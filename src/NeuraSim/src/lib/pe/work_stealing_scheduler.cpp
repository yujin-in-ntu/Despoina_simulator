// SPDX-License-Identifier: Apache-2.0
/**
 * Work Stealing Scheduler Implementation
 * 
 * Distributes SpMM work (CSR rows) to clusters and handles
 * dynamic load balancing through work stealing.
 */

#include "work_stealing_scheduler.h"
#include <algorithm>
#include <numeric>
#include <cstring>
#include <cmath>
#include <utility>

WorkStealingScheduler::WorkStealingScheduler()
    : num_clusters(0), total_rows_processed(0), steals_count(0),
      noc_steal_latency_total_cycles(0), noc_steal_count(0), noc_steal_hops_total(0) {
}

WorkStealingScheduler::~WorkStealingScheduler() {
    work_queues.clear();
}

void WorkStealingScheduler::init(int num_clusters) {
    this->num_clusters = std::max(1, num_clusters);
    
    // Clear state
    work_queues.assign(this->num_clusters, std::deque<RowWork>());
    rows_per_cluster.assign(this->num_clusters, 0);
    total_rows_processed = 0;
    steals_count = 0;
    noc_steal_latency_total_cycles = 0;
    noc_steal_count = 0;
    noc_steal_hops_total = 0;
    pending_steals_.clear();
}

void WorkStealingScheduler::load_matrix(const CSRMatrix& csr) {
    distribute_rows(csr);
}

void WorkStealingScheduler::distribute_rows(const CSRMatrix& csr) {
    // Create work items for all rows
    std::vector<RowWork> all_rows;
    all_rows.reserve(csr.n);
    
    for (int i = 0; i < csr.n; i++) {
        int start = csr.row_ptr[i];
        int end = csr.row_ptr[i + 1];
        int degree = end - start;
        
        if (degree > 0) {  // Only add non-empty rows
            all_rows.emplace_back(i, start, end, degree);
        }
    }
    
    // Sort by degree descending for better load balancing
    // High-degree rows are distributed first
    std::sort(all_rows.begin(), all_rows.end(),
              [](const RowWork& a, const RowWork& b) {
                  return a.degree > b.degree;
              });
    
    // Track total work per cluster for balanced distribution
    std::vector<long long> cluster_work(num_clusters, 0);
    
    // Distribute using least-loaded-first strategy
    for (const auto& row : all_rows) {
        // Find cluster with least work
        int min_cluster = 0;
        long long min_work = cluster_work[0];
        
        for (int c = 1; c < num_clusters; c++) {
            if (cluster_work[c] < min_work) {
                min_work = cluster_work[c];
                min_cluster = c;
            }
        }
        
        // Assign to least loaded cluster
        work_queues[min_cluster].push_back(row);
        cluster_work[min_cluster] += row.degree;
    }
}

bool WorkStealingScheduler::get_work(int cluster_id, PECluster::WorkItem& work, 
                                     const CSRMatrix& csr) {
    if (cluster_id < 0 || cluster_id >= num_clusters) {
        return false;
    }
    
    // Try to get work from own queue first
    if (!work_queues[cluster_id].empty()) {
        RowWork row = work_queues[cluster_id].front();
        work_queues[cluster_id].pop_front();
        
        // Convert to WorkItem
        work.row_id = row.row_id;
        work.csr_start_ptr = row.start_ptr;
        work.nnz = row.degree;
        work.valid = true;
        
        // Copy data from CSR
        work.col_indices.clear();
        work.values.clear();
        work.col_indices.reserve(row.degree);
        work.values.reserve(row.degree);
        
        for (int i = row.start_ptr; i < row.end_ptr; i++) {
            work.col_indices.push_back(csr.col_idx[i]);
            work.values.push_back(csr.values[i]);
        }
        
        total_rows_processed++;
        rows_per_cluster[cluster_id]++;
        return true;
    }

    // A NoC steal is already in-flight for this thief.
    if (has_pending_steal_for(cluster_id)) {
        work.valid = false;
        return false;
    }
    
    // Own queue empty - try to steal
    int victim = find_victim_cluster(cluster_id);
    if (victim >= 0 && steal_from(cluster_id, victim)) {
        work.valid = false;
        return false;
    }
    
    // No work available
    work.valid = false;
    return false;
}

bool WorkStealingScheduler::all_done() const {
    if (!pending_steals_.empty()) {
        return false;
    }
    for (int i = 0; i < num_clusters; i++) {
        if (!work_queues[i].empty()) {
            return false;
        }
    }
    return true;
}

void WorkStealingScheduler::tick() {
    if (pending_steals_.empty()) {
        return;
    }

    for (auto it = pending_steals_.begin(); it != pending_steals_.end(); ) {
        if (it->remaining_cycles > 0) {
            it->remaining_cycles--;
        }

        if (it->remaining_cycles <= 0) {
            if (it->thief_id >= 0 && it->thief_id < num_clusters) {
                for (const auto& stolen : it->stolen_work) {
                    work_queues[it->thief_id].push_back(stolen);
                }
            }
            it = pending_steals_.erase(it);
            continue;
        }
        ++it;
    }
}

float WorkStealingScheduler::get_load_imbalance() const {
    if (num_clusters == 0 || total_rows_processed == 0) {
        return 0.0f;
    }
    
    // Calculate coefficient of variation of rows per cluster
    double mean = (double)total_rows_processed / num_clusters;
    double variance = 0.0;
    
    for (int i = 0; i < num_clusters; i++) {
        double diff = rows_per_cluster[i] - mean;
        variance += diff * diff;
    }
    variance /= num_clusters;
    
    double std_dev = sqrt(variance);
    return (float)(std_dev / mean);  // CV = std_dev / mean
}

float WorkStealingScheduler::get_noc_steal_avg_hops() const {
    if (noc_steal_count <= 0) {
        return 0.0f;
    }
    return static_cast<float>(noc_steal_hops_total) / static_cast<float>(noc_steal_count);
}

int WorkStealingScheduler::find_victim_cluster(int thief_id) {
    int best_victim = -1;
    size_t max_queue_size = STEAL_THRESHOLD;
    
    // Find cluster with most work
    for (int i = 0; i < num_clusters; i++) {
        if (i == thief_id) continue;
        
        if (work_queues[i].size() > max_queue_size) {
            max_queue_size = work_queues[i].size();
            best_victim = i;
        }
    }
    
    return best_victim;
}

bool WorkStealingScheduler::steal_from(int thief_id, int victim_id) {
    if (victim_id < 0 || work_queues[victim_id].size() <= STEAL_THRESHOLD) {
        return false;
    }
    
    // Steal half of victim's remaining work (up to some limit)
    size_t steal_count = work_queues[victim_id].size() / 2;
    steal_count = std::min(steal_count, (size_t)8);  // Max steal 8 at a time
    steal_count = std::max(steal_count, (size_t)1);  // At least steal 1
    
    PendingSteal pending;
    pending.thief_id = thief_id;
    pending.victim_id = victim_id;

    const int hops = manhattan_distance(thief_id, victim_id);
    const int latency = hops * NOC_CYCLES_PER_HOP * 2;  // request + data
    pending.remaining_cycles = std::max(1, latency);

    for (size_t i = 0; i < steal_count && !work_queues[victim_id].empty(); i++) {
        // Steal from back (LIFO stealing for cache locality)
        RowWork stolen = work_queues[victim_id].back();
        work_queues[victim_id].pop_back();
        pending.stolen_work.push_back(stolen);
    }

    pending_steals_.push_back(std::move(pending));

    steals_count++;
    noc_steal_count++;
    noc_steal_hops_total += hops;
    noc_steal_latency_total_cycles += latency;
    return true;
}

int WorkStealingScheduler::manhattan_distance(int cluster_a, int cluster_b) const {
    const int mesh_width = cluster_mesh_width();
    const int ax = cluster_a % mesh_width;
    const int ay = cluster_a / mesh_width;
    const int bx = cluster_b % mesh_width;
    const int by = cluster_b / mesh_width;
    return std::abs(ax - bx) + std::abs(ay - by);
}

int WorkStealingScheduler::cluster_mesh_width() const {
    return std::max(1, static_cast<int>(std::ceil(std::sqrt(static_cast<double>(num_clusters)))));
}

bool WorkStealingScheduler::has_pending_steal_for(int cluster_id) const {
    for (const auto& ps : pending_steals_) {
        if (ps.thief_id == cluster_id) {
            return true;
        }
    }
    return false;
}
