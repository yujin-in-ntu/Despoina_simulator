// SPDX-License-Identifier: Apache-2.0
#ifndef WORK_STEALING_SCHEDULER_H
#define WORK_STEALING_SCHEDULER_H

#include <vector>
#include <deque>
#include <algorithm>
#include "pe_cluster.h"

/**
 * Work Stealing Scheduler for PE Array SpMM
 * 
 * Features:
 * - Distributes CSR rows to clusters based on degree (load balancing)
 * - Implements work stealing between clusters
 * - Tracks load balance metrics
 * 
 * Algorithm:
 * 1. Sort rows by degree (descending)
 * 2. Distribute rows round-robin to balance load
 * 3. When a cluster becomes idle, it steals from busy clusters
 */
class WorkStealingScheduler {
public:
    // Configuration
    static constexpr int STEAL_THRESHOLD = 2;  // Min queue size to allow stealing
    static constexpr int NOC_CYCLES_PER_HOP = 2;
    
    // CSR Matrix representation
    struct CSRMatrix {
        std::vector<int> row_ptr;      // Row pointers (size n+1)
        std::vector<int> col_idx;      // Column indices
        std::vector<float> values;     // Non-zero values
        int n;                         // Number of rows
        int nnz;                       // Total non-zeros
        
        CSRMatrix() : n(0), nnz(0) {}
    };
    
    // Row work item with degree info
    struct RowWork {
        int row_id;
        int start_ptr;     // Start index in CSR arrays
        int end_ptr;       // End index in CSR arrays
        int degree;        // Number of non-zeros (for load balancing)
        
        RowWork() : row_id(-1), start_ptr(0), end_ptr(0), degree(0) {}
        RowWork(int r, int s, int e, int d) : row_id(r), start_ptr(s), end_ptr(e), degree(d) {}
    };

    struct PendingSteal {
        int thief_id = -1;
        int victim_id = -1;
        int remaining_cycles = 0;
        std::vector<RowWork> stolen_work;
    };

    // Constructor
    WorkStealingScheduler();
    ~WorkStealingScheduler();
    
    // Initialize with number of clusters
    void init(int num_clusters);
    
    // Load CSR matrix and distribute rows
    void load_matrix(const CSRMatrix& csr);
    
    // Get next work item for a cluster (may involve stealing)
    bool get_work(int cluster_id, PECluster::WorkItem& work, const CSRMatrix& csr);
    void tick();
    
    // Check if all work is done
    bool all_done() const;
    
    // Statistics
    long long get_total_rows_processed() const { return total_rows_processed; }
    long long get_steals_count() const { return steals_count; }
    long long get_noc_steal_latency_total_cycles() const { return noc_steal_latency_total_cycles; }
    long long get_noc_steal_count() const { return noc_steal_count; }
    float get_noc_steal_avg_hops() const;
    float get_load_imbalance() const;

private:
    int num_clusters;
    
    // Per-cluster work queues
    std::vector<std::deque<RowWork>> work_queues;
    
    // Statistics
    long long total_rows_processed;
    long long steals_count;
    long long noc_steal_latency_total_cycles;
    long long noc_steal_count;
    long long noc_steal_hops_total;
    std::vector<long long> rows_per_cluster;
    std::vector<PendingSteal> pending_steals_;

    // Internal methods
    void distribute_rows(const CSRMatrix& csr);
    int find_victim_cluster(int thief_id);
    bool steal_from(int thief_id, int victim_id);
    int cluster_mesh_width() const;
    int manhattan_distance(int cluster_a, int cluster_b) const;
    bool has_pending_steal_for(int cluster_id) const;
};

#endif // WORK_STEALING_SCHEDULER_H
