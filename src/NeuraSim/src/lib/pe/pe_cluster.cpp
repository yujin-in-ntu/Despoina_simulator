// SPDX-License-Identifier: Apache-2.0
/**
 * PE Cluster Implementation
 * 
 * Groups 4 PEs with shared F buffer for bandwidth reduction.
 * Uses MRU (Most Recently Used) cache for F row reuse.
 */

#include "pe_cluster.h"
#include "a_load_burst_model.h"
#include "shared_f_cache.h"
#include "energy_params_generated.h"
#include <algorithm>
#include <cstring>

namespace {
// Model L2(shared) cache access latency using CACTI-derived F-cache access cycles.
// This is the latency to read an F row from the shared cache SRAM bank(s), excluding NoC.
constexpr int kSharedCacheHitLatencyCycles = energy::sram::f_cache::ACCESS_CYCLES;

// Model L1(local MRU) access latency using CACTI-derived PE register file access cycles.
// This approximates a small, close-to-PE SRAM/RF-style access (cluster-local).
constexpr int kLocalCacheHitLatencyCycles = energy::sram::pe_registers::ACCESS_CYCLES;
}

PECluster::PECluster() 
    : gcp(nullptr), cluster_id(-1), shared_f_cache(nullptr),
      f_cache(nullptr), cache_age_counter(0), f_matrix(nullptr), f_n(0), f_d(0),
      f_cache_hits(0), f_cache_misses(0),
      memory_stall_cycles(0), pending_output_cycles(0), output_write_active(false),
      output_write_pe_idx(-1), output_rr_next_pe(0), output_backpressure_stalls(0) {
    // Allocate f_cache on heap to avoid stack overflow (256 entries × 8KB = 2MB)
    f_cache = new FCacheEntry[MRU_CACHE_ENTRIES];
    for (int i = 0; i < MRU_CACHE_ENTRIES; i++) {
        f_cache[i].data = new float[MAX_FEATURE_DIM]();
        f_cache[i].valid = false;
    }
    
    for (int i = 0; i < PES_PER_CLUSTER; i++) {
        work_ptr[i] = 0;
        pending_f_reqs[i].active = false;
#ifdef ENABLE_DRAMSIM3_PE
        pending_a_load_reqs[i].active = false;
#else
        pending_a_load_cycles[i] = 0;
#endif
    }
    current_cycle = 0;
    reset_l1_bank_access_tracker();
}

PECluster::PECluster(GlobalCounters* gcp, int cluster_id) : PECluster() {
    init(gcp, cluster_id);
}

PECluster::~PECluster() {
    // Clean up work queue
    while (!work_queue.empty()) {
        work_queue.pop();
    }
    
    // Free heap-allocated f_cache
    if (f_cache) {
        for (int i = 0; i < MRU_CACHE_ENTRIES; i++) {
            delete[] f_cache[i].data;
        }
        delete[] f_cache;
        f_cache = nullptr;
    }
}

void PECluster::init(GlobalCounters* gcp, int cluster_id) {
    this->gcp = gcp;
    this->cluster_id = cluster_id;
    
    // Initialize PEs
    for (int i = 0; i < PES_PER_CLUSTER; i++) {
        int pe_id = cluster_id * PES_PER_CLUSTER + i;
        pes[i].init(gcp, pe_id, cluster_id);
        current_work[i].valid = false;
        work_ptr[i] = 0;
        pending_f_reqs[i].active = false;
#ifdef ENABLE_DRAMSIM3_PE
        pending_a_load_reqs[i].active = false;
#else
        pending_a_load_cycles[i] = 0;
#endif
    }
    
    // Clear cache (with null check for safety)
    if (f_cache) {
        for (int i = 0; i < MRU_CACHE_ENTRIES; i++) {
            f_cache[i].valid = false;
        }
    }
    cache_age_counter = 0;
    f_cache_hits = 0;
    f_cache_misses = 0;
    memory_stall_cycles = 0;
    for (int i = 0; i < PES_PER_CLUSTER; i++) {
#ifdef ENABLE_DRAMSIM3_PE
        pending_a_load_reqs[i].active = false;
        pending_a_load_reqs[i].needs_submit = false;
        pending_a_load_reqs[i].waiting_for_dram = false;
#else
        pending_a_load_cycles[i] = 0;
#endif
    }
    pending_output_cycles = 0;
    output_write_active = false;
    output_write_pe_idx = -1;
    output_rr_next_pe = 0;
    output_backpressure_stalls = 0;
    l1_f_cache_hits = 0;
    l2_shared_hits = 0;
    l2_shared_hits_pre_submit = 0;
    dram_fills = 0;
    l1_bank_conflict_stalls = 0;
    l2_bank_conflict_stalls = 0;
    current_cycle = 0;
    reset_l1_bank_access_tracker();
    
#ifdef ENABLE_DRAMSIM3_PE
    dram_actual_latency = 0;
#endif
}

void PECluster::reset_for_new_spmm(bool clear_local_cache) {
    // Clear pending work / state
    while (!work_queue.empty()) work_queue.pop();
    while (!output_queue.empty()) output_queue.pop();

    for (int i = 0; i < PES_PER_CLUSTER; i++) {
        current_work[i].valid = false;
        work_ptr[i] = 0;
        pending_f_reqs[i].active = false;
#ifdef ENABLE_DRAMSIM3_PE
        pending_f_reqs[i].needs_submit = false;
        pending_f_reqs[i].waiting_for_dram = false;
#endif
        // Reset all execution registers.  The PE feature register is part of
        // the cache epoch: it is reusable only when the L1/shared caches are
        // also proven to belong to the same feature payload.
        pes[i].reset_for_new_spmm(/*keep_feature_register=*/!clear_local_cache);
#ifdef ENABLE_DRAMSIM3_PE
        pending_a_load_reqs[i].active = false;
        pending_a_load_reqs[i].needs_submit = false;
        pending_a_load_reqs[i].waiting_for_dram = false;
#else
        pending_a_load_cycles[i] = 0;
#endif
    }

    // Optionally clear local MRU cache contents (data), not just stats.
    if (clear_local_cache && f_cache) {
        for (int i = 0; i < MRU_CACHE_ENTRIES; i++) {
            f_cache[i].valid = false;
        }
        cache_age_counter = 0;
    }

    // Reset per-job counters (these are used for energy modeling)
    f_cache_hits = 0;
    f_cache_misses = 0;
    memory_stall_cycles = 0;
    for (int i = 0; i < PES_PER_CLUSTER; i++) {
#ifdef ENABLE_DRAMSIM3_PE
        pending_a_load_reqs[i].active = false;
        pending_a_load_reqs[i].needs_submit = false;
        pending_a_load_reqs[i].waiting_for_dram = false;
#else
        pending_a_load_cycles[i] = 0;
#endif
    }
    pending_output_cycles = 0;
    output_write_active = false;
    output_write_pe_idx = -1;
    output_rr_next_pe = 0;
    output_backpressure_stalls = 0;
    l1_f_cache_hits = 0;
    l2_shared_hits = 0;
    l2_shared_hits_pre_submit = 0;
    dram_fills = 0;
    l1_bank_conflict_stalls = 0;
    l2_bank_conflict_stalls = 0;
    current_cycle = 0;
    reset_l1_bank_access_tracker();

#ifdef ENABLE_DRAMSIM3_PE
    dram_actual_latency = 0;
#endif
}

void PECluster::invalidate_local_f_row(int row_id) {
    if (!f_cache) return;
    for (int i = 0; i < MRU_CACHE_ENTRIES; i++) {
        if (f_cache[i].valid && f_cache[i].row_id == row_id) {
            f_cache[i].valid = false;
        }
    }
}

bool PECluster::tick() {
    bool did_work = false;
    bool any_stalled = false;
    current_cycle++;
    reset_l1_bank_access_tracker();
    
    // =========================================================================
    // Process pending memory requests (DRAM latency simulation)
    // =========================================================================
    for (int i = 0; i < PES_PER_CLUSTER; i++) {
        if (pending_f_reqs[i].active) {
#ifdef ENABLE_DRAMSIM3_PE
            // DRAMsim3 mode: three states
            // 1. needs_submit = true: waiting for PE Array to submit
            // 2. needs_submit = false && waiting_for_dram = true: submitted, waiting for callback
            // 3. needs_submit = false && waiting_for_dram = false && remaining_cycles < 0: ready to load
            if (pending_f_reqs[i].needs_submit) {
                // If a different cluster has already fetched this row into the shared cache,
                // satisfy it immediately and cancel the pending submit.
                if (shared_f_cache && f_matrix != nullptr) {
                    int col_idx = pending_f_reqs[i].col_idx;
                    int l2_bank_stall = 0;
                    const float* shared_row = shared_f_cache->get_ptr(col_idx, current_cycle, &l2_bank_stall);
                    if (shared_row != nullptr) {
                        l2_shared_hits++;
                        l2_shared_hits_pre_submit++;
                        l2_bank_conflict_stalls += l2_bank_stall;
                        insert_f_cache(col_idx, shared_row, f_d);
                        pending_f_reqs[i].needs_submit = false;
                        pending_f_reqs[i].waiting_for_dram = false;
                        pending_f_reqs[i].remaining_cycles =
                            kSharedCacheHitLatencyCycles + l2_bank_stall;
                        any_stalled = true;
                        continue;
                    }
                }
                // State 1: waiting for PE Array to submit
                any_stalled = true;
            } else if (pending_f_reqs[i].waiting_for_dram) {
                // State 2: submitted, waiting for DRAMsim3 callback
                any_stalled = true;
            } else if (pending_f_reqs[i].remaining_cycles < 0) {
                // State 3: callback received, load data now
                int col_idx = pending_f_reqs[i].col_idx;
                if (col_idx >= 0 && col_idx < f_n && f_matrix != nullptr) {
                    const float* row_data = f_matrix + col_idx * f_d;
                    if (shared_f_cache) {
                        int l2_bank_stall = 0;
                        shared_f_cache->put(col_idx, row_data, current_cycle, &l2_bank_stall);
                        l2_bank_conflict_stalls += l2_bank_stall;
                    }
                    dram_fills++;
                    insert_f_cache(col_idx, row_data, f_d);
                    pes[i].load_f_row(row_data, f_d);
                }
                pending_f_reqs[i].active = false;
                did_work = true;
            } else if (pending_f_reqs[i].remaining_cycles > 0) {
                // Shared-cache hit (modeled latency) or other local timed fetch.
                pending_f_reqs[i].remaining_cycles--;
                any_stalled = true;
                if (pending_f_reqs[i].remaining_cycles <= 0) {
                    int col_idx = pending_f_reqs[i].col_idx;
                    // Row data should already be in local cache (inserted at hit time).
                    FCacheEntry* cached = lookup_f_cache(col_idx);
                    if (cached != nullptr) {
                        pes[i].load_f_row(cached->data, cached->dim);
                    } else if (shared_f_cache) {
                        // Fallback: re-query shared cache (should usually hit).
                        int l2_bank_stall = 0;
                        const float* shared_row = shared_f_cache->get_ptr(col_idx, current_cycle, &l2_bank_stall);
                        if (shared_row != nullptr) {
                            l2_bank_conflict_stalls += l2_bank_stall;
                            insert_f_cache(col_idx, shared_row, f_d);
                            pes[i].load_f_row(shared_row, f_d);
                        }
                    }
                    pending_f_reqs[i].active = false;
                    did_work = true;
                }
            }
#else
            // Constant latency mode: count down remaining cycles
            pending_f_reqs[i].remaining_cycles--;
            if (pending_f_reqs[i].remaining_cycles <= 0) {
                // Memory request completed - load F row to PE
                int col_idx = pending_f_reqs[i].col_idx;
                if (col_idx >= 0 && col_idx < f_n && f_matrix != nullptr) {
                    const float* row_data = f_matrix + col_idx * f_d;
                    if (shared_f_cache) {
                        int l2_bank_stall = 0;
                        shared_f_cache->put(col_idx, row_data, current_cycle, &l2_bank_stall);
                        l2_bank_conflict_stalls += l2_bank_stall;
                    }
                    dram_fills++;
                    insert_f_cache(col_idx, row_data, f_d);
                    pes[i].load_f_row(row_data, f_d);
                }
                pending_f_reqs[i].active = false;
                did_work = true;
            } else {
                any_stalled = true;
            }
#endif
        }
    }
    
    // Process pending A load (per-PE)
    for (int i = 0; i < PES_PER_CLUSTER; i++) {
#ifdef ENABLE_DRAMSIM3_PE
        if (pending_a_load_reqs[i].active) {
            if (pending_a_load_reqs[i].needs_submit || pending_a_load_reqs[i].waiting_for_dram) {
                any_stalled = true;
            } else {
                // Callback was received and request is complete.
                pending_a_load_reqs[i].active = false;
                did_work = true;
            }
        }
#else
        if (pending_a_load_cycles[i] > 0) {
            pending_a_load_cycles[i]--;
            any_stalled = true;
        }
#endif
    }
    
    // Process in-flight output writeback (latency + queue backpressure)
    if (output_write_active) {
        if (pending_output_cycles > 0) {
            pending_output_cycles--;
            any_stalled = true;
        }

        // Write latency finished; now retire into finite output queue.
        if (pending_output_cycles == 0) {
            if ((int)output_queue.size() >= OUTPUT_QUEUE_DEPTH) {
                output_backpressure_stalls++;
                any_stalled = true;
            } else {
                commit_output_write();
                did_work = true;
            }
        }
    }
    
    if (any_stalled) {
        memory_stall_cycles++;
    }
    
    // =========================================================================
    // Distribute work to idle PEs
    // =========================================================================
    distribute_work_to_pes();
    
    // =========================================================================
    // Process each PE
    // =========================================================================
    for (int i = 0; i < PES_PER_CLUSTER; i++) {
        // Check if PE needs F row (but don't re-issue if a request is already pending)
        if (!pending_f_reqs[i].active && pes[i].needs_f_row()) {
            bool fetched = fetch_f_row_for_pe(i);
            if (fetched) did_work = true;
        }
        
        // Tick the PE every cycle, even while a memory request is pending.
        // This is cycle-accurate: PE may still drain its internal pipeline while waiting for F.
        if (pes[i].tick()) {
            did_work = true;
        }
    }
    
    // =========================================================================
    // Collect outputs (starts at most one writeback at a time)
    // =========================================================================
    if (!output_write_active) {
        collect_pe_outputs();
    }
    
    return did_work || any_stalled;
}

void PECluster::assign_work(const WorkItem& work) {
    work_queue.push(work);
}

bool PECluster::can_accept_work() const {
    // Accept work if queue is not too full
    return work_queue.size() < 16;  // Max queue depth
}

bool PECluster::is_done() const {
    // Check if all PEs are done and no pending work
    if (!work_queue.empty()) return false;
    if (output_write_active || !output_queue.empty()) return false;
    
    for (int i = 0; i < PES_PER_CLUSTER; i++) {
        if (!pes[i].is_done() || current_work[i].valid) {
            return false;
        }
    }
    return true;
}

void PECluster::set_f_matrix(const float* f_matrix, int n, int d) {
    this->f_matrix = f_matrix;
    this->f_n = n;
    this->f_d = d;
}

bool PECluster::get_output(int* row_id, float* output, int max_dim) {
    if (output_queue.empty()) return false;
    
    OutputEntry& entry = output_queue.front();
    *row_id = entry.row_id;
    
    int copy_dim = std::min(max_dim, f_d);
    std::memcpy(output, entry.data, copy_dim * sizeof(float));
    
    output_queue.pop();
    return true;
}

bool PECluster::has_stealable_work() const {
    return work_queue.size() > 1;  // Keep at least one for ourselves
}

PECluster::WorkItem PECluster::steal_work() {
    WorkItem stolen;
    stolen.valid = false;
    
    if (work_queue.size() > 1) {
        stolen = work_queue.front();
        work_queue.pop();
    }
    
    return stolen;
}

long long PECluster::get_total_mac_ops() const {
    long long total = 0;
    for (int i = 0; i < PES_PER_CLUSTER; i++) {
        total += pes[i].get_mac_count();
    }
    return total;
}

// ============ Internal Methods ============

void PECluster::distribute_work_to_pes() {
    for (int i = 0; i < PES_PER_CLUSTER; i++) {
        // Skip if this PE is waiting for F row
        if (pending_f_reqs[i].active) continue;
        // Skip if this PE is still loading A batch (per-PE latency model)
        if (is_a_load_pending(i)) continue;
        
        // If PE has finished its work, assign new work
        if (pes[i].is_done() && !current_work[i].valid && !work_queue.empty()) {
            current_work[i] = work_queue.front();
            work_queue.pop();
            work_ptr[i] = 0;
            
            // Set output row
            pes[i].set_output_row(current_work[i].row_id);
            pes[i].clear_psum();

#ifdef ENABLE_DRAMSIM3_PE
            // Model every 64-byte cacheline intersecting the exact CSR payload.
            // Lines are issued serially for this PE, so no unmodeled overlap is
            // introduced.  Empty rows issue no A-load transaction.
            PendingALoadReq& req = pending_a_load_reqs[i];
            req.pe_idx = i;
            req.row_id = current_work[i].row_id;
            const ALoadBurstSpan span = make_a_load_burst_span(
                A_LOAD_ADDR_BASE, current_work[i].csr_start_ptr, current_work[i].nnz);
            req.payload_bytes = static_cast<int>(span.payload_bytes);
            req.addr = span.begin;
            req.end_addr = span.end;
            req.active = req.payload_bytes > 0;
            req.needs_submit = req.active;
            req.waiting_for_dram = false;
#else
            // Fallback non-DRAMsim3 build: fixed A-load latency.
            pending_a_load_cycles[i] = A_LOAD_LATENCY;
#endif
            // Do not feed the first A batch until the row load completes.
            continue;
        }
        
        // Feed A values to PE if it's done with previous batch
        if (current_work[i].valid && pes[i].is_done() && work_ptr[i] < current_work[i].nnz) {
            int remaining = current_work[i].nnz - work_ptr[i];
            int batch_size = std::min(remaining, PE::A_REG_SIZE);
            
            pes[i].load_a_values(
                current_work[i].values.data() + work_ptr[i],
                current_work[i].col_indices.data() + work_ptr[i],
                batch_size
            );
            
            work_ptr[i] += batch_size;
        }
        
        // Check if work is complete for this PE
        if (current_work[i].valid && pes[i].is_done() && work_ptr[i] >= current_work[i].nnz) {
            // Row computation complete, mark for output collection
        }
    }
}

void PECluster::reset_l1_bank_access_tracker() {
    l1_bank_access_count.fill(0);
}

int PECluster::reserve_l1_bank_port(int row_id) {
    int bank_id = row_id % L1_NUM_BANKS;
    if (bank_id < 0) {
        bank_id += L1_NUM_BANKS;
    }

    const int current_accesses = l1_bank_access_count[bank_id];
    const int ports = std::max(1, L1_PORTS_PER_BANK);
    const int stall_cycles = current_accesses / ports;
    l1_bank_access_count[bank_id] = current_accesses + 1;
    return stall_cycles;
}

bool PECluster::is_a_load_pending(int pe_idx) const {
#ifdef ENABLE_DRAMSIM3_PE
    return pending_a_load_reqs[pe_idx].active;
#else
    return pending_a_load_cycles[pe_idx] > 0;
#endif
}

bool PECluster::fetch_f_row_for_pe(int pe_idx) {
    int col_idx = pes[pe_idx].get_needed_f_col();
    if (col_idx < 0 || col_idx >= f_n) return false;
    
    // Check cache first
    FCacheEntry* cached = lookup_f_cache(col_idx);
    if (cached != nullptr) {
        // L1 (local MRU) hit: model local SRAM/RF access latency (CACTI).
        f_cache_hits++;
        l1_f_cache_hits++;
        const int l1_bank_stall = reserve_l1_bank_port(col_idx);
        l1_bank_conflict_stalls += l1_bank_stall;

        // Start a timed local request to model L1 hit latency.
        pending_f_reqs[pe_idx].pe_idx = pe_idx;
        pending_f_reqs[pe_idx].col_idx = col_idx;
        pending_f_reqs[pe_idx].active = true;
#ifdef ENABLE_DRAMSIM3_PE
        pending_f_reqs[pe_idx].needs_submit = false;
        pending_f_reqs[pe_idx].waiting_for_dram = false;
#endif
        // Even though data is already local, we delay the load to PE to account for access time.
        pending_f_reqs[pe_idx].remaining_cycles = kLocalCacheHitLatencyCycles + l1_bank_stall;
        return false;
    }

    // Shared cache (chip-level) check before issuing a DRAM request.
    if (shared_f_cache) {
        int l2_bank_stall = 0;
        const float* shared_row = shared_f_cache->get_ptr(col_idx, current_cycle, &l2_bank_stall);
        if (shared_row != nullptr) {
            // Shared (L2) hit: model SRAM access latency (CACTI) and avoid DRAM.
            insert_f_cache(col_idx, shared_row, f_d);
            f_cache_hits++;
            l2_shared_hits++;
            l2_bank_conflict_stalls += l2_bank_stall;
            // Start a timed local request to model L2 hit latency.
            pending_f_reqs[pe_idx].pe_idx = pe_idx;
            pending_f_reqs[pe_idx].col_idx = col_idx;
            pending_f_reqs[pe_idx].active = true;
#ifdef ENABLE_DRAMSIM3_PE
            pending_f_reqs[pe_idx].needs_submit = false;
            pending_f_reqs[pe_idx].waiting_for_dram = false;
#endif
            pending_f_reqs[pe_idx].remaining_cycles = kSharedCacheHitLatencyCycles + l2_bank_stall;
            return false;
        }
    }
    
    // Cache miss - need to fetch from DRAM
    f_cache_misses++;
    
    if (f_matrix == nullptr) return false;
    
    // Start pending memory request
    pending_f_reqs[pe_idx].pe_idx = pe_idx;
    pending_f_reqs[pe_idx].col_idx = col_idx;
    pending_f_reqs[pe_idx].active = true;
    
#ifdef ENABLE_DRAMSIM3_PE
    // DRAMsim3 mode: mark as needs_submit for PE Array to pick up
    pending_f_reqs[pe_idx].needs_submit = true;
    pending_f_reqs[pe_idx].waiting_for_dram = false;
    pending_f_reqs[pe_idx].remaining_cycles = 0;
#else
    // Constant latency mode
    pending_f_reqs[pe_idx].remaining_cycles = DRAM_LATENCY;
#endif
    
    // Data will be loaded when DRAMsim3 callback or remaining_cycles reaches 0
    return false;  // Not yet available
}

void PECluster::collect_pe_outputs() {
    if (output_write_active) {
        return;
    }

    if ((int)output_queue.size() >= OUTPUT_QUEUE_DEPTH) {
        output_backpressure_stalls++;
        return;
    }

    // Round-robin selection among done PEs so one writeback starts per cycle max.
    for (int offset = 0; offset < PES_PER_CLUSTER; offset++) {
        const int i = (output_rr_next_pe + offset) % PES_PER_CLUSTER;
        if (pending_f_reqs[i].active) continue;
        if (!(current_work[i].valid && pes[i].is_done() && work_ptr[i] >= current_work[i].nnz)) {
            continue;
        }

        output_write_active = true;
        output_write_pe_idx = i;
        pending_output_cycles = OUTPUT_WRITE_LATENCY;
        output_rr_next_pe = (i + 1) % PES_PER_CLUSTER;
        return;
    }
}

void PECluster::commit_output_write() {
    if (!output_write_active || output_write_pe_idx < 0 || output_write_pe_idx >= PES_PER_CLUSTER) {
        output_write_active = false;
        output_write_pe_idx = -1;
        pending_output_cycles = 0;
        return;
    }

    const int pe_idx = output_write_pe_idx;
    if (!(current_work[pe_idx].valid && pes[pe_idx].is_done() && work_ptr[pe_idx] >= current_work[pe_idx].nnz)) {
        output_write_active = false;
        output_write_pe_idx = -1;
        pending_output_cycles = 0;
        return;
    }

    OutputEntry entry;
    entry.row_id = current_work[pe_idx].row_id;
    const float* psum = pes[pe_idx].get_psum();
    const int copy_dim = std::min(f_d, (int)MAX_FEATURE_DIM);
    std::memcpy(entry.data, psum, copy_dim * sizeof(float));
    output_queue.push(entry);

    current_work[pe_idx].valid = false;
    output_write_active = false;
    output_write_pe_idx = -1;
    pending_output_cycles = 0;
}

PECluster::FCacheEntry* PECluster::lookup_f_cache(int row_id) {
    if (!f_cache) return nullptr;
    for (int i = 0; i < MRU_CACHE_ENTRIES; i++) {
        if (f_cache[i].valid && f_cache[i].row_id == row_id) {
            // Update MRU age
            f_cache[i].age = cache_age_counter++;
            return &f_cache[i];
        }
    }
    return nullptr;
}

void PECluster::insert_f_cache(int row_id, const float* data, int dim) {
    if (!f_cache || !data) return;
    
    // Find empty slot or evict LRU
    int slot = -1;
    
    for (int i = 0; i < MRU_CACHE_ENTRIES; i++) {
        if (!f_cache[i].valid) {
            slot = i;
            break;
        }
    }
    
    if (slot < 0) {
        evict_lru_cache_entry();
        // Find the newly freed slot
        for (int i = 0; i < MRU_CACHE_ENTRIES; i++) {
            if (!f_cache[i].valid) {
                slot = i;
                break;
            }
        }
    }
    
    if (slot >= 0 && f_cache[slot].data) {
        f_cache[slot].row_id = row_id;
        f_cache[slot].dim = std::min(dim, (int)MAX_FEATURE_DIM);
        std::memcpy(f_cache[slot].data, data, f_cache[slot].dim * sizeof(float));
        f_cache[slot].age = cache_age_counter++;
        f_cache[slot].valid = true;
    }
}

void PECluster::evict_lru_cache_entry() {
    if (!f_cache) return;
    int lru_idx = 0;
    int lru_age = f_cache[0].age;
    
    for (int i = 1; i < MRU_CACHE_ENTRIES; i++) {
        if (f_cache[i].valid && f_cache[i].age < lru_age) {
            lru_age = f_cache[i].age;
            lru_idx = i;
        }
    }
    
    f_cache[lru_idx].valid = false;
}

// =============================================================================
// DRAMsim3 Integration
// =============================================================================
#ifdef ENABLE_DRAMSIM3_PE

void PECluster::notify_dram_complete(int col_idx, long long actual_latency) {
    // Find the PE waiting for this col_idx
    for (int i = 0; i < PES_PER_CLUSTER; i++) {
        if (pending_f_reqs[i].active && 
            pending_f_reqs[i].col_idx == col_idx &&
            pending_f_reqs[i].waiting_for_dram) {
            // Mark as callback received (remaining_cycles = -1 signals completion)
            pending_f_reqs[i].waiting_for_dram = false;
            pending_f_reqs[i].remaining_cycles = -1;  // Signal: callback received
            dram_actual_latency = actual_latency;
            break;
        }
    }
}

bool PECluster::has_pending_dram_requests() const {
    // Check if any PE needs its request submitted to DRAMsim3
    for (int i = 0; i < PES_PER_CLUSTER; i++) {
        if (pending_f_reqs[i].active && pending_f_reqs[i].needs_submit) {
            return true;
        }
    }
    return false;
}

void PECluster::get_pending_dram_request(int& pe_idx, int& col_idx) const {
    // Return the first PE that needs its request submitted
    for (int i = 0; i < PES_PER_CLUSTER; i++) {
        if (pending_f_reqs[i].active && pending_f_reqs[i].needs_submit) {
            pe_idx = i;
            col_idx = pending_f_reqs[i].col_idx;
            return;
        }
    }
    pe_idx = -1;
    col_idx = -1;
}

void PECluster::set_waiting_for_dram(int pe_idx, int col_idx) {
    // Called by PE Array after submitting to DRAMsim3
    if (pe_idx >= 0 && pe_idx < PES_PER_CLUSTER) {
        if (pending_f_reqs[pe_idx].active && 
            pending_f_reqs[pe_idx].col_idx == col_idx) {
            pending_f_reqs[pe_idx].needs_submit = false;
            pending_f_reqs[pe_idx].waiting_for_dram = true;
        }
    }
}

void PECluster::notify_a_load_complete(int pe_idx, uint64_t addr) {
    if (pe_idx < 0 || pe_idx >= PES_PER_CLUSTER) {
        return;
    }
    auto& req = pending_a_load_reqs[pe_idx];
    if (req.active && req.addr == addr && req.waiting_for_dram) {
        req.waiting_for_dram = false;
        const uint64_t next_addr = req.addr + 64ULL;
        if (next_addr < req.end_addr) {
            req.addr = next_addr;
            req.needs_submit = true;
        } else {
            req.needs_submit = false;
        }
    }
}

bool PECluster::has_pending_a_load_requests() const {
    for (int i = 0; i < PES_PER_CLUSTER; i++) {
        if (pending_a_load_reqs[i].active && pending_a_load_reqs[i].needs_submit) {
            return true;
        }
    }
    return false;
}

void PECluster::get_pending_a_load_request(int& pe_idx, uint64_t& addr, int& bytes, int& row_id) const {
    for (int i = 0; i < PES_PER_CLUSTER; i++) {
        if (pending_a_load_reqs[i].active && pending_a_load_reqs[i].needs_submit) {
            pe_idx = i;
            addr = pending_a_load_reqs[i].addr;
            bytes = 64;
            row_id = pending_a_load_reqs[i].row_id;
            return;
        }
    }
    pe_idx = -1;
    addr = 0;
    bytes = 0;
    row_id = -1;
}

void PECluster::set_waiting_for_a_load_dram(int pe_idx, uint64_t addr) {
    if (pe_idx < 0 || pe_idx >= PES_PER_CLUSTER) {
        return;
    }
    auto& req = pending_a_load_reqs[pe_idx];
    if (req.active && req.addr == addr) {
        req.needs_submit = false;
        req.waiting_for_dram = true;
    }
}

#endif // ENABLE_DRAMSIM3_PE
