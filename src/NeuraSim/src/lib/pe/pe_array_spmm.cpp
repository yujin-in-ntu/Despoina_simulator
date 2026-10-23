// SPDX-License-Identifier: Apache-2.0
/**
 * PE Array SpMM Engine Implementation
 * 
 * Cycle-accurate simulation of PE Array-based SpMM.
 * Coordinates 64 clusters (256 PEs) with work stealing.
 */

#include "pe_array_spmm.h"
#include "finalized_dcim_v3.h"
#include <cstdint>
#include <cstring>
#include <algorithm>
#include <iostream>
#ifdef _WIN32
#include <direct.h>
#else
#include <sys/stat.h>
#endif

#ifdef _WIN32
#define CREATE_DIR(path) _mkdir((path))
#else
#define CREATE_DIR(path) mkdir((path), 0777)
#endif

#ifdef ENABLE_DRAMSIM3_PE
#include "../counters/globalcounters.h"
#endif

PEArraySpMM::PEArraySpMM()
    : gcp(nullptr), state(IDLE), f_matrix(nullptr), f_n(0), f_d(0),
      total_cycles(0), compute_cycles(0), stall_cycles(0),
      outputs_received(0), outputs_expected(0), streaming_output_enabled(false),
      staging_full_stall_cycles(0), staging_max_occupancy_rows(0)
#ifdef ENABLE_DRAMSIM3_PE
      , dram_system(nullptr), dram_system_a_load(nullptr)
#endif
{
}

PEArraySpMM::PEArraySpMM(GlobalCounters* gcp) : PEArraySpMM() {
    init(gcp);
}

PEArraySpMM::~PEArraySpMM() {
    // Clear output matrix
    y_matrix.clear();
#ifdef ENABLE_DRAMSIM3_PE
    delete dram_system;
    dram_system = nullptr;
    delete dram_system_a_load;
    dram_system_a_load = nullptr;
#endif
}

void PEArraySpMM::init(GlobalCounters* gcp) {
    this->gcp = gcp;
    
    // Initialize clusters
    for (int i = 0; i < NUM_CLUSTERS; i++) {
        clusters[i].init(gcp, i);
    }
    
    // Initialize scheduler
    scheduler.init(NUM_CLUSTERS);
    
#ifdef ENABLE_DRAMSIM3_PE
    // Initialize DRAMsim3
    // Use runtime-selected DRAM config (CLI --dram-config / --memory-type).
    if (this->gcp != nullptr) {
        dram_config_file = this->gcp->get_dram_config_path();
    } else {
        dram_config_file = "./ktb_dram_configs/ktb_HBM2_8Gb_x128.ini";
    }
    // Put DRAMsim3 output under results directory for reproducibility.
    if (this->gcp != nullptr) {
        dram_output_dir = this->gcp->get_results_dir() + "/dramsim3_pe_output";
    } else {
        dram_output_dir = "./dramsim3_pe_output";
    }
    CREATE_DIR(dram_output_dir.c_str());
    
    dram_system = dramsim3::GetMemorySystem(
        dram_config_file,
        dram_output_dir,
        std::bind(&PEArraySpMM::dram_read_callback, this, std::placeholders::_1),
        std::bind(&PEArraySpMM::dram_write_callback, this, std::placeholders::_1)
    );
    
    dram_requests_total = 0;
    dram_latency_sum = 0;
    dram_row_buffer_hits = 0;
    dram_requests_a_load_total = 0;
    dram_latency_a_load_sum = 0;
    cam_lookup_cycles_total = 0;
    mshr_full_stalls = 0;
    piggyback_overflow = 0;

    if (this->gcp != nullptr) {
        dram_output_dir_a_load = this->gcp->get_results_dir() + "/dramsim3_a_load_output";
    } else {
        dram_output_dir_a_load = "./dramsim3_a_load_output";
    }
    CREATE_DIR(dram_output_dir_a_load.c_str());
    dram_system_a_load = dramsim3::GetMemorySystem(
        dram_config_file,
        dram_output_dir_a_load,
        std::bind(&PEArraySpMM::dram_read_callback_a_load, this, std::placeholders::_1),
        std::bind(&PEArraySpMM::dram_write_callback_a_load, this, std::placeholders::_1)
    );
    
    // Calculate clock ratio: NC clock (1.0 ns) / DRAM tCK
    // HBM2: tCK=1.0ns -> ratio=1.0, HBM3: tCK=0.625ns -> ratio=1.6, HBM4: tCK=0.5ns -> ratio=2.0
    constexpr double NC_CLOCK_PERIOD_NS = 1.0;
    double dram_tck_ns = (this->gcp != nullptr) ? this->gcp->get_memory_tck_ns() : 1.0;
    dram_clock_ratio = NC_CLOCK_PERIOD_NS / dram_tck_ns;
    dram_tick_accumulator = 0.0;
    dram_tick_accumulator_a_load = 0.0;
    
    std::string memory_type = (this->gcp != nullptr) ? this->gcp->get_memory_type() : "hbm2";
    std::cout << "[PE Array] DRAMsim3 initialized (" << memory_type 
              << ", tCK=" << dram_tck_ns << "ns, clock_ratio=" << dram_clock_ratio
              << ", MSHR=" << MSHR_SIZE 
              << ", MaxPiggyback=" << MAX_PIGGYBACK_PER_MSHR << ")" << std::endl;
#endif
    
    reset();
}

void PEArraySpMM::load_spmm(const int* row_ptr, const int* col_idx, const float* values,
                            int n, int nnz, const float* f_matrix, int d) {
    load_spmm(row_ptr, col_idx, values, n, nnz, f_matrix, d, false);
}

void PEArraySpMM::load_spmm(const int* row_ptr, const int* col_idx, const float* values,
                            int n, int nnz, const float* f_matrix, int d,
                            bool keep_shared_f_cache) {
    load_spmm(row_ptr, col_idx, values, n, nnz, f_matrix, d,
              keep_shared_f_cache, 0);
}

void PEArraySpMM::load_spmm(const int* row_ptr, const int* col_idx, const float* values,
                            int n, int nnz, const float* f_matrix, int d,
                            bool keep_shared_f_cache, uint64_t matrix_epoch) {
#if DESPOINA_V3_CACHE_EPOCH
    const bool exact_epoch_reuse = despoina_v3::cache_epoch_can_reuse(
        keep_shared_f_cache, shared_f_cache_epoch_valid,
        shared_f_cache_matrix_epoch, matrix_epoch);
#else
    const bool exact_epoch_reuse = keep_shared_f_cache;
    (void)matrix_epoch;
#endif
    // Reset state
    reset(/*clear_local_cache=*/!exact_epoch_reuse);
    
    // Store F matrix reference
    this->f_matrix = f_matrix;
    this->f_n = n;
    this->f_d = d;

    // Configure shared cache capacity (~4MB for FP32 row=128 floats => 512B/row => 8192 rows).
    // This models a chip-level shared feature cache (distinct from per-cluster tiny buffers).
    // Override at compile time: -DPE_SHARED_F_CACHE_ENTRIES=4096
#ifndef PE_SHARED_F_CACHE_ENTRIES
#define PE_SHARED_F_CACHE_ENTRIES 8192
#endif
    const std::size_t shared_entries = PE_SHARED_F_CACHE_ENTRIES;
    if (!exact_epoch_reuse ||
        shared_f_cache_configured_dim != d ||
        shared_f_cache_configured_entries != shared_entries) {
        shared_f_cache.configure(shared_entries, d);
        shared_f_cache_configured_dim = d;
        shared_f_cache_configured_entries = shared_entries;
    }
#if DESPOINA_V3_CACHE_EPOCH
    shared_f_cache_matrix_epoch = matrix_epoch;
    shared_f_cache_epoch_valid = true;
#endif
    // New job: reset shared-cache hit/miss counters without flushing data.
    shared_f_cache.reset_stats();
    
    // Copy CSR data
    csr.n = n;
    csr.nnz = nnz;
    csr.row_ptr.assign(row_ptr, row_ptr + n + 1);
    csr.col_idx.assign(col_idx, col_idx + nnz);
    csr.values.assign(values, values + nnz);
    
    // Allocate output matrix (initialized to zero)
    y_matrix.resize(n * d, 0.0f);
    
    // Set F matrix for all clusters
    for (int i = 0; i < NUM_CLUSTERS; i++) {
        clusters[i].set_f_matrix(f_matrix, n, d);
        clusters[i].set_shared_f_cache(&shared_f_cache);
    }
    
    // Count expected outputs (non-empty rows)
    outputs_expected = 0;
    for (int i = 0; i < n; i++) {
        if (row_ptr[i + 1] > row_ptr[i]) {
            outputs_expected++;
        }
    }
    
    // Load matrix into scheduler
    scheduler.load_matrix(csr);
    
    // Distribute initial work
    distribute_initial_work();
    
    state = COMPUTING;
}

void PEArraySpMM::invalidate_f_rows(const std::vector<int>& row_ids) {
    for (int row_id : row_ids) {
        shared_f_cache.invalidate(row_id);
    }
    for (int i = 0; i < NUM_CLUSTERS; i++) {
        for (int row_id : row_ids) {
            clusters[i].invalidate_local_f_row(row_id);
        }
    }
}

bool PEArraySpMM::tick() {
    if (state == DONE || state == IDLE) {
        return false;
    }
    
    total_cycles++;
    bool any_work = false;

    // Advance pending NoC work-steal transfers.
    scheduler.tick();
    
#ifdef ENABLE_DRAMSIM3_PE
    // Tick DRAMsim3 with proper clock ratio.
    // NC runs at 1 GHz (1ns), but DRAMsim3 may run faster (HBM3/4).
    // Use accumulator to handle fractional ticks.
    if (dram_system != nullptr) {
        dram_tick_accumulator += dram_clock_ratio;
        while (dram_tick_accumulator >= 1.0) {
            dram_system->ClockTick();
            dram_tick_accumulator -= 1.0;
        }
    }
    if (dram_system_a_load != nullptr) {
        dram_tick_accumulator_a_load += dram_clock_ratio;
        while (dram_tick_accumulator_a_load >= 1.0) {
            dram_system_a_load->ClockTick();
            dram_tick_accumulator_a_load -= 1.0;
        }
    }
    if (dram_system != nullptr) {
        
        // =========================================================================
        // STEP 1: Process CAM lookup queue (decrement remaining cycles)
        // =========================================================================
        int requests_this_cycle = 0;
        const int MAX_REQUESTS_PER_CYCLE = 8;
        
        for (auto it = cam_lookup_queue.begin(); it != cam_lookup_queue.end(); ) {
            it->remaining_cycles--;
            
            if (it->remaining_cycles <= 0 && requests_this_cycle < MAX_REQUESTS_PER_CYCLE) {
                // CAM lookup complete - now check pending and submit
                int cluster_id = it->cluster_id;
                int col_idx = it->col_idx;
                
                uint64_t addr = static_cast<uint64_t>(col_idx) * f_d * sizeof(float);
                addr = (addr / 64) * 64;  // Align to cache line
                
                auto pending_it = pending_dram_requests.find(addr);
                if (pending_it == pending_dram_requests.end()) {
                    // No pending request for this address - need new MSHR entry
                    
                    // CHECK: MSHR size limit (realistic HW constraint)
                    if (pending_dram_requests.size() >= MSHR_SIZE) {
                        // MSHR full - must wait for an entry to free up
                        mshr_full_stalls++;
                        it->remaining_cycles = 1;  // Retry next cycle
                        ++it;
                        continue;
                    }
                    
                    if (dram_system->WillAcceptTransaction(addr, false)) {
                        dram_system->AddTransaction(addr, false);
                        
                        DRAMRequest req;
                        req.cluster_id = cluster_id;
                        req.col_idx = col_idx;
                        req.start_cycle = total_cycles;
                        pending_dram_requests[addr] = req;
                        
                        dram_requests_total++;
                        requests_this_cycle++;
                        cam_lookup_cycles_total += CAM_LOOKUP_LATENCY;
                        
                        // Debug: print first few submissions
                        static int submit_count = 0;
                        if (submit_count < 10) {
                            std::cerr << "[DRAMsim3] Submit #" << submit_count 
                                      << ": cluster=" << cluster_id << ", col=" << col_idx 
                                      << ", addr=" << addr << " (MSHR: " << pending_dram_requests.size() << "/" << MSHR_SIZE << ")" << std::endl;
                        }
                        submit_count++;
                        
                        it = cam_lookup_queue.erase(it);
                        continue;
                    }
                    // WillAcceptTransaction false - keep in queue, retry next cycle
                    it->remaining_cycles = 1;  // Retry next cycle
                    ++it;
                } else {
                    // Another cluster already requested this address - try to piggyback
                    auto& pb_list = piggybacked_requests[addr];
                    
                    // CHECK: Piggybacking limit per MSHR entry (realistic HW constraint)
                    if (pb_list.size() >= MAX_PIGGYBACK_PER_MSHR) {
                        // Piggyback list full - must wait for DRAM to complete
                        piggyback_overflow++;
                        it->remaining_cycles = 1;  // Retry next cycle
                        ++it;
                        continue;
                    }
                    
                    pb_list.push_back({cluster_id, col_idx});
                    cam_lookup_cycles_total += CAM_LOOKUP_LATENCY;
                    
                    // Debug: count piggybacked requests
                    static int piggyback_count = 0;
                    piggyback_count++;
                    if (piggyback_count <= 5 || piggyback_count % 1000 == 0) {
                        std::cerr << "[Piggyback] count=" << piggyback_count << ", addr=" << addr 
                                  << " (waiters: " << pb_list.size() << "/" << MAX_PIGGYBACK_PER_MSHR << ")" << std::endl;
                    }
                    
                    it = cam_lookup_queue.erase(it);
                }
            } else {
                ++it;
            }
        }
        
        // =========================================================================
        // STEP 2: Collect new requests from clusters into CAM lookup queue
        // =========================================================================
        const int MAX_NEW_REQUESTS_PER_CYCLE = 8;  // Limit new requests entering pipeline
        int new_requests = 0;
        
        for (int i = 0; i < NUM_CLUSTERS && new_requests < MAX_NEW_REQUESTS_PER_CYCLE; i++) {
            if (clusters[i].has_pending_dram_requests()) {
                int pe_idx, col_idx;
                clusters[i].get_pending_dram_request(pe_idx, col_idx);
                
                if (pe_idx >= 0 && col_idx >= 0) {
                    // Add to CAM lookup queue with latency
                    PendingCAMLookup lookup;
                    lookup.cluster_id = i;
                    lookup.pe_idx = pe_idx;
                    lookup.col_idx = col_idx;
                    lookup.remaining_cycles = CAM_LOOKUP_LATENCY;
                    cam_lookup_queue.push_back(lookup);
                    
                    // Mark this PE as waiting for DRAM callback
                    clusters[i].set_waiting_for_dram(pe_idx, col_idx);
                    new_requests++;
                }
            }
        }

        // =========================================================================
        // STEP 3: Submit A-load DRAM transactions (separate DRAMsim3 instance)
        // =========================================================================
        if (dram_system_a_load != nullptr) {
            for (int i = 0; i < NUM_CLUSTERS; i++) {
                if (!clusters[i].has_pending_a_load_requests()) {
                    continue;
                }

                int pe_idx = -1;
                uint64_t addr = 0;
                int bytes = 0;
                int row_id = -1;
                clusters[i].get_pending_a_load_request(pe_idx, addr, bytes, row_id);
                if (pe_idx < 0) {
                    continue;
                }

                auto pending_it = pending_a_load_requests.find(addr);
                if (pending_it == pending_a_load_requests.end()) {
                    if (pending_a_load_requests.size() >= MSHR_SIZE) {
                        continue;
                    }
                    if (!dram_system_a_load->WillAcceptTransaction(addr, false)) {
                        continue;
                    }
                    dram_system_a_load->AddTransaction(addr, false);

                    ALoadRequest req;
                    req.cluster_id = i;
                    req.pe_idx = pe_idx;
                    req.row_id = row_id;
                    req.bytes = bytes;
                    req.start_cycle = total_cycles;
                    pending_a_load_requests[addr] = req;
                    clusters[i].set_waiting_for_a_load_dram(pe_idx, addr);
                    dram_requests_a_load_total++;
                } else {
                    auto& pb_list = piggybacked_a_load_requests[addr];
                    if (pb_list.size() >= MAX_PIGGYBACK_PER_MSHR) {
                        continue;
                    }
                    pb_list.push_back({i, pe_idx, row_id, bytes});
                    clusters[i].set_waiting_for_a_load_dram(pe_idx, addr);
                }
            }
        }
    }
#endif
    
    // Tick all clusters
    for (int i = 0; i < NUM_CLUSTERS; i++) {
        if (clusters[i].tick()) {
            any_work = true;
        }
    }
    
    if (any_work) {
        compute_cycles++;
    } else {
        stall_cycles++;
    }
    
    // Collect outputs from clusters
    process_outputs();
    
    // Feed more work to clusters that need it
    feed_clusters();
    
    // Check if we're done
    if (state == COMPUTING) {
        if (scheduler.all_done() && all_clusters_done()) {
            state = DRAINING;
        }
    }
    
    if (state == DRAINING) {
        // Wait for all outputs
        if (outputs_received >= outputs_expected) {
            state = DONE;
            return false;
        }
    }
    
    return true;
}

PEArraySpMM::Stats PEArraySpMM::get_stats() const {
    Stats stats;
    stats.total_cycles = total_cycles;
    stats.compute_cycles = compute_cycles;
    stats.stall_cycles = stall_cycles;
    stats.memory_stall_cycles = 0;
    stats.total_mac_ops = 0;
    stats.f_cache_hits = 0;
    stats.f_cache_misses = 0;
    stats.l1_f_cache_hits = 0;
    stats.l2_shared_hits = 0;
    stats.l2_shared_hits_pre_submit = 0;
    stats.l1_bank_conflict_stalls = 0;
    stats.l2_bank_conflict_stalls = 0;
    stats.dram_fills = 0;
    stats.dram_requests = 0;
    stats.a_load_dram_requests = 0;
    stats.a_load_dram_latency_sum = 0;
    stats.dram_row_buffer_hits = 0;
    stats.shared_f_cache_hits = 0;
    stats.shared_f_cache_misses = 0;
    stats.output_egress_bytes = egress_bytes_total;
    stats.output_egress_rows = egress_rows_total;
    stats.output_egress_stall_cycles = egress_stall_cycles;
    stats.output_cluster_backpressure_stalls = 0;
    stats.noc_steal_latency_total_cycles = 0;
    stats.noc_steal_count = 0;
    stats.noc_steal_avg_hops = 0.0f;
    
    for (int i = 0; i < NUM_CLUSTERS; i++) {
        stats.total_mac_ops += clusters[i].get_total_mac_ops();
        stats.f_cache_hits += clusters[i].get_f_cache_hits();
        stats.f_cache_misses += clusters[i].get_f_cache_misses();
        stats.memory_stall_cycles += clusters[i].get_memory_stall_cycles();
        stats.l1_f_cache_hits += clusters[i].get_l1_f_cache_hits();
        stats.l2_shared_hits += clusters[i].get_l2_shared_hits();
        stats.l2_shared_hits_pre_submit += clusters[i].get_l2_shared_hits_pre_submit();
        stats.l1_bank_conflict_stalls += clusters[i].get_l1_bank_conflict_stalls();
        stats.l2_bank_conflict_stalls += clusters[i].get_l2_bank_conflict_stalls();
        stats.dram_fills += clusters[i].get_dram_fills();
        stats.output_cluster_backpressure_stalls += clusters[i].get_output_backpressure_stalls();
    }
    
    stats.work_steals = scheduler.get_steals_count();
    stats.noc_steal_latency_total_cycles = scheduler.get_noc_steal_latency_total_cycles();
    stats.noc_steal_count = scheduler.get_noc_steal_count();
    stats.noc_steal_avg_hops = scheduler.get_noc_steal_avg_hops();
    stats.pe_utilization = total_cycles > 0 ? 
        (float)compute_cycles / total_cycles : 0.0f;
    stats.load_imbalance = scheduler.get_load_imbalance();
    
#ifdef ENABLE_DRAMSIM3_PE
    stats.dram_requests = dram_requests_total;
    stats.a_load_dram_requests = dram_requests_a_load_total;
    stats.a_load_dram_latency_sum = dram_latency_a_load_sum;
    stats.dram_row_buffer_hits = dram_row_buffer_hits;
    stats.shared_f_cache_hits = shared_f_cache.hits();
    stats.shared_f_cache_misses = shared_f_cache.misses();

    // Print DRAMsim3 statistics
    std::cout << "\n========== DRAMsim3 Statistics ==========" << std::endl;
    std::cout << "Total DRAM Requests:    " << dram_requests_total << std::endl;
    std::cout << "Avg DRAM Latency:       " << (dram_requests_total > 0 ? dram_latency_sum / dram_requests_total : 0) << " cycles" << std::endl;
    std::cout << "Row Buffer Hits:        " << dram_row_buffer_hits << std::endl;
    std::cout << "CAM Lookup Overhead:    " << cam_lookup_cycles_total << " cycles (total)" << std::endl;
    std::cout << "MSHR Full Stalls:       " << mshr_full_stalls << " (MSHR=" << MSHR_SIZE << ")" << std::endl;
    std::cout << "Piggyback Overflow:     " << piggyback_overflow << " (max=" << MAX_PIGGYBACK_PER_MSHR << "/entry)" << std::endl;
    std::cout << "F Cache Hits:           " << stats.f_cache_hits << std::endl;
    std::cout << "F Cache Misses:         " << stats.f_cache_misses << std::endl;
    std::cout << "Cache Hit Rate:         " << (stats.f_cache_hits + stats.f_cache_misses > 0 ? 
        100.0 * stats.f_cache_hits / (stats.f_cache_hits + stats.f_cache_misses) : 0) << "%" << std::endl;
    std::cout << "Shared F-Cache:         size=" << shared_f_cache.size()
              << "/" << shared_f_cache.capacity_entries()
              << ", hits=" << shared_f_cache.hits()
              << ", misses=" << shared_f_cache.misses() << std::endl;
    std::cout << "L1(local) hits:         " << stats.l1_f_cache_hits << std::endl;
    std::cout << "L2(shared) hits:        " << stats.l2_shared_hits
              << " (pre-submit=" << stats.l2_shared_hits_pre_submit << ")" << std::endl;
    std::cout << "L1 bank conflict stalls:" << stats.l1_bank_conflict_stalls << std::endl;
    std::cout << "L2 bank conflict stalls:" << stats.l2_bank_conflict_stalls << std::endl;
    std::cout << "NoC steal count:        " << stats.noc_steal_count << std::endl;
    std::cout << "NoC steal latency(sum): " << stats.noc_steal_latency_total_cycles << " cycles" << std::endl;
    std::cout << "NoC steal avg hops:     " << stats.noc_steal_avg_hops << std::endl;
    std::cout << "A-load DRAM requests:   " << stats.a_load_dram_requests << std::endl;
    std::cout << "A-load DRAM latency(sum): " << stats.a_load_dram_latency_sum << " cycles" << std::endl;
    std::cout << "DRAM fills (clusters):  " << stats.dram_fills << std::endl;
    std::cout << "==========================================\n" << std::endl;
    
    // Flush DRAMsim3 energy statistics to JSON file
    if (dram_system != nullptr) {
        dram_system->PrintStats();
        std::cout << "[DRAMsim3] Energy statistics saved to output directory" << std::endl;
    }
#endif
    
    return stats;
}

void PEArraySpMM::reset(bool clear_local_cache) {
    state = IDLE;
    total_cycles = 0;
    compute_cycles = 0;
    stall_cycles = 0;
    outputs_received = 0;
    outputs_expected = 0;
    y_matrix.clear();
    csr.row_ptr.clear();
    csr.col_idx.clear();
    csr.values.clear();
    csr.n = 0;
    csr.nnz = 0;

    // New SpMM job boundary: clear per-cluster state and counters so stats are per-job,
    // not accumulated across timesteps/phases.
    for (int i = 0; i < NUM_CLUSTERS; i++) {
        clusters[i].reset_for_new_spmm(clear_local_cache);
    }
    
#ifdef ENABLE_DRAMSIM3_PE
    // Clear pending DRAM request tracking
    pending_dram_requests.clear();
    piggybacked_requests.clear();
    pending_a_load_requests.clear();
    piggybacked_a_load_requests.clear();
    cam_lookup_queue.clear();
    cam_lookup_cycles_total = 0;
    mshr_full_stalls = 0;
    piggyback_overflow = 0;
    
    // Reset per-job DRAM statistics (FIX: was not being reset, causing accumulation across phases)
    dram_requests_total = 0;
    dram_latency_sum = 0;
    dram_row_buffer_hits = 0;
    dram_requests_a_load_total = 0;
    dram_latency_a_load_sum = 0;
#endif
    
    // Re-initialize scheduler
    scheduler.init(NUM_CLUSTERS);
    
    // Reset egress state
    egress_last_cluster = -1;
    egress_bytes_credit = 0;
    egress_bytes_total = 0;
    egress_rows_total = 0;
    egress_stall_cycles = 0;
    egress_backpressure_cycles = 0;

    while (!completed_rows.empty()) {
        completed_rows.pop();
    }
}

void PEArraySpMM::distribute_initial_work() {
    // Give initial work to each cluster
    for (int i = 0; i < NUM_CLUSTERS; i++) {
        if (clusters[i].can_accept_work()) {
            PECluster::WorkItem work;
            if (scheduler.get_work(i, work, csr)) {
                clusters[i].assign_work(work);
            }
        }
    }
}

void PEArraySpMM::process_outputs() {
    // =========================================================================
    // Hardware-realistic output egress model:
    // - Bytes/cycle credit system: accumulate EGRESS_BYTES_PER_CYCLE each tick
    // - Drain rows until credit exhausted (credit -= row_bytes)
    // - Round-robin arbitration across clusters for fairness
    // Backpressure is modeled at cluster finite output queues.
    // =========================================================================
    
    const int row_bytes = f_d * sizeof(float);  // Bytes per output row
    if (row_bytes <= 0) {
        return;
    }
    
    // Accumulate bandwidth credit for this cycle
    egress_bytes_credit += EGRESS_BYTES_PER_CYCLE;
    
    float output_buffer[PECluster::MAX_FEATURE_DIM];
    int row_id;
    int rows_drained = 0;
    bool any_cluster_has_output = false;
    
    // Round-robin: start from cluster after the last one served
    // Continue while we have credit for at least one row
    while (egress_bytes_credit >= row_bytes) {
		if (streaming_output_enabled &&
		    !pe_hagg_staging::can_accept(completed_rows.size())) {
			staging_full_stall_cycles++;
			egress_backpressure_cycles++;
			break;
		}
        
        bool drained_this_round = false;
        
        for (int offset = 0; offset < NUM_CLUSTERS; offset++) {
            int i = (egress_last_cluster + 1 + offset) % NUM_CLUSTERS;
            
            if (clusters[i].get_output(&row_id, output_buffer, f_d)) {
                any_cluster_has_output = true;
                
                // Check if we still have credit
                if (egress_bytes_credit < row_bytes) {
                    break;  // No more credit this cycle
                }
                
                // Consume credit
                egress_bytes_credit -= row_bytes;
                
                // Write to output matrix
                if (row_id >= 0 && row_id < f_n) {
                    float* y_row = y_matrix.data() + row_id * f_d;
                    for (int j = 0; j < f_d; j++) {
                        y_row[j] = output_buffer[j];
                    }
                    
                    if (streaming_output_enabled) {
                        CompletedRow cr;
                        cr.row_id = row_id;
                        cr.data.assign(output_buffer, output_buffer + f_d);
                        completed_rows.push(std::move(cr));
                        staging_max_occupancy_rows = std::max(
                            staging_max_occupancy_rows, completed_rows.size());
                    }
                }
                
                outputs_received++;
                rows_drained++;
                egress_last_cluster = i;
                egress_bytes_total += row_bytes;
                egress_rows_total++;
                drained_this_round = true;
                
                break;  // Move to next round-robin slot
            }
        }
        
        if (!drained_this_round) {
            break;  // No cluster had output ready
        }
    }
    
    // Track stall cycles (had credit but nothing to drain)
    if (rows_drained == 0 && !any_cluster_has_output) {
        egress_stall_cycles++;
    }
    
    // Cap accumulated credit to prevent unbounded growth during stalls
    // Max credit = 4 rows worth (prevents burst after long stall)
    int max_credit = 4 * row_bytes;
    if (egress_bytes_credit > max_credit) {
        egress_bytes_credit = max_credit;
    }
}

int PEArraySpMM::pop_completed_row(float* buffer, int max_dim) {
    if (completed_rows.empty()) {
        return -1;
    }
    
    CompletedRow& cr = completed_rows.front();
    int row_id = cr.row_id;
    
    // Copy data to buffer
    int copy_dim = std::min(max_dim, (int)cr.data.size());
    std::memcpy(buffer, cr.data.data(), copy_dim * sizeof(float));
    
    completed_rows.pop();
    return row_id;
}

bool PEArraySpMM::all_clusters_done() const {
    for (int i = 0; i < NUM_CLUSTERS; i++) {
        if (!clusters[i].is_done()) {
            return false;
        }
    }
    return true;
}

void PEArraySpMM::feed_clusters() {
    // Feed more work to clusters that can accept it
    for (int i = 0; i < NUM_CLUSTERS; i++) {
        // Each cluster can have multiple work items queued
        while (clusters[i].can_accept_work() && !scheduler.all_done()) {
            PECluster::WorkItem work;
            if (scheduler.get_work(i, work, csr)) {
                clusters[i].assign_work(work);
            } else {
                break;  // No more work available
            }
        }
    }
}

// =============================================================================
// DRAMsim3 Integration
// =============================================================================
#ifdef ENABLE_DRAMSIM3_PE

void PEArraySpMM::dram_read_callback(uint64_t addr) {
    // Find the pending request for this address
    auto it = pending_dram_requests.find(addr);
    if (it != pending_dram_requests.end()) {
        DRAMRequest& req = it->second;
        long long latency = total_cycles - req.start_cycle;
        dram_latency_sum += latency;
        
        // Debug: print first few callbacks
        static int callback_count = 0;
        if (callback_count < 10) {
            std::cerr << "[DRAMsim3] Callback #" << callback_count 
                      << ": addr=" << addr << ", latency=" << latency << " cycles" << std::endl;
        }
        callback_count++;
        
        // Track row buffer hits (latency < 50 cycles suggests row buffer hit)
        dram_row_buffer_hits += (latency < 50) ? 1 : 0;
        
        // Notify original requester cluster that F row is ready
        if (req.cluster_id >= 0 && req.cluster_id < NUM_CLUSTERS) {
            clusters[req.cluster_id].notify_dram_complete(req.col_idx, latency);
        }
        
        // Notify all piggybacked requests
        auto pb_it = piggybacked_requests.find(addr);
        if (pb_it != piggybacked_requests.end()) {
            for (const auto& pb_req : pb_it->second) {
                if (pb_req.cluster_id >= 0 && pb_req.cluster_id < NUM_CLUSTERS) {
                    clusters[pb_req.cluster_id].notify_dram_complete(pb_req.col_idx, latency);
                }
            }
            piggybacked_requests.erase(pb_it);
        }
        
        pending_dram_requests.erase(it);
    }
}

void PEArraySpMM::dram_write_callback(uint64_t addr) {
    // Write callbacks not used for F matrix reads
}

void PEArraySpMM::dram_read_callback_a_load(uint64_t addr) {
    auto it = pending_a_load_requests.find(addr);
    if (it == pending_a_load_requests.end()) {
        return;
    }

    const ALoadRequest& req = it->second;
    const long long latency = total_cycles - req.start_cycle;
    dram_latency_a_load_sum += latency;

    if (req.cluster_id >= 0 && req.cluster_id < NUM_CLUSTERS) {
        clusters[req.cluster_id].notify_a_load_complete(req.pe_idx, addr);
    }

    auto pb_it = piggybacked_a_load_requests.find(addr);
    if (pb_it != piggybacked_a_load_requests.end()) {
        for (const auto& pb_req : pb_it->second) {
            if (pb_req.cluster_id >= 0 && pb_req.cluster_id < NUM_CLUSTERS) {
                clusters[pb_req.cluster_id].notify_a_load_complete(pb_req.pe_idx, addr);
            }
        }
        piggybacked_a_load_requests.erase(pb_it);
    }

    pending_a_load_requests.erase(it);
}

void PEArraySpMM::dram_write_callback_a_load(uint64_t addr) {
    // A-load path models reads only.
}

void PEArraySpMM::submit_dram_request(int cluster_id, int col_idx) {
    // Calculate address for F row
    // Assume each float is 4 bytes, rows are contiguous
    uint64_t addr = static_cast<uint64_t>(col_idx) * f_d * sizeof(float);
    
    // Align to cache line (64 bytes)
    addr = (addr / 64) * 64;
    
    // Check if DRAM can accept transaction
    if (dram_system->WillAcceptTransaction(addr, false)) {
        dram_system->AddTransaction(addr, false);  // false = read
        
        DRAMRequest req;
        req.cluster_id = cluster_id;
        req.col_idx = col_idx;
        req.start_cycle = total_cycles;
        pending_dram_requests[addr] = req;
        
        dram_requests_total++;
    }
}

void PEArraySpMM::tick_dram() {
    dram_system->ClockTick();
}

void PEArraySpMM::tick_dram_a_load() {
    dram_system_a_load->ClockTick();
}

bool PEArraySpMM::check_dram_completion(int cluster_id, int col_idx) {
    // Check if the request for this F row has completed
    uint64_t addr = static_cast<uint64_t>(col_idx) * f_d * sizeof(float);
    addr = (addr / 64) * 64;
    
    return pending_dram_requests.find(addr) == pending_dram_requests.end();
}

#endif // ENABLE_DRAMSIM3_PE
