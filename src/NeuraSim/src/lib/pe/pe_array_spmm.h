// SPDX-License-Identifier: Apache-2.0
#ifndef PE_ARRAY_SPMM_H
#define PE_ARRAY_SPMM_H

#include <vector>
#include <map>
#include <queue>
#include <functional>
#include <cstdint>
#include "pe_cluster.h"
#include "work_stealing_scheduler.h"
#include "shared_f_cache.h"

#ifndef DESPOINA_V2_CACHE_EPOCH
#define DESPOINA_V2_CACHE_EPOCH 0
#endif
#include "hagg_staging_policy.h"

// DRAMsim3 integration
#ifdef ENABLE_DRAMSIM3_PE
#include "dramsim3.h"
#endif

// Forward declaration - GlobalCounters is optional for standalone tests
class GlobalCounters;

/**
 * PE Array SpMM Engine
 * 
 * Main controller for PE Array-based SpMM computation.
 * Coordinates 64 clusters (256 PEs total) to compute Y = A × F.
 * 
 * From config.py:
 * - pe_count: 256 (total PEs)
 * - pes_per_cluster: 4 (4 PEs share F buffer)
 * - num_clusters: 64 (256 / 4)
 * 
 * Features:
 * - CSR format input for A matrix
 * - Work stealing for load balancing
 * - F row caching in clusters
 * - Cycle-accurate simulation
 */
class PEArraySpMM {
public:
    // Configuration
#ifndef PE_NUM_CLUSTERS
#define PE_NUM_CLUSTERS 64
#endif
    static constexpr int NUM_CLUSTERS = PE_NUM_CLUSTERS;
    static constexpr int TOTAL_PES = NUM_CLUSTERS * PECluster::PES_PER_CLUSTER;  // 256
    
    // ==========================================================================
    // Output Egress Bandwidth Parameters (hardware-realistic)
    // ==========================================================================
    // Models the physical write port bandwidth from PE clusters to output staging buffer.
    // Uses bytes/cycle credit system so fractional rows/cycle can be modeled.
    //
    // Default: 512B/cycle = 1 row/cycle for d=128 (matches dual-port 256B SRAM)
    // Override at compile time: -DPE_OUTPUT_EGRESS_BYTES_PER_CYCLE=<bytes>
#ifndef PE_OUTPUT_EGRESS_BYTES_PER_CYCLE
#define PE_OUTPUT_EGRESS_BYTES_PER_CYCLE 512
#endif
    static constexpr int EGRESS_BYTES_PER_CYCLE = PE_OUTPUT_EGRESS_BYTES_PER_CYCLE;
    
    // State machine states
    enum State {
        IDLE,
        LOADING,      // Loading CSR matrix
        COMPUTING,    // SpMM computation in progress
        DRAINING,     // Waiting for clusters to finish
        DONE
    };

    // Constructor
    PEArraySpMM();
    PEArraySpMM(GlobalCounters* gcp);
    ~PEArraySpMM();
    
    void init(GlobalCounters* gcp);
    
    // Load matrices for SpMM: Y = A × F
    // @param row_ptr: CSR row pointers (n+1 elements)
    // @param col_idx: CSR column indices (nnz elements)
    // @param values: CSR values (nnz elements)
    // @param n: Number of rows in A (and F)
    // @param nnz: Number of non-zeros in A
    // @param f_matrix: Dense F matrix (n × d, row-major)
    // @param d: Feature dimension
    void load_spmm(const int* row_ptr, const int* col_idx, const float* values,
                   int n, int nnz, const float* f_matrix, int d);

    // Same as load_spmm(), but optionally keeps the shared feature cache contents across calls.
    // This is important for multi-phase incremental execution (Phase2↔Phase3 reuse).
    void load_spmm(const int* row_ptr, const int* col_idx, const float* values,
                   int n, int nnz, const float* f_matrix, int d,
                   bool keep_shared_f_cache);

    // v2 cache identity. Cache contents may survive a job boundary only when
    // the caller proves the dense input matrix identity by reusing epoch.
    void load_spmm(const int* row_ptr, const int* col_idx, const float* values,
                   int n, int nnz, const float* f_matrix, int d,
                   bool keep_shared_f_cache, uint64_t matrix_epoch);
    
    // Main simulation tick - call every cycle
    // Returns true if computation is still in progress
    bool tick();
    
    // Check if SpMM is complete
    bool is_done() const { return state == DONE; }
    
    // Get output matrix Y (n × d, row-major)
    const float* get_output() const { return y_matrix.data(); }

    // ΔF coherence: invalidate cached F rows (shared + per-cluster local MRU).
    void invalidate_f_rows(const std::vector<int>& row_ids);
    
    // Get total cycles taken
    long long get_total_cycles() const { return total_cycles; }
    
    // Row-wise streaming support
    // Check if any row has been completed this cycle
    bool has_completed_row() const { return !completed_rows.empty(); }
    
    // Get completed row (returns row_id, copies data to buffer)
    // Returns -1 if no completed row
    int pop_completed_row(float* buffer, int max_dim);

    void set_streaming_output_enabled(bool enabled) { streaming_output_enabled = enabled; }
    bool get_streaming_output_enabled() const { return streaming_output_enabled; }
    long long get_staging_full_stall_cycles() const { return staging_full_stall_cycles; }
    std::size_t get_staging_max_occupancy_rows() const { return staging_max_occupancy_rows; }
    std::size_t get_staging_current_occupancy_rows() const { return completed_rows.size(); }
    
    // Get feature dimension
    int get_feature_dim() const { return f_d; }
    
    // Get current state
    State get_state() const { return state; }
    
    // Statistics
    struct Stats {
        long long total_cycles;
        long long compute_cycles;
        long long stall_cycles;
        long long memory_stall_cycles;  // DRAM latency stalls
        long long total_mac_ops;
        long long f_cache_hits;
        long long f_cache_misses;
        long long l1_f_cache_hits;
        long long l2_shared_hits;
        long long l2_shared_hits_pre_submit;
        long long l1_bank_conflict_stalls;
        long long l2_bank_conflict_stalls;
        long long dram_fills;
        long long dram_requests;
        long long a_load_dram_requests;
        long long a_load_dram_latency_sum;
        long long dram_row_buffer_hits;
        long long shared_f_cache_hits;
        long long shared_f_cache_misses;
        long long output_egress_bytes;
        long long output_egress_rows;
        long long output_egress_stall_cycles;
        long long output_cluster_backpressure_stalls;
        long long work_steals;
        long long noc_steal_latency_total_cycles;
        long long noc_steal_count;
        float noc_steal_avg_hops;
        float pe_utilization;
        float load_imbalance;
    };
    Stats get_stats() const;
#ifdef ENABLE_DRAMSIM3_PE
    dramsim3::CommandEnergySnapshot get_pe_hbm_command_energy() const {
        return dram_system != nullptr
            ? dram_system->GetCommandEnergySnapshot()
            : dramsim3::CommandEnergySnapshot{};
    }
#endif
    
    // Reset for new computation. Multi-phase callers can preserve per-cluster
    // local MRUs when shared-cache reuse is requested across phase boundaries.
    void reset(bool clear_local_cache = true);

private:
    GlobalCounters* gcp;
    State state;

    // Shared (chip-level) feature cache to reduce cross-cluster DRAM fetch duplication.
    // Configured on load_spmm() based on feature dim and target capacity.
    SharedFCache shared_f_cache;
    int shared_f_cache_configured_dim = 0;
    std::size_t shared_f_cache_configured_entries = 0;
    uint64_t shared_f_cache_matrix_epoch = 0;
    bool shared_f_cache_epoch_valid = false;
    
    // Clusters
    PECluster clusters[NUM_CLUSTERS];
    
    // Work stealing scheduler
    WorkStealingScheduler scheduler;
    WorkStealingScheduler::CSRMatrix csr;
    
    // F matrix reference
    const float* f_matrix;
    int f_n;
    int f_d;
    
    // Output matrix
    std::vector<float> y_matrix;
    
    // Cycle counters
    long long total_cycles;
    long long compute_cycles;
    long long stall_cycles;
    
    // Pending outputs
    int outputs_received;
    int outputs_expected;
    
    // Row-wise streaming: completed rows queue
    struct CompletedRow {
        int row_id;
        std::vector<float> data;
    };
    std::queue<CompletedRow> completed_rows;
    bool streaming_output_enabled;
    long long staging_full_stall_cycles;
    std::size_t staging_max_occupancy_rows;
    
    // Output egress state (round-robin arbitration + credit system)
    int egress_last_cluster;
    int egress_bytes_credit;  // Accumulates EGRESS_BYTES_PER_CYCLE each tick
    
    // Egress statistics (for hardware bottleneck analysis)
    long long egress_bytes_total;         // Total bytes actually drained
    long long egress_rows_total;          // Total rows drained
    long long egress_stall_cycles;        // Cycles stalled due to empty clusters
    long long egress_backpressure_cycles; // Cycles backpressured by cluster-side finite queues
    
    // Internal methods
    void distribute_initial_work();
    void process_outputs();
    bool all_clusters_done() const;
    void feed_clusters();

#ifdef ENABLE_DRAMSIM3_PE
    // DRAMsim3 integration for accurate memory latency
    dramsim3::MemorySystem* dram_system;
    std::string dram_config_file;
    std::string dram_output_dir;

    // Separate DRAMsim3 model for A-load transactions.
    dramsim3::MemorySystem* dram_system_a_load;
    std::string dram_output_dir_a_load;
    
    // Clock domain ratio: NC runs at 1 GHz (1ns), but DRAMsim3 may run faster.
    // clock_ratio = NC_CLOCK_PERIOD_NS / DRAM_tCK_NS
    // e.g., HBM4 (tCK=0.5ns) -> clock_ratio=2.0, tick DRAMsim3 twice per NC cycle
    double dram_clock_ratio;
    double dram_tick_accumulator;       // Fractional tick accumulator
    double dram_tick_accumulator_a_load;
    
    // Pending DRAM requests: address -> (cluster_id, col_idx, start_cycle)
    struct DRAMRequest {
        int cluster_id;
        int col_idx;       // F row being fetched
        long long start_cycle;
    };
    std::map<uint64_t, DRAMRequest> pending_dram_requests;
    
    // Piggybacked requests: address -> list of (cluster_id, col_idx)
    struct PiggybackedRequest {
        int cluster_id;
        int col_idx;
    };
    std::map<uint64_t, std::vector<PiggybackedRequest>> piggybacked_requests;

    // Pending A-load requests: address -> (cluster_id, pe_idx, row_id, bytes, start_cycle)
    struct ALoadRequest {
        int cluster_id;
        int pe_idx;
        int row_id;
        int bytes;
        long long start_cycle;
    };
    std::map<uint64_t, ALoadRequest> pending_a_load_requests;

    struct ALoadPiggybackedRequest {
        int cluster_id;
        int pe_idx;
        int row_id;
        int bytes;
    };
    std::map<uint64_t, std::vector<ALoadPiggybackedRequest>> piggybacked_a_load_requests;
    
    // MSHR/CAM lookup overhead (realistic hardware model)
    static constexpr int CAM_LOOKUP_LATENCY = 2;  // CAM lookup + arbitration
#ifndef PE_MSHR_SIZE
#define PE_MSHR_SIZE 64
#endif
    static constexpr int MSHR_SIZE = PE_MSHR_SIZE;
    static constexpr int MAX_PIGGYBACK_PER_MSHR = 8;  // Max waiters per MSHR entry
    static constexpr int DATA_BROADCAST_LATENCY = 4;  // Cycles to broadcast data to clusters
    
    struct PendingCAMLookup {
        int cluster_id;
        int pe_idx;
        int col_idx;
        int remaining_cycles;
    };
    std::vector<PendingCAMLookup> cam_lookup_queue;
    long long cam_lookup_cycles_total;  // Total CAM lookup overhead
    long long mshr_full_stalls;         // Stalls due to MSHR full
    long long piggyback_overflow;       // Requests that couldn't piggyback (list full)
    
    // Statistics
    long long dram_requests_total;
    long long dram_latency_sum;
    long long dram_row_buffer_hits;
    long long dram_requests_a_load_total;
    long long dram_latency_a_load_sum;
    
    // Callbacks
    void dram_read_callback(uint64_t addr);
    void dram_write_callback(uint64_t addr);
    void dram_read_callback_a_load(uint64_t addr);
    void dram_write_callback_a_load(uint64_t addr);
    
    // Submit DRAM read request for F row
    void submit_dram_request(int cluster_id, int col_idx);
    
    // Tick DRAMsim3
    void tick_dram();
    void tick_dram_a_load();
    
    // Check if DRAM request completed
    bool check_dram_completion(int cluster_id, int col_idx);
    
public:
    // Get DRAM statistics
    long long get_dram_requests() const { return dram_requests_total; }
    double get_avg_dram_latency() const { 
        return dram_requests_total > 0 ? (double)dram_latency_sum / dram_requests_total : 0; 
    }
#endif
};

#endif // PE_ARRAY_SPMM_H
