// SPDX-License-Identifier: Apache-2.0
#ifndef PE_CLUSTER_H
#define PE_CLUSTER_H

#include <vector>
#include <map>
#include <queue>
#include <array>
#include <cstdint>
#include "pe.h"

// Forward declaration - GlobalCounters is optional for standalone tests
class GlobalCounters;
class SharedFCache;

/**
 * PE Cluster - Groups 4 PEs with shared F buffer
 * 
 * From config.py ClusteringConfig:
 * - pes_per_cluster: 4
 * - shared_f_buffer_kb: 4 (4KB = 1024 floats)
 * - mru_cache_entries: 64 (MRU cache for F rows)
 * 
 * Benefits:
 * - F row sharing reduces DRAM bandwidth by ~40%
 * - MRU cache exploits locality in graph access patterns
 */
class PECluster {
public:
    // Configuration constants (from config.py ClusteringConfig)
    static constexpr int PES_PER_CLUSTER = 4;
    static constexpr int SHARED_F_BUFFER_KB = 4;
    static constexpr int SHARED_F_BUFFER_FLOATS = SHARED_F_BUFFER_KB * 1024 / sizeof(float);
    // L1 (per-cluster) MRU entries.
    // With SharedFCache enabled, this acts as a small L1 to avoid repeatedly
    // querying shared cache for short-term reuse within a cluster.
    //
    // To experiment, you can override at compile time:
    //   -DPE_L1_MRU_ENTRIES=256
#ifndef PE_L1_MRU_ENTRIES
#define PE_L1_MRU_ENTRIES 64
#endif
    static constexpr int MRU_CACHE_ENTRIES = PE_L1_MRU_ENTRIES;

#ifndef PE_L1_MRU_BANKS
#define PE_L1_MRU_BANKS 16
#endif
#ifndef PE_L1_MRU_PORTS_PER_BANK
#define PE_L1_MRU_PORTS_PER_BANK 1
#endif
    static constexpr int L1_NUM_BANKS = PE_L1_MRU_BANKS;
    static constexpr int L1_PORTS_PER_BANK = PE_L1_MRU_PORTS_PER_BANK;
    static_assert(L1_NUM_BANKS > 0, "PE_L1_MRU_BANKS must be > 0");

    static constexpr int MAX_FEATURE_DIM = 128;    // Realistic: matches JODIE datasets (Wikipedia/Reddit=128)
    static constexpr int PREFETCH_DEPTH = 4;       // Prefetch next N F rows
    
    // Memory latency model (for fair comparison with NeuraChip/DRAMsim3)
    // These values match typical HBM/DDR4 latencies in NeuraSim
    static constexpr int DRAM_LATENCY = 100;       // DRAM read latency (cycles)
    static constexpr int CACHE_HIT_LATENCY = 1;    // F cache hit latency
#ifndef PE_CLUSTER_A_LOAD_LATENCY
#define PE_CLUSTER_A_LOAD_LATENCY 50
#endif
    static constexpr int A_LOAD_LATENCY = PE_CLUSTER_A_LOAD_LATENCY;
    static constexpr uint64_t A_LOAD_ADDR_BASE = 0x30000000ULL;
#ifndef PE_CLUSTER_OUTPUT_WRITE_LATENCY
#define PE_CLUSTER_OUTPUT_WRITE_LATENCY 50
#endif
    static constexpr int OUTPUT_WRITE_LATENCY = PE_CLUSTER_OUTPUT_WRITE_LATENCY;
#ifndef PE_CLUSTER_OUTPUT_QUEUE_DEPTH
#define PE_CLUSTER_OUTPUT_QUEUE_DEPTH 8
#endif
    static constexpr int OUTPUT_QUEUE_DEPTH = PE_CLUSTER_OUTPUT_QUEUE_DEPTH;

    // Work item for SpMM computation
    struct WorkItem {
        int row_id;                    // Output row
        int csr_start_ptr;             // CSR row_ptr[row_id], for A-load address model
        std::vector<int> col_indices;  // A matrix column indices
        std::vector<float> values;     // A matrix values
        int nnz;                       // Number of non-zeros
        bool valid;
        
        WorkItem() : row_id(-1), csr_start_ptr(0), nnz(0), valid(false) {}
    };

    // Constructor
    PECluster();
    PECluster(GlobalCounters* gcp, int cluster_id);
    ~PECluster();
    
    void init(GlobalCounters* gcp, int cluster_id);
    
    // Main simulation tick - returns true if any PE did work
    bool tick();
    
    // Assign work to this cluster
    void assign_work(const WorkItem& work);
    
    // Check if cluster can accept more work
    bool can_accept_work() const;
    
    // Check if all PEs are done
    bool is_done() const;
    
    // Set F matrix pointer for row fetching
    void set_f_matrix(const float* f_matrix, int n, int d);
    
    // Get output partial sums for a completed row
    bool get_output(int* row_id, float* output, int max_dim);
    bool has_output() const { return !output_queue.empty(); }
    
    // Work stealing interface
    bool has_stealable_work() const;
    WorkItem steal_work();
    
    // Statistics
    long long get_total_mac_ops() const;
    long long get_f_cache_hits() const { return f_cache_hits; }
    long long get_f_cache_misses() const { return f_cache_misses; }
    long long get_memory_stall_cycles() const { return memory_stall_cycles; }

    // Breakdown: where F-row requests were satisfied.
    long long get_l1_f_cache_hits() const { return l1_f_cache_hits; }
    long long get_l2_shared_hits() const { return l2_shared_hits; }
    long long get_l2_shared_hits_pre_submit() const { return l2_shared_hits_pre_submit; }
    long long get_dram_fills() const { return dram_fills; }
    long long get_l1_bank_conflict_stalls() const { return l1_bank_conflict_stalls; }
    long long get_l2_bank_conflict_stalls() const { return l2_bank_conflict_stalls; }
    long long get_output_backpressure_stalls() const { return output_backpressure_stalls; }
    
    int get_cluster_id() const { return cluster_id; }

    // Optional shared (chip-level) F-cache shared by all clusters.
    // If set, cluster will consult it on local cache miss and will populate it on DRAM fills.
    void set_shared_f_cache(SharedFCache* cache) { shared_f_cache = cache; }

    // Reset per-SpMM-job state/counters so multi-timestep aggregation is correct.
    // If clear_local_cache=true, clears the per-cluster MRU (L1) contents.
    void reset_for_new_spmm(bool clear_local_cache = true);

    // Invalidate a single row in the local MRU cache (ΔF coherence).
    void invalidate_local_f_row(int row_id);

#ifdef ENABLE_DRAMSIM3_PE
    // DRAMsim3 integration: notify that F row fetch completed
    void notify_dram_complete(int col_idx, long long actual_latency);
    void notify_a_load_complete(int pe_idx, uint64_t addr);
    
    // Check if any PE needs its request submitted to DRAMsim3
    bool has_pending_dram_requests() const;
    bool has_pending_a_load_requests() const;
    
    // Get one pending DRAM request (pe_idx, col_idx). Returns -1 for pe_idx if none.
    void get_pending_dram_request(int& pe_idx, int& col_idx) const;
    void get_pending_a_load_request(int& pe_idx, uint64_t& addr, int& bytes, int& row_id) const;
    
    // Set waiting state after PE Array submits to DRAMsim3
    void set_waiting_for_dram(int pe_idx, int col_idx);
    void set_waiting_for_a_load_dram(int pe_idx, uint64_t addr);
#endif

private:
    GlobalCounters* gcp;
    int cluster_id;

    // Shared cache pointer (owned by PEArraySpMM). May be null.
    SharedFCache* shared_f_cache = nullptr;
    
    // PEs in this cluster
    PE pes[PES_PER_CLUSTER];
    
    // Shared F buffer with MRU cache (heap-allocated to avoid stack overflow)
    struct FCacheEntry {
        int row_id;
        float* data;  // Heap-allocated: MAX_FEATURE_DIM floats
        int dim;
        int age;  // For MRU eviction
        bool valid;
        
        FCacheEntry() : row_id(-1), data(nullptr), dim(0), age(0), valid(false) {}
    };
    FCacheEntry* f_cache;  // Heap-allocated: MRU_CACHE_ENTRIES entries
    int cache_age_counter;
    
    // Work queue for this cluster
    std::queue<WorkItem> work_queue;
    WorkItem current_work[PES_PER_CLUSTER];
    int work_ptr[PES_PER_CLUSTER];  // Progress within current work
    
    // F matrix reference
    const float* f_matrix;
    int f_n;  // Number of rows in F
    int f_d;  // Feature dimension
    
    // Output buffer
    struct OutputEntry {
        int row_id;
        float data[MAX_FEATURE_DIM];
    };
    std::queue<OutputEntry> output_queue;
    
    // Statistics
    long long f_cache_hits;
    long long f_cache_misses;
    long long memory_stall_cycles;  // Total cycles stalled on memory

    // Breakdown counters (for profiling)
    long long l1_f_cache_hits = 0;            // satisfied by local MRU
    long long l2_shared_hits = 0;             // satisfied by shared cache (total)
    long long l2_shared_hits_pre_submit = 0;  // satisfied from shared while in needs_submit
    long long dram_fills = 0;                 // DRAM fetch completions for this cluster
    long long l1_bank_conflict_stalls = 0;    // bank-port conflict stalls in local MRU
    long long l2_bank_conflict_stalls = 0;    // bank-port conflict stalls in shared cache

    long long current_cycle = 0;
    std::array<int, L1_NUM_BANKS> l1_bank_access_count{};
    
    // Memory request tracking (for latency simulation)
    struct PendingMemReq {
        int pe_idx;           // Which PE requested (-1 for prefetch)
        int col_idx;          // F row requested
        int remaining_cycles; // Cycles until completion (or -1 for callback received in DRAMsim3 mode)
        bool active;
#ifdef ENABLE_DRAMSIM3_PE
        // DRAMsim3 state machine:
        // - needs_submit = true: waiting for PE Array to submit to DRAMsim3
        // - needs_submit = false && waiting_for_dram = true: submitted, waiting for callback
        // - needs_submit = false && waiting_for_dram = false && remaining_cycles < 0: callback received
        bool needs_submit;     // True if PE Array hasn't submitted yet
        bool waiting_for_dram; // True if waiting for DRAMsim3 callback
#endif
        
        PendingMemReq() : pe_idx(-1), col_idx(-1), remaining_cycles(0), active(false)
#ifdef ENABLE_DRAMSIM3_PE
            , needs_submit(false), waiting_for_dram(false)
#endif
        {}
    };
    PendingMemReq pending_f_reqs[PES_PER_CLUSTER];
#ifdef ENABLE_DRAMSIM3_PE
    struct PendingALoadReq {
        int pe_idx = -1;
        int row_id = -1;
        uint64_t addr = 0;
        uint64_t end_addr = 0;
        int payload_bytes = 0;
        bool active = false;
        bool needs_submit = false;
        bool waiting_for_dram = false;
    };
    PendingALoadReq pending_a_load_reqs[PES_PER_CLUSTER];
#else
    int pending_a_load_cycles[PES_PER_CLUSTER];   // Cycles remaining for A load (per-PE)
#endif
    int pending_output_cycles;   // Cycles remaining for active output write
    bool output_write_active;
    int output_write_pe_idx;
    int output_rr_next_pe;
    long long output_backpressure_stalls;
    
#ifdef ENABLE_DRAMSIM3_PE
    long long dram_actual_latency;  // Actual latency from DRAMsim3 (for stats)
#endif
    
    // Internal methods
    void distribute_work_to_pes();
    bool fetch_f_row_for_pe(int pe_idx);
    void collect_pe_outputs();
    void commit_output_write();
    void issue_prefetch_requests();  // Prefetch next F rows
    void process_prefetch_completions();  // Handle completed prefetches
    void reset_l1_bank_access_tracker();
    int reserve_l1_bank_port(int row_id);
    bool is_a_load_pending(int pe_idx) const;
    
    // Cache management
    FCacheEntry* lookup_f_cache(int row_id);
    void insert_f_cache(int row_id, const float* data, int dim);
    void evict_lru_cache_entry();
};

#endif // PE_CLUSTER_H
