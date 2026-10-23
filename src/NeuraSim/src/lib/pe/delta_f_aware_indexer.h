// SPDX-License-Identifier: Apache-2.0
#ifndef DELTA_F_AWARE_INDEXER_H
#define DELTA_F_AWARE_INDEXER_H

#include <cstdint>
#include <list>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <set>

#include "work_stealing_scheduler.h"

// DRAMsim3 integration for indexer/feature store traffic (exact accounting)
#ifdef ENABLE_DRAMSIM3_PE
#include "dramsim3.h"
#endif

/**
 * DeltaFAwareIndexer
 *
 * 목적:
 * - ΔF(변경된 노드 집합) 입력을 받아, Phase2(A_new × ΔF)에서 필요한 edge만
 *   (≈ l * avg(deg)) 추출해서 CSR(row-centric) 형태로 PEArraySpMM에 공급.
 *
 * 공정성(MICRO):
 * - host가 공짜로 A_sub를 주는 대신, index store(개념적 CSC/adjacency list) 접근과
 *   ΔA에 의한 index update의 트래픽/카운터를 수집한다.
 *
 * 주의:
 * - DRAMsim3 트랜잭션 기반 정확 계상은 별도 구현 단계(no_approx_modeling)에서
 *   MemorySystem과 연결된다. 이 클래스는 그를 위한 주소/바이트 카운터를 제공한다.
 */
class DeltaFAwareIndexer {
public:
    using CSRMatrix = WorkStealingScheduler::CSRMatrix;

#ifndef INDEX_STORE_MAX_ENTRIES
#define INDEX_STORE_MAX_ENTRIES 16384
#endif
#ifndef INDEX_STORE_TAG_CHECK_CYCLES
#define INDEX_STORE_TAG_CHECK_CYCLES 1
#endif
#ifndef INDEX_STORE_SRAM_HIT_CYCLES
#define INDEX_STORE_SRAM_HIT_CYCLES 5
#endif
#ifndef INDEX_STORE_DRAM_MISS_CYCLES
#define INDEX_STORE_DRAM_MISS_CYCLES 32
#endif

    struct EdgeRec {
        int src;        // row i
        float w;        // A[i,j]
        bool valid;     // deletion tombstone

        EdgeRec() : src(-1), w(0.0f), valid(true) {}
        EdgeRec(int src_, float w_, bool valid_) : src(src_), w(w_), valid(valid_) {}
    };

    struct DeltaAUpdate {
        int src = -1;
        int dst = -1;
        float w = 0.0f;
        bool is_add = true;  // false => delete (or negative weight)
    };

    struct DeltaFUpdate {
        int node = -1;                 // changed node id (j)
        const float* delta_vec = nullptr; // optional pointer (owned elsewhere)
    };

    struct Counters {
        // Metadata traffic intent (for exact DRAMsim3 accounting later)
        uint64_t index_bytes_read = 0;
        uint64_t index_bytes_write = 0;

        // FeatureStore update intent
        uint64_t feature_bytes_write = 0;

        // Logical operations (for Accelergy action counts later)
        uint64_t index_lookup_ops = 0;
        uint64_t index_update_ops = 0;
        // Exact logical payload owned by sparse operator maintenance.  This is
        // distinct from cacheline-rounded index_bytes_write.
        uint64_t index_update_payload_bytes = 0;

        // Cycle-accurate DRAM stalls for index/feature traffic (when enabled)
        long long index_dram_cycles = 0;
        // Owner-disjoint subset of index_dram_cycles caused only by applying
        // delta-A records to the maintained incoming-edge index.
        long long index_update_dram_cycles = 0;
        long long feature_dram_cycles = 0;

        // Batching/coalescing statistics (shows optimization effectiveness)
        long long coalesced_requests = 0;   // Requests saved by cacheline coalescing
        long long piggyback_hits = 0;       // Requests saved by piggybacking
        long long mshr_full_stalls = 0;     // Cycles stalled due to MSHR full

        // Indexer controller micro-ops (for CACTI/Accelergy control energy)
        // - cam_searches: number of tag searches performed
        // - mshr_writes: number of MSHR allocations (request buffer writes)
        // - mshr_reads: number of MSHR reads on completion (response)
        uint64_t cam_searches = 0;
        uint64_t mshr_writes = 0;
        uint64_t mshr_reads = 0;

        // Partial-SRAM IndexStore behavior stats.
        uint64_t index_sram_hits = 0;
        uint64_t index_sram_misses = 0;
        uint64_t index_sram_fills = 0;
        uint64_t index_sram_evictions = 0;
        long long index_lookup_cycles = 0;

        void reset() {
            index_bytes_read = 0;
            index_bytes_write = 0;
            feature_bytes_write = 0;
            index_lookup_ops = 0;
            index_update_ops = 0;
            index_update_payload_bytes = 0;
            index_dram_cycles = 0;
            index_update_dram_cycles = 0;
            feature_dram_cycles = 0;
            coalesced_requests = 0;
            piggyback_hits = 0;
            mshr_full_stalls = 0;
            cam_searches = 0;
            mshr_writes = 0;
            mshr_reads = 0;
            index_sram_hits = 0;
            index_sram_misses = 0;
            index_sram_fills = 0;
            index_sram_evictions = 0;
            index_lookup_cycles = 0;
        }
    };

    DeltaFAwareIndexer();

    // Optional: enable cycle-accurate DRAMsim3 accounting for index + feature updates.
    // Provide output dirs so energy can be read from dramsim3.json later.
    // tck_ns: DRAM clock period in nanoseconds (e.g., 1.0 for HBM2, 0.5 for HBM4)
    void enable_dramsim3(const std::string& config_file,
                         const std::string& index_output_dir,
                         const std::string& feature_output_dir,
                         double tck_ns = 1.0);

    bool dramsim3_enabled() const;

    // Flush DRAMsim3 stats (writes JSON/TXT into output dirs).
    void flush_dramsim3_stats();
#ifdef ENABLE_DRAMSIM3_PE
    dramsim3::CommandEnergySnapshot index_hbm_command_energy() const {
        return index_dram_ != nullptr
            ? index_dram_->GetCommandEnergySnapshot()
            : dramsim3::CommandEnergySnapshot{};
    }
    dramsim3::CommandEnergySnapshot feature_hbm_command_energy() const {
        return feature_dram_ != nullptr
            ? feature_dram_->GetCommandEnergySnapshot()
            : dramsim3::CommandEnergySnapshot{};
    }
#endif

    // Build index store from A_new CSR (one-time init or full rebuild).
    void build_from_csr(const CSRMatrix& a_new);

    // Apply ΔA updates to the index store (add/del). Updates counters.
    void apply_delta_a(const std::vector<DeltaAUpdate>& delta_a);

    // Record ΔF updates for FeatureStore update traffic (l×d×4B). d is feature dim.
    void account_feature_updates(int d, int num_changed_nodes);

    // Cycle-accurate FeatureStore write transactions (when DRAMsim3 enabled).
    void apply_feature_store_updates(const std::vector<int>& changed_nodes, int d);

    // Extract Phase2 matrix A_sub in CSR(row-centric): edges (i,j) where j ∈ changed_nodes.
    // Returns CSRMatrix with same n rows as original (rows without edges allowed).
    CSRMatrix extract_a_sub_for_delta_f(const std::vector<int>& changed_nodes);

    // Account for fullscan DRAM cost: reading entire A matrix CSR from DRAM.
    // Models the cost of scanning row_ptr(n+1) + col_idx(nnz) + values(nnz) via DRAMsim3.
    // Returns cycles spent. Updates index_dram_cycles in counters.
    long long account_fullscan_dram_cost(int n, int nnz);

    // For analysis/debug: affected rows/cols sizes
    std::set<int> get_affected_rows_last() const { return affected_rows_last_; }
    std::set<int> get_affected_cols_last() const { return affected_cols_last_; }

    const Counters& get_counters() const { return ctrs_; }
    void reset_counters() { ctrs_.reset(); }

    int n() const { return n_; }

private:
    static constexpr int INDEX_STORE_CAPACITY_ENTRIES = INDEX_STORE_MAX_ENTRIES;
    static constexpr int INDEX_STORE_TAG_CHECK_LATENCY = INDEX_STORE_TAG_CHECK_CYCLES;
    static constexpr int INDEX_STORE_SRAM_HIT_LATENCY = INDEX_STORE_SRAM_HIT_CYCLES;
    static constexpr int INDEX_STORE_DRAM_MISS_LATENCY = INDEX_STORE_DRAM_MISS_CYCLES;

    // index: dst(j) -> list of incoming edges (i, w)
    std::vector<std::vector<EdgeRec>> incoming_;
    int n_ = 0;

    // =========================================================================
    // Index Store Memory Layout (Hardware-Realistic)
    // =========================================================================
    // Real accelerators use compact index storage:
    // - Each node's incoming edge list is allocated based on max expected degree
    // - Adjacency data is stored contiguously to maximize cacheline utilization
    // - Header table (node → offset) enables direct access
    //
    // Layout: base_addr + node_id * stride_bytes
    // Stride = max_edges_per_node * bytes_per_edge = 32 * 8 = 256B = 4 cachelines
    //
    // This enables:
    // - Spatial locality when querying nearby nodes
    // - Cacheline coalescing for nodes mapped to same/adjacent lines
    // =========================================================================
    uint64_t index_base_addr_ = 0x10000000ULL;
    uint64_t bytes_per_edge_rec_ = 8;   // src(int32) + w(float32)
    uint64_t cacheline_bytes_ = 64;
    uint64_t index_region_stride_bytes_ = 256;   // 4 cachelines per node (32 edges max)
    uint64_t feature_base_addr_ = 0x20000000ULL;

    // Partial-SRAM index placement:
    // - onchip_entries_: node ids currently resident in on-chip IndexStore
    // - onchip_lru_list_: front=MRU, back=LRU
    // - onchip_lru_pos_: node id -> iterator into onchip_lru_list_
    size_t index_store_capacity_entries_ = 0;
    std::unordered_set<int> onchip_entries_;
    std::list<int> onchip_lru_list_;
    std::unordered_map<int, std::list<int>::iterator> onchip_lru_pos_;

    // Snapshot of last extract for reporting
    std::set<int> affected_rows_last_;
    std::set<int> affected_cols_last_;

    Counters ctrs_;

    void add_index_read_bytes_for_dst(int dst, uint64_t bytes);
    void add_index_write_bytes_for_dst(int dst, uint64_t bytes);
    bool is_index_onchip(int node) const;
    void touch_index_entry(int node);
    bool insert_index_entry_lru(int node);  // returns true if eviction happened
    void reset_index_residency();

#ifdef ENABLE_DRAMSIM3_PE
    dramsim3::MemorySystem* index_dram_ = nullptr;
    dramsim3::MemorySystem* feature_dram_ = nullptr;
    std::string index_dram_output_dir_;
    std::string feature_dram_output_dir_;

    // Clock domain ratio: NC runs at 1 GHz (1ns), but DRAMsim3 may run faster (HBM3/4).
    // clock_ratio = NC_CLOCK_PERIOD_NS / DRAM_tCK_NS
    double dram_clock_ratio_ = 1.0;
    double dram_tick_accumulator_index_ = 0.0;
    double dram_tick_accumulator_feature_ = 0.0;
    
    // Helper to tick DRAMsim3 with proper clock ratio
    void tick_dramsim3_index();
    void tick_dramsim3_feature();

    // Outstanding transaction tracking
    std::unordered_map<uint64_t, int> inflight_index_;
    std::unordered_map<uint64_t, int> inflight_feature_;

    static void noop_read_cb(uint64_t) {}

    void index_read_cb(uint64_t addr);
    void index_write_cb(uint64_t addr);
    void feature_read_cb(uint64_t addr);
    void feature_write_cb(uint64_t addr);

    // Issue cacheline-granular transactions and tick until completion (legacy, blocking).
    long long issue_transactions(dramsim3::MemorySystem* sys,
                                std::unordered_map<uint64_t, int>& inflight,
                                uint64_t start_addr,
                                uint64_t bytes,
                                bool is_write);

    // Batched, bandwidth-limited transaction issuing (hardware-realistic).
    // - Coalesces to cachelines
    // - Issues up to INDEX_MAX_REQUESTS_PER_CYCLE per simulated cycle
    // - Uses MSHR-size limit (INDEX_MSHR_SIZE) as max outstanding unique lines
    long long issue_batched_transactions(dramsim3::MemorySystem* sys,
                                         std::unordered_map<uint64_t, int>& inflight,
                                         const std::vector<std::pair<uint64_t, uint64_t>>& addr_size_pairs,
                                         bool is_write);

    // =========================================================================
    // Batched Index Lookup Unit (Hardware-realistic)
    // =========================================================================
    // Models a dedicated hardware unit for index lookups with:
    // - MSHR-based coalescing (same cacheline → single DRAM request)
    // - Piggybacking (in-flight request → add to waiters, no new request)
    // - Bandwidth-limited request issuing (max N requests/cycle)
    // - Bank-level parallelism via DRAMsim3
    //
    // This is the key optimization for incremental SpMM Phase 2.
    // =========================================================================

    static constexpr int INDEX_MSHR_SIZE = 32;           // Max outstanding unique addresses
    static constexpr int INDEX_MAX_REQUESTS_PER_CYCLE = 4;  // Memory controller BW limit
    static constexpr int INDEX_MAX_PIGGYBACK = 16;       // Max waiters per MSHR entry

    struct IndexMSHREntry {
        uint64_t addr;
        int waiters;          // Number of requests coalesced to this address
        long long start_cycle;
        bool completed;
    };

    std::unordered_map<uint64_t, IndexMSHREntry> index_mshr_;
    long long index_mshr_full_stalls_ = 0;
    long long index_piggyback_count_ = 0;
    long long index_coalesced_requests_ = 0;

    // Batched + coalesced transaction issuing
    // Returns total cycles including DRAM latency with proper overlap
    long long issue_batched_index_lookups(
        const std::vector<std::pair<uint64_t, uint64_t>>& addr_size_pairs);

    // Helper: expand address range to cacheline-aligned set
    std::set<uint64_t> expand_to_cachelines(uint64_t start, uint64_t bytes) const;
#endif
};

#endif  // DELTA_F_AWARE_INDEXER_H
