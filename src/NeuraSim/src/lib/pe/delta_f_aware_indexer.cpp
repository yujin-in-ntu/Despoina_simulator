// SPDX-License-Identifier: Apache-2.0
#include "delta_f_aware_indexer.h"

#include <algorithm>
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

DeltaFAwareIndexer::DeltaFAwareIndexer() {
    ctrs_.reset();
    index_store_capacity_entries_ = (INDEX_STORE_CAPACITY_ENTRIES > 0)
        ? static_cast<size_t>(INDEX_STORE_CAPACITY_ENTRIES)
        : 0;
}

#ifdef ENABLE_DRAMSIM3_PE
void DeltaFAwareIndexer::enable_dramsim3(const std::string& config_file,
                                         const std::string& index_output_dir,
                                         const std::string& feature_output_dir,
                                         double tck_ns) {
    // Idempotent: if already enabled, keep accumulating into the existing systems.
    if (index_dram_ != nullptr && feature_dram_ != nullptr) {
        return;
    }
    index_dram_output_dir_ = index_output_dir;
    feature_dram_output_dir_ = feature_output_dir;
    CREATE_DIR(index_dram_output_dir_.c_str());
    CREATE_DIR(feature_dram_output_dir_.c_str());

    // Calculate clock ratio: NC clock (1.0 ns) / DRAM tCK
    // HBM2: tCK=1.0ns -> ratio=1.0, HBM3: tCK=0.625ns -> ratio=1.6, HBM4: tCK=0.5ns -> ratio=2.0
    constexpr double NC_CLOCK_PERIOD_NS = 1.0;
    dram_clock_ratio_ = NC_CLOCK_PERIOD_NS / tck_ns;
    dram_tick_accumulator_index_ = 0.0;
    dram_tick_accumulator_feature_ = 0.0;

    index_dram_ = dramsim3::GetMemorySystem(
        config_file,
        index_dram_output_dir_,
        std::bind(&DeltaFAwareIndexer::index_read_cb, this, std::placeholders::_1),
        std::bind(&DeltaFAwareIndexer::index_write_cb, this, std::placeholders::_1)
    );
    feature_dram_ = dramsim3::GetMemorySystem(
        config_file,
        feature_dram_output_dir_,
        std::bind(&DeltaFAwareIndexer::feature_read_cb, this, std::placeholders::_1),
        std::bind(&DeltaFAwareIndexer::feature_write_cb, this, std::placeholders::_1)
    );
}

void DeltaFAwareIndexer::tick_dramsim3_index() {
    dram_tick_accumulator_index_ += dram_clock_ratio_;
    while (dram_tick_accumulator_index_ >= 1.0) {
        index_dram_->ClockTick();
        dram_tick_accumulator_index_ -= 1.0;
    }
}

void DeltaFAwareIndexer::tick_dramsim3_feature() {
    dram_tick_accumulator_feature_ += dram_clock_ratio_;
    while (dram_tick_accumulator_feature_ >= 1.0) {
        feature_dram_->ClockTick();
        dram_tick_accumulator_feature_ -= 1.0;
    }
}

bool DeltaFAwareIndexer::dramsim3_enabled() const {
    return (index_dram_ != nullptr) && (feature_dram_ != nullptr);
}

void DeltaFAwareIndexer::flush_dramsim3_stats() {
    if (index_dram_) index_dram_->PrintStats();
    if (feature_dram_) feature_dram_->PrintStats();
}

void DeltaFAwareIndexer::index_read_cb(uint64_t addr) {
    std::unordered_map<uint64_t, int>::iterator it = inflight_index_.find(addr);
    if (it == inflight_index_.end()) return;
    it->second -= 1;
    if (it->second <= 0) inflight_index_.erase(it);
}
void DeltaFAwareIndexer::index_write_cb(uint64_t addr) { index_read_cb(addr); }

void DeltaFAwareIndexer::feature_read_cb(uint64_t addr) {
    std::unordered_map<uint64_t, int>::iterator it = inflight_feature_.find(addr);
    if (it == inflight_feature_.end()) return;
    it->second -= 1;
    if (it->second <= 0) inflight_feature_.erase(it);
}
void DeltaFAwareIndexer::feature_write_cb(uint64_t addr) { feature_read_cb(addr); }

long long DeltaFAwareIndexer::issue_transactions(dramsim3::MemorySystem* sys,
                                                 std::unordered_map<uint64_t, int>& inflight,
                                                 uint64_t start_addr,
                                                 uint64_t bytes,
                                                 bool is_write) {
    if (!sys) return 0;
    long long nc_cycles = 0;  // NC cycles (not DRAMsim3 cycles)
    const uint64_t line = cacheline_bytes_;
    uint64_t first = (start_addr / line) * line;
    uint64_t last = ((start_addr + bytes - 1) / line) * line;
    
    // Determine which accumulator to use based on which DRAM system this is
    double* accum = (sys == index_dram_) ? &dram_tick_accumulator_index_ 
                                         : &dram_tick_accumulator_feature_;
    
    auto tick_dram = [&]() {
        *accum += dram_clock_ratio_;
        while (*accum >= 1.0) {
            sys->ClockTick();
            *accum -= 1.0;
        }
        nc_cycles++;
    };
    
    for (uint64_t addr = first; addr <= last; addr += line) {
        while (!sys->WillAcceptTransaction(addr, is_write)) {
            tick_dram();
        }
        sys->AddTransaction(addr, is_write);
        inflight[addr] += 1;
    }
    while (!inflight.empty()) {
        tick_dram();
    }
    return nc_cycles;
}

long long DeltaFAwareIndexer::issue_batched_transactions(
    dramsim3::MemorySystem* sys,
    std::unordered_map<uint64_t, int>& inflight,
    const std::vector<std::pair<uint64_t, uint64_t>>& addr_size_pairs,
    bool is_write) {

    if (!sys || addr_size_pairs.empty()) return 0;

    // Count total cachelines before coalescing.
    size_t total_cachelines_before = 0;
    for (const auto& p : addr_size_pairs) {
        const uint64_t start_addr = p.first;
        const uint64_t bytes = p.second;
        auto lines = expand_to_cachelines(start_addr, bytes);
        total_cachelines_before += lines.size();
    }

    // Unique cachelines (after coalescing).
    std::set<uint64_t> unique_cachelines;
    for (const auto& p : addr_size_pairs) {
        const uint64_t start_addr = p.first;
        const uint64_t bytes = p.second;
        auto lines = expand_to_cachelines(start_addr, bytes);
        unique_cachelines.insert(lines.begin(), lines.end());
    }
    const size_t unique_lines = unique_cachelines.size();

    // Update coalescing stats.
    if (total_cachelines_before >= unique_lines) {
        index_coalesced_requests_ += static_cast<long long>(total_cachelines_before - unique_lines);
    }

    // -------------------------------------------------------------------------
    // Indexer controller micro-ops:
    // - Each original (pre-coalescing) cacheline request corresponds to a tag/CAM search.
    // - Each unique cacheline issued allocates one MSHR entry (request buffer write).
    // - Each unique cacheline completion consumes one MSHR entry (request buffer read).
    //
    // Energy/area for CAM + MSHR SRAM are sourced from CACTI and added in the
    // Python energy pipeline (no hardcoded constants).
    // -------------------------------------------------------------------------
    ctrs_.cam_searches += static_cast<uint64_t>(total_cachelines_before);
    ctrs_.mshr_writes += static_cast<uint64_t>(unique_lines);

    std::vector<uint64_t> pending_addrs(unique_cachelines.begin(), unique_cachelines.end());
    size_t next_to_issue = 0;
    long long nc_cycles = 0;  // NC cycles (not DRAMsim3 cycles)

    // Clear any stale outstanding tracking for this batch.
    inflight.clear();

    // Determine which accumulator to use based on which DRAM system this is
    double* accum = (sys == index_dram_) ? &dram_tick_accumulator_index_ 
                                         : &dram_tick_accumulator_feature_;
    
    auto tick_dram = [&]() {
        *accum += dram_clock_ratio_;
        while (*accum >= 1.0) {
            sys->ClockTick();
            *accum -= 1.0;
        }
        nc_cycles++;
    };

    while (next_to_issue < pending_addrs.size() || !inflight.empty()) {
        // One simulated NC cycle (with proper DRAM clock ratio)
        tick_dram();

        // inflight is decremented/erased by callbacks; nothing to do here.

        int issued_this_cycle = 0;
        while (next_to_issue < pending_addrs.size() &&
               issued_this_cycle < INDEX_MAX_REQUESTS_PER_CYCLE) {
            // Enforce MSHR capacity (unique outstanding cachelines)
            if (inflight.size() >= static_cast<size_t>(INDEX_MSHR_SIZE)) {
                index_mshr_full_stalls_++;
                break;
            }

            uint64_t addr = pending_addrs[next_to_issue];
            if (!sys->WillAcceptTransaction(addr, is_write)) {
                break;
            }
            sys->AddTransaction(addr, is_write);
            inflight[addr] = 1;
            issued_this_cycle++;
            next_to_issue++;
        }
    }

    // Propagate internal stats to ctrs_ for reporting
    ctrs_.coalesced_requests = index_coalesced_requests_;
    ctrs_.piggyback_hits = index_piggyback_count_;
    ctrs_.mshr_full_stalls = index_mshr_full_stalls_;
    ctrs_.mshr_reads += static_cast<uint64_t>(unique_lines);

    return nc_cycles;
}

std::set<uint64_t> DeltaFAwareIndexer::expand_to_cachelines(uint64_t start, uint64_t bytes) const {
    std::set<uint64_t> lines;
    if (bytes == 0) return lines;
    const uint64_t line = cacheline_bytes_;
    uint64_t first = (start / line) * line;
    uint64_t last = ((start + bytes - 1) / line) * line;
    for (uint64_t addr = first; addr <= last; addr += line) {
        lines.insert(addr);
    }
    return lines;
}

// =============================================================================
// Batched Index Lookup Unit - Hardware-Realistic Implementation
// =============================================================================
// This models a dedicated hardware unit for index lookups that:
// 1. Collects all lookup addresses from changed_nodes
// 2. Coalesces requests to the same cacheline (deduplication)
// 3. Uses MSHR to track outstanding requests and enable piggybacking
// 4. Issues requests respecting bandwidth limits (N requests/cycle)
// 5. Exploits bank-level parallelism via DRAMsim3's internal modeling
//
// Key insight: With 82 changed nodes mapping to potentially many fewer unique
// cachelines, we can dramatically reduce DRAM requests through coalescing.
// =============================================================================

long long DeltaFAwareIndexer::issue_batched_index_lookups(
    const std::vector<std::pair<uint64_t, uint64_t>>& addr_size_pairs) {

    if (!index_dram_ || addr_size_pairs.empty()) return 0;

    // =========================================================================
    // Phase 1: Collect and coalesce all required cachelines
    // =========================================================================
    // Count total cachelines that would be accessed without coalescing
    size_t total_cachelines_before = 0;
    for (const auto& p : addr_size_pairs) {
        const uint64_t start_addr = p.first;
        const uint64_t bytes = p.second;
        auto lines = expand_to_cachelines(start_addr, bytes);
        total_cachelines_before += lines.size();
    }

    // Now collect unique cachelines (after coalescing)
    std::set<uint64_t> unique_cachelines;
    for (const auto& p : addr_size_pairs) {
        const uint64_t start_addr = p.first;
        const uint64_t bytes = p.second;
        auto lines = expand_to_cachelines(start_addr, bytes);
        unique_cachelines.insert(lines.begin(), lines.end());
    }

    const size_t unique_lines = unique_cachelines.size();
    // Coalesced = cachelines saved by deduplication
    index_coalesced_requests_ += (total_cachelines_before - unique_lines);

    // Controller micro-ops (tag searches happen before coalescing in the front-end).
    ctrs_.cam_searches += static_cast<uint64_t>(total_cachelines_before);

    // =========================================================================
    // Phase 2: Issue requests with MSHR tracking and bandwidth limiting
    // =========================================================================
    // Convert set to vector for indexed access
    std::vector<uint64_t> pending_addrs(unique_cachelines.begin(), unique_cachelines.end());
    size_t next_to_issue = 0;
    long long nc_cycles = 0;  // NC cycles (not DRAMsim3 cycles)

    // Clear MSHR for this batch
    index_mshr_.clear();

    while (next_to_issue < pending_addrs.size() || !index_mshr_.empty()) {
        // Tick DRAMsim3 with proper clock ratio (index DRAM)
        tick_dramsim3_index();
        nc_cycles++;

        // Check for completed requests via callback (inflight_index_ is decremented by callback)
        for (auto it = index_mshr_.begin(); it != index_mshr_.end(); ) {
            if (inflight_index_.find(it->first) == inflight_index_.end() ||
                inflight_index_[it->first] <= 0) {
                // Request completed
                it->second.completed = true;
                inflight_index_.erase(it->first);
                it = index_mshr_.erase(it);
                ctrs_.mshr_reads += 1;
            } else {
                ++it;
            }
        }

        // Issue new requests up to bandwidth limit
        int issued_this_cycle = 0;
        while (next_to_issue < pending_addrs.size() &&
               issued_this_cycle < INDEX_MAX_REQUESTS_PER_CYCLE) {

            uint64_t addr = pending_addrs[next_to_issue];

            // Check if already in MSHR (piggybacking opportunity)
            auto mshr_it = index_mshr_.find(addr);
            if (mshr_it != index_mshr_.end()) {
                // Piggyback: request already in flight, just add waiter
                if (mshr_it->second.waiters < INDEX_MAX_PIGGYBACK) {
                    mshr_it->second.waiters++;
                    index_piggyback_count_++;
                }
                next_to_issue++;
                continue;  // No new DRAM request needed
            }

            // Check MSHR capacity
            if (index_mshr_.size() >= INDEX_MSHR_SIZE) {
                // MSHR full, must wait for a slot to free up
                index_mshr_full_stalls_++;
                break;  // Can't issue more this cycle
            }

            // Check if DRAM controller will accept
            if (!index_dram_->WillAcceptTransaction(addr, false)) {
                break;  // DRAM busy, try next cycle
            }

            // Issue the request
            index_dram_->AddTransaction(addr, false);
            inflight_index_[addr] = 1;

            // Add to MSHR
            IndexMSHREntry entry;
            entry.addr = addr;
            entry.waiters = 1;
            entry.start_cycle = nc_cycles;
            entry.completed = false;
            index_mshr_[addr] = entry;
            ctrs_.mshr_writes += 1;

            issued_this_cycle++;
            next_to_issue++;
        }
    }

    // Propagate internal stats to ctrs_ for reporting
    ctrs_.coalesced_requests = index_coalesced_requests_;
    ctrs_.piggyback_hits = index_piggyback_count_;
    ctrs_.mshr_full_stalls = index_mshr_full_stalls_;

    return nc_cycles;
}
#endif

bool DeltaFAwareIndexer::is_index_onchip(int node) const {
    return onchip_entries_.find(node) != onchip_entries_.end();
}

void DeltaFAwareIndexer::touch_index_entry(int node) {
    auto it = onchip_lru_pos_.find(node);
    if (it == onchip_lru_pos_.end()) return;
    onchip_lru_list_.erase(it->second);
    onchip_lru_list_.push_front(node);
    onchip_lru_pos_[node] = onchip_lru_list_.begin();
}

bool DeltaFAwareIndexer::insert_index_entry_lru(int node) {
    if (index_store_capacity_entries_ == 0) return false;

    if (is_index_onchip(node)) {
        touch_index_entry(node);
        return false;
    }

    bool evicted = false;
    if (onchip_entries_.size() >= index_store_capacity_entries_) {
        if (!onchip_lru_list_.empty()) {
            int victim = onchip_lru_list_.back();
            onchip_lru_list_.pop_back();
            onchip_lru_pos_.erase(victim);
            onchip_entries_.erase(victim);
            evicted = true;
        }
    }

    onchip_lru_list_.push_front(node);
    onchip_lru_pos_[node] = onchip_lru_list_.begin();
    onchip_entries_.insert(node);
    return evicted;
}

void DeltaFAwareIndexer::reset_index_residency() {
    onchip_entries_.clear();
    onchip_lru_list_.clear();
    onchip_lru_pos_.clear();
}

void DeltaFAwareIndexer::build_from_csr(const CSRMatrix& a_new) {
    n_ = a_new.n;
    incoming_.clear();
    incoming_.resize(n_);

    // Build incoming adjacency lists: for each edge (row=i, col=j) push into incoming_[j]
    for (int i = 0; i < n_; i++) {
        int start = a_new.row_ptr[i];
        int end = a_new.row_ptr[i + 1];
        for (int p = start; p < end; p++) {
            int j = a_new.col_idx[p];
            float w = a_new.values[p];
            if (j < 0 || j >= n_) continue;
            incoming_[j].push_back(EdgeRec(i, w, true));
        }
    }

    // Canonicalize: sort each incoming list by src for determinism (helps stable CSR build)
    for (int j = 0; j < n_; j++) {
        auto& vec = incoming_[j];
        std::sort(vec.begin(), vec.end(), [](const EdgeRec& a, const EdgeRec& b) {
            return a.src < b.src;
        });
    }

    // Partial-SRAM model:
    // preload only up to INDEX_STORE_MAX_ENTRIES entries on-chip.
    reset_index_residency();
    const size_t preload = std::min(static_cast<size_t>(n_), index_store_capacity_entries_);
    for (size_t node = 0; node < preload; ++node) {
        insert_index_entry_lru(static_cast<int>(node));
    }
}

void DeltaFAwareIndexer::apply_delta_a(const std::vector<DeltaAUpdate>& delta_a) {
    ctrs_.index_update_ops += delta_a.size();
    ctrs_.index_update_payload_bytes +=
        static_cast<uint64_t>(delta_a.size()) * bytes_per_edge_rec_;
    // For simplicity: update incoming_[dst] by insertion or tombstone.
    // This is “functional correctness” for A_sub extraction; exact DRAMsim3 modeling will
    // be implemented later by mapping these updates to transactions + latency.

#ifdef ENABLE_DRAMSIM3_PE
    std::vector<std::pair<uint64_t, uint64_t>> write_reqs;
    write_reqs.reserve(delta_a.size());
#endif

    for (const auto& up : delta_a) {
        if (up.dst < 0 || up.dst >= n_) continue;
        auto& vec = incoming_[up.dst];

        // Model metadata write: one edge record updated.
        add_index_write_bytes_for_dst(up.dst, bytes_per_edge_rec_);
#ifdef ENABLE_DRAMSIM3_PE
        if (index_dram_) {
            uint64_t base = index_base_addr_ + static_cast<uint64_t>(up.dst) * index_region_stride_bytes_;
            write_reqs.emplace_back(base, bytes_per_edge_rec_);
        }
#endif

        if (up.is_add) {
            vec.push_back(EdgeRec(up.src, up.w, true));
        } else {
            // delete: find first matching src and mark invalid
            for (auto& e : vec) {
                if (e.valid && e.src == up.src) {
                    e.valid = false;
                    e.w = 0.0f;
                    break;
                }
            }
        }

        // Keep LRU state coherent: updates to resident entries refresh recency.
        if (is_index_onchip(up.dst)) {
            touch_index_entry(up.dst);
        }
    }

#ifdef ENABLE_DRAMSIM3_PE
    if (index_dram_ && !write_reqs.empty()) {
        // Batched index metadata writes (bandwidth-limited, coalesced).
        long long c = issue_batched_transactions(index_dram_, inflight_index_, write_reqs, /*is_write=*/true);
        ctrs_.index_dram_cycles += c;
        ctrs_.index_update_dram_cycles += c;
    } else if (!delta_a.empty()) {
        const long long c = std::max<long long>(
            1, static_cast<long long>((ctrs_.index_update_payload_bytes + 31ULL) / 32ULL));
        ctrs_.index_dram_cycles += c;
        ctrs_.index_update_dram_cycles += c;
    }
#else
    if (!delta_a.empty()) {
        const long long c = std::max<long long>(
            1, static_cast<long long>((ctrs_.index_update_payload_bytes + 31ULL) / 32ULL));
        ctrs_.index_dram_cycles += c;
        ctrs_.index_update_dram_cycles += c;
    }
#endif
}

void DeltaFAwareIndexer::account_feature_updates(int d, int num_changed_nodes) {
    if (d <= 0 || num_changed_nodes <= 0) return;
    // Each updated row writes d floats.
    ctrs_.feature_bytes_write += static_cast<uint64_t>(num_changed_nodes) * static_cast<uint64_t>(d) * 4ULL;
}

void DeltaFAwareIndexer::apply_feature_store_updates(const std::vector<int>& changed_nodes, int d) {
    if (d <= 0) return;
    // Count traffic intent.
    account_feature_updates(d, static_cast<int>(changed_nodes.size()));

#ifdef ENABLE_DRAMSIM3_PE
    if (!feature_dram_) return;
    const uint64_t row_bytes = static_cast<uint64_t>(d) * 4ULL;
    std::vector<std::pair<uint64_t, uint64_t>> write_reqs;
    write_reqs.reserve(changed_nodes.size());
    for (size_t idx = 0; idx < changed_nodes.size(); idx++) {
        int node = changed_nodes[idx];
        if (node < 0 || node >= n_) continue;
        uint64_t addr = feature_base_addr_ + static_cast<uint64_t>(node) * row_bytes;
        write_reqs.emplace_back(addr, row_bytes);
    }
    if (!write_reqs.empty()) {
        long long c = issue_batched_transactions(feature_dram_, inflight_feature_, write_reqs, /*is_write=*/true);
        ctrs_.feature_dram_cycles += c;
    }
#endif
}

DeltaFAwareIndexer::CSRMatrix
DeltaFAwareIndexer::extract_a_sub_for_delta_f(const std::vector<int>& changed_nodes) {
    CSRMatrix out;
    out.n = n_;
    out.row_ptr.assign(n_ + 1, 0);

    affected_rows_last_.clear();
    affected_cols_last_.clear();

    // Phase2 wants edges where col=j is changed. We'll build per-row buffers then CSR.
    std::vector<std::vector<std::pair<int, float>>> per_row;
    per_row.resize(n_);

    // =========================================================================
    // Step 1: Partial-SRAM index lookup model
    // =========================================================================
    // node j lookup path:
    //   tag check (1 cyc)
    //   hit  -> on-chip SRAM read (5 cyc)
    //   miss -> DRAM fetch + on-chip fill (LRU eviction if needed)
    //
    // Functional data still lives in incoming_[] for correctness; this step models
    // latency/traffic and on-chip residency transitions.
    // =========================================================================

#ifdef ENABLE_DRAMSIM3_PE
    std::vector<std::pair<uint64_t, uint64_t>> miss_requests;  // (addr, size) pairs
#endif
    std::vector<int> lookup_nodes;
    lookup_nodes.reserve(changed_nodes.size());

    long long lookup_cycles = 0;
    size_t miss_count = 0;

    for (int j : changed_nodes) {
        if (j < 0 || j >= n_) continue;
        lookup_nodes.push_back(j);
        affected_cols_last_.insert(j);
        ctrs_.index_lookup_ops += 1;
        ctrs_.cam_searches += 1;
        lookup_cycles += INDEX_STORE_TAG_CHECK_LATENCY;

        const auto& inc = incoming_[j];
        const uint64_t bytes_needed = static_cast<uint64_t>(inc.size()) * bytes_per_edge_rec_;
        const bool onchip_hit = is_index_onchip(j);

        if (onchip_hit) {
            ctrs_.index_sram_hits += 1;
            lookup_cycles += INDEX_STORE_SRAM_HIT_LATENCY;
            add_index_read_bytes_for_dst(j, bytes_needed);
            touch_index_entry(j);
            continue;
        }

        ctrs_.index_sram_misses += 1;
        miss_count += 1;

        // One node entry occupies fixed stride in IndexStore (e.g., 256B).
        const uint64_t fill_bytes = std::max<uint64_t>(index_region_stride_bytes_, bytes_needed);
        add_index_read_bytes_for_dst(j, fill_bytes);

        if (index_store_capacity_entries_ > 0) {
            if (insert_index_entry_lru(j)) {
                ctrs_.index_sram_evictions += 1;
            }
            ctrs_.index_sram_fills += 1;
            add_index_write_bytes_for_dst(j, fill_bytes);
        }

#ifdef ENABLE_DRAMSIM3_PE
        if (index_dram_ && fill_bytes > 0) {
            uint64_t base = index_base_addr_ + static_cast<uint64_t>(j) * index_region_stride_bytes_;
            miss_requests.emplace_back(base, fill_bytes);
        }
#endif
    }

    ctrs_.index_lookup_cycles += lookup_cycles;
    ctrs_.index_dram_cycles += lookup_cycles;

    // =========================================================================
    // Step 2: DRAM miss handling
    // =========================================================================
    // If DRAMsim3 is enabled, use cycle-accurate batched transactions for misses.
    // Otherwise fall back to fixed miss penalty per lookup.
    // =========================================================================
#ifdef ENABLE_DRAMSIM3_PE
    if (index_dram_ && !miss_requests.empty()) {
        long long batch_cycles = issue_batched_transactions(
            index_dram_, inflight_index_, miss_requests, /*is_write=*/false);
        ctrs_.index_dram_cycles += batch_cycles;
    } else if (!index_dram_ && miss_count > 0) {
        ctrs_.index_dram_cycles += static_cast<long long>(miss_count) * INDEX_STORE_DRAM_MISS_LATENCY;
    }
#else
    if (miss_count > 0) {
        ctrs_.index_dram_cycles += static_cast<long long>(miss_count) * INDEX_STORE_DRAM_MISS_LATENCY;
    }
#endif

    // =========================================================================
    // Step 3: Build A_sub CSR from fetched index data
    // =========================================================================
    // At this point, all index data has been "fetched" from DRAM.
    // In simulation, incoming_[] already holds the data (functional model).
    // The cycle cost was accounted for in Step 2.
    // =========================================================================

    for (int j : lookup_nodes) {
        const auto& inc = incoming_[j];
        for (const auto& e : inc) {
            if (!e.valid) continue;
            int i = e.src;
            if (i < 0 || i >= n_) continue;
            per_row[i].push_back({ j, e.w });
            affected_rows_last_.insert(i);
        }
    }

    // Build row_ptr
    int nnz = 0;
    for (int i = 0; i < n_; i++) {
        out.row_ptr[i] = nnz;
        nnz += static_cast<int>(per_row[i].size());
    }
    out.row_ptr[n_] = nnz;
    out.nnz = nnz;
    out.col_idx.resize(nnz);
    out.values.resize(nnz);

    // Fill
    int ptr = 0;
    for (int i = 0; i < n_; i++) {
        // sort by col for determinism
        auto& row = per_row[i];
        std::sort(row.begin(), row.end(),
                  [](const std::pair<int, float>& a, const std::pair<int, float>& b) {
                      return a.first < b.first;
                  });
        for (const auto& kv : row) {
            out.col_idx[ptr] = kv.first;
            out.values[ptr] = kv.second;
            ptr++;
        }
    }

    return out;
}

void DeltaFAwareIndexer::add_index_read_bytes_for_dst(int /*dst*/, uint64_t bytes) {
    // Round to cacheline granularity (intent) so later DRAM txn generation is consistent.
    uint64_t lines = (bytes + cacheline_bytes_ - 1) / cacheline_bytes_;
    ctrs_.index_bytes_read += lines * cacheline_bytes_;
}

void DeltaFAwareIndexer::add_index_write_bytes_for_dst(int /*dst*/, uint64_t bytes) {
    uint64_t lines = (bytes + cacheline_bytes_ - 1) / cacheline_bytes_;
    ctrs_.index_bytes_write += lines * cacheline_bytes_;
}

long long DeltaFAwareIndexer::account_fullscan_dram_cost(int n, int nnz) {
    // Fullscan reads entire A matrix CSR from DRAM:
    //   row_ptr: (n+1) × 4B
    //   col_idx: nnz × 4B
    //   values:  nnz × 4B
    // Total bytes = (n+1)*4 + nnz*8
    uint64_t total_bytes = static_cast<uint64_t>(n + 1) * 4 + static_cast<uint64_t>(nnz) * 8;
    ctrs_.index_bytes_read += total_bytes;
    ctrs_.index_lookup_ops += n;  // scanning all rows

#ifdef ENABLE_DRAMSIM3_PE
    if (index_dram_) {
        // Model as sequential read from a base address
        uint64_t fullscan_base = 0x30000000ULL;  // separate address space
        std::vector<std::pair<uint64_t, uint64_t>> reqs;
        reqs.emplace_back(fullscan_base, total_bytes);
        long long cycles = issue_batched_transactions(index_dram_, inflight_index_, reqs, /*is_write=*/false);
        ctrs_.index_dram_cycles += cycles;
        return cycles;
    }
#endif
    // Fallback: estimate cycles based on bytes / bandwidth
    // Assume HBM2 ~32GB/s effective → ~32 bytes/cycle @ 1GHz
    long long est_cycles = static_cast<long long>((total_bytes + 31) / 32);
    ctrs_.index_dram_cycles += est_cycles;
    return est_cycles;
}
