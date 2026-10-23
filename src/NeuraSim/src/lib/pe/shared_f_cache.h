// SPDX-License-Identifier: Apache-2.0
#ifndef SHARED_F_CACHE_H
#define SHARED_F_CACHE_H

#include <unordered_map>
#include <list>
#include <vector>
#include <cstddef>
#include <algorithm>

#ifndef PE_SHARED_F_CACHE_BANKS
#define PE_SHARED_F_CACHE_BANKS 64
#endif

#ifndef PE_SHARED_F_CACHE_PORTS_PER_BANK
#define PE_SHARED_F_CACHE_PORTS_PER_BANK 1
#endif

/**
 * Shared F-Cache (row cache) across all PE clusters.
 *
 * Motivation:
 * - Current design has per-cluster MRU caches, which cannot exploit cross-cluster reuse
 *   (same F row often needed by multiple clusters).
 * - This shared cache models a chip-level feature cache (e.g., ~4MB) to reduce DRAM reads.
 *
 * Implementation:
 * - LRU replacement over row_id
 * - Stores full row data (dim floats) per entry
 */
class SharedFCache {
public:
    static constexpr int NUM_BANKS = PE_SHARED_F_CACHE_BANKS;
    static constexpr int PORTS_PER_BANK = PE_SHARED_F_CACHE_PORTS_PER_BANK;
    static constexpr int SAFE_NUM_BANKS = (NUM_BANKS > 0 ? NUM_BANKS : 1);
    static constexpr int SAFE_PORTS_PER_BANK = (PORTS_PER_BANK > 0 ? PORTS_PER_BANK : 1);

    SharedFCache() = default;
    explicit SharedFCache(std::size_t capacity_entries, int feature_dim)
        : capacity_entries_(capacity_entries), feature_dim_(feature_dim) {
        init_bank_tracker();
    }

    void configure(std::size_t capacity_entries, int feature_dim) {
        capacity_entries_ = capacity_entries;
        feature_dim_ = feature_dim;
        init_bank_tracker();
        clear();
    }

    void clear() {
        map_.clear();
        lru_.clear();
        hits_ = 0;
        misses_ = 0;
        bank_conflict_stalls_ = 0;
        bank_accesses_ = 0;
        reset_bank_tracker();
    }

    // Reset hit/miss counters without evicting data.
    void reset_stats() {
        hits_ = 0;
        misses_ = 0;
        bank_conflict_stalls_ = 0;
        bank_accesses_ = 0;
    }

    std::size_t capacity_entries() const { return capacity_entries_; }
    std::size_t size() const { return map_.size(); }
    int feature_dim() const { return feature_dim_; }

    long long hits() const { return hits_; }
    long long misses() const { return misses_; }
    long long bank_conflict_stalls() const { return bank_conflict_stalls_; }
    long long bank_accesses() const { return bank_accesses_; }

    // Return pointer to cached row data if present, else nullptr.
    // Updates LRU on hit.
    const float* get_ptr(int row_id, long long request_cycle = 0, int* extra_stall_cycles = nullptr) {
        int bank_stall = reserve_bank_port(row_id, request_cycle);
        if (extra_stall_cycles != nullptr) {
            *extra_stall_cycles = bank_stall;
        }

        auto it = map_.find(row_id);
        if (it == map_.end()) {
            misses_++;
            return nullptr;
        }
        hits_++;
        // move to back (most recently used)
        lru_.splice(lru_.end(), lru_, it->second);
        return it->second->data.data();
    }

    // Insert/overwrite row data and update LRU.
    // Assumes data has at least feature_dim floats.
    void put(int row_id, const float* data, long long request_cycle = 0, int* extra_stall_cycles = nullptr) {
        if (!data || feature_dim_ <= 0 || capacity_entries_ == 0) return;

        int bank_stall = reserve_bank_port(row_id, request_cycle);
        if (extra_stall_cycles != nullptr) {
            *extra_stall_cycles = bank_stall;
        }

        auto it = map_.find(row_id);
        if (it != map_.end()) {
            // overwrite existing
            auto node_it = it->second;
            node_it->data.assign(data, data + feature_dim_);
            lru_.splice(lru_.end(), lru_, node_it);
            return;
        }

        // evict if full
        if (map_.size() >= capacity_entries_) {
            auto& victim = lru_.front();
            map_.erase(victim.row_id);
            lru_.pop_front();
        }

        Node node;
        node.row_id = row_id;
        node.data.assign(data, data + feature_dim_);
        lru_.push_back(std::move(node));
        auto new_it = std::prev(lru_.end());
        map_[row_id] = new_it;
    }

    // Invalidate one cached row if present (used for ΔF coherence).
    void invalidate(int row_id) {
        auto it = map_.find(row_id);
        if (it == map_.end()) return;
        lru_.erase(it->second);
        map_.erase(it);
    }

private:
    struct Node {
        int row_id = -1;
        std::vector<float> data;
    };

    std::size_t capacity_entries_ = 0;
    int feature_dim_ = 0;

    std::list<Node> lru_;
    std::unordered_map<int, std::list<Node>::iterator> map_;

    // [bank][port] -> next cycle the port becomes available.
    std::vector<std::vector<long long>> bank_port_next_free_cycle_;

    long long hits_ = 0;
    long long misses_ = 0;
    long long bank_conflict_stalls_ = 0;
    long long bank_accesses_ = 0;

    void init_bank_tracker() {
        bank_port_next_free_cycle_.assign(SAFE_NUM_BANKS, std::vector<long long>(SAFE_PORTS_PER_BANK, 0));
    }

    void reset_bank_tracker() {
        for (auto& bank_ports : bank_port_next_free_cycle_) {
            for (auto& next_free : bank_ports) {
                next_free = 0;
            }
        }
    }

    int get_bank_id(int row_id) const {
        int bank = row_id % SAFE_NUM_BANKS;
        if (bank < 0) bank += SAFE_NUM_BANKS;
        return bank;
    }

    int reserve_bank_port(int row_id, long long request_cycle) {
        if (bank_port_next_free_cycle_.empty()) {
            init_bank_tracker();
        }
        const int bank_id = get_bank_id(row_id);
        auto& ports = bank_port_next_free_cycle_[bank_id];

        int selected_port = 0;
        long long selected_start = std::max(request_cycle, ports[0]);
        for (int p = 1; p < SAFE_PORTS_PER_BANK; ++p) {
            const long long candidate_start = std::max(request_cycle, ports[p]);
            if (candidate_start < selected_start) {
                selected_start = candidate_start;
                selected_port = p;
            }
        }

        const int stall_cycles = static_cast<int>(selected_start - request_cycle);
        ports[selected_port] = selected_start + 1;
        bank_accesses_++;
        if (stall_cycles > 0) {
            bank_conflict_stalls_ += stall_cycles;
        }
        return stall_cycles;
    }
};

#endif  // SHARED_F_CACHE_H

