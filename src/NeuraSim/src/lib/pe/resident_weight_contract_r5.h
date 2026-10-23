// SPDX-License-Identifier: Apache-2.0
#ifndef RESIDENT_WEIGHT_CONTRACT_R5_H
#define RESIDENT_WEIGHT_CONTRACT_R5_H

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace despoina_r5 {

struct ResidentWeightKey {
    ResidentWeightKey() = default;
    ResidentWeightKey(int layer, uint64_t ver, uint64_t hash, int r, int c)
        : layer_id(layer), version(ver), source_hash(hash), rows(r), cols(c) {}
    int layer_id = -1;
    uint64_t version = 0;
    uint64_t source_hash = 0;
    int rows = 0;
    int cols = 0;
};

inline bool same_weight_key(const ResidentWeightKey& a, const ResidentWeightKey& b) {
    return a.layer_id == b.layer_id && a.version == b.version &&
           a.source_hash == b.source_hash && a.rows == b.rows && a.cols == b.cols;
}

// One physical DCIM stores exactly one current-layer W.  A layer/version/shape
// transition requires one load; repeated finalized-row commits for the same key
// reuse it.  The quantized hash prevents a private/legacy load from silently
// replacing the tracked resident image.
class ResidentWeightContract {
public:
    bool requires_load(const ResidentWeightKey& requested) const {
        validate_key(requested);
        return !valid_ || !same_weight_key(current_, requested);
    }

    void commit_load(const ResidentWeightKey& loaded, uint64_t quantized_hash) {
        validate_key(loaded);
        if (quantized_hash == 0) throw std::runtime_error("resident INT8 hash is invalid");
        current_ = loaded;
        quantized_hash_ = quantized_hash;
        valid_ = true;
        ++load_events_;
    }

    bool validate(const ResidentWeightKey& requested, uint64_t quantized_hash) const {
        validate_key(requested);
        return valid_ && same_weight_key(current_, requested) && quantized_hash != 0 &&
               quantized_hash == quantized_hash_;
    }

    uint64_t load_events() const { return load_events_; }
    const ResidentWeightKey& current() const { return current_; }

private:
    static void validate_key(const ResidentWeightKey& key) {
        if (key.layer_id < 1 || key.version == 0 || key.source_hash == 0 ||
            key.rows <= 0 || key.cols <= 0) {
            throw std::runtime_error("invalid resident weight key/version/shape");
        }
    }

    bool valid_ = false;
    ResidentWeightKey current_{};
    uint64_t quantized_hash_ = 0;
    uint64_t load_events_ = 0;
};

inline uint64_t fnv1a64(const void* data, size_t bytes) {
    const uint8_t* p = static_cast<const uint8_t*>(data);
    uint64_t h = 1469598103934665603ULL;
    for (size_t i = 0; i < bytes; ++i) {
        h ^= static_cast<uint64_t>(p[i]);
        h *= 1099511628211ULL;
    }
    return h;
}

// R5.1 cold-start gate shared by production and unit tests.  A tile is usable
// only after its full replicated INT8 image has a nonzero matching hash and a
// causal programming-completion timestamp.  Merely creating the NeuraCiM
// object or quantizing the source is not readiness.
inline bool all_replica_tiles_ready(
    size_t expected_tiles,
    uint64_t expected_quantized_hash,
    const std::vector<uint8_t>& loaded,
    const std::vector<uint64_t>& quantized_hash_per_tile,
    const std::vector<uint64_t>& program_done_cycle_per_tile) {
    if (expected_tiles == 0 || expected_quantized_hash == 0 ||
        loaded.size() != expected_tiles ||
        quantized_hash_per_tile.size() != expected_tiles ||
        program_done_cycle_per_tile.size() != expected_tiles) {
        return false;
    }
    for (size_t tile = 0; tile < expected_tiles; ++tile) {
        if (loaded[tile] != 1 ||
            quantized_hash_per_tile[tile] != expected_quantized_hash ||
            program_done_cycle_per_tile[tile] == 0) {
            return false;
        }
    }
    return true;
}

inline uint64_t replicated_int8_program_bytes(size_t weight_elements,
                                               size_t physical_tiles) {
    if (weight_elements == 0 || physical_tiles == 0) {
        throw std::runtime_error("invalid replicated weight program shape");
    }
    return static_cast<uint64_t>(weight_elements) *
           static_cast<uint64_t>(physical_tiles) * sizeof(int8_t);
}

}  // namespace despoina_r5

#endif
