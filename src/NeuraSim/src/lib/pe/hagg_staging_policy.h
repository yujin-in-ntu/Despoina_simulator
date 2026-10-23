// SPDX-License-Identifier: Apache-2.0
#ifndef PE_HAGG_STAGING_POLICY_H
#define PE_HAGG_STAGING_POLICY_H

#include <cstddef>

#ifndef PE_HAGG_STAGING_MODEL
#define PE_HAGG_STAGING_MODEL 1
#endif
#ifndef PE_HAGG_STAGING_BYTES
#define PE_HAGG_STAGING_BYTES 262144
#endif
#ifndef PE_HAGG_STAGING_LINE_BYTES
#define PE_HAGG_STAGING_LINE_BYTES 256
#endif

namespace pe_hagg_staging {
constexpr int kModel = PE_HAGG_STAGING_MODEL;
constexpr bool kBounded = (kModel == 1);
constexpr std::size_t kBytes = static_cast<std::size_t>(PE_HAGG_STAGING_BYTES);
constexpr std::size_t kLineBytes = static_cast<std::size_t>(PE_HAGG_STAGING_LINE_BYTES);
constexpr std::size_t kCapacityRows = kBytes / kLineBytes;
static_assert(kModel == 0 || kModel == 1, "PE_HAGG_STAGING_MODEL must be 0 or 1");
static_assert(kBytes > 0 && kLineBytes > 0, "staging bytes and line bytes must be positive");
static_assert(kBytes % kLineBytes == 0, "staging capacity must contain whole row packets");

inline bool can_accept(std::size_t occupancy_rows) {
    return !kBounded || occupancy_rows < kCapacityRows;
}
inline std::size_t occupancy_bytes(std::size_t occupancy_rows) {
    return occupancy_rows * kLineBytes;
}
}  // namespace pe_hagg_staging

#endif
