// SPDX-License-Identifier: Apache-2.0
#include <cstddef>
#include <iostream>
#include <queue>
#include "lib/pe/hagg_staging_policy.h"

int main() {
    std::queue<int> q;
    std::size_t stalls = 0;
    for (std::size_t i = 0; i < pe_hagg_staging::kCapacityRows; ++i) {
        if (!pe_hagg_staging::can_accept(q.size())) return 10;
        q.push(static_cast<int>(i));
    }
    const bool accepted_1025 = pe_hagg_staging::can_accept(q.size());
    if (accepted_1025) q.push(1024); else stalls++;
    if (pe_hagg_staging::kBounded && accepted_1025) return 11;
    if (!pe_hagg_staging::kBounded && !accepted_1025) return 12;
    q.pop();
    if (!pe_hagg_staging::can_accept(q.size())) return 13;
    q.push(2048);
    std::cout << "{\"model\":" << pe_hagg_staging::kModel
              << ",\"bounded\":" << (pe_hagg_staging::kBounded ? "true" : "false")
              << ",\"bytes\":" << pe_hagg_staging::kBytes
              << ",\"line_bytes\":" << pe_hagg_staging::kLineBytes
              << ",\"capacity_rows\":" << pe_hagg_staging::kCapacityRows
              << ",\"accepted_1025\":" << (accepted_1025 ? "true" : "false")
              << ",\"stalls\":" << stalls
              << ",\"resume_after_pop\":true}" << std::endl;
    return 0;
}
