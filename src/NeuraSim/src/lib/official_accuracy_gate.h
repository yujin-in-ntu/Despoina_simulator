// SPDX-License-Identifier: Apache-2.0
#ifndef DESPOINA_OFFICIAL_ACCURACY_GATE_H
#define DESPOINA_OFFICIAL_ACCURACY_GATE_H

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

namespace despoina {
namespace official_accuracy {

inline bool same_domain_pass(const std::vector<float>& produced,
                             const std::vector<float>& golden,
                             double relative_tolerance = 0.05,
                             double absolute_tolerance = 1e-4,
                             double required_hit_rate = 0.99) {
    if (produced.size() != golden.size() || produced.empty()) return false;
    std::size_t hits = 0;
    for (std::size_t i = 0; i < produced.size(); ++i) {
        const double error =
            std::abs(static_cast<double>(produced[i]) - golden[i]);
        const double relative =
            error / std::max(std::abs(static_cast<double>(golden[i])), 1e-6);
        if (error <= absolute_tolerance || relative <= relative_tolerance) ++hits;
    }
    return static_cast<double>(hits) / produced.size() >= required_hit_rate;
}

}  // namespace official_accuracy
}  // namespace despoina

#endif
