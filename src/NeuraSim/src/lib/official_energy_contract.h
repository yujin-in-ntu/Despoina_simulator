// SPDX-License-Identifier: Apache-2.0
#ifndef DESPOINA_OFFICIAL_ENERGY_CONTRACT_H
#define DESPOINA_OFFICIAL_ENERGY_CONTRACT_H

#include <cstdint>

namespace despoina {
namespace official_energy {

// Final FP32 PE authority: Accelergy 0.4 / Aladdin estimator at 45 nm,
// scaled to 22 nm with E22 = E45 * (22/45)^2.
constexpr double kFp32ActionEnergy45nmPj = 38.55419140625;
constexpr double kTechnologyScale22From45 =
    (22.0 / 45.0) * (22.0 / 45.0);
constexpr double kFp32ActionEnergy22nmPj =
    kFp32ActionEnergy45nmPj * kTechnologyScale22From45;

// PE::total_mac_ops counts one multiply and one add as two scalar operations.
constexpr std::int64_t fused_mac_events_from_raw_scalar_ops(
    std::int64_t raw_scalar_ops) {
    return raw_scalar_ops / 2;
}

constexpr double fused_mac_energy_pj(std::int64_t raw_scalar_ops) {
    return static_cast<double>(
               fused_mac_events_from_raw_scalar_ops(raw_scalar_ops)) *
           kFp32ActionEnergy22nmPj;
}

// Dynamic row-mean operations are scalar FP32 operations, not fused MACs.
constexpr double row_mean_scalar_energy_pj(std::int64_t scalar_ops) {
    return static_cast<double>(scalar_ops) * kFp32ActionEnergy22nmPj;
}

}  // namespace official_energy
}  // namespace despoina

#endif
