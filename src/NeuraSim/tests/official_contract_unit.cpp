// SPDX-License-Identifier: Apache-2.0
#include "../src/lib/official_accuracy_gate.h"
#include "../src/lib/official_energy_contract.h"

#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>

int main() {
    using namespace despoina::official_energy;
    assert(std::abs(kFp32ActionEnergy22nmPj -
                    9.214927723765432) < 1e-12);
    assert(fused_mac_events_from_raw_scalar_ops(200) == 100);
    assert(std::abs(fused_mac_energy_pj(200) -
                    100.0 * kFp32ActionEnergy22nmPj) < 1e-10);
    assert(std::abs(row_mean_scalar_energy_pj(200) -
                    200.0 * kFp32ActionEnergy22nmPj) < 1e-10);
    assert(row_mean_scalar_energy_pj(200) == 2.0 * fused_mac_energy_pj(200));

    std::vector<float> golden(256);
    for (std::size_t i = 0; i < golden.size(); ++i) {
        golden[i] = static_cast<float>((static_cast<int>(i % 17) - 8) / 8.0);
    }
    const std::vector<float> produced = golden;
    assert(despoina::official_accuracy::same_domain_pass(produced, golden));

    std::vector<float> corrupted = produced;
    for (std::size_t i = 0; i < corrupted.size(); i += 8) {
        corrupted[i] += 0.5f;
    }
    assert(!despoina::official_accuracy::same_domain_pass(corrupted, golden));
    std::cout << "OFFICIAL_CONTRACT_UNIT_PASS\n";
    return 0;
}
