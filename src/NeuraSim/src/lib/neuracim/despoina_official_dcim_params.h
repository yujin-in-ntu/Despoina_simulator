// SPDX-License-Identifier: Apache-2.0
#ifndef DESPOINA_OFFICIAL_DCIM_PARAMS_H
#define DESPOINA_OFFICIAL_DCIM_PARAMS_H

#include <cstdint>
#include <cstdlib>
#include <stdexcept>

#include "sha256_file.h"

namespace despoina_official_dcim {
struct Contract final {
    static constexpr const char* kSchema =
        "despoina.official-dcim.factorized-row-mean.v4";
    static constexpr const char* kProfile =
        "despoina-h6-dcim-final";
    static constexpr const char* kParamsSha256 =
        "de42eca133c9fea3bf064be6446b35f6afd265258b0f7dacb13053293cb15f1f";
    static constexpr const char* kOfficialPrimitiveSha256 =
        "c8f5bb82189618fbe8f10331d04fb444b57d9308a404f19c28e75be8c87cf794";
    static constexpr int kDimension = 128;
    static constexpr int kLanes = 16;
    static constexpr int kFifoRows = 16;
    static constexpr int kSramReadCycles = 2;
    static constexpr int kFullInt8ComputeCycles = 1;
    static constexpr int kAdcCycles = 0;
    static constexpr int kAccumulateCycles = 1;
    static constexpr int kOutputCycles = 2;
    static constexpr double kSramSensePrechargePjPerRow = 639.6399270054075;
    static constexpr double kSramWordlineDecoderPjPerRow = 35.5113137375654;
    static constexpr double kSramReadPjPerRow = 675.1512407429729;
    static constexpr double kCustomNoFifoPjPerRow = 4464.951643;
    static constexpr double kOfficialNoFifoPjPerRow = 5140.102883742973;
    static constexpr bool kWeightProgramLatencySupported = false;
    static constexpr bool kWeightProgramEnergySupported = false;
};

static_assert(Contract::kDimension == 128, "official DCIM dimension drift");
static_assert(Contract::kLanes == 16, "official DCIM lane-count drift");
static_assert(Contract::kFifoRows == 16, "official DCIM FIFO-depth drift");
static_assert(Contract::kAdcCycles == 0, "official digital profile cannot consume ADC timing");
static_assert(
    Contract::kSramReadPjPerRow ==
        Contract::kSramSensePrechargePjPerRow +
        Contract::kSramWordlineDecoderPjPerRow,
    "official NeuroSim SRAM read decomposition drift");
static_assert(!Contract::kWeightProgramLatencySupported,
              "official steady-state scope cannot claim program latency");
static_assert(!Contract::kWeightProgramEnergySupported,
              "official steady-state scope cannot claim program energy");

inline void validate_runtime_files() {
    static bool validated = false;
    if (validated) return;
    if (native_dcim_profile::sha256_file(std::getenv("DESPOINA_DCIM_PARAMS_JSON")) !=
        Contract::kParamsSha256) {
        throw std::runtime_error(
            "Despoina official DCIM immutable parameter SHA256 mismatch");
    }
    if (native_dcim_profile::sha256_file(
            std::getenv("DESPOINA_DCIM_PRIMITIVE_JSON")) !=
        Contract::kOfficialPrimitiveSha256) {
        throw std::runtime_error(
            "Despoina official DCIM NeuroSim primitive SHA256 mismatch");
    }
}

inline void require_runtime_contract(int d_in, int d_out, int lanes, int fifo) {
    validate_runtime_files();
    if (d_in != Contract::kDimension || d_out != Contract::kDimension ||
        lanes != Contract::kLanes || fifo != Contract::kFifoRows ||
        Contract::kWeightProgramLatencySupported ||
        Contract::kWeightProgramEnergySupported) {
        throw std::runtime_error(
            "Despoina official DCIM requires D128/N16/FIFO16 and resident weights");
    }
}
}  // namespace despoina_official_dcim

#endif
