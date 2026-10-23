// SPDX-License-Identifier: Apache-2.0
/**
 * neurosim_params.h
 * 
 * NeuroSim V1.4/V1.5 derived parameters for NeuraCiM module.
 * Parameters extracted from NeuroSim circuit-level models for SRAM CiM.
 * 
 * This file handles CiM (Compute-in-Memory) energy and timing estimation.
 * For on-chip SRAM caches/buffers, see cacti_params.h (CACTI 7.0 based).
 * 
 * Design Target: 22nm LSTP (optimal for Analog SRAM CiM)
 * ADC Configuration: 2-bit per cycle (current design)
 * 
 * References:
 * - NeuroSim V1.4: SubArray.cpp, SarADC.cpp, Param.cpp
 * - NeuroSim V1.5: Enhanced noise modeling, faster runtime
 * - Technology: 22nm LSTP (Low Standby Power)
 * 
 * HOW TO UPDATE:
 * 1. Download NeuroSim V1.5 from GitHub
 * 2. Configure for 22nm SRAM CiM with 2-bit ADC
 * 3. Run simulation and update values marked [NeuroSim]
 */

#ifndef NEUROSIM_PARAMS_H
#define NEUROSIM_PARAMS_H

namespace neurosim {

// =============================================================================
// Technology Parameters (22nm LSTP)
// =============================================================================
constexpr int TECH_NODE_NM = 22;
constexpr double FEATURE_SIZE_M = 22e-9;
constexpr double VDD_V = 0.8;                     // Supply voltage @ 22nm LSTP

// =============================================================================
// SRAM SubArray Parameters
// Ref: NeuroSim Param.cpp:143-144
// =============================================================================
constexpr int SUBARRAY_ROWS = 128;
constexpr int SUBARRAY_COLS = 128;
constexpr int NUM_COL_MUXED = 8;                  // Columns sharing one ADC

// SRAM Cell (6T)
// Ref: ITRS 2015 roadmap, 160F² for 6T SRAM
constexpr double CELL_HEIGHT_F = 10.0;
constexpr double CELL_WIDTH_F = 16.0;
constexpr double CELL_AREA_UM2 = 0.116;           // 160F² @ 22nm + 50% routing

// =============================================================================
// ADC Configuration - 2-bit per cycle (Current Design)
// Ref: DyGNN config.py SRAMCiMConfig
// =============================================================================
namespace adc_2bit {
    constexpr int BITS_PER_CYCLE = 2;             // 2-bit parallelism
    constexpr int NUM_LEVELS = 4;                 // 2^2 levels
    constexpr double BL_SWING_MV = 200.0;         // Bit-line voltage swing
    constexpr double MV_PER_LEVEL = 50.0;         // 200mV / 4 levels
    constexpr double NOISE_MARGIN_MV = 25.0;      // Half of mV_per_level
    
    // Timing [NeuroSim V1.5]
    constexpr double ADC_DELAY_NS = 2.5;          // 2-bit SAR: faster than 8-bit
    constexpr int ADC_CYCLES = 3;                 // ceil(2.5ns / 1ns)
    
    // Energy [NeuroSim V1.5]
    constexpr double ADC_ENERGY_PJ = 0.04;        // Per 2-bit conversion
    
    // Area [NeuroSim V1.5]
    constexpr double ADC_AREA_UM2 = 56.0;         // Smaller than 8-bit
}

// =============================================================================
// ADC Configuration - 8-bit (Reference, not used in current design)
// =============================================================================
namespace adc_8bit {
    constexpr int BITS_PER_CYCLE = 8;
    constexpr int NUM_LEVELS = 256;
    
    // Timing
    constexpr double ADC_DELAY_NS = 9.0;          // 8-bit SAR: (log2(256)+1)*1ns
    constexpr int ADC_CYCLES = 9;
    
    // Energy
    constexpr double ADC_ENERGY_PJ = 0.16;        // Per 8-bit conversion
    
    // Area
    constexpr double ADC_AREA_UM2 = 225.0;        // 8-bit SAR ADC
}

// =============================================================================
// Input DAC Parameters (Activation quantizer for Analog CIM)
// WL voltage modulation via R-2R ladder DAC @ 22nm
// =============================================================================
namespace input_dac {
    constexpr double DELAY_8BIT_NS = 1.0;          // 8-bit DAC settling time
    constexpr double ENERGY_PJ = 0.03;             // Per subarray activation (128 WL DACs)
}

// =============================================================================
// Timing Parameters (ns) - 22nm CiM with 2-bit ADC
// Ref: NeuroSim SubArray.cpp::CalculateLatency()
// =============================================================================
constexpr double PRECHARGE_DELAY_NS = 0.5;        // BL precharge
constexpr double SENSE_DELAY_NS = 1.0;            // Row activation + sensing
constexpr double ADC_DELAY_NS = adc_2bit::ADC_DELAY_NS;  // Using 2-bit ADC
constexpr double SHIFT_ADD_DELAY_NS = 2.0;        // Shift-add accumulation

// Pipeline stage latencies (cycles @ 1GHz)
constexpr int CYCLES_PRECHARGE = 1;
constexpr int CYCLES_SENSE = 1;
constexpr int CYCLES_ADC = adc_2bit::ADC_CYCLES;  // 3 cycles for 2-bit
constexpr int CYCLES_SHIFT_ADD = 2;

// Total read latency per row activation (cycles)
constexpr int CYCLES_PER_ROW = CYCLES_PRECHARGE + CYCLES_SENSE + 
                               CYCLES_ADC + CYCLES_SHIFT_ADD;  // ~7 cycles

// =============================================================================
// Energy Parameters (pJ) - 22nm CiM
// Ref: NeuroSim SubArray.cpp::CalculatePower(), SarADC.cpp
// Values calibrated with NeuroSim V1.5 output
// =============================================================================

// Array energy [NeuroSim]
constexpr double ARRAY_READ_ENERGY_PJ = 0.25;     // Per row activation (128 cols)
constexpr double ARRAY_WRITE_ENERGY_PJ = 0.35;    // Per row write (higher due to WL driver)

// ADC energy - using 2-bit configuration
constexpr double ADC_ENERGY_PJ = adc_2bit::ADC_ENERGY_PJ;

// Shift-add accumulator energy [NeuroSim]
constexpr double SHIFT_ADD_ENERGY_PJ = 0.2;       // Per accumulation

// Leakage power [NeuroSim]
constexpr double LEAKAGE_POWER_MW = 0.13;         // Per 128x128 subarray

// =============================================================================
// Energy Breakdown per Operation
// =============================================================================

// Energy per MAC operation in CiM (amortized over all columns)
// One row activation processes SUBARRAY_COLS MACs simultaneously
constexpr double ENERGY_PER_CIM_MAC_PJ = 
    (ARRAY_READ_ENERGY_PJ + ADC_ENERGY_PJ * NUM_COL_MUXED + SHIFT_ADD_ENERGY_PJ) / SUBARRAY_COLS;
    // ≈ (0.25 + 0.04*8 + 0.2) / 128 ≈ 0.006 pJ/MAC

// For comparison: typical digital MAC @ 22nm ≈ 0.1-0.5 pJ/MAC
// CiM advantage: ~20-80x more energy efficient per MAC

// =============================================================================
// Area Parameters (um²)
// Ref: NeuroSim SubArray.cpp::CalculateArea()
// =============================================================================

// Using 2-bit ADC area
constexpr double ADC_AREA_UM2 = adc_2bit::ADC_AREA_UM2;

// Peripheral circuit overhead
constexpr double PERIPHERAL_OVERHEAD = 1.3;       // 30% for decoders, drivers, etc.

// Subarray area calculation
constexpr double SUBARRAY_AREA_UM2 = 
    SUBARRAY_ROWS * SUBARRAY_COLS * CELL_AREA_UM2 * PERIPHERAL_OVERHEAD;
    // ≈ 128 * 128 * 0.116 * 1.3 ≈ 2470 um²

// =============================================================================
// Derived Parameters
// =============================================================================

// Clock frequency assumption (1 GHz)
constexpr double CLOCK_FREQ_GHZ = 1.0;
constexpr double CLOCK_PERIOD_NS = 1.0 / CLOCK_FREQ_GHZ;

// Total energy per row of CiM computation (one row of H_agg × W slice)
constexpr double ENERGY_PER_ROW_PJ = 
    ARRAY_READ_ENERGY_PJ + (ADC_ENERGY_PJ * NUM_COL_MUXED) + SHIFT_ADD_ENERGY_PJ;
    // ≈ 0.25 + 0.32 + 0.2 = 0.77 pJ

// =============================================================================
// NeuraCiM Module Parameters (16 tiles configuration)
// =============================================================================
namespace neuracim_config {
    constexpr int NUM_TILES = 16;                 // Parallel CiM tiles
    constexpr int TILE_ROWS = SUBARRAY_ROWS;      // 128 rows per tile
    constexpr int TILE_COLS = SUBARRAY_COLS;      // 128 cols per tile
    
    // Throughput: NUM_TILES rows per cycle (pipelined)
    constexpr int ROWS_PER_CYCLE = NUM_TILES;     // 16 rows/cycle
    
    // Total CiM array size
    constexpr int TOTAL_ROWS = NUM_TILES * TILE_ROWS;  // 2048
    constexpr int TOTAL_COLS = TILE_COLS;              // 128
    
    // Total CiM SRAM size
    constexpr double TOTAL_SRAM_KB = 
        (double)(NUM_TILES * TILE_ROWS * TILE_COLS * 2) / 1024.0;  // 512 KB (FP16 weights)
    
    // Total ADC area
    constexpr double TOTAL_ADC_AREA_UM2 = NUM_TILES * ADC_AREA_UM2;
    constexpr double TOTAL_ADC_AREA_MM2 = TOTAL_ADC_AREA_UM2 / 1e6;
}

// =============================================================================
// Noise Analysis Parameters (for Monte Carlo simulation)
// =============================================================================
namespace noise {
    // Thermal noise @ 22nm
    constexpr double THERMAL_NOISE_MV = 8.0;      // σ at room temperature
    
    // Process variation
    constexpr double VTH_VARIATION_PCT = 5.0;     // Threshold voltage variation
    
    // Acceptable bit error rate
    constexpr double MAX_BER = 0.01;              // 1% maximum
    
    // 2-bit ADC noise margin analysis
    // At 22nm with 50mV/level and 10mV noise: adequate margin
    // At 7nm: noise increases to ~20mV, margin becomes insufficient
    constexpr bool IS_NOISE_SAFE_22NM = true;
    constexpr bool IS_NOISE_SAFE_14NM = true;     // Borderline
    constexpr bool IS_NOISE_SAFE_7NM = false;     // Requires Digital CiM
}

// =============================================================================
// Backward Compatibility
// =============================================================================

// Legacy parameters (deprecated, use specific namespaces above)
constexpr int ADC_PRECISION = adc_2bit::BITS_PER_CYCLE;
constexpr int ADC_LEVELS = adc_2bit::NUM_LEVELS;

// Legacy energy per MAC (for old code compatibility)
constexpr double ENERGY_PER_MAC_PJ = ENERGY_PER_CIM_MAC_PJ;

// Legacy cycles per activation
constexpr int CYCLES_PER_ACTIVATION = CYCLES_PER_ROW;

// =============================================================================
// Digital CiM (DCIM) Timing Parameters
// INT8×INT8 MAC in/near SRAM — no ADC/DAC needed.
// Timing only; energy comes from energy_params_generated.h (Accelergy pipeline).
// =============================================================================
namespace dcim {
    constexpr int CYCLES_SRAM_READ = 2;     // Precharge + sense (same physical SRAM)
    constexpr int CYCLES_MULTIPLY = 1;      // INT8×INT8 parallel multiply
    constexpr int CYCLES_ACCUMULATE = 1;    // Adder tree reduction
    // ADC = 0 cycles (no ADC in digital path)
    // DAC = 0 cycles (no DAC in digital path)
    // serial_steps = 1 (full 8-bit processed at once)

    // Activation quantizer (FP32→INT8): comparator-tree max-abs + per-element
    // scale-round-clamp.  Overlapped with SRAM precharge (2 cycles slack).
    constexpr int CYCLES_QUANTIZE = 1;

    // Dequantizer (INT32→FP32 + scale multiply): per-element conversion at
    // CiM output, before ReLU.  Adds 1 cycle to OUTPUT_READY stage.
    constexpr int CYCLES_DEQUANTIZE = 1;
}

}  // namespace neurosim

#endif  // NEUROSIM_PARAMS_H
