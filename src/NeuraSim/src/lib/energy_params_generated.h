// SPDX-License-Identifier: Apache-2.0
/**
 * energy_params_generated.h
 * 
 * Integrated energy parameters for DyGNN Accelerator.
 * 
 * AUTO-GENERATED FILE - DO NOT EDIT MANUALLY
 * Generated: 2026-03-27 07:35:32
 * Generator: energy_curator.py
 * 
 * Sources:
 * - CACTI 7.0: SRAM (F-Cache, IndexStore, L1 MRU, Indexer MSHR SRAM/CAM, PE Registers, Shared Buffer)
 * - Accelergy: PE Array + NoC (NeuraRouter 2D Mesh)
 * - NeuroSim V1.5: CiM
 * - DRAMsim3: Off-chip DRAM (separate integration)
 * 
 * Architecture: DyGNN Accelerator
 * Technology: 22nm
 */

#ifndef ENERGY_PARAMS_GENERATED_H
#define ENERGY_PARAMS_GENERATED_H

namespace energy {

// =============================================================================
// Architecture Configuration
// =============================================================================
namespace config {
    constexpr int TECH_NODE_NM = 22;
    constexpr double CLOCK_FREQ_GHZ = 1.0;
    constexpr int TOTAL_PES = 256;
    constexpr int NUM_CLUSTERS = 64;
    constexpr int SIMD_WIDTH = 8;
}

// =============================================================================
// SRAM Energy (from CACTI 7.0)
// NOTE: H-tree energy excluded - we use 2D Mesh NoC instead
// =============================================================================
namespace sram {

namespace f_cache {
    // 4MB MRU Cache, 64 banks, 4-way
    constexpr double AREA_MM2 = 6.9975;
    constexpr double ACCESS_TIME_NS = 5.2110;
    
    // Energy (H-tree EXCLUDED for NoC designs)
    constexpr double READ_ENERGY_PJ = 10.43;   // Core SRAM only
    constexpr double WRITE_ENERGY_PJ = 16.79;
    constexpr double LEAKAGE_POWER_MW = 0.1633;
    
    // Original CACTI values (for reference)
    constexpr double TOTAL_READ_ENERGY_PJ = 828.03;  // Including H-tree
    constexpr double HTREE_ENERGY_PJ = 817.59;
    
    constexpr int ACCESS_CYCLES = 6;
}

namespace index_store {
    // IndexStore SRAM, 32 banks
    constexpr double AREA_MM2 = 1.6119;
    constexpr double ACCESS_TIME_NS = 2.4904;
    
    // Energy (H-tree EXCLUDED for NoC designs)
    constexpr double READ_ENERGY_PJ = 8.02;
    constexpr double WRITE_ENERGY_PJ = 1.89;
    constexpr double LEAKAGE_POWER_MW = 0.0355;
    
    // Original CACTI values (for reference)
    constexpr double TOTAL_READ_ENERGY_PJ = 97.82;
    constexpr double HTREE_ENERGY_PJ = 89.80;
    
    constexpr int ACCESS_CYCLES = 3;
}

namespace l1_mru {
    // Per-cluster local MRU cache, aggregated across clusters.
    constexpr int ENTRIES_PER_CLUSTER = 64;
    constexpr int LINE_SIZE_BYTES = 512;
    constexpr int CLUSTERS = 64;

    constexpr double AREA_MM2_TOTAL = 4.896614;
    constexpr double AREA_MM2_PER_CLUSTER = 0.076510;
    constexpr double ACCESS_TIME_NS = 0.9242;

    // Energy per local cache access
    constexpr double READ_ENERGY_PJ = 0.51;
    constexpr double WRITE_ENERGY_PJ = 0.62;

    constexpr double LEAKAGE_POWER_MW_TOTAL = 1.350406;
    constexpr double LEAKAGE_POWER_MW_PER_CLUSTER = 0.021100;
    constexpr int ACCESS_CYCLES = 1;
}

namespace indexer_mshr_sram {
    // Indexer controller MSHR request buffer (SRAM)
    constexpr double AREA_MM2 = 0.003241;
    constexpr double ACCESS_TIME_NS = 0.3861;
    constexpr double READ_ENERGY_PJ = 0.01;
    constexpr double WRITE_ENERGY_PJ = 0.01;
    constexpr double LEAKAGE_POWER_MW = 0.000929;
    constexpr int ACCESS_CYCLES = 1;
}

namespace indexer_mshr_cam {
    // Indexer controller tag store (CACTI SRAM proxy; CACTI CAM mode is unstable in this build)
    constexpr double AREA_MM2 = 0.004203;
    constexpr double ACCESS_TIME_NS = 0.4490;
    constexpr double SEARCH_ENERGY_PJ = 0.02;
    constexpr double LEAKAGE_POWER_MW = 0.001580;
    constexpr int ACCESS_CYCLES = 1;
}

namespace pe_registers {
    // 128KB total, 512B per PE
    constexpr double AREA_MM2 = 0.848042;
    constexpr double AREA_PER_PE_MM2 = 0.003313;
    constexpr double ACCESS_TIME_NS = 1.8149;
    
    constexpr double READ_ENERGY_PJ = 1.73;
    constexpr double WRITE_ENERGY_PJ = 1.90;
    constexpr double LEAKAGE_POWER_MW = 0.017235;
    
    constexpr int ACCESS_CYCLES = 2;
}

namespace shared_buffer {
    // 256KB total, 4KB per cluster
    constexpr double AREA_MM2 = 0.850786;
    constexpr double AREA_PER_CLUSTER_MM2 = 0.013294;
    constexpr double ACCESS_TIME_NS = 1.9200;
    
    constexpr double READ_ENERGY_PJ = 8.67;
    constexpr double WRITE_ENERGY_PJ = 9.53;
    constexpr double LEAKAGE_POWER_MW = 0.057568;
    
    constexpr int ACCESS_CYCLES = 2;
}

}  // namespace sram

// =============================================================================
// NoC Energy (from Accelergy: crossbar + FIFO + wire)
// =============================================================================
namespace noc {
    // 2D Mesh: 8x4 routers, 9 ports each
    constexpr double ROUTER_ENERGY_PJ_PER_FLIT = 0.97;
    constexpr double LINK_ENERGY_PJ_PER_BIT_MM = 0.15;
    constexpr double ROUTER_AREA_MM2 = 0.0079;
    constexpr double ROUTER_LEAKAGE_MW = 0.00;
    
    constexpr int TOTAL_ROUTERS = 32;
    constexpr double AVG_HOPS = 4.0;
    constexpr double LINK_LENGTH_MM = 0.50;
    
    // Pre-calculated: F row transfer (256 bytes, avg hops)
    constexpr double F_ROW_TRANSFER_PJ = 739.01;
    
    // Total NoC area
    constexpr double TOTAL_NOC_AREA_MM2 = ROUTER_AREA_MM2 * TOTAL_ROUTERS;
}

// =============================================================================
// PE Array Energy (from Accelergy)
// Source: accelergy
// =============================================================================
namespace pe_array {
    // SIMD MAC operation (8-wide FP16)
    constexpr double SIMD_MAC_ENERGY_PJ = 4.71;    // 8x FP16 MAC
    constexpr double MAC_PER_UNIT_PJ = 0.59;       // Per FP16 MAC
    
    // Register file access
    constexpr double F_REG_READ_ENERGY_PJ = 2.02;   // Read 8 values
    constexpr double F_REG_WRITE_ENERGY_PJ = 32.30; // Write 128 values
    constexpr double PSUM_RMW_ENERGY_PJ = 4.04;       // Read-modify-write 8 values
    constexpr double A_REG_ACCESS_ENERGY_PJ = 0.00;      // A value read
    
    // PE metrics
    constexpr double PE_AREA_MM2 = 0.0085;
    constexpr double PE_LEAKAGE_MW = 0.3720;
    constexpr int TOTAL_PES = 256;
    
    // Total PE Array
    constexpr double TOTAL_PE_AREA_MM2 = PE_AREA_MM2 * TOTAL_PES;
    constexpr double TOTAL_PE_LEAKAGE_MW = PE_LEAKAGE_MW * TOTAL_PES;
}

// =============================================================================
// CiM Energy (from NeuroSim V1.5)
// =============================================================================
namespace cim {
    constexpr double MAC_ENERGY_PJ = 0.0269;
    constexpr double ROW_ENERGY_PJ = 0.22;
    constexpr double ADC_ENERGY_PJ = 0.02;
    constexpr double ARRAY_READ_ENERGY_PJ = 3.44;
    constexpr double LEAKAGE_POWER_MW = 2.24;
    constexpr double AREA_MM2 = 0.0044;
}

// =============================================================================
// Digital CiM (DCIM) Peripheral Energy/Area (from Accelergy)
// =============================================================================
namespace dcim {
    // Per-op energies
    constexpr double MULTIPLIER_ENERGY_PJ = 0.239731;      // INT8 multiply
    constexpr double ADDER_ENERGY_PJ = 0.031763;          // single adder access
    constexpr double COMPARATOR_ENERGY_PJ = 0.063525;       // comparator proxy access

    // Aggregated per-output tree energies
    constexpr int ADDER_COUNT = 127;
    constexpr int COMPARATOR_COUNT = 127;
    constexpr int MULTIPLIER_COUNT = 128;
    constexpr double ADDER_TREE_ENERGY_PJ = 4.033838;       // per output element
    constexpr double COMPARATOR_TREE_ENERGY_PJ = 8.067675;    // per output element

    // Despoina official DCIM: direct NeuroSim V1.4 digital sequential SRAM
    // macro only.
    // Generic NeuroSim accumulation/transport remain excluded because the
    // downstream datapath is the custom Accelergy-backed logic below.
    constexpr double SRAM_SENSE_PRECHARGE_ARRAY_ENERGY_PJ = 639.6399270054075;
    constexpr double SRAM_WORDLINE_DECODER_ENERGY_PJ = 35.5113137375654;
    constexpr double SRAM_READ_ENERGY_PJ =
        SRAM_SENSE_PRECHARGE_ARRAY_ENERGY_PJ +
        SRAM_WORDLINE_DECODER_ENERGY_PJ;

    // Legacy compatibility metrics
    constexpr double ROW_ENERGY_PJ = SRAM_READ_ENERGY_PJ +
                                     128.0 * MULTIPLIER_ENERGY_PJ +
                                     ADDER_TREE_ENERGY_PJ;
    constexpr double MAC_ENERGY_PJ = ROW_ENERGY_PJ / 128.0;

    // Area decomposition (per tile)
    constexpr double MULTIPLIER_AREA_MM2 = 0.015367;
    constexpr double ADDER_TREE_AREA_MM2 = 0.005340;
    constexpr double COMPARATOR_TREE_AREA_MM2 = 0.010680;
    constexpr double PERIPHERAL_AREA_MM2 = 0.031387;
    constexpr double AREA_MM2 = 0.035829;  // SRAM CiM + digital peripherals

    constexpr double LEAKAGE_POWER_MW = 0.326140;

    // Quantizer/Dequantizer energy (FP32↔INT8 conversion at PE-DCIM boundary)
    constexpr double QUANTIZER_ENERGY_PER_ELEMENT_PJ = 0.05;    // FP32→INT8 per element
    constexpr double DEQUANTIZER_ENERGY_PER_ELEMENT_PJ = 0.05;  // INT8→FP32 per element
}

// =============================================================================
// Row Detection Scanner (RDS) — CSR row_ptr comparator at PE/CiM boundary
// =============================================================================
namespace row_detect {
    constexpr double AREA_MM2 = 0.002000;
    constexpr double SRAM_READ_PER_LINE_PJ = sram::index_store::READ_ENERGY_PJ;
    constexpr double COMPARATOR_PER_CHECK_PJ = 0.01;
    constexpr double LEAKAGE_POWER_MW = 0.000100;
}

// =============================================================================
// DRAM Energy (Literature / DRAMsim3)
// =============================================================================
namespace dram {
    // HBM2 estimates (per 64-byte cacheline)
    constexpr double READ_ENERGY_PJ = 3200.0;   // ~50 pJ/bit x 512 bits
    constexpr double WRITE_ENERGY_PJ = 3840.0;  // ~1.2x read
    constexpr double READ_PER_BYTE_PJ = 50.0;
    constexpr double WRITE_PER_BYTE_PJ = 60.0;
}

// =============================================================================
// Aggregate Totals
// =============================================================================
namespace total {
    constexpr double SRAM_AREA_MM2 = sram::f_cache::AREA_MM2 + 
                                     sram::index_store::AREA_MM2 +
                                     sram::indexer_mshr_sram::AREA_MM2 +
                                     sram::indexer_mshr_cam::AREA_MM2 +
                                     sram::pe_registers::AREA_MM2 + 
                                     sram::shared_buffer::AREA_MM2;
    constexpr double SRAM_AREA_WITH_L1_MRU_MM2 = SRAM_AREA_MM2 + sram::l1_mru::AREA_MM2_TOTAL;
    
    constexpr double PE_AREA_MM2 = pe_array::TOTAL_PE_AREA_MM2;
    
    constexpr double NOC_AREA_MM2 = noc::TOTAL_NOC_AREA_MM2;
    
    constexpr double CIM_AREA_MM2 = cim::AREA_MM2 * 16;          // ACIM-like tile area
    constexpr double DCIM_AREA_MM2 = dcim::AREA_MM2 * 16;        // DCIM digital tile area
    constexpr double ROW_DETECT_AREA_MM2 = row_detect::AREA_MM2;
    
    // Backward-compatible default total uses ACIM area model.
    constexpr double TOTAL_AREA_MM2 = SRAM_AREA_MM2 + PE_AREA_MM2 + NOC_AREA_MM2 + CIM_AREA_MM2 + ROW_DETECT_AREA_MM2;
    constexpr double TOTAL_AREA_WITH_L1_MRU_MM2 = SRAM_AREA_WITH_L1_MRU_MM2 + PE_AREA_MM2 + NOC_AREA_MM2 + CIM_AREA_MM2 + ROW_DETECT_AREA_MM2;

    // Explicit per-mode area totals.
    constexpr double TOTAL_AREA_ACIM_MM2 = SRAM_AREA_MM2 + PE_AREA_MM2 + NOC_AREA_MM2 + CIM_AREA_MM2 + ROW_DETECT_AREA_MM2;
    constexpr double TOTAL_AREA_DCIM_MM2 = SRAM_AREA_MM2 + PE_AREA_MM2 + NOC_AREA_MM2 + DCIM_AREA_MM2 + ROW_DETECT_AREA_MM2;
    constexpr double TOTAL_AREA_WITH_L1_MRU_ACIM_MM2 = SRAM_AREA_WITH_L1_MRU_MM2 + PE_AREA_MM2 + NOC_AREA_MM2 + CIM_AREA_MM2 + ROW_DETECT_AREA_MM2;
    constexpr double TOTAL_AREA_WITH_L1_MRU_DCIM_MM2 = SRAM_AREA_WITH_L1_MRU_MM2 + PE_AREA_MM2 + NOC_AREA_MM2 + DCIM_AREA_MM2 + ROW_DETECT_AREA_MM2;
    
    constexpr double SRAM_LEAKAGE_MW = sram::f_cache::LEAKAGE_POWER_MW + 
                                       sram::index_store::LEAKAGE_POWER_MW +
                                       sram::indexer_mshr_sram::LEAKAGE_POWER_MW +
                                       sram::indexer_mshr_cam::LEAKAGE_POWER_MW +
                                       sram::pe_registers::LEAKAGE_POWER_MW + 
                                       sram::shared_buffer::LEAKAGE_POWER_MW;
    constexpr double SRAM_LEAKAGE_WITH_L1_MRU_MW = SRAM_LEAKAGE_MW + sram::l1_mru::LEAKAGE_POWER_MW_TOTAL;
    
    constexpr double PE_LEAKAGE_MW = pe_array::TOTAL_PE_LEAKAGE_MW;
    
    constexpr double NOC_LEAKAGE_MW = noc::ROUTER_LEAKAGE_MW * noc::TOTAL_ROUTERS;
    
    constexpr double CIM_LEAKAGE_MW = cim::LEAKAGE_POWER_MW * 16;
    constexpr double DCIM_LEAKAGE_MW = dcim::LEAKAGE_POWER_MW * 16;
    
    constexpr double TOTAL_LEAKAGE_MW = SRAM_LEAKAGE_MW + PE_LEAKAGE_MW + NOC_LEAKAGE_MW + CIM_LEAKAGE_MW;
    constexpr double TOTAL_LEAKAGE_DCIM_MW = SRAM_LEAKAGE_MW + PE_LEAKAGE_MW + NOC_LEAKAGE_MW + DCIM_LEAKAGE_MW;
}

}  // namespace energy

#endif  // ENERGY_PARAMS_GENERATED_H
