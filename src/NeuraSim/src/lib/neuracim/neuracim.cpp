// SPDX-License-Identifier: Apache-2.0
/**
 * neuracim.cpp
 * 
 * Implementation of NeuraCiM - SRAM Compute-in-Memory module.
 * 5-Stage Pipelined Architecture:
 *   Stage 0: PRECHARGE
 *   Stage 1: COMPUTE
 *   Stage 2: ADC_CONVERT
 *   Stage 3: SHIFT_ADD
 *   Stage 4: OUTPUT_READY
 * 
 * Throughput: 1 row/cycle (steady state)
 * Latency: 5 + (N-1) cycles for N rows
 */

#include "neuracim.h"
#include <cmath>
#include <algorithm>
#include <iostream>
#include <cstdio>
#include <memory>
#include <stdexcept>
#include <string>
#include <array>
#include <sstream>
#include <fstream>
#include <functional>
#include "../energy_params_generated.h"
#include "../pe/resident_weight_contract_r5.h"

#ifdef _WIN32
#include <direct.h>
#define CIM_CREATE_DIR(path) _mkdir((path))
#else
#include <sys/stat.h>
#define CIM_CREATE_DIR(path) mkdir((path), 0777)
#endif

namespace {
constexpr uint64_t kWeightDramBaseAddr = 0x50000000ULL;
constexpr uint64_t kWeightProgramBaseAddr = 0x60000000ULL;
constexpr uint64_t kWeightProgramTileStride = 0x00100000ULL;
constexpr uint64_t kWeightDramTxnBytes = 64ULL;
constexpr int kWeightDramMaxRequestsPerCycle = 8;
constexpr int kWeightDramMshrSize = 64;
constexpr uint64_t kWeightFallbackCyclesPerTxn = 20ULL;
constexpr double kBoltzmannJPerK = 1.380649e-23;
uint64_t fnv1a64_bytes(const void* data, size_t bytes) {
    const uint8_t* p = static_cast<const uint8_t*>(data);
    uint64_t h = 1469598103934665603ULL;
    for (size_t i = 0; i < bytes; ++i) {
        h ^= static_cast<uint64_t>(p[i]);
        h *= 1099511628211ULL;
    }
    return h;
}

bool same_official_dcim_bank_key(const OfficialDcimWeightBankKey& a,
                            const OfficialDcimWeightBankKey& b) {
    return a.layer_id == b.layer_id && a.version == b.version &&
           a.source_hash == b.source_hash && a.rows == b.rows &&
           a.cols == b.cols;
}

void official_dcim_trace_mix(uint64_t& hash, uint64_t value) {
    for (int byte = 0; byte < 8; ++byte) {
        hash ^= (value >> (byte * 8)) & 0xffULL;
        hash *= 1099511628211ULL;
    }
}
}  // namespace

// =============================================================================
// Constructor / Destructor
// =============================================================================

NeuraCiM::NeuraCiM(int d_in, int d_out, int num_tiles, int bits_per_cycle, int fifo_depth)
    : d_in_(d_in)
    , d_out_(d_out)
    , num_tiles_(num_tiles)           // 64 parallel CiM tiles (from config.py)
    , bits_per_cycle_(bits_per_cycle) // 2-bit parallelism (from config.py)
    , fifo_depth_(std::max(1, fifo_depth))
    , weights_loaded_(false)
    , module_state_(NeuraCiMModuleState::IDLE)
    , total_rows_(0)
    , rows_completed_(0)
    , rows_entered_(0)
    , is_incremental_(false)
    , total_col_tiles_(0)
    , compute_cycles_(0)
    , idle_cycles_(0)
    , total_col_tile_passes_(0)
    , col_tile_pipeline_restarts_(0)
    , total_energy_pj_(0.0)
    , array_energy_pj_(0.0)
    , adc_energy_pj_(0.0)
    , shift_add_energy_pj_(0.0)
    , fifo_energy_pj_(0.0)
    , input_dac_energy_pj_(0.0)
    , multiplier_energy_pj_(0.0)
    , adder_tree_energy_pj_(0.0)
    , comparator_energy_pj_(0.0)
    , quantizer_energy_pj_(0.0)
    , dequantizer_energy_pj_(0.0)
    , weight_load_cycles_(0)
    , weight_load_dram_requests_(0)
    , weight_load_bytes_(0)
    , weight_resident_int8_bytes_(0)
    , weight_load_events_(0)
    , weight_program_tile_events_(0)
    , last_weight_program_cycles_(0)
    , last_weight_source_fp32_bytes_(0)
    , last_weight_programmed_int8_bytes_(0)
    , pipeline_counters_()
    , native_trace_stats_()
    , native_cycle_(0)
    , weight_dram_output_dir_("")
    , weight_dram_config_file_("")
#ifdef ENABLE_DRAMSIM3_PE
    , weight_dram_(nullptr)
    , weight_source_pending_txns_(0)
    , weight_dram_current_cycle_(0)
#endif
    , compute_mode_(CiMComputeMode::IDEAL_FP32)
    , variation_seed_(42)
    , variation_initialized_(false)
    , adc_bits_(8)
    , fp_activations_(false)
    , enable_pv_(true)
    , enable_thermal_(true)
    , enable_pv_compensation_(true)
    , thermal_sigma_override_mv_(-1.0f)
    , thermal_sigma_scale_(1.0f)
    , temperature_k_(300.0f)
    , thermal_model_("physics_ktc")
    , thermal_bl_cap_ff_(100.0f)
    , thermal_adc_comp_sigma_mv_(0.5f)
    , resolved_thermal_sigma_mv_(static_cast<float>(neurosim::noise::THERMAL_NOISE_MV))
    , read_avg_n_(1)
    , w_scale_(1.0f)
    , activation_scale_(1.0f)
    , activation_scale_ready_(false)
    , activation_scale_manual_(false)
    , activation_scale_mode_(ActivationScaleMode::FIXED)
    , act_calib_rows_(64)
    , act_calib_clip_(1.0f)
    , warmup_maxabs_(0.0f)
    , warmup_target_rows_(0)
    , warmup_rows_seen_(0)
{
    std::cout << "[CiM] Initialized with " << num_tiles_ << " parallel tiles, "
              << bits_per_cycle_ << "-bit parallelism, FIFO depth "
              << fifo_depth_ << std::endl;
    // Model stage timing from DCIM configuration (ADC bits + bit-serial steps).
    refresh_timing_from_dcim_config();
    
    // Calculate number of output column tiles
    total_col_tiles_ = std::max(1, static_cast<int>(std::ceil(static_cast<double>(d_out_) / neurosim::SUBARRAY_COLS)));
    
    // Reserve weight storage
    weights_.resize(d_in_ * d_out_, 0.0f);
    
    // Initialize per-tile pipeline lanes
    pipeline_lanes_.resize(num_tiles_);
    for (auto& lane : pipeline_lanes_) {
        for (auto& slot : lane) {
            slot.reset();
        }
    }
}

NeuraCiM::~NeuraCiM() {
#ifdef ENABLE_DRAMSIM3_PE
    if (weight_dram_ != nullptr) {
        delete weight_dram_;
        weight_dram_ = nullptr;
    }
#endif
}

// =============================================================================
// Weight Management
// =============================================================================

void NeuraCiM::configure_weight_dram_output_dir(const std::string& output_dir) {
    weight_dram_output_dir_ = output_dir;
}

void NeuraCiM::configure_weight_dram_config_file(const std::string& config_file) {
    weight_dram_config_file_ = config_file;
}

#ifdef ENABLE_DRAMSIM3_PE
void NeuraCiM::ensure_weight_dram_enabled() {
    if (weight_dram_ != nullptr) {
        return;
    }

    std::string cfg = weight_dram_config_file_;
    if (cfg.empty()) {
        cfg = "./ktb_dram_configs/ktb_HBM2_8Gb_x128.ini";
    }
    if (weight_dram_output_dir_.empty()) {
        weight_dram_output_dir_ = "./dramsim3_cim_weight_output";
    }
    CIM_CREATE_DIR(weight_dram_output_dir_.c_str());

    weight_dram_ = dramsim3::GetMemorySystem(
        cfg,
        weight_dram_output_dir_,
        std::bind(&NeuraCiM::weight_dram_read_cb, this, std::placeholders::_1),
        std::bind(&NeuraCiM::weight_dram_write_cb, this, std::placeholders::_1));
}

void NeuraCiM::weight_dram_read_cb(uint64_t addr) {
    std::unordered_map<uint64_t, int>::iterator it = weight_dram_inflight_.find(addr);
    if (it == weight_dram_inflight_.end()) {
        return;
    }
    it->second -= 1;
    if (it->second <= 0) {
        weight_dram_inflight_.erase(it);
    }
    const auto owner = weight_dram_addr_tile_.find(addr);
    if (owner != weight_dram_addr_tile_.end()) {
        const int tile = owner->second;
        if (tile < 0) {
            if (weight_source_pending_txns_ > 0) --weight_source_pending_txns_;
        } else if (tile < static_cast<int>(weight_tile_pending_txns_.size()) &&
                   weight_tile_pending_txns_[static_cast<size_t>(tile)] > 0) {
            uint64_t& pending = weight_tile_pending_txns_[static_cast<size_t>(tile)];
            --pending;
            if (pending == 0) {
                last_weight_tile_program_done_cycles_[static_cast<size_t>(tile)] =
                    weight_dram_current_cycle_;
            }
        }
    }
}

void NeuraCiM::weight_dram_write_cb(uint64_t addr) {
    weight_dram_read_cb(addr);
}
#endif

void NeuraCiM::load_weights(const std::vector<float>& W) {
    // Invalidate readiness before inspecting or replacing any part of W.  A
    // failed cold load must never leave the previous layer dispatchable.
    weights_loaded_ = false;
    resident_weight_loaded_per_tile_.assign(
        static_cast<size_t>(num_tiles_), static_cast<uint8_t>(0));
    if (W.size() != static_cast<size_t>(d_in_ * d_out_)) {
        throw std::runtime_error("NeuraCiM::load_weights resident shape mismatch");
    }

    // Source transfer is FP32 in this implementation; the physical resident
    // image produced below is INT8.  Keep both byte domains explicit.
    const uint64_t bytes = static_cast<uint64_t>(W.size()) * sizeof(float);
    const uint64_t resident_int8_bytes_per_tile =
        static_cast<uint64_t>(W.size()) * sizeof(int8_t);
    const uint64_t programmed_int8_bytes =
        despoina_r5::replicated_int8_program_bytes(
            W.size(), static_cast<size_t>(num_tiles_));
    const uint64_t source_txns =
        (bytes + kWeightDramTxnBytes - 1ULL) / kWeightDramTxnBytes;
    const uint64_t program_txns_per_tile =
        (resident_int8_bytes_per_tile + kWeightDramTxnBytes - 1ULL) /
        kWeightDramTxnBytes;
    const uint64_t program_txns =
        program_txns_per_tile * static_cast<uint64_t>(num_tiles_);
    const uint64_t txns = source_txns + program_txns;

    weights_ = W;
    if (compute_mode_ == CiMComputeMode::ACIM_FAITHFUL ||
        compute_mode_ == CiMComputeMode::DCIM_DIGITAL) {
        quantize_weights_for_dcim();
        if (compute_mode_ == CiMComputeMode::ACIM_FAITHFUL &&
            enable_pv_ && !variation_initialized_) {
            initialize_process_variation();
        }
    }
    if (compute_mode_ == CiMComputeMode::DCIM_DIGITAL &&
        weights_quantized_.size() != W.size()) {
        throw std::runtime_error("R5 INT8 resident image generation failed");
    }
    const uint64_t resident_hash = weights_quantized_.empty()
        ? 0 : fnv1a64_bytes(weights_quantized_.data(), weights_quantized_.size());
    if (compute_mode_ == CiMComputeMode::DCIM_DIGITAL && resident_hash == 0) {
        throw std::runtime_error("R5 resident INT8 hash failed");
    }
    resident_weight_hash_per_tile_.assign(
        static_cast<size_t>(num_tiles_), resident_hash);
    last_weight_program_cycles_ = 0;
    last_weight_source_fp32_bytes_ = bytes;
    last_weight_programmed_int8_bytes_ = programmed_int8_bytes;
    last_weight_tile_program_done_cycles_.assign(
        static_cast<size_t>(num_tiles_), 0);

    // Despoina official DCIM read/compute sensitivity. NeuroSim sequential-SRAM program
    // aggregation is unsupported: retain one layer-entry event, replicated
    // resident bytes and readiness, but exclude invented program time/energy.
    if (compute_mode_ == CiMComputeMode::DCIM_DIGITAL) {
        weight_load_bytes_ += bytes;
        weight_resident_int8_bytes_ += programmed_int8_bytes;
        weight_load_events_ += 1;
        weight_program_tile_events_ += static_cast<uint64_t>(num_tiles_);
        resident_weight_loaded_per_tile_.assign(
            static_cast<size_t>(num_tiles_), static_cast<uint8_t>(1));
        weights_loaded_ = all_resident_tiles_ready();
        if (!weights_loaded_) {
            throw std::runtime_error("Despoina official DCIM resident W readiness failed");
        }
        return;
    }

#ifdef ENABLE_DRAMSIM3_PE
    ensure_weight_dram_enabled();
    if (weight_dram_ != nullptr && txns > 0) {
        std::vector<uint64_t> cachelines;
        cachelines.reserve(static_cast<size_t>(txns));
        weight_dram_addr_tile_.clear();
        weight_tile_pending_txns_.assign(
            static_cast<size_t>(num_tiles_), program_txns_per_tile);
        weight_source_pending_txns_ = source_txns;
        for (uint64_t i = 0; i < source_txns; i++) {
            cachelines.push_back(kWeightDramBaseAddr + i * kWeightDramTxnBytes);
            weight_dram_addr_tile_[cachelines.back()] = -1;
        }
        for (int tile = 0; tile < num_tiles_; ++tile) {
            for (uint64_t i = 0; i < program_txns_per_tile; ++i) {
                const uint64_t addr = kWeightProgramBaseAddr +
                    static_cast<uint64_t>(tile) * kWeightProgramTileStride +
                    i * kWeightDramTxnBytes;
                cachelines.push_back(addr);
                weight_dram_addr_tile_[addr] = tile;
            }
        }

        uint64_t cycles = 0;
        size_t next_to_issue = 0;
        weight_dram_inflight_.clear();

        while (next_to_issue < cachelines.size() || !weight_dram_inflight_.empty()) {
            weight_dram_current_cycle_ = cycles + 1;
            weight_dram_->ClockTick();
            cycles++;

            int issued_this_cycle = 0;
            while (next_to_issue < cachelines.size() &&
                   issued_this_cycle < kWeightDramMaxRequestsPerCycle) {
                if (weight_dram_inflight_.size() >= static_cast<size_t>(kWeightDramMshrSize)) {
                    break;
                }

                const uint64_t addr = cachelines[next_to_issue];
                if (!weight_dram_->WillAcceptTransaction(addr, false)) {
                    break;
                }

                weight_dram_->AddTransaction(addr, false);
                weight_dram_inflight_[addr] += 1;
                issued_this_cycle++;
                next_to_issue++;
            }
        }
        if (std::any_of(last_weight_tile_program_done_cycles_.begin(),
                        last_weight_tile_program_done_cycles_.end(),
                        [](uint64_t v) { return v == 0; })) {
            throw std::runtime_error("R5 per-tile W programming did not complete");
        }

        weight_load_cycles_ += cycles;
        weight_load_dram_requests_ += txns;
        weight_load_bytes_ += bytes;
        last_weight_program_cycles_ = cycles;

        // Emit dramsim3.{txt,json} for energy scripts and audits.
        weight_dram_->PrintStats();
    } else {
        const uint64_t cycles = txns * kWeightFallbackCyclesPerTxn;
        weight_load_cycles_ += cycles;
        weight_load_dram_requests_ += txns;
        weight_load_bytes_ += bytes;
        last_weight_program_cycles_ = cycles;
        uint64_t done = source_txns * kWeightFallbackCyclesPerTxn;
        for (int tile = 0; tile < num_tiles_; ++tile) {
            done += program_txns_per_tile * kWeightFallbackCyclesPerTxn;
            last_weight_tile_program_done_cycles_[static_cast<size_t>(tile)] = done;
        }
    }
#else
    const uint64_t cycles = txns * kWeightFallbackCyclesPerTxn;
    weight_load_cycles_ += cycles;
    weight_load_dram_requests_ += txns;
    weight_load_bytes_ += bytes;
    last_weight_program_cycles_ = cycles;
    uint64_t done = source_txns * kWeightFallbackCyclesPerTxn;
    for (int tile = 0; tile < num_tiles_; ++tile) {
        done += program_txns_per_tile * kWeightFallbackCyclesPerTxn;
        last_weight_tile_program_done_cycles_[static_cast<size_t>(tile)] = done;
    }
#endif

    weight_resident_int8_bytes_ += programmed_int8_bytes;
    weight_load_events_ += 1;
    weight_program_tile_events_ += static_cast<uint64_t>(num_tiles_);

    // Publish per-tile readiness only after every causal program completion.
    // start_matmul independently rechecks the complete vector and hashes.
    if (compute_mode_ == CiMComputeMode::DCIM_DIGITAL) {
        resident_weight_loaded_per_tile_.assign(
            static_cast<size_t>(num_tiles_), static_cast<uint8_t>(1));
        weights_loaded_ = all_resident_tiles_ready();
        if (!weights_loaded_) {
            throw std::runtime_error("R5.1 resident replicas are not program-ready");
        }
    } else {
        weights_loaded_ = true;
    }
}

NeuraCiM::OfficialDcimWeightBankImage NeuraCiM::build_official_dcim_weight_bank(
    const std::vector<float>& W,
    const OfficialDcimWeightBankKey& key,
    uint64_t program_cycles) const {
    if (compute_mode_ != CiMComputeMode::DCIM_DIGITAL ||
        key.layer_id < 1 || key.version == 0 || key.source_hash == 0 ||
        key.rows != d_in_ || key.cols != d_out_ ||
        W.size() != static_cast<size_t>(d_in_) * static_cast<size_t>(d_out_) ||
        program_cycles == 0) {
        throw std::runtime_error("invalid Despoina official DCIM W-bank prefetch request");
    }
    OfficialDcimWeightBankImage image;
    image.valid = true;
    image.key = key;
    image.fp32 = W;
    image.int8.resize(W.size());
    float w_max = 0.0f;
    for (float value : W) w_max = std::max(w_max, std::abs(value));
    image.scale = w_max > 1e-10f ? w_max / 127.0f : 1.0f;
    for (size_t i = 0; i < W.size(); ++i) {
        float q = std::round(W[i] / image.scale);
        q = std::max(-128.0f, std::min(127.0f, q));
        image.int8[i] = static_cast<int8_t>(q);
    }
    image.sum_w.assign(static_cast<size_t>(d_out_), 0.0f);
    image.sum_abs_w.assign(static_cast<size_t>(d_out_), 0.0f);
    for (int i = 0; i < d_in_; ++i) {
        for (int j = 0; j < d_out_; ++j) {
            const float q = static_cast<float>(
                image.int8[static_cast<size_t>(i) * d_out_ + j]);
            image.sum_w[static_cast<size_t>(j)] += q;
            image.sum_abs_w[static_cast<size_t>(j)] += std::abs(q);
        }
    }
    image.quantized_hash = fnv1a64_bytes(image.int8.data(), image.int8.size());
    image.program_cycles = program_cycles;
    if (image.quantized_hash == 0) {
        throw std::runtime_error("Despoina official DCIM W-bank quantized hash is zero");
    }
    return image;
}

void NeuraCiM::official_dcim_bind_active_bank(const OfficialDcimWeightBankKey& key) {
    if (compute_mode_ != CiMComputeMode::DCIM_DIGITAL ||
        weights_quantized_.size() != static_cast<size_t>(d_in_) * d_out_) {
        throw std::runtime_error("cannot bind invalid active Despoina official DCIM W bank");
    }
    OfficialDcimWeightBankImage& bank = official_dcim_weight_banks_[official_dcim_active_bank_];
    bank.valid = true;
    bank.key = key;
    bank.fp32 = weights_;
    bank.int8 = weights_quantized_;
    bank.sum_w = sum_w_q_per_col_;
    bank.sum_abs_w = sum_abs_w_q_per_col_;
    bank.scale = w_scale_;
    bank.quantized_hash = fnv1a64_bytes(bank.int8.data(), bank.int8.size());
    bank.program_cycles = 0;
}

bool NeuraCiM::official_dcim_active_bank_matches(const OfficialDcimWeightBankKey& key) const {
    const OfficialDcimWeightBankImage& bank = official_dcim_weight_banks_[official_dcim_active_bank_];
    return bank.valid && same_official_dcim_bank_key(bank.key, key);
}

bool NeuraCiM::official_dcim_inactive_bank_matches(const OfficialDcimWeightBankKey& key) const {
    const OfficialDcimWeightBankImage& bank = official_dcim_weight_banks_[1 - official_dcim_active_bank_];
    return bank.valid && same_official_dcim_bank_key(bank.key, key);
}

bool NeuraCiM::official_dcim_prefetch_inactive_bank(
    const std::vector<float>& W,
    const OfficialDcimWeightBankKey& key,
    uint64_t program_cycles) {
    if (official_dcim_active_bank_matches(key) || official_dcim_inactive_bank_matches(key)) {
        return false;
    }
    const int inactive = 1 - official_dcim_active_bank_;
    official_dcim_weight_banks_[inactive] =
        build_official_dcim_weight_bank(W, key, program_cycles);
    ++official_dcim_bank_prefetch_events_;
    official_dcim_bank_program_tile_events_ += static_cast<uint64_t>(num_tiles_);
    official_dcim_bank_program_bytes_ += static_cast<uint64_t>(W.size()) *
        sizeof(int8_t) * static_cast<uint64_t>(num_tiles_);
    std::cout << "[DESPOINA_DCIM_BANK_PREFETCH_JSON] {\"target_bank\":" << inactive
              << ",\"layer\":" << key.layer_id
              << ",\"version\":" << key.version
              << ",\"source_hash\":" << key.source_hash
              << ",\"program_cycles\":" << program_cycles
              << ",\"tile_events\":" << num_tiles_
              << ",\"program_bytes\":"
              << static_cast<uint64_t>(W.size()) * static_cast<uint64_t>(num_tiles_)
              << "}" << std::endl;
    return true;
}

uint64_t NeuraCiM::official_dcim_swap_to_prefetched(
    const OfficialDcimWeightBankKey& key,
    uint64_t overlap_available_cycles) {
    if (official_dcim_active_bank_matches(key)) return 0;
    const int inactive = 1 - official_dcim_active_bank_;
    OfficialDcimWeightBankImage& bank = official_dcim_weight_banks_[inactive];
    if (!bank.valid || !same_official_dcim_bank_key(bank.key, key)) {
        throw std::runtime_error("Despoina official DCIM boundary swap missing requested bank key");
    }
    const uint64_t stall = bank.program_cycles > overlap_available_cycles
        ? bank.program_cycles - overlap_available_cycles : 0;
    weights_ = bank.fp32;
    weights_quantized_ = bank.int8;
    sum_w_q_per_col_ = bank.sum_w;
    sum_abs_w_q_per_col_ = bank.sum_abs_w;
    w_scale_ = bank.scale;
    resident_weight_hash_per_tile_.assign(static_cast<size_t>(num_tiles_),
                                           bank.quantized_hash);
    resident_weight_loaded_per_tile_.assign(static_cast<size_t>(num_tiles_), 1);
    weights_loaded_ = true;
    official_dcim_active_bank_ = inactive;
    ++official_dcim_bank_swap_events_;
    official_dcim_bank_exposed_stall_cycles_ += stall;
    official_dcim_start_stall_remaining_ += stall;
    std::cout << "[DESPOINA_DCIM_BANK_SWAP_JSON] {\"active_bank\":" << official_dcim_active_bank_
              << ",\"layer\":" << key.layer_id
              << ",\"version\":" << key.version
              << ",\"source_hash\":" << key.source_hash
              << ",\"overlap_available_cycles\":" << overlap_available_cycles
              << ",\"exposed_stall_cycles\":" << stall << "}" << std::endl;
    return stall;
}

// =============================================================================
// Simulation Interface
// =============================================================================

bool NeuraCiM::start_matmul(int n, int d_in, int d_out, bool is_incremental) {
    if (!weights_loaded_ || !all_resident_tiles_ready()) {
        std::cerr << "NeuraCiM::start_matmul: all resident tile weights are not program-ready!" << std::endl;
        return false;
    }
    
    if (d_in != d_in_ || d_out != d_out_) {
        std::cerr << "NeuraCiM::start_matmul: dimension mismatch!" << std::endl;
        return false;
    }
    
    // Initialize operation
    total_rows_ = n;
    rows_completed_ = 0;
    rows_entered_ = 0;
    is_incremental_ = is_incremental;
    total_col_tile_passes_ = 0;
    col_tile_pipeline_restarts_ = 0;
    native_cycle_ = 0;
    native_trace_stats_ = NativeTraceStats{};
    official_dcim_servers_.assign(static_cast<size_t>(num_tiles_), OfficialDcimTileServer{});
    native_trace_stats_.row_issue_cycles.assign(static_cast<size_t>(std::max(0, n)), 0);
    native_trace_stats_.row_completion_cycles.assign(static_cast<size_t>(std::max(0, n)), 0);
    
    // Reset module state
    module_state_ = NeuraCiMModuleState::IDLE;
    
    // Clear pipeline
    for (auto& lane : pipeline_lanes_) {
        for (auto& slot : lane) {
            slot.reset();
        }
    }
    
    // Clear FIFOs
    while (!input_fifo_.empty()) input_fifo_.pop();
    while (!output_fifo_.empty()) output_fifo_.pop();

    // Hardware-like activation calibration: scale is fixed for the whole operation.
    if (!fp_activations_ && !activation_scale_manual_) {
        activation_scale_ = 1.0f;
        activation_scale_ready_ = false;
    }
    warmup_maxabs_ = 0.0f;
    warmup_rows_seen_ = 0;
    warmup_target_rows_ = warmup_calibration_active()
        ? std::max(1, std::min(act_calib_rows_, total_rows_))
        : 0;
    warmup_input_queue_.clear();
    last_quantized_inputs_.assign(
        static_cast<size_t>(std::max(0, total_rows_)) * static_cast<size_t>(d_in_), 0);
    last_input_scales_.assign(static_cast<size_t>(std::max(0, total_rows_)), 1.0f);
    last_input_trace_valid_.assign(static_cast<size_t>(std::max(0, total_rows_)), 0);
    resolved_thermal_sigma_mv_ = resolve_thermal_sigma_mv();
    return true;
}

bool NeuraCiM::all_resident_tiles_ready() const {
    if (compute_mode_ != CiMComputeMode::DCIM_DIGITAL) {
        return weights_loaded_;
    }
    const uint64_t resident_hash = weights_quantized_.empty()
        ? 0 : fnv1a64_bytes(weights_quantized_.data(), weights_quantized_.size());
    if (resident_hash == 0 ||
        resident_weight_loaded_per_tile_.size() != static_cast<size_t>(num_tiles_) ||
        resident_weight_hash_per_tile_.size() != static_cast<size_t>(num_tiles_)) {
        return false;
    }
    for (int tile = 0; tile < num_tiles_; ++tile) {
        if (resident_weight_loaded_per_tile_[static_cast<size_t>(tile)] != 1 ||
            resident_weight_hash_per_tile_[static_cast<size_t>(tile)] != resident_hash) {
            return false;
        }
    }
    return true;
}

bool NeuraCiM::push_input(const CiMInputPacket& packet) {
    if (warmup_calibration_active() && !activation_scale_ready_) {
        warmup_input_queue_.push_back(packet);
        for (size_t i = 0; i < packet.features.size(); ++i) {
            warmup_maxabs_ = std::max(warmup_maxabs_, std::abs(packet.features[i]));
        }
        warmup_rows_seen_++;
        try_finalize_warmup_calibration();
        flush_warmup_packets_to_input_fifo();
        return true;
    }

    flush_warmup_packets_to_input_fifo();
    if (input_fifo_.size() >= static_cast<size_t>(fifo_depth_)) {
        pipeline_counters_.input_fifo_full_rejections++;
        return false;  // FIFO full
    }
    
    input_fifo_.push(packet);

    // CACTI-based FIFO write energy (store input row into FIFO)
    // Use PE register file (CACTI) as proxy for small on-chip FIFO storage.
    int in_bytes = d_in_ * static_cast<int>(sizeof(float));
    int in_accesses = std::max(1, (in_bytes + 15) / 16);
    fifo_energy_pj_ += in_accesses * energy::sram::pe_registers::WRITE_ENERGY_PJ;
    update_total_energy();

    return true;
}

bool NeuraCiM::pop_output(CiMOutputPacket& packet) {
    if (output_fifo_.empty()) {
        return false;  // FIFO empty
    }
    
    packet = output_fifo_.front();
    output_fifo_.pop();

    // CACTI-based FIFO read energy (read result row out of FIFO)
    int out_bytes = d_out_ * static_cast<int>(sizeof(float));
    int out_accesses = std::max(1, (out_bytes + 15) / 16);
    fifo_energy_pj_ += out_accesses * energy::sram::pe_registers::READ_ENERGY_PJ;
    update_total_energy();

    return true;
}

bool NeuraCiM::tick() {
    ++native_cycle_;
    flush_warmup_packets_to_input_fifo();
    process_pipeline();
    native_trace_stats_.elapsed_cycles = native_cycle_;
    
    // Return true if still busy
    return !(module_state_ == NeuraCiMModuleState::DONE && 
             output_fifo_.empty() && 
             input_fifo_.empty() &&
             is_pipeline_empty());
}

bool NeuraCiM::is_pipeline_empty() const {
    if (compute_mode_ == CiMComputeMode::DCIM_DIGITAL) {
        return std::none_of(official_dcim_servers_.begin(), official_dcim_servers_.end(),
                            [](const OfficialDcimTileServer& s) { return s.busy; });
    }
    for (const auto& lane : pipeline_lanes_) {
        for (const auto& slot : lane) {
            if (slot.valid) return false;
        }
    }
    return true;
}

// =============================================================================
// Pipeline Processing
// =============================================================================

int NeuraCiM::get_cycles_for_stage(NeuraCiMState stage) const {
    switch (stage) {
        case NeuraCiMState::PRECHARGE:   return cycles_precharge_;
        case NeuraCiMState::COMPUTE:     return cycles_compute_;
        case NeuraCiMState::ADC_CONVERT: return cycles_adc_;
        case NeuraCiMState::SHIFT_ADD:   return cycles_shift_add_;
        case NeuraCiMState::OUTPUT_READY:
            // DCIM: dequantize (INT32→FP32 + scale) + ReLU comparator
            if (compute_mode_ == CiMComputeMode::DCIM_DIGITAL)
                return 1 + neurosim::dcim::CYCLES_DEQUANTIZE;  // 2 cycles
            return 1;  // ReLU only for ACIM/FP32
        default: return 1;
    }
}

bool NeuraCiM::refresh_params_from_neurosim(const std::string& neurosim_exe) {
    // External executable invocation is intentionally absent. Runtime parameters
    // must come from the checked, attributed JSON configuration files.
    (void)neurosim_exe;
    std::cerr << "[NeuraCiM] external parameter refresh is disabled" << std::endl;
    return false;
}

void NeuraCiM::process_pipeline() {
    if (compute_mode_ == CiMComputeMode::DCIM_DIGITAL) {
        process_official_dcim_logical_tiles();
        return;
    }
    // =========================================================================
    // REAL tile-parallel pipeline:
    // - num_tiles_ independent lanes, each lane has 5-stage pipeline
    // - Throughput: up to num_tiles_ rows/cycle (steady state)
    // - No "batch mocking": each row has its own row_id/features/result
    // =========================================================================

    // Phase 1: retire OUTPUT_READY for each lane
    // OUTPUT_READY stage includes ReLU comparator (1 cycle): sign-bit check + conditional zeroing
    // fused at CiM output port — no separate activation phase needed
    for (auto& lane : pipeline_lanes_) {
        PipelineSlot& out = lane[4];
        if (out.valid && out.stage == NeuraCiMState::OUTPUT_READY) {
            int req = get_cycles_for_stage(NeuraCiMState::OUTPUT_READY);
            if (out.cycle_in_stage >= req && output_fifo_.size() < static_cast<size_t>(fifo_depth_)) {
                CiMOutputPacket output;
                output.row_id = out.row_id;
                output.result = out.partial_sum;
                output_fifo_.push(output);

                rows_completed_++;
                native_trace_stats_.row_completion_events++;
                native_trace_stats_.functional_apply_events++;
                if (out.row_id >= 0 &&
                    static_cast<size_t>(out.row_id) <
                        native_trace_stats_.row_completion_cycles.size()) {
                    native_trace_stats_.row_completion_cycles[static_cast<size_t>(out.row_id)] =
                        native_cycle_;
                }
                if (compute_mode_ == CiMComputeMode::DCIM_DIGITAL) {
                    accumulate_energy_dcim_digital(1, d_out_);
                } else {
                    accumulate_energy(1, d_out_);
                }

                // CACTI-based FIFO write energy (store result row)
                // Use PE register file as proxy for small on-chip FIFO storage.
                // Result row bytes = d_out_ * 4. CACTI line = 16B → accesses = ceil(bytes/16).
                int out_bytes = d_out_ * static_cast<int>(sizeof(float));
                int out_accesses = std::max(1, (out_bytes + 15) / 16);
                fifo_energy_pj_ += out_accesses * energy::sram::pe_registers::WRITE_ENERGY_PJ;

                out.reset();
            } else if (out.cycle_in_stage >= req) {
                pipeline_counters_.output_fifo_full_stall_cycles++;
            }
        }
    }

    // Phase 2: advance per-lane pipeline (back to front)
    for (auto& lane : pipeline_lanes_) {
        // Stage 3 retirement: either advance to next output tile or emit final output.
        if (lane[3].valid) {
            int req = get_cycles_for_stage(lane[3].stage);
            if (lane[3].cycle_in_stage >= req) {
                if (compute_mode_ == CiMComputeMode::ACIM_FAITHFUL) {
                    compute_row_matmul_dcim(lane[3]);
                } else if (compute_mode_ == CiMComputeMode::DCIM_DIGITAL) {
                    compute_row_matmul_dcim_digital(lane[3]);
                } else {
                    compute_row_matmul(lane[3]);
                }
                total_col_tile_passes_++;
                if (lane[3].col_tile < total_col_tiles_ - 1) {
                    lane[3].col_tile++;
                    lane[3].stage = NeuraCiMState::PRECHARGE;
                    lane[3].cycle_in_stage = 0;
                    col_tile_pipeline_restarts_++;
                } else if (!lane[4].valid) {
                    lane[4] = lane[3];
                    lane[4].stage = NeuraCiMState::OUTPUT_READY;
                    lane[4].cycle_in_stage = 0;
                    lane[3].reset();
                } else {
                    pipeline_counters_.stage3_output_block_cycles++;
                }
            }
        }

        // Stage 2 -> Stage 3
        if (lane[2].valid && !lane[3].valid) {
            int req = get_cycles_for_stage(lane[2].stage);
            if (lane[2].cycle_in_stage >= req) {
                lane[3] = lane[2];
                lane[3].stage = NeuraCiMState::SHIFT_ADD;
                lane[3].cycle_in_stage = 0;
                lane[2].reset();
            }
        }

        // Stage 1 -> Stage 2 (or bypass ADC for DCIM_DIGITAL)
        if (lane[1].valid && !lane[2].valid) {
            int req = get_cycles_for_stage(lane[1].stage);
            if (lane[1].cycle_in_stage >= req) {
                if (cycles_adc_ <= 0 && !lane[3].valid) {
                    // Bypass ADC: COMPUTE → SHIFT_ADD directly
                    lane[3] = lane[1];
                    lane[3].stage = NeuraCiMState::SHIFT_ADD;
                    lane[3].cycle_in_stage = 0;
                    lane[1].reset();
                } else {
                    // Normal path: COMPUTE → ADC_CONVERT
                    lane[2] = lane[1];
                    lane[2].stage = NeuraCiMState::ADC_CONVERT;
                    lane[2].cycle_in_stage = 0;
                    lane[1].reset();
                }
            }
        }

        // Stage 0 -> Stage 1
        if (lane[0].valid && !lane[1].valid) {
            int req = get_cycles_for_stage(lane[0].stage);
            if (lane[0].cycle_in_stage >= req) {
                lane[1] = lane[0];
                lane[1].stage = NeuraCiMState::COMPUTE;
                lane[1].cycle_in_stage = 0;
                lane[0].reset();
            }
        }
    }

    // Phase 3: inject new rows into empty lanes (one row per lane per cycle)
    bool input_waiting_for_inject = !input_fifo_.empty();
    bool any_empty_stage0_lane = false;
    for (const auto& lane : pipeline_lanes_) {
        if (!lane[0].valid) {
            any_empty_stage0_lane = true;
            break;
        }
    }
    if (input_waiting_for_inject && !any_empty_stage0_lane) {
        pipeline_counters_.inject_no_empty_lane_cycles++;
    }
    for (auto& lane : pipeline_lanes_) {
        if (rows_entered_ >= total_rows_) break;
        if (input_fifo_.empty()) break;

        if (!lane[0].valid) {
            CiMInputPacket input = input_fifo_.front();
            input_fifo_.pop();

            lane[0].valid = true;
            lane[0].row_id = input.row_id;
            lane[0].stage = NeuraCiMState::PRECHARGE;
            lane[0].cycle_in_stage = 0;
            lane[0].col_tile = 0;
            lane[0].features = input.features;
            lane[0].partial_sum.resize(d_out_, 0.0f);

            rows_entered_++;
            native_trace_stats_.row_issue_events++;
            if (input.row_id >= 0 &&
                static_cast<size_t>(input.row_id) <
                    native_trace_stats_.row_issue_cycles.size()) {
                native_trace_stats_.row_issue_cycles[static_cast<size_t>(input.row_id)] =
                    native_cycle_;
            }
            module_state_ = NeuraCiMModuleState::RUNNING;

            // CACTI-based FIFO read energy (consume input row from FIFO)
            int in_bytes = d_in_ * static_cast<int>(sizeof(float));
            int in_accesses = std::max(1, (in_bytes + 15) / 16);
            fifo_energy_pj_ += in_accesses * energy::sram::pe_registers::READ_ENERGY_PJ;
        }
    }

    // Phase 4: increment cycle counters for all active slots (all lanes)
    bool any_active = false;
    int active_lanes = 0;
    int active_slots = 0;
    for (auto& lane : pipeline_lanes_) {
        bool lane_active = false;
        for (auto& slot : lane) {
            if (slot.valid) {
                slot.cycle_in_stage++;
                any_active = true;
                lane_active = true;
                active_slots++;
            }
        }
        if (lane_active) {
            active_lanes++;
        }
    }

    pipeline_counters_.active_lane_cycles += static_cast<uint64_t>(active_lanes);
    pipeline_counters_.active_slot_cycles += static_cast<uint64_t>(active_slots);
    pipeline_counters_.max_active_lanes =
        std::max<uint64_t>(pipeline_counters_.max_active_lanes,
                           static_cast<uint64_t>(active_lanes));
    pipeline_counters_.input_fifo_occupancy_cycles += input_fifo_.size();
    pipeline_counters_.output_fifo_occupancy_cycles += output_fifo_.size();
    native_trace_stats_.max_input_fifo_rows = std::max<uint64_t>(
        native_trace_stats_.max_input_fifo_rows, input_fifo_.size());
    native_trace_stats_.max_output_fifo_rows = std::max<uint64_t>(
        native_trace_stats_.max_output_fifo_rows, output_fifo_.size());
    if (input_fifo_.size() >= static_cast<size_t>(fifo_depth_)) {
        pipeline_counters_.input_fifo_full_cycles++;
    }
    if (output_fifo_.size() >= static_cast<size_t>(fifo_depth_)) {
        pipeline_counters_.output_fifo_full_cycles++;
    }

    if (any_active) {
        compute_cycles_++;
    } else if (module_state_ != NeuraCiMModuleState::DONE) {
        idle_cycles_++;
    }

    // Completion check
    if (total_rows_ > 0 &&
        rows_completed_ >= total_rows_ &&
        input_fifo_.empty() &&
        is_pipeline_empty()) {
        module_state_ = NeuraCiMModuleState::DONE;
    }
}

void NeuraCiM::process_official_dcim_logical_tiles() {
    constexpr uint64_t kCyclesPerBitStep = 128ULL;
    uint64_t active = 0;

    if (official_dcim_start_stall_remaining_ > 0) {
        --official_dcim_start_stall_remaining_;
        ++idle_cycles_;
        if (input_fifo_.size() >= static_cast<size_t>(fifo_depth_)) {
            ++pipeline_counters_.input_fifo_full_cycles;
        }
        return;
    }

    // Advance one real temporal input-bit event only at its 128-cycle boundary.
    // The signed b7 significance is accumulated separately; no atomic full-row
    // compute is deferred to cycle 1024.
    for (size_t tile = 0; tile < official_dcim_servers_.size(); ++tile) {
        OfficialDcimTileServer& server = official_dcim_servers_[tile];
        if (!server.busy) continue;
        ++active;
        if (server.compute_done || server.next_bit_completion_cycle > native_cycle_) continue;
        if (server.next_bit_completion_cycle != native_cycle_) {
            ++native_trace_stats_.bit_step_temporal_order_violations;
            continue;
        }
        if (server.bit_step < 0 || server.bit_step >= 8 ||
            server.quantized_input.size() != static_cast<size_t>(d_in_) ||
            server.accumulator.size() != static_cast<size_t>(d_out_)) {
            throw std::runtime_error("Despoina official DCIM temporal bit-step server state corrupt");
        }
        const int bit = server.bit_step;
        const int32_t significance = bit == 7 ? -128 : (1 << bit);
        for (int j = 0; j < d_out_; ++j) {
            int32_t partial = 0;
            for (int i = 0; i < d_in_; ++i) {
                const uint8_t ux = static_cast<uint8_t>(server.quantized_input[static_cast<size_t>(i)]);
                if (((ux >> bit) & 1U) != 0U) {
                    partial += static_cast<int32_t>(
                        weights_quantized_[static_cast<size_t>(i) * d_out_ + j]);
                }
            }
            server.accumulator[static_cast<size_t>(j)] += significance * partial;
        }
        ++native_trace_stats_.bit_step_completion_events;
        native_trace_stats_.last_bit_step_completion_cycle = native_cycle_;
        official_dcim_trace_mix(native_trace_stats_.bit_step_trace_fnv1a64,
                           static_cast<uint64_t>(server.row_id));
        official_dcim_trace_mix(native_trace_stats_.bit_step_trace_fnv1a64,
                           static_cast<uint64_t>(tile));
        official_dcim_trace_mix(native_trace_stats_.bit_step_trace_fnv1a64,
                           static_cast<uint64_t>(bit));
        official_dcim_trace_mix(native_trace_stats_.bit_step_trace_fnv1a64,
                           server.next_bit_completion_cycle - kCyclesPerBitStep);
        official_dcim_trace_mix(native_trace_stats_.bit_step_trace_fnv1a64,
                           native_cycle_);
        if (bit == 7) {
            // Charge the calibrated finalized-row read/compute event exactly
            // once at b7 completion. The eight temporal bit events remain
            // separately counted timing/work evidence.
            accumulate_energy_dcim_digital(1, d_out_);
            ++native_trace_stats_.full_int8_energy_charge_events;
            server.compute_done = true;
            server.completion_cycle = native_cycle_;
            server.result.resize(static_cast<size_t>(d_out_));
            for (int j = 0; j < d_out_; ++j) {
                server.result[static_cast<size_t>(j)] =
                    static_cast<float>(server.accumulator[static_cast<size_t>(j)]) *
                    server.input_scale * w_scale_;
            }
        } else {
            ++server.bit_step;
            server.next_bit_completion_cycle += kCyclesPerBitStep;
            ++native_trace_stats_.bit_step_issue_events;
        }
    }

    // Completion is owned only by the issuing logical tile. Output credit is
    // finite; a full output FIFO stalls retirement without duplicating a row.
    for (auto& server : official_dcim_servers_) {
        if (!server.busy || !server.compute_done) continue;
        if (output_fifo_.size() >= static_cast<size_t>(fifo_depth_)) {
            pipeline_counters_.output_fifo_full_stall_cycles++;
            continue;
        }
        CiMOutputPacket out;
        out.row_id = server.row_id;
        out.result.swap(server.result);
        output_fifo_.push(out);
        rows_completed_++;
        native_trace_stats_.row_completion_events++;
        native_trace_stats_.functional_apply_events++;
        if (server.row_id >= 0 &&
            static_cast<size_t>(server.row_id) < native_trace_stats_.row_completion_cycles.size()) {
            native_trace_stats_.row_completion_cycles[static_cast<size_t>(server.row_id)] = native_cycle_;
        }
        server = OfficialDcimTileServer{};
    }

    // Dispatch consumes FIFO16 credit only while a distinct logical tile is
    // free. A server cannot accept a second row before its 1024-cycle result.
    for (auto& server : official_dcim_servers_) {
        if (server.busy || input_fifo_.empty() || rows_entered_ >= total_rows_) continue;
        CiMInputPacket in = input_fifo_.front();
        input_fifo_.pop();
        server.busy = true;
        server.compute_done = false;
        server.row_id = in.row_id;
        server.bit_step = 0;
        server.issue_cycle = native_cycle_;
        server.completion_cycle = native_cycle_ + 8ULL * kCyclesPerBitStep;
        server.next_bit_completion_cycle = native_cycle_ + kCyclesPerBitStep;
        float max_abs = 0.0f;
        for (float value : in.features) max_abs = std::max(max_abs, std::abs(value));
        server.input_scale = max_abs > 1e-10f ? max_abs / 127.0f : 1.0f;
        server.quantized_input.assign(static_cast<size_t>(d_in_), 0);
        for (int i = 0; i < d_in_ && i < static_cast<int>(in.features.size()); ++i) {
            float q = std::round(in.features[static_cast<size_t>(i)] / server.input_scale);
            q = std::max(-128.0f, std::min(127.0f, q));
            server.quantized_input[static_cast<size_t>(i)] = static_cast<int8_t>(q);
        }
        server.accumulator.assign(static_cast<size_t>(d_out_), 0);
        if (server.row_id >= 0 && server.row_id < total_rows_ &&
            last_quantized_inputs_.size() ==
                static_cast<size_t>(total_rows_) * static_cast<size_t>(d_in_)) {
            const size_t row = static_cast<size_t>(server.row_id);
            std::copy(server.quantized_input.begin(), server.quantized_input.end(),
                      last_quantized_inputs_.begin() + row * static_cast<size_t>(d_in_));
            last_input_scales_[row] = server.input_scale;
            last_input_trace_valid_[row] = 1;
        }
        rows_entered_++;
        native_trace_stats_.row_issue_events++;
        native_trace_stats_.bit_step_issue_events++;
        if (native_trace_stats_.first_bit_step_issue_cycle == 0) {
            native_trace_stats_.first_bit_step_issue_cycle = native_cycle_;
        }
        if (server.row_id >= 0 &&
            static_cast<size_t>(server.row_id) < native_trace_stats_.row_issue_cycles.size()) {
            native_trace_stats_.row_issue_cycles[static_cast<size_t>(server.row_id)] = native_cycle_;
        }
        module_state_ = NeuraCiMModuleState::RUNNING;
    }

    active = 0;
    for (const auto& server : official_dcim_servers_) if (server.busy) ++active;
    pipeline_counters_.active_lane_cycles += active;
    pipeline_counters_.active_slot_cycles += active;
    pipeline_counters_.max_active_lanes = std::max<uint64_t>(pipeline_counters_.max_active_lanes, active);
    pipeline_counters_.input_fifo_occupancy_cycles += input_fifo_.size();
    pipeline_counters_.output_fifo_occupancy_cycles += output_fifo_.size();
    native_trace_stats_.max_input_fifo_rows = std::max<uint64_t>(native_trace_stats_.max_input_fifo_rows, input_fifo_.size());
    native_trace_stats_.max_output_fifo_rows = std::max<uint64_t>(native_trace_stats_.max_output_fifo_rows, output_fifo_.size());
    if (input_fifo_.size() >= static_cast<size_t>(fifo_depth_)) ++pipeline_counters_.input_fifo_full_cycles;
    if (active > 0) ++compute_cycles_; else ++idle_cycles_;
    if (total_rows_ > 0 && rows_completed_ >= total_rows_ && input_fifo_.empty() && is_pipeline_empty()) {
        module_state_ = NeuraCiMModuleState::DONE;
    }
}

void NeuraCiM::compute_row_matmul(PipelineSlot& slot) {
    // Compute only the active output column tile and accumulate partial sums.
    if (slot.partial_sum.size() != static_cast<size_t>(d_out_)) {
        slot.partial_sum.resize(static_cast<size_t>(d_out_), 0.0f);
    }
    const int col_start = slot.col_tile * neurosim::SUBARRAY_COLS;
    const int col_end = std::min((slot.col_tile + 1) * neurosim::SUBARRAY_COLS, d_out_);

    for (int j = col_start; j < col_end; j++) {
        float sum = 0.0f;
        for (int i = 0; i < d_in_; i++) {
            if (i < static_cast<int>(slot.features.size())) {
                sum += slot.features[i] * weights_[i * d_out_ + j];
            }
        }
        slot.partial_sum[j] += sum;
    }
}

void NeuraCiM::accumulate_energy(int rows_processed, int cols_processed) {
    const int serial_steps = fp_activations_ ? 1 : get_serial_steps();
    const double array_e =
        static_cast<double>(rows_processed) *
        static_cast<double>(total_col_tiles_) *
        static_cast<double>(serial_steps) *
        neurosim::ARRAY_READ_ENERGY_PJ;
    const double adc_e =
        static_cast<double>(rows_processed) *
        static_cast<double>(cols_processed) *
        static_cast<double>(serial_steps) *
        adc_energy_pj_for_bits(adc_bits_) *
        static_cast<double>(read_avg_n_);
    const double sa_e =
        static_cast<double>(rows_processed) *
        static_cast<double>(cols_processed) *
        static_cast<double>(serial_steps) *
        neurosim::SHIFT_ADD_ENERGY_PJ;
    const double dac_e = fp_activations_ ? 0.0 :
        static_cast<double>(rows_processed) *
        static_cast<double>(total_col_tiles_) *
        static_cast<double>(serial_steps) *
        neurosim::input_dac::ENERGY_PJ;
    
    array_energy_pj_ += array_e;
    adc_energy_pj_ += adc_e;
    shift_add_energy_pj_ += sa_e;
    input_dac_energy_pj_ += dac_e;
    update_total_energy();
}

void NeuraCiM::accumulate_energy_dcim_digital(int rows_processed, int cols_processed) {
    // Energy accounting:
    // - Multiplier: per scalar INT8 multiply -> rows * d_out * d_in
    // - Adder tree: per output-element reduction (128-term reference) -> rows * d_out
    //   scaled linearly by active reduction depth (d_in / 128).
    const double rows_d = static_cast<double>(rows_processed);
    const double cols_d = static_cast<double>(cols_processed);

    const double sram_e =
        rows_d *
        static_cast<double>(total_col_tiles_) *
        energy::dcim::SRAM_READ_ENERGY_PJ;
    const double mult_e =
        rows_d * cols_d *
        static_cast<double>(d_in_) *
        energy::dcim::MULTIPLIER_ENERGY_PJ;
    const double adder_scale = std::max(1.0, static_cast<double>(d_in_) / 128.0);
    const double adder_e =
        rows_d * cols_d * adder_scale *
        energy::dcim::ADDER_TREE_ENERGY_PJ;

    // Activation quantization comparator tree: finds max-abs per row for
    // dynamic INT8 scaling.  One comparator tree pass per row processed.
    const double comparator_e =
        rows_d * energy::dcim::COMPARATOR_TREE_ENERGY_PJ;

    // Activation quantizer array: 128 parallel FP32→INT8 scale-round-clamp
    // units.  One pass per row (d_in elements quantized).
    const double quantizer_e =
        rows_d *
        static_cast<double>(std::min(d_in_, 128)) *
        energy::dcim::QUANTIZER_ENERGY_PER_ELEMENT_PJ;

    // Dequantizer array: 128 parallel INT32→FP32 + scale multiply units.
    // One pass per column tile (cols_processed elements dequantized).
    const double dequantizer_e =
        rows_d * cols_d *
        energy::dcim::DEQUANTIZER_ENERGY_PER_ELEMENT_PJ;

    array_energy_pj_ += sram_e;
    multiplier_energy_pj_ += mult_e;
    adder_tree_energy_pj_ += adder_e;
    comparator_energy_pj_ += comparator_e;
    quantizer_energy_pj_ += quantizer_e;
    dequantizer_energy_pj_ += dequantizer_e;
    // ADC = 0, DAC = 0 for DCIM_DIGITAL
    update_total_energy();
}

// =============================================================================
// Statistics
// =============================================================================

void NeuraCiM::reset_counters() {
    compute_cycles_ = 0;
    idle_cycles_ = 0;
    total_col_tile_passes_ = 0;
    col_tile_pipeline_restarts_ = 0;
    total_energy_pj_ = 0.0;
    array_energy_pj_ = 0.0;
    adc_energy_pj_ = 0.0;
    shift_add_energy_pj_ = 0.0;
    fifo_energy_pj_ = 0.0;
    input_dac_energy_pj_ = 0.0;
    multiplier_energy_pj_ = 0.0;
    adder_tree_energy_pj_ = 0.0;
    comparator_energy_pj_ = 0.0;
    quantizer_energy_pj_ = 0.0;
    dequantizer_energy_pj_ = 0.0;
    pipeline_counters_ = PipelineCounters{};
    native_trace_stats_ = NativeTraceStats{};
    native_cycle_ = 0;
}

void NeuraCiM::set_compute_mode(CiMComputeMode mode) {
    if (compute_mode_ == mode) {
        return;
    }
    if (mode == CiMComputeMode::DCIM_DIGITAL) {
        despoina_official_dcim::require_runtime_contract(
            d_in_, d_out_, num_tiles_, fifo_depth_);
    }
    compute_mode_ = mode;
    if ((compute_mode_ == CiMComputeMode::ACIM_FAITHFUL ||
         compute_mode_ == CiMComputeMode::DCIM_DIGITAL) && weights_loaded_) {
        quantize_weights_for_dcim();
    }
    // Refresh timing — DCIM_DIGITAL sets cycles_adc_=0 for ADC bypass
    refresh_timing_from_dcim_config();
}

void NeuraCiM::set_variation_seed(uint32_t seed) {
    variation_seed_ = seed;
    variation_initialized_ = false;
    if (compute_mode_ == CiMComputeMode::ACIM_FAITHFUL && weights_loaded_ && enable_pv_) {
        initialize_process_variation();
    }
}

void NeuraCiM::set_adc_bits(int bits) {
    adc_bits_ = std::max(1, std::min(16, bits));
    refresh_timing_from_dcim_config();
}

void NeuraCiM::set_fp_activations(bool fp) {
    if (fp_activations_ == fp) {
        return;
    }
    fp_activations_ = fp;
    refresh_timing_from_dcim_config();
}

void NeuraCiM::set_enable_pv(bool en) {
    if (enable_pv_ == en) {
        return;
    }
    enable_pv_ = en;
    if ((compute_mode_ == CiMComputeMode::ACIM_FAITHFUL ||
         compute_mode_ == CiMComputeMode::DCIM_DIGITAL) && weights_loaded_) {
        if (enable_pv_ && !variation_initialized_ &&
            compute_mode_ == CiMComputeMode::ACIM_FAITHFUL) {
            initialize_process_variation();
        } else {
            quantize_weights_for_dcim();
        }
    }
}

void NeuraCiM::set_enable_thermal(bool en) {
    enable_thermal_ = en;
    resolved_thermal_sigma_mv_ = resolve_thermal_sigma_mv();
}

void NeuraCiM::set_enable_pv_compensation(bool en) {
    if (enable_pv_compensation_ == en) {
        return;
    }
    enable_pv_compensation_ = en;
    if ((compute_mode_ == CiMComputeMode::ACIM_FAITHFUL ||
         compute_mode_ == CiMComputeMode::DCIM_DIGITAL) && weights_loaded_) {
        quantize_weights_for_dcim();
    }
}

void NeuraCiM::set_thermal_sigma_override_mv(float sigma_mv) {
    thermal_sigma_override_mv_ = sigma_mv;
    resolved_thermal_sigma_mv_ = resolve_thermal_sigma_mv();
}

void NeuraCiM::set_thermal_sigma_scale(float scale) {
    thermal_sigma_scale_ = std::max(1e-6f, scale);
    resolved_thermal_sigma_mv_ = resolve_thermal_sigma_mv();
}

void NeuraCiM::set_temperature_k(float temp_k) {
    temperature_k_ = std::max(1e-3f, temp_k);
    resolved_thermal_sigma_mv_ = resolve_thermal_sigma_mv();
}

void NeuraCiM::set_thermal_model(const std::string& model) {
    if (model == "physics_ktc") {
        thermal_model_ = "physics_ktc";
    } else {
        thermal_model_ = "legacy";
    }
    resolved_thermal_sigma_mv_ = resolve_thermal_sigma_mv();
}

void NeuraCiM::set_thermal_bl_cap_ff(float cap_ff) {
    thermal_bl_cap_ff_ = std::max(1e-6f, cap_ff);
    resolved_thermal_sigma_mv_ = resolve_thermal_sigma_mv();
}

void NeuraCiM::set_thermal_adc_comp_sigma_mv(float sigma_mv) {
    thermal_adc_comp_sigma_mv_ = std::max(0.0f, sigma_mv);
    resolved_thermal_sigma_mv_ = resolve_thermal_sigma_mv();
}

void NeuraCiM::set_read_avg_n(int n) {
    read_avg_n_ = std::max(1, std::min(8, n));
}

void NeuraCiM::set_activation_scale_mode(const std::string& mode) {
    if (mode == "warmup_maxabs") {
        activation_scale_mode_ = ActivationScaleMode::WARMUP_MAXABS;
    } else {
        activation_scale_mode_ = ActivationScaleMode::FIXED;
    }
}

std::string NeuraCiM::get_activation_scale_mode_str() const {
    return activation_scale_mode_ == ActivationScaleMode::WARMUP_MAXABS ? "warmup_maxabs" : "fixed";
}

void NeuraCiM::set_act_calib_rows(int rows) {
    act_calib_rows_ = std::max(1, rows);
}

void NeuraCiM::set_act_calib_clip(float clip) {
    act_calib_clip_ = std::max(1e-6f, clip);
}

void NeuraCiM::refresh_timing_from_dcim_config() {
    if (compute_mode_ == CiMComputeMode::DCIM_DIGITAL) {
        // Preserve the original H6 array-wide full-INT8 native pipeline.
        cycles_precharge_ = despoina_official_dcim::Contract::kSramReadCycles;
        cycles_compute_ = despoina_official_dcim::Contract::kFullInt8ComputeCycles;
        cycles_adc_ = despoina_official_dcim::Contract::kAdcCycles;
        cycles_shift_add_ = despoina_official_dcim::Contract::kAccumulateCycles;
        cycles_input_dac_ = 0;
        return;
    }

    // ACIM / IDEAL_FP32 path (unchanged)
    const int base_precharge = std::max(
        1, static_cast<int>(std::ceil(neurosim::PRECHARGE_DELAY_NS / neurosim::CLOCK_PERIOD_NS)));
    const int base_compute = std::max(
        1, static_cast<int>(std::ceil(neurosim::SENSE_DELAY_NS / neurosim::CLOCK_PERIOD_NS)));
    const int base_shift_add = std::max(
        1, static_cast<int>(std::ceil(neurosim::SHIFT_ADD_DELAY_NS / neurosim::CLOCK_PERIOD_NS)));
    const int adc_cycles_per_read = std::max(
        1, static_cast<int>(std::ceil(adc_delay_ns_for_bits(adc_bits_) / neurosim::CLOCK_PERIOD_NS)));
    const int serial_steps = fp_activations_ ? 1 : get_serial_steps();

    cycles_input_dac_ = fp_activations_ ? 0 : std::max(
        1, static_cast<int>(std::ceil(dac_delay_ns_for_bits(8) / neurosim::CLOCK_PERIOD_NS)));
    cycles_precharge_ = std::max(base_precharge, cycles_input_dac_);
    cycles_compute_ = std::max(1, base_compute * serial_steps);
    cycles_adc_ = std::max(1, adc_cycles_per_read * serial_steps);
    cycles_shift_add_ = std::max(1, base_shift_add * serial_steps);
}

float NeuraCiM::resolve_thermal_sigma_mv() const {
    if (!enable_thermal_) {
        return 0.0f;
    }
    const float scale = std::max(1e-6f, thermal_sigma_scale_);
    const float temp_k = std::max(1e-3f, temperature_k_);

    if (thermal_sigma_override_mv_ > 0.0f) {
        return thermal_sigma_override_mv_ * scale * std::sqrt(temp_k / 300.0f);
    }

    if (thermal_model_ == "physics_ktc") {
        const double c_bl_f = std::max(1e-9, static_cast<double>(thermal_bl_cap_ff_));
        const double c_bl = c_bl_f * 1e-15;
        const double sigma_ktc_v = std::sqrt(kBoltzmannJPerK * static_cast<double>(temp_k) / c_bl);
        const double sigma_ktc_mv = sigma_ktc_v * 1e3;
        const double sigma_comp_mv =
            static_cast<double>(std::max(0.0f, thermal_adc_comp_sigma_mv_)) *
            std::sqrt(static_cast<double>(temp_k) / 300.0);
        const double sigma_total_mv = std::hypot(sigma_ktc_mv, sigma_comp_mv);
        return static_cast<float>(sigma_total_mv * static_cast<double>(scale));
    }

    const float base_sigma = static_cast<float>(neurosim::noise::THERMAL_NOISE_MV);
    return base_sigma * scale * std::sqrt(temp_k / 300.0f);
}

bool NeuraCiM::warmup_calibration_active() const {
    return (!fp_activations_) &&
           (!activation_scale_manual_) &&
           (activation_scale_mode_ == ActivationScaleMode::WARMUP_MAXABS);
}

void NeuraCiM::try_finalize_warmup_calibration() {
    if (!warmup_calibration_active() || activation_scale_ready_) {
        return;
    }
    if (warmup_rows_seen_ < warmup_target_rows_) {
        return;
    }
    const float clipped_max = warmup_maxabs_ * std::max(1e-6f, act_calib_clip_);
    activation_scale_ = (clipped_max > 1e-10f) ? (clipped_max / 127.0f) : 1.0f;
    activation_scale_ready_ = true;
}

void NeuraCiM::flush_warmup_packets_to_input_fifo() {
    if (!activation_scale_ready_) {
        return;
    }
    while (!warmup_input_queue_.empty() && input_fifo_.size() < static_cast<size_t>(fifo_depth_)) {
        input_fifo_.push(warmup_input_queue_.front());
        warmup_input_queue_.pop_front();
        // Keep FIFO write energy accounting consistent with direct push_input path.
        int in_bytes = d_in_ * static_cast<int>(sizeof(float));
        int in_accesses = std::max(1, (in_bytes + 15) / 16);
        fifo_energy_pj_ += in_accesses * energy::sram::pe_registers::WRITE_ENERGY_PJ;
        update_total_energy();
    }
}

void NeuraCiM::update_total_energy() {
    if (compute_mode_ == CiMComputeMode::DCIM_DIGITAL) {
        total_energy_pj_ =
            array_energy_pj_ + multiplier_energy_pj_ + adder_tree_energy_pj_ +
            comparator_energy_pj_ + quantizer_energy_pj_ + dequantizer_energy_pj_ +
            fifo_energy_pj_;
    } else {
        total_energy_pj_ =
            array_energy_pj_ + adc_energy_pj_ + shift_add_energy_pj_ + fifo_energy_pj_ + input_dac_energy_pj_;
    }
}

int NeuraCiM::get_serial_steps() const {
    const int bpc = std::max(1, std::min(8, bits_per_cycle_));
    return (8 + bpc - 1) / bpc;
}

double NeuraCiM::adc_delay_ns_for_bits(int bits) const {
    const int b = std::max(1, bits);
    // Anchor points from design-space characterization (22nm SRAM-CiM SAR ADC).
    if (b <= 2) return neurosim::adc_2bit::ADC_DELAY_NS;
    if (b <= 4) return 4.7;
    if (b <= 6) return 6.9;
    if (b <= 8) return neurosim::adc_8bit::ADC_DELAY_NS;
    return neurosim::adc_8bit::ADC_DELAY_NS + 1.1 * static_cast<double>(b - 8);
}

double NeuraCiM::dac_delay_ns_for_bits(int bits) const {
    const int b = std::max(1, bits);
    return neurosim::input_dac::DELAY_8BIT_NS * static_cast<double>(b) / 8.0;
}

double NeuraCiM::adc_energy_pj_for_bits(int bits) const {
    const int b = std::max(1, bits);
    if (b <= 2) return neurosim::adc_2bit::ADC_ENERGY_PJ;
    if (b <= 4) return 0.08;
    if (b <= 6) return 0.12;
    if (b <= 8) return neurosim::adc_8bit::ADC_ENERGY_PJ;
    return neurosim::adc_8bit::ADC_ENERGY_PJ + 0.02 * static_cast<double>(b - 8);
}

// =============================================================================
// DCIM-Faithful Computation
// =============================================================================

void NeuraCiM::initialize_process_variation() {
    cell_vth_offset_mv_.resize(static_cast<size_t>(d_in_) * d_out_);
    std::mt19937 rng(variation_seed_);
    // Vth variation: VTH_VARIATION_PCT of Vdd in millivolts
    float sigma_mv = static_cast<float>(neurosim::VDD_V * 1000.0 *
                     neurosim::noise::VTH_VARIATION_PCT / 100.0);
    std::normal_distribution<float> dist(0.0f, sigma_mv);
    for (size_t i = 0; i < cell_vth_offset_mv_.size(); i++) {
        cell_vth_offset_mv_[i] = dist(rng);
    }
    variation_initialized_ = true;
    std::cout << "[CiM-DCIM] Process variation initialized: sigma=" << sigma_mv
              << " mV, seed=" << variation_seed_
              << ", cells=" << cell_vth_offset_mv_.size() << std::endl;

    // PV table changed -> refresh quantized weights if DCIM mode is active.
    if (compute_mode_ == CiMComputeMode::ACIM_FAITHFUL && weights_loaded_) {
        quantize_weights_for_dcim();
    }
}

void NeuraCiM::quantize_weights_for_dcim() {
    weights_quantized_.resize(static_cast<size_t>(d_in_) * d_out_);
    float w_max = 0.0f;
    for (size_t i = 0; i < weights_.size(); i++) {
        w_max = std::max(w_max, std::abs(weights_[i]));
    }
    w_scale_ = (w_max > 1e-10f) ? (w_max / 127.0f) : 1.0f;
    const float vdd_mv = static_cast<float>(neurosim::VDD_V * 1000.0);
    const bool use_pv_comp =
        enable_pv_ &&
        enable_pv_compensation_ &&
        variation_initialized_ &&
        cell_vth_offset_mv_.size() == weights_quantized_.size();
    for (size_t i = 0; i < weights_.size(); i++) {
        float q_target = weights_[i] / w_scale_;
        if (use_pv_comp) {
            const float vth_off = cell_vth_offset_mv_[i];
            const float pv_factor = std::max(0.5f, 1.0f + vth_off / vdd_mv);
            q_target /= pv_factor;
        }
        float q = std::round(q_target);
        q = std::max(-128.0f, std::min(127.0f, q));
        weights_quantized_[i] = static_cast<int8_t>(q);
    }
    // Precompute per-column sum of quantized weights for unsigned bias correction
    sum_w_q_per_col_.assign(static_cast<size_t>(d_out_), 0.0f);
    for (int i = 0; i < d_in_; i++) {
        for (int j = 0; j < d_out_; j++) {
            const float w_q = static_cast<float>(
                weights_quantized_[static_cast<size_t>(i) * d_out_ + j]);
            sum_w_q_per_col_[j] += w_q;
        }
    }
    sum_abs_w_q_per_col_.assign(static_cast<size_t>(d_out_), 0.0f);
    for (int i = 0; i < d_in_; i++) {
        for (int j = 0; j < d_out_; j++) {
            const float w_q = static_cast<float>(
                weights_quantized_[static_cast<size_t>(i) * d_out_ + j]);
            sum_abs_w_q_per_col_[j] += std::abs(w_q);
        }
    }
    std::cout << "[CiM-DCIM] Weights quantized to INT8: scale=" << w_scale_
              << ", w_max=" << w_max
              << (use_pv_comp ? " (pv-comp:on)" : " (pv-comp:off)")
              << std::endl;
}

float NeuraCiM::adc_quantize(float bl_voltage_mv, int num_levels) const {
    if (num_levels <= 1) {
        return 0.0f;
    }
    // Multi-bit SAR ADC: num_levels uniformly spanning [0, BL_SWING_MV]
    constexpr float bl_swing = static_cast<float>(neurosim::adc_2bit::BL_SWING_MV);
    const float mv_per_level = bl_swing / static_cast<float>(num_levels - 1);

    float clamped = std::max(0.0f, std::min(bl_swing, bl_voltage_mv));
    int level = static_cast<int>(std::round(clamped / mv_per_level));
    level = std::min(level, num_levels - 1);
    return static_cast<float>(level);
}

void NeuraCiM::compute_row_matmul_dcim(PipelineSlot& slot) {
    // =========================================================================
    // DCIM-Faithful Computation (DNN+NeuroSim behavioral model)
    //
    // Noise sources modeled:
    //   1. INT8 quantization of weights (always — SRAM stores INT8)
    //   2. INT8 quantization of activations (optional, fp_activations_ disables)
    //   3. Per-cell process variation (optional, enable_pv_)
    //   4. Thermal read noise per column (optional, enable_thermal_)
    //
    // This follows the DNN+NeuroSim/PuMa/ISAAC convention: direct INT8
    // dot-product with hardware noise injection.
    // =========================================================================

    const size_t expected_w = static_cast<size_t>(d_in_) * static_cast<size_t>(d_out_);
    if (weights_quantized_.size() != expected_w) {
        quantize_weights_for_dcim();
    }
    if (weights_quantized_.size() != expected_w) {
        compute_row_matmul(slot);
        return;
    }
    if (enable_pv_ && cell_vth_offset_mv_.size() != expected_w) {
        initialize_process_variation();
    }

    if (slot.partial_sum.size() != static_cast<size_t>(d_out_)) {
        slot.partial_sum.resize(static_cast<size_t>(d_out_), 0.0f);
    }
    const int col_start = slot.col_tile * neurosim::SUBARRAY_COLS;
    const int col_end = std::min((slot.col_tile + 1) * neurosim::SUBARRAY_COLS, d_out_);
    const int feat_size = static_cast<int>(slot.features.size());

    // ---- Input quantization ----
    float x_scale = 1.0f;
    float x_abs_max = 0.0f;
    std::vector<int> x_q;
    std::vector<int> x_u;

    for (int i = 0; i < d_in_ && i < feat_size; i++) {
        x_abs_max = std::max(x_abs_max, std::abs(slot.features[i]));
    }

    if (!fp_activations_) {
        // Per-row dynamic scale: each row computes its own max-abs for INT8 quantization.
        // Hardware-realistic for modern CiM designs (comparator tree + programmable DAC ref).
        x_scale = (x_abs_max > 1e-10f) ? (x_abs_max / 127.0f) : 1.0f;

        x_q.resize(static_cast<size_t>(d_in_), 0);
        x_u.resize(static_cast<size_t>(d_in_), 0);
        for (int i = 0; i < d_in_ && i < feat_size; i++) {
            float q = std::round(slot.features[i] / x_scale);
            x_q[i] = static_cast<int>(std::max(-128.0f, std::min(127.0f, q)));
            x_u[i] = x_q[i] + 128;
        }
    }

    // ---- Noise RNG ----
    const float vdd_mv = static_cast<float>(neurosim::VDD_V) * 1000.0f;
    std::mt19937 thermal_rng(variation_seed_ ^
        static_cast<uint32_t>(slot.row_id * 7919 + slot.col_tile * 7927));
    constexpr float bl_swing_mv = static_cast<float>(neurosim::adc_2bit::BL_SWING_MV);
    resolved_thermal_sigma_mv_ = resolve_thermal_sigma_mv();
    const float thermal_sigma_mv = resolved_thermal_sigma_mv_;
    std::normal_distribution<float> thermal_mv_dist(0.0f, thermal_sigma_mv);

    const int adc_levels = (adc_bits_ >= 16) ? 0 : (1 << std::max(1, adc_bits_));
    const float adc_headroom = enable_pv_ ? 1.25f : 1.0f;

    const int bpc = std::max(1, std::min(8, bits_per_cycle_));
    const int serial_steps = get_serial_steps();
    const int chunk_mask = (1 << bpc) - 1;

    const int read_n = read_avg_n_;

    auto analog_adc_roundtrip = [&](float partial, float full_scale) -> float {
        if (full_scale <= 1e-8f) {
            return 0.0f;
        }
        const float clipped = std::max(-full_scale, std::min(full_scale, partial));
        const float base_bl_mv = (clipped / full_scale * 0.5f + 0.5f) * bl_swing_mv;

        if (read_n <= 1) {
            // Single read (original path)
            float bl_mv = base_bl_mv;
            if (enable_thermal_) {
                bl_mv += thermal_mv_dist(thermal_rng);
            }
            bl_mv = std::max(0.0f, std::min(bl_swing_mv, bl_mv));

            if (adc_levels > 1) {
                const float level = adc_quantize(bl_mv, adc_levels);
                const float denom = static_cast<float>(adc_levels - 1);
                const float norm = level / denom;
                return (norm * 2.0f - 1.0f) * full_scale;
            }
            const float norm = bl_mv / bl_swing_mv;
            return (norm * 2.0f - 1.0f) * full_scale;
        }

        // Multi-read averaging: N independent reads with independent thermal noise
        float acc = 0.0f;
        for (int r = 0; r < read_n; r++) {
            float bl_mv = base_bl_mv;
            if (enable_thermal_) {
                bl_mv += thermal_mv_dist(thermal_rng);
            }
            bl_mv = std::max(0.0f, std::min(bl_swing_mv, bl_mv));

            if (adc_levels > 1) {
                const float level = adc_quantize(bl_mv, adc_levels);
                const float denom = static_cast<float>(adc_levels - 1);
                acc += level / denom;
            } else {
                acc += bl_mv / bl_swing_mv;
            }
        }
        const float norm = acc / static_cast<float>(read_n);
        return (norm * 2.0f - 1.0f) * full_scale;
    };

    // ---- Per-column MAC with noise ----
    for (int j = col_start; j < col_end; j++) {
        std::vector<float> w_eff_col(static_cast<size_t>(d_in_), 0.0f);
        float sum_w_eff = 0.0f;
        float sum_abs_w_eff = 0.0f;
        for (int i = 0; i < d_in_ && i < feat_size; i++) {
            float w_eff = static_cast<float>(
                weights_quantized_[static_cast<size_t>(i) * d_out_ + j]);
            if (enable_pv_) {
                const float vth_off = cell_vth_offset_mv_[
                    static_cast<size_t>(i) * d_out_ + j];
                const float pv_factor = std::max(0.5f, 1.0f + vth_off / vdd_mv);
                w_eff *= pv_factor;
            }
            w_eff_col[static_cast<size_t>(i)] = w_eff;
            sum_w_eff += w_eff;
            sum_abs_w_eff += std::abs(w_eff);
        }
        const float sum_abs_fallback = (j < static_cast<int>(sum_abs_w_q_per_col_.size()))
                                           ? sum_abs_w_q_per_col_[static_cast<size_t>(j)]
                                           : 0.0f;
        const float sum_abs_nonzero = std::max(1.0f, std::max(sum_abs_w_eff, sum_abs_fallback));

        if (fp_activations_) {
            float dot_q = 0.0f;
            for (int i = 0; i < d_in_ && i < feat_size; i++) {
                dot_q += slot.features[i] * w_eff_col[static_cast<size_t>(i)];
            }
            const float dot_full_scale = std::max(
                1e-6f, x_abs_max * sum_abs_nonzero * adc_headroom);
            const float dot_q_adc = analog_adc_roundtrip(dot_q, dot_full_scale);
            slot.partial_sum[j] += dot_q_adc * w_scale_;
            continue;
        }

        // Bit-serial activation path: per-cycle ADC quantization + digital shift-add.
        float dot_u_acc = 0.0f;
        for (int step = 0; step < serial_steps; step++) {
            const int shift = step * bpc;
            float partial = 0.0f;
            for (int i = 0; i < d_in_ && i < feat_size; i++) {
                const int xu = x_u[static_cast<size_t>(i)];
                const int x_chunk = (xu >> shift) & chunk_mask;
                if (x_chunk == 0) {
                    continue;
                }
                partial += static_cast<float>(x_chunk) * w_eff_col[static_cast<size_t>(i)];
            }

            const float partial_full_scale = std::max(
                1e-6f, static_cast<float>(chunk_mask) * sum_abs_nonzero * adc_headroom);
            const float partial_adc = analog_adc_roundtrip(partial, partial_full_scale);
            dot_u_acc += partial_adc * static_cast<float>(1 << shift);
        }

        const float dot_signed = dot_u_acc - 128.0f * sum_w_eff;
        slot.partial_sum[j] += dot_signed * x_scale * w_scale_;
    }
}

// =============================================================================
// DCIM Digital Computation — pure INT8×INT8 MAC, no analog noise
// =============================================================================

void NeuraCiM::compute_row_matmul_dcim_digital(PipelineSlot& slot) {
    // =========================================================================
    // DCIM Digital: INT8 weight × INT8 activation → INT32 accumulate → FP32
    //
    // No noise sources: no ADC, no DAC, no thermal, no process variation.
    // This is the "pure INT8 SW" equivalent executed on CiM SRAM hardware.
    // =========================================================================

    const size_t expected_w = static_cast<size_t>(d_in_) * static_cast<size_t>(d_out_);
    if (weights_quantized_.size() != expected_w) {
        quantize_weights_for_dcim();
    }
    if (weights_quantized_.size() != expected_w) {
        compute_row_matmul(slot);  // fallback to FP32
        return;
    }

    if (slot.partial_sum.size() != static_cast<size_t>(d_out_)) {
        slot.partial_sum.resize(static_cast<size_t>(d_out_), 0.0f);
    }
    const int col_start = slot.col_tile * neurosim::SUBARRAY_COLS;
    const int col_end = std::min((slot.col_tile + 1) * neurosim::SUBARRAY_COLS, d_out_);
    const int feat_size = static_cast<int>(slot.features.size());

    // ---- Input quantization (symmetric INT8, per-row dynamic scale) ----
    // Each row computes its own scale from max-abs.  In digital hardware this
    // is a simple comparator tree before the MAC array — no analog calibration.
    float x_scale = 1.0f;
    {
        float x_abs_max = 0.0f;
        for (int i = 0; i < d_in_ && i < feat_size; i++) {
            x_abs_max = std::max(x_abs_max, std::abs(slot.features[i]));
        }
        x_scale = (x_abs_max > 1e-10f) ? (x_abs_max / 127.0f) : 1.0f;
    }

    // Quantize activations to INT8
    std::vector<int8_t> x_q(static_cast<size_t>(d_in_), 0);
    for (int i = 0; i < d_in_ && i < feat_size; i++) {
        float q = std::round(slot.features[i] / x_scale);
        q = std::max(-128.0f, std::min(127.0f, q));
        x_q[i] = static_cast<int8_t>(q);
    }

    // Capture the exact values consumed by the DCIM_DIGITAL MAC.  Capture once
    // at the first column tile; later tiles must see the same quantization.
    if (slot.col_tile == 0 && slot.row_id >= 0 && slot.row_id < total_rows_) {
        const size_t row = static_cast<size_t>(slot.row_id);
        if (last_quantized_inputs_.size() ==
                static_cast<size_t>(total_rows_) * static_cast<size_t>(d_in_) &&
            last_input_scales_.size() == static_cast<size_t>(total_rows_) &&
            last_input_trace_valid_.size() == static_cast<size_t>(total_rows_)) {
            std::copy(x_q.begin(), x_q.end(),
                      last_quantized_inputs_.begin() + row * static_cast<size_t>(d_in_));
            last_input_scales_[row] = x_scale;
            last_input_trace_valid_[row] = 1;
        }
    }

    // ---- Signed input-bit-serial INT8×INT8 MAC ----
    for (int j = col_start; j < col_end; j++) {
        int32_t acc = 0;
        for (int bit = 0; bit < 8; ++bit) {
            const int32_t significance = bit == 7 ? -128 : (1 << bit);
            int32_t partial = 0;
            for (int i = 0; i < d_in_ && i < feat_size; i++) {
                const uint8_t ux = static_cast<uint8_t>(x_q[i]);
                if (((ux >> bit) & 1U) != 0U) {
                    partial += static_cast<int32_t>(weights_quantized_[static_cast<size_t>(i) * d_out_ + j]);
                }
            }
            acc += significance * partial;
        }
        // Dequantize: INT32 result × x_scale × w_scale → FP32
        slot.partial_sum[j] += static_cast<float>(acc) * x_scale * w_scale_;
    }
}
