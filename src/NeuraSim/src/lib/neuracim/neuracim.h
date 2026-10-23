// SPDX-License-Identifier: Apache-2.0
/**
 * neuracim.h
 * 
 * NeuraCiM - SRAM Compute-in-Memory module for NeuraChip.
 * Performs dense matrix multiplication (Combination step) using CiM.
 * 
 * Architecture (from config.py SRAMCiMConfig):
 * - 16 parallel CiM tiles (reduced from 64 for realistic area)
 * - Each tile: 128×128 subarray
 * - Throughput: 64 rows/cycle (pipelined)
 * - 2-bit parallelism → ADC latency reduced
 * 
 * Cycle Calculation:
 *   Total = ceil(n / num_tiles) + pipeline_latency
 *   For n=19717, tiles=64: ceil(19717/64) + 8 = 309 + 8 = 317 cycles
 * 
 * Based on NeuroSim V1.4 timing/energy parameters.
 */

#ifndef NEURACIM_H
#define NEURACIM_H

#include <queue>
#include "despoina_official_dcim_params.h"
#include <vector>
#include <array>
#include <string>
#include <cstdint>
#include <random>
#include <unordered_map>
#include <deque>
#include "../neurosim_params.h"

/**
 * CiM computation fidelity mode
 */
enum class CiMComputeMode {
    IDEAL_FP32,       // Current behavior: ideal float multiply-accumulate
    ACIM_FAITHFUL,    // Analog CiM: bit-serial ADC + thermal noise + PV
    DCIM_DIGITAL      // Digital CiM: INT8×INT8 pure digital MAC, no ADC/DAC/analog noise
};

enum class ActivationScaleMode {
    FIXED,
    WARMUP_MAXABS
};
#ifdef ENABLE_DRAMSIM3_PE
#include "dramsim3.h"
#endif

/**
 * CiM pipeline stage states
 */
enum class NeuraCiMState {
    EMPTY,          // Slot is empty
    PRECHARGE,      // Precharging bit lines
    COMPUTE,        // Row activation + sensing
    ADC_CONVERT,    // ADC conversion
    SHIFT_ADD,      // Accumulate partial sums
    OUTPUT_READY    // Ready to output
};

/**
 * Pipeline module state
 */
enum class NeuraCiMModuleState {
    IDLE,           // Waiting for input
    RUNNING,        // Pipeline active
    DONE            // Operation complete
};

/**
 * Pipeline slot - holds one row being processed
 */
struct PipelineSlot {
    bool valid;                         // Slot contains valid data
    int row_id;                         // Which row
    NeuraCiMState stage;                // Current pipeline stage
    int cycle_in_stage;                 // Cycles spent in current stage
    int col_tile;                       // Current column tile (for multi-tile)
    std::vector<float> features;        // Input features
    std::vector<float> partial_sum;     // Accumulated result
    
    PipelineSlot() : valid(false), row_id(-1), stage(NeuraCiMState::EMPTY), 
                     cycle_in_stage(0), col_tile(0) {}
    
    void reset() {
        valid = false;
        row_id = -1;
        stage = NeuraCiMState::EMPTY;
        cycle_in_stage = 0;
        col_tile = 0;
        features.clear();
        partial_sum.clear();
    }
};

// Despoina official DCIM logical server: one full replicated INT8 W image per logical tile,
// one distinct row in flight, and no inter-tile partial-sum reduction.
struct OfficialDcimTileServer {
    bool busy = false;
    bool compute_done = false;
    int row_id = -1;
    int bit_step = 0;
    uint64_t issue_cycle = 0;
    uint64_t completion_cycle = 0;
    uint64_t next_bit_completion_cycle = 0;
    float input_scale = 1.0f;
    std::vector<int8_t> quantized_input;
    std::vector<int32_t> accumulator;
    std::vector<float> result;
};

struct OfficialDcimWeightBankKey {
    int layer_id = -1;
    uint64_t version = 0;
    uint64_t source_hash = 0;
    int rows = 0;
    int cols = 0;
};

/**
 * Input packet for CiM matmul
 */
struct CiMInputPacket {
    int row_id;                     // Which row of H_agg
    std::vector<float> features;    // Feature vector [d_in]
    bool is_incremental;            // Incremental mode flag
};

/**
 * Output packet from CiM matmul
 */
struct CiMOutputPacket {
    int row_id;                     // Which row of H_comb
    std::vector<float> result;      // Result vector [d_out]
};

/**
 * NeuraCiM - Compute-in-Memory Module
 */
class NeuraCiM {
public:
    struct PipelineCounters {
        uint64_t input_fifo_full_rejections = 0;
        uint64_t input_fifo_full_cycles = 0;
        uint64_t output_fifo_full_cycles = 0;
        uint64_t output_fifo_full_stall_cycles = 0;
        uint64_t stage3_output_block_cycles = 0;
        uint64_t inject_no_empty_lane_cycles = 0;
        uint64_t active_lane_cycles = 0;
        uint64_t active_slot_cycles = 0;
        uint64_t input_fifo_occupancy_cycles = 0;
        uint64_t output_fifo_occupancy_cycles = 0;
        uint64_t max_active_lanes = 0;
    };

    struct NativeTraceStats {
        uint64_t elapsed_cycles = 0;
        uint64_t row_issue_events = 0;
        uint64_t row_completion_events = 0;
        uint64_t functional_apply_events = 0;
        uint64_t bit_step_issue_events = 0;
        uint64_t bit_step_completion_events = 0;
        uint64_t full_int8_energy_charge_events = 0;
        uint64_t first_bit_step_issue_cycle = 0;
        uint64_t last_bit_step_completion_cycle = 0;
        uint64_t bit_step_trace_fnv1a64 = 1469598103934665603ULL;
        uint64_t bit_step_temporal_order_violations = 0;
        uint64_t max_input_fifo_rows = 0;
        uint64_t max_output_fifo_rows = 0;
        std::vector<uint64_t> row_issue_cycles;
        std::vector<uint64_t> row_completion_cycles;
    };

    // =========================================================================
    // Constructor / Destructor
    // =========================================================================
    
    /**
     * Initialize NeuraCiM module.
     * 
     * @param d_in Input feature dimension
     * @param d_out Output feature dimension
     * @param num_tiles Number of parallel CiM tiles (default: 16, reduced for area)
     * @param bits_per_cycle Bits processed per cycle (default: 2 from config.py)
     */
    NeuraCiM(int d_in, int d_out, int num_tiles = 16, int bits_per_cycle = 2, int fifo_depth = 16);
    ~NeuraCiM();
    
    // =========================================================================
    // Weight Management
    // =========================================================================
    
    /**
     * Load weight matrix into CiM array.
     * Call once at initialization or when weights change.
     * 
     * @param W Weight matrix [d_in x d_out], row-major
     */
    void load_weights(const std::vector<float>& W);

    // Two tile-local logical W banks.  The simulator stores one canonical
    // image per bank and accounts the 16 replicated physical copies in bytes
    // and program events; there is no 16x software payload duplication.
    void official_dcim_bind_active_bank(const OfficialDcimWeightBankKey& key);
    bool official_dcim_active_bank_matches(const OfficialDcimWeightBankKey& key) const;
    bool official_dcim_inactive_bank_matches(const OfficialDcimWeightBankKey& key) const;
    bool official_dcim_prefetch_inactive_bank(const std::vector<float>& W,
                                         const OfficialDcimWeightBankKey& key,
                                         uint64_t program_cycles);
    uint64_t official_dcim_swap_to_prefetched(const OfficialDcimWeightBankKey& key,
                                         uint64_t overlap_available_cycles);
    uint64_t get_official_dcim_bank_prefetch_events() const { return official_dcim_bank_prefetch_events_; }
    uint64_t get_official_dcim_bank_swap_events() const { return official_dcim_bank_swap_events_; }
    uint64_t get_official_dcim_bank_exposed_stall_cycles() const { return official_dcim_bank_exposed_stall_cycles_; }
    uint64_t get_official_dcim_bank_program_bytes() const { return official_dcim_bank_program_bytes_; }
    uint64_t get_official_dcim_bank_program_tile_events() const { return official_dcim_bank_program_tile_events_; }
    int get_official_dcim_active_bank() const { return official_dcim_active_bank_; }

    /**
     * Configure DRAMsim3 output directory for weight load traffic.
     * If not called, a default local directory is used.
     */
    void configure_weight_dram_output_dir(const std::string& output_dir);
    void configure_weight_dram_config_file(const std::string& config_file);
#ifdef ENABLE_DRAMSIM3_PE
    dramsim3::CommandEnergySnapshot get_weight_hbm_command_energy() const {
        return weight_dram_ != nullptr
            ? weight_dram_->GetCommandEnergySnapshot()
            : dramsim3::CommandEnergySnapshot{};
    }
#endif
    
    // =========================================================================
    // Simulation Interface
    // =========================================================================
    
    /**
     * Start a matrix multiplication operation.
     * 
     * @param n Number of input rows
     * @param d_in Input dimension (must match constructor)
     * @param d_out Output dimension (must match constructor)
     * @param is_incremental Whether this is incremental mode
     */
    bool start_matmul(int n, int d_in, int d_out, bool is_incremental = false);
    
    /**
     * Push an input row to the input FIFO.
     * 
     * @param packet Input packet with row_id and features
     * @return true if accepted, false if FIFO full
     */
    bool push_input(const CiMInputPacket& packet);
    
    /**
     * Check if input FIFO can accept more data.
     * Used for backpressure signaling in streaming mode.
     * 
     * @return true if FIFO has space, false if full
     */
    bool can_accept_input() const { return input_fifo_.size() < static_cast<size_t>(fifo_depth_); }
    
    /**
     * Pop a result row from the output FIFO.
     * 
     * @param packet Output packet to fill
     * @return true if available, false if FIFO empty
     */
    bool pop_output(CiMOutputPacket& packet);
    
    /**
     * Advance simulation by one cycle.
     * Call this every clock cycle.
     * 
     * @return true if module is still busy
     */
    bool tick();
    
    /**
     * Check if operation is complete.
     */
    bool is_done() const { return module_state_ == NeuraCiMModuleState::DONE && output_fifo_.empty(); }
    
    /**
     * Compute CiM cycles using analytical model (accurate, fast).
     * 
     * From config.py SRAMCiMConfig:
     * - num_tiles = 64 (parallel CiM tiles)
     * - bits_per_cycle = 2 (ADC latency reduced by 2x)
     * - Pipeline: precharge(1) + compute(1) + ADC(5) + shift_add(2) = ~8 cycles
     * 
     * Throughput: num_tiles rows/cycle (pipelined)
     * Total cycles = ceil(n / num_tiles) + pipeline_latency
     * 
     * @param n Number of input rows
     * @return Total cycles for CiM matmul
     */
    uint64_t compute_cycles_analytical(int n) const {
        if (compute_mode_ == CiMComputeMode::DCIM_DIGITAL) {
            // Eight input-bit events across D128 are 1024 cycles per logical
            // tile server. Sixteen servers give the 64-cycle/row aggregate.
            const int waves = (n + num_tiles_ - 1) / num_tiles_;
            return static_cast<uint64_t>(waves) * 1024ULL;
        }
        // Pipeline latency = sum of all stage latencies
        int output_cycles = get_cycles_for_stage(NeuraCiMState::OUTPUT_READY);
        int pipeline_latency = cycles_precharge_ + cycles_compute_ +
                               cycles_adc_ + cycles_shift_add_ + output_cycles;
        
        // Throughput: num_tiles_ rows/cycle (64 tiles in parallel)
        // Each tile processes 1 row per cycle (pipelined)
        int rows_per_tile = (n + num_tiles_ - 1) / num_tiles_;  // ceil division
        
        // Total = rows_per_tile + pipeline_latency - 1
        return static_cast<uint64_t>(rows_per_tile + pipeline_latency - 1);
    }
    
    /**
     * Run CiM using analytical model (accurate, no simulation overhead).
     * 
     * @param n Number of input rows
     * @return Total cycles
     */
    uint64_t run_fast(int n) {
        compute_cycles_ = compute_cycles_analytical(n);
        module_state_ = NeuraCiMModuleState::DONE;
        // Accumulate energy based on compute mode
        if (compute_mode_ == CiMComputeMode::DCIM_DIGITAL) {
            accumulate_energy_dcim_digital(n, d_out_);
        } else {
            accumulate_energy(n, d_out_);
        }
        return compute_cycles_;
    }
    
    int get_num_tiles() const { return num_tiles_; }
    int get_fifo_depth() const { return fifo_depth_; }
    const NativeTraceStats& get_native_trace_stats() const { return native_trace_stats_; }
    
    /**
     * Check if module is idle.
     */
    bool is_idle() const { return module_state_ == NeuraCiMModuleState::IDLE; }
    
    /**
     * Check if pipeline is empty (all slots empty).
     */
    bool is_pipeline_empty() const;
    
    /**
     * Refresh timing/energy parameters by calling NeuroSim V1.4 executable.
     * 
     * @param neurosim_exe Path to NeuroSim 'main' executable
     * @return true if successful
     */
    bool refresh_params_from_neurosim(const std::string& neurosim_exe);
    
    // =========================================================================
    // Statistics / Counters
    // =========================================================================
    
    /**
     * Get total cycles spent in computation.
     */
    uint64_t get_compute_cycles() const { return compute_cycles_; }
    
    /**
     * Get total cycles spent idle.
     */
    uint64_t get_idle_cycles() const { return idle_cycles_; }
    
    /**
     * Get total energy consumed (pJ).
     */
    double get_total_energy_pj() const { return total_energy_pj_; }
    
    /**
     * Get energy breakdown.
     */
    double get_array_energy_pj() const { return array_energy_pj_; }
    double get_adc_energy_pj() const { return adc_energy_pj_; }
    double get_shift_add_energy_pj() const { return shift_add_energy_pj_; }
    double get_fifo_energy_pj() const { return fifo_energy_pj_; }
    double get_input_dac_energy_pj() const { return input_dac_energy_pj_; }
    double get_multiplier_energy_pj() const { return multiplier_energy_pj_; }
    double get_adder_tree_energy_pj() const { return adder_tree_energy_pj_; }
    double get_comparator_energy_pj() const { return comparator_energy_pj_; }
    double get_quantizer_energy_pj() const { return quantizer_energy_pj_; }
    double get_dequantizer_energy_pj() const { return dequantizer_energy_pj_; }
    int get_cycles_input_dac() const { return cycles_input_dac_; }
    uint64_t get_total_col_tile_passes() const { return total_col_tile_passes_; }
    uint64_t get_col_tile_pipeline_restarts() const { return col_tile_pipeline_restarts_; }
    uint64_t get_weight_load_cycles() const { return weight_load_cycles_; }
    uint64_t get_weight_load_dram_requests() const { return weight_load_dram_requests_; }
    uint64_t get_weight_load_bytes() const { return weight_load_bytes_; }
    uint64_t get_weight_source_fp32_bytes() const { return weight_load_bytes_; }
    uint64_t get_weight_resident_int8_bytes() const { return weight_resident_int8_bytes_; }
    uint64_t get_weight_load_events() const { return weight_load_events_; }
    uint64_t get_weight_program_tile_events() const { return weight_program_tile_events_; }
    uint64_t get_last_weight_program_cycles() const { return last_weight_program_cycles_; }
    uint64_t get_last_weight_source_fp32_bytes() const { return last_weight_source_fp32_bytes_; }
    uint64_t get_last_weight_programmed_int8_bytes() const { return last_weight_programmed_int8_bytes_; }
    const std::vector<uint64_t>& get_last_weight_tile_program_done_cycles() const {
        return last_weight_tile_program_done_cycles_;
    }
    const std::vector<uint64_t>& get_resident_weight_hash_per_tile() const {
        return resident_weight_hash_per_tile_;
    }
    const std::vector<uint8_t>& get_resident_weight_loaded_per_tile() const {
        return resident_weight_loaded_per_tile_;
    }
    bool all_resident_tiles_ready() const;
    PipelineCounters get_pipeline_counters() const { return pipeline_counters_; }
    
    /**
     * Reset all counters.
     */
    void reset_counters();
    
    // =========================================================================
    // Configuration
    // =========================================================================
    
    int get_d_in() const { return d_in_; }
    int get_d_out() const { return d_out_; }
    int get_bits_per_cycle() const { return bits_per_cycle_; }
    const std::vector<int8_t>& get_quantized_weights() const { return weights_quantized_; }
    float get_weight_scale() const { return w_scale_; }
    const std::vector<int8_t>& get_last_quantized_inputs() const {
        return last_quantized_inputs_;
    }
    const std::vector<float>& get_last_input_scales() const {
        return last_input_scales_;
    }
    const std::vector<uint8_t>& get_last_input_trace_valid() const {
        return last_input_trace_valid_;
    }

    // =========================================================================
    // DCIM Computation Mode
    // =========================================================================

    void set_compute_mode(CiMComputeMode mode);
    CiMComputeMode get_compute_mode() const { return compute_mode_; }

    /**
     * Set seed for reproducible per-cell process variation offsets.
     * Must be called before load_weights() in ACIM_FAITHFUL mode.
     */
    void set_variation_seed(uint32_t seed);
    void set_adc_bits(int bits);
    int get_adc_bits() const { return adc_bits_; }

    // High-precision mode: keep activations in FP32, only quantize weights to INT8
    void set_fp_activations(bool fp);
    bool get_fp_activations() const { return fp_activations_; }

    // Disable noise sources for ablation study
	    void set_enable_pv(bool en);
	    void set_enable_thermal(bool en);
	    void set_enable_pv_compensation(bool en);
	    void set_thermal_sigma_override_mv(float sigma_mv);
	    void set_thermal_sigma_scale(float scale);
	    void set_temperature_k(float temp_k);
	    void set_thermal_model(const std::string& model);
	    void set_thermal_bl_cap_ff(float cap_ff);
	    void set_thermal_adc_comp_sigma_mv(float sigma_mv);
	    float get_resolved_thermal_sigma_mv() const { return resolved_thermal_sigma_mv_; }
	    std::string get_thermal_model() const { return thermal_model_; }
	    float get_thermal_bl_cap_ff() const { return thermal_bl_cap_ff_; }
	    float get_thermal_adc_comp_sigma_mv() const { return thermal_adc_comp_sigma_mv_; }
	    void set_read_avg_n(int n);
	    int get_read_avg_n() const { return read_avg_n_; }
	    void set_activation_scale_mode(const std::string& mode);
	    ActivationScaleMode get_activation_scale_mode() const { return activation_scale_mode_; }
	    std::string get_activation_scale_mode_str() const;
	    void set_act_calib_rows(int rows);
	    int get_act_calib_rows() const { return act_calib_rows_; }
	    void set_act_calib_clip(float clip);
	    float get_act_calib_clip() const { return act_calib_clip_; }
	    bool get_enable_pv() const { return enable_pv_; }
	    bool get_enable_thermal() const { return enable_thermal_; }
	    bool get_enable_pv_compensation() const { return enable_pv_compensation_; }

    /**
     * Initialize per-cell Vth offsets using neurosim::noise::VTH_VARIATION_PCT.
     * Called automatically from load_weights() when in ACIM_FAITHFUL or DCIM_DIGITAL mode.
     */
    void initialize_process_variation();
    
private:
    // Configuration (from config.py SRAMCiMConfig)
    int d_in_;
    int d_out_;
    int num_tiles_;       // 64 parallel CiM tiles (was num_subarrays_)
    int bits_per_cycle_;  // 2-bit parallelism
    int fifo_depth_;      // Runtime input/output FIFO depth in rows
    
    // Weight storage
    std::vector<float> weights_;  // [d_in x d_out]
    bool weights_loaded_;
    
    // FIFOs
    std::queue<CiMInputPacket> input_fifo_;
    std::queue<CiMOutputPacket> output_fifo_;
    
    // Module state
    NeuraCiMModuleState module_state_;
    int total_rows_;            // Total rows to process
    int rows_completed_;        // Rows completed
    int rows_entered_;          // Rows entered pipeline
    bool is_incremental_;       // Incremental mode flag
    
    // 5-Stage Pipeline
    static constexpr int NUM_PIPELINE_STAGES = 5;
    // True parallelism: num_tiles_ independent pipeline lanes.
    // lane[t][s] where s: 0=PRECHARGE, 1=COMPUTE, 2=ADC, 3=SHIFT_ADD, 4=OUTPUT_READY
    std::vector<std::array<PipelineSlot, NUM_PIPELINE_STAGES>> pipeline_lanes_;
    
    // Tiling
    int total_col_tiles_;       // Total output column tiles
    
    // Timing (derived from neurosim_params.h)
    int cycles_precharge_;
    int cycles_compute_;
    int cycles_adc_;
    int cycles_shift_add_;
    int cycles_input_dac_;    // Input activation DAC latency (overlaps with precharge)
    
    // Counters
    uint64_t compute_cycles_;
    uint64_t idle_cycles_;
    uint64_t total_col_tile_passes_;
    uint64_t col_tile_pipeline_restarts_;
    double total_energy_pj_;
    double array_energy_pj_;
    double adc_energy_pj_;
    double shift_add_energy_pj_;
    double fifo_energy_pj_;
    double input_dac_energy_pj_;
    double multiplier_energy_pj_;       // DCIM_DIGITAL: INT8 multiplier energy
    double adder_tree_energy_pj_;       // DCIM_DIGITAL: adder tree energy
    double comparator_energy_pj_;       // DCIM_DIGITAL: max-abs comparator tree for quantizer
    double quantizer_energy_pj_;        // DCIM_DIGITAL: per-element FP32→INT8 scale-round-clamp
    double dequantizer_energy_pj_;      // DCIM_DIGITAL: per-element INT32→FP32 + scale multiply

    // One-time weight load accounting
    uint64_t weight_load_cycles_;
    uint64_t weight_load_dram_requests_;
    uint64_t weight_load_bytes_;
    uint64_t weight_resident_int8_bytes_;
    uint64_t weight_load_events_;
    uint64_t weight_program_tile_events_;
    uint64_t last_weight_program_cycles_;
    uint64_t last_weight_source_fp32_bytes_;
    uint64_t last_weight_programmed_int8_bytes_;
    std::vector<uint64_t> last_weight_tile_program_done_cycles_;
    std::vector<uint64_t> resident_weight_hash_per_tile_;
    std::vector<uint8_t> resident_weight_loaded_per_tile_;
    PipelineCounters pipeline_counters_;
    NativeTraceStats native_trace_stats_;
    uint64_t native_cycle_;
    std::string weight_dram_output_dir_;
    std::string weight_dram_config_file_;
#ifdef ENABLE_DRAMSIM3_PE
    dramsim3::MemorySystem* weight_dram_;
    std::unordered_map<uint64_t, int> weight_dram_inflight_;
    std::unordered_map<uint64_t, int> weight_dram_addr_tile_;
    std::vector<uint64_t> weight_tile_pending_txns_;
    uint64_t weight_source_pending_txns_;
    uint64_t weight_dram_current_cycle_;
    void ensure_weight_dram_enabled();
    void weight_dram_read_cb(uint64_t addr);
    void weight_dram_write_cb(uint64_t addr);
#endif
    
    // DCIM-faithful computation state
    CiMComputeMode compute_mode_;
    uint32_t variation_seed_;
    bool variation_initialized_;
    int adc_bits_;                            // ADC resolution (default 8 for INT8 weights)
    bool fp_activations_;                     // true = keep activations FP32 (only quantize weights)
    bool enable_pv_;                          // enable process variation noise
    bool enable_thermal_;                     // enable thermal read noise
	    bool enable_pv_compensation_;             // pre-distort quantized weight to offset static PV
	    float thermal_sigma_override_mv_;         // <= 0 means use default NeuroSim thermal sigma
	    float thermal_sigma_scale_;               // multiplicative thermal sigma scale
	    float temperature_k_;                     // absolute temperature in Kelvin
	    std::string thermal_model_;               // legacy | physics_ktc
	    float thermal_bl_cap_ff_;                 // bitline capacitance for kT/C calibration
	    float thermal_adc_comp_sigma_mv_;         // ADC comparator sigma for kT/C calibration
	    float resolved_thermal_sigma_mv_;         // resolved sigma used by latest compute pass
	    int read_avg_n_;                          // Number of ADC reads to average (1=off)
	    std::vector<float> cell_vth_offset_mv_;   // Per-cell static Vth offset [d_in*d_out]
	    std::vector<int8_t> weights_quantized_;   // INT8 quantized weights [d_in*d_out]
	    float w_scale_;                           // Weight quantization scale
	    float activation_scale_;                  // Fixed INT8 activation scale for current operation
	    bool activation_scale_ready_;             // Activation scale calibrated for current operation
	    bool activation_scale_manual_;            // Reserved for manual calibration mode
	    ActivationScaleMode activation_scale_mode_;   // Activation scale calibration mode
	    int act_calib_rows_;                          // Warmup rows for activation calibration
	    float act_calib_clip_;                        // Warmup scale clipping coefficient
	    float warmup_maxabs_;                         // Running max|x| during warmup
	    int warmup_target_rows_;                      // Number of rows required before freezing scale
	    int warmup_rows_seen_;                        // Warmup rows collected so far
	    std::deque<CiMInputPacket> warmup_input_queue_; // Buffered warmup packets before scale is fixed
	    std::vector<float> sum_w_q_per_col_;     // Per-column sum of w_q for unsigned bias correction
	    std::vector<float> sum_abs_w_q_per_col_; // Per-column sum of |w_q| for ADC full-scale estimate
	    // Exact DCIM_DIGITAL input quantization trace for the current matmul.
	    // Indexed by operation-local row_id, not graph row_id.
	    std::vector<int8_t> last_quantized_inputs_;
	    std::vector<float> last_input_scales_;
    std::vector<uint8_t> last_input_trace_valid_;

    // Dedicated Despoina official DCIM path. This is separate from the legacy five-stage
    // pipelined model so its 1024-cycle individual-row ownership cannot be
    // collapsed into an aggregate post-hoc multiplier.
    std::vector<OfficialDcimTileServer> official_dcim_servers_;

    struct OfficialDcimWeightBankImage {
        bool valid = false;
        OfficialDcimWeightBankKey key{};
        std::vector<float> fp32;
        std::vector<int8_t> int8;
        std::vector<float> sum_w;
        std::vector<float> sum_abs_w;
        float scale = 1.0f;
        uint64_t quantized_hash = 0;
        uint64_t program_cycles = 0;
    };
    std::array<OfficialDcimWeightBankImage, 2> official_dcim_weight_banks_{};
    int official_dcim_active_bank_ = 0;
    uint64_t official_dcim_bank_prefetch_events_ = 0;
    uint64_t official_dcim_bank_swap_events_ = 0;
    uint64_t official_dcim_bank_exposed_stall_cycles_ = 0;
    uint64_t official_dcim_bank_program_bytes_ = 0;
    uint64_t official_dcim_bank_program_tile_events_ = 0;
    uint64_t official_dcim_start_stall_remaining_ = 0;

	    // Internal methods
    void process_pipeline();
    void advance_stage(int stage_idx);
    void compute_row_matmul(PipelineSlot& slot);
	    void compute_row_matmul_dcim(PipelineSlot& slot);
	    void compute_row_matmul_dcim_digital(PipelineSlot& slot);
	    void process_official_dcim_logical_tiles();
	    OfficialDcimWeightBankImage build_official_dcim_weight_bank(
	        const std::vector<float>& W,
	        const OfficialDcimWeightBankKey& key,
	        uint64_t program_cycles) const;
	    float adc_quantize(float bl_voltage_mv, int num_levels) const;
	    void quantize_weights_for_dcim();
	    void update_total_energy();
	    void accumulate_energy(int rows_processed, int cols_processed);
	    void accumulate_energy_dcim_digital(int rows_processed, int cols_processed);
		    void refresh_timing_from_dcim_config();
	    float resolve_thermal_sigma_mv() const;
	    bool warmup_calibration_active() const;
	    void try_finalize_warmup_calibration();
	    void flush_warmup_packets_to_input_fifo();
	    int get_serial_steps() const;
    double adc_delay_ns_for_bits(int bits) const;
    double adc_energy_pj_for_bits(int bits) const;
    double dac_delay_ns_for_bits(int bits) const;
    int get_cycles_for_stage(NeuraCiMState stage) const;
};

#endif  // NEURACIM_H
