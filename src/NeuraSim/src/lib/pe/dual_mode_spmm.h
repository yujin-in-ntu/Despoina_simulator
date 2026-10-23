// SPDX-License-Identifier: Apache-2.0
#ifndef DUAL_MODE_SPMM_H
#define DUAL_MODE_SPMM_H

#include <cstdint>
#include <fstream>
#include <map>
#include <string>
#include <unordered_map>
#include <vector>

#include "pe_array_spmm.h"
#include "delta_f_aware_indexer.h"
#include "delta_converter.h"
#include "arch_variant.h"
#include "finalized_dcim_v3.h"
#include "resident_weight_contract_r5.h"
#include "dynamic_row_mean_operator.h"
#include "../neuracim/neuracim.h"  // CiMComputeMode

#ifndef DESPOINA_V3_FUNCTIONAL_DCIM
#define DESPOINA_V3_FUNCTIONAL_DCIM 0
#endif

#ifndef DESPOINA_V3_CACHE_EPOCH
#define DESPOINA_V3_CACHE_EPOCH 0
#endif

#if DESPOINA_V3_FUNCTIONAL_DCIM && !DESPOINA_V3_CACHE_EPOCH
#error "R5 functional DCIM requires matrix/job cache epochs; cross-matrix reuse corrupts P2/P3"
#endif

#ifdef ENABLE_DRAMSIM3_PE
#include "dramsim3.h"
#endif

class GlobalCounters;
class NeuraCiM;

enum class CiMMode {
    NONE,
    DELTA_ONLY,
    FULL_YNEW,
    PER_PHASE_LINEAR
};

enum class PhaseMode {
    THREE_PHASE,
    EDGE_ONLY,
    TWO_PHASE,
    ANEW_FULL
};

/**
 * DualModeSpMM — Despoina factorized row-mean controller
 *
 * - FULL / INCREMENTAL / SCHEDULED 3모드
 * - 단일 PEArraySpMM 엔진 재사용 (cycle-accurate)
 * - FeatureStore(F_current)는 내부 벡터로 유지, 매 timestep ΔF로 업데이트
 * - Phase2(A_new×ΔF)는 DeltaFAwareIndexer가 추출한 A_sub CSR을 사용
 *
 * Supported-component contract: the production path exercises the vendored
 * DRAMsim3 transaction model and accounts for official DCIM read/compute and
 * FIFO events with SHA-256-gated NeuroSim-, Accelergy-, and CACTI-derived
 * constants. This is not a full-chip or model-deployment energy model.
 */
class DualModeSpMM {
public:
    enum class Mode {
        FULL,
        INCREMENTAL,
        SCHEDULED
    };

    struct StepResult {
        struct AccuracyMetrics {
            bool executed = false;
            bool valid = false;
            bool pass = false;
            bool pass_hitrate = false;
            double rmse = 0.0;
            double cosine_similarity = 0.0;
            double max_abs_error = 0.0;
            double mean_abs_error = 0.0;
            double rel_error_mean = 0.0;
            double rel_error_max = 0.0;
            double hit_rate_rel_1pct = 0.0;
            double hit_rate_rel_2pct = 0.0;
            double hit_rate_rel_5pct = 0.0;
            int rows_compared = 0;
            int elements_compared = 0;
            bool hw_complete = false;
            bool hw_timeout = false;
            int hw_expected_rows = 0;
            int hw_received_unique_rows = 0;
            int hw_duplicate_packets = 0;
            int hw_invalid_packets = 0;
            double rmse_tolerance = 1e-6;
            double cosine_tolerance = 1e-6;
            double rel_error_epsilon = 1e-6;
            double abs_error_tolerance = 1e-4;
            double hitrate_pass_threshold = 0.99;
        };

        struct PerPhaseInfo {
            int nnz = 0;
            long long mac_ops = 0;
            long long cycles = 0;
        };

        struct YAccuracy {
            bool valid = false;
            bool exact = false;
            double rmse = 0.0;
            double cosine_similarity = 1.0;
        };

        struct ArchAblationCounters {
            bool behavior_forced = false;
            bool dcim_enabled = false;
            bool streaming_enabled = false;
            bool fullscan_enabled = false;
            bool indexstore_enabled = false;

            uint64_t y_materialize_read_bytes = 0;
            uint64_t y_materialize_write_bytes = 0;
            uint64_t z_materialize_read_bytes = 0;
            uint64_t z_materialize_write_bytes = 0;
            uint64_t materialization_eliminated_bytes = 0;

            uint64_t fullscan_bytes_read = 0;
            uint64_t indexstore_bytes_read = 0;
            uint64_t indexstore_bytes_write = 0;
            uint64_t index_lookup_ops = 0;
            uint64_t index_lookup_cycles = 0;
            uint64_t index_bytes_read = 0;
            uint64_t index_bytes_write = 0;

            uint64_t crb_bytes_read = 0;
            uint64_t crb_bytes_write = 0;
            uint64_t crb_popcount_cycles = 0;
            uint64_t crb_bytes_read_estimated = 0;
            uint64_t crb_bytes_write_estimated = 0;
            uint64_t crb_popcount_cycles_estimated = 0;

            uint64_t row_detect_cycles = 0;
            uint64_t row_detect_checks = 0;
            uint64_t row_detect_sram_reads = 0;

            uint64_t dcim_cycles = 0;
            uint64_t dcim_exposed_cycles = 0;
            uint64_t dcim_stream_cycles = 0;
            uint64_t tile_busy_work_cycles = 0;
            uint64_t causal_stream_elapsed_cycles = 0;
            uint64_t exposed_tail_cycles = 0;
            uint64_t cim_rows = 0;
            uint64_t cim_active_lane_cycles = 0;
            uint64_t cim_active_slot_cycles = 0;
            uint64_t cim_input_fifo_occupancy_cycles = 0;
            uint64_t cim_output_fifo_occupancy_cycles = 0;
            uint64_t cim_pe_backpressure_cycles = 0;
            uint64_t stream_fifo_bytes = 0;
            uint64_t stream_fifo_bytes_estimated = 0;
            uint64_t digital_stream_estimated_cycles = 0;
            uint64_t digital_stream_saved_cycles = 0;
            uint64_t digital_stream_fifo_bytes_estimated = 0;
            uint64_t digital_stream_measured_cycles = 0;
            uint64_t digital_stream_compute_cycles = 0;
            uint64_t digital_stream_overlap_cycles = 0;
            uint64_t digital_stream_fifo_bytes = 0;
            uint64_t digital_stream_input_fifo_occupancy_cycles = 0;
            uint64_t digital_stream_output_fifo_occupancy_cycles = 0;
            uint64_t digital_stream_backpressure_cycles = 0;
            uint64_t digital_stream_rows = 0;
            uint64_t digital_stream_mac_ops = 0;
            uint64_t digital_stream_count = 0;
            uint64_t digital_stream_drain_sensitivity_cycles = 0;
            uint64_t digital_stream_plus6_exposed_cycles = 0;
            uint64_t digital_stream_packets = 0;
            uint64_t digital_stream_duplicate_packets = 0;
            uint64_t digital_stream_invalid_packets = 0;
            uint64_t digital_stream_timeouts = 0;
            uint64_t digital_stream_activation_quantized_elements = 0;
            uint64_t digital_stream_dequantized_elements = 0;
            uint64_t digital_stream_comparator_tree_passes = 0;
            uint64_t digital_stream_bit_step_issue_events = 0;
            uint64_t digital_stream_bit_step_completion_events = 0;
            uint64_t digital_stream_full_int8_energy_charge_events = 0;
            uint64_t digital_stream_tile_dispatch_events = 0;
            uint64_t digital_stream_weight_program_cycles = 0;
            uint64_t digital_stream_weight_program_tile_events = 0;
            uint64_t digital_stream_weight_source_fp32_bytes = 0;
            uint64_t digital_stream_weight_programmed_int8_bytes = 0;
            uint64_t digital_stream_weight_bank_prefetch_events = 0;
            uint64_t digital_stream_weight_bank_swap_events = 0;
            uint64_t digital_stream_weight_bank_exposed_stall_cycles = 0;
            uint64_t digital_stream_overlap_sensitivity_finish_cycles = 0;
            uint64_t digital_stream_overlap_sensitivity_exposed_cycles = 0;
            uint64_t digital_stream_event_trace_xor = 0;
            uint64_t digital_stream_first_bit_step_issue_cycle = 0;
            uint64_t digital_stream_last_bit_step_completion_cycle = 0;
            // Exact logical PE-output-staging -> DCIM-input-FIFO transfers.
            // The path is direct and has no inferred mesh-hop field.
            uint64_t pe_cim_noc_flits = 0;
            uint64_t pe_cim_noc_bytes = 0;
            uint64_t z_sram_read_bytes = 0;
            uint64_t z_sram_write_bytes = 0;
            uint64_t z_sram_rw_bytes = 0;
            int digital_stream_tile_rows = 0;
            int digital_stream_tile_cols = 0;
            int digital_stream_lanes = 0;

            uint64_t systolic_cycles = 0;
            uint64_t systolic_compute_cycles = 0;
            uint64_t systolic_pack_cycles = 0;
            uint64_t systolic_materialize_cycles = 0;
            uint64_t systolic_active_rows = 0;
            uint64_t systolic_mac_ops = 0;
            uint64_t systolic_pack_bytes = 0;
            uint64_t systolic_materialize_bytes = 0;
            int systolic_tile_rows = 0;
            int systolic_tile_cols = 0;

            void add(const ArchAblationCounters& other) {
                behavior_forced = behavior_forced || other.behavior_forced;
                dcim_enabled = dcim_enabled || other.dcim_enabled;
                streaming_enabled = streaming_enabled || other.streaming_enabled;
                fullscan_enabled = fullscan_enabled || other.fullscan_enabled;
                indexstore_enabled = indexstore_enabled || other.indexstore_enabled;
                y_materialize_read_bytes += other.y_materialize_read_bytes;
                y_materialize_write_bytes += other.y_materialize_write_bytes;
                z_materialize_read_bytes += other.z_materialize_read_bytes;
                z_materialize_write_bytes += other.z_materialize_write_bytes;
                materialization_eliminated_bytes += other.materialization_eliminated_bytes;
                fullscan_bytes_read += other.fullscan_bytes_read;
                indexstore_bytes_read += other.indexstore_bytes_read;
                indexstore_bytes_write += other.indexstore_bytes_write;
                index_lookup_ops += other.index_lookup_ops;
                index_lookup_cycles += other.index_lookup_cycles;
                index_bytes_read += other.index_bytes_read;
                index_bytes_write += other.index_bytes_write;
                crb_bytes_read += other.crb_bytes_read;
                crb_bytes_write += other.crb_bytes_write;
                crb_popcount_cycles += other.crb_popcount_cycles;
                crb_bytes_read_estimated += other.crb_bytes_read_estimated;
                crb_bytes_write_estimated += other.crb_bytes_write_estimated;
                crb_popcount_cycles_estimated += other.crb_popcount_cycles_estimated;
                row_detect_cycles += other.row_detect_cycles;
                row_detect_checks += other.row_detect_checks;
                row_detect_sram_reads += other.row_detect_sram_reads;
                dcim_cycles += other.dcim_cycles;
                dcim_exposed_cycles += other.dcim_exposed_cycles;
                dcim_stream_cycles += other.dcim_stream_cycles;
                tile_busy_work_cycles += other.tile_busy_work_cycles;
                causal_stream_elapsed_cycles += other.causal_stream_elapsed_cycles;
                exposed_tail_cycles += other.exposed_tail_cycles;
                cim_rows += other.cim_rows;
                cim_active_lane_cycles += other.cim_active_lane_cycles;
                cim_active_slot_cycles += other.cim_active_slot_cycles;
                cim_input_fifo_occupancy_cycles += other.cim_input_fifo_occupancy_cycles;
                cim_output_fifo_occupancy_cycles += other.cim_output_fifo_occupancy_cycles;
                cim_pe_backpressure_cycles += other.cim_pe_backpressure_cycles;
                stream_fifo_bytes += other.stream_fifo_bytes;
                stream_fifo_bytes_estimated += other.stream_fifo_bytes_estimated;
                digital_stream_estimated_cycles += other.digital_stream_estimated_cycles;
                digital_stream_saved_cycles += other.digital_stream_saved_cycles;
                digital_stream_fifo_bytes_estimated += other.digital_stream_fifo_bytes_estimated;
                digital_stream_measured_cycles += other.digital_stream_measured_cycles;
                digital_stream_compute_cycles += other.digital_stream_compute_cycles;
                digital_stream_overlap_cycles += other.digital_stream_overlap_cycles;
                digital_stream_fifo_bytes += other.digital_stream_fifo_bytes;
                digital_stream_input_fifo_occupancy_cycles += other.digital_stream_input_fifo_occupancy_cycles;
                digital_stream_output_fifo_occupancy_cycles += other.digital_stream_output_fifo_occupancy_cycles;
                digital_stream_backpressure_cycles += other.digital_stream_backpressure_cycles;
                digital_stream_rows += other.digital_stream_rows;
                digital_stream_mac_ops += other.digital_stream_mac_ops;
                digital_stream_count += other.digital_stream_count;
                digital_stream_drain_sensitivity_cycles +=
                    other.digital_stream_drain_sensitivity_cycles;
                digital_stream_plus6_exposed_cycles +=
                    other.digital_stream_plus6_exposed_cycles;
                digital_stream_packets += other.digital_stream_packets;
                digital_stream_duplicate_packets += other.digital_stream_duplicate_packets;
                digital_stream_invalid_packets += other.digital_stream_invalid_packets;
                digital_stream_timeouts += other.digital_stream_timeouts;
                digital_stream_activation_quantized_elements +=
                    other.digital_stream_activation_quantized_elements;
                digital_stream_dequantized_elements +=
                    other.digital_stream_dequantized_elements;
                digital_stream_comparator_tree_passes +=
                    other.digital_stream_comparator_tree_passes;
                digital_stream_bit_step_issue_events += other.digital_stream_bit_step_issue_events;
                digital_stream_bit_step_completion_events += other.digital_stream_bit_step_completion_events;
                digital_stream_full_int8_energy_charge_events +=
                    other.digital_stream_full_int8_energy_charge_events;
                digital_stream_tile_dispatch_events += other.digital_stream_tile_dispatch_events;
                digital_stream_weight_program_cycles += other.digital_stream_weight_program_cycles;
                digital_stream_weight_program_tile_events += other.digital_stream_weight_program_tile_events;
                digital_stream_weight_source_fp32_bytes += other.digital_stream_weight_source_fp32_bytes;
                digital_stream_weight_programmed_int8_bytes += other.digital_stream_weight_programmed_int8_bytes;
                digital_stream_weight_bank_prefetch_events += other.digital_stream_weight_bank_prefetch_events;
                digital_stream_weight_bank_swap_events += other.digital_stream_weight_bank_swap_events;
                digital_stream_weight_bank_exposed_stall_cycles +=
                    other.digital_stream_weight_bank_exposed_stall_cycles;
                digital_stream_overlap_sensitivity_finish_cycles +=
                    other.digital_stream_overlap_sensitivity_finish_cycles;
                digital_stream_overlap_sensitivity_exposed_cycles +=
                    other.digital_stream_overlap_sensitivity_exposed_cycles;
                digital_stream_event_trace_xor ^= other.digital_stream_event_trace_xor;
                if (digital_stream_first_bit_step_issue_cycle == 0 ||
                    (other.digital_stream_first_bit_step_issue_cycle != 0 &&
                     other.digital_stream_first_bit_step_issue_cycle <
                         digital_stream_first_bit_step_issue_cycle)) {
                    digital_stream_first_bit_step_issue_cycle =
                        other.digital_stream_first_bit_step_issue_cycle;
                }
                digital_stream_last_bit_step_completion_cycle = std::max(
                    digital_stream_last_bit_step_completion_cycle,
                    other.digital_stream_last_bit_step_completion_cycle);
                pe_cim_noc_flits += other.pe_cim_noc_flits;
                pe_cim_noc_bytes += other.pe_cim_noc_bytes;
                z_sram_read_bytes += other.z_sram_read_bytes;
                z_sram_write_bytes += other.z_sram_write_bytes;
                z_sram_rw_bytes += other.z_sram_rw_bytes;
                if (digital_stream_tile_rows == 0) {
                    digital_stream_tile_rows = other.digital_stream_tile_rows;
                }
                if (digital_stream_tile_cols == 0) {
                    digital_stream_tile_cols = other.digital_stream_tile_cols;
                }
                if (digital_stream_lanes == 0) {
                    digital_stream_lanes = other.digital_stream_lanes;
                }
                systolic_cycles += other.systolic_cycles;
                systolic_compute_cycles += other.systolic_compute_cycles;
                systolic_pack_cycles += other.systolic_pack_cycles;
                systolic_materialize_cycles += other.systolic_materialize_cycles;
                systolic_active_rows += other.systolic_active_rows;
                systolic_mac_ops += other.systolic_mac_ops;
                systolic_pack_bytes += other.systolic_pack_bytes;
                systolic_materialize_bytes += other.systolic_materialize_bytes;
                if (systolic_tile_rows == 0) {
                    systolic_tile_rows = other.systolic_tile_rows;
                }
                if (systolic_tile_cols == 0) {
                    systolic_tile_cols = other.systolic_tile_cols;
                }
            }
        };

        Mode mode_used = Mode::FULL;
        PhaseMode phase_mode_used = PhaseMode::THREE_PHASE;
        long long total_cycles = 0;
        PEArraySpMM::Stats pe_stats{}; // phase 합산
        PerPhaseInfo phase1{};
        PerPhaseInfo phase2{};
        PerPhaseInfo phase3{};

        // Exact dynamic row-mean metadata/correction ownership.  These fields
        // are exact owners for the fixed Factorized row-mean operator.
        bool row_mean_enabled = false;
        long long row_mean_scaled_rows = 0;
        long long row_mean_graph_changed_degree_rows = 0;
        long long row_mean_changed_degree_rows = 0;
        long long row_mean_scale_elements = 0;
        long long row_mean_residual_correction_elements = 0;
        long long row_mean_scalar_ops = 0;
        long long row_mean_stream_delay_cycles = 0;
        long long row_mean_degree_read_events = 0;
        long long row_mean_degree_write_events = 0;
        long long row_mean_degree_read_bytes = 0;
        long long row_mean_degree_write_bytes = 0;
        long long row_mean_reciprocal_events = 0;
        long long factorized_native_pe_cycles = 0;
        long long factorized_native_owner_cycles = 0;
        long long factorized_state_capacity_bytes = 0;
        long long factorized_state_sram_bytes = 0;
        long long factorized_state_hbm_bytes = 0;
        long long factorized_state_hbm_read_bytes = 0;
        long long factorized_state_hbm_write_bytes = 0;
        long long factorized_state_hbm_bursts = 0;
        long long factorized_state_hbm_cycles = 0;
        long long factorized_old_y_snapshot_bytes = 0;
        long long factorized_old_y_sram_bytes = 0;
        long long factorized_old_y_hbm_write_bytes = 0;
        long long factorized_old_y_hbm_read_bytes = 0;
        long long factorized_old_y_hbm_bursts = 0;
        long long factorized_old_y_hbm_cycles = 0;
        double factorized_pe_dynamic_energy_pj = 0.0;

        // Indexer/FeatureStore counters (for later modeling integration)
        DeltaFAwareIndexer::Counters indexer_ctrs{};

        // CiM integration fields (shared across A/B/C modes)
        CiMMode cim_mode = CiMMode::NONE;
        long long combine_cycles = 0;
        long long cim_cycles = 0;
        long long cim_stream_cycles = 0;
        long long cim_exposed_cycles = 0;
        long long cim_pe_backpressure_cycles = 0;
        long long cim_input_fifo_full_rejections = 0;
        long long cim_input_fifo_full_cycles = 0;
        long long cim_output_fifo_full_cycles = 0;
        long long cim_output_fifo_full_stall_cycles = 0;
        long long cim_stage3_output_block_cycles = 0;
        long long cim_inject_no_empty_lane_cycles = 0;
        long long cim_active_lane_cycles = 0;
        long long cim_active_slot_cycles = 0;
        long long cim_input_fifo_occupancy_cycles = 0;
        long long cim_output_fifo_occupancy_cycles = 0;
        int cim_max_active_lanes = 0;
        int cim_rows = 0;
        long long finalized_rows = 0;
        long long finalized_packets = 0;
        long long dcim_scalar_macs = 0;
        despoina_v3::TileBusyWorkCycles tile_busy_work_cycles{};
        despoina_v3::CausalStreamElapsedCycles causal_stream_elapsed_cycles{};
        despoina_v3::ExposedTailCycles exposed_tail_cycles{};
        long long dcim_bit_serial_steps = 0;
        long long dcim_drain_cycles = 0; // NEW +6/stream sensitivity only
        long long dcim_streams = 0;
        long long finalized_stream_cycles_plus6 = 0;
        long long finalized_exposed_cycles_plus6 = 0;
        long long finalized_backpressure_cycles = 0;
        long long finalized_max_staging_rows = 0;
        long long finalized_max_fifo_rows = 0;
        long long finalized_timeout = 0;
        long long finalized_duplicate_packets = 0;
        long long finalized_invalid_packets = 0;
        long long dcim_activation_quantized_elements = 0;
        long long dcim_dequantized_elements = 0;
        long long dcim_comparator_tree_passes = 0;
        long long y_dram_read_cycles = 0;
        long long y_dram_write_cycles = 0;
        long long z_dram_rw_cycles = 0;
        long long z_hbm_rw_bytes = 0;  // persistent-Z RMW for HBM-resident layers

        // Z SRAM accumulation (F-Cache time-multiplexed during CiM phase)
        bool z_sram_accum_used = false;
        long long z_sram_read_bytes = 0;
        long long z_sram_write_bytes = 0;
        long long z_sram_rw_bytes = 0;  // checksum = read + write
        void account_z_sram_bytes(long long read_bytes, long long write_bytes) {
            if (read_bytes < 0 || write_bytes < 0) {
                throw std::runtime_error("negative exact Z-SRAM byte count");
            }
            z_sram_read_bytes += read_bytes;
            z_sram_write_bytes += write_bytes;
            z_sram_rw_bytes += read_bytes + write_bytes;
            arch_ctrs.z_sram_read_bytes += static_cast<uint64_t>(read_bytes);
            arch_ctrs.z_sram_write_bytes += static_cast<uint64_t>(write_bytes);
            arch_ctrs.z_sram_rw_bytes += static_cast<uint64_t>(read_bytes + write_bytes);
        }

        // Row Detection Scanner (RDS): CSR row_ptr scan cost
        long long row_detect_cycles = 0;
        int row_detect_checks = 0;         // Total row_ptr entries compared
        int row_detect_sram_reads = 0;     // SRAM cache line reads for energy
        AccuracyMetrics accuracy{};
        YAccuracy y_accuracy{};

        // Layer-1 affected rows (exposed for Layer-2 propagation)
        std::vector<int> affected_rows;

        ArchAblationCounters arch_ctrs{};
    };

    // Reuse an existing PEArraySpMM instance (cycle-accurate, DRAMsim3, caches).
    // `pe_array` must outlive this controller.
    explicit DualModeSpMM(GlobalCounters* gcp, PEArraySpMM* pe_array);
    ~DualModeSpMM();

    void set_mode_from_string(const std::string& mode_str);
    void set_cim_mode_from_string(const std::string& mode_str);
    void set_phase_mode_from_string(const std::string& mode_str);
    void set_arch_variant_from_string(const std::string& variant_str);
    ArchVariant get_arch_variant() const { return arch_variant_; }
    const char* get_arch_variant_string() const { return arch_variant_to_string(arch_variant_); }
    const char* get_effective_backend_policy_string() const;
    const char* get_effective_indexer_mode_string() const;
#ifdef ENABLE_DRAMSIM3_PE
    dramsim3::CommandEnergySnapshot get_index_hbm_command_energy() const {
        return indexer_.index_hbm_command_energy();
    }
    dramsim3::CommandEnergySnapshot get_feature_hbm_command_energy() const {
        return indexer_.feature_hbm_command_energy();
    }
    dramsim3::CommandEnergySnapshot get_yz_hbm_command_energy() const {
        return yz_dram_ != nullptr
            ? yz_dram_->GetCommandEnergySnapshot()
            : dramsim3::CommandEnergySnapshot{};
    }
#endif
    void set_cim(NeuraCiM* cim, int d_out, CiMMode mode);
    CiMMode get_cim_mode() const { return cim_mode_; }
    PhaseMode get_phase_mode() const { return phase_mode_; }
    void set_threshold(double t) { threshold_ = t; }
    void set_indexer_mode(const std::string& mode) { indexer_mode_ = mode; }
    void set_compute_accuracy(bool v) { compute_accuracy_ = v; }
    void set_fail_closed_accuracy(bool v) { fail_closed_accuracy_ = v; }
    void set_cim_noise(bool v) { cim_noise_ = v; }
    void set_z_sram_accum(bool v) { z_sram_accum_ = v; }
    void set_dedicated_z_sram(bool v) { dedicated_z_sram_ = v; }
    void set_propagation_tau(double tau) { propagation_tau_ = tau; }
    double get_propagation_tau() const { return propagation_tau_; }
    void set_propagation_scope_enabled(bool v) { propagation_scope_enabled_ = v; }
    bool get_propagation_scope_enabled() const { return propagation_scope_enabled_; }
    void set_dedicated_z_sram_capacity_bytes(long long bytes) { dedicated_z_sram_capacity_bytes_ = bytes; }
    void load_weights(const std::string& path, int d_in, int d_out);
    const std::vector<float>& get_weights() const { return weights_; }
    const std::vector<float>& get_y_current() const { return y_current_; }
    const std::vector<float>& get_y2_current() const { return get_y_layer(2); }
    void set_y2_current(const std::vector<float>& y2) { set_y_layer(2, y2); }

    // Generic per-layer Y state (layer_id >= 2)
    const std::vector<float>& get_y_layer(int layer_id) const;
    void set_y_layer(int layer_id, const std::vector<float>& y);
    bool initialize_higher_layer_state(int layer_id,
                                       const WorkStealingScheduler::CSRMatrix& a_new,
                                       const std::vector<float>& h_prev,
                                       StepResult* t0_direct_accounting = nullptr);
    const std::vector<float>& get_f_current() const { return f_current_; }
    const std::vector<float>& get_z_current() const { return z_current_; }
    const std::vector<float>& get_z_layer(int layer_id) const;
    const std::vector<float>& get_h_layer(int layer_id) const;
    bool set_h_layer(int layer_id, const std::vector<float>& h);
    // NPF state traffic for one layer, in HBM: H_prev[v] is read for the rows
    // the caller must check exactly (changed rows whose distance bound may
    // exceed tau), and H'[v] is written for rows that pass ||dH||inf > tau
    // (rows below tau keep H_prev).
    struct NpfStateAccounting {
        long long h_read_rows = 0;
        long long h_write_rows = 0;
        long long h_hbm_read_bytes = 0;
        long long h_hbm_write_bytes = 0;
        long long h_hbm_cycles = 0;
    };
    NpfStateAccounting account_npf_h_state(int layer_id,
                                           const std::vector<int>& read_rows,
                                           const std::vector<int>& write_rows);
    // Distance-bound metadata: one FP32 bound per (layer, row), read and
    // written in HBM for every changed row of a tau-gated layer.
    NpfStateAccounting account_npf_bound_state(int layer_id, const std::vector<int>& rows);
    static bool build_norm_relu(const std::vector<float>& z,
                                int n,
                                bool apply_l2_norm,
                                std::vector<float>* h);
    struct RuntimeLayerTrace {
        bool executed = false;
        std::vector<int> graph_rows;
        std::vector<int8_t> quantized_inputs;
        std::vector<float> input_scales;
        std::vector<int8_t> quantized_weights;
        float weight_scale = 1.0f;
    };
    struct RuntimeActivationTrace {
        std::vector<float> norm_relu_output;
        std::vector<float> next_layer_delta;
        std::vector<int> next_layer_tau_rows;
    };
    void begin_runtime_timestep_trace();
    void set_runtime_activation_trace(int layer_id,
                                      const std::vector<float>& norm_relu_output,
                                      const std::vector<float>& next_layer_delta,
                                      const std::vector<int>& next_layer_tau_rows);
    const RuntimeLayerTrace* get_runtime_layer_trace(int layer_id) const;
    const RuntimeActivationTrace* get_runtime_activation_trace(int layer_id) const;
    const std::vector<float>& get_weights_for_layer(int layer_id) const {
        return weights_for_layer(layer_id);
    }
    StepResult run_full_from_existing_y(const WorkStealingScheduler::CSRMatrix& a_new,
                                        const std::vector<float>& y_full,
                                        const PEArraySpMM::Stats& pe_stats);

    // Initialize for dataset shape and (optional) feature init file.
    void init_from_loaded_graph(const PEArraySpMM::Stats& /*unused*/, int n, int nnz, int d);
    void load_feature_init(const std::string& f_init_path, int n, int d);

    // Run one timestep (for now single timestep driver; multi-timestep wiring later).
    StepResult run_once_full(const WorkStealingScheduler::CSRMatrix& a_new);
    StepResult run_once_incremental(const WorkStealingScheduler::CSRMatrix& a_new,
                                    const std::vector<DeltaConverter::DeltaAUpdate>& delta_a,
                                    const DeltaConverter::DeltaFParsed& delta_f);

    // Update FeatureStore: F_current += ΔF
    void apply_delta_f(const DeltaConverter::DeltaFParsed& df);

    const std::vector<float>& feature_store() const { return f_current_; }
    const std::vector<float>& y_current() const { return y_current_; }
    const std::vector<float>& z2_current() const { return z2_current_; }
    const std::vector<float>& z2_golden() const { return z2_golden_; }

    // =========================================================================
    // Multi-Layer GCN (2-Layer End-to-End)
    // =========================================================================

    struct LayerResult {
        int layer_id = 0;
        long long spmm_cycles = 0;
        long long cim_cycles = 0;
        long long activation_cycles = 0;
        double cim_energy_pj = 0.0;
        StepResult::AccuracyMetrics accuracy{};
    };

    struct MultiLayerResult {
        int num_layers = 0;
        std::vector<LayerResult> layers;
        long long total_cycles = 0;
        double total_energy_pj = 0.0;
        // Split CiM energy by stage:
        // - runtime: multi-timestep (online) stage before 2-layer eval
        // - eval: 2-layer end-to-end validation stage
        double runtime_cim_energy_pj = 0.0;
        double eval_cim_energy_pj = 0.0;
        StepResult::AccuracyMetrics final_accuracy{};
    };

    void load_weights_l2(const std::string& path, int d_in, int d_out);
    void load_weights_l3(const std::string& path, int d_in, int d_out);
    void load_weights_l4(const std::string& path, int d_in, int d_out);
    void load_weights_l5(const std::string& path, int d_in, int d_out);
    MultiLayerResult run_two_layer_gcn(
        const WorkStealingScheduler::CSRMatrix& a_new,
        bool export_z,
        const std::string& export_dir);
    MultiLayerResult run_multi_layer_gcn(
        int num_layers,
        const WorkStealingScheduler::CSRMatrix& a_new,
        bool export_z,
        const std::string& export_dir);

    // Run higher-layer incremental SpMM (3-phase), generic for layers 2, 3, 4, 5, ...:
    //   Phase 1: ΔA × H_{l-1}_new  (edge changes × current prev-layer activation)
    //   Phase 2: A_sub × ΔH_{l-1}  (column-filtered A × prev-layer feature delta)
    //   Phase 3: -ΔA × ΔH_{l-1}    (correction term, subtracted)
    //   Combine: accumulate phases into Y_l
    //   Materialize Y_l state for downstream layer propagation.
    // layer_id: 2 for Layer-2, 3 for Layer-3, etc.
    StepResult run_higher_layer_spmm(int layer_id,
                                     const WorkStealingScheduler::CSRMatrix& a_new,
                                     const std::vector<DeltaConverter::DeltaAUpdate>& delta_a,
                                     const std::vector<float>& h_prev_new,
                                     const std::vector<float>& delta_h_prev,
                                     const std::vector<int>& prev_affected_rows);

    // Backward compat wrappers
    StepResult run_layer2_spmm(const WorkStealingScheduler::CSRMatrix& a_new,
                               const std::vector<DeltaConverter::DeltaAUpdate>& delta_a,
                               const std::vector<float>& h1_new,
                               const std::vector<float>& delta_h,
                               const std::vector<int>& l1_affected_rows) {
        return run_higher_layer_spmm(2, a_new, delta_a, h1_new, delta_h, l1_affected_rows);
    }
    StepResult run_layer3_spmm(const WorkStealingScheduler::CSRMatrix& a_new,
                               const std::vector<DeltaConverter::DeltaAUpdate>& delta_a,
                               const std::vector<float>& h2_new,
                               const std::vector<float>& delta_h2,
                               const std::vector<int>& l2_affected_rows) {
        return run_higher_layer_spmm(3, a_new, delta_a, h2_new, delta_h2, l2_affected_rows);
    }

    static void export_binary(const std::string& path, const std::vector<float>& data) {
        std::ofstream out(path, std::ios::binary);
        out.write(reinterpret_cast<const char*>(data.data()),
                  static_cast<std::streamsize>(data.size() * sizeof(float)));
    }

private:
    GlobalCounters* gcp_;
    PEArraySpMM* pe_;
    DeltaFAwareIndexer indexer_;

    Mode mode_ = Mode::FULL;
    NeuraCiM* cim_ = nullptr;
    CiMMode cim_mode_ = CiMMode::NONE;
    PhaseMode phase_mode_ = PhaseMode::THREE_PHASE;
    ArchVariant arch_variant_ = ArchVariant::H6_FULL_DESPOINA;
    int d_out_ = 128;
    double threshold_ = 0.5;
    std::string indexer_mode_ = "indexstore"; // or "fullscan"
    double propagation_tau_ = 0.0;  // τ threshold for propagating set filtering
    bool propagation_scope_enabled_ = true;
    DynamicRowMeanOperator row_mean_operator_;
    DynamicRowMeanOperator::Transition row_mean_transition_;

    int n_ = 0;
    int d_ = 0;
    int nnz_ = 0;
    bool compute_accuracy_ = false;
    bool fail_closed_accuracy_ = false;
    bool cim_noise_ = false;
    bool z_sram_accum_ = true;  // Use F-Cache SRAM for Z accumulation
    bool dedicated_z_sram_ = false;  // Dedicated Z-SRAM: eliminates all Z-DRAM write-back
    long long dedicated_z_sram_capacity_bytes_ = -1;  // <=0: unlimited legacy behavior
    std::vector<float> weights_;

    // FeatureStore (F_current)
    std::vector<float> f_current_;
    // Output state (Y_prev / Y_current)
    std::vector<float> y_current_;
    // Combination output state (Z_prev / Z_current)
    std::vector<float> z_current_;
    std::vector<float> z_prev_;

    // Multi-layer GCN state
    std::vector<float> weights_l2_;
    std::vector<float> weights_l3_;
    std::vector<float> weights_l4_;
    std::vector<float> weights_l5_;
    int d_out_l2_ = 128;
    int d_out_l3_ = 128;
    int d_out_l4_ = 128;
    int d_out_l5_ = 128;
    std::vector<float> h1_current_;   // ReLU(Z1) — Layer 2 input
    std::map<int, std::vector<float>> y_layer_state_; // Per-layer Y state (layer_id >= 2)
    std::map<int, std::vector<float>> z_layer_state_; // Post-W state consumed by NPF/next layer
    // Paper-defined H state.  This is the sole cross-timestep owner used to
    // form dH; old Z must never be privately re-normalized at t=1+.
    std::map<int, std::vector<float>> h_layer_state_; // NormReLU(Z), layer_id >= 1
    std::map<int, RuntimeLayerTrace> runtime_layer_trace_;
    std::map<int, RuntimeActivationTrace> runtime_activation_trace_;
    despoina_r5::ResidentWeightContract resident_weight_contract_;
    std::map<int, uint64_t> observed_weight_hash_by_layer_;
    std::map<int, uint64_t> weight_version_by_layer_;
    bool pending_weight_program_event_ = false;
    uint64_t pending_weight_program_cycles_ = 0;
    uint64_t pending_weight_source_fp32_bytes_ = 0;
    uint64_t pending_weight_programmed_int8_bytes_ = 0;
    uint64_t pending_weight_program_tile_events_ = 0;
    uint64_t pending_weight_bank_prefetch_events_ = 0;
    uint64_t pending_weight_bank_swap_events_ = 0;
    uint64_t pending_weight_bank_exposed_stall_cycles_ = 0;
    std::vector<uint64_t> pending_weight_tile_program_done_cycles_;
    std::vector<float> z2_current_;   // Layer 2 CiM output
    std::vector<float> h_current_;    // Generic hidden state for multi-layer
    std::vector<float> z_final_;          // Final layer output
    std::vector<float> z_final_golden_;   // Final layer FP32 golden (N-layer)
    std::vector<float> z2_golden_;        // Layer 2 FP32 golden

    struct CimHardwareRunResult {
        bool ok = false;
        bool timeout = false;
        int expected_rows = 0;
        int received_unique_rows = 0;
        int duplicate_packets = 0;
        int invalid_packets = 0;
        long long cycles = 0;
        long long pe_to_cim_backpressure_cycles = 0;
        long long producer_first_ready_cycle = 0;
        long long producer_last_ready_cycle = 0;
        long long producer_last_enqueue_cycle = 0;
        long long native_first_issue_cycle = 0;
        long long native_last_completion_cycle = 0;
        long long overlap_cycles = 0;
        long long exposed_tail_cycles = 0;
        long long max_staging_rows = 0;
        long long staging_capacity_rows = 0;
        uint64_t pe_cim_noc_flits = 0;
        uint64_t pe_cim_noc_bytes = 0;
        long long row_mean_scaled_rows = 0;
        long long row_mean_changed_degree_rows = 0;
        long long row_mean_scale_elements = 0;
        long long row_mean_residual_correction_elements = 0;
        long long row_mean_scalar_ops = 0;
        long long row_mean_stream_delay_cycles = 0;
        long long row_mean_degree_read_events = 0;
        long long row_mean_degree_read_bytes = 0;
        long long row_mean_reciprocal_events = 0;
        NeuraCiM::PipelineCounters cim_pipeline{};
        // Streaming path: output holds Q(dY)·W per ordinal and the committed
        // Z row is Z_old * z_old_scale[ordinal] + output[ordinal].
        std::vector<float> output;
        std::vector<float> z_old_scale;
    };

    struct LinearStreamPhase {
        const WorkStealingScheduler::CSRMatrix* csr = nullptr;
        const float* f_matrix = nullptr;
        float y_scale = 1.0f;
        int expected_rows = 0;
        bool keep_shared_f_cache = false;
        uint64_t matrix_epoch = 0;
    };

    struct DigitalBackendConfig {
        int tile_rows = 16;
        int tile_cols = 16;
        int input_fifo_rows = 64;
        int output_fifo_rows = 64;
        int ingress_bytes_per_cycle = 512;
        int output_bytes_per_cycle = 512;
    };

    struct DigitalStreamStats {
        long long measured_cycles = 0;
        long long compute_cycles = 0;
        long long overlap_cycles = 0;
        long long fifo_bytes = 0;
        long long input_fifo_occupancy_cycles = 0;
        long long output_fifo_occupancy_cycles = 0;
        long long backpressure_cycles = 0;
        long long rows = 0;
        long long mac_ops = 0;
        int tile_rows = 0;
        int tile_cols = 0;
        int lanes = 0;
    };

    // Helpers
    static Mode parse_mode(const std::string& s);
    static CiMMode parse_cim_mode(const std::string& s);
    static PhaseMode parse_phase_mode(const std::string& s);
    static bool should_incremental(int k, int l, int nnz, double avg_deg, double threshold);
    static const char* phase_mode_to_string(PhaseMode mode);
    bool arch_forces_sparse_materialized() const;
    bool arch_uses_systolic_estimator() const;
    bool arch_uses_digital_stream_backend() const;
    bool arch_collects_dcim_counters() const;
    CiMMode effective_cim_mode_for_arch() const;
    std::string effective_indexer_mode_for_arch() const;
    DigitalStreamStats simulate_digital_stream_backend(
        long long producer_cycles,
        int rows,
        int d_in,
        int d_out,
        const DigitalBackendConfig& cfg) const;
    void apply_digital_stream_backend_accounting(StepResult& r,
                                                 long long producer_cycles,
                                                 int rows,
                                                 int d_in,
                                                 int d_out,
                                                 long long z_materialize_cycles);
    void populate_arch_ablation_counters(StepResult& r,
                                         bool y_materialized,
                                         bool z_materialized,
                                         bool higher_layer_materialization_estimate,
                                         int active_phase_count,
                                         int y_materialized_rows = -1,
                                         int materialization_eliminated_rows = 0) const;
    // Persistent Z of every layer is the accumulation state.  Layers are
    // allocated to the Post-Combination Buffer in order (L1, L2, ...) while
    // whole layers fit; the remaining layers live in HBM.
    bool z_layer_resident_in_post_combination_buffer(int layer_id) const;
    void account_z_state_rmw(int layer_id, const std::vector<int>& rows, StepResult& result);

    bool dedicated_z_sram_can_hold(long long z_bytes) const {
        if (!dedicated_z_sram_) {
            return false;
        }
        if (dedicated_z_sram_capacity_bytes_ <= 0) {
            return true;
        }
        return z_bytes <= dedicated_z_sram_capacity_bytes_;
    }

    WorkStealingScheduler::CSRMatrix fullscan_extract(const WorkStealingScheduler::CSRMatrix& a_new,
                                                      const std::vector<int>& changed_nodes) const;

    // Execute one PEArraySpMM job and return stats. (cycle-accurate)
    struct PhaseCapture {
        PEArraySpMM::Stats stats{};
        std::unordered_map<int, long long> row_ready_cycle;
    };
    PEArraySpMM::Stats run_pe_job(const WorkStealingScheduler::CSRMatrix& csr,
                                  const float* f_matrix,
                                  bool keep_shared_f_cache = false,
                                  uint64_t matrix_epoch = 0);
    PhaseCapture run_pe_job_capture(const WorkStealingScheduler::CSRMatrix& csr,
                                    const float* f_matrix,
                                    bool keep_shared_f_cache,
                                    uint64_t matrix_epoch);
    const std::vector<float>& weights_for_layer(int layer_id) const;
    OfficialDcimWeightBankKey official_dcim_weight_key_for_layer(int layer_id);
    void official_dcim_schedule_layer_ahead_prefetch(int current_layer_id);
    bool load_resident_weights_for_layer(int layer_id);
    bool record_direct_dcim_accounting(
        const std::vector<uint64_t>& row_ready_cycles,
        long long producer_cycles,
        const CimHardwareRunResult& hw,
        StepResult& result);
    bool capture_runtime_layer_trace(int layer_id,
                                     const std::vector<int>& graph_rows);
    bool commit_finalized_dcim_rows(int layer_id,
                                    const std::vector<int>& finalized_rows,
                                    const std::vector<uint64_t>& row_ready_cycles,
                                    long long producer_cycles,
                                    StepResult& result,
                                    const CimHardwareRunResult* concurrent_hw = nullptr);
    uint64_t next_cache_epoch_ = 1;
    std::vector<int> collect_nonzero_rows(const WorkStealingScheduler::CSRMatrix& csr) const;
    void run_full_cim_validation(const WorkStealingScheduler::CSRMatrix& a_new, StepResult& r);
    CimHardwareRunResult run_cim_hardware_rows(const std::vector<float>& Y,
                                               const std::vector<int>& rows);
    CimHardwareRunResult run_cim_hardware_rows_causal(
        const std::vector<float>& Y,
        const std::vector<int>& rows,
        const std::vector<uint64_t>& producer_ready_cycles);
    CimHardwareRunResult run_cim_hardware_streaming_phases(
        const std::vector<LinearStreamPhase>& phases,
        const std::vector<int>& unique_rows,
        std::vector<float>* y_state,
        long long* cim_cycles,
        std::vector<PEArraySpMM::Stats>* phase_stats);
    void account_row_mean_stream_result(const CimHardwareRunResult& hw,
                                        StepResult* result) const;
    void account_factorized_native_owners(const std::vector<int>& rows,
                                          StepResult* result);
    std::vector<float> golden_full_forward_rows(
        const WorkStealingScheduler::CSRMatrix& a_new,
        const std::vector<int>& rows) const;
    std::vector<float> golden_full_spmm(const WorkStealingScheduler::CSRMatrix& a_new) const;
    std::vector<float> golden_full_spmm_rows(
        const WorkStealingScheduler::CSRMatrix& a_new,
        const std::vector<int>& rows) const;
    std::vector<float> golden_dense_matmul_rows(const std::vector<float>& Y,
                                                const std::vector<int>& rows) const;
    StepResult::AccuracyMetrics compute_accuracy_metrics(
        const std::vector<float>& cim_output,
        const std::vector<float>& golden,
        const CimHardwareRunResult& hw) const;
    StepResult::YAccuracy compute_y_accuracy_metrics(
        const std::vector<float>& y_eval,
        const std::vector<float>& y_golden) const;
    void apply_cim_noise(std::vector<float>& output, int rows, int cols) const;

#ifdef ENABLE_DRAMSIM3_PE
    void ensure_yz_dramsim3_enabled();
    void flush_yz_dramsim3_stats();
    void yz_dram_read_cb(uint64_t addr);
    void yz_dram_write_cb(uint64_t addr);

    long long issue_yz_cacheline_batch(const std::vector<uint64_t>& cachelines, bool is_write);
    long long issue_yz_all_rows(uint64_t base_addr, int rows, int dim, bool is_write);
    long long issue_yz_selected_rows(uint64_t base_addr,
                                     const std::vector<int>& rows,
                                     int dim,
                                     bool is_write);

    dramsim3::MemorySystem* yz_dram_ = nullptr;
    std::unordered_map<uint64_t, int> yz_inflight_;
    std::string yz_dram_output_dir_;
    
    // Clock domain ratio for DRAMsim3 tick adjustment
    double yz_dram_clock_ratio_ = 1.0;
    double yz_dram_tick_accumulator_ = 0.0;
#endif
};

#endif  // DUAL_MODE_SPMM_H
