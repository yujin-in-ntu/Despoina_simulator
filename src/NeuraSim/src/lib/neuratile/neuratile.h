// SPDX-License-Identifier: CC-BY-4.0
// Derived from NeuraChip/NeuraSim (DOI: 10.5281/zenodo.10896280); modified for Despoina and this standalone release in 2026.
#ifndef NEURA_TILE_H
#define NEURA_TILE_H

#include <iostream>
#include <queue>
#include <deque>
#include <unordered_map>
#include "../counters/globalcounters.h"
#include "../neuracore/neuracore.h"
#include "../neuramem/neuramem.h"
#include "../neurarouter/neurarouter.h"
#include "../memreq/memrequest.h"
#include "../functions.h"
#include <map>
#include <vector>
#include "../connector/conn.h"
#include "../dispatcher/dispatcher.h"
#include "../mc/memorycontroller.h"
#include "../writer/writer.h"
#include "../result_sink/result_sink.h"

// CiM Integration (optional, compile with -DENABLE_CIM)
#ifdef ENABLE_CIM
#include "../neuracim/neuracim.h"
#endif

// PE Array SpMM (our DyGNN design, compile with -DENABLE_PE_ARRAY)
#ifdef ENABLE_PE_ARRAY
#include "../pe/pe_array_spmm.h"
#include "../pe/dual_mode_spmm.h"
#endif

// Activation Unit (compile with -DENABLE_ACTIVATION)
#ifdef ENABLE_ACTIVATION
#include "../activation/activation.h"
#endif

// H_agg Buffer (compile with -DENABLE_HAGG_BUFFER)
#ifdef ENABLE_HAGG_BUFFER
#include "../buffer/sram_buffer.h"
#endif

using namespace std;

class StreamingNoCEndpoint;
struct StreamingNoCPacket;

// A single NeuraTile
class NeuraTile {
	public:
		// Constructor
		// @param id: NeuraTile ID
		// @param x_cood: NeuraTile X Coordinate
		// @param y_cood: NeuraTile Y Coordinate
		NeuraTile(GlobalCounters *gcp, int x_cood, int y_cood, int port_count);
		NeuraTile();
		void custom_constructor(GlobalCounters *gcp, int x, int y, int port_count);
		~NeuraTile();
		void print();
		bool tick();
		void statistics();
		bool sanity_check();
		// In Port array
		queue<InstX *> *in_port_array;
		// Out Port array
		queue<InstX *> *out_port_array;
		queue<InstX *> *get_in_port_ref(int port_id);
		queue<InstX *> *get_out_port_ref(int port_id);
		string get_vertex_name();
		bool get_sim_end();
		void print_sim_end();
		void enable_max_buffer_sizes_computation();
		void post_sim_metrics();
		void self_post_sim_metrics();
		void interval_metrics();
		void self_interval_metrics();
		void set_sim_start_time();
		bool return_routine(bool sim_end, bool computed);
		int get_id();
		// int get_uid();
		GlobalCounters *get_gcp();
		void connect_units(string unit1, int port1, string unit2, int port2);
		void generate_cgra_mesh_topology();
		void create_unit_components();
		void create_arch_graph_distances_for_all_units();
		void print_unit_distances();
		void smart_stats();
		string vertex_name;
	private:
		int id;
		// NOTE: Not allocating UID to neura tile
		// Because you dont have to send data to entire tile but spcific units on the tile.
		// So not adding Neura Tile to the graph.
		// To match the vertex descriptions to UIDs, not adding UID to NeuraTile
		// int uid;
		GlobalCounters *gcp;
		int x_cood;
		int y_cood;
		int core_x_count;
		int core_y_count;
		int mem_x_count;
		int mem_y_count;
		int mem_controller_count;
		int router_x_count;
		int router_y_count;
		int writer_count;
		int port_count;
		bool sim_end;
		NeuraCore **core_array;
		NeuraMem **mem_array;
		NeuraRouter **router_array;
		MemoryController *mem_controller_array;
		Writer *writer_array;
		vector<Conn *> conn_array;
		NeuraDispatcher dispatcher;
		
		// CiM Unit for Dense MatMul (Combination step)
#ifdef ENABLE_CIM
		NeuraCiM *cim_unit;
		bool cim_enabled;
		bool spmm_done;      // SpMM phase completed
		bool cim_started;    // CiM phase started
		int cim_d_in;        // CiM input dimension (= mat_B y_dim)
		int cim_d_out;       // CiM output dimension (= hidden layer size)
		void init_cim(int d_in, int d_out);
		void start_cim_phase();  // Start CiM after SpMM
		void tick_cim();
		bool is_cim_done() const;
		long long cim_start_cycle;  // Cycle when CiM started
		long long cim_weight_load_cycles;
		long long cim_weight_load_dram_requests;
		long long cim_weight_load_bytes;
#endif

		// PE Array SpMM (our DyGNN design)
#ifdef ENABLE_PE_ARRAY
		PEArraySpMM *pe_array_spmm;
		bool pe_array_enabled;
		bool pe_array_initialized;
		void init_pe_array();
		void tick_pe_array();
		bool is_pe_array_done() const;
		void print_pe_array_stats();
#ifdef USE_PE_ARRAY_MODE
		// PE Array standalone mode (replaces NeuraCore)
		bool load_pe_array_data();
		bool tick_pe_array_mode();
		long long pe_array_start_cycle;
		long long pe_array_cycles;  // Final SpMM cycles (saved for E2E calculation)
		int pe_array_n;
		int pe_array_nnz;
		int pe_array_d;
		int pe_array_expected_rows; // Non-empty A rows that actually produce PE outputs
		std::vector<int> pe_array_row_ptr;
		std::vector<int> pe_array_col_idx;
		std::vector<float> pe_array_values;
		std::vector<float> pe_array_F;  // F matrix storage (must outlive PE Array)

		// Despoina factorized row-mean controller (FULL/INCR/SCHED)
		DualModeSpMM* dygnn_controller;
		// NPF distance bound: per layer, the last NormReLU(Z) seen for each row
		// and an upper bound on ||NormReLU(Z) - H_prev||_inf, so H_prev reads
		// can be skipped without changing any propagation decision.
		std::map<int, std::vector<float>> npf_g_last_;
		std::map<int, std::vector<float>> npf_bound_;

		// Multi-timestep driver (t=0 full, t=1..N-1 incremental)
		int multi_ts_current;
		int multi_ts_total;
		bool multi_ts_done;
		bool multi_ts_full_done;
		long long multi_ts_full_cycles;
		double multi_ts_runtime_cim_energy_pj;
		double multi_ts_eval_cim_energy_pj;
		bool multi_ts_eval_energy_valid;
		struct TimestepResult {
			int timestep;
			DualModeSpMM::Mode mode_used;
			PhaseMode phase_mode_used = PhaseMode::THREE_PHASE;
			long long cycles;
			std::vector<long long> higher_layer_cycles; // index 0 = Layer-2, 1 = Layer-3, etc.
			// Depth-selectivity ablation: per-layer propagation tracking
			std::vector<int> per_layer_propagating_rows;  // index 0 = L1 affected, 1 = L2 propagating, ...
			std::vector<long long> per_layer_mac_ops;     // per-layer MAC ops
			PEArraySpMM::Stats pe_stats;
			DualModeSpMM::StepResult::PerPhaseInfo phase1{};
			DualModeSpMM::StepResult::PerPhaseInfo phase2{};
			DualModeSpMM::StepResult::PerPhaseInfo phase3{};
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
			long long npf_h_read_rows = 0;
			long long npf_h_write_rows = 0;
			long long npf_h_hbm_read_bytes = 0;
			long long npf_h_hbm_write_bytes = 0;
			long long npf_h_hbm_cycles = 0;
			long long z_hbm_rw_bytes = 0;
			double factorized_pe_dynamic_energy_pj = 0.0;
			DeltaFAwareIndexer::Counters indexer_ctrs;
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
			long long dcim_compute_cycles = 0;
			long long tile_busy_work_cycles = 0;
			long long causal_stream_elapsed_cycles = 0;
			long long exposed_tail_cycles = 0;
			long long dcim_streams = 0;
			long long dcim_drain_sensitivity_cycles = 0;
			long long dcim_compute_only_exposed_cycles = 0;
			long long dcim_plus6_exposed_cycles = 0;
			long long finalized_duplicate_packets = 0;
			long long finalized_invalid_packets = 0;
			long long finalized_timeouts = 0;
			long long dcim_activation_quantized_elements = 0;
			long long dcim_dequantized_elements = 0;
			long long dcim_comparator_tree_passes = 0;
			long long y_dram_read_cycles = 0;
			long long y_dram_write_cycles = 0;
			long long z_dram_rw_cycles = 0;
			long long z_sram_read_bytes = 0;
			long long z_sram_write_bytes = 0;
			long long z_sram_rw_bytes = 0;
			long long row_detect_cycles = 0;
			int row_detect_checks = 0;
			int row_detect_sram_reads = 0;
			DualModeSpMM::StepResult::ArchAblationCounters arch_ctrs{};
			DualModeSpMM::StepResult::AccuracyMetrics accuracy{};
			DualModeSpMM::StepResult::YAccuracy y_accuracy{};
			DualModeSpMM::StepResult::AccuracyMetrics e2e_accuracy{};
			bool e2e_exported = false;
			std::string e2e_export_dir;
#ifdef ENABLE_DRAMSIM3_PE
			dramsim3::CommandEnergySnapshot hbm_pe_cumulative;
			dramsim3::CommandEnergySnapshot hbm_index_cumulative;
			dramsim3::CommandEnergySnapshot hbm_feature_cumulative;
			dramsim3::CommandEnergySnapshot hbm_yz_cumulative;
			dramsim3::CommandEnergySnapshot hbm_weight_cumulative;
#endif
		};
		std::vector<TimestepResult> multi_ts_results;
		void write_multi_ts_stats();
		void capture_exact_hbm_snapshot(TimestepResult& result);
		void write_exact_hbm_sidecar();
		void write_arch_ablation_stats();
		
		// Row-wise Streaming: PE Array → CiM (always enabled, no memory path)
			int streaming_rows_completed;
			int streaming_outputs_completed;
			long long streaming_cim_cycles;
			void tick_streaming_cim();  // Tick CiM in streaming mode
			bool setup_streaming_noc_endpoints();
			void tick_streaming_noc_fabric();
			bool can_inject_streaming_row(int feature_dim);
			bool inject_pe_to_cim_packet(int row_id, const float* row_data, int feature_dim);
			bool can_inject_cim_output(const CiMOutputPacket& output);
			bool inject_cim_to_pe_packet(const CiMOutputPacket& output);
			void drain_streaming_cim_endpoint();
			void drain_streaming_pe_endpoint();
			bool streaming_noc_quiesced();
			bool is_router_extra_port_in_use(int router_x, int router_y);
			int streaming_pe_router_x;
			int streaming_pe_router_y;
			int streaming_cim_router_x;
			int streaming_cim_router_y;
			StreamingNoCEndpoint* streaming_pe_endpoint;
			StreamingNoCEndpoint* streaming_cim_endpoint;
			std::unordered_map<int, std::vector<float>> streaming_cim_partial_rows;
			std::unordered_map<int, int> streaming_cim_partial_flits;
			std::unordered_map<int, int> streaming_cim_expected_flits;
			std::deque<int> streaming_cim_ready_rows;
			std::unordered_map<int, std::vector<float>> streaming_pe_partial_rows;
			std::unordered_map<int, int> streaming_pe_partial_flits;
			std::unordered_map<int, int> streaming_pe_expected_flits;
			std::deque<int> streaming_pe_ready_rows;
			std::deque<CiMOutputPacket> streaming_pending_outputs;
			long long pe_cim_noc_packets;
			long long pe_cim_noc_flits;
			long long pe_cim_noc_latency_cycles;
			long long pe_cim_noc_output_packets;
#endif
#endif

		// Activation Unit
#ifdef ENABLE_ACTIVATION
		ActivationUnit *activation_unit;
		bool activation_enabled;
		bool activation_started;
		bool activation_done;
		long long activation_start_cycle;
		long long activation_cycles;
		void init_activation(ActivationType type = ActivationType::RELU);
		void start_activation_phase(int n, int d);
		bool is_activation_done() const;
#endif

		// H_agg Buffer (SpMM output → CiM input)
#ifdef ENABLE_HAGG_BUFFER
		SRAMBuffer *hagg_buffer;
		bool hagg_buffer_enabled;
		long long hagg_buffer_stall_cycles;
		long long hagg_buffer_write_cycles;  // Total write latency cycles
		long long hagg_buffer_read_cycles;   // Total read latency cycles
		long long hagg_noc_cycles;           // Total NoC transfer cycles (model)
		double hagg_sram_energy_pj;          // CACTI-based SRAM energy for H_agg buffering
		double hagg_noc_energy_pj;           // Accelergy-based NoC energy for H_agg transfers
		std::queue<int> hagg_row_id_queue;   // Track row IDs in buffer (SRAMBuffer doesn't store them)
		void init_hagg_buffer(
			int capacity_kb = static_cast<int>(pe_hagg_staging::kBytes / 1024u),
			bool double_buffer = false);
#endif
		
		string program_file;
		// Name of the component
		string component_name;
		// Name of the header component
		string header_component_name;
		// Collection
		ResultCollection collection;
		// Collection for the header
		ResultCollection header_collection;
		// Metrics
		long long idle_cycles;
		long long interval_idle_cycles;
		long long busy_cycles;
		long long interval_busy_cycles;
		long long interval_id;
		bool *component_sim_end_array;

};

#endif // !NEURA_TILE_H
