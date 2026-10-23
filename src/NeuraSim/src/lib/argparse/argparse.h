// SPDX-License-Identifier: CC-BY-4.0
// Derived from NeuraChip/NeuraSim (DOI: 10.5281/zenodo.10896280); modified for Despoina and this standalone release in 2026.
#ifndef ARG_PARSE_H
#define ARG_PARSE_H

#include <string>
#include <iostream>
#include <boost/program_options.hpp>
#include <boost/program_options/parsers.hpp>
#include <math.h>
#include "../pe/arch_variant.h"

using namespace std;

class ArgParse {
public:
	ArgParse(int argc, char *argv[]);
	~ArgParse();
	void parse(int argc, char *argv[]);
	void print();
	string get_exp_name() { return exp_name; }
	string get_program_root() { return program_root; }
	string get_description() { return description; }
	string get_dataset() { return dataset; }
	string get_dataset_root() { return dataset_root; }
	string get_profile() const { return profile; }
	bool is_official_profile() const {
		return profile == "despoina-h6-dcim-final";
	}
	bool get_experimental() const { return experimental; }

	// =============================================================================
	// DyGNN PE Array mode options (FULL / INCREMENTAL / SCHEDULED)
	// =============================================================================
	// Mode: "full" | "incremental" | "scheduled"
	string get_spmm_mode() const { return spmm_mode; }
	double get_mode_threshold() const { return mode_threshold; }
	string get_f_init_path() const { return f_init_path; }
	bool get_compute_accuracy() const { return compute_accuracy; }
	string get_w_init_path() const { return w_init_path; }
	bool get_cim_noise() const { return cim_noise; }
	string get_delta_a_path() const { return delta_a_path; }
	string get_delta_f_path() const { return delta_f_path; }
	string get_indexer_mode() const { return indexer_mode; }
	string get_cim_mode() const { return cim_mode; }
	string get_phase_mode() const { return phase_mode; }
	string get_arch_variant() const { return arch_variant; }
	int get_timesteps() const { return timesteps; }
	string get_delta_dir() const { return delta_dir; }
	string get_memory_type() const { return memory_type; }
	string get_dram_config_path() const { return dram_config_path; }
	double get_memory_bw_gbps() const { return memory_bw_gbps; }
	int get_memory_latency_cycles() const { return memory_latency_cycles; }
	double get_memory_tck_ns() const { return memory_tck_ns; }
	double get_memory_hit_pj_per_txn() const { return memory_hit_pj_per_txn; }
	double get_memory_miss_pj_per_txn() const { return memory_miss_pj_per_txn; }
	bool get_cxl_mode() const { return cxl_mode; }
	double get_cxl_link_bw_gbps() const { return cxl_link_bw_gbps; }
	int get_cxl_extra_latency_cycles() const { return cxl_extra_latency_cycles; }
	bool get_force_incremental_after_warmup() const { return force_incremental_after_warmup; }
	bool get_skip_warmup_sim() const { return skip_warmup_sim; }

	// DCIM-faithful accuracy + multi-layer GCN
	bool get_dcim_accuracy() const { return dcim_accuracy; }
	bool get_dcim_digital() const { return dcim_digital; }
	int get_num_layers() const { return num_layers; }
	double get_propagation_tau() const { return propagation_tau; }
	bool get_no_layer_norm() const { return no_layer_norm; }
	string get_scope_mode() const { return scope_mode; }
	string get_layer_norm_mode() const { return layer_norm_mode; }
	bool get_propagation_scope_enabled() const { return scope_mode != "none"; }
	bool get_effective_layer_norm_enabled() const {
		if (layer_norm_mode == "on") return !no_layer_norm;
		if (layer_norm_mode == "off") return false;
		return (propagation_tau > 0.0) && !no_layer_norm;
	}
	string get_w_init_l1_path() const { return w_init_l1_path; }
	string get_w_init_l2_path() const { return w_init_l2_path; }
	string get_w_init_l3_path() const { return w_init_l3_path; }
	string get_w_init_l4_path() const { return w_init_l4_path; }
	string get_w_init_l5_path() const { return w_init_l5_path; }
	string get_isolation_mode() const { return isolation_mode; }
	double get_nl_threshold_override() const { return nl_threshold_override; }
	bool get_export_z() const { return export_z; }
	bool get_export_z_per_timestep() const { return export_z_per_timestep; }
	string get_export_z_timestep_list() const { return export_z_timestep_list; }
	bool get_dedicated_z_sram() const { return dedicated_z_sram; }
	double get_dedicated_z_sram_size_mb() const { return dedicated_z_sram_size_mb; }
	uint32_t get_variation_seed() const { return variation_seed; }
	int get_bits_per_cycle() const { return bits_per_cycle; }
	int get_cim_fifo_depth() const { return cim_fifo_depth; }
		int get_adc_bits() const { return adc_bits; }
		bool get_fp_activations() const { return fp_activations; }
		bool get_no_pv() const { return no_pv; }
		bool get_no_thermal() const { return no_thermal; }
		bool get_no_pv_comp() const { return no_pv_comp; }
		double get_thermal_sigma_mv() const { return thermal_sigma_mv; }
		double get_thermal_sigma_scale() const { return thermal_sigma_scale; }
		double get_temperature_k() const { return temperature_k; }
		string get_thermal_model() const { return thermal_model; }
		double get_thermal_bl_cap_ff() const { return thermal_bl_cap_ff; }
		double get_thermal_adc_comp_sigma_mv() const { return thermal_adc_comp_sigma_mv; }
		string get_act_scale_mode() const { return act_scale_mode; }
		int get_act_calib_rows() const { return act_calib_rows; }
		double get_act_calib_clip() const { return act_calib_clip; }

	int get_tile_size() {
		// if tile size is 4, 16 or 64, convert to int and return square root of it in int
		if (tile_size == "4" || tile_size == "16" || tile_size == "64") {
			return (int)sqrt(stoi(tile_size));
		}
		// if tile size is Tile 4, Tile 16 or Tile 64, convert to int and return square root of it in int
		else if (tile_size == "Tile 4" || tile_size == "Tile 16" || tile_size == "Tile 64") {
			return (int)sqrt(stoi(tile_size.substr(5)));
		}
		else {
			cout << "ERROR: Invalid tile size" << endl;
			return stoi(tile_size);
		}
	}
	int get_stride() { return stride; }
	int get_cache_lines_count() { return stoi(cache_lines_count); }
	int get_pipeline_count() { return stoi(pipeline_count); }
	int get_comparator_count() { return stoi(comparator_count); }
private:
	string profile;
	bool experimental;
	string exp_name;
	string program_root;
	string description;
	string dataset;
	string dataset_root;
	string tile_size;
	string pipeline_count;
	int stride;
	string cache_lines_count;
	string comparator_count;

	// DyGNN PE Array mode (used in USE_PE_ARRAY_MODE path)
	string spmm_mode;
	double mode_threshold;
	string f_init_path;
	bool compute_accuracy;
	string w_init_path;
	bool cim_noise;
	string delta_a_path;
	string delta_f_path;
	string indexer_mode;
	string cim_mode;
	string phase_mode;
	string arch_variant;
	int timesteps;
	string delta_dir;
	string memory_type;
	string dram_config_path;
	double memory_bw_gbps;
	int memory_latency_cycles;
	double memory_tck_ns;
	double memory_hit_pj_per_txn;
	double memory_miss_pj_per_txn;
	bool cxl_mode;
	double cxl_link_bw_gbps;
	int cxl_extra_latency_cycles;
	double cxl_extra_latency_ns;
	bool force_incremental_after_warmup;
	bool skip_warmup_sim;

	// Depth-selectivity ablation
	string isolation_mode;
	double nl_threshold_override;

	// DCIM-faithful accuracy + multi-layer GCN
	bool dcim_accuracy;
	bool dcim_digital;
	int num_layers;
	double propagation_tau;
	bool no_layer_norm;
	string scope_mode;
	string layer_norm_mode;
	string w_init_l1_path;
	string w_init_l2_path;
	string w_init_l3_path;
	string w_init_l4_path;
	string w_init_l5_path;
	bool export_z;
	bool export_z_per_timestep;
	string export_z_timestep_list;
	bool dedicated_z_sram;
	double dedicated_z_sram_size_mb;
	uint32_t variation_seed;
		int bits_per_cycle;
		int cim_fifo_depth;
		int adc_bits;
		bool fp_activations;
		bool no_pv;
		bool no_thermal;
		bool no_pv_comp;
		double thermal_sigma_mv;
		double thermal_sigma_scale;
		double temperature_k;
		string thermal_model;
		double thermal_bl_cap_ff;
		double thermal_adc_comp_sigma_mv;
		string act_scale_mode;
		int act_calib_rows;
		double act_calib_clip;
		int read_avg_n;

public:
	int get_read_avg_n() const { return read_avg_n; }

};

#endif // !ARG_PARSE_H
