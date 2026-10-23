// SPDX-License-Identifier: CC-BY-4.0
// Derived from NeuraChip/NeuraSim (DOI: 10.5281/zenodo.10896280); modified for Despoina and this standalone release in 2026.
// Created by Kaustubh Shivdikar
//

#include "argparse.h"
#include "../debugger/debugger.h"
#include <cctype>
#include <set>

namespace po = boost::program_options;
using namespace std;

ArgParse::ArgParse(int argc, char *argv[])
{
	parse(argc, argv);
}

ArgParse::~ArgParse()
{
}

void ArgParse::parse(int argc, char *argv[])
{
	try {
		po::options_description desc("Allowed options");
		desc.add_options()
			("help,h", "produce help message")
			("profile", po::value<string>(&profile)->default_value(""),
			 "Official profile: despoina-h6-dcim-final")
			("experimental", po::bool_switch(&experimental)->default_value(false),
			 "Expose non-official architecture and analog/noise research controls")
			("exp_name,n", po::value<string>(&exp_name)->default_value("NeuraChip_V000"), "Experiment Name")
			("program_root,r", po::value<string>(&program_root)->default_value("../NeuraCompiler/programs"), "Program Root File")
			("description,d", po::value<string>(&description)->default_value("Default experiment description"), "Experiment Description")
			("dataset,x", po::value<string>(&dataset)->default_value("cora"), "Dataset")
			("tile_size,t", po::value<string>(&tile_size)->default_value("16"), "Tile Size")
			("stride,s", po::value<int>(&stride)->default_value(1), "Stride")
			("cache_lines_count,c", po::value<string>(&cache_lines_count)->default_value("16384"), "Cach Lines Count")
			("pipeline_count,p", po::value<string>(&pipeline_count)->default_value("128"), "Pipeline Count")
			("comparator_count,m", po::value<string>(&comparator_count)->default_value("16"), "Comparator Count")

			// =========================================================================
			// DyGNN PE Array mode options
			// =========================================================================
			("spmm-mode", po::value<string>(&spmm_mode)->default_value("full"),
			 "SpMM mode for PE Array path: full | incremental | scheduled")
			("mode-threshold", po::value<double>(&mode_threshold)->default_value(0.5),
			 "Scheduling threshold for scheduled mode (rho_e + rho_w < T => incremental)")
			("f-init", po::value<string>(&f_init_path)->default_value(""),
			 "t=0 feature initialization file for FeatureStore")
			("compute-accuracy", po::bool_switch(&compute_accuracy)->default_value(false),
			 "Enable hardware-path CiM accuracy measurement (CiM output vs CPU golden)")
			("w-init", po::value<string>(&w_init_path)->default_value(""),
			 "Legacy layer-1 weight binary file [d_in x d_out] row-major float32; no generated fallback")
			("cim-noise", po::bool_switch(&cim_noise)->default_value(false),
			 "Enable optional CiM noise injection for robustness test")
			("delta-a", po::value<string>(&delta_a_path)->default_value(""),
			 "Per-timestep delta-A file (required for incremental/scheduled)")
			("delta-f", po::value<string>(&delta_f_path)->default_value(""),
			 "Per-timestep delta-F file (required for incremental/scheduled; also updates FeatureStore in full)")
			("indexer-mode", po::value<string>(&indexer_mode)->default_value("indexstore"),
			 "Phase2(A×ΔF) edge filtering: indexstore | fullscan")
			("cim-mode", po::value<string>(&cim_mode)->default_value("none"),
			 "CiM mode for incremental combination: none | delta | full | linear")
			("phase-mode", po::value<string>(&phase_mode)->default_value("three_phase"),
			 "Incremental phase strategy: three_phase | edge_only | two_phase | anew_full")
			("arch-variant", po::value<string>(&arch_variant)->default_value(default_arch_variant_string()),
			 "Despoina architecture variant: H0_systolic_scoped | H1_sparsefe_digitalbe | H2_sparsefe_index_digitalbe | H3_sparsefe_index_digital_stream | H4_sparsefe_index_dcim | H5_sparsefe_index_dcim_stream | H6_full_despoina")
			("timesteps", po::value<int>(&timesteps)->default_value(1),
			 "Number of timesteps to run (t=0 is full, t=1..N-1 are incremental)")
			("delta-dir", po::value<string>(&delta_dir)->default_value(""),
			 "Directory containing per-timestep delta files (delta_a_t{i}.txt, delta_f_t{i}.txt)")
			("memory-type", po::value<string>(&memory_type)->default_value("hbm2"),
			 "Memory profile: hbm2 | hbm3 | hbm4 | cxl_ddr5 | cxl_ddr5_5600")
			("dram-config", po::value<string>(&dram_config_path)->default_value(""),
			 "DRAMSim3 .ini path override (if empty, chosen from --memory-type)")
			("memory-bw-gbps", po::value<double>(&memory_bw_gbps)->default_value(-1.0),
			 "Off-chip bandwidth override in GB/s")
			("memory-latency-cycles", po::value<int>(&memory_latency_cycles)->default_value(-1),
			 "Off-chip latency override in cycles")
			("memory-hit-pj-per-txn", po::value<double>(&memory_hit_pj_per_txn)->default_value(-1.0),
			 "Analytical row-buffer-hit DRAM energy override in pJ/64B txn")
			("memory-miss-pj-per-txn", po::value<double>(&memory_miss_pj_per_txn)->default_value(-1.0),
			 "Analytical row-buffer-miss DRAM energy override in pJ/64B txn")
			("cxl-link-bw-gbps", po::value<double>(&cxl_link_bw_gbps)->default_value(-1.0),
			 "CXL wrapper link BW cap in GB/s (only used for cxl_* memory-type)")
			("cxl-extra-latency-ns", po::value<double>(&cxl_extra_latency_ns)->default_value(-1.0),
			 "Extra CXL latency in ns added before DRAM transaction issue")
			("force-incremental-after-warmup", po::bool_switch(&force_incremental_after_warmup)->default_value(false),
			 "Force t=1..N-1 to stay incremental after the t=0 full warmup step (disables post-warmup FULL fallback)")
			("skip-warmup-sim", po::bool_switch(&skip_warmup_sim)->default_value(false),
			 "Skip t=0 cycle-accurate PE simulation: compute Y_t0=A*F analytically, then run t=1+ normally")

				// =========================================================================
				// DCIM-faithful accuracy + multi-layer GCN
				// =========================================================================
			("dcim-accuracy", po::bool_switch(&dcim_accuracy)->default_value(false),
			 "Enable ACIM-faithful CiM computation (bit-serial ADC + thermal noise + process variation)")
			("dcim-digital", po::bool_switch(&dcim_digital)->default_value(false),
			 "Enable DCIM (Digital CiM) computation: pure INT8 MAC, no ADC/DAC/analog noise")
			("isolation-mode", po::value<string>(&isolation_mode)->default_value("full"),
			 "Nonlinear isolation ablation: full | none | weak")
			("nl-threshold", po::value<double>(&nl_threshold_override)->default_value(-1.0),
			 "Nonlinear isolation threshold override (default -1 = use 1e-6 for full, 0 for weak)")
			("num-layers", po::value<int>(&num_layers)->default_value(1),
			 "Number of GCN layers (1 to 10)")
			("propagation-tau", po::value<double>(&propagation_tau)->default_value(0.0),
			 "Propagation threshold tau: rows with ||delta_H||_inf <= tau are filtered. 0=exact (default)")
			("scope-mode", po::value<string>(&scope_mode)->default_value("tau"),
			 "Propagation scoping mode: tau | none. tau preserves existing threshold filtering; none disables propagation row thresholding without changing backend path")
			("layer-norm-mode", po::value<string>(&layer_norm_mode)->default_value("auto"),
			 "Layer normalization mode: auto | on | off. auto preserves legacy behavior (on iff propagation_tau > 0 and --no-layer-norm is absent)")
			("no-layer-norm", po::bool_switch(&no_layer_norm)->default_value(false),
			 "Disable per-row L2 normalization between layers (norm is on by default when tau > 0)")
			("w-init-l1", po::value<string>(&w_init_l1_path)->default_value(""),
			 "Layer 1 weight binary file [d_in x d_hidden] row-major float32")
			("w-init-l2", po::value<string>(&w_init_l2_path)->default_value(""),
			 "Layer 2 weight binary file [d_hidden x d_out] row-major float32")
			("w-init-l3", po::value<string>(&w_init_l3_path)->default_value(""),
			 "Layer 3 weight binary file [d_hidden x d_hidden] row-major float32")
			("w-init-l4", po::value<string>(&w_init_l4_path)->default_value(""),
			 "Layer 4 weight binary file [d_hidden x d_hidden] row-major float32")
			("w-init-l5", po::value<string>(&w_init_l5_path)->default_value(""),
			 "Layer 5 weight binary file [d_hidden x d_out] row-major float32")
			("export-z", po::bool_switch(&export_z)->default_value(false),
			 "Export Z output matrices to binary files for Python consumption")
			("export-z-per-timestep", po::bool_switch(&export_z_per_timestep)->default_value(false),
			 "Export per-timestep 2-layer Z outputs under results_dir/per_timestep_z/t*/")
			("export-z-timestep-list", po::value<string>(&export_z_timestep_list)->default_value(""),
			 "Optional comma-separated timestep list for per-timestep export (e.g. 0,1,25,50,75,99). Empty means export all timesteps")
			("dedicated-z-sram", po::bool_switch(&dedicated_z_sram)->default_value(false),
			 "Use dedicated Z-SRAM: eliminates all Z-DRAM write-back, Z stays on-chip")
			("dedicated-z-sram-size-mb", po::value<double>(&dedicated_z_sram_size_mb)->default_value(-1.0),
			 "Dedicated Z-SRAM capacity in MB (<=0 keeps legacy unlimited behavior when --dedicated-z-sram is set)")
			("variation-seed", po::value<uint32_t>(&variation_seed)->default_value(42),
			 "Seed for reproducible per-cell process variation in DCIM mode")
			("bits-per-cycle", po::value<int>(&bits_per_cycle)->default_value(2),
			 "Input bit-serial parallelism in DCIM mode (bits processed per cycle, 1..8)")
			("cim-fifo-depth", po::value<int>(&cim_fifo_depth)->default_value(16),
			 "CiM input/output FIFO depth in rows for streaming/backpressure studies (default 16)")
			("adc-bits", po::value<int>(&adc_bits)->default_value(8),
			 "ADC resolution in bits for DCIM mode (default 8 for INT8 weights)")
			("read-avg-n", po::value<int>(&read_avg_n)->default_value(1),
			 "Number of ADC reads to average for thermal noise reduction (1=off, 2/4/8)")
			("fp-activations", po::bool_switch(&fp_activations)->default_value(false),
			 "Keep activations in FP32 (only quantize weights to INT8)")
			("no-pv", po::bool_switch(&no_pv)->default_value(false),
			 "Disable process variation noise (ablation study)")
			("no-thermal", po::bool_switch(&no_thermal)->default_value(false),
			 "Disable thermal read noise (ablation study)")
			("no-pv-comp", po::bool_switch(&no_pv_comp)->default_value(false),
			 "Disable PV write-verify compensation (accuracy-oriented calibration)")
			("thermal-sigma-mv", po::value<double>(&thermal_sigma_mv)->default_value(-1.0),
			 "Thermal noise sigma override in mV (default: use neurosim::noise::THERMAL_NOISE_MV)")
			("thermal-sigma-scale", po::value<double>(&thermal_sigma_scale)->default_value(1.0),
			 "Thermal sigma scale multiplier (default 1.0)")
			("temperature-k", po::value<double>(&temperature_k)->default_value(300.0),
			 "Temperature in Kelvin for thermal scaling (sigma scales by sqrt(T/300))")
			("thermal-model", po::value<string>(&thermal_model)->default_value("physics_ktc"),
			 "Thermal sigma source: legacy | physics_ktc (default: physics_ktc)")
			("thermal-bl-cap-ff", po::value<double>(&thermal_bl_cap_ff)->default_value(100.0),
			 "Bitline capacitance in fF for physics_ktc thermal model (default 100)")
			("thermal-adc-comp-sigma-mv", po::value<double>(&thermal_adc_comp_sigma_mv)->default_value(0.5),
			 "ADC comparator input-referred sigma in mV for physics_ktc thermal model (default 0.5)")
			("act-scale-mode", po::value<string>(&act_scale_mode)->default_value("fixed"),
			 "Activation quant scale mode: fixed | warmup_maxabs")
			("act-calib-rows", po::value<int>(&act_calib_rows)->default_value(64),
			 "Rows used for warmup activation scale calibration")
			("act-calib-clip", po::value<double>(&act_calib_clip)->default_value(1.0),
			 "Warmup activation scale clipping coefficient (default 1.0)")

			;

		po::variables_map vm;
		po::store(po::parse_command_line(argc, argv, desc), vm);
		po::notify(vm);
		this->dataset_root = this->program_root + "/" + this->dataset;

		if (vm.count("help")) {
			if (experimental) {
				cout << desc << "\n";
				exit(0);
			}
			cout
				<< "Despoina standalone simulator\n\n"
				<< "Official reviewer interface:\n"
				<< "  --profile despoina-h6-dcim-final (required)\n"
				<< "  --program_root PATH --dataset NAME --timesteps N --delta-dir PATH\n"
				<< "  --f-init PATH --w-init-l1 PATH --w-init-l2 PATH\n"
				<< "  [--propagation-tau VALUE] [--dram-config PATH]\n"
				<< "  [--exp_name NAME] [--description TEXT]\n\n"
				<< "The official profile fixes H6, incremental three-phase SpMM,\n"
				<< "IndexStore, linear DCIM, D=128, 16 lanes/FIFO=16, two layers,\n"
				<< "exact row-mean semantics, FP32/22nm energy, and final hashes.\n";
			exit(0);
		}

		if (!profile.empty() && profile != "despoina-h6-dcim-final") {
			cerr << "error: unknown --profile; official value is despoina-h6-dcim-final\n";
			exit(1);
		}
		if (profile.empty() && !experimental) {
			cerr << "error: select --profile despoina-h6-dcim-final; "
			     << "non-official controls require --experimental\n";
			exit(1);
		}
		if (profile == "despoina-h6-dcim-final") {
			const std::set<std::string> official_options = {
				"profile", "program_root", "dataset", "timesteps", "delta-dir",
				"f-init", "w-init-l1", "w-init-l2", "dram-config",
				"propagation-tau", "exp_name", "description"
			};
			for (const auto& entry : vm) {
				if (!entry.second.defaulted() &&
				    official_options.count(entry.first) == 0) {
					cerr << "error: --" << entry.first
					     << " is unavailable in the closed official profile\n";
					exit(1);
				}
			}
			spmm_mode = "incremental";
			indexer_mode = "indexstore";
			cim_mode = "linear";
			phase_mode = "three_phase";
			arch_variant = "H6_full_despoina";
			memory_type = "hbm2";
			force_incremental_after_warmup = true;
			compute_accuracy = true;
			dcim_digital = true;
			dcim_accuracy = false;
			cim_noise = false;
			isolation_mode = "full";
			num_layers = 2;
			scope_mode = "tau";
			layer_norm_mode = "auto";
			no_layer_norm = false;
			dedicated_z_sram = true;
			dedicated_z_sram_size_mb = 4.0;
			export_z_per_timestep = true;
			bits_per_cycle = 8;
			cim_fifo_depth = 16;
			adc_bits = 8;
			if (timesteps < 2 || f_init_path.empty() ||
			    w_init_l1_path.empty() || w_init_l2_path.empty() ||
			    delta_dir.empty() || vm["program_root"].defaulted() ||
			    vm["dataset"].defaulted()) {
				cerr << "error: official profile requires explicit --program_root, --dataset, "
				     << "--timesteps >= 2, --f-init, "
				     << "--w-init-l1, --w-init-l2, and --delta-dir\n";
				exit(1);
			}
		}

		// Basic validation (keep CLI UX friendly)
		if (skip_warmup_sim && timesteps < 2) {
			cerr << "error: --skip-warmup-sim requires --timesteps >= 2 (t=0 is skipped, t=1+ run normally)\n";
			exit(1);
		}
		if (skip_warmup_sim && spmm_mode == "full") {
			cerr << "error: --skip-warmup-sim requires --spmm-mode incremental|scheduled (not full)\n";
			exit(1);
		}
		if (spmm_mode != "full" && spmm_mode != "incremental" && spmm_mode != "scheduled") {
			cerr << "error: --spmm-mode must be one of: full | incremental | scheduled\n";
			exit(1);
		}
		if (indexer_mode != "indexstore" && indexer_mode != "fullscan") {
			cerr << "error: --indexer-mode must be one of: indexstore | fullscan\n";
			exit(1);
		}
			if (phase_mode != "three_phase" &&
			    phase_mode != "edge_only" &&
			    phase_mode != "two_phase" &&
			    phase_mode != "anew_full") {
				cerr << "error: --phase-mode must be one of: three_phase | edge_only | two_phase | anew_full\n";
				exit(1);
			}
			if (cim_mode != "none" && cim_mode != "delta" && cim_mode != "full" && cim_mode != "linear") {
				cerr << "error: --cim-mode must be one of: none | delta | full | linear\n";
				exit(1);
			}
			if (!is_valid_arch_variant(arch_variant)) {
				cerr << "error: --arch-variant must be one of: "
				     << "H0_systolic_scoped | H1_sparsefe_digitalbe | H2_sparsefe_index_digitalbe | "
				     << "H3_sparsefe_index_digital_stream | H4_sparsefe_index_dcim | "
				     << "H5_sparsefe_index_dcim_stream | H6_full_despoina "
				     << "(legacy aliases accepted: H1_sparse_pe, H2_metadata_indexstore, "
				     << "H3_dcim_no_stream, H4_dcim_stream, H5_full_despoina)\n";
				exit(1);
			}
			if (compute_accuracy && cim_mode == "none") {
				cerr << "error: --compute-accuracy requires --cim-mode to be one of: delta | full | linear\n";
				exit(1);
			}
			if (compute_accuracy &&
			    !(spmm_mode == "incremental" || spmm_mode == "scheduled")) {
				cerr << "error: --compute-accuracy currently supports --spmm-mode incremental|scheduled only\n";
				exit(1);
			}
			if (compute_accuracy && timesteps < 2) {
				cerr << "error: --compute-accuracy requires --timesteps >= 2 (t=0 full + incremental/scheduled updates)\n";
				exit(1);
			}
			if (mode_threshold <= 0.0 || mode_threshold >= 2.0) {
				cerr << "error: --mode-threshold should be in (0, 2) (typical 0.3~0.7)\n";
				exit(1);
			}
			if (dedicated_z_sram_size_mb < 0.0 && dedicated_z_sram_size_mb != -1.0) {
				cerr << "error: --dedicated-z-sram-size-mb must be -1(unlimited) or >= 0\n";
				exit(1);
			}
			if (num_layers < 1 || num_layers > 10) {
				cerr << "error: --num-layers must be 1 to 10\n";
				exit(1);
			}
			if (isolation_mode != "full" && isolation_mode != "none" && isolation_mode != "weak") {
				cerr << "error: --isolation-mode must be one of: full | none | weak\n";
				exit(1);
			}
			if (scope_mode != "tau" && scope_mode != "none") {
				cerr << "error: --scope-mode must be one of: tau | none\n";
				exit(1);
			}
			if (layer_norm_mode != "auto" && layer_norm_mode != "on" && layer_norm_mode != "off") {
				cerr << "error: --layer-norm-mode must be one of: auto | on | off\n";
				exit(1);
			}
			if (export_z_per_timestep && num_layers < 2) {
				cerr << "error: --export-z-per-timestep requires --num-layers >= 2\n";
				exit(1);
			}
			if (!export_z_timestep_list.empty() && !export_z_per_timestep) {
				cerr << "error: --export-z-timestep-list requires --export-z-per-timestep\n";
				exit(1);
			}
			if (dcim_accuracy && dcim_digital) {
				cerr << "error: --dcim-accuracy and --dcim-digital are mutually exclusive\n";
				exit(1);
			}
			if (dcim_accuracy && cim_mode == "none") {
				cerr << "error: --dcim-accuracy requires --cim-mode != none\n";
				exit(1);
			}
			if (dcim_accuracy && !compute_accuracy) {
				cerr << "error: --dcim-accuracy requires --compute-accuracy\n";
				exit(1);
			}
			if (dcim_digital && !compute_accuracy) {
				cerr << "error: --dcim-digital requires --compute-accuracy\n";
				exit(1);
			}
			if (dcim_digital && cim_mode == "none") {
				cerr << "error: --dcim-digital requires --cim-mode != none\n";
				exit(1);
			}
			if (compute_accuracy) {
				if (w_init_path.empty() && w_init_l1_path.empty()) {
					cerr << "error: --compute-accuracy requires --w-init-l1; random weight fallback is forbidden\n";
					exit(1);
				}
				if (!w_init_path.empty() && !w_init_l1_path.empty() && w_init_path != w_init_l1_path) {
					cerr << "error: ambiguous layer-1 weights; use --w-init-l1 only\n";
					exit(1);
				}
				const string layer_paths[] = {w_init_l1_path, w_init_l2_path, w_init_l3_path,
				                              w_init_l4_path, w_init_l5_path};
				for (int layer = 2; layer <= num_layers && layer <= 5; ++layer) {
					if (layer_paths[layer - 1].empty()) {
						cerr << "error: --compute-accuracy requires --w-init-l" << layer
						     << "; random weight fallback is forbidden\n";
						exit(1);
					}
				}
			}
			if (bits_per_cycle < 1 || bits_per_cycle > 8) {
				cerr << "error: --bits-per-cycle must be in [1, 8]\n";
				exit(1);
			}
			if (cim_fifo_depth <= 0) {
				cerr << "error: --cim-fifo-depth must be > 0\n";
				exit(1);
			}
			if (adc_bits < 1 || adc_bits > 16) {
				cerr << "error: --adc-bits must be in [1, 16]\n";
				exit(1);
			}
			if (read_avg_n < 1 || read_avg_n > 8) {
				cerr << "error: --read-avg-n must be in [1, 8]\n";
				exit(1);
			}
			if (thermal_sigma_mv != -1.0 && thermal_sigma_mv <= 0.0) {
				cerr << "error: --thermal-sigma-mv must be > 0, or unset\n";
				exit(1);
			}
			if (thermal_sigma_scale <= 0.0) {
				cerr << "error: --thermal-sigma-scale must be > 0\n";
				exit(1);
			}
			if (temperature_k <= 0.0) {
				cerr << "error: --temperature-k must be > 0\n";
				exit(1);
			}
			if (thermal_model != "legacy" && thermal_model != "physics_ktc") {
				cerr << "error: --thermal-model must be one of: legacy | physics_ktc\n";
				exit(1);
			}
			if (thermal_bl_cap_ff <= 0.0) {
				cerr << "error: --thermal-bl-cap-ff must be > 0\n";
				exit(1);
			}
			if (thermal_adc_comp_sigma_mv < 0.0) {
				cerr << "error: --thermal-adc-comp-sigma-mv must be >= 0\n";
				exit(1);
			}
			if (act_scale_mode != "fixed" && act_scale_mode != "warmup_maxabs") {
				cerr << "error: --act-scale-mode must be one of: fixed | warmup_maxabs\n";
				exit(1);
			}
			if (act_calib_rows < 1) {
				cerr << "error: --act-calib-rows must be >= 1\n";
				exit(1);
			}
			if (act_calib_clip <= 0.0) {
				cerr << "error: --act-calib-clip must be > 0\n";
				exit(1);
			}

			if (memory_bw_gbps != -1.0 && memory_bw_gbps <= 0.0) {
				cerr << "error: --memory-bw-gbps must be > 0, or unset\n";
				exit(1);
			}
			if (memory_latency_cycles != -1 && memory_latency_cycles <= 0) {
				cerr << "error: --memory-latency-cycles must be > 0, or unset\n";
				exit(1);
			}
			if (memory_hit_pj_per_txn != -1.0 && memory_hit_pj_per_txn <= 0.0) {
				cerr << "error: --memory-hit-pj-per-txn must be > 0, or unset\n";
				exit(1);
			}
			if (memory_miss_pj_per_txn != -1.0 && memory_miss_pj_per_txn <= 0.0) {
				cerr << "error: --memory-miss-pj-per-txn must be > 0, or unset\n";
				exit(1);
			}
			if (cxl_link_bw_gbps != -1.0 && cxl_link_bw_gbps <= 0.0) {
				cerr << "error: --cxl-link-bw-gbps must be > 0, or unset\n";
				exit(1);
			}
			if (cxl_extra_latency_ns != -1.0 && cxl_extra_latency_ns < 0.0) {
				cerr << "error: --cxl-extra-latency-ns must be >= 0, or unset\n";
				exit(1);
			}

			auto to_lower_copy = [](string s) {
				for (char& c : s) {
					c = static_cast<char>(tolower(static_cast<unsigned char>(c)));
				}
				return s;
			};
			this->memory_type = to_lower_copy(this->memory_type);

			struct MemoryProfile {
				string config_path;
				double bw_gbps;
				int latency_cycles;
				double hit_pj_per_txn;
				double miss_pj_per_txn;
				bool cxl_mode;
				double cxl_link_bw_gbps;
				double cxl_extra_latency_ns;
				double tck_ns;  // DRAM clock period in nanoseconds
			};

			MemoryProfile profile;
			if (this->memory_type == "hbm2") {
				// HBM2: tCK = 1.0ns (1.0 GHz), matches NC clock
				profile = {"./ktb_dram_configs/ktb_HBM2_8Gb_x128.ini", 256.0, 70, 358.0, 1894.0, false, 0.0, 0.0, 1.0};
			} else if (this->memory_type == "hbm3") {
				// HBM3: tCK = 0.625ns (1.6 GHz), faster than NC clock
				profile = {"./ktb_dram_configs/ktb_HBM3_16Gb_x128.ini", 819.2, 60, 230.0, 1215.0, false, 0.0, 0.0, 0.625};
			} else if (this->memory_type == "hbm4") {
				// HBM4: tCK = 0.5ns (2.0 GHz), 2x faster than NC clock
				profile = {"./ktb_dram_configs/ktb_HBM4_24Gb_x128.ini", 1536.0, 55, 165.0, 875.0, false, 0.0, 0.0, 0.5};
			} else if (this->memory_type == "cxl_ddr5") {
				// DDR5-4800: tCK = 0.4167ns
				profile = {"./ktb_dram_configs/ktb_DDR5_8Gb_x8_4800.ini", 64.0, 220, 3584.0, 4301.0, true, 64.0, 150.0, 0.4167};
			} else if (this->memory_type == "cxl_ddr5_5600") {
				// DDR5-5600: tCK = 0.357ns
				profile = {"./ktb_dram_configs/ktb_DDR5_8Gb_x8_5600.ini", 128.0, 170, 3584.0, 4301.0, true, 128.0, 100.0, 0.357};
			} else {
				cerr << "error: --memory-type must be one of: hbm2 | hbm3 | hbm4 | cxl_ddr5 | cxl_ddr5_5600\n";
				exit(1);
			}

			const bool memory_bw_overridden = (this->memory_bw_gbps > 0.0);
			const bool memory_latency_overridden = (this->memory_latency_cycles > 0);
			const bool memory_hit_overridden = (this->memory_hit_pj_per_txn > 0.0);
			const bool memory_miss_overridden = (this->memory_miss_pj_per_txn > 0.0);
			const bool cxl_bw_overridden = (this->cxl_link_bw_gbps > 0.0);
			const bool cxl_extra_latency_overridden = (this->cxl_extra_latency_ns >= 0.0);

			this->cxl_mode = profile.cxl_mode;
			if (this->dram_config_path.empty()) {
				this->dram_config_path = profile.config_path;
			}
			if (!memory_bw_overridden) {
				this->memory_bw_gbps = profile.bw_gbps;
			}
			if (!memory_latency_overridden) {
				this->memory_latency_cycles = profile.latency_cycles;
			}
			if (!memory_hit_overridden) {
				this->memory_hit_pj_per_txn = profile.hit_pj_per_txn;
			}
			if (!memory_miss_overridden) {
				this->memory_miss_pj_per_txn = profile.miss_pj_per_txn;
			}
			// tCK is always set from profile (no CLI override)
			this->memory_tck_ns = profile.tck_ns;

			if (this->cxl_mode) {
				if (!cxl_bw_overridden) {
					this->cxl_link_bw_gbps = profile.cxl_link_bw_gbps;
				}
				if (!cxl_extra_latency_overridden) {
					this->cxl_extra_latency_ns = profile.cxl_extra_latency_ns;
				}
				this->cxl_extra_latency_cycles =
					static_cast<int>(round(max(0.0, this->cxl_extra_latency_ns)));
				if (!memory_bw_overridden) {
					this->memory_bw_gbps = this->cxl_link_bw_gbps;
				}
				if (!memory_latency_overridden) {
					const int base_hbm_latency_cycles = 70;
					this->memory_latency_cycles = base_hbm_latency_cycles + this->cxl_extra_latency_cycles;
				}
			} else {
				this->cxl_link_bw_gbps = 0.0;
				this->cxl_extra_latency_ns = 0.0;
				this->cxl_extra_latency_cycles = 0;
			}
	}
	catch (exception& e) {
		cerr << "error: " << e.what() << "\n";
		exit(1);
	}
	catch (...) {
		cerr << "Exception of unknown type!\n";
		exit(1);
	}
}

void ArgParse::print()
{
	cout << "Experiment Name: " << this->exp_name << endl;
	// cout << "Program Root: " << this->program_root << endl;
	cout << "Description: " << this->description << endl;
	cout << "Dataset: " << this->dataset << endl;
	// cout << "Dataset Root: " << this->dataset_root << endl;
	cout << "Tile Size: " << this->tile_size << endl;
	cout << "Cache Lines Count: " << this->cache_lines_count << endl;
	cout << "Pipeline Count: " << this->pipeline_count << endl;
	cout << "Comparator Count: " << this->comparator_count << endl;

	cout << "SpMM Mode: " << this->spmm_mode << endl;
	cout << "Mode Threshold: " << this->mode_threshold << endl;
	if (!this->f_init_path.empty()) cout << "F init: " << this->f_init_path << endl;
	if (!this->delta_a_path.empty()) cout << "Delta A: " << this->delta_a_path << endl;
	if (!this->delta_f_path.empty()) cout << "Delta F: " << this->delta_f_path << endl;
	cout << "Indexer Mode: " << this->indexer_mode << endl;
	cout << "CiM Mode: " << this->cim_mode << endl;
	cout << "Arch Variant: " << this->arch_variant << endl;
	cout << "Compute Accuracy: " << (this->compute_accuracy ? "true" : "false") << endl;
	if (!this->w_init_path.empty()) cout << "W init: " << this->w_init_path << endl;
	cout << "CiM Noise: " << (this->cim_noise ? "true" : "false") << endl;
	cout << "Timesteps: " << this->timesteps << endl;
	if (!this->delta_dir.empty()) cout << "Delta Dir: " << this->delta_dir << endl;
	cout << "Memory Type: " << this->memory_type << endl;
	cout << "DRAM Config: " << this->dram_config_path << endl;
	cout << "Memory BW (GB/s): " << this->memory_bw_gbps << endl;
	cout << "Memory Latency (cycles): " << this->memory_latency_cycles << endl;
	cout << "Memory Hit/Miss Energy (pJ/txn): "
	     << this->memory_hit_pj_per_txn << " / " << this->memory_miss_pj_per_txn << endl;
	if (this->cxl_mode) {
		cout << "CXL Link BW (GB/s): " << this->cxl_link_bw_gbps << endl;
		cout << "CXL Extra Latency (cycles): " << this->cxl_extra_latency_cycles << endl;
	}
	cout << "DCIM Accuracy (ACIM): " << (this->dcim_accuracy ? "true" : "false") << endl;
	cout << "DCIM Digital: " << (this->dcim_digital ? "true" : "false") << endl;
	cout << "Num Layers: " << this->num_layers << endl;
	cout << "Isolation Mode: " << this->isolation_mode << endl;
	cout << "Scope Mode: " << this->scope_mode << endl;
	cout << "Layer Norm Mode: " << this->layer_norm_mode
	     << " (effective=" << (this->get_effective_layer_norm_enabled() ? "on" : "off") << ")" << endl;
	if (this->nl_threshold_override >= 0.0) cout << "NL Threshold Override: " << this->nl_threshold_override << endl;
	if (!this->w_init_l1_path.empty()) cout << "W init L1: " << this->w_init_l1_path << endl;
	if (!this->w_init_l2_path.empty()) cout << "W init L2: " << this->w_init_l2_path << endl;
	cout << "Export Z: " << (this->export_z ? "true" : "false") << endl;
	cout << "Export Z Per Timestep: " << (this->export_z_per_timestep ? "true" : "false") << endl;
	if (!this->export_z_timestep_list.empty()) cout << "Export Z Timesteps: " << this->export_z_timestep_list << endl;
	cout << "Variation Seed: " << this->variation_seed << endl;
	cout << "Bits per Cycle: " << this->bits_per_cycle << endl;
	cout << "CiM FIFO Depth: " << this->cim_fifo_depth << endl;
	cout << "Read Avg N: " << this->read_avg_n << endl;
	cout << "PV Compensation: " << (this->no_pv_comp ? "off" : "on") << endl;
	cout << "Thermal Sigma Override (mV): " << this->thermal_sigma_mv << endl;
	cout << "Thermal Sigma Scale: " << this->thermal_sigma_scale << endl;
	cout << "Temperature (K): " << this->temperature_k << endl;
	cout << "Thermal Model: " << this->thermal_model << endl;
	cout << "Thermal BL Cap (fF): " << this->thermal_bl_cap_ff << endl;
	cout << "Thermal ADC Comp Sigma (mV): " << this->thermal_adc_comp_sigma_mv << endl;
	cout << "Activation Scale Mode: " << this->act_scale_mode << endl;
	cout << "Activation Calib Rows: " << this->act_calib_rows << endl;
	cout << "Activation Calib Clip: " << this->act_calib_clip << endl;
}
