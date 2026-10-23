// SPDX-License-Identifier: CC-BY-4.0
// Derived from NeuraChip/NeuraSim (DOI: 10.5281/zenodo.10896280); modified for Despoina and this standalone release in 2026.
// Created by Kaustubh Shivdikar
//

#include "globalcounters.h"
#include "../debugger/debugger.h"
#include "../config.h"
#include "../energy_params_generated.h"
#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <iostream>
#ifdef _WIN32
#include <direct.h>
#include <io.h>
#else
#include <unistd.h>
#include <sys/stat.h>
#endif

#ifdef _WIN32
#define DIR_EXISTS(path) (_access((path), 0) != -1)
#define CREATE_DIR(path) _mkdir((path))
#else
#define DIR_EXISTS(path) (access((path), F_OK) != -1)
#define CREATE_DIR(path) mkdir((path), 0777)
#endif

using chrono::high_resolution_clock;

using namespace std;

namespace {

struct AreaModelBreakdown {
	double total_mm2 = 0.0;
	double sram_mm2 = 0.0;
	double noc_mm2 = 0.0;
	double pe_logic_mm2 = 0.0;
	double cim_mm2 = 0.0;
	double yz_policy_mm2 = 0.0;
	double y_buffer_mm2 = 0.0;
	double z_buffer_mm2 = 0.0;
	double yz_ctrl_mm2 = 0.0;
	double f_cache_mm2 = 0.0;
	double index_store_mm2 = 0.0;
	double l1_mru_mm2 = 0.0;
	double pe_reg_mm2 = 0.0;
	double shared_buffer_mm2 = 0.0;
	double indexer_ctrl_mm2 = 0.0;
	double on_chip_memory_mb = 0.0;
	long long y_buffer_bytes = 0;
	long long z_buffer_bytes = 0;
	int active_cim_tiles = 0;
	bool y_dram_enabled = false;
	bool z_dram_enabled = false;
	bool z_rmw_enabled = false;
	std::string spmm_mode = "full";
	std::string cim_mode = "none";
	bool dcim_digital = false;
	std::string indexer_mode = "indexstore";
	std::string model_version = "cacti_neurosim_policy_v2_dcimaware";
};

static std::string to_lower_copy(std::string s) {
	for (char& c : s) {
		c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
	}
	return s;
}

static AreaModelBreakdown estimate_chip_area(const GlobalCounters* gc) {
	AreaModelBreakdown out;
	// Derive actual PE Array counts from compile-time PE_NUM_CLUSTERS.
#ifndef PE_NUM_CLUSTERS
#define PE_NUM_CLUSTERS 64
#endif
	const int clusters = std::max(1, PE_NUM_CLUSTERS);
	const int total_pes = clusters * 4;  // PES_PER_CLUSTER = 4
	const int routers = clusters * 2;    // 2 routers per cluster (mesh)
	const double cluster_scale = static_cast<double>(clusters) / 64.0;
	const double router_scale = static_cast<double>(routers) / 128.0;

	if (gc->arg_parse != nullptr) {
		out.spmm_mode = to_lower_copy(gc->arg_parse->get_spmm_mode());
		out.cim_mode = to_lower_copy(gc->arg_parse->get_cim_mode());
		out.dcim_digital = gc->arg_parse->get_dcim_digital();
		out.indexer_mode = to_lower_copy(gc->arg_parse->get_indexer_mode());
	}

	if (out.cim_mode != "none" && out.cim_mode != "delta" && out.cim_mode != "full" && out.cim_mode != "linear") {
		out.cim_mode = "none";
	}

	const bool incremental_path = (out.spmm_mode == "incremental" || out.spmm_mode == "scheduled");
	const bool index_store_enabled = (out.indexer_mode != "fullscan");
	const bool cim_enabled = (out.cim_mode != "none");
	out.y_dram_enabled = incremental_path && (out.cim_mode == "none" || out.cim_mode == "full");
	out.z_dram_enabled = incremental_path && (out.cim_mode == "delta" || out.cim_mode == "linear" || out.cim_mode == "full");
	out.z_rmw_enabled = incremental_path && (out.cim_mode == "delta" || out.cim_mode == "linear");

	// F-Cache: scale by compile-time entry count vs CACTI reference (16384 entries = 4MB).
	double cache_line_scale = 1.0;
	if (gc->AREA_F_CACHE_ENTRIES > 0) {
		cache_line_scale = static_cast<double>(gc->AREA_F_CACHE_ENTRIES) / 16384.0;
	}
	out.f_cache_mm2 = energy::sram::f_cache::AREA_MM2 * cache_line_scale;

	// IndexStore: shared chip-level SRAM. Scale by entry count only (NOT per-cluster).
	double is_entry_scale = 1.0;
	if (gc->AREA_INDEX_STORE_MAX_ENTRIES > 0) {
		is_entry_scale = static_cast<double>(gc->AREA_INDEX_STORE_MAX_ENTRIES) / 16384.0;
	}
	out.index_store_mm2 = index_store_enabled
		? (energy::sram::index_store::AREA_MM2 * is_entry_scale) : 0.0;

	// L1 MRU: scale by cluster count AND compile-time entries per cluster vs CACTI reference (64 entries).
	double l1_entry_scale = static_cast<double>(gc->AREA_L1_MRU_ENTRIES_PER_CLUSTER)
	                      / static_cast<double>(energy::sram::l1_mru::ENTRIES_PER_CLUSTER);
	out.l1_mru_mm2 = energy::sram::l1_mru::AREA_MM2_PER_CLUSTER * static_cast<double>(clusters) * l1_entry_scale;
	out.pe_reg_mm2 = energy::sram::pe_registers::AREA_PER_PE_MM2 * static_cast<double>(total_pes);
	out.shared_buffer_mm2 = energy::sram::shared_buffer::AREA_PER_CLUSTER_MM2 * static_cast<double>(clusters);
	out.indexer_ctrl_mm2 = index_store_enabled
		? (energy::sram::indexer_mshr_sram::AREA_MM2 + energy::sram::indexer_mshr_cam::AREA_MM2) * cluster_scale
		: 0.0;

	out.sram_mm2 = out.f_cache_mm2 + out.index_store_mm2 + out.l1_mru_mm2 +
	               out.pe_reg_mm2 + out.shared_buffer_mm2 + out.indexer_ctrl_mm2;
	out.noc_mm2 = energy::noc::TOTAL_NOC_AREA_MM2 * router_scale;
	out.pe_logic_mm2 = energy::pe_array::PE_AREA_MM2 * static_cast<double>(total_pes);

	if (cim_enabled) {
		const double cim_tiles_ref = 16.0;
		out.active_cim_tiles = std::max(1, static_cast<int>(std::llround(cim_tiles_ref * cluster_scale)));
		const double cim_tile_area_mm2 = out.dcim_digital ? energy::dcim::AREA_MM2 : energy::cim::AREA_MM2;
		out.cim_mm2 = cim_tile_area_mm2 * static_cast<double>(out.active_cim_tiles);
	}

	// Policy-specific SRAM: Y materialization (none/full) vs Z R-M-W accumulation (delta/linear).
	// Converted via strict IndexStore density (5MB @ AREA_MM2 from energy curator).
	const double index_store_bytes_ref = 5.0 * 1024.0 * 1024.0;
	const double bytes_per_mm2 = index_store_bytes_ref / std::max(1e-12, energy::sram::index_store::AREA_MM2);
	double policy_scale = std::max(0.25, cluster_scale);
	if (out.y_dram_enabled) {
		out.y_buffer_bytes = static_cast<long long>(std::llround(256.0 * 1024.0 * policy_scale));
	}
	if (out.z_dram_enabled) {
		const double z_bytes = out.z_rmw_enabled ? (256.0 * 1024.0) : (128.0 * 1024.0);
		out.z_buffer_bytes = static_cast<long long>(std::llround(z_bytes * policy_scale));
		// The Post-Combination Buffer that holds persistent Z is the configured
		// dedicated Z-SRAM, priced at its full capacity.
		if (gc->arg_parse != nullptr && gc->arg_parse->get_dedicated_z_sram() &&
		    gc->arg_parse->get_dedicated_z_sram_size_mb() > 0.0) {
			out.z_buffer_bytes = static_cast<long long>(std::llround(
				gc->arg_parse->get_dedicated_z_sram_size_mb() * 1024.0 * 1024.0));
		}
	}

	out.y_buffer_mm2 = static_cast<double>(out.y_buffer_bytes) / bytes_per_mm2;
	out.z_buffer_mm2 = static_cast<double>(out.z_buffer_bytes) / bytes_per_mm2;
	out.yz_ctrl_mm2 = (out.y_dram_enabled || out.z_dram_enabled)
		? (energy::sram::indexer_mshr_sram::AREA_MM2 + energy::sram::indexer_mshr_cam::AREA_MM2) * std::max(1.0, cluster_scale)
		: 0.0;
	out.yz_policy_mm2 = out.y_buffer_mm2 + out.z_buffer_mm2 + out.yz_ctrl_mm2;

	out.total_mm2 = out.sram_mm2 + out.noc_mm2 + out.pe_logic_mm2 + out.cim_mm2 + out.yz_policy_mm2;

	// On-chip memory estimate in MB (for backward-compatible reporting field).
	double on_chip_bytes = 0.0;
	on_chip_bytes += static_cast<double>(gc->AREA_F_CACHE_ENTRIES) * 256.0;  // F-cache
	on_chip_bytes += index_store_enabled
		? (index_store_bytes_ref * cluster_scale * is_entry_scale) : 0.0;    // IndexStore
	on_chip_bytes += static_cast<double>(gc->AREA_L1_MRU_ENTRIES_PER_CLUSTER)
	              * 512.0 * static_cast<double>(clusters);                   // L1 MRU
	on_chip_bytes += static_cast<double>(total_pes) * 512.0;             // PE regs
	on_chip_bytes += 4.0 * 1024.0 * static_cast<double>(clusters);       // Shared buffers
	on_chip_bytes += static_cast<double>(out.y_buffer_bytes + out.z_buffer_bytes);
	out.on_chip_memory_mb = on_chip_bytes / (1024.0 * 1024.0);

	return out;
}

}  // namespace

// Constructor
GlobalCounters::GlobalCounters(ArgParse *arg_parse)
{
	// Keep a pointer to parsed CLI options for downstream modules (e.g., DyGNN PE Array modes).
	this->arg_parse = arg_parse;

	// Start cycle count from 1
	// To avoid confusion of first INTERVAL being INTERVAL+1
	this->set_configurable_parameters(arg_parse);
	this->cycle = 1;
	this->mem_req_id = -1;
	this->mem_resp_id = -1;
	this->core_id = -1;
	this->mem_id = -1;
	this->router_id = -1;
	this->tile_id = -1;
	this->conn_id = -1;
	this->uid = -1;
	this->dispatcher_id = -1;
	this->mc_id = -1;
	this->writer_id = -1;
	this->hash_req_id = -1;
	this->hash_resp_id = -1;
	this->mcps = 0;
	this->cps = 0;
	this->sim_time = 0;
	this->interval_sim_time = 0;
	this->frequency = this->FREQUENCY;
	// Use relative path from NeuraSim directory
	this->results_base_dir = "./results_neurachip/";
	this->interval_sim_progress = 0;

	this->exp_name = arg_parse->get_exp_name();
	this->test_stride = arg_parse->get_stride();
	this->results_dir = this->results_base_dir + this->exp_name;

	// Disabled when running through script
	// Check if the results directory exists
	// If it exists, delete it
	if (DIR_EXISTS(this->results_dir.c_str())) {
		cerr << "Error: result directory already exists; choose a fresh --exp_name: "
		     << this->results_dir << endl;
		exit(EXIT_FAILURE);
	} else {
		cout << "Directory does not exist" << endl;
	}

	// Create the results directory
	if (CREATE_DIR(this->results_dir.c_str()) == -1) {
		cout << "Error creating directory: " << this->results_dir << endl;
	} else {
		// cout << "Directory created: " << this->results_dir << endl;
		cout << "Directory created." << endl;
	}




	this->dram_data_file_name_0 = arg_parse->get_dataset_root() + "/" + "raw_data" + "/" + "dram_raw_data_0.csv";
	this->dram_data_file_name_1 = arg_parse->get_dataset_root() + "/" + "raw_data" + "/" + "dram_raw_data_1.csv";
	this->program_file = arg_parse->get_dataset_root() + "/" + "bin" + "/" + "spmm.csv";
	// cout << "Program File: " << this->program_file << endl;
	this->program_data_folder = arg_parse->get_dataset_root() + "/" + "matrix";
	// Create a new database with the experiment name
	this->result_sink = new ResultSink(arg_parse->get_exp_name());
	this->result_sink->drop();
	this->component_name = "GlobalCounters";
	this->header_component_name = "H_" + this->component_name;
	this->collection = this->result_sink->get_component_collection(this->component_name);
	this->header_collection = this->result_sink->get_component_collection(this->header_component_name);
	this->interval_id = 0;
	this->total_cores = this->NEURA_CORE_X_COUNT * this->NEURA_CORE_Y_COUNT;
	this->total_tiles = this->NEURA_TILE_X_COUNT * this->NEURA_TILE_Y_COUNT;
	this->total_mems = this->NEURA_MEM_X_COUNT * this->NEURA_MEM_Y_COUNT;
	this->total_mcs = this->MC_COUNT;
	this->total_writers = this->WRITER_COUNT;
	this->total_dispatchers = this->DISPATCHER_COUNT;
	this->total_routers = this->NEURA_ROUTER_X_COUNT * this->NEURA_ROUTER_Y_COUNT;
	this->core_id_array = new int[this->total_cores];
	for (int i = 0; i < this->total_cores; i++)
	{
		this->core_id_array[i] = -1;
	}
	this->tile_id_array = new int[this->total_tiles];
	for (int i = 0; i < this->total_tiles; i++)
	{
		this->tile_id_array[i] = -1;
	}
	this->mem_id_array = new int[this->total_mems];
	for (int i = 0; i < this->total_mems; i++)
	{
		this->mem_id_array[i] = -1;
	}
	this->mc_id_array = new int[this->total_mcs];
	for (int i = 0; i < this->total_mcs; i++)
	{
		this->mc_id_array[i] = -1;
	}
	this->writer_id_array = new int[this->total_writers];
	for (int i = 0; i < this->total_writers; i++)
	{
		this->writer_id_array[i] = -1;
	}
	this->dispatcher_id_array = new int[this->total_dispatchers];
	for (int i = 0; i < this->total_dispatchers; i++)
	{
		this->dispatcher_id_array[i] = -1;
	}
	this->router_id_array = new int[this->total_routers];
	for (int i = 0; i < this->total_routers; i++)
	{
		this->router_id_array[i] = -1;
	}

	this->test_arch_graph();
}



void GlobalCounters::set_configurable_parameters(ArgParse *arg_parse)
{
	/*
	  ____ ___  _   _ _____ ___ ____
 	 / ___/ _ \| \ | |  ___|_ _/ ___|
	| |  | | | |  \| | |_   | | |  _
	| |__| |_| | |\  |  _|  | | |_| |
 	 \____\___/|_| \_|_|   |___\____|
	*/
	// Tile Config
	this->NEURA_TILE_SIZE = arg_parse->get_tile_size();
	this->NEURA_TILE_X_COUNT = 1;
	this->NEURA_TILE_Y_COUNT = 1;
	this->NEURA_TILE_PORT_COUNT = 4;

	// Core Config
	this->NEURA_CORE_X_COUNT = this->NEURA_TILE_SIZE;
	this->NEURA_CORE_Y_COUNT = this->NEURA_TILE_SIZE;

	// Last port is for Instructions from Dispatcher
	this->NEURA_CORE_PORT_COUNT = 5;
	this->NEURA_CORE_DATA_PORT_COUNT = 4;
	// Instruction Port Index for NeuraCores
	this->INSTX_PORT_INDEX = 4;
	// Core register file size
	// Number of Instructions that can be stored in register file
	// this->PIPELINE_COUNT = 16;
	// this->PIPELINE_COUNT = 64;
	this->PIPELINE_COUNT = arg_parse->get_pipeline_count();

	// Memory Config
	this->NEURA_MEM_X_COUNT = this->NEURA_TILE_SIZE;
	this->NEURA_MEM_Y_COUNT = this->NEURA_TILE_SIZE;
	this->NEURA_MEM_PORT_COUNT = 4;

	// Hashpad
	this->HASHPAD_COMPARATOR_COUNT = arg_parse->get_comparator_count();

	// Memory Controller Config
	this->MC_COUNT = 1;
	this->MC_PORT_COUNT = 10;

	// Router Config
	// this->NEURA_ROUTER_X_COUNT = 2 * NEURA_TILE_SIZE + 1;
	// this->NEURA_ROUTER_Y_COUNT = NEURA_TILE_SIZE + 1;
	this->NEURA_ROUTER_X_COUNT = 2 * NEURA_TILE_SIZE;
	this->NEURA_ROUTER_Y_COUNT = NEURA_TILE_SIZE;
	// Last port is for writer and memory controller
	this->NEURA_ROUTER_PORT_COUNT = 9;
	this->NEURA_ROUTER_PROCESSING_CYCLES = 0;
	this->NEURA_ROUTER_BUFFER_SIZE = 32;

	// Writer Config
	this->WRITER_COUNT = 2;
	this->WRITER_PORT_COUNT = 2;

	// Cache Config
	// this->CACHE_LINES_COUNT 4096

	this->CACHE_LINES_COUNT = arg_parse->get_cache_lines_count();

	// Sync compile-time SRAM params for area model.
	// Compile-time macros control the actual SRAM sizes used in simulation,
	// but the area model historically read from CLI defaults that were never synced.
#ifdef PE_SHARED_F_CACHE_ENTRIES
	this->AREA_F_CACHE_ENTRIES = PE_SHARED_F_CACHE_ENTRIES;
#else
	this->AREA_F_CACHE_ENTRIES = this->CACHE_LINES_COUNT;
#endif

#ifdef PE_L1_MRU_ENTRIES
	this->AREA_L1_MRU_ENTRIES_PER_CLUSTER = PE_L1_MRU_ENTRIES;
#else
	this->AREA_L1_MRU_ENTRIES_PER_CLUSTER = energy::sram::l1_mru::ENTRIES_PER_CLUSTER;
#endif

#ifdef INDEX_STORE_MAX_ENTRIES
	this->AREA_INDEX_STORE_MAX_ENTRIES = INDEX_STORE_MAX_ENTRIES;
#else
	this->AREA_INDEX_STORE_MAX_ENTRIES = 16384;
#endif

	// Timings
	this->DRAM_CONST_LATENCY = 300;
	this->IF_PROCESSING_CYCLES = 1;
	this->MULT_PROCESSING_CYCLES = 1;

	// Constant
	this->DISPATCH_INF_INSTX = -10; // Constant, do not comment

	// Dispatcher Config
	// this->DISPATCHER_X_COUNT = this->NEURA_CORE_X_COUNT;
	// this->DISPATCHER_Y_COUNT = this->NEURA_CORE_Y_COUNT;


	// Create 1 dispatcher and connect it to all cores
	this->DISPATCHER_COUNT = 1;
	// Dispatcher port count

	// Using 1 dispatcher per core
	this->DISPATCHER_PORT_COUNT = this->NEURA_CORE_X_COUNT * this->NEURA_CORE_Y_COUNT;

	// Configurable
	// this->DISPATCH_INSTX_COUNT = 1;
	// this->DISPATCH_INSTX_COUNT = 0;
	// Just Table 0
	// this->DISPATCH_INSTX_COUNT = 53;
	// this->DISPATCH_INSTX_COUNT = 118;
	// this->DISPATCH_INSTX_COUNT = 572; // 10 Rows
	// Entire Workload
	// this->DISPATCH_INSTX_COUNT = 192885;
	// Sample Instruction | Warning: This will never HALT

	this->DISPATCH_INSTX_COUNT = DISPATCH_INF_INSTX;

	this->INTERVAL = 1000;
	this->MIN_CYCLES = 0;
	this->HARD_STOP = 400000000;
	// this->HARD_STOP = 100;

	// this->INTERMEDIATE_STAT_PRINT = 10000;

	// Frequency 1.0 GHz
	this->FREQUENCY = 1000000000;
	this->frequency = this->FREQUENCY;
	this->HBM_FREQUENCY = 3200000000;
}

void GlobalCounters::test_arch_graph()
{
	cout << "Testing architecture graph generated \u2713" << endl;

	// Add vertices
	// this->add_vertex_to_arch_graph("C_0_0");
	// this->add_vertex_to_arch_graph("C_0_1");
	// this->add_vertex_to_arch_graph("C_1_0");
	// this->add_vertex_to_arch_graph("C_1_1");

	// this->add_edge_to_arch_graph("C_0_0", "C_0_1", BIDIRECTIONAL_EDGE);
	// this->add_edge_to_arch_graph("C_0_0", "C_1_0", BIDIRECTIONAL_EDGE);
	// this->add_edge_to_arch_graph("C_0_1", "C_1_1", BIDIRECTIONAL_EDGE);
	// this->add_edge_to_arch_graph("C_1_0", "C_1_1", BIDIRECTIONAL_EDGE);

	// vector<int> distances(boost::num_vertices(this->arch_graph));
	// boost::dijkstra_shortest_paths(this->arch_graph, this->inverse_vertex_map["C_0_0"] , boost::distance_map(&distances[0]));

	// std::cout << "Distances from vertex 0:" << std::endl;
	// for(int i = 0; i < distances.size(); ++i) {
	//     std::cout << i << ": " << distances[i] << std::endl;
	// }

	// cout << "Testing Arch Graph Complete" << endl;
}

void GlobalCounters::visualize_arch_graph()
{
	// The standalone release has no external Graphviz process hook.
	cout << "Architecture graph visualization is disabled in the standalone release" << endl;
}

void GlobalCounters::add_vertex_to_arch_graph(string vertex_name, int uid)
{
	VertexDescriptor v_desc = boost::add_vertex(this->arch_graph);
	// cout << "Adding vertex " << vertex_name << " with uid " << uid << " and desc " << v_desc << endl;
	this->vertex_desc_to_name_map[v_desc] = vertex_name;
	this->vertex_name_to_desc_map[vertex_name] = v_desc;
	this->vertex_uid_to_desc_map[uid] = v_desc;
	this->vertex_desc_to_uid_map[v_desc] = uid;
	this->uid_to_vertex_name_map[uid] = vertex_name;
}

// Just dont create UNIDIRECTIONAL edges
// Creates a UNIDIRECTIONAL edge from vertex_1 to vertex_2
// void GlobalCounters::add_edge_to_arch_graph(string vertex_1, string vertex_2, int edge_1_weight) {
// 	if (edge_1_weight == 0) {
// 		cout << FATAL_ERROR << "Edge weight cannot be 0" << endl;
// 	}
// 	VertexDescriptor v1 = this->inverse_vertex_map[vertex_1];
// 	VertexDescriptor v2 = this->inverse_vertex_map[vertex_2];
// 	boost::add_edge(v1, v2, EdgeWeightProperty(edge_1_weight), this->arch_graph);
// }

// Create a BIDIRECTIONAL edge from vertex_1 to vertex_2
void GlobalCounters::add_edge_to_arch_graph(string vertex_1, string vertex_2, int edge_1_weight, int edge_2_weight)
{
	if (edge_1_weight == 0 || edge_2_weight == 0)
	{
		cout << FATAL_ERROR << "Edge weight cannot be 0" << endl;
	}
	VertexDescriptor v1 = this->vertex_name_to_desc_map[vertex_1];
	VertexDescriptor v2 = this->vertex_name_to_desc_map[vertex_2];
	boost::add_edge(v1, v2, EdgeWeightProperty(edge_1_weight), this->arch_graph);
	boost::add_edge(v2, v1, EdgeWeightProperty(edge_2_weight), this->arch_graph);
}

GlobalCounters::~GlobalCounters()
{
	delete this->result_sink;
	delete[] this->core_id_array;
	delete[] this->tile_id_array;
	delete[] this->mem_id_array;
	delete[] this->mc_id_array;
	delete[] this->writer_id_array;
	delete[] this->dispatcher_id_array;
	delete[] this->router_id_array;
}

void GlobalCounters::print()
{
	cout << "Tile ID: " << this->tile_id << endl;
	cout << "Core ID: " << this->core_id << endl;
	cout << "Memory ID: " << this->mem_id << endl;
	cout << "Router ID: " << this->router_id << endl;
	cout << "Memory Request ID: " << this->mem_req_id << endl;
	cout << "Memory Response ID: " << this->mem_resp_id << endl;
	cout << "Connection ID: " << this->conn_id << endl;
	cout << "UID: " << this->uid << endl;
	cout << "Dispatcher ID: " << this->dispatcher_id << endl;
}

long long GlobalCounters::generate_mem_req_id()
{
	this->mem_req_id++;
	return (this->mem_req_id);
}

long long GlobalCounters::generate_mem_resp_id()
{
	this->mem_resp_id++;
	return (this->mem_resp_id);
}

long long GlobalCounters::generate_hash_req_id()
{
	this->hash_req_id++;
	return (this->hash_req_id);
}

long long GlobalCounters::generate_hash_resp_id()
{
	this->hash_resp_id++;
	return (this->hash_resp_id);
}

int GlobalCounters::generate_core_id()
{
	this->core_id++;
	return (this->core_id);
}

int GlobalCounters::generate_tile_id()
{
	this->tile_id++;
	return (this->tile_id);
}

int GlobalCounters::generate_mem_id()
{
	this->mem_id++;
	return (this->mem_id);
}

int GlobalCounters::generate_router_id()
{
	this->router_id++;
	return (this->router_id);
}

int GlobalCounters::generate_conn_id()
{
	this->conn_id++;
	return (this->conn_id);
}

int GlobalCounters::generate_uid()
{
	this->uid++;
	return (this->uid);
}

int GlobalCounters::generate_dispatcher_id()
{
	this->dispatcher_id++;
	return (this->dispatcher_id);
}

int GlobalCounters::generate_mc_id()
{
	this->mc_id++;
	return (this->mc_id);
}

int GlobalCounters::generate_writer_id()
{
	this->writer_id++;
	return (this->writer_id);
}

void GlobalCounters::set_core_id_array(int index, int uid)
{
	this->core_id_array[index] = uid;
}

void GlobalCounters::set_tile_id_array(int index, int uid)
{
	this->tile_id_array[index] = uid;
}

void GlobalCounters::set_mem_id_array(int index, int uid)
{
	this->mem_id_array[index] = uid;
}

void GlobalCounters::set_mc_id_array(int index, int uid)
{
	this->mc_id_array[index] = uid;
}

void GlobalCounters::set_writer_id_array(int index, int uid)
{
	this->writer_id_array[index] = uid;
}

void GlobalCounters::set_dispatcher_id_array(int index, int uid)
{
	this->dispatcher_id_array[index] = uid;
}

void GlobalCounters::set_router_id_array(int index, int uid)
{
	this->router_id_array[index] = uid;
}

int *GlobalCounters::get_core_id_array()
{
	return (this->core_id_array);
}

int *GlobalCounters::get_tile_id_array()
{
	return (this->tile_id_array);
}

int *GlobalCounters::get_mem_id_array()
{
	return (this->mem_id_array);
}

int *GlobalCounters::get_mc_id_array()
{
	return (this->mc_id_array);
}

int *GlobalCounters::get_writer_id_array()
{
	return (this->writer_id_array);
}

int *GlobalCounters::get_dispatcher_id_array()
{
	return (this->dispatcher_id_array);
}

int *GlobalCounters::get_router_id_array()
{
	return (this->router_id_array);
}

int GlobalCounters::get_total_cores()
{
	return (this->total_cores);
}

int GlobalCounters::get_total_tiles()
{
	return (this->total_tiles);
}

int GlobalCounters::get_total_mems()
{
	return (this->total_mems);
}

int GlobalCounters::get_total_routers()
{
	return (this->total_routers);
}

int GlobalCounters::get_total_mcs()
{
	return (this->total_mcs);
}

int GlobalCounters::get_total_writers()
{
	return (this->total_writers);
}

int GlobalCounters::get_total_dispatchers()
{
	return (this->total_dispatchers);
}

void GlobalCounters::set_program_file(string program_file)
{
	this->program_file = program_file;
}

string GlobalCounters::get_program_file()
{
	return (this->program_file);
}

// Ticks the Cycle Counter
void GlobalCounters::tick()
{
	this->cycle++;
}

void GlobalCounters::statistics()
{
	cout << "********** Final Statistics **********" << endl;
	this->sim_time_printer();
	cout << "Total Cycles: " << CYAN_START << this->cycle << COLOR_RESET << endl;
	cout << "Execution Time: ";
	this->cycles_to_time_printer();
	cout << endl;
	cout << "**************************************" << endl;
}

void GlobalCounters::cycles_to_time_printer()
{
	// Cycles are in nanoseconds
	if (this->cycle < 1000)
	{
		cout << UNDERLINE_START;
		cout << this->cycle << " ns";
		cout << COLOR_RESET;
		return;
	}
	if (this->cycle >= 1000 && this->cycle < 1000000)
	{
		cout << GREEN_START;
		cout << this->cycle / 1000.0 << " us";
		cout << COLOR_RESET;
		return;
	}
	if (this->cycle >= 1000000 && this->cycle < 1000000000)
	{
		cout << YELLOW_START;
		cout << this->cycle / 1000000.0 << " ms";
		cout << COLOR_RESET;
		return;
	}
	if (this->cycle >= 1000000000)
	{
		cout << RED_START;
		cout << this->cycle / 1000000000.0 << " s";
		cout << COLOR_RESET;
		return;
	}
	cout << FATAL_ERROR << "Should not reach here" << endl;
}

void GlobalCounters::sim_time_printer()
{
	// Sim time is in microseconds
	// Print Hours, minutes, seconds
	cout << "Sim Time Raw: " << this->sim_time << endl;
	// Hours
	int hours = this->sim_time / 3600000000;
	// Minutes
	int minutes = (this->sim_time - (hours * 3600000000)) / 60000000;
	// Seconds
	int seconds = (this->sim_time - (hours * 3600000000) - (minutes * 60000000)) / 1000000;

	cout << "Simulation Time: ";
	cout << RED_START << hours << COLOR_RESET << "h::";
	cout << YELLOW_START << minutes << COLOR_RESET << "m::";
	cout << GREEN_START << seconds << COLOR_RESET << "s" << endl;
}

void GlobalCounters::smart_stats()
{
	// Component statistics are authoritative; no duplicate aggregate is emitted.
}

bool GlobalCounters::sanity_check()
{
	// Check if at least one tile is present
	if (this->tile_id == -2)
	{
		cout << SANITY_CHECK << "No Tiles Present" << endl;
		return (false);
	}
	// Check if at least one core is present
	if (this->core_id == -1)
	{
		cout << SANITY_CHECK << "No Cores Present" << endl;
		return (false);
	}
	// Check if at least one memory is present
	if (this->mem_id == -1)
	{
		cout << SANITY_CHECK << "No Memories Present" << endl;
		return (false);
	}
	// Check if at least one router is present
	if (this->router_id == -1)
	{
		cout << SANITY_CHECK << "No Routers Present" << endl;
		// return (false);
	}
	// Check if at least one connection is present
	if (this->conn_id == -1)
	{
		cout << SANITY_CHECK << "No Connections Present" << endl;
		// return (false);
	}
	// Check if at least one dispatcher is present
	if (this->dispatcher_id == -1)
	{
		cout << SANITY_CHECK << "No Dispatchers Present" << endl;
		// return (false);
	}

	// Check if core_id_array is populated
	for (int i = 0; i < this->total_cores; i++)
	{
		if (this->core_id_array[i] == -1)
		{
			cout << SANITY_CHECK << "Core ID Array Not Populated" << endl;
			return (false);
		}
	}

	// Check if tile_id_array is populated
	for (int i = 0; i < this->total_tiles; i++)
	{
		if (this->tile_id_array[i] == -2)
		{
			cout << SANITY_CHECK << "Tile ID Array Not Populated" << endl;
			return (false);
		}
	}

	// Check if mem_id_array is populated
	for (int i = 0; i < this->total_mems; i++)
	{
		if (this->mem_id_array[i] == -1)
		{
			cout << SANITY_CHECK << "Mem ID Array Not Populated" << endl;
			return (false);
		}
	}

	// Check if mc_id_array is populated
	for (int i = 0; i < this->total_mcs; i++)
	{
		if (this->mc_id_array[i] == -1)
		{
			cout << SANITY_CHECK << "MC ID Array Not Populated" << endl;
			return (false);
		}
	}

	// Check if writer_id_array is populated
	for (int i = 0; i < this->total_writers; i++)
	{
		if (this->writer_id_array[i] == -1)
		{
			cout << SANITY_CHECK << "Writer ID Array Not Populated" << endl;
			return (false);
		}
	}

	// Check if dispatcher_id_array is populated
	for (int i = 0; i < this->total_dispatchers; i++)
	{
		if (this->dispatcher_id_array[i] == -1)
		{
			cout << SANITY_CHECK << "Dispatcher ID Array Not Populated" << endl;
			return (false);
		}
	}

	// Check if router_id_array is populated
	for (int i = 0; i < this->total_routers; i++)
	{
		if (this->router_id_array[i] == -1)
		{
			cout << SANITY_CHECK << "Router ID Array Not Populated" << endl;
			return (false);
		}
	}

	// ResultSink Sanity Check
	if (!this->result_sink->sanity_check())
	{
		cout << SANITY_CHECK << "ResultSink Sanity Check Failed" << endl;
		return (false);
	}
	return (true);
}

long long GlobalCounters::get_cycle()
{
	return (this->cycle);
}

void GlobalCounters::post_sim_metrics()
{
	// End sim time is computed in the end_sim_time function within the run_sim function in main.cpp
	// this->end_time = high_resolution_clock::now();
	// this->sim_time = chrono::duration_cast<chrono::microseconds>(this->end_time - this->start_time).count();
	this->mcps = (double)this->cycle / (double)this->sim_time;
	this->cps = ((double)this->cycle * 1000000) / (double)this->sim_time;
	long peak_performance = 2 * (long)this->NEURA_TILE_SIZE;
	AreaModelBreakdown area = estimate_chip_area(this);
	float on_chip_memory = static_cast<float>(area.on_chip_memory_mb);
	long off_chip_memory = 128;
	long technology = 7;
	float chip_area = static_cast<float>(area.total_mm2);
	float spgemm_performance;
	float area_efficiency;
	spgemm_performance = (float)((this->hash_req_id + 1) * 2) / ((float)this->cycle); // * 2 because multiply and add operations
	area_efficiency = (float)spgemm_performance / chip_area;
	
	// Save results to JSON file (works regardless of legacy database status)
	{
		string json_file_name = this->results_dir + "/simulation_results.json";
		ofstream json_file(json_file_name);
		json_file << "{\n";
		json_file << "  \"name\": \"post_sim_metrics\",\n";
		json_file << "  \"exp_name\": \"" << this->exp_name << "\",\n";
		json_file << "  \"tile_size\": " << this->NEURA_TILE_SIZE << ",\n";
		json_file << "  \"peak_performance\": " << peak_performance << ",\n";
		json_file << "  \"on_chip_memory\": " << on_chip_memory << ",\n";
		json_file << "  \"off_chip_memory\": " << off_chip_memory << ",\n";
		json_file << "  \"technology\": " << technology << ",\n";
		json_file << "  \"chip_area\": " << chip_area << ",\n";
		json_file << "  \"chip_area_model\": \"" << area.model_version << "\",\n";
		json_file << "  \"spmm_mode\": \"" << area.spmm_mode << "\",\n";
		json_file << "  \"cim_mode\": \"" << area.cim_mode << "\",\n";
		json_file << "  \"dcim_digital\": " << (area.dcim_digital ? "true" : "false") << ",\n";
		json_file << "  \"indexer_mode\": \"" << area.indexer_mode << "\",\n";
		json_file << "  \"y_dram_policy_enabled\": " << (area.y_dram_enabled ? "true" : "false") << ",\n";
		json_file << "  \"z_dram_policy_enabled\": " << (area.z_dram_enabled ? "true" : "false") << ",\n";
		json_file << "  \"z_dram_rmw_policy\": " << (area.z_rmw_enabled ? "true" : "false") << ",\n";
		json_file << "  \"active_cim_tiles\": " << area.active_cim_tiles << ",\n";
		json_file << "  \"area_sram_mm2\": " << area.sram_mm2 << ",\n";
		json_file << "  \"area_noc_mm2\": " << area.noc_mm2 << ",\n";
		json_file << "  \"area_pe_logic_mm2\": " << area.pe_logic_mm2 << ",\n";
		json_file << "  \"area_cim_mm2\": " << area.cim_mm2 << ",\n";
		json_file << "  \"area_yz_policy_mm2\": " << area.yz_policy_mm2 << ",\n";
		json_file << "  \"area_y_buffer_mm2\": " << area.y_buffer_mm2 << ",\n";
		json_file << "  \"area_z_buffer_mm2\": " << area.z_buffer_mm2 << ",\n";
		json_file << "  \"area_yz_ctrl_mm2\": " << area.yz_ctrl_mm2 << ",\n";
		json_file << "  \"area_f_cache_mm2\": " << area.f_cache_mm2 << ",\n";
		json_file << "  \"area_index_store_mm2\": " << area.index_store_mm2 << ",\n";
		json_file << "  \"area_l1_mru_mm2\": " << area.l1_mru_mm2 << ",\n";
		json_file << "  \"area_pe_reg_mm2\": " << area.pe_reg_mm2 << ",\n";
		json_file << "  \"area_shared_buffer_mm2\": " << area.shared_buffer_mm2 << ",\n";
		json_file << "  \"area_indexer_ctrl_mm2\": " << area.indexer_ctrl_mm2 << ",\n";
		json_file << "  \"y_buffer_bytes\": " << area.y_buffer_bytes << ",\n";
		json_file << "  \"z_buffer_bytes\": " << area.z_buffer_bytes << ",\n";
		json_file << "  \"spgemm_performance\": " << spgemm_performance << ",\n";
		json_file << "  \"area_efficiency\": " << area_efficiency << ",\n";
		json_file << "  \"pipeline_count\": " << this->PIPELINE_COUNT << ",\n";
		json_file << "  \"hashpad_comparator_count\": " << this->HASHPAD_COMPARATOR_COUNT << ",\n";
		json_file << "  \"cycles\": " << this->cycle << ",\n";
		json_file << "  \"sim_time\": " << this->sim_time << ",\n";
		json_file << "  \"mcps\": " << this->mcps << ",\n";
		json_file << "  \"cps\": " << this->cps << ",\n";
		json_file << "  \"total_mem_reqs\": " << (this->mem_req_id + 1) << ",\n";
		json_file << "  \"total_mem_resps\": " << (this->mem_resp_id + 1) << ",\n";
		json_file << "  \"total_hash_reqs\": " << (this->hash_req_id + 1) << ",\n";
		json_file << "  \"total_hash_resps\": " << (this->hash_resp_id + 1) << "\n";
		json_file << "}\n";
		json_file.close();
		cout << "[INFO] Simulation results saved to: " << json_file_name << endl;
	}

	if (MONGO_FLAG)
	{
		this->header_collection.insert_one(make_result_record(
			result_field("name", "post_sim_metrics"),
			result_field("exp_name", this->exp_name),
			result_field("tile_size", (long)this->NEURA_TILE_SIZE),
			result_field("peak_performance", (long)peak_performance),
			result_field("on_chip_memory", (double)on_chip_memory),
			result_field("off_chip_memory", (long)off_chip_memory),
			result_field("technology", (long)technology),
			result_field("chip_area", (double)chip_area),
			result_field("chip_area_model", area.model_version),
			result_field("spmm_mode", area.spmm_mode),
			result_field("cim_mode", area.cim_mode),
			result_field("dcim_digital", area.dcim_digital),
			result_field("indexer_mode", area.indexer_mode),
			result_field("y_dram_policy_enabled", area.y_dram_enabled),
			result_field("z_dram_policy_enabled", area.z_dram_enabled),
			result_field("z_dram_rmw_policy", area.z_rmw_enabled),
			result_field("active_cim_tiles", (long)area.active_cim_tiles),
			result_field("area_sram_mm2", area.sram_mm2),
			result_field("area_noc_mm2", area.noc_mm2),
			result_field("area_pe_logic_mm2", area.pe_logic_mm2),
			result_field("area_cim_mm2", area.cim_mm2),
			result_field("area_yz_policy_mm2", area.yz_policy_mm2),
			result_field("area_y_buffer_mm2", area.y_buffer_mm2),
			result_field("area_z_buffer_mm2", area.z_buffer_mm2),
			result_field("area_yz_ctrl_mm2", area.yz_ctrl_mm2),
			result_field("area_f_cache_mm2", area.f_cache_mm2),
			result_field("area_index_store_mm2", area.index_store_mm2),
			result_field("area_l1_mru_mm2", area.l1_mru_mm2),
			result_field("area_pe_reg_mm2", area.pe_reg_mm2),
			result_field("area_shared_buffer_mm2", area.shared_buffer_mm2),
			result_field("area_indexer_ctrl_mm2", area.indexer_ctrl_mm2),
			result_field("y_buffer_bytes", (long long)area.y_buffer_bytes),
			result_field("z_buffer_bytes", (long long)area.z_buffer_bytes),
			result_field("spgemm_performance", (double)spgemm_performance),
			result_field("area_efficiency", (double)area_efficiency),
			result_field("pipeline_count", (long)this->PIPELINE_COUNT),
			result_field("hashpad_comparator_count", (long)this->HASHPAD_COMPARATOR_COUNT),
			result_field("cycles", (long)this->cycle),
			result_field("sim_time", (double)this->sim_time),
			result_field("mcps", (double)this->mcps),
			result_field("cps", (double)this->cps),
			result_field("total_mem_reqs", (long)(this->mem_req_id + 1)),
			result_field("total_mem_resps", (long)(this->mem_resp_id + 1)),
			result_field("total_hash_reqs", (long)(this->hash_req_id + 1)),
			result_field("total_hash_resps", (long)(this->hash_resp_id + 1))));
	}

	// Create a folder in results directory called neuracore
	string core_results_dir = this->results_dir + "/neuracore";
	CREATE_DIR(core_results_dir.c_str());
	// rf_histogram.csv
	string rf_histogram_file_name = core_results_dir + "/rf_histogram.csv";
	this->write_vector_to_csv(rf_histogram_file_name, this->rf_histogram);
}

void GlobalCounters::end_sim_time()
{
	this->end_time = high_resolution_clock::now();
	this->sim_time = chrono::duration_cast<chrono::microseconds>(this->end_time - this->start_time).count();
}

void GlobalCounters::interval_metrics()
{
	this->interval_end_time = high_resolution_clock::now();
	this->interval_sim_time = chrono::duration_cast<chrono::microseconds>(this->interval_end_time - this->interval_start_time).count();
	this->interval_start_time = high_resolution_clock::now();
	this->interval_mcps = (double)INTERVAL / (double)this->interval_sim_time;
	this->interval_cps = ((double)INTERVAL * 1000000) / (double)this->interval_sim_time;
	if (MONGO_FLAG)
	{
		this->collection.insert_one(make_result_record(
			result_field("name", "interval_metrics"),
			result_field("interval_id", (long)this->interval_id),
			result_field("interval_cycles", (long)INTERVAL),
			result_field("interval_sim_time", (double)this->interval_sim_time),
			result_field("interval_mcps", (double)this->interval_mcps),
			result_field("interval_cps", (double)this->interval_cps),
			result_field("interval_sim_progress", (double)this->interval_sim_progress)));
	}
	this->interval_id++;
}

void GlobalCounters::set_sim_start_time()
{
	this->start_time = high_resolution_clock::now();
	this->interval_start_time = high_resolution_clock::now();
}

void GlobalCounters::print_sim_speed()
{
	cout << "Sim Speed: " << GREEN_START << (int)(this->interval_cps) << COLOR_RESET << " Cycles/s    ";
	cout << "Cycles: ";
	this->cycle_printer();
	// Show progress percentage
	if (this->interval_sim_progress > 0)
	{
		cout << "    Progress: " << CYAN_START << this->interval_sim_progress << "%" << COLOR_RESET;
	}
	cout << endl;
}

void GlobalCounters::cycle_printer()
{
	if (this->cycle < 1000)
	{
		cout << GREEN_START;
		cout << this->cycle;
		cout << COLOR_RESET;
		return;
	}
	if (this->cycle >= 1000 && this->cycle < 1000000)
	{
		cout << YELLOW_START;
		cout << int(this->cycle / 1000) << "K";
		cout << COLOR_RESET;
		return;
	}
	if (this->cycle >= 1000000 && this->cycle < 1000000000)
	{
		cout << RED_START;
		cout << int(this->cycle / 1000000) << "M";
		cout << COLOR_RESET;
		return;
	}
	if (this->cycle >= 1000000000 && this->cycle < 1000000000000)
	{
		cout << CYAN_START;
		cout << int(this->cycle / 1000000000) << "B";
		cout << COLOR_RESET;
		return;
	}
	if (this->cycle >= 1000000000000)
	{
		cout << RED_START;
		cout << int(this->cycle / 1000000000000) << "T";
		cout << COLOR_RESET;
		return;
	}
	cout << FATAL_ERROR << "Should not reach here: ";
	cout << this->cycle;
	return;
}

string GlobalCounters::get_exp_name()
{
	return (this->exp_name);
}

string GlobalCounters::get_program_data_folder()
{
	return (this->program_data_folder);
}

string GlobalCounters::get_dram_data_file_name_0()
{
	return (this->dram_data_file_name_0);
}

string GlobalCounters::get_dram_data_file_name_1()
{
	return (this->dram_data_file_name_1);
}



string GlobalCounters::get_results_dir()
{
	return (this->results_dir);
}

string GlobalCounters::get_dram_config_path()
{
	return (this->arg_parse != nullptr) ? this->arg_parse->get_dram_config_path()
	                                     : "./ktb_dram_configs/ktb_HBM2_8Gb_x128.ini";
}

string GlobalCounters::get_memory_type()
{
	return (this->arg_parse != nullptr) ? this->arg_parse->get_memory_type() : "hbm2";
}

double GlobalCounters::get_memory_bw_gbps()
{
	return (this->arg_parse != nullptr) ? this->arg_parse->get_memory_bw_gbps() : 256.0;
}

int GlobalCounters::get_memory_latency_cycles()
{
	return (this->arg_parse != nullptr) ? this->arg_parse->get_memory_latency_cycles() : 70;
}

double GlobalCounters::get_memory_tck_ns()
{
	return (this->arg_parse != nullptr) ? this->arg_parse->get_memory_tck_ns() : 1.0;
}

double GlobalCounters::get_memory_hit_pj_per_txn()
{
	return (this->arg_parse != nullptr) ? this->arg_parse->get_memory_hit_pj_per_txn() : 358.0;
}

double GlobalCounters::get_memory_miss_pj_per_txn()
{
	return (this->arg_parse != nullptr) ? this->arg_parse->get_memory_miss_pj_per_txn() : 1894.0;
}

bool GlobalCounters::get_cxl_mode()
{
	return (this->arg_parse != nullptr) ? this->arg_parse->get_cxl_mode() : false;
}

double GlobalCounters::get_cxl_link_bw_gbps()
{
	return (this->arg_parse != nullptr) ? this->arg_parse->get_cxl_link_bw_gbps() : 0.0;
}

int GlobalCounters::get_cxl_extra_latency_cycles()
{
	return (this->arg_parse != nullptr) ? this->arg_parse->get_cxl_extra_latency_cycles() : 0;
}

long long GlobalCounters::get_frequency()
{
	return (this->frequency);
}

long long GlobalCounters::get_hbm_frequency()
{
	return (this->HBM_FREQUENCY);
}

void GlobalCounters::set_sim_progress(int progress)
{
	this->interval_sim_progress = progress;
}

bool GlobalCounters::is_router(int uid)
{
	for (int i = 0; i < this->total_routers; i++)
	{
		if (this->router_id_array[i] == uid)
		{
			return (true);
		}
	}
	return (false);
}

long long GlobalCounters::get_clock_frequency()
{
	return (this->frequency);
}

void GlobalCounters::write_vector_to_csv(string file_name, vector<long long> data)
{
	ofstream myfile;
	myfile.open(file_name);
	for (size_t i = 0; i < data.size(); i++)
	{
		myfile << data[i] << endl;
	}
	myfile.close();
}

void GlobalCounters::write_vector_to_csv(string file_name, vector<long> data)
{
	ofstream myfile;
	myfile.open(file_name);
	for (size_t i = 0; i < data.size(); i++)
	{
		myfile << data[i] << endl;
	}
	myfile.close();
}

void GlobalCounters::write_vector_to_csv(string file_name, vector<int> data)
{
	ofstream myfile;
	myfile.open(file_name);
	for (size_t i = 0; i < data.size(); i++)
	{
		myfile << data[i] << endl;
	}
	myfile.close();
}
