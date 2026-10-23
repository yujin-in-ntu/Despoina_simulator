// SPDX-License-Identifier: CC-BY-4.0
// Derived from NeuraChip/NeuraSim (DOI: 10.5281/zenodo.10896280); modified for Despoina and this standalone release in 2026.
// Created by Kaustubh Shivdikar
//

#include "neuratile.h"
#include "../neuracore/neuracore.h"
#include "../debugger/debugger.h"
#include "../config.h"
#include "../neuramem/neuramem.h"
#include "../functions.h"
#include "../connector/conn.h"
#include "../dispatcher/dispatcher.h"
#include "../neurarouter/neurarouter.h"
#include "../mc/memorycontroller.h"
#include "../writer/writer.h"
#include "../energy_params_generated.h"
#include "../official_energy_contract.h"
#include "runtime_export_r4.h"
#include <iomanip>  // For setw, setprecision in streaming output
#include <algorithm>
#include <array>
#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <deque>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <sys/stat.h>
#include <sys/types.h>
#include <unordered_set>
#include <vector>

using namespace std;

constexpr int kStreamingEndpointPort = 0;
constexpr int kStreamingEndpointPortCount = 1;
constexpr int kStreamingFlitFloats = 8;  // 32B payload per flit

namespace {

using despoina_r4_export::RuntimeExportError;
using despoina_r4_export::ensure_dir_recursive;
using despoina_r4_export::vector_hash;
using despoina_r4_export::write_text_checked;
using despoina_r4_export::write_vector_checked;

std::set<int> parse_timestep_set(const std::string& csv) {
	std::set<int> out;
	if (csv.empty()) return out;
	std::stringstream ss(csv);
	std::string token;
	while (std::getline(ss, token, ',')) {
		if (token.empty()) continue;
		std::stringstream ts(token);
		int value = -1;
		ts >> value;
		if (!ts.fail() && value >= 0) out.insert(value);
	}
	return out;
}

}  // namespace

class StreamingNoCEndpoint {
public:
	StreamingNoCEndpoint(GlobalCounters* gcp, const std::string& name) : gcp_(gcp), vertex_name_(name) {
		this->uid_ = gcp_->generate_uid();
		this->in_port_array = new queue<InstX*>[kStreamingEndpointPortCount];
		this->out_port_array = new queue<InstX*>[kStreamingEndpointPortCount];
		this->gcp_->add_vertex_to_arch_graph(this->vertex_name_, this->uid_);
	}

	~StreamingNoCEndpoint() {
		delete[] this->in_port_array;
		delete[] this->out_port_array;
	}

	queue<InstX*>* get_in_port_ref(int port_id) { return &this->in_port_array[port_id]; }
	queue<InstX*>* get_out_port_ref(int port_id) { return &this->out_port_array[port_id]; }
	string get_vertex_name() { return this->vertex_name_; }
	int get_uid() const { return this->uid_; }

	queue<InstX*>* in_port_array;
	queue<InstX*>* out_port_array;

private:
	GlobalCounters* gcp_;
	string vertex_name_;
	int uid_;
};

struct StreamingNoCPacket : public InstX {
	StreamingNoCPacket(int src_uid,
	                  int dest_uid,
	                  int row_id,
	                  bool pe_to_cim,
	                  int flit_index,
	                  int total_flits,
	                  int feature_offset,
	                  std::vector<float>&& payload,
	                  int payload_bytes,
	                  long long inject_cycle)
	    : src_uid(src_uid),
	      dest_uid(dest_uid),
	      row_id(row_id),
	      pe_to_cim(pe_to_cim),
	      flit_index(flit_index),
	      total_flits(total_flits),
	      feature_offset(feature_offset),
	      payload(std::move(payload)),
	      payload_bytes(payload_bytes),
	      inject_cycle(inject_cycle) {
		this->type = INSTX_TYPE_CIM_PACKET;
		this->dest_unit_uid_ptr = &this->dest_uid;
	}

	int src_uid;
	int dest_uid;
	int row_id;
	bool pe_to_cim;
	int flit_index;
	int total_flits;
	int feature_offset;
	std::vector<float> payload;
	int payload_bytes;
	long long inject_cycle;
};
// Constructor
// @param id: NeuraTile ID
// @param gcp: Global Counters Pointer
// @param x: X Coordinate
// @param y: Y Coordinate
// @param port_count: Number of Ports
NeuraTile::NeuraTile(GlobalCounters *gcp, int x, int y, int port_count)
{
	this->custom_constructor(gcp, x, y, port_count);
}

NeuraTile::NeuraTile()
{
}

// Custom Constructor
// @param id: NeuraTile ID
// @param gcp: Global Counters Pointer
// @param x: X Coordinate
// @param y: Y Coordinate
// @param port_count: Number of Ports
void NeuraTile::custom_constructor(GlobalCounters *gcp, int x, int y, int port_count)
{
	// cout << "NeuraTile::custom_constructor()" << endl;
	this->gcp = gcp;
	this->id = this->gcp->generate_tile_id();
	// this->uid = this->gcp->generate_uid();
	// this->gcp->set_tile_id_array(this->id, this->uid);
	this->gcp->set_tile_id_array(this->id, -1);
	this->x_cood = x;
	this->y_cood = y;
	this->core_x_count = this->gcp->NEURA_CORE_X_COUNT;
	this->core_y_count = this->gcp->NEURA_CORE_Y_COUNT;
	this->mem_x_count = this->gcp->NEURA_MEM_X_COUNT;
	this->mem_y_count = this->gcp->NEURA_MEM_Y_COUNT;
	this->mem_controller_count = this->gcp->MC_COUNT;
	this->router_x_count = this->gcp->NEURA_ROUTER_X_COUNT;
	this->router_y_count = this->gcp->NEURA_ROUTER_Y_COUNT;
	this->port_count = this->gcp->NEURA_TILE_PORT_COUNT;
	this->writer_count = this->gcp->WRITER_COUNT;
	this->in_port_array = new queue<InstX *>[port_count];
	this->out_port_array = new queue<InstX *>[port_count];
	this->sim_end = false;
	this->program_file = this->gcp->get_program_file();
	this->busy_cycles = 0;
	this->idle_cycles = 0;
	this->interval_busy_cycles = 0;
	this->interval_idle_cycles = 0;
	this->interval_id = 0;
	this->component_name = "NeuraTile";
	this->header_component_name = "H_" + this->component_name;
	this->collection = this->gcp->result_sink->get_component_collection(this->component_name);
	this->header_collection = this->gcp->result_sink->get_component_collection(this->header_component_name);
	this->vertex_name = "T_" + to_string(this->id);
	this->create_unit_components();
	this->component_sim_end_array = new bool[COMPONENT_COUNT];

	// Connect 1 core to 1 memory Port 0 <-> Port 0
	// this->conn_array.push_back(create_link(this->gcp, &this->core_array[0][0], 0, &this->mem_array[0][0], 0));
	// Connect 1 core to 1 router Port 1 <-> Port 1
	// this->conn_array.push_back(create_link(this->gcp, &this->core_array[0][0], 1, &this->router_array[0][0], 1));

	// Dispatcher-to-core connections are created by the topology generator below.
	// for (auto i = 0; i < this->core_x_count; i++)
	// {
	// 	for (auto j = 0; j < this->core_y_count; j++)
	// 	{
	// 		this->conn_array.push_back(create_link(this->gcp, this->dispatcher, this->core_array[i][j].get_id(), &this->core_array[i][j], this->gcp->INSTX_PORT_INDEX));
	// 	}
	// }

	this->generate_cgra_mesh_topology();

	// =========================================================================
	// CiM Initialization
	// =========================================================================
#ifdef ENABLE_CIM
	this->cim_unit = nullptr;
	this->cim_enabled = false;
	this->spmm_done = false;
	this->cim_started = false;
	this->cim_start_cycle = 0;
	this->cim_weight_load_cycles = 0;
	this->cim_weight_load_dram_requests = 0;
	this->cim_weight_load_bytes = 0;
	// Default dimensions - will be set based on matrix B's y_dim and hidden layer size
	// For GNN: d_in = feature_dim (mat_B y_dim), d_out = hidden_dim (typically 128)
	this->cim_d_in = 128;   // Default, can be overridden
	this->cim_d_out = 128;  // Default hidden layer size
	cout << "[CiM] Module initialized (will start after SpMM completion)" << endl;
#endif

	// =========================================================================
	// PE Array SpMM Initialization (our DyGNN design)
	// =========================================================================
#ifdef ENABLE_PE_ARRAY
	this->pe_array_spmm = new PEArraySpMM();
	this->pe_array_spmm->init(this->gcp);
	this->pe_array_enabled = true;
	this->pe_array_initialized = false;
#ifdef USE_PE_ARRAY_MODE
	// Row-wise Streaming: PE Array ??CiM direct (always enabled)
	this->streaming_rows_completed = 0;
	this->streaming_outputs_completed = 0;
	this->streaming_cim_cycles = 0;
	this->streaming_pe_router_x = -1;
	this->streaming_pe_router_y = -1;
	this->streaming_cim_router_x = -1;
	this->streaming_cim_router_y = -1;
	this->streaming_pe_endpoint = nullptr;
	this->streaming_cim_endpoint = nullptr;
	this->streaming_cim_partial_rows.clear();
	this->streaming_cim_partial_flits.clear();
	this->streaming_cim_expected_flits.clear();
	this->streaming_cim_ready_rows.clear();
	this->streaming_pe_partial_rows.clear();
	this->streaming_pe_partial_flits.clear();
	this->streaming_pe_expected_flits.clear();
	this->streaming_pe_ready_rows.clear();
	this->streaming_pending_outputs.clear();
	this->pe_cim_noc_packets = 0;
	this->pe_cim_noc_flits = 0;
	this->pe_cim_noc_latency_cycles = 0;
	this->pe_cim_noc_output_packets = 0;
	this->pe_array_expected_rows = 0;
	this->dygnn_controller = new DualModeSpMM(this->gcp, this->pe_array_spmm);
	this->dygnn_controller->set_fail_closed_accuracy(
		this->gcp != nullptr && this->gcp->arg_parse != nullptr &&
		this->gcp->arg_parse->is_official_profile());
	// Multi-timestep driver init (t=0 full, t=1..N-1 incremental)
	this->multi_ts_current = 0;
	this->multi_ts_total = 1;
	this->multi_ts_done = false;
	this->multi_ts_full_done = false;
	this->multi_ts_full_cycles = 0;
	this->multi_ts_runtime_cim_energy_pj = 0.0;
	this->multi_ts_eval_cim_energy_pj = 0.0;
	this->multi_ts_eval_energy_valid = false;
#ifdef ENABLE_CIM
	if (!this->setup_streaming_noc_endpoints()) {
		cerr << "[Streaming] Warning: failed to reserve router endpoints for PE-CiM traffic" << endl;
	}
#endif
	cout << "[PE Array] SpMM module initialized (256 PEs, 64 clusters, STREAMING)" << endl;
#else
	cout << "[PE Array] SpMM module initialized (256 PEs, 64 clusters, work stealing)" << endl;
#endif

	// =========================================================================
	// Activation Unit Initialization
	// =========================================================================
#ifdef ENABLE_ACTIVATION
	this->activation_unit = new ActivationUnit();
	this->activation_unit->init(ActivationType::RELU);
	this->activation_enabled = true;
	this->activation_started = false;
	this->activation_done = false;
	this->activation_start_cycle = 0;
	this->activation_cycles = 0;
	cout << "[Activation] Unit initialized (ReLU, 16 lanes)" << endl;
#endif

	// =========================================================================
	// H_agg Buffer Initialization (SpMM ??CiM data transfer)
	// =========================================================================
#ifdef ENABLE_HAGG_BUFFER
	this->hagg_buffer = new SRAMBuffer();
	const int hagg_capacity_kb = static_cast<int>(pe_hagg_staging::kBytes / 1024u);
#ifdef ENABLE_DOUBLE_BUFFER
	this->hagg_buffer->init(hagg_capacity_kb, true);
	cout << "[H_agg Buffer] Initialized (" << hagg_capacity_kb << "KB total; capacity enforced by completed-row policy)" << endl;
#else
	this->hagg_buffer->init(hagg_capacity_kb, false);
	cout << "[H_agg Buffer] Initialized (" << hagg_capacity_kb << "KB; capacity enforced by completed-row policy)" << endl;
#endif
	this->hagg_buffer_enabled = true;
	this->hagg_buffer_stall_cycles = 0;
	this->hagg_buffer_write_cycles = 0;
	this->hagg_buffer_read_cycles = 0;
	this->hagg_noc_cycles = 0;
	this->hagg_sram_energy_pj = 0.0;
	this->hagg_noc_energy_pj = 0.0;
	// hagg_row_id_queue is default-constructed (empty)
#endif
#endif // ENABLE_PE_ARRAY
	// =========================================================================

	// Unit connections are created by the topology generator.
	// this->connect_units("C_0_0", 0, "M_0_0", 1);

	// Connect NeuraMem Port 1 ro Memory Controller Port 0
	// this->conn_array.push_back(create_link(this->gcp, &this->mem_array[0][0], 1, this->mem_controller, 0));

	// Connect NeuraMem Port 2 to Writer Port 0
	// this->conn_array.push_back(create_link(this->gcp, &this->mem_array[0][0], 2, &this->writer_array[0], 0));

	// Connect Writer Port 1 to Memory Controller Port 1
	// this->conn_array.push_back(create_link(this->gcp, &this->writer_array[0], 1, this->mem_controller, 1));

	// Connect Core Port 2 to Writer Port 2
	// this->conn_array.push_back(create_link(this->gcp, &this->core_array[0][0], 2, &this->writer_array[0], 2));

	this->create_arch_graph_distances_for_all_units();

	// this->gcp->visualize_arch_graph();

	// this->print_unit_distances();

	// This has to be called after all connections are created
	this->enable_max_buffer_sizes_computation();
}

void NeuraTile::create_unit_components()
{
	// Create a 2D core array
	this->core_array = new NeuraCore *[this->core_x_count];
	for (int i = 0; i < this->core_x_count; i++)
	{
		this->core_array[i] = new NeuraCore[this->core_y_count];
		for (int j = 0; j < this->core_y_count; j++)
		{
			this->core_array[i][j].custom_constructor(this->gcp, i, j, this->gcp->NEURA_CORE_PORT_COUNT);
		}
	}

	// Create 2D array of memory
	this->mem_array = new NeuraMem *[this->mem_x_count];
	for (int i = 0; i < this->mem_x_count; i++)
	{
		this->mem_array[i] = new NeuraMem[this->mem_y_count];
		for (int j = 0; j < this->mem_y_count; j++)
		{
			this->mem_array[i][j].custom_constructor(this->gcp, i, j, this->gcp->NEURA_MEM_PORT_COUNT);
		}
	}

	// Create 2D array of Routers
	this->router_array = new NeuraRouter *[this->router_x_count];
	for (int i = 0; i < this->router_x_count; i++)
	{
		this->router_array[i] = new NeuraRouter[this->router_y_count];
		for (int j = 0; j < this->router_y_count; j++)
		{
			this->router_array[i][j].custom_constructor(this->gcp, i, j);
		}
	}

	// Create 1D array of Writers
	this->writer_array = new Writer[this->writer_count];
	for (int i = 0; i < this->writer_count; i++)
	{
		this->writer_array[i].custom_constructor(this->gcp, i);
	}

	// Create 1D Memory Controller array
	this->mem_controller_array = new MemoryController[this->mem_controller_count];
	// this->mem_controller = new MemoryController(this->gcp);
	for (int i = 0; i < this->mem_controller_count; i++)
	{
		this->mem_controller_array[i].custom_constructor(this->gcp, i);
	}

	// The standalone topology uses one shared dispatcher.
	// this->dispatcher_array = new NeuraDispatcher *[this->core_x_count];
	// for (int i = 0; i < this->core_x_count; i++)
	// {
	// 	this->dispatcher_array[i] = new NeuraDispatcher[this->core_y_count];
	// 	for (int j = 0; j < this->core_y_count; j++)
	// 	{
	// 		// Each dispatcher would require its own program file.
	// 		// Use a different program file for each dispatcher
	// 		this->dispatcher_array[i][j].custom_constructor(this->gcp, this->program_file, this->gcp->DISPATCHER_COUNT,  i, j);
	// 		// A per-dispatcher row count would be required.
	// 		this->dispatcher_array[i][j].set_dispatch_count(this->gcp->DISPATCH_INSTX_COUNT);
	// 	}
	// }
	// this->dispatcher = new NeuraDispatcher();

	int total_core_count = this->core_x_count * this->core_y_count;
	this->dispatcher.custom_constructor(this->gcp, this->program_file, total_core_count);
	// For debugging, just dispatch 1 instruction
	this->dispatcher.set_dispatch_count(this->gcp->DISPATCH_INSTX_COUNT);
}

void NeuraTile::create_arch_graph_distances_for_all_units()
{
	// cout << "Creating arch graph distances for all units" << endl;

	// Create arch graph distances for the 2D core array
	for (int i = 0; i < this->core_x_count; i++)
	{
		for (int j = 0; j < this->core_y_count; j++)
		{
			this->core_array[i][j].create_arch_graph_distances();
		}
	}

	// Create arch graph distances for the 2D memory array
	for (int i = 0; i < this->mem_x_count; i++)
	{
		for (int j = 0; j < this->mem_y_count; j++)
		{
			this->mem_array[i][j].create_arch_graph_distances();
		}
	}

	// Create arch graph distances for the 2D router array
	for (int i = 0; i < this->router_x_count; i++)
	{
		for (int j = 0; j < this->router_y_count; j++)
		{
			this->router_array[i][j].create_arch_graph_distances();
		}
	}

	// Create arch graph distances for the 1D writer array
	for (int i = 0; i < this->writer_count; i++)
	{
		this->writer_array[i].create_arch_graph_distances();
	}

	// Create arch graph distances for the 1D memory controller array
	for (int i = 0; i < this->mem_controller_count; i++)
	{
		this->mem_controller_array[i].create_arch_graph_distances();
	}

	// Create arch graph distances for dispatcher
	this->dispatcher.create_arch_graph_distances();
}

void NeuraTile::print_unit_distances()
{
	cout << "Arch Distances" << endl;

	// Create arch graph distances for the 2D core array
	for (int i = 0; i < this->core_x_count; i++)
	{
		for (int j = 0; j < this->core_y_count; j++)
		{
			cout << "Core[" << i << "][" << j << "]: ";
			for (size_t k = 0; k < this->core_array[i][j].arch_graph_distances.size(); k++)
			{
				cout << k << ":" << this->core_array[i][j].arch_graph_distances[k] << " ";
			}
			cout << endl;
		}
	}

	// Create arch graph distances for the 2D memory array
	for (int i = 0; i < this->mem_x_count; i++)
	{
		for (int j = 0; j < this->mem_y_count; j++)
		{
			cout << "Mem[" << i << "][" << j << "]: ";
			for (size_t k = 0; k < this->mem_array[i][j].arch_graph_distances.size(); k++)
			{
				cout << k << ":" << this->mem_array[i][j].arch_graph_distances[k] << " ";
			}
			cout << endl;
		}
	}

	// Create arch graph distances for the 2D router array
	for (int i = 0; i < this->router_x_count; i++)
	{
		for (int j = 0; j < this->router_y_count; j++)
		{
			cout << "Router[" << i << "][" << j << "]: ";
			for (size_t k = 0; k < this->router_array[i][j].arch_graph_distances.size(); k++)
			{
				cout << k << ":" << this->router_array[i][j].arch_graph_distances[k] << " ";
			}
			cout << endl;
		}
	}

	// Create arch graph distances for the 1D writer array
	for (int i = 0; i < this->writer_count; i++)
	{
		cout << "Writer[" << i << "]: ";
		for (size_t k = 0; k < this->writer_array[i].arch_graph_distances.size(); k++)
		{
			cout << k << ":" << this->writer_array[i].arch_graph_distances[k] << " ";
		}
		cout << endl;
	}

	// Create arch graph distances for the 1D memory controller array
	for (int i = 0; i < this->mem_controller_count; i++)
	{
		cout << "MemController[" << i << "]: ";
		for (size_t k = 0; k < this->mem_controller_array[i].arch_graph_distances.size(); k++)
		{
			cout << k << ":" << this->mem_controller_array[i].arch_graph_distances[k] << " ";
		}
		cout << endl;
	}

	// Create arch graph distances for the 1D dispatcher array
	cout << "Dispatcher: ";
	for (size_t k = 0; k < this->dispatcher.arch_graph_distances.size(); k++)
	{
		cout << k << ":" << this->dispatcher.arch_graph_distances[k] << " ";
	}
	cout << endl;
}

// Unit Connector
void NeuraTile::connect_units(string unit1, int port1, string unit2, int port2)
{
	// cout << "Connecting " << unit1 << " to " << unit2 << endl;
	// Unit 1
	char unit1_type = unit1[0];
	int unit1_x = stoi(unit1.substr(2, unit1.find("_", 2)));
	int unit1_y = -1;
	if (unit1_type == 'C' || unit1_type == 'M' || unit1_type == 'R')
	{ // Not including writer, mem controller, or dispatcher
		unit1_y = stoi(unit1.substr(unit1.find("_", 2) + 1, unit1.length()));
	}
	// Unit 2
	char unit2_type = unit2[0];
	int unit2_x = stoi(unit2.substr(2, unit2.find("_", 2)));
	int unit2_y = -1;
	if (unit2_type == 'C' || unit2_type == 'M' || unit2_type == 'R')
	{ // Not including writer, mem controller, or dispatcher
		unit2_y = stoi(unit2.substr(unit2.find("_", 2) + 1, unit2.length()));
	}

	// // C
	// NeuraCore *core_ptr;
	// // M
	// NeuraMem *mem_ptr;
	// // X
	// MemoryController *mem_controller_ptr;
	// // W
	// Writer *writer_ptr;
	// // R
	// NeuraRouter *router_ptr;
	// // D
	// NeuraDispatcher *dispatcher_ptr;

	// this->conn_array.push_back(create_link(this->gcp, &this->mem_array[0][0], 1, this->mem_controller_array[this->id]], 0));

	if (unit1_type == 'C')
	{
		if (unit2_type == 'C')
		{
			this->gcp->add_edge_to_arch_graph(unit1, unit2, PENALTY_WEIGHT, PENALTY_WEIGHT);
			this->conn_array.push_back(create_link(this->gcp, &this->core_array[unit1_x][unit1_y], port1, &this->core_array[unit2_x][unit2_y], port2));
			this->core_array[unit1_x][unit1_y].uid_to_port_map[this->core_array[unit2_x][unit2_x].get_uid()] = port1;
			this->core_array[unit2_x][unit2_y].uid_to_port_map[this->core_array[unit1_y][unit1_y].get_uid()] = port2;
		}
		else if (unit2_type == 'M')
		{
			this->gcp->add_edge_to_arch_graph(unit1, unit2, PENALTY_WEIGHT, PENALTY_WEIGHT);
			this->conn_array.push_back(create_link(this->gcp, &this->core_array[unit1_x][unit1_y], port1, &this->mem_array[unit2_x][unit2_y], port2));
			this->core_array[unit1_x][unit1_y].uid_to_port_map[this->mem_array[unit2_x][unit2_y].get_uid()] = port1;
			this->mem_array[unit2_x][unit2_y].uid_to_port_map[this->core_array[unit1_x][unit1_y].get_uid()] = port2;
		}
		else if (unit2_type == 'X')
		{
			this->gcp->add_edge_to_arch_graph(unit1, unit2, PENALTY_WEIGHT, PENALTY_WEIGHT);
			this->conn_array.push_back(create_link(this->gcp, &this->core_array[unit1_x][unit1_y], port1, &this->mem_controller_array[this->id], port2));
			this->core_array[unit1_x][unit1_y].uid_to_port_map[this->mem_controller_array[this->id].get_uid()] = port1;
			this->mem_controller_array[this->id].uid_to_port_map[this->core_array[unit1_x][unit1_y].get_uid()] = port2;
		}
		else if (unit2_type == 'W')
		{
			this->gcp->add_edge_to_arch_graph(unit1, unit2, PENALTY_WEIGHT, PENALTY_WEIGHT);
			this->conn_array.push_back(create_link(this->gcp, &this->core_array[unit1_x][unit1_y], port1, &this->writer_array[unit2_x], port2));
			this->core_array[unit1_x][unit1_y].uid_to_port_map[this->writer_array[unit2_x].get_uid()] = port1;
			this->writer_array[unit2_x].uid_to_port_map[this->core_array[unit1_x][unit1_y].get_uid()] = port2;
		}
		else if (unit2_type == 'R')
		{
			this->gcp->add_edge_to_arch_graph(unit1, unit2, STANDARD_WEIGHT, PENALTY_WEIGHT);
			this->conn_array.push_back(create_link(this->gcp, &this->core_array[unit1_x][unit1_y], port1, &this->router_array[unit2_x][unit2_y], port2));
			this->core_array[unit1_x][unit1_y].uid_to_port_map[this->router_array[unit2_x][unit2_y].get_uid()] = port1;
			this->router_array[unit2_x][unit2_y].uid_to_port_map[this->core_array[unit1_x][unit1_y].get_uid()] = port2;
		}
		else if (unit2_type == 'D')
		{
			this->gcp->add_edge_to_arch_graph(unit1, unit2, PENALTY_WEIGHT, PENALTY_WEIGHT);
			this->conn_array.push_back(create_link(this->gcp, &this->core_array[unit1_x][unit1_y], port1, &this->dispatcher, port2));
			this->core_array[unit1_x][unit1_y].uid_to_port_map[this->dispatcher.get_uid()] = port1;
			this->dispatcher.uid_to_port_map[this->core_array[unit1_x][unit1_y].get_uid()] = port2;
			this->dispatcher.port_to_uid_map[port2] = this->core_array[unit1_x][unit1_y].get_uid();
		}
		else
		{
			cout << FATAL_ERROR << "Error creating link between " << unit1_type << " and " << unit2_type << endl;
		}
	}
	else if (unit1_type == 'M')
	{
		if (unit2_type == 'C')
		{
			this->gcp->add_edge_to_arch_graph(unit1, unit2, PENALTY_WEIGHT, PENALTY_WEIGHT);
			this->conn_array.push_back(create_link(this->gcp, &this->mem_array[unit1_x][unit1_y], port1, &this->core_array[unit2_x][unit2_y], port2));
			this->mem_array[unit1_x][unit1_y].uid_to_port_map[this->core_array[unit2_x][unit2_y].get_uid()] = port1;
			this->core_array[unit2_x][unit2_y].uid_to_port_map[this->mem_array[unit1_x][unit1_y].get_uid()] = port2;
		}
		else if (unit2_type == 'M')
		{
			this->gcp->add_edge_to_arch_graph(unit1, unit2, PENALTY_WEIGHT, PENALTY_WEIGHT);
			this->conn_array.push_back(create_link(this->gcp, &this->mem_array[unit1_x][unit1_y], port1, &this->mem_array[unit2_x][unit2_y], port2));
			this->mem_array[unit1_x][unit1_y].uid_to_port_map[this->mem_array[unit2_x][unit2_y].get_uid()] = port1;
			this->mem_array[unit2_x][unit2_y].uid_to_port_map[this->mem_array[unit1_x][unit1_y].get_uid()] = port2;
		}
		else if (unit2_type == 'X')
		{
			this->gcp->add_edge_to_arch_graph(unit1, unit2, PENALTY_WEIGHT, PENALTY_WEIGHT);
			this->conn_array.push_back(create_link(this->gcp, &this->mem_array[unit1_x][unit1_y], port1, &this->mem_controller_array[this->id], port2));
			this->mem_array[unit1_x][unit1_y].uid_to_port_map[this->mem_controller_array[this->id].get_uid()] = port1;
			this->mem_controller_array[this->id].uid_to_port_map[this->mem_array[unit1_x][unit1_y].get_uid()] = port2;
		}
		else if (unit2_type == 'W')
		{
			this->gcp->add_edge_to_arch_graph(unit1, unit2, PENALTY_WEIGHT, PENALTY_WEIGHT);
			this->conn_array.push_back(create_link(this->gcp, &this->mem_array[unit1_x][unit1_y], port1, &this->writer_array[unit2_x], port2));
			this->mem_array[unit1_x][unit1_y].uid_to_port_map[this->writer_array[unit2_x].get_uid()] = port1;
			this->writer_array[unit2_x].uid_to_port_map[this->mem_array[unit1_x][unit1_y].get_uid()] = port2;
		}
		else if (unit2_type == 'R')
		{
			this->gcp->add_edge_to_arch_graph(unit1, unit2, STANDARD_WEIGHT, PENALTY_WEIGHT);
			this->conn_array.push_back(create_link(this->gcp, &this->mem_array[unit1_x][unit1_y], port1, &this->router_array[unit2_x][unit2_y], port2));
			this->mem_array[unit1_x][unit1_y].uid_to_port_map[this->router_array[unit2_x][unit2_y].get_uid()] = port1;
			this->router_array[unit2_x][unit2_y].uid_to_port_map[this->mem_array[unit1_x][unit1_y].get_uid()] = port2;
		}
		else if (unit2_type == 'D')
		{
			this->gcp->add_edge_to_arch_graph(unit1, unit2, PENALTY_WEIGHT, PENALTY_WEIGHT);
			this->conn_array.push_back(create_link(this->gcp, &this->mem_array[unit1_x][unit1_y], port1, &this->dispatcher, port2));
			this->mem_array[unit1_x][unit1_y].uid_to_port_map[this->dispatcher.get_uid()] = port1;
			this->dispatcher.uid_to_port_map[this->mem_array[unit1_x][unit1_y].get_uid()] = port2;
			this->dispatcher.port_to_uid_map[port2] = this->mem_array[unit1_x][unit1_y].get_uid();
		}
		else
		{
			cout << FATAL_ERROR << "Error creating link between " << unit1_type << " and " << unit2_type << endl;
		}
	}
	else if (unit1_type == 'X')
	{
		if (unit2_type == 'C')
		{
			this->gcp->add_edge_to_arch_graph(unit1, unit2, PENALTY_WEIGHT, PENALTY_WEIGHT);
			this->conn_array.push_back(create_link(this->gcp, &this->mem_controller_array[this->id], port1, &this->core_array[unit2_x][unit2_y], port2));
			this->mem_controller_array[this->id].uid_to_port_map[this->core_array[unit2_x][unit2_y].get_uid()] = port1;
			this->core_array[unit2_x][unit2_y].uid_to_port_map[this->mem_controller_array[this->id].get_uid()] = port2;
		}
		else if (unit2_type == 'M')
		{
			this->gcp->add_edge_to_arch_graph(unit1, unit2, PENALTY_WEIGHT, PENALTY_WEIGHT);
			this->conn_array.push_back(create_link(this->gcp, &this->mem_controller_array[this->id], port1, &this->mem_array[unit2_x][unit2_y], port2));
			this->mem_controller_array[this->id].uid_to_port_map[this->mem_array[unit2_x][unit2_y].get_uid()] = port1;
			this->mem_array[unit2_x][unit2_y].uid_to_port_map[this->mem_controller_array[this->id].get_uid()] = port2;
		}
		else if (unit2_type == 'X')
		{
			this->gcp->add_edge_to_arch_graph(unit1, unit2, PENALTY_WEIGHT, PENALTY_WEIGHT);
			this->conn_array.push_back(create_link(this->gcp, &this->mem_controller_array[this->id], port1, &this->mem_controller_array[this->id], port2));
			this->mem_controller_array[this->id].uid_to_port_map[this->mem_controller_array[this->id].get_uid()] = port1;
			this->mem_controller_array[this->id].uid_to_port_map[this->mem_controller_array[this->id].get_uid()] = port2;
		}
		else if (unit2_type == 'W')
		{
			this->gcp->add_edge_to_arch_graph(unit1, unit2, PENALTY_WEIGHT, PENALTY_WEIGHT);
			this->conn_array.push_back(create_link(this->gcp, &this->mem_controller_array[this->id], port1, &this->writer_array[unit2_x], port2));
			this->mem_controller_array[this->id].uid_to_port_map[this->writer_array[unit2_x].get_uid()] = port1;
			this->writer_array[unit2_x].uid_to_port_map[this->mem_controller_array[this->id].get_uid()] = port2;
		}
		else if (unit2_type == 'R')
		{
			this->gcp->add_edge_to_arch_graph(unit1, unit2, STANDARD_WEIGHT, PENALTY_WEIGHT);
			this->conn_array.push_back(create_link(this->gcp, &this->mem_controller_array[this->id], port1, &this->router_array[unit2_x][unit2_y], port2));
			this->mem_controller_array[this->id].uid_to_port_map[this->router_array[unit2_x][unit2_y].get_uid()] = port1;
			this->router_array[unit2_x][unit2_y].uid_to_port_map[this->mem_controller_array[this->id].get_uid()] = port2;
		}
		else if (unit2_type == 'D')
		{
			this->gcp->add_edge_to_arch_graph(unit1, unit2, PENALTY_WEIGHT, PENALTY_WEIGHT);
			this->conn_array.push_back(create_link(this->gcp, &this->mem_controller_array[this->id], port1, &this->dispatcher, port2));
			this->mem_controller_array[this->id].uid_to_port_map[this->dispatcher.get_uid()] = port1;
			this->dispatcher.uid_to_port_map[this->mem_controller_array[this->id].get_uid()] = port2;
			this->dispatcher.port_to_uid_map[port2] = this->mem_controller_array[this->id].get_uid();
		}
		else
		{
			cout << FATAL_ERROR << "Error creating link between " << unit1_type << " and " << unit2_type << endl;
		}
	}
	else if (unit1_type == 'W')
	{
		if (unit2_type == 'C')
		{
			this->gcp->add_edge_to_arch_graph(unit1, unit2, PENALTY_WEIGHT, PENALTY_WEIGHT);
			this->conn_array.push_back(create_link(this->gcp, &this->writer_array[unit1_x], port1, &this->core_array[unit2_x][unit2_y], port2));
			this->writer_array[unit1_x].uid_to_port_map[this->core_array[unit2_x][unit2_y].get_uid()] = port1;
			this->core_array[unit2_x][unit2_y].uid_to_port_map[this->writer_array[unit1_x].get_uid()] = port2;
		}
		else if (unit2_type == 'M')
		{
			this->gcp->add_edge_to_arch_graph(unit1, unit2, PENALTY_WEIGHT, PENALTY_WEIGHT);
			this->conn_array.push_back(create_link(this->gcp, &this->writer_array[unit1_x], port1, &this->mem_array[unit2_x][unit2_y], port2));
			this->writer_array[unit1_x].uid_to_port_map[this->mem_array[unit2_x][unit2_y].get_uid()] = port1;
			this->mem_array[unit2_x][unit2_y].uid_to_port_map[this->writer_array[unit1_x].get_uid()] = port2;
		}
		else if (unit2_type == 'X')
		{
			this->gcp->add_edge_to_arch_graph(unit1, unit2, PENALTY_WEIGHT, PENALTY_WEIGHT);
			this->conn_array.push_back(create_link(this->gcp, &this->writer_array[unit1_x], port1, &this->mem_controller_array[this->id], port2));
			this->writer_array[unit1_x].uid_to_port_map[this->mem_controller_array[this->id].get_uid()] = port1;
			this->mem_controller_array[this->id].uid_to_port_map[this->writer_array[unit1_x].get_uid()] = port2;
		}
		else if (unit2_type == 'W')
		{
			this->gcp->add_edge_to_arch_graph(unit1, unit2, PENALTY_WEIGHT, PENALTY_WEIGHT);
			this->conn_array.push_back(create_link(this->gcp, &this->writer_array[unit1_x], port1, &this->writer_array[unit2_x], port2));
			this->writer_array[unit1_x].uid_to_port_map[this->writer_array[unit2_x].get_uid()] = port1;
			this->writer_array[unit2_x].uid_to_port_map[this->writer_array[unit1_x].get_uid()] = port2;
		}
		else if (unit2_type == 'R')
		{
			this->gcp->add_edge_to_arch_graph(unit1, unit2, STANDARD_WEIGHT, PENALTY_WEIGHT);
			this->conn_array.push_back(create_link(this->gcp, &this->writer_array[unit1_x], port1, &this->router_array[unit2_x][unit2_y], port2));
			this->writer_array[unit1_x].uid_to_port_map[this->router_array[unit2_x][unit2_y].get_uid()] = port1;
			this->router_array[unit2_x][unit2_y].uid_to_port_map[this->writer_array[unit1_x].get_uid()] = port2;
		}
		else if (unit2_type == 'D')
		{
			this->gcp->add_edge_to_arch_graph(unit1, unit2, PENALTY_WEIGHT, PENALTY_WEIGHT);
			this->conn_array.push_back(create_link(this->gcp, &this->writer_array[unit1_x], port1, &this->dispatcher, port2));
			this->writer_array[unit1_x].uid_to_port_map[this->dispatcher.get_uid()] = port1;
			this->dispatcher.uid_to_port_map[this->writer_array[unit1_x].get_uid()] = port2;
			this->dispatcher.port_to_uid_map[port2] = this->writer_array[unit1_x].get_uid();
		}
		else
		{
			cout << FATAL_ERROR << "Error creating link between " << unit1_type << " and " << unit2_type << endl;
		}
	}
	else if (unit1_type == 'R')
	{
		if (unit2_type == 'C')
		{
			this->gcp->add_edge_to_arch_graph(unit1, unit2, PENALTY_WEIGHT, STANDARD_WEIGHT);
			this->conn_array.push_back(create_link(this->gcp, &this->router_array[unit1_x][unit1_y], port1, &this->core_array[unit2_x][unit2_y], port2));
			this->router_array[unit1_x][unit1_y].uid_to_port_map[this->core_array[unit2_x][unit2_y].get_uid()] = port1;
			this->core_array[unit2_x][unit2_y].uid_to_port_map[this->router_array[unit1_x][unit1_y].get_uid()] = port2;
		}
		else if (unit2_type == 'M')
		{
			this->gcp->add_edge_to_arch_graph(unit1, unit2, PENALTY_WEIGHT, STANDARD_WEIGHT);
			this->conn_array.push_back(create_link(this->gcp, &this->router_array[unit1_x][unit1_y], port1, &this->mem_array[unit2_x][unit2_y], port2));
			this->router_array[unit1_x][unit1_y].uid_to_port_map[this->mem_array[unit2_x][unit2_y].get_uid()] = port1;
			this->mem_array[unit2_x][unit2_y].uid_to_port_map[this->router_array[unit1_x][unit1_y].get_uid()] = port2;
		}
		else if (unit2_type == 'X')
		{
			this->gcp->add_edge_to_arch_graph(unit1, unit2, PENALTY_WEIGHT, STANDARD_WEIGHT);
			this->conn_array.push_back(create_link(this->gcp, &this->router_array[unit1_x][unit1_y], port1, &this->mem_controller_array[this->id], port2));
			this->router_array[unit1_x][unit1_y].uid_to_port_map[this->mem_controller_array[this->id].get_uid()] = port1;
			this->mem_controller_array[this->id].uid_to_port_map[this->router_array[unit1_x][unit1_y].get_uid()] = port2;
		}
		else if (unit2_type == 'W')
		{
			this->gcp->add_edge_to_arch_graph(unit1, unit2, PENALTY_WEIGHT, STANDARD_WEIGHT);
			this->conn_array.push_back(create_link(this->gcp, &this->router_array[unit1_x][unit1_y], port1, &this->writer_array[unit2_x], port2));
			this->router_array[unit1_x][unit1_y].uid_to_port_map[this->writer_array[unit2_x].get_uid()] = port1;
			this->writer_array[unit2_x].uid_to_port_map[this->router_array[unit1_x][unit1_y].get_uid()] = port2;
		}
		else if (unit2_type == 'R')
		{
			this->gcp->add_edge_to_arch_graph(unit1, unit2, PENALTY_WEIGHT, STANDARD_WEIGHT);
			this->conn_array.push_back(create_link(this->gcp, &this->router_array[unit1_x][unit1_y], port1, &this->router_array[unit2_x][unit2_y], port2));
			this->router_array[unit1_x][unit1_y].uid_to_port_map[this->router_array[unit2_x][unit2_y].get_uid()] = port1;
			this->router_array[unit2_x][unit2_y].uid_to_port_map[this->router_array[unit1_x][unit1_y].get_uid()] = port2;
		}
		else if (unit2_type == 'D')
		{
			this->gcp->add_edge_to_arch_graph(unit1, unit2, PENALTY_WEIGHT, STANDARD_WEIGHT);
			this->conn_array.push_back(create_link(this->gcp, &this->router_array[unit1_x][unit1_y], port1, &this->dispatcher, port2));
			this->router_array[unit1_x][unit1_y].uid_to_port_map[this->dispatcher.get_uid()] = port1;
			this->dispatcher.uid_to_port_map[this->router_array[unit1_x][unit1_y].get_uid()] = port2;
			this->dispatcher.port_to_uid_map[port2] = this->router_array[unit1_x][unit1_y].get_uid();
		}
		else
		{
			cout << FATAL_ERROR << "Error creating link between " << unit1_type << " and " << unit2_type << endl;
		}
	}
	else if (unit1_type == 'D')
	{
		if (unit2_type == 'C')
		{
			this->gcp->add_edge_to_arch_graph(unit1, unit2, PENALTY_WEIGHT, PENALTY_WEIGHT);
			this->conn_array.push_back(create_link(this->gcp, &this->dispatcher, port1, &this->core_array[unit2_x][unit2_y], port2));
			this->dispatcher.uid_to_port_map[this->core_array[unit2_x][unit2_y].get_uid()] = port1;
			this->dispatcher.port_to_uid_map[port1] = this->core_array[unit2_x][unit2_y].get_uid();
			this->core_array[unit2_x][unit2_y].uid_to_port_map[this->dispatcher.get_uid()] = port2;
		}
		else if (unit2_type == 'M')
		{
			this->gcp->add_edge_to_arch_graph(unit1, unit2, PENALTY_WEIGHT, PENALTY_WEIGHT);
			this->conn_array.push_back(create_link(this->gcp, &this->dispatcher, port1, &this->mem_array[unit2_x][unit2_y], port2));
			this->dispatcher.uid_to_port_map[this->mem_array[unit2_x][unit2_y].get_uid()] = port1;
			this->dispatcher.port_to_uid_map[port1] = this->mem_array[unit2_x][unit2_y].get_uid();
			this->mem_array[unit2_x][unit2_y].uid_to_port_map[this->dispatcher.get_uid()] = port2;
		}
		else if (unit2_type == 'X')
		{
			this->gcp->add_edge_to_arch_graph(unit1, unit2, PENALTY_WEIGHT, PENALTY_WEIGHT);
			this->conn_array.push_back(create_link(this->gcp, &this->dispatcher, port1, &this->mem_controller_array[this->id], port2));
			this->dispatcher.uid_to_port_map[this->mem_controller_array[this->id].get_uid()] = port1;
			this->dispatcher.port_to_uid_map[port1] = this->mem_controller_array[this->id].get_uid();
			this->mem_controller_array[this->id].uid_to_port_map[this->dispatcher.get_uid()] = port2;
		}
		else if (unit2_type == 'W')
		{
			this->gcp->add_edge_to_arch_graph(unit1, unit2, PENALTY_WEIGHT, PENALTY_WEIGHT);
			this->conn_array.push_back(create_link(this->gcp, &this->dispatcher, port1, &this->writer_array[unit2_x], port2));
			this->dispatcher.uid_to_port_map[this->writer_array[unit2_x].get_uid()] = port1;
			this->dispatcher.port_to_uid_map[port1] = this->writer_array[unit2_x].get_uid();
			this->writer_array[unit2_x].uid_to_port_map[this->dispatcher.get_uid()] = port2;
		}
		else if (unit2_type == 'R')
		{
			this->gcp->add_edge_to_arch_graph(unit1, unit2, STANDARD_WEIGHT, PENALTY_WEIGHT);
			this->conn_array.push_back(create_link(this->gcp, &this->dispatcher, port1, &this->router_array[unit2_x][unit2_y], port2));
			this->dispatcher.uid_to_port_map[this->router_array[unit2_x][unit2_y].get_uid()] = port1;
			this->dispatcher.port_to_uid_map[port1] = this->router_array[unit2_x][unit2_y].get_uid();
			this->router_array[unit2_x][unit2_y].uid_to_port_map[this->dispatcher.get_uid()] = port2;
		}
		else if (unit2_type == 'D')
		{
			this->gcp->add_edge_to_arch_graph(unit1, unit2, PENALTY_WEIGHT, PENALTY_WEIGHT);
			this->conn_array.push_back(create_link(this->gcp, &this->dispatcher, port1, &this->dispatcher, port2));
			this->dispatcher.uid_to_port_map[this->dispatcher.get_uid()] = port1;
			this->dispatcher.uid_to_port_map[this->dispatcher.get_uid()] = port2;
			this->dispatcher.port_to_uid_map[port1] = this->dispatcher.get_uid();
			this->dispatcher.port_to_uid_map[port2] = this->dispatcher.get_uid();
		}
		else
		{
			cout << FATAL_ERROR << "Error creating link between " << unit1_type << " and " << unit2_type << endl;
		}
	}
	else
	{
		cout << FATAL_ERROR << "Error creating link between " << unit1_type << " and " << unit2_type << endl;
	}
}

// Generate CGRA Mesh topology
void NeuraTile::generate_cgra_mesh_topology()
{
	cout << "Generating CGRA mesh topology \u2713" << endl;
	// NeuraMem Connections
	for (auto i = 0; i < this->mem_x_count; i++)
	{
		for (auto j = 0; j < this->mem_y_count; j++)
		{
			string neuramem_name = "M_" + to_string(i) + "_" + to_string(j);
			if (j % 2 == 0)
			{
				// TOP LEFT Port
				int router_x_cood = i * 2;
				int router_y_cood = j;
				if (router_x_cood >= this->router_x_count)
				{
					router_x_cood = router_x_cood % this->router_x_count;
				}
				if (router_y_cood >= this->router_y_count)
				{
					router_y_cood = router_y_cood % this->router_y_count;
				}
				if (router_x_cood >= this->router_x_count || router_y_cood >= this->router_y_count)
				{
					cout << FATAL_ERROR << "Router coordinates out of bounds" << endl;
				}
				string router_name = "R_" + to_string(router_x_cood) + "_" + to_string(router_y_cood);
				this->connect_units(neuramem_name, MEM_TOP_LEFT_PORT, router_name, RTR_BOTTOM_RIGHT_PORT);
				// TOP RIGHT Port
				router_x_cood = (i * 2) + 1;
				router_y_cood = j;
				if (router_x_cood >= this->router_x_count)
				{
					router_x_cood = router_x_cood % this->router_x_count;
				}
				if (router_y_cood >= this->router_y_count)
				{
					router_y_cood = router_y_cood % this->router_y_count;
				}
				if (router_x_cood >= this->router_x_count || router_y_cood >= this->router_y_count)
				{
					cout << FATAL_ERROR << "Router coordinates out of bounds" << endl;
				}
				router_name = "R_" + to_string(router_x_cood) + "_" + to_string(router_y_cood);
				this->connect_units(neuramem_name, MEM_TOP_RIGHT_PORT, router_name, RTR_BOTTOM_LEFT_PORT);
				// BOTTOM LEFT Port
				router_x_cood = i * 2;
				router_y_cood = j + 1;
				if (router_x_cood >= this->router_x_count)
				{
					router_x_cood = router_x_cood % this->router_x_count;
				}
				if (router_y_cood >= this->router_y_count)
				{
					router_y_cood = router_y_cood % this->router_y_count;
				}
				if (router_x_cood >= this->router_x_count || router_y_cood >= this->router_y_count)
				{
					cout << FATAL_ERROR << "Router coordinates out of bounds" << endl;
				}
				router_name = "R_" + to_string(router_x_cood) + "_" + to_string(router_y_cood);
				this->connect_units(neuramem_name, MEM_BOTTOM_LEFT_PORT, router_name, RTR_TOP_RIGHT_PORT);
				// BOTTOM RIGHT Port
				router_x_cood = (i * 2) + 1;
				router_y_cood = j + 1;
				if (router_x_cood >= this->router_x_count)
				{
					router_x_cood = router_x_cood % this->router_x_count;
				}
				if (router_y_cood >= this->router_y_count)
				{
					router_y_cood = router_y_cood % this->router_y_count;
				}
				if (router_x_cood >= this->router_x_count || router_y_cood >= this->router_y_count)
				{
					cout << FATAL_ERROR << "Router coordinates out of bounds" << endl;
				}
				router_name = "R_" + to_string(router_x_cood) + "_" + to_string(router_y_cood);
				this->connect_units(neuramem_name, MEM_BOTTOM_RIGHT_PORT, router_name, RTR_TOP_LEFT_PORT);
			}
			else
			{
				// TOP LEFT Port
				int router_x_cood = (i * 2) + 1;
				int router_y_cood = j;
				if (router_x_cood >= this->router_x_count)
				{
					router_x_cood = router_x_cood % this->router_x_count;
				}
				if (router_y_cood >= this->router_y_count)
				{
					router_y_cood = router_y_cood % this->router_y_count;
				}
				if (router_x_cood >= this->router_x_count || router_y_cood >= this->router_y_count)
				{
					cout << FATAL_ERROR << "Router coordinates out of bounds" << endl;
				}

				string router_name = "R_" + to_string(router_x_cood) + "_" + to_string(router_y_cood);
				this->connect_units(neuramem_name, MEM_TOP_LEFT_PORT, router_name, RTR_BOTTOM_RIGHT_PORT);
				// TOP RIGHT Port
				router_x_cood = (i * 2) + 2;
				router_y_cood = j;
				if (router_x_cood >= this->router_x_count)
				{
					router_x_cood = router_x_cood % this->router_x_count;
				}
				if (router_y_cood >= this->router_y_count)
				{
					router_y_cood = router_y_cood % this->router_y_count;
				}
				if (router_x_cood >= this->router_x_count || router_y_cood >= this->router_y_count)
				{
					cout << FATAL_ERROR << "Router coordinates out of bounds" << endl;
				}
				router_name = "R_" + to_string(router_x_cood) + "_" + to_string(router_y_cood);
				this->connect_units(neuramem_name, MEM_TOP_RIGHT_PORT, router_name, RTR_BOTTOM_LEFT_PORT);
				// BOTTOM LEFT Port
				router_x_cood = (i * 2) + 1;
				router_y_cood = j + 1;
				if (router_x_cood >= this->router_x_count)
				{
					router_x_cood = router_x_cood % this->router_x_count;
				}
				if (router_y_cood >= this->router_y_count)
				{
					router_y_cood = router_y_cood % this->router_y_count;
				}
				if (router_x_cood >= this->router_x_count || router_y_cood >= this->router_y_count)
				{
					cout << FATAL_ERROR << "Router coordinates out of bounds" << endl;
				}
				router_name = "R_" + to_string(router_x_cood) + "_" + to_string(router_y_cood);
				this->connect_units(neuramem_name, MEM_BOTTOM_LEFT_PORT, router_name, RTR_TOP_RIGHT_PORT);
				// BOTTOM RIGHT Port
				router_x_cood = (i * 2) + 2;
				router_y_cood = j + 1;
				if (router_x_cood >= this->router_x_count)
				{
					router_x_cood = router_x_cood % this->router_x_count;
				}
				if (router_y_cood >= this->router_y_count)
				{
					router_y_cood = router_y_cood % this->router_y_count;
				}
				if (router_x_cood >= this->router_x_count || router_y_cood >= this->router_y_count)
				{
					cout << FATAL_ERROR << "Router coordinates out of bounds" << endl;
				}
				router_name = "R_" + to_string(router_x_cood) + "_" + to_string(router_y_cood);
				this->connect_units(neuramem_name, MEM_BOTTOM_RIGHT_PORT, router_name, RTR_TOP_LEFT_PORT);
			}
		}
	}

	// NeuraCore Connections
	for (auto i = 0; i < this->core_x_count; i++)
	{
		for (auto j = 0; j < this->core_y_count; j++)
		{
			string neuracore_name = "C_" + to_string(i) + "_" + to_string(j);
			if (j % 2 == 0)
			{
				// TOP LEFT Port
				int router_x_cood = (i * 2) + 1;
				int router_y_cood = j;
				if (router_x_cood >= this->router_x_count)
				{
					router_x_cood = router_x_cood % this->router_x_count;
				}
				if (router_y_cood >= this->router_y_count)
				{
					router_y_cood = router_y_cood % this->router_y_count;
				}
				if (router_x_cood >= this->router_x_count || router_y_cood >= this->router_y_count)
				{
					cout << FATAL_ERROR << "Router coordinates out of bounds" << endl;
				}
				string router_name = "R_" + to_string(router_x_cood) + "_" + to_string(router_y_cood);
				this->connect_units(neuracore_name, CORE_TOP_LEFT_PORT, router_name, RTR_BOTTOM_RIGHT_PORT);
				// TOP RIGHT Port
				router_x_cood = (i * 2) + 2;
				router_y_cood = j;
				if (router_x_cood >= this->router_x_count)
				{
					router_x_cood = router_x_cood % this->router_x_count;
				}
				if (router_y_cood >= this->router_y_count)
				{
					router_y_cood = router_y_cood % this->router_y_count;
				}
				if (router_x_cood >= this->router_x_count || router_y_cood >= this->router_y_count)
				{
					cout << FATAL_ERROR << "Router coordinates out of bounds" << endl;
				}
				router_name = "R_" + to_string(router_x_cood) + "_" + to_string(router_y_cood);
				this->connect_units(neuracore_name, CORE_TOP_RIGHT_PORT, router_name, RTR_BOTTOM_LEFT_PORT);
				// BOTTOM LEFT Port
				router_x_cood = (i * 2) + 1;
				router_y_cood = j + 1;
				if (router_x_cood >= this->router_x_count)
				{
					router_x_cood = router_x_cood % this->router_x_count;
				}
				if (router_y_cood >= this->router_y_count)
				{
					router_y_cood = router_y_cood % this->router_y_count;
				}
				if (router_x_cood >= this->router_x_count || router_y_cood >= this->router_y_count)
				{
					cout << FATAL_ERROR << "Router coordinates out of bounds" << endl;
				}
				router_name = "R_" + to_string(router_x_cood) + "_" + to_string(router_y_cood);
				this->connect_units(neuracore_name, CORE_BOTTOM_LEFT_PORT, router_name, RTR_TOP_RIGHT_PORT);
				// BOTTOM RIGHT Port
				router_x_cood = (i * 2) + 2;
				router_y_cood = j + 1;
				if (router_x_cood >= this->router_x_count)
				{
					router_x_cood = router_x_cood % this->router_x_count;
				}
				if (router_y_cood >= this->router_y_count)
				{
					router_y_cood = router_y_cood % this->router_y_count;
				}
				if (router_x_cood >= this->router_x_count || router_y_cood >= this->router_y_count)
				{
					cout << FATAL_ERROR << "Router coordinates out of bounds" << endl;
				}
				router_name = "R_" + to_string(router_x_cood) + "_" + to_string(router_y_cood);
				this->connect_units(neuracore_name, CORE_BOTTOM_RIGHT_PORT, router_name, RTR_TOP_LEFT_PORT);
			}
			else
			{
				// TOP LEFT Port
				int router_x_cood = i * 2;
				int router_y_cood = j;
				if (router_x_cood >= this->router_x_count)
				{
					router_x_cood = router_x_cood % this->router_x_count;
				}
				if (router_y_cood >= this->router_y_count)
				{
					router_y_cood = router_y_cood % this->router_y_count;
				}
				if (router_x_cood >= this->router_x_count || router_y_cood >= this->router_y_count)
				{
					cout << FATAL_ERROR << "Router coordinates out of bounds" << endl;
				}
				string router_name = "R_" + to_string(router_x_cood) + "_" + to_string(router_y_cood);
				this->connect_units(neuracore_name, CORE_TOP_LEFT_PORT, router_name, RTR_BOTTOM_RIGHT_PORT);
				// TOP RIGHT Port
				router_x_cood = (i * 2) + 1;
				router_y_cood = j;
				if (router_x_cood >= this->router_x_count)
				{
					router_x_cood = router_x_cood % this->router_x_count;
				}
				if (router_y_cood >= this->router_y_count)
				{
					router_y_cood = router_y_cood % this->router_y_count;
				}
				if (router_x_cood >= this->router_x_count || router_y_cood >= this->router_y_count)
				{
					cout << FATAL_ERROR << "Router coordinates out of bounds" << endl;
				}
				router_name = "R_" + to_string(router_x_cood) + "_" + to_string(router_y_cood);
				this->connect_units(neuracore_name, CORE_TOP_RIGHT_PORT, router_name, RTR_BOTTOM_LEFT_PORT);
				// BOTTOM LEFT Port
				router_x_cood = i * 2;
				router_y_cood = j + 1;
				if (router_x_cood >= this->router_x_count)
				{
					router_x_cood = router_x_cood % this->router_x_count;
				}
				if (router_y_cood >= this->router_y_count)
				{
					router_y_cood = router_y_cood % this->router_y_count;
				}
				if (router_x_cood >= this->router_x_count || router_y_cood >= this->router_y_count)
				{
					cout << FATAL_ERROR << "Router coordinates out of bounds" << endl;
				}
				router_name = "R_" + to_string(router_x_cood) + "_" + to_string(router_y_cood);
				this->connect_units(neuracore_name, CORE_BOTTOM_LEFT_PORT, router_name, RTR_TOP_RIGHT_PORT);
				// BOTTOM RIGHT Port
				router_x_cood = (i * 2) + 1;
				router_y_cood = j + 1;
				if (router_x_cood >= this->router_x_count)
				{
					router_x_cood = router_x_cood % this->router_x_count;
				}
				if (router_y_cood >= this->router_y_count)
				{
					router_y_cood = router_y_cood % this->router_y_count;
				}
				if (router_x_cood >= this->router_x_count || router_y_cood >= this->router_y_count)
				{
					cout << FATAL_ERROR << "Router coordinates out of bounds" << endl;
				}
				router_name = "R_" + to_string(router_x_cood) + "_" + to_string(router_y_cood);
				this->connect_units(neuracore_name, CORE_BOTTOM_RIGHT_PORT, router_name, RTR_TOP_LEFT_PORT);
			}
		}
	}

	// NeuraDispatcher Connections
	int connect_dispatcher_port = 0;
	for (auto y = 0; y < this->core_y_count; y++)
	{
		for (auto x = 0; x < this->core_x_count; x++)
		{
			string neuradispatcher_name = "D_0"; // + to_string(i) + "_" + to_string(j);
			string core_name = "C_" + to_string(x) + "_" + to_string(y);
			this->connect_units(neuradispatcher_name, connect_dispatcher_port, core_name, CORE_INSTX_PORT);
			this->dispatcher.insert_uid_to_port_map_entry(this->core_array[x][y].get_uid(), connect_dispatcher_port);
			connect_dispatcher_port++;
		}
	}

	// NeuraRouter Connections
	for (auto i = 0; i < this->router_x_count; i++)
	{
		for (auto j = 0; j < this->router_y_count; j++)
		{
			string router_name = "R_" + to_string(i) + "_" + to_string(j);
			// Each router has to connect only to TOP and RIGHT routers
			// TOP Port
			int router_x_cood = i;
			int router_y_cood = (j + this->router_y_count - 1) % this->router_y_count;
			string top_router_name = "R_" + to_string(router_x_cood) + "_" + to_string(router_y_cood);
			this->connect_units(router_name, RTR_TOP_PORT, top_router_name, RTR_BOTTOM_PORT);
			// RIGHT Port
			router_x_cood = (i + 1) % this->router_x_count;
			router_y_cood = j;
			string right_router_name = "R_" + to_string(router_x_cood) + "_" + to_string(router_y_cood);
			this->connect_units(router_name, RTR_RIGHT_PORT, right_router_name, RTR_LEFT_PORT);
		}
	}

	// NeuraWriters Connections
	// The supported tile sizes use the explicit writer/router mapping below.
	if (this->gcp->NEURA_TILE_SIZE == 2)
	{
		string writer_name = "W_0";
		string router_name = "R_2_1";
		this->connect_units(writer_name, WRITER_RTR_PORT, router_name, RTR_EXTRA_PORT);
		writer_name = "W_0";
		string mc_name = "X_0";
		this->connect_units(writer_name, WRITER_MC_PORT, mc_name, MC_WRITER_0_PORT);
		writer_name = "W_1";
		router_name = "R_0_3";
		this->connect_units(writer_name, WRITER_RTR_PORT, router_name, RTR_EXTRA_PORT);
		writer_name = "W_1";
		mc_name = "X_0";
		this->connect_units(writer_name, WRITER_MC_PORT, mc_name, MC_WRITER_1_PORT);
	}
	else
	{
		string writer_name = "W_0";
		string router_name = "R_6_1";
		this->connect_units(writer_name, WRITER_RTR_PORT, router_name, RTR_EXTRA_PORT);
		writer_name = "W_0";
		string mc_name = "X_0";
		this->connect_units(writer_name, WRITER_MC_PORT, mc_name, MC_WRITER_0_PORT);
		writer_name = "W_1";
		router_name = "R_2_3";
		this->connect_units(writer_name, WRITER_RTR_PORT, router_name, RTR_EXTRA_PORT);
		writer_name = "W_1";
		mc_name = "X_0";
		this->connect_units(writer_name, WRITER_MC_PORT, mc_name, MC_WRITER_1_PORT);
	}

	// Memory Controller Connections
	// The supported tile sizes use the explicit controller/router mapping below.
	if (this->gcp->NEURA_TILE_SIZE == 2)
	{
		string mc_name = "X_0";
		string router_name = "R_2_0";
		this->connect_units(mc_name, MC_RTR_0_PORT, router_name, RTR_EXTRA_PORT);
		mc_name = "X_0";
		router_name = "R_0_2";
		this->connect_units(mc_name, MC_RTR_1_PORT, router_name, RTR_EXTRA_PORT);
		mc_name = "X_0";
		router_name = "R_1_1";
		this->connect_units(mc_name, MC_RTR_2_PORT, router_name, RTR_EXTRA_PORT);
		mc_name = "X_0";
		router_name = "R_0_3";
		this->connect_units(mc_name, MC_RTR_3_PORT, router_name, RTR_EXTRA_PORT);
		mc_name = "X_0";
		router_name = "R_3_0";
		this->connect_units(mc_name, MC_RTR_4_PORT, router_name, RTR_EXTRA_PORT);
		mc_name = "X_0";
		router_name = "R_1_3";
		this->connect_units(mc_name, MC_RTR_5_PORT, router_name, RTR_EXTRA_PORT);
		mc_name = "X_0";
		router_name = "R_3_3";
		this->connect_units(mc_name, MC_RTR_6_PORT, router_name, RTR_EXTRA_PORT);
		mc_name = "X_0";
		router_name = "R_2_2";
		this->connect_units(mc_name, MC_RTR_7_PORT, router_name, RTR_EXTRA_PORT);
	}
	else
	{
		string mc_name = "X_0";
		string router_name = "R_0_2";
		this->connect_units(mc_name, MC_RTR_0_PORT, router_name, RTR_EXTRA_PORT);
		mc_name = "X_0";
		router_name = "R_4_0";
		this->connect_units(mc_name, MC_RTR_1_PORT, router_name, RTR_EXTRA_PORT);
		mc_name = "X_0";
		router_name = "R_1_1";
		this->connect_units(mc_name, MC_RTR_2_PORT, router_name, RTR_EXTRA_PORT);
		mc_name = "X_0";
		router_name = "R_3_2";
		this->connect_units(mc_name, MC_RTR_3_PORT, router_name, RTR_EXTRA_PORT);
		mc_name = "X_0";
		router_name = "R_7_0";
		this->connect_units(mc_name, MC_RTR_4_PORT, router_name, RTR_EXTRA_PORT);
		mc_name = "X_0";
		router_name = "R_5_3";
		this->connect_units(mc_name, MC_RTR_5_PORT, router_name, RTR_EXTRA_PORT);
		mc_name = "X_0";
		router_name = "R_7_3";
		this->connect_units(mc_name, MC_RTR_6_PORT, router_name, RTR_EXTRA_PORT);
		mc_name = "X_0";
		router_name = "R_0_3";
		this->connect_units(mc_name, MC_RTR_7_PORT, router_name, RTR_EXTRA_PORT);
	}
}

// Destructor
NeuraTile::~NeuraTile()
{
	// cout << "NeuraTile::~NeuraTile()" << endl;
	// DyGNN controller (PE Array mode)
#ifdef ENABLE_PE_ARRAY
#ifdef USE_PE_ARRAY_MODE
	if (this->dygnn_controller) {
		delete this->dygnn_controller;
		this->dygnn_controller = nullptr;
	}
	auto drain_endpoint_queue = [](queue<InstX*>* q_ptr) {
		if (q_ptr == nullptr) {
			return;
		}
		while (!q_ptr->empty()) {
			InstX* inst = q_ptr->front();
			q_ptr->pop();
			if (inst && inst->type == INSTX_TYPE_CIM_PACKET) {
				delete static_cast<StreamingNoCPacket*>(inst);
			}
		}
	};
	if (this->streaming_pe_endpoint) {
		drain_endpoint_queue(this->streaming_pe_endpoint->get_in_port_ref(kStreamingEndpointPort));
		drain_endpoint_queue(this->streaming_pe_endpoint->get_out_port_ref(kStreamingEndpointPort));
		delete this->streaming_pe_endpoint;
		this->streaming_pe_endpoint = nullptr;
	}
	if (this->streaming_cim_endpoint) {
		drain_endpoint_queue(this->streaming_cim_endpoint->get_in_port_ref(kStreamingEndpointPort));
		drain_endpoint_queue(this->streaming_cim_endpoint->get_out_port_ref(kStreamingEndpointPort));
		delete this->streaming_cim_endpoint;
		this->streaming_cim_endpoint = nullptr;
	}
#endif
#endif
#ifdef ENABLE_CIM
	if (this->cim_unit != nullptr) {
		delete this->cim_unit;
		this->cim_unit = nullptr;
	}
#endif
	// Delete all cores
	// cout << "Deleting Cores" << endl;
	for (auto i = 0; i < this->core_x_count; i++)
	{
		delete[] this->core_array[i];
	}
	delete[] this->core_array;

	// Delete all memories
	// cout << "Deleting Memories" << endl;
	for (auto i = 0; i < this->mem_x_count; i++)
	{
		delete[] this->mem_array[i];
	}
	delete[] this->mem_array;

	// Delete all routers
	// cout << "Deleting Routers" << endl;
	for (auto i = 0; i < this->router_x_count; i++)
	{
		delete[] this->router_array[i];
	}
	delete[] this->router_array;

	// Delete all connections
	// cout << "Deleting Connections" << endl;
	for (size_t i = 0; i < this->conn_array.size(); i++)
	{
		delete this->conn_array[i];
	}

	// Delete all ports
	// cout << "Deleting Ports" << endl;
	delete[] this->in_port_array;
	delete[] this->out_port_array;

	// Its not dynamic memory allocation
	// You don't need to delete it
	// Delete dispatcher array
	// for (auto i=0; i<this->core_x_count; i++) {
	// 	delete[] this->dispatcher_array[i];
	// }
	// delete[] this->dispatcher_array;

	// Delete memory controller array
	// cout << "Deleting Memory Controllers" << endl;
	delete[] this->mem_controller_array;

	// Delete all writers
	// for (auto i=0; i<this->writer_count; i++) {
	// 	delete[] this->writer_array[i];
	// }
	// cout << "Deleting Writers" << endl;
	delete[] this->writer_array;

	// Delete GlobalCounters
	// cout << "Deleting Global Counters" << endl;
	delete this->gcp;

#ifdef ENABLE_PE_ARRAY
	if (this->pe_array_spmm != nullptr) {
		delete this->pe_array_spmm;
	}
#endif
}

int NeuraTile::get_id()
{
	return this->id;
}

// int NeuraTile::get_uid()
// {
// 	return this->uid;
// }

// Print NeuraTile Information
void NeuraTile::print()
{
	cout << "NeuraTile ID: " << this->id << endl;
	cout << "NeuraTile X Coordinate: " << this->x_cood << endl;
	cout << "NeuraTile Y Coordinate: " << this->y_cood << endl;
}

// Tick
// @return: True if simulation is over
bool NeuraTile::tick()
{
#ifdef TRACE_INSTX
	cout << "******** Cycle: " << CYAN_START << this->gcp->get_cycle() << COLOR_RESET << "********" << endl;
#endif

// =============================================================================
// PE ARRAY MODE: Use our PE Array SpMM instead of original NeuraCore (Hash)
// =============================================================================
#ifdef ENABLE_PE_ARRAY
#ifdef USE_PE_ARRAY_MODE
	return this->tick_pe_array_mode();
#endif
#endif
// =============================================================================

	bool sim_end = true;
	bool interval_tick;
	if (this->gcp->get_cycle() % this->gcp->INTERVAL == 0)
	{
		interval_tick = true;
	}
	else
	{
		interval_tick = false;
	}
	// Storing tick output before because "C++ short-circuits logical AND"
	bool tick_output;
	// bool unit_sim_end = true;
	// Tick all connections
	// It is important to tick all the connections before the components because
	// the components create data and the connections send data.
	// We dont send data in the same cycle that we create it.

	for (auto conn_tick = 0; conn_tick < MULTI_TICK_CONN; conn_tick++)
	{
		for (size_t i = 0; i < this->conn_array.size(); i++)
		{
			// cout << "Ticking Connection [" << i << "]" << endl;
			tick_output = this->conn_array[i]->tick();
			sim_end = sim_end && tick_output;
		}
	}

// #ifdef DUAL_TX_CONN
// 	for (size_t i = 0; i < this->conn_array.size(); i++)
// 	{
// 		// cout << "Ticking Connection [" << i << "]" << endl;
// 		tick_output = this->conn_array[i]->tick();
// 		sim_end = sim_end && tick_output;
// 	}
// #endif // DUAL_TX_CONN

	// #pragma omp parallel for
	for (int component_id = 0; component_id < 6; component_id++)
	{

		if (component_id == DISPATCHER_COMPONENT_INDEX)
		{
			// Tick dispatcher
			bool dispatcher_tick_output = this->dispatcher.tick();
			// sim_end = sim_end && tick_output;

			// cout << "Dispatcher Tick: ";
			// sim_end_printer(unit_sim_end);
			// cout << endl;

			// for (auto i=0; i<this->core_x_count; i++) {
			// 	for (auto j=0; j<this->core_y_count; j++) {
			// 		// cout << "Ticking Dispatcher [" << i << "][" << j << "]" << endl;
			// 		tick_output = this->dispatcher_array[i][j].tick();
			// 		sim_end = sim_end && tick_output;
			// 	}
			// }
			// tick_output = this->dispatcher->tick();

			this->component_sim_end_array[DISPATCHER_COMPONENT_INDEX] = dispatcher_tick_output;
		}
		else if (component_id == CORE_COMPONENT_INDEX)
		{
			// Tick all cores
			bool core_sim_end = true;
			// #pragma omp parallel for collapse(2) reduction(&& : core_sim_end)
			for (auto c_tick = 0; c_tick < MULTI_TICK_NEURACORE; c_tick++)
			{
			for (auto i = 0; i < this->core_x_count; i++)
			{
				for (auto j = 0; j < this->core_y_count; j++)
				{
					// cout << "Ticking Core [" << i << "][" << j << "]" << endl;
					bool core_tick_output = this->core_array[i][j].tick();
					core_sim_end = core_sim_end && core_tick_output;
				}
			}
			}
			this->component_sim_end_array[CORE_COMPONENT_INDEX] = core_sim_end;
		}
		else if (component_id == MEM_COMPONENT_INDEX)
		{

			// cout << "Core Tick: ";
			// sim_end_printer(unit_sim_end);
			// cout << endl;

			// Tick all memories
			bool mem_sim_end = true;
			// #pragma omp parallel for collapse(2) reduction(&& : mem_sim_end)
			for (auto m_tick = 0; m_tick < MULTI_TICK_NEURAMEM; m_tick++)
			{
				for (auto i = 0; i < this->mem_x_count; i++)
				{
					for (auto j = 0; j < this->mem_y_count; j++)
					{
						// cout << "Ticking Memory [" << i << "][" << j << "]" << endl;
						bool mem_tick_output = this->mem_array[i][j].tick();
						mem_sim_end = mem_sim_end && mem_tick_output;
					}
				}
			}
			this->component_sim_end_array[MEM_COMPONENT_INDEX] = mem_sim_end;
		}
		else if (component_id == RTR_COMPONENT_INDEX)
		{

			// cout << "Mem Tick: ";
			// sim_end_printer(unit_sim_end);
			// cout << endl;

			// Tick all routers
			bool router_sim_end = true;
			// #pragma omp parallel for collapse(2)
			for (auto r_tick = 0; r_tick < MULTI_TICK_ROUTER; r_tick++)
			{
				for (auto i = 0; i < this->router_x_count; i++)
				{
					for (auto j = 0; j < this->router_y_count; j++)
					{
						// cout << "Ticking Router [" << i << "][" << j << "]" << endl;
						bool router_tick_output = this->router_array[i][j].tick();
						router_sim_end = router_sim_end && router_tick_output;
					}
				}
			}
			this->component_sim_end_array[RTR_COMPONENT_INDEX] = router_sim_end;

			// cout << "Router Tick: ";
			// sim_end_printer(unit_sim_end);
			// cout << endl;
		}
		else if (component_id == WRITER_COMPONENT_INDEX)
		{
			// Tick all Writers
			bool writer_sim_end = true;
			// #NOpragma omp parallel for
			for (auto w_tick = 0; w_tick < MULTI_TICK_WRITER; w_tick++)
			{
				for (auto i = 0; i < this->writer_count; i++)
				{
					// cout << "Ticking Writer [" << i << "][" << j << "]" << endl;
					bool writer_tick_output = this->writer_array[i].tick();
					writer_sim_end = writer_sim_end && writer_tick_output;
				}
			}
			this->component_sim_end_array[WRITER_COMPONENT_INDEX] = writer_sim_end;
		}
		else if (component_id == MC_COMPONENT_INDEX)
		{

			// cout << "Writer Tick: ";
			// sim_end_printer(unit_sim_end);
			// cout << endl;

			// Tick Memory Controller
			bool mc_sim_end = true;
			for (auto mc_tick = 0; mc_tick < MULTI_TICK_MC; mc_tick++)
			{
				this->mem_controller_array[this->id].tick();
			}
			

			this->component_sim_end_array[MC_COMPONENT_INDEX] = mc_sim_end;
		}
	}

	// Reduce all component sim end flags and store in sim_end
	for (int component_id = 0; component_id < 6; component_id++)
	{
		sim_end = sim_end && this->component_sim_end_array[component_id];
	}

	// =========================================================================
	// CiM Integration: Dense MatMul (Combination) after SpMM (Aggregation)
	// =========================================================================
#ifdef ENABLE_CIM
	if (sim_end && !this->spmm_done)
	{
		// SpMM phase just completed, transition to CiM phase
		this->spmm_done = true;
		this->cim_start_cycle = this->gcp->get_cycle();
		cout << "[CiM] SpMM completed at cycle " << this->cim_start_cycle << endl;
		cout << "[CiM] Starting Dense MatMul (Combination) phase..." << endl;
		this->start_cim_phase();
	}

	if (this->spmm_done && this->cim_started && !this->is_cim_done())
	{
		// Continue CiM phase
		this->tick_cim();
		sim_end = false;  // Keep simulation running until CiM completes
	}
	else if (this->spmm_done && this->cim_started && this->is_cim_done())
	{
		// CiM phase completed
		static bool cim_done_printed = false;
		if (!cim_done_printed)
		{
			long long cim_cycles = this->gcp->get_cycle() - this->cim_start_cycle;
			cout << "[CiM] Dense MatMul completed. CiM cycles: " << cim_cycles << endl;
			cout << "[CiM] Total energy: " << this->cim_unit->get_total_energy_pj() << " pJ" << endl;
			cim_done_printed = true;
		}
		sim_end = true;
	}
#endif
	// =========================================================================

	// cout << "Mem Controller Tick: ";
	// sim_end_printer(unit_sim_end);
	// cout << endl;

	// Tick Global Counters
	this->gcp->tick();

	if (interval_tick)
	{
		this->interval_metrics();
		this->gcp->print_sim_speed();
	}

	// Barrier to limit the number of cycles
	if (this->gcp->get_cycle() > this->gcp->HARD_STOP)
	{
		// Print Simulation End Flags
		this->print_sim_end();
		return (this->return_routine(SIM_END_TRUE, COMPUTED));
	}

	return (this->return_routine(sim_end, COMPUTED));
}

void NeuraTile::interval_metrics()
{
	// Tick all interval metrics
	// Tick all connections
	for (size_t i = 0; i < this->conn_array.size(); i++)
	{
		this->conn_array[i]->interval_metrics();
	}
	// Tick dispatcher
	this->dispatcher.interval_metrics();
	// for (auto i=0; i<this->core_x_count; i++) {
	// 	for (auto j=0; j<this->core_y_count; j++) {
	// 		this->dispatcher_array[i][j].interval_metrics();
	// 	}
	// }
	// this->dispatcher->interval_metrics();

	// Tick all cores
	for (auto i = 0; i < this->core_x_count; i++)
	{
		for (auto j = 0; j < this->core_y_count; j++)
		{
			this->core_array[i][j].interval_metrics();
		}
	}

	// Tick all memories
	for (auto i = 0; i < this->mem_x_count; i++)
	{
		for (auto j = 0; j < this->mem_y_count; j++)
		{
			this->mem_array[i][j].interval_metrics();
		}
	}

	// Tick all routers
	for (auto i = 0; i < this->router_x_count; i++)
	{
		for (auto j = 0; j < this->router_y_count; j++)
		{
			this->router_array[i][j].interval_metrics();
		}
	}

	// Tick all Writers
	for (auto i = 0; i < this->writer_count; i++)
	{
		this->writer_array[i].interval_metrics();
	}

	// Tick Memory Controller
	this->mem_controller_array[this->id].interval_metrics();

	// Tick Global Counters
	this->gcp->interval_metrics();

	// Tick self
	this->self_interval_metrics();
}

// Statistics
void NeuraTile::statistics()
{
	cout << "************* NeuraTile ID: " << this->id << " *************" << endl;
	// Print Statistics for dispatcher
	this->dispatcher.statistics();
	// for (auto i=0; i<this->core_x_count; i++) {
	// 	for (auto j=0; j<this->core_y_count; j++) {
	// 		this->dispatcher_array[i][j].statistics();
	// 	}
	// }
	// this->dispatcher->statistics();
	// Print Statistics for all cores
	for (auto i = 0; i < this->core_x_count; i++)
	{
		for (auto j = 0; j < this->core_y_count; j++)
		{
			this->core_array[i][j].statistics();
		}
	}
	// Print Statistics for all memories
	for (auto i = 0; i < this->mem_x_count; i++)
	{
		for (auto j = 0; j < this->mem_y_count; j++)
		{
			this->mem_array[i][j].statistics();
		}
	}
	// Print Statistics for all routers
	for (auto i = 0; i < this->router_x_count; i++)
	{
		for (auto j = 0; j < this->router_y_count; j++)
		{
			// cout << "Neura Tile Router [" << i << "][" << j << "]" << endl;
			this->router_array[i][j].statistics();
		}
	}
	// Print Statistics for Memory Controller
	this->mem_controller_array[this->id].statistics();

	// Print Statistics for all Writers
	for (auto i = 0; i < this->writer_count; i++)
	{
		this->writer_array[i].statistics();
	}

	// Print Statistics for all connections
	for (size_t i = 0; i < this->conn_array.size(); i++)
	{
		this->conn_array[i]->statistics();
	}
	// Print final statistics
	this->gcp->statistics();
}

// Statistics
void NeuraTile::smart_stats()
{
	// Print Statistics for dispatcher
	this->dispatcher.smart_stats();
	// for (auto i=0; i<this->core_x_count; i++) {
	// 	for (auto j=0; j<this->core_y_count; j++) {
	// 		this->dispatcher_array[i][j].statistics();
	// 	}
	// }
	// this->dispatcher->statistics();
	// Print Statistics for all cores
	for (auto i = 0; i < this->core_x_count; i++)
	{
		for (auto j = 0; j < this->core_y_count; j++)
		{
			this->core_array[i][j].smart_stats();
		}
	}
	// Print Statistics for all memories
	for (auto i = 0; i < this->mem_x_count; i++)
	{
		for (auto j = 0; j < this->mem_y_count; j++)
		{
			// this->mem_array[i][j].statistics();
		}
	}
	// Print Statistics for all routers
	for (auto i = 0; i < this->router_x_count; i++)
	{
		for (auto j = 0; j < this->router_y_count; j++)
		{
			// cout << "Neura Tile Router [" << i << "][" << j << "]" << endl;
			this->router_array[i][j].smart_stats();
		}
	}
	// Print Statistics for Memory Controller
	this->mem_controller_array[this->id].smart_stats();

	// Print Statistics for all connections
	for (size_t i = 0; i < this->conn_array.size(); i++)
	{
		this->conn_array[i]->smart_stats();
	}
	// Print final statistics
	this->gcp->smart_stats();
}

void NeuraTile::print_sim_end()
{
	cout << "********** Sim End Flags **********" << endl;
	cout << "Dispatcher: ";
	sim_end_printer(this->dispatcher.get_sim_end());
	cout << endl;
	// for (auto i = 0; i < this->core_x_count; i++)
	// {
	// 	for (auto j = 0; j < this->core_y_count; j++)
	// 	{
	// 		cout << "Dispatcher [" << i << "][" << j << "]: ";
	// 		sim_end_printer(this->dispatcher_array[i][j].get_sim_end());
	// 		cout << endl;
	// 	}
	// }
	// cout << "Dispatcher: ";
	// sim_end_printer(this->dispatcher->get_sim_end());
	// cout << endl;
	for (auto i = 0; i < this->core_x_count; i++)
	{
		for (auto j = 0; j < this->core_y_count; j++)
		{
			cout << "Core [" << i << "][" << j << "]: ";
			sim_end_printer(this->core_array[i][j].get_sim_end());
			cout << endl;
		}
	}
	// Print Statistics for all memories
	for (auto i = 0; i < this->mem_x_count; i++)
	{
		for (auto j = 0; j < this->mem_y_count; j++)
		{
			// this->mem_array[i][j].statistics();
			cout << "Memory [" << i << "][" << j << "]: ";
			sim_end_printer(this->mem_array[i][j].get_sim_end());
			cout << endl;
		}
	}
	// Print Statistics for all routers
	for (auto i = 0; i < this->router_x_count; i++)
	{
		for (auto j = 0; j < this->router_y_count; j++)
		{
			cout << "Router [" << i << "][" << j << "]: ";
			sim_end_printer(this->router_array[i][j].get_sim_end());
			cout << endl;
		}
	}
	// Print Statistics for Memory Controller
	cout << "Memory Controller: ";
	sim_end_printer(this->mem_controller_array[this->id].get_sim_end());
	cout << endl;

	// Print Statistics for all connections
	for (size_t i = 0; i < this->conn_array.size(); i++)
	{
		cout << "Connection [" << i << "]: ";
		sim_end_printer(this->conn_array[i]->get_sim_end());
		cout << endl;
	}

	cout << "********** End of Sim End Flags **********" << endl;

	this->mem_controller_array[this->id].print_json();
}

// Post Sim Metrics
void NeuraTile::post_sim_metrics()
{
	// Post Sim Metrics for all cores
	// cout << "Post Sim Metrics for all cores" << endl;
	for (auto i = 0; i < this->core_x_count; i++)
	{
		for (auto j = 0; j < this->core_y_count; j++)
		{
			this->core_array[i][j].post_sim_metrics();
		}
	}
	// Post Sim Metrics for all memories
	// cout << "Post Sim Metrics for all memories" << endl;
	for (auto i = 0; i < this->mem_x_count; i++)
	{
		for (auto j = 0; j < this->mem_y_count; j++)
		{
			this->mem_array[i][j].post_sim_metrics();
		}
	}
	// Post Sim Metrics for all connections
	// cout << "Post Sim Metrics for all connections" << endl;
	for (size_t i = 0; i < this->conn_array.size(); i++)
	{
		this->conn_array[i]->post_sim_metrics();
	}
	// Post Sim Metrics for Global Counters
	this->gcp->post_sim_metrics();

	// Post Sim Metrics for all routers
	// cout << "Post Sim Metrics for all routers" << endl;
	for (auto i = 0; i < this->router_x_count; i++)
	{
		for (auto j = 0; j < this->router_y_count; j++)
		{
			this->router_array[i][j].post_sim_metrics();
		}
	}
	// Post Sim Metrics for Memory Controller
	// cout << "Post Sim Metrics for Memory Controller" << endl;
	this->mem_controller_array[this->id].post_sim_metrics();

	// Post Sim Metrics for Dispatcher
	// cout << "Post Sim Metrics for Dispatcher" << endl;
	this->dispatcher.post_sim_metrics();
	// for (auto i = 0; i < this->core_x_count; i++)
	// {
	// 	for (auto j = 0; j < this->core_y_count; j++)
	// 	{
	// 		this->dispatcher_array[i][j].post_sim_metrics();
	// 	}
	// }
	// this->dispatcher->post_sim_metrics();

	// Post Sim Metrics for all Writers
	// cout << "Post Sim Metrics for all Writers" << endl;
	for (auto i = 0; i < this->writer_count; i++)
	{
		this->writer_array[i].post_sim_metrics();
	}

	// Post Sim Metrics for NeuraTile
	// Self Post Sim Metrics
	// cout << "Post Sim Metrics for NeuraTile" << endl;
	this->self_post_sim_metrics();
}

// Sanity Check
bool NeuraTile::sanity_check()
{
	bool sanity = true;
	// Sanity Check for all cores
	for (auto i = 0; i < this->core_x_count; i++)
	{
		for (auto j = 0; j < this->core_y_count; j++)
		{
			sanity = sanity && this->core_array[i][j].sanity_check();
			if (!sanity)
			{
				cout << FATAL_ERROR << "Core [" << i << "][" << j << "] Sanity Check Failed" << endl;
			}
		}
	}
	// Sanity Check for all memories
	for (auto i = 0; i < this->mem_x_count; i++)
	{
		for (auto j = 0; j < this->mem_y_count; j++)
		{
			sanity = sanity && this->mem_array[i][j].sanity_check();
			if (!sanity)
			{
				cout << FATAL_ERROR << "Memory [" << i << "][" << j << "] Sanity Check Failed" << endl;
			}
		}
	}
	// Sanity Check for Global Counters
	sanity = sanity && this->gcp->sanity_check();
	if (!sanity)
	{
		cout << FATAL_ERROR << "Global Counters Sanity Check Failed" << endl;
	}

	// Sanity Check for all routers
	for (auto i = 0; i < this->router_x_count; i++)
	{
		for (auto j = 0; j < this->router_y_count; j++)
		{
			sanity = sanity && this->router_array[i][j].sanity_check();
			if (!sanity)
			{
				cout << FATAL_ERROR << "Router [" << i << "][" << j << "] Sanity Check Failed" << endl;
			}
		}
	}

	// Sanity Check for Memory Controller
	sanity = sanity && this->mem_controller_array[this->id].sanity_check();
	if (!sanity)
	{
		cout << FATAL_ERROR << "Memory Controller [" << this->id << "] Sanity Check Failed" << endl;
	}

	// Sanity Check for Dispatcher
	sanity = sanity && this->dispatcher.sanity_check();
	if (!sanity)
	{
		cout << FATAL_ERROR << "Dispatcher Sanity Check Failed" << endl;
	}
	// for (auto i = 0; i < this->core_x_count; i++)
	// {
	// 	for (auto j = 0; j < this->core_y_count; j++)
	// 	{
	// 		sanity = sanity && this->dispatcher_array[i][j].sanity_check();
	// 	}
	// }
	// sanity = sanity && this->dispatcher->sanity_check();

	// Sanity Check for all Writers
	for (auto i = 0; i < this->writer_count; i++)
	{
		sanity = sanity && this->writer_array[i].sanity_check();
		if (!sanity)
		{
			cout << FATAL_ERROR << "Writer [" << i << "] Sanity Check Failed" << endl;
		}
	}

	// Sanity Check for all connections
	for (size_t i = 0; i < this->conn_array.size(); i++)
	{
		sanity = sanity && this->conn_array[i]->sanity_check();
		if (!sanity)
		{
			cout << FATAL_ERROR << "Connection [" << i << "] connecting " << this->conn_array[i]->connected_vertices_names << " Sanity Check Failed" << endl;
		}
	}

	// Print Sanity Check Result
	if (sanity)
	{
		cout << "Sanity Check Passed" << endl;
	}
	else
	{
		cout << FATAL_ERROR << "Sanity Check Failed" << endl;
		exit(1);
	}
	return sanity;
}

queue<InstX *> *NeuraTile::get_in_port_ref(int port_id)
{
	return &this->in_port_array[port_id];
}

queue<InstX *> *NeuraTile::get_out_port_ref(int port_id)
{
	return &this->out_port_array[port_id];
}

bool NeuraTile::get_sim_end()
{
	return this->sim_end;
}

void NeuraTile::enable_max_buffer_sizes_computation()
{
	for (auto it : this->conn_array)
	{
		it->measure_max_buffer_sizes();
	}
}

void NeuraTile::set_sim_start_time()
{
	this->gcp->set_sim_start_time();
}

bool NeuraTile::return_routine(bool sim_end, bool computed)
{
	this->sim_end = sim_end;
	if (computed)
	{
		this->busy_cycles++;
		this->interval_busy_cycles++;
	}
	else
	{
		this->idle_cycles++;
		this->interval_idle_cycles++;
	}
	return this->sim_end;
}

void NeuraTile::self_post_sim_metrics()
{
	if (MONGO_FLAG)
	{
		this->header_collection.insert_one(make_result_record(
			result_field("name", "post_sim_metrics"),
			result_field("busy_cycles", (long)this->busy_cycles),
			result_field("idle_cycles", (long)this->idle_cycles)));
	}
}

void NeuraTile::self_interval_metrics()
{
	if (MONGO_FLAG)
	{
		this->collection.insert_one(make_result_record(
			result_field("name", "interval_metrics"),
			result_field("interval_id", (long)this->interval_id),
			result_field("interval_busy_cycles", (long)this->interval_busy_cycles),
			result_field("interval_idle_cycles", (long)this->interval_idle_cycles)));
	}
	this->interval_busy_cycles = 0;
	this->interval_idle_cycles = 0;
	this->interval_id++;
}

GlobalCounters *NeuraTile::get_gcp()
{
	return this->gcp;
}

string NeuraTile::get_vertex_name()
{
	return this->vertex_name;
}

// =============================================================================
// CiM Integration Methods
// =============================================================================

#ifdef ENABLE_CIM

void NeuraTile::init_cim(int d_in, int d_out)
{
	this->cim_d_in = d_in;
	this->cim_d_out = d_out;
	
	// From config.py SRAMCiMConfig:
	// - num_tiles: 16 (reduced from 64 for realistic area)
	// - bits_per_cycle: configurable via CLI (default: 2)
	int num_cim_tiles = 16;
	if (const char* configured = std::getenv("DESPOINA_R5_PHYSICAL_TILES")) {
		char* end = nullptr;
		const long parsed = std::strtol(configured, &end, 10);
		if (end == configured || *end != '\0' ||
		    (parsed != 16 && parsed != 32 && parsed != 128)) {
			throw std::runtime_error(
			    "DESPOINA_R5_PHYSICAL_TILES must be exactly 16, 32, or 128");
		}
		num_cim_tiles = static_cast<int>(parsed);
	}
	int bits_per_cycle = 2;
	int fifo_depth = 16;
	if (this->gcp != nullptr && this->gcp->arg_parse != nullptr) {
		bits_per_cycle = std::max(1, std::min(8, this->gcp->arg_parse->get_bits_per_cycle()));
		fifo_depth = std::max(1, this->gcp->arg_parse->get_cim_fifo_depth());
	}
	
	this->cim_unit = new NeuraCiM(d_in, d_out, num_cim_tiles, bits_per_cycle, fifo_depth);
	this->cim_enabled = true;

#ifdef ENABLE_DRAMSIM3_PE
	if (this->gcp != nullptr) {
		const std::string weight_out_dir =
		    this->gcp->get_results_dir() + "/dramsim3_cim_weight_output";
		this->cim_unit->configure_weight_dram_output_dir(weight_out_dir);
		this->cim_unit->configure_weight_dram_config_file(this->gcp->get_dram_config_path());
	}
#endif

	// R5: no placeholder/preload. The controller emits one explicit resident-W
	// load immediately before the first operation of each layer key.
	this->cim_weight_load_cycles = static_cast<long long>(this->cim_unit->get_weight_load_cycles());
	this->cim_weight_load_dram_requests = static_cast<long long>(this->cim_unit->get_weight_load_dram_requests());
	this->cim_weight_load_bytes = static_cast<long long>(this->cim_unit->get_weight_load_bytes());
	cout << "[CiM] Weight load: "
	     << this->cim_weight_load_dram_requests << " DRAM txns, "
	     << this->cim_weight_load_cycles << " cycles, "
	     << this->cim_weight_load_bytes << " bytes" << endl;
}

void NeuraTile::start_cim_phase()
{
	if (!this->cim_enabled)
	{
		// Initialize CiM if not already done
		// Use matrix B's y_dim as d_in, and default 128 as d_out
		init_cim(this->cim_d_in, this->cim_d_out);
	}
	
	// Get number of output rows from SpMM result
	// For SpMM: C = A ??B, where A is [n ??n] and B is [n ??d]
	// Result C is [n ??d], so we process n rows through CiM
#ifdef USE_PE_ARRAY_MODE
	int n_rows = this->pe_array_n;  // Use actual node count from PE Array
	
	// Use analytical model for accurate cycle calculation
	// From config.py: 64 tiles parallel, 2-bit parallelism
	// Throughput: 64 rows/cycle (pipelined)
	// Total = ceil(n/64) + pipeline_latency
	uint64_t cim_cycles = this->cim_unit->run_fast(n_rows);
	
	cout << "[CiM] Dense MatMul: " << n_rows << " rows x "
	     << this->cim_d_in << "x" << this->cim_d_out << " weights" << endl;
	cout << "[CiM] Config: " << this->cim_unit->get_num_tiles() << " tiles parallel, "
	     << this->cim_unit->get_bits_per_cycle() << "-bit parallelism" << endl;
	cout << "[CiM] Analytical cycles: " << cim_cycles << " (throughput: "
	     << this->cim_unit->get_num_tiles() << " rows/cycle)" << endl;
	
	this->cim_started = true;
	// run_fast() sets module_state_ to DONE, so is_cim_done() will return true
#else
	throw std::runtime_error(
	    "tick-by-tick CiM start requires explicit SpMM rows; generated input is forbidden");
#endif
}

void NeuraTile::tick_cim()
{
	if (this->cim_enabled && this->cim_unit != nullptr)
	{
		this->cim_unit->tick();
		
		// Drain output FIFO (discard for now, just to keep CiM running)
		CiMOutputPacket output;
		while (this->cim_unit->pop_output(output))
		{
			// Output processed - in real integration, would forward to next stage
		}
	}
}

bool NeuraTile::is_cim_done() const
{
	if (!this->cim_enabled || this->cim_unit == nullptr)
	{
		return true;
	}
	return this->cim_unit->is_done();
}

#endif // ENABLE_CIM

// =============================================================================
// PE Array SpMM Methods (our DyGNN design)
// =============================================================================
#ifdef ENABLE_PE_ARRAY

void NeuraTile::init_pe_array()
{
	if (this->pe_array_initialized) return;
	
	// PE Array is initialized in constructor
	// This method is for loading matrix data
	this->pe_array_initialized = true;
	cout << "[PE Array] Ready for SpMM computation" << endl;
}

void NeuraTile::tick_pe_array()
{
	if (this->pe_array_enabled && this->pe_array_spmm != nullptr)
	{
		this->pe_array_spmm->tick();
	}
}

bool NeuraTile::is_pe_array_done() const
{
	if (!this->pe_array_enabled || this->pe_array_spmm == nullptr)
	{
		return true;
	}
	return this->pe_array_spmm->is_done();
}

void NeuraTile::print_pe_array_stats()
{
	if (this->pe_array_spmm == nullptr) return;
	
	PEArraySpMM::Stats stats = this->pe_array_spmm->get_stats();
	
	cout << "\n========== PE Array SpMM Statistics ==========" << endl;
	cout << "Total Cycles:      " << stats.total_cycles << endl;
	cout << "Compute Cycles:    " << stats.compute_cycles << endl;
	cout << "Stall Cycles:      " << stats.stall_cycles << endl;
	cout << "Memory Stalls:     " << stats.memory_stall_cycles << endl;
	cout << "PE Utilization:    " << (stats.pe_utilization * 100.0f) << "%" << endl;
	cout << "Total MAC Ops:     " << stats.total_mac_ops << endl;
	cout << "F Cache Hits:      " << stats.f_cache_hits << endl;
	cout << "F Cache Misses:    " << stats.f_cache_misses << endl;
	float cache_hit_rate = (stats.f_cache_hits + stats.f_cache_misses) > 0 ?
		(float)stats.f_cache_hits / (stats.f_cache_hits + stats.f_cache_misses) * 100.0f : 0.0f;
	cout << "F Cache Hit Rate:  " << cache_hit_rate << "%" << endl;
	cout << "L1 Bank Conflicts: " << stats.l1_bank_conflict_stalls << " cycles" << endl;
	cout << "L2 Bank Conflicts: " << stats.l2_bank_conflict_stalls << " cycles" << endl;
	cout << "Work Steals:       " << stats.work_steals << endl;
	cout << "NoC Steal Count:   " << stats.noc_steal_count << endl;
	cout << "NoC Steal Latency: " << stats.noc_steal_latency_total_cycles << " cycles" << endl;
	cout << "NoC Avg Hops:      " << stats.noc_steal_avg_hops << endl;
	cout << "A-load DRAM Req:   " << stats.a_load_dram_requests << endl;
	cout << "Load Imbalance:    " << (stats.load_imbalance * 100.0f) << "%" << endl;
	cout << "===============================================\n" << endl;
}

#ifdef USE_PE_ARRAY_MODE

// ---------------------------------------------------------------------------
// Binary CSR cache helpers — speeds up repeated runs by skipping text parsing
// Cache format:
//   [uint64_t magic = 0xC5B0CACE]
//   [uint64_t n_rows]
//   [uint64_t nnz]
//   [int      row_ptr[n_rows+1]]
//   [int      col_idx[nnz]]
//   [float    values[nnz]]
// ---------------------------------------------------------------------------
static constexpr uint64_t CSR_CACHE_MAGIC = 0xC5B0CACE;

// Return the newest mtime (seconds since epoch) among the three source text
// files.  Returns 0 on any stat failure so the cache will be treated as stale.
static time_t newest_source_mtime(const string& matrix_path) {
	struct stat sb;
	time_t newest = 0;
	for (const char* name : {"row_ptr.txt", "col_indices.txt", "data.txt"}) {
		string p = matrix_path + "/" + name;
		if (::stat(p.c_str(), &sb) != 0) return 0;
		if (sb.st_mtime > newest) newest = sb.st_mtime;
	}
	return newest;
}

static bool try_load_csr_cache(const string& cache_path,
                               time_t source_mtime,
                               vector<int>& row_ptr,
                               vector<int>& col_idx,
                               vector<float>& values,
                               int& n_out, int& nnz_out) {
	struct stat sb;
	if (::stat(cache_path.c_str(), &sb) != 0) return false;   // no cache file
	if (sb.st_mtime < source_mtime) {
		cout << "[CSR-Cache] Stale cache, will rebuild" << endl;
		return false;
	}

	FILE* fp = fopen(cache_path.c_str(), "rb");
	if (!fp) return false;

	uint64_t magic = 0, nr = 0, nz = 0;
	if (fread(&magic, sizeof(magic), 1, fp) != 1 || magic != CSR_CACHE_MAGIC) {
		fclose(fp); return false;
	}
	if (fread(&nr, sizeof(nr), 1, fp) != 1) { fclose(fp); return false; }
	if (fread(&nz, sizeof(nz), 1, fp) != 1) { fclose(fp); return false; }

	n_out   = static_cast<int>(nr);
	nnz_out = static_cast<int>(nz);

	row_ptr.resize(n_out + 1);
	col_idx.resize(nnz_out);
	values.resize(nnz_out);

	bool ok = true;
	ok = ok && fread(row_ptr.data(), sizeof(int), n_out + 1, fp) == static_cast<size_t>(n_out + 1);
	ok = ok && fread(col_idx.data(), sizeof(int), nnz_out,   fp) == static_cast<size_t>(nnz_out);
	ok = ok && fread(values.data(),  sizeof(float), nnz_out, fp) == static_cast<size_t>(nnz_out);
	fclose(fp);

	if (!ok) {
		cerr << "[CSR-Cache] Truncated cache file, will rebuild" << endl;
		row_ptr.clear(); col_idx.clear(); values.clear();
		return false;
	}
	return true;
}

static bool save_csr_cache(const string& cache_path,
                           const vector<int>& row_ptr,
                           const vector<int>& col_idx,
                           const vector<float>& values,
                           int n, int nnz) {
	FILE* fp = fopen(cache_path.c_str(), "wb");
	if (!fp) {
		cerr << "[CSR-Cache] Cannot write cache: " << cache_path << endl;
		return false;
	}
	uint64_t magic = CSR_CACHE_MAGIC;
	uint64_t nr = static_cast<uint64_t>(n);
	uint64_t nz = static_cast<uint64_t>(nnz);
	fwrite(&magic, sizeof(magic), 1, fp);
	fwrite(&nr,    sizeof(nr),    1, fp);
	fwrite(&nz,    sizeof(nz),    1, fp);
	fwrite(row_ptr.data(), sizeof(int),   n + 1, fp);
	fwrite(col_idx.data(), sizeof(int),   nnz,   fp);
	fwrite(values.data(),  sizeof(float), nnz,   fp);
	fclose(fp);
	return true;
}

// Load CSR data from dataset
bool NeuraTile::load_pe_array_data()
{
	// Get matrix path from global counters
	// program_data_folder = dataset_root + "/matrix"
	string matrix_path = this->gcp->get_program_data_folder() + "/mat_A";

	cout << "[PE Array] Loading CSR data from: " << matrix_path << endl;

	// Load header
	ifstream header_file(matrix_path + "/header.csv");
	if (!header_file.is_open()) {
		cerr << "[PE Array] Cannot open header: " << matrix_path << "/header.csv" << endl;
		return false;
	}

	int n = 0, nnz = 0;
	string line;
	while (getline(header_file, line)) {
		if (line.find("x_dim,") == 0) n = stoi(line.substr(6));
		else if (line.find("nnz,") == 0) nnz = stoi(line.substr(4));
	}
	header_file.close();

	cout << "[PE Array] Matrix: n=" << n << ", nnz=" << nnz << endl;

	// --- Try binary CSR cache first ---
	vector<int> row_ptr;
	vector<int> col_idx;
	vector<float> values;
	const string cache_path = matrix_path + "/csr.cache";
	const time_t src_mtime = newest_source_mtime(matrix_path);
	int cached_n = 0, cached_nnz = 0;

	if (try_load_csr_cache(cache_path, src_mtime, row_ptr, col_idx, values,
	                        cached_n, cached_nnz)) {
		n = cached_n;
		nnz = cached_nnz;
		cout << "[CSR-Cache] Loaded binary cache (" << cache_path << "): n="
		     << n << ", nnz=" << nnz << endl;
	} else {
		// Fall back to text file parsing
		// Load row_ptr
		ifstream row_ptr_file(matrix_path + "/row_ptr.txt");
		if (!row_ptr_file.is_open()) {
			cerr << "[PE Array] Cannot open row_ptr.txt" << endl;
			return false;
		}
		row_ptr.reserve(n + 1);
		int val;
		while (row_ptr_file >> val) row_ptr.push_back(val);
		row_ptr_file.close();

		// Load col_indices
		ifstream col_idx_file(matrix_path + "/col_indices.txt");
		if (!col_idx_file.is_open()) {
			cerr << "[PE Array] Cannot open col_indices.txt" << endl;
			return false;
		}
		col_idx.reserve(nnz);
		while (col_idx_file >> val) col_idx.push_back(val);
		col_idx_file.close();

		// Load values
		ifstream data_file(matrix_path + "/data.txt");
		if (!data_file.is_open()) {
			cerr << "[PE Array] Cannot open data.txt" << endl;
			return false;
		}
		values.reserve(nnz);
		float fval;
		while (data_file >> fval) values.push_back(fval);
		data_file.close();

		// Save cache for subsequent runs
		if (save_csr_cache(cache_path, row_ptr, col_idx, values, n, nnz)) {
			cout << "[CSR-Cache] Saved binary cache: " << cache_path << endl;
		}
	}
	
	// Use fixed feature dimension (GNN hidden layer typically 128-256)
	// Note: Original Cora has 1433 features, but we use hidden dim for SpMM
	int feature_dim = 128;  // GNN hidden dimension (not raw input features)
	cout << "[PE Array] Using hidden dim (not raw features): " << feature_dim << endl;
	
	cout << "[PE Array] Feature dim: " << feature_dim << endl;
	
		// Official execution requires an exact-size FP32 feature matrix.
		{
			const std::string f_init_path = this->gcp->arg_parse->get_f_init_path();
			if (!f_init_path.empty()) {
			std::ifstream fin(f_init_path, std::ios::binary);
			if (fin.is_open()) {
				fin.seekg(0, std::ios::end);
				const auto bytes = fin.tellg();
				fin.seekg(0, std::ios::beg);
				const size_t expected = static_cast<size_t>(n) * static_cast<size_t>(feature_dim) * sizeof(float);
				if (static_cast<size_t>(bytes) == expected) {
					this->pe_array_F.resize(static_cast<size_t>(n) * feature_dim);
					fin.read(reinterpret_cast<char*>(this->pe_array_F.data()),
					         static_cast<std::streamsize>(expected));
					cout << "[PE Array] Loaded F from f-init: " << f_init_path
					     << " (" << n << "x" << feature_dim << ")" << endl;
					} else {
						cerr << "[PE Array] f-init size mismatch (" << bytes << " vs " << expected
						     << ")" << endl;
						return false;
					}
				} else {
					cerr << "[PE Array] Cannot open f-init: " << f_init_path << endl;
					return false;
				}
			} else {
				cerr << "[PE Array] --f-init is required" << endl;
				return false;
			}
		}

	// Persist CSR arrays for incremental controller
	this->pe_array_row_ptr = row_ptr;
	this->pe_array_col_idx = col_idx;
	this->pe_array_values = values;
	
	// Load into PE Array
	bool outer_streaming_enabled = true;
	if (this->gcp && this->gcp->arg_parse) {
		const std::string mode = this->gcp->arg_parse->get_spmm_mode();
		outer_streaming_enabled = (mode != "incremental" && mode != "scheduled");
	}
	this->pe_array_spmm->set_streaming_output_enabled(outer_streaming_enabled);
	this->pe_array_spmm->load_spmm(this->pe_array_row_ptr.data(),
	                               this->pe_array_col_idx.data(),
	                               this->pe_array_values.data(),
	                               n, nnz, this->pe_array_F.data(), feature_dim);
	
	this->pe_array_n = n;
	this->pe_array_nnz = nnz;
	this->pe_array_d = feature_dim;
	this->pe_array_expected_rows = 0;
	for (int i = 0; i < n; i++) {
		if (this->pe_array_row_ptr[i + 1] > this->pe_array_row_ptr[i]) {
			this->pe_array_expected_rows++;
		}
	}
	cout << "[PE Array] Expected output rows (non-empty): " << this->pe_array_expected_rows << endl;
	
	cout << "[PE Array] Data loaded successfully!" << endl;
	return true;
}

// Tick function for PE Array mode (replaces original tick())
bool NeuraTile::tick_pe_array_mode()
{
	// First tick: Initialize and load data
	if (!this->pe_array_initialized) {
		cout << "\n========================================" << endl;
		cout << "  PE ARRAY MODE (DyGNN Accelerator)" << endl;
		cout << "  [STREAMING] PE Array ??CiM (direct)" << endl;
		cout << "========================================\n" << endl;
		
		if (!this->load_pe_array_data()) {
			cerr << "[PE Array] Failed to load data, aborting." << endl;
			return true;  // End simulation
		}

		// Configure DyGNN controller (parsed by ArgParse, stored in GlobalCounters)
		if (this->gcp && this->gcp->arg_parse) {
			const auto spmm_mode = this->gcp->arg_parse->get_spmm_mode();
			const auto cim_mode = this->gcp->arg_parse->get_cim_mode();
			const auto phase_mode = this->gcp->arg_parse->get_phase_mode();
			const auto arch_variant = this->gcp->arg_parse->get_arch_variant();
			const bool compute_accuracy = this->gcp->arg_parse->get_compute_accuracy();
			const bool cim_noise = this->gcp->arg_parse->get_cim_noise();
			const std::string w_init_path = this->gcp->arg_parse->get_w_init_path();
			const std::string w_init_l1_path = this->gcp->arg_parse->get_w_init_l1_path();
			const std::string effective_w_init_l1 =
				!w_init_l1_path.empty() ? w_init_l1_path : w_init_path;
			this->multi_ts_total = this->gcp->arg_parse->get_timesteps();
			this->multi_ts_runtime_cim_energy_pj = 0.0;
			this->multi_ts_eval_cim_energy_pj = 0.0;
			this->multi_ts_eval_energy_valid = false;
			const auto delta_dir = this->gcp->arg_parse->get_delta_dir();
			cout << "[DyGNN] spmm-mode=" << spmm_mode
			     << " cim-mode=" << cim_mode
			     << " phase-mode=" << phase_mode
			     << " arch-variant=" << arch_variant
			     << " threshold=" << this->gcp->arg_parse->get_mode_threshold()
			     << " timesteps=" << this->multi_ts_total << endl;
			cout << "[DyGNN] aggregation=exact_dynamic_row_mean"
			     << " self_loop=none destination_degree=csr_row_nnz" << endl;
			if (this->gcp->arg_parse->get_force_incremental_after_warmup()) {
				cout << "[DyGNN] policy=t0_full_then_force_incremental" << endl;
			}
			if (compute_accuracy) {
				cout << "[DyGNN] compute-accuracy=on" << endl;
				if (!effective_w_init_l1.empty()) {
					cout << "[DyGNN] w-init-l1=" << effective_w_init_l1 << endl;
				}
			}
			if (cim_noise) {
				cout << "[DyGNN] cim-noise=on" << endl;
			}
			if (!delta_dir.empty()) {
				cout << "[DyGNN] delta-dir=" << delta_dir << endl;
			}
			if (!this->gcp->arg_parse->get_f_init_path().empty()) {
				cout << "[DyGNN] f-init=" << this->gcp->arg_parse->get_f_init_path() << endl;
			}
			if (!this->gcp->arg_parse->get_delta_a_path().empty()) {
				cout << "[DyGNN] delta-a=" << this->gcp->arg_parse->get_delta_a_path() << endl;
			}
			if (!this->gcp->arg_parse->get_delta_f_path().empty()) {
				cout << "[DyGNN] delta-f=" << this->gcp->arg_parse->get_delta_f_path() << endl;
			}

			if (this->dygnn_controller) {
				this->dygnn_controller->set_mode_from_string(spmm_mode);
				this->dygnn_controller->set_cim_mode_from_string(cim_mode);
				this->dygnn_controller->set_phase_mode_from_string(phase_mode);
				this->dygnn_controller->set_arch_variant_from_string(arch_variant);
				this->dygnn_controller->set_threshold(this->gcp->arg_parse->get_mode_threshold());
				this->dygnn_controller->set_propagation_tau(this->gcp->arg_parse->get_propagation_tau());
				this->dygnn_controller->set_propagation_scope_enabled(
					this->gcp->arg_parse->get_propagation_scope_enabled());
				this->dygnn_controller->set_indexer_mode(this->gcp->arg_parse->get_indexer_mode());
				this->dygnn_controller->set_compute_accuracy(compute_accuracy);
				this->dygnn_controller->set_cim_noise(cim_noise);
				const bool dedicated_z_sram = this->gcp->arg_parse->get_dedicated_z_sram();
				const double dedicated_z_sram_size_mb = this->gcp->arg_parse->get_dedicated_z_sram_size_mb();
				long long dedicated_z_sram_capacity_bytes = -1;
				if (dedicated_z_sram_size_mb > 0.0) {
					dedicated_z_sram_capacity_bytes =
					    static_cast<long long>(dedicated_z_sram_size_mb * 1024.0 * 1024.0);
				}
				this->dygnn_controller->set_dedicated_z_sram(dedicated_z_sram);
				this->dygnn_controller->set_dedicated_z_sram_capacity_bytes(dedicated_z_sram_capacity_bytes);
				if (dedicated_z_sram) {
					if (dedicated_z_sram_capacity_bytes > 0) {
						cout << "[DyGNN] dedicated-z-sram-capacity=" << dedicated_z_sram_size_mb << "MB" << endl;
					} else {
						cout << "[DyGNN] dedicated-z-sram-capacity=unlimited(legacy)" << endl;
					}
				}
				this->dygnn_controller->load_feature_init(this->gcp->arg_parse->get_f_init_path(),
				                                          this->pe_array_n, this->pe_array_d);

				const CiMMode parsed_cim_mode = this->dygnn_controller->get_cim_mode();
				if (compute_accuracy && parsed_cim_mode == CiMMode::NONE) {
					cerr << "[DyGNN] ERROR: --compute-accuracy requires --cim-mode != none." << endl;
					exit(1);
				}
#ifdef ENABLE_CIM
				if (parsed_cim_mode != CiMMode::NONE) {
					if (!this->cim_enabled) {
						init_cim(this->cim_d_in, this->cim_d_out);
					}
					this->dygnn_controller->set_cim(this->cim_unit, this->cim_d_out, parsed_cim_mode);
					if (compute_accuracy) {
						this->dygnn_controller->load_weights(effective_w_init_l1, this->pe_array_d, this->cim_d_out);
						if (this->cim_unit == nullptr) {
							cerr << "[DyGNN] ERROR: CiM unit is null in compute-accuracy mode." << endl;
							exit(1);
						}
						// R5 defers physical load until the first finalized L1 operation.
					}

					// ---------------------------------------------------------
					// DCIM Digital mode: pure INT8 MAC, no ADC/DAC/analog noise
					// ---------------------------------------------------------
					const bool dcim_digital = this->gcp->arg_parse->get_dcim_digital();
					if (dcim_digital) {
						this->cim_unit->set_compute_mode(CiMComputeMode::DCIM_DIGITAL);
						// Quantize weights to INT8 (same as ACIM but no PV/thermal)
						this->cim_unit->set_fp_activations(false);
						this->cim_unit->set_enable_pv(false);
						this->cim_unit->set_enable_thermal(false);
						cout << "[DyGNN] DCIM Digital mode: pure INT8 MAC, no ADC/DAC/analog noise\n";
					}

					// ---------------------------------------------------------
					// Analog CiM accuracy mode: INT8 quant + PV + thermal noise
					// ---------------------------------------------------------
					const bool dcim_accuracy = this->gcp->arg_parse->get_dcim_accuracy();
						if (dcim_accuracy) {
							const uint32_t vseed = this->gcp->arg_parse->get_variation_seed();
								const int adc_bits = this->gcp->arg_parse->get_adc_bits();
								const double thermal_sigma_mv = this->gcp->arg_parse->get_thermal_sigma_mv();
								const double thermal_sigma_scale = this->gcp->arg_parse->get_thermal_sigma_scale();
								const double temperature_k = this->gcp->arg_parse->get_temperature_k();
								const std::string thermal_model = this->gcp->arg_parse->get_thermal_model();
								const double thermal_bl_cap_ff = this->gcp->arg_parse->get_thermal_bl_cap_ff();
								const double thermal_adc_comp_sigma_mv =
								    this->gcp->arg_parse->get_thermal_adc_comp_sigma_mv();
								this->cim_unit->set_compute_mode(CiMComputeMode::ACIM_FAITHFUL);
								this->cim_unit->set_variation_seed(vseed);
								this->cim_unit->set_adc_bits(adc_bits);
								this->cim_unit->set_fp_activations(this->gcp->arg_parse->get_fp_activations());
								this->cim_unit->set_enable_pv(!this->gcp->arg_parse->get_no_pv());
								this->cim_unit->set_enable_thermal(!this->gcp->arg_parse->get_no_thermal());
								this->cim_unit->set_enable_pv_compensation(!this->gcp->arg_parse->get_no_pv_comp());
									this->cim_unit->set_thermal_sigma_override_mv(static_cast<float>(thermal_sigma_mv));
									this->cim_unit->set_thermal_sigma_scale(static_cast<float>(thermal_sigma_scale));
									this->cim_unit->set_temperature_k(static_cast<float>(temperature_k));
									this->cim_unit->set_thermal_model(thermal_model);
									this->cim_unit->set_thermal_bl_cap_ff(static_cast<float>(thermal_bl_cap_ff));
									this->cim_unit->set_thermal_adc_comp_sigma_mv(
									    static_cast<float>(thermal_adc_comp_sigma_mv));
									this->cim_unit->set_read_avg_n(this->gcp->arg_parse->get_read_avg_n());
									this->cim_unit->set_activation_scale_mode(this->gcp->arg_parse->get_act_scale_mode());
									this->cim_unit->set_act_calib_rows(this->gcp->arg_parse->get_act_calib_rows());
									this->cim_unit->set_act_calib_clip(static_cast<float>(this->gcp->arg_parse->get_act_calib_clip()));
								if (this->cim_unit->get_enable_pv()) {
									this->cim_unit->initialize_process_variation();
								}
									cout << "[DyGNN] Analog CiM accuracy mode (seed=" << vseed
									     << ", adc=" << adc_bits << "b"
								     << ", bpc=" << this->cim_unit->get_bits_per_cycle()
								     << ", pv=" << (this->cim_unit->get_enable_pv() ? "on" : "off")
								     << ", thermal=" << (this->cim_unit->get_enable_thermal() ? "on" : "off")
								     << ", pv_comp=" << (this->cim_unit->get_enable_pv_compensation() ? "on" : "off")
								     << ", thermal_sigma_mv=" << this->cim_unit->get_resolved_thermal_sigma_mv()
								     << ", thermal_model=" << this->cim_unit->get_thermal_model()
								     << ", thermal_bl_cap_ff=" << this->cim_unit->get_thermal_bl_cap_ff()
								     << ", thermal_adc_comp_sigma_mv=" << this->cim_unit->get_thermal_adc_comp_sigma_mv()
								     << ", act_scale_mode=" << this->cim_unit->get_activation_scale_mode_str()
								     << ", act_calib_rows=" << this->cim_unit->get_act_calib_rows()
							     << ", read_avg_n=" << this->cim_unit->get_read_avg_n()
							     << (this->cim_unit->get_fp_activations() ? ", FP32 act" : "")
							     << ")\n";
						}

					// ---------------------------------------------------------
					// Multi-layer GCN: load L2+ weights
					// ---------------------------------------------------------
					const int num_layers = this->gcp->arg_parse->get_num_layers();
					if (num_layers >= 2) {
						std::string wl2 = this->gcp->arg_parse->get_w_init_l2_path();
						// Load L2 weights (d_hidden=cim_d_out → d_out_l2 defaults to same)
						this->dygnn_controller->load_weights_l2(wl2, this->cim_d_out, this->cim_d_out);
						int num_layers = this->gcp->arg_parse->get_num_layers();
						cout << "[DyGNN] " << num_layers << "-layer GCN: L2 weights loaded ("
						     << this->cim_d_out << "x" << this->cim_d_out << ")\n";
						// Load L3-L5 weights if needed
						if (num_layers >= 3) {
							std::string wl3 = this->gcp->arg_parse->get_w_init_l3_path();
							this->dygnn_controller->load_weights_l3(wl3, this->cim_d_out, this->cim_d_out);
						}
						if (num_layers >= 4) {
							std::string wl4 = this->gcp->arg_parse->get_w_init_l4_path();
							this->dygnn_controller->load_weights_l4(wl4, this->cim_d_out, this->cim_d_out);
						}
						if (num_layers >= 5) {
							std::string wl5 = this->gcp->arg_parse->get_w_init_l5_path();
							this->dygnn_controller->load_weights_l5(wl5, this->cim_d_out, this->cim_d_out);
						}
					}
				} else {
					this->dygnn_controller->set_cim(nullptr, this->cim_d_out, CiMMode::NONE);
				}
#else
				if (compute_accuracy) {
					cerr << "[DyGNN] ERROR: --compute-accuracy requires ENABLE_CIM build." << endl;
					exit(1);
				}
				this->dygnn_controller->set_cim(nullptr, this->cim_d_out, parsed_cim_mode);
				if (parsed_cim_mode != CiMMode::NONE) {
					cerr << "[DyGNN] WARN: --cim-mode requested but ENABLE_CIM is disabled at build-time; falling back to none." << endl;
				}
#endif
			}
		}
		this->pe_array_initialized = true;
		this->pe_array_start_cycle = this->gcp->get_cycle();
		cout << "[PE Array] Starting simulation at cycle " << this->pe_array_start_cycle << endl;

		bool streaming_enabled_for_mode = true;
		if (this->gcp && this->gcp->arg_parse) {
			const std::string mode = this->gcp->arg_parse->get_spmm_mode();
			if (mode == "incremental" || mode == "scheduled") {
				streaming_enabled_for_mode = false;
			}
		}
		
#ifdef ENABLE_CIM
		// Initialize CiM for streaming (starts immediately with PE Array)
		{
			if (streaming_enabled_for_mode) {
				if (!this->cim_enabled) {
					init_cim(this->cim_d_in, this->cim_d_out);
				}
				// Start CiM matmul in tick-by-tick mode
				if (!this->cim_unit->start_matmul(this->pe_array_expected_rows, this->cim_d_in, this->cim_d_out, false)) {
					cerr << "[Streaming] ERROR: failed to start CiM matmul." << endl;
					return true;
				}
				this->cim_started = true;
				this->cim_start_cycle = this->pe_array_start_cycle;
				cout << "[Streaming] CiM initialized for TICK-BY-TICK simulation" << endl;
				cout << "[Streaming] CiM config: " << this->cim_unit->get_num_tiles() << " tiles, "
				     << this->cim_unit->get_bits_per_cycle() << "-bit parallelism" << endl;
			}
		}
#endif
	}

	// =========================================================================
	// --skip-warmup-sim: bypass t=0 cycle-accurate PE simulation
	// Compute Y_t0 = A × F analytically, then jump to multi-TS driver
	// =========================================================================
	if (this->gcp && this->gcp->arg_parse &&
	    this->gcp->arg_parse->get_skip_warmup_sim() &&
	    !this->multi_ts_full_done && this->pe_array_initialized) {

#if DESPOINA_V3_FUNCTIONAL_DCIM
		if (this->gcp->arg_parse->get_num_layers() >= 2) {
			throw std::runtime_error(
				"v3 closed-loop publication path forbids --skip-warmup-sim");
		}
#endif

		cout << "\n[Skip-Warmup] Bypassing t=0 cycle-accurate simulation" << endl;
		const int n = this->pe_array_n;
		const int d = this->pe_array_d;
		const int nnz = this->pe_array_nnz;

		// 1. Compute Y_t0 = A × F in software (SpMM)
		std::vector<float> y_t0(static_cast<size_t>(n) * d, 0.0f);
		for (int i = 0; i < n; i++) {
			const int row_start = this->pe_array_row_ptr[i];
			const int row_end = this->pe_array_row_ptr[i + 1];
			for (int p = row_start; p < row_end; p++) {
				const int j = this->pe_array_col_idx[p];
				const float v = this->pe_array_values[p];
				const size_t dst_base = static_cast<size_t>(i) * d;
				const size_t src_base = static_cast<size_t>(j) * d;
				for (int k = 0; k < d; k++) {
					y_t0[dst_base + k] += v * this->pe_array_F[src_base + k];
				}
			}
		}
		cout << "[Skip-Warmup] Y_t0 computed: " << n << "x" << d
		     << " (" << nnz << " nnz)" << endl;

		// 2. Build synthetic PE stats (analytical estimates)
		PEArraySpMM::Stats synth_stats = {};
		synth_stats.total_mac_ops = static_cast<long long>(nnz) * d;
		// Analytical cycle estimate: nnz*d / (num_PEs * SIMD_width)
		// 256 PEs × 8 SIMD = 2048 MACs/cycle
		const long long macs_per_cycle = 2048;
		synth_stats.total_cycles = std::max(1LL, synth_stats.total_mac_ops / macs_per_cycle);
		synth_stats.compute_cycles = synth_stats.total_cycles;
		synth_stats.f_cache_misses = static_cast<long long>(n);  // cold start
		synth_stats.dram_requests = synth_stats.f_cache_misses;
		synth_stats.pe_utilization = 1.0f;
		cout << "[Skip-Warmup] Synthetic stats: " << synth_stats.total_mac_ops
		     << " MACs, ~" << synth_stats.total_cycles << " cycles (analytical)" << endl;

		// 3. Build CSR for DyGNN controller
		WorkStealingScheduler::CSRMatrix a_new;
		a_new.n = n;
		a_new.nnz = nnz;
		a_new.row_ptr = this->pe_array_row_ptr;
		a_new.col_idx = this->pe_array_col_idx;
		a_new.values = this->pe_array_values;

		// 4. Initialize DyGNN controller with synthetic stats
		this->multi_ts_full_done = true;
		this->multi_ts_full_cycles = synth_stats.total_cycles;
		this->dygnn_controller->init_from_loaded_graph(synth_stats, n, nnz, d);
		DualModeSpMM::StepResult t0_eval =
			this->dygnn_controller->run_full_from_existing_y(a_new, y_t0, synth_stats);

		// 5. Record t=0 result
		TimestepResult t0_result;
		t0_result.timestep = 0;
		t0_result.mode_used = t0_eval.mode_used;
		t0_result.phase_mode_used = t0_eval.phase_mode_used;
		t0_result.cycles = t0_eval.total_cycles;
		t0_result.pe_stats = t0_eval.pe_stats;
					t0_result.phase1 = t0_eval.phase1;
					t0_result.phase2 = t0_eval.phase2;
					t0_result.phase3 = t0_eval.phase3;
					t0_result.row_mean_enabled = t0_eval.row_mean_enabled;
					t0_result.row_mean_scaled_rows = t0_eval.row_mean_scaled_rows;
					t0_result.row_mean_graph_changed_degree_rows = t0_eval.row_mean_graph_changed_degree_rows;
					t0_result.row_mean_changed_degree_rows = t0_eval.row_mean_changed_degree_rows;
					t0_result.row_mean_scale_elements = t0_eval.row_mean_scale_elements;
					t0_result.row_mean_residual_correction_elements = t0_eval.row_mean_residual_correction_elements;
					t0_result.row_mean_scalar_ops = t0_eval.row_mean_scalar_ops;
					t0_result.row_mean_stream_delay_cycles = t0_eval.row_mean_stream_delay_cycles;
					t0_result.row_mean_degree_read_events = t0_eval.row_mean_degree_read_events;
					t0_result.row_mean_degree_write_events = t0_eval.row_mean_degree_write_events;
					t0_result.row_mean_degree_read_bytes = t0_eval.row_mean_degree_read_bytes;
					t0_result.row_mean_degree_write_bytes = t0_eval.row_mean_degree_write_bytes;
					t0_result.row_mean_reciprocal_events = t0_eval.row_mean_reciprocal_events;
		t0_result.row_mean_enabled = t0_eval.row_mean_enabled;
		t0_result.row_mean_scaled_rows = t0_eval.row_mean_scaled_rows;
		t0_result.row_mean_graph_changed_degree_rows = t0_eval.row_mean_graph_changed_degree_rows;
		t0_result.row_mean_changed_degree_rows = t0_eval.row_mean_changed_degree_rows;
		t0_result.row_mean_scale_elements = t0_eval.row_mean_scale_elements;
		t0_result.row_mean_residual_correction_elements = t0_eval.row_mean_residual_correction_elements;
		t0_result.row_mean_scalar_ops = t0_eval.row_mean_scalar_ops;
		t0_result.row_mean_stream_delay_cycles = t0_eval.row_mean_stream_delay_cycles;
		t0_result.row_mean_degree_read_events = t0_eval.row_mean_degree_read_events;
		t0_result.row_mean_degree_write_events = t0_eval.row_mean_degree_write_events;
		t0_result.row_mean_degree_read_bytes = t0_eval.row_mean_degree_read_bytes;
		t0_result.row_mean_degree_write_bytes = t0_eval.row_mean_degree_write_bytes;
		t0_result.row_mean_reciprocal_events = t0_eval.row_mean_reciprocal_events;
		t0_result.factorized_native_pe_cycles = t0_eval.factorized_native_pe_cycles;
		t0_result.factorized_native_owner_cycles = t0_eval.factorized_native_owner_cycles;
		t0_result.factorized_state_capacity_bytes = t0_eval.factorized_state_capacity_bytes;
		t0_result.factorized_state_sram_bytes = t0_eval.factorized_state_sram_bytes;
		t0_result.factorized_state_hbm_bytes = t0_eval.factorized_state_hbm_bytes;
		t0_result.factorized_state_hbm_read_bytes = t0_eval.factorized_state_hbm_read_bytes;
		t0_result.factorized_state_hbm_write_bytes = t0_eval.factorized_state_hbm_write_bytes;
		t0_result.factorized_state_hbm_bursts = t0_eval.factorized_state_hbm_bursts;
		t0_result.factorized_state_hbm_cycles = t0_eval.factorized_state_hbm_cycles;
		t0_result.factorized_old_y_snapshot_bytes = t0_eval.factorized_old_y_snapshot_bytes;
		t0_result.factorized_old_y_sram_bytes = t0_eval.factorized_old_y_sram_bytes;
		t0_result.factorized_old_y_hbm_write_bytes = t0_eval.factorized_old_y_hbm_write_bytes;
		t0_result.factorized_old_y_hbm_read_bytes = t0_eval.factorized_old_y_hbm_read_bytes;
		t0_result.factorized_old_y_hbm_bursts = t0_eval.factorized_old_y_hbm_bursts;
		t0_result.factorized_old_y_hbm_cycles = t0_eval.factorized_old_y_hbm_cycles;
		t0_result.factorized_pe_dynamic_energy_pj = t0_eval.factorized_pe_dynamic_energy_pj;
		t0_result.indexer_ctrs = t0_eval.indexer_ctrs;
		t0_result.cim_mode = t0_eval.cim_mode;
		t0_result.combine_cycles = t0_eval.combine_cycles;
		t0_result.cim_cycles = t0_eval.cim_cycles;
		t0_result.cim_stream_cycles = t0_eval.cim_stream_cycles;
		t0_result.cim_exposed_cycles = t0_eval.cim_exposed_cycles;
		t0_result.cim_pe_backpressure_cycles = t0_eval.cim_pe_backpressure_cycles;
		t0_result.cim_input_fifo_full_rejections = t0_eval.cim_input_fifo_full_rejections;
		t0_result.cim_input_fifo_full_cycles = t0_eval.cim_input_fifo_full_cycles;
		t0_result.cim_output_fifo_full_cycles = t0_eval.cim_output_fifo_full_cycles;
		t0_result.cim_output_fifo_full_stall_cycles = t0_eval.cim_output_fifo_full_stall_cycles;
		t0_result.cim_stage3_output_block_cycles = t0_eval.cim_stage3_output_block_cycles;
		t0_result.cim_inject_no_empty_lane_cycles = t0_eval.cim_inject_no_empty_lane_cycles;
		t0_result.cim_active_lane_cycles = t0_eval.cim_active_lane_cycles;
		t0_result.cim_active_slot_cycles = t0_eval.cim_active_slot_cycles;
		t0_result.cim_input_fifo_occupancy_cycles = t0_eval.cim_input_fifo_occupancy_cycles;
		t0_result.cim_output_fifo_occupancy_cycles = t0_eval.cim_output_fifo_occupancy_cycles;
		t0_result.cim_max_active_lanes = t0_eval.cim_max_active_lanes;
		t0_result.cim_rows = t0_eval.cim_rows;
		t0_result.y_dram_read_cycles = t0_eval.y_dram_read_cycles;
		t0_result.y_dram_write_cycles = t0_eval.y_dram_write_cycles;
		t0_result.z_dram_rw_cycles = t0_eval.z_dram_rw_cycles;
		t0_result.z_sram_rw_bytes = t0_eval.z_sram_rw_bytes;
		t0_result.row_detect_cycles = t0_eval.row_detect_cycles;
		t0_result.row_detect_checks = t0_eval.row_detect_checks;
		t0_result.row_detect_sram_reads = t0_eval.row_detect_sram_reads;
		t0_result.arch_ctrs = t0_eval.arch_ctrs;
#if DESPOINA_V3_FUNCTIONAL_DCIM
		t0_result.finalized_rows = static_cast<long long>(t0_result.arch_ctrs.digital_stream_rows);
		t0_result.finalized_packets = static_cast<long long>(t0_result.arch_ctrs.digital_stream_packets);
		t0_result.dcim_scalar_macs = static_cast<long long>(t0_result.arch_ctrs.digital_stream_mac_ops);
		t0_result.dcim_compute_cycles = static_cast<long long>(t0_result.arch_ctrs.digital_stream_compute_cycles);
		t0_result.tile_busy_work_cycles = static_cast<long long>(t0_result.arch_ctrs.tile_busy_work_cycles);
		t0_result.causal_stream_elapsed_cycles = static_cast<long long>(t0_result.arch_ctrs.causal_stream_elapsed_cycles);
		t0_result.exposed_tail_cycles = static_cast<long long>(t0_result.arch_ctrs.exposed_tail_cycles);
		t0_result.dcim_streams = static_cast<long long>(t0_result.arch_ctrs.digital_stream_count);
		t0_result.dcim_drain_sensitivity_cycles = static_cast<long long>(t0_result.arch_ctrs.digital_stream_drain_sensitivity_cycles);
		t0_result.dcim_compute_only_exposed_cycles = static_cast<long long>(t0_result.arch_ctrs.dcim_exposed_cycles);
		t0_result.dcim_plus6_exposed_cycles = static_cast<long long>(t0_result.arch_ctrs.digital_stream_plus6_exposed_cycles);
#endif
		t0_result.accuracy = t0_eval.accuracy;
		t0_result.y_accuracy = t0_eval.y_accuracy;
		this->capture_exact_hbm_snapshot(t0_result);
		this->multi_ts_results.push_back(t0_result);
		this->write_multi_ts_stats();

		cout << "[Skip-Warmup] t=0 analytical done: " << t0_eval.total_cycles << " cycles" << endl;
		if (this->gcp->arg_parse->get_force_incremental_after_warmup()) {
			this->dygnn_controller->set_mode_from_string("incremental");
			cout << "[Skip-Warmup] policy lock: t=1..N-1 forced to incremental" << endl;
		}

		// 6. Build adjacency sets for multi-TS driver
		std::vector<std::unordered_set<int>> adj_sets(static_cast<size_t>(n));
		for (int i = 0; i < n; i++) {
			const int start = a_new.row_ptr[i];
			const int end = a_new.row_ptr[i + 1];
			for (int p = start; p < end; p++) {
				const int j = a_new.col_idx[p];
				if (j >= 0 && j < n) adj_sets[static_cast<size_t>(i)].insert(j);
			}
		}
		auto rebuild_csr_from_adj_sets = [&]() {
			a_new.row_ptr.assign(n + 1, 0);
			a_new.col_idx.clear();
			a_new.values.clear();
			int ptr = 0;
			for (int i = 0; i < n; i++) {
				a_new.row_ptr[i] = ptr;
				std::vector<int> cols(adj_sets[static_cast<size_t>(i)].begin(),
				                      adj_sets[static_cast<size_t>(i)].end());
				std::sort(cols.begin(), cols.end());
				for (int v : cols) {
					a_new.col_idx.push_back(v);
					a_new.values.push_back(1.0f);
					ptr++;
				}
			}
			a_new.row_ptr[n] = ptr;
			a_new.nnz = ptr;
		};

		// 7. Run incremental timesteps t=1..N-1
		const int total_ts = this->multi_ts_total;
		const std::string delta_dir = this->gcp->arg_parse->get_delta_dir();
		const int num_layers_runtime = this->gcp->arg_parse->get_num_layers();
		long long incr_total_cycles = 0;

		for (int t = 1; t < total_ts; ++t) {
			std::string delta_a_file, delta_f_file;
			if (!delta_dir.empty()) {
				delta_a_file = delta_dir + "/delta_a_t" + std::to_string(t) + ".txt";
				delta_f_file = delta_dir + "/delta_f_t" + std::to_string(t) + ".txt";
			} else {
				delta_a_file = this->gcp->arg_parse->get_delta_a_path();
				delta_f_file = this->gcp->arg_parse->get_delta_f_path();
			}

			try {
				auto delta_a = DeltaConverter::parse_delta_a_file(delta_a_file);
				auto delta_f = DeltaConverter::parse_delta_f_file(delta_f_file, d);
				cout << "[Multi-TS] t=" << t << " loading: " << delta_a_file << endl;

				for (const auto& u : delta_a) {
					if (u.src < 0 || u.src >= n) continue;
					if (u.dst < 0 || u.dst >= n) continue;
					auto& s = adj_sets[static_cast<size_t>(u.src)];
					if (u.is_add) s.insert(u.dst); else s.erase(u.dst);
				}
				rebuild_csr_from_adj_sets();

				// Snapshot Y for each layer before L1 runs
				std::vector<float> y_before_l1;
				if (num_layers_runtime >= 2) {
#if DESPOINA_V3_FUNCTIONAL_DCIM
					y_before_l1 = this->dygnn_controller->get_z_current();
#else
					y_before_l1 = this->dygnn_controller->get_y_current();
#endif
				}
					std::map<int, std::vector<float>> y_before_layer; // key = layer_id (2,3,...)
							for (int l = 2; l < num_layers_runtime; l++) {
#if DESPOINA_V3_FUNCTIONAL_DCIM
								y_before_layer[l] = this->dygnn_controller->get_z_layer(l);
#else
						y_before_layer[l] = this->dygnn_controller->get_y_layer(l);
#endif
				}

							this->dygnn_controller->begin_runtime_timestep_trace();
							DualModeSpMM::StepResult res = this->dygnn_controller->run_once_incremental(a_new, delta_a, delta_f);

				// Generic N-layer cascade
				const int num_higher_layers = std::max(0, num_layers_runtime - 1);
				std::vector<long long> higher_layer_cycles_vec(num_higher_layers, 0);
				// Accumulate higher-layer stats for energy accounting
				PEArraySpMM::Stats higher_layer_pe_stats_accum{};
				long long higher_cim_cycles_accum = 0;
				long long higher_cim_stream_cycles_accum = 0;
				long long higher_cim_exposed_cycles_accum = 0;
				long long higher_combine_cycles_accum = 0;
				long long higher_cim_pe_bp_accum = 0;
				long long higher_cim_if_full_accum = 0;
				long long higher_cim_active_lane_accum = 0;
				long long higher_cim_active_slot_accum = 0;
				long long higher_y_dram_read_accum = 0;
				long long higher_y_dram_write_accum = 0;
				long long higher_z_dram_rw_accum = 0;
				DualModeSpMM::StepResult::ArchAblationCounters higher_arch_accum{};
				// Depth-selectivity ablation: per-layer propagation tracking
				std::vector<int> ablation_per_layer_prop_rows;
				std::vector<long long> ablation_per_layer_mac_ops;

				if (num_layers_runtime >= 2 && !res.affected_rows.empty()) {
					// Isolation mode from CLI
					const std::string iso_mode = this->gcp->arg_parse->get_isolation_mode();
					const double nl_override = this->gcp->arg_parse->get_nl_threshold_override();
					const double prop_tau = this->gcp->arg_parse->get_propagation_tau();
					const bool scope_enabled = this->gcp->arg_parse->get_propagation_scope_enabled();
					const bool use_layer_norm = this->gcp->arg_parse->get_effective_layer_norm_enabled();
					// Unify: if propagation_tau is set and no explicit nl-threshold override,
					// use propagation_tau as the nonlinear isolation threshold
					const float nl_threshold = (!scope_enabled || iso_mode == "none") ? std::numeric_limits<float>::max()
					                         : (iso_mode == "weak") ? 0.0f
					                         : (nl_override >= 0.0) ? static_cast<float>(nl_override)
					                         : (prop_tau > 0.0) ? static_cast<float>(prop_tau)
					                         : 1e-6f;

					// Record L1 affected rows
					ablation_per_layer_prop_rows.push_back(static_cast<int>(res.affected_rows.size()));
					ablation_per_layer_mac_ops.push_back(res.pe_stats.total_mac_ops);

					// Layer-1 output: Z1 = Y1 × W1 (CiM output), H1 = ReLU(Z1)
					// BUG FIX: previously used ReLU(Y1) instead of ReLU(Z1=Y1×W1)
					const auto& z_after_l1 = this->dygnn_controller->get_z_current();
					const int d_l1 = static_cast<int>(z_after_l1.size()) / std::max(1, n);

					std::vector<float> h_prev_new(z_after_l1.size());
					for (size_t i = 0; i < z_after_l1.size(); i++)
						h_prev_new[i] = std::max(0.0f, z_after_l1[i]);

					// Per-row L2 normalization on new L1 activation (when tau > 0)
					if (use_layer_norm) {
						for (int row = 0; row < n; row++) {
							float norm_sq = 0.0f;
							const size_t base = static_cast<size_t>(row) * d_l1;
							for (int k = 0; k < d_l1; k++)
								norm_sq += h_prev_new[base + k] * h_prev_new[base + k];
							const float inv_norm = 1.0f / (std::sqrt(norm_sq) + 1e-12f);
							for (int k = 0; k < d_l1; k++)
								h_prev_new[base + k] *= inv_norm;
						}
					}

					// v3 snapshots the previously dequantized post-W Z directly.  It
					// must not execute a second FP32 dense pass to reconstruct old Z.
					std::vector<float> z_before_l1(z_after_l1.size(), 0.0f);
#if DESPOINA_V3_FUNCTIONAL_DCIM
					z_before_l1 = y_before_l1;
#else
					{
						const auto& w = this->dygnn_controller->get_weights();
						const int d_in = static_cast<int>(y_before_l1.size()) / std::max(1, n);
						if (!w.empty() && d_in > 0 && d_l1 > 0) {
							for (int row = 0; row < n; row++) {
								for (int j = 0; j < d_l1; j++) {
									float sum = 0.0f;
									for (int k = 0; k < d_in; k++) {
										sum += y_before_l1[static_cast<size_t>(row) * d_in + k] *
										       w[static_cast<size_t>(k) * d_l1 + j];
									}
									z_before_l1[static_cast<size_t>(row) * d_l1 + j] = sum;
								}
							}
						}
					}
#endif

					// Old activation + norm
					std::vector<float> h_old_l1(z_after_l1.size());
					for (size_t i = 0; i < z_after_l1.size(); i++)
						h_old_l1[i] = std::max(0.0f, z_before_l1[i]);
					if (use_layer_norm) {
						for (int row = 0; row < n; row++) {
							float norm_sq = 0.0f;
							const size_t base = static_cast<size_t>(row) * d_l1;
							for (int k = 0; k < d_l1; k++)
								norm_sq += h_old_l1[base + k] * h_old_l1[base + k];
							const float inv_norm = 1.0f / (std::sqrt(norm_sq) + 1e-12f);
							for (int k = 0; k < d_l1; k++)
								h_old_l1[base + k] *= inv_norm;
						}
					}

					std::vector<float> delta_h_prev(z_after_l1.size(), 0.0f);
					for (size_t i = 0; i < z_after_l1.size(); i++) {
						delta_h_prev[i] = h_prev_new[i] - h_old_l1[i];
					}

					std::vector<int> prev_propagating;
					if (!scope_enabled || iso_mode == "none") {
						// No filtering: all L1 affected rows propagate
						prev_propagating = res.affected_rows;
					} else {
						for (int row : res.affected_rows) {
							if (row < 0 || row >= n) continue;
							float max_abs = 0.0f;
							const size_t base = static_cast<size_t>(row) * d_l1;
							for (int j = 0; j < d_l1; j++) {
								const float v = std::fabs(delta_h_prev[base + j]);
								if (v > max_abs) max_abs = v;
							}
							if (max_abs > nl_threshold) prev_propagating.push_back(row);
						}
					}

					// No inter-layer PE overlap: L(l+1) PE depends on L(l) CiM output
					// (H_new = ReLU+Norm of CiM result, needed for all 3 phases)
#if !DESPOINA_V3_FUNCTIONAL_DCIM
					long long prev_overlap_budget = 0;
#endif
					// A tau-empty predecessor suppresses only P2/P3.  A canonical
					// nonzero ΔA still requires P1=ΔA×H_new at every higher layer.
					// Use the CSR, rather than raw update-list length, so cancelled
					// structural updates cannot fabricate work.
					const bool has_structural_delta_a_work =
						DeltaConverter::delta_a_to_csr(delta_a, n).nnz > 0;

					for (int l = 2; l <= num_layers_runtime; l++) {
						if (prev_propagating.empty() && !has_structural_delta_a_work) break;

						// Record propagating rows entering this layer
						ablation_per_layer_prop_rows.push_back(static_cast<int>(prev_propagating.size()));

						auto l_res = this->dygnn_controller->run_higher_layer_spmm(
							l, a_new, delta_a, h_prev_new, delta_h_prev, prev_propagating);
							const long long overlap =
#if DESPOINA_V3_FUNCTIONAL_DCIM
								0;
#else
								std::min(prev_overlap_budget, l_res.total_cycles);
#endif
						higher_layer_cycles_vec[l - 2] = std::max(0LL, l_res.total_cycles - overlap);

						// Record per-layer MAC ops
						ablation_per_layer_mac_ops.push_back(l_res.pe_stats.total_mac_ops);

						// Accumulate pe_stats from higher layers
						higher_layer_pe_stats_accum.total_mac_ops += l_res.pe_stats.total_mac_ops;
						higher_layer_pe_stats_accum.f_cache_hits += l_res.pe_stats.f_cache_hits;
						higher_layer_pe_stats_accum.f_cache_misses += l_res.pe_stats.f_cache_misses;
						higher_layer_pe_stats_accum.l1_f_cache_hits += l_res.pe_stats.l1_f_cache_hits;
						higher_layer_pe_stats_accum.l2_shared_hits += l_res.pe_stats.l2_shared_hits;
						higher_layer_pe_stats_accum.l1_bank_conflict_stalls += l_res.pe_stats.l1_bank_conflict_stalls;
						higher_layer_pe_stats_accum.l2_bank_conflict_stalls += l_res.pe_stats.l2_bank_conflict_stalls;
						higher_layer_pe_stats_accum.dram_fills += l_res.pe_stats.dram_fills;
						higher_layer_pe_stats_accum.dram_requests += l_res.pe_stats.dram_requests;
						higher_layer_pe_stats_accum.a_load_dram_requests += l_res.pe_stats.a_load_dram_requests;
						higher_layer_pe_stats_accum.a_load_dram_latency_sum += l_res.pe_stats.a_load_dram_latency_sum;
						higher_layer_pe_stats_accum.dram_row_buffer_hits += l_res.pe_stats.dram_row_buffer_hits;
						higher_layer_pe_stats_accum.noc_steal_count += l_res.pe_stats.noc_steal_count;
						higher_layer_pe_stats_accum.noc_steal_latency_total_cycles += l_res.pe_stats.noc_steal_latency_total_cycles;
						higher_layer_pe_stats_accum.output_egress_bytes += l_res.pe_stats.output_egress_bytes;
						higher_layer_pe_stats_accum.output_egress_rows += l_res.pe_stats.output_egress_rows;
						// Accumulate CiM/DRAM counters from higher layers
						higher_cim_cycles_accum += l_res.cim_cycles;
						higher_cim_stream_cycles_accum += l_res.cim_stream_cycles;
						higher_cim_exposed_cycles_accum += l_res.cim_exposed_cycles;
						higher_combine_cycles_accum += l_res.combine_cycles;
						higher_cim_pe_bp_accum += l_res.cim_pe_backpressure_cycles;
						higher_cim_if_full_accum += l_res.cim_input_fifo_full_cycles;
						higher_cim_active_lane_accum += l_res.cim_active_lane_cycles;
						higher_cim_active_slot_accum += l_res.cim_active_slot_cycles;
						higher_y_dram_read_accum += l_res.y_dram_read_cycles;
						higher_y_dram_write_accum += l_res.y_dram_write_cycles;
						higher_z_dram_rw_accum += l_res.z_dram_rw_cycles;
						higher_arch_accum.add(l_res.arch_ctrs);
						// CiM energy is accumulated in the CiM hardware unit directly

						if (l < num_layers_runtime && !l_res.affected_rows.empty()) {
							// Prepare next layer's input: snapshot Y_l, compute ΔH_l
#if DESPOINA_V3_FUNCTIONAL_DCIM
							const auto& yl_after = this->dygnn_controller->get_z_layer(l);
#else
							const auto& yl_after = this->dygnn_controller->get_y_layer(l);
#endif
							const auto& yl_before = y_before_layer[l];
							const int dl = static_cast<int>(yl_after.size()) / std::max(1, n);

							h_prev_new.resize(yl_after.size());
							for (size_t i = 0; i < yl_after.size(); i++)
								h_prev_new[i] = std::max(0.0f, yl_after[i]);

							// Per-row L2 normalization (new activations)
							if (use_layer_norm) {
								for (int row = 0; row < n; row++) {
									float norm_sq = 0.0f;
									const size_t base = static_cast<size_t>(row) * dl;
									for (int k = 0; k < dl; k++)
										norm_sq += h_prev_new[base + k] * h_prev_new[base + k];
									const float inv_norm = 1.0f / (std::sqrt(norm_sq) + 1e-12f);
									for (int k = 0; k < dl; k++)
										h_prev_new[base + k] *= inv_norm;
								}
							}

							// Compute old H with ReLU + norm, then delta
							delta_h_prev.assign(yl_after.size(), 0.0f);
							for (size_t i = 0; i < yl_after.size(); i++)
								delta_h_prev[i] = h_prev_new[i];
							// old activation + norm
							{
								std::vector<float> h_old_tmp(yl_after.size());
								for (size_t i = 0; i < yl_before.size(); i++)
									h_old_tmp[i] = std::max(0.0f, yl_before[i]);
								if (use_layer_norm) {
									for (int row = 0; row < n; row++) {
										float norm_sq = 0.0f;
										const size_t base = static_cast<size_t>(row) * dl;
										for (int k = 0; k < dl; k++)
											norm_sq += h_old_tmp[base + k] * h_old_tmp[base + k];
										const float inv_norm = 1.0f / (std::sqrt(norm_sq) + 1e-12f);
										for (int k = 0; k < dl; k++)
											h_old_tmp[base + k] *= inv_norm;
									}
								}
								for (size_t i = 0; i < yl_after.size(); i++)
									delta_h_prev[i] = h_prev_new[i] - h_old_tmp[i];
							}

							prev_propagating.clear();
							if (!scope_enabled || iso_mode == "none") {
								// No filtering: all affected rows propagate
								for (int row : l_res.affected_rows) {
									if (row >= 0 && row < n) prev_propagating.push_back(row);
								}
							} else {
								for (int row : l_res.affected_rows) {
									if (row < 0 || row >= n) continue;
									float max_abs = 0.0f;
									const size_t base = static_cast<size_t>(row) * dl;
									for (int j = 0; j < dl; j++) {
										const float v = std::fabs(delta_h_prev[base + j]);
										if (v > max_abs) max_abs = v;
									}
									if (max_abs > nl_threshold) prev_propagating.push_back(row);
								}
							}
								this->dygnn_controller->set_runtime_activation_trace(
									l, h_prev_new, delta_h_prev, prev_propagating);

							// Next layer's overlap budget = this layer's combine cycles
#if !DESPOINA_V3_FUNCTIONAL_DCIM
								prev_overlap_budget = l_res.combine_cycles;
#endif
									} else {
										break;
									}
					}
				}

				TimestepResult ts_result;
				ts_result.timestep = t;
				ts_result.mode_used = res.mode_used;
				ts_result.phase_mode_used = res.phase_mode_used;
				ts_result.higher_layer_cycles = higher_layer_cycles_vec;
				ts_result.per_layer_propagating_rows = ablation_per_layer_prop_rows;
				ts_result.per_layer_mac_ops = ablation_per_layer_mac_ops;
				long long sum_higher = 0;
				for (auto c : higher_layer_cycles_vec) sum_higher += c;
				ts_result.cycles = res.total_cycles + sum_higher;
				// Merge L1 pe_stats with accumulated higher-layer stats
				ts_result.pe_stats = res.pe_stats;
				ts_result.pe_stats.total_mac_ops += higher_layer_pe_stats_accum.total_mac_ops;
				ts_result.pe_stats.f_cache_hits += higher_layer_pe_stats_accum.f_cache_hits;
				ts_result.pe_stats.f_cache_misses += higher_layer_pe_stats_accum.f_cache_misses;
				ts_result.pe_stats.l1_f_cache_hits += higher_layer_pe_stats_accum.l1_f_cache_hits;
				ts_result.pe_stats.l2_shared_hits += higher_layer_pe_stats_accum.l2_shared_hits;
				ts_result.pe_stats.l1_bank_conflict_stalls += higher_layer_pe_stats_accum.l1_bank_conflict_stalls;
				ts_result.pe_stats.l2_bank_conflict_stalls += higher_layer_pe_stats_accum.l2_bank_conflict_stalls;
				ts_result.pe_stats.dram_fills += higher_layer_pe_stats_accum.dram_fills;
				ts_result.pe_stats.dram_requests += higher_layer_pe_stats_accum.dram_requests;
				ts_result.pe_stats.a_load_dram_requests += higher_layer_pe_stats_accum.a_load_dram_requests;
				ts_result.pe_stats.a_load_dram_latency_sum += higher_layer_pe_stats_accum.a_load_dram_latency_sum;
				ts_result.pe_stats.dram_row_buffer_hits += higher_layer_pe_stats_accum.dram_row_buffer_hits;
				ts_result.pe_stats.noc_steal_count += higher_layer_pe_stats_accum.noc_steal_count;
				ts_result.pe_stats.noc_steal_latency_total_cycles += higher_layer_pe_stats_accum.noc_steal_latency_total_cycles;
				ts_result.pe_stats.output_egress_bytes += higher_layer_pe_stats_accum.output_egress_bytes;
				ts_result.pe_stats.output_egress_rows += higher_layer_pe_stats_accum.output_egress_rows;
				ts_result.phase1 = res.phase1;
				ts_result.phase2 = res.phase2;
				ts_result.phase3 = res.phase3;
				ts_result.row_mean_enabled = res.row_mean_enabled;
				ts_result.row_mean_scaled_rows = res.row_mean_scaled_rows;
				ts_result.row_mean_graph_changed_degree_rows = res.row_mean_graph_changed_degree_rows;
				ts_result.row_mean_changed_degree_rows = res.row_mean_changed_degree_rows;
				ts_result.row_mean_scale_elements = res.row_mean_scale_elements;
				ts_result.row_mean_residual_correction_elements = res.row_mean_residual_correction_elements;
				ts_result.row_mean_scalar_ops = res.row_mean_scalar_ops;
				ts_result.row_mean_stream_delay_cycles = res.row_mean_stream_delay_cycles;
				ts_result.row_mean_degree_read_events = res.row_mean_degree_read_events;
				ts_result.row_mean_degree_write_events = res.row_mean_degree_write_events;
				ts_result.row_mean_degree_read_bytes = res.row_mean_degree_read_bytes;
				ts_result.row_mean_degree_write_bytes = res.row_mean_degree_write_bytes;
				ts_result.row_mean_reciprocal_events = res.row_mean_reciprocal_events;
				ts_result.factorized_native_pe_cycles = res.factorized_native_pe_cycles;
				ts_result.factorized_native_owner_cycles = res.factorized_native_owner_cycles;
				ts_result.factorized_state_capacity_bytes = res.factorized_state_capacity_bytes;
				ts_result.factorized_state_sram_bytes = res.factorized_state_sram_bytes;
				ts_result.factorized_state_hbm_bytes = res.factorized_state_hbm_bytes;
				ts_result.factorized_state_hbm_read_bytes = res.factorized_state_hbm_read_bytes;
				ts_result.factorized_state_hbm_write_bytes = res.factorized_state_hbm_write_bytes;
				ts_result.factorized_state_hbm_bursts = res.factorized_state_hbm_bursts;
				ts_result.factorized_state_hbm_cycles = res.factorized_state_hbm_cycles;
				ts_result.factorized_old_y_snapshot_bytes = res.factorized_old_y_snapshot_bytes;
				ts_result.factorized_old_y_sram_bytes = res.factorized_old_y_sram_bytes;
				ts_result.factorized_old_y_hbm_write_bytes = res.factorized_old_y_hbm_write_bytes;
				ts_result.factorized_old_y_hbm_read_bytes = res.factorized_old_y_hbm_read_bytes;
				ts_result.factorized_old_y_hbm_bursts = res.factorized_old_y_hbm_bursts;
				ts_result.factorized_old_y_hbm_cycles = res.factorized_old_y_hbm_cycles;
				ts_result.factorized_pe_dynamic_energy_pj = res.factorized_pe_dynamic_energy_pj;
				ts_result.indexer_ctrs = res.indexer_ctrs;
				ts_result.cim_mode = res.cim_mode;
				ts_result.combine_cycles = res.combine_cycles + higher_combine_cycles_accum;
				ts_result.cim_cycles = res.cim_cycles + higher_cim_cycles_accum;
				ts_result.cim_stream_cycles = res.cim_stream_cycles + higher_cim_stream_cycles_accum;
				ts_result.cim_exposed_cycles = res.cim_exposed_cycles + higher_cim_exposed_cycles_accum;
				ts_result.cim_pe_backpressure_cycles = res.cim_pe_backpressure_cycles + higher_cim_pe_bp_accum;
				ts_result.cim_input_fifo_full_rejections = res.cim_input_fifo_full_rejections;
				ts_result.cim_input_fifo_full_cycles = res.cim_input_fifo_full_cycles + higher_cim_if_full_accum;
				ts_result.cim_output_fifo_full_cycles = res.cim_output_fifo_full_cycles;
				ts_result.cim_output_fifo_full_stall_cycles = res.cim_output_fifo_full_stall_cycles;
				ts_result.cim_stage3_output_block_cycles = res.cim_stage3_output_block_cycles;
				ts_result.cim_inject_no_empty_lane_cycles = res.cim_inject_no_empty_lane_cycles;
				ts_result.cim_active_lane_cycles = res.cim_active_lane_cycles + higher_cim_active_lane_accum;
				ts_result.cim_active_slot_cycles = res.cim_active_slot_cycles + higher_cim_active_slot_accum;
				ts_result.cim_input_fifo_occupancy_cycles = res.cim_input_fifo_occupancy_cycles;
				ts_result.cim_output_fifo_occupancy_cycles = res.cim_output_fifo_occupancy_cycles;
				ts_result.cim_max_active_lanes = res.cim_max_active_lanes;
				ts_result.cim_rows = res.cim_rows;
				ts_result.y_dram_read_cycles = res.y_dram_read_cycles + higher_y_dram_read_accum;
				ts_result.y_dram_write_cycles = res.y_dram_write_cycles + higher_y_dram_write_accum;
				ts_result.z_dram_rw_cycles = res.z_dram_rw_cycles + higher_z_dram_rw_accum;
				ts_result.z_sram_rw_bytes = res.z_sram_rw_bytes;
				ts_result.row_detect_cycles = res.row_detect_cycles;
				ts_result.row_detect_checks = res.row_detect_checks;
				ts_result.row_detect_sram_reads = res.row_detect_sram_reads;
				ts_result.arch_ctrs = res.arch_ctrs;
				ts_result.arch_ctrs.add(higher_arch_accum);
#if DESPOINA_V3_FUNCTIONAL_DCIM
				ts_result.finalized_rows = static_cast<long long>(ts_result.arch_ctrs.digital_stream_rows);
				ts_result.finalized_packets = static_cast<long long>(ts_result.arch_ctrs.digital_stream_packets);
				ts_result.dcim_scalar_macs = static_cast<long long>(ts_result.arch_ctrs.digital_stream_mac_ops);
				ts_result.dcim_compute_cycles = static_cast<long long>(ts_result.arch_ctrs.digital_stream_compute_cycles);
				ts_result.tile_busy_work_cycles = static_cast<long long>(ts_result.arch_ctrs.tile_busy_work_cycles);
				ts_result.causal_stream_elapsed_cycles = static_cast<long long>(ts_result.arch_ctrs.causal_stream_elapsed_cycles);
				ts_result.exposed_tail_cycles = static_cast<long long>(ts_result.arch_ctrs.exposed_tail_cycles);
				ts_result.dcim_streams = static_cast<long long>(ts_result.arch_ctrs.digital_stream_count);
				ts_result.dcim_drain_sensitivity_cycles = static_cast<long long>(ts_result.arch_ctrs.digital_stream_drain_sensitivity_cycles);
				ts_result.dcim_compute_only_exposed_cycles = static_cast<long long>(ts_result.arch_ctrs.dcim_exposed_cycles);
				ts_result.dcim_plus6_exposed_cycles = static_cast<long long>(ts_result.arch_ctrs.digital_stream_plus6_exposed_cycles);
				ts_result.finalized_duplicate_packets = static_cast<long long>(ts_result.arch_ctrs.digital_stream_duplicate_packets);
				ts_result.finalized_invalid_packets = static_cast<long long>(ts_result.arch_ctrs.digital_stream_invalid_packets);
				ts_result.finalized_timeouts = static_cast<long long>(ts_result.arch_ctrs.digital_stream_timeouts);
				ts_result.dcim_activation_quantized_elements = static_cast<long long>(ts_result.arch_ctrs.digital_stream_activation_quantized_elements);
				ts_result.dcim_dequantized_elements = static_cast<long long>(ts_result.arch_ctrs.digital_stream_dequantized_elements);
				ts_result.dcim_comparator_tree_passes = static_cast<long long>(ts_result.arch_ctrs.digital_stream_comparator_tree_passes);
				ts_result.cim_rows = static_cast<int>(ts_result.finalized_rows);
#endif
				ts_result.accuracy = res.accuracy;
				ts_result.y_accuracy = res.y_accuracy;
				this->capture_exact_hbm_snapshot(ts_result);
				this->multi_ts_results.push_back(ts_result);
				this->write_multi_ts_stats();

				const long long total_step_cycles = res.total_cycles + sum_higher;
				incr_total_cycles += total_step_cycles;
				cout << "[Multi-TS] t=" << t << " done: " << total_step_cycles
				     << " cycles ("
				     << ((res.mode_used == DualModeSpMM::Mode::INCREMENTAL) ? "incremental" : "full");
				for (int l = 0; l < (int)higher_layer_cycles_vec.size(); l++) {
					if (higher_layer_cycles_vec[l] > 0)
						cout << ", L" << (l + 2) << "=" << higher_layer_cycles_vec[l];
				}
				cout << ")" << endl;
						} catch (const RuntimeExportError&) {
							throw;
						} catch (const std::exception& e) {
				cerr << "[Multi-TS] WARNING t=" << t << ": " << e.what() << endl;
			}
		}

		cout << "\n[Skip-Warmup] All " << (total_ts - 1) << " incremental timesteps done."
		     << " Total incremental cycles: " << incr_total_cycles << endl;

		this->multi_ts_done = true;
		return true;
	}

	// Multi-timestep driver: already done? Skip.
	if (this->multi_ts_done) {
		return true;
	}

	// Tick PE Array
	try {
		this->pe_array_spmm->tick();
	} catch (const exception& e) {
		cerr << "[PE Array] Exception during tick: " << e.what() << endl;
		return true;
	}
	
	// =========================================================================
	// ROW-WISE STREAMING: Immediately forward completed rows to CiM
	// =========================================================================
#ifdef ENABLE_CIM
	// Collect completed rows and feed to CiM (zero memory latency)
	bool streaming_enabled_for_mode = true;
	if (this->gcp && this->gcp->arg_parse) {
		const std::string mode = this->gcp->arg_parse->get_spmm_mode();
		if (mode == "incremental" || mode == "scheduled") {
			streaming_enabled_for_mode = false;
		}
	}
	if (streaming_enabled_for_mode) {
		this->tick_streaming_cim();
	}
#endif
	// =========================================================================
	
	// Tick global counters
	this->gcp->tick();
	
	// Progress update
	bool interval_tick = (this->gcp->get_cycle() % this->gcp->INTERVAL == 0);
	if (interval_tick) {
		this->gcp->print_sim_speed();
	}
	
	// Check completion
	if (this->pe_array_spmm->is_done()) {
		// =====================================================================
		// Multi-timestep driver: t=0 full SpMM done, now run t=1..N-1 incremental
		// =====================================================================
		if (this->gcp && this->gcp->arg_parse && this->dygnn_controller) {
			const std::string mode = this->gcp->arg_parse->get_spmm_mode();
			const int total_ts = this->multi_ts_total;
			const std::string delta_dir = this->gcp->arg_parse->get_delta_dir();

			// Check if we should run multi-timestep incremental
			bool run_multi_ts = (total_ts > 1) &&
			                    (mode == "incremental" || mode == "scheduled") &&
			                    !this->multi_ts_full_done;

			if (run_multi_ts) {
				// Record t=0 full stats
				this->multi_ts_full_done = true;
				this->multi_ts_full_cycles = this->gcp->get_cycle() - this->pe_array_start_cycle;
				PEArraySpMM::Stats full_stats = this->pe_array_spmm->get_stats();
				// Build A_new CSR for incremental driver
				WorkStealingScheduler::CSRMatrix a_new;
				a_new.n = this->pe_array_n;
				a_new.nnz = this->pe_array_nnz;
				a_new.row_ptr = this->pe_array_row_ptr;
				a_new.col_idx = this->pe_array_col_idx;
				a_new.values = this->pe_array_values;

				// Maintain a mutable adjacency (append-only / optional deletions) so that
				// A_new is consistent across timesteps (A_t), enabling correct A_sub extraction.
				std::vector<std::unordered_set<int>> adj_sets(static_cast<size_t>(a_new.n));
				for (int i = 0; i < a_new.n; i++) {
					const int start = a_new.row_ptr[i];
					const int end = a_new.row_ptr[i + 1];
					for (int p = start; p < end; p++) {
						const int j = a_new.col_idx[p];
						if (j >= 0 && j < a_new.n) {
							adj_sets[static_cast<size_t>(i)].insert(j);
						}
					}
				}

				auto rebuild_csr_from_adj_sets = [&]() {
					a_new.row_ptr.assign(a_new.n + 1, 0);
					a_new.col_idx.clear();
					a_new.values.clear();
					a_new.col_idx.reserve(static_cast<size_t>(a_new.nnz));
					a_new.values.reserve(static_cast<size_t>(a_new.nnz));
					int ptr = 0;
					for (int i = 0; i < a_new.n; i++) {
						a_new.row_ptr[i] = ptr;
						auto& s = adj_sets[static_cast<size_t>(i)];
						std::vector<int> cols;
						cols.reserve(s.size());
						for (int v : s) cols.push_back(v);
						std::sort(cols.begin(), cols.end());
						for (int v : cols) {
							a_new.col_idx.push_back(v);
							a_new.values.push_back(1.0f);
							ptr++;
						}
					}
					a_new.row_ptr[a_new.n] = ptr;
					a_new.nnz = ptr;
				};

				// Init DyGNN controller for incremental
				this->dygnn_controller->init_from_loaded_graph(full_stats, this->pe_array_n, this->pe_array_nnz, this->pe_array_d);

					// Store t=0 result (including optional full-step hardware accuracy)
					std::vector<float> y_t0(
						this->pe_array_spmm->get_output(),
						this->pe_array_spmm->get_output() + static_cast<size_t>(this->pe_array_n) * static_cast<size_t>(this->pe_array_d));
					this->dygnn_controller->begin_runtime_timestep_trace();
						DualModeSpMM::StepResult t0_eval =
							this->dygnn_controller->run_full_from_existing_y(a_new, y_t0, full_stats);

				TimestepResult t0_result;
				t0_result.timestep = 0;
				t0_result.mode_used = t0_eval.mode_used;
				t0_result.phase_mode_used = t0_eval.phase_mode_used;
				t0_result.cycles = t0_eval.total_cycles;
				t0_result.pe_stats = t0_eval.pe_stats;
				t0_result.phase1 = t0_eval.phase1;
				t0_result.phase2 = t0_eval.phase2;
				t0_result.phase3 = t0_eval.phase3;
				t0_result.indexer_ctrs = t0_eval.indexer_ctrs;
				t0_result.cim_mode = t0_eval.cim_mode;
				t0_result.combine_cycles = t0_eval.combine_cycles;
				t0_result.cim_cycles = t0_eval.cim_cycles;
				t0_result.cim_stream_cycles = t0_eval.cim_stream_cycles;
				t0_result.cim_exposed_cycles = t0_eval.cim_exposed_cycles;
				t0_result.cim_pe_backpressure_cycles = t0_eval.cim_pe_backpressure_cycles;
				t0_result.cim_input_fifo_full_rejections = t0_eval.cim_input_fifo_full_rejections;
				t0_result.cim_input_fifo_full_cycles = t0_eval.cim_input_fifo_full_cycles;
				t0_result.cim_output_fifo_full_cycles = t0_eval.cim_output_fifo_full_cycles;
				t0_result.cim_output_fifo_full_stall_cycles = t0_eval.cim_output_fifo_full_stall_cycles;
				t0_result.cim_stage3_output_block_cycles = t0_eval.cim_stage3_output_block_cycles;
				t0_result.cim_inject_no_empty_lane_cycles = t0_eval.cim_inject_no_empty_lane_cycles;
				t0_result.cim_active_lane_cycles = t0_eval.cim_active_lane_cycles;
				t0_result.cim_active_slot_cycles = t0_eval.cim_active_slot_cycles;
				t0_result.cim_input_fifo_occupancy_cycles = t0_eval.cim_input_fifo_occupancy_cycles;
				t0_result.cim_output_fifo_occupancy_cycles = t0_eval.cim_output_fifo_occupancy_cycles;
				t0_result.cim_max_active_lanes = t0_eval.cim_max_active_lanes;
				t0_result.cim_rows = t0_eval.cim_rows;
				t0_result.y_dram_read_cycles = t0_eval.y_dram_read_cycles;
				t0_result.y_dram_write_cycles = t0_eval.y_dram_write_cycles;
				t0_result.z_dram_rw_cycles = t0_eval.z_dram_rw_cycles;
				t0_result.z_sram_rw_bytes = t0_eval.z_sram_rw_bytes;
				t0_result.row_detect_cycles = t0_eval.row_detect_cycles;
				t0_result.row_detect_checks = t0_eval.row_detect_checks;
				t0_result.row_detect_sram_reads = t0_eval.row_detect_sram_reads;
				t0_result.arch_ctrs = t0_eval.arch_ctrs;
#if DESPOINA_V3_FUNCTIONAL_DCIM
				t0_result.finalized_rows = static_cast<long long>(t0_result.arch_ctrs.digital_stream_rows);
				t0_result.finalized_packets = static_cast<long long>(t0_result.arch_ctrs.digital_stream_packets);
				t0_result.dcim_scalar_macs = static_cast<long long>(t0_result.arch_ctrs.digital_stream_mac_ops);
				t0_result.dcim_compute_cycles = static_cast<long long>(t0_result.arch_ctrs.digital_stream_compute_cycles);
				t0_result.tile_busy_work_cycles = static_cast<long long>(t0_result.arch_ctrs.tile_busy_work_cycles);
				t0_result.causal_stream_elapsed_cycles = static_cast<long long>(t0_result.arch_ctrs.causal_stream_elapsed_cycles);
				t0_result.exposed_tail_cycles = static_cast<long long>(t0_result.arch_ctrs.exposed_tail_cycles);
				t0_result.dcim_streams = static_cast<long long>(t0_result.arch_ctrs.digital_stream_count);
				t0_result.dcim_drain_sensitivity_cycles = static_cast<long long>(t0_result.arch_ctrs.digital_stream_drain_sensitivity_cycles);
				t0_result.dcim_compute_only_exposed_cycles = static_cast<long long>(t0_result.arch_ctrs.dcim_exposed_cycles);
				t0_result.dcim_plus6_exposed_cycles = static_cast<long long>(t0_result.arch_ctrs.digital_stream_plus6_exposed_cycles);
#endif
				t0_result.accuracy = t0_eval.accuracy;
				t0_result.y_accuracy = t0_eval.y_accuracy;
					cout << "\n[Multi-TS] t=0 Full SpMM done" << endl;
				cout << "[Multi-TS] t=0 cycles: " << t0_eval.total_cycles << endl;
				if (this->gcp->arg_parse->get_force_incremental_after_warmup()) {
					this->dygnn_controller->set_mode_from_string("incremental");
					cout << "[Multi-TS] policy lock: t=1..N-1 forced to incremental" << endl;
				}

				// Initialize higher-layer pre-W Y and post-W Z state.
				const int num_layers_runtime = this->gcp->arg_parse->get_num_layers();
				if (num_layers_runtime >= 2) {
					const int n_init = a_new.n;
					const bool t0_use_layer_norm =
					    this->gcp->arg_parse->get_effective_layer_norm_enabled();
					// Start from Layer-1 post-combination output in v3.
#if DESPOINA_V3_FUNCTIONAL_DCIM
					std::vector<float> prev_y = this->dygnn_controller->get_z_current();
#else
					std::vector<float> prev_y = this->dygnn_controller->get_y_current();
#endif
					const int d_init = static_cast<int>(prev_y.size()) / std::max(1, n_init);
					for (int l = 2; l <= num_layers_runtime; l++) {
						// Paper NPF boundary: H_{l-1}=L2Norm(ReLU(Z_{l-1})).
							std::vector<float> h_prev;
							if (!DualModeSpMM::build_norm_relu(
							        prev_y, n_init, t0_use_layer_norm, &h_prev) ||
							    !this->dygnn_controller->set_h_layer(l - 1, h_prev)) {
								throw std::runtime_error("R5 t0 NormReLU state initialization failed closed");
							}
							std::vector<float> bootstrap_delta(h_prev.size(), 0.0f);
							std::vector<int> bootstrap_rows(static_cast<size_t>(n_init));
							for (int row = 0; row < n_init; ++row)
								bootstrap_rows[static_cast<size_t>(row)] = row;
							this->dygnn_controller->set_runtime_activation_trace(
								l - 1, h_prev, bootstrap_delta, bootstrap_rows);
#if DESPOINA_V3_FUNCTIONAL_DCIM
							if (!this->dygnn_controller->initialize_higher_layer_state(
									l, a_new, h_prev, &t0_eval)) {
								throw std::runtime_error("v3 t0 higher-layer closed-loop initialization failed");
						}
#else
						// Legacy nonfunctional path only.  The R5 functional path above owns
						// the single A*H application and must not run this duplicate dense pass.
						std::vector<float> y_l(prev_y.size(), 0.0f);
						for (int row = 0; row < n_init; row++) {
							for (int ptr = a_new.row_ptr[row]; ptr < a_new.row_ptr[row + 1]; ptr++) {
								const int col = a_new.col_idx[ptr];
								const float val = a_new.values.empty() ? 1.0f : a_new.values[ptr];
								for (int k = 0; k < d_init; k++) {
									y_l[static_cast<size_t>(row) * d_init + k] +=
										val * h_prev[static_cast<size_t>(col) * d_init + k];
								}
							}
						}
						this->dygnn_controller->set_y_layer(l, y_l);
#endif
						cout << "[Multi-TS] Layer-" << l << " closed-loop Y/Z initialized: "
						     << n_init << "x" << d_init << endl;
#if DESPOINA_V3_FUNCTIONAL_DCIM
							prev_y = this->dygnn_controller->get_z_layer(l);
#else
							prev_y = y_l;
#endif
						}
						std::vector<float> final_h;
						if (!DualModeSpMM::build_norm_relu(
						        prev_y, n_init, t0_use_layer_norm, &final_h) ||
						    !this->dygnn_controller->set_h_layer(num_layers_runtime, final_h)) {
							throw std::runtime_error("R5 t0 terminal NormReLU state failed closed");
						}
						this->dygnn_controller->set_runtime_activation_trace(
							num_layers_runtime, final_h, std::vector<float>(), std::vector<int>());
					}

					// Higher-layer t0 DCIM executions are complete before the t0 snapshot.
					const uint64_t t0_prior_exposed_tail =
						static_cast<uint64_t>(std::max(0LL, t0_result.exposed_tail_cycles));
					if (t0_eval.exposed_tail_cycles.value < t0_prior_exposed_tail) {
						throw std::runtime_error("R5.2 t0 exposed-tail counter regressed");
					}
					const despoina_v3::ExposedTailCycles t0_higher_tail{
						t0_eval.exposed_tail_cycles.value - t0_prior_exposed_tail};
					t0_result.cycles = static_cast<long long>(
						despoina_v3::add_exposed_tail_once(
							static_cast<uint64_t>(std::max(0LL, t0_result.cycles)),
							t0_higher_tail));
					t0_result.combine_cycles = 0;
					t0_result.cim_cycles = static_cast<long long>(
						t0_eval.causal_stream_elapsed_cycles.value);
					t0_result.cim_stream_cycles = t0_result.cim_cycles;
					t0_result.arch_ctrs = t0_eval.arch_ctrs;
					t0_result.finalized_rows = static_cast<long long>(t0_eval.arch_ctrs.digital_stream_rows);
					t0_result.finalized_packets = static_cast<long long>(t0_eval.arch_ctrs.digital_stream_packets);
					t0_result.dcim_scalar_macs = static_cast<long long>(t0_eval.arch_ctrs.digital_stream_mac_ops);
					t0_result.dcim_compute_cycles = static_cast<long long>(t0_eval.arch_ctrs.digital_stream_compute_cycles);
					t0_result.tile_busy_work_cycles = static_cast<long long>(t0_eval.arch_ctrs.tile_busy_work_cycles);
					t0_result.causal_stream_elapsed_cycles = static_cast<long long>(t0_eval.arch_ctrs.causal_stream_elapsed_cycles);
					t0_result.exposed_tail_cycles = static_cast<long long>(t0_eval.arch_ctrs.exposed_tail_cycles);
					t0_result.dcim_streams = static_cast<long long>(t0_eval.arch_ctrs.digital_stream_count);
					t0_result.dcim_drain_sensitivity_cycles = static_cast<long long>(t0_eval.arch_ctrs.digital_stream_drain_sensitivity_cycles);
					t0_result.dcim_compute_only_exposed_cycles = static_cast<long long>(t0_eval.arch_ctrs.dcim_exposed_cycles);
					t0_result.dcim_plus6_exposed_cycles = static_cast<long long>(t0_eval.arch_ctrs.digital_stream_plus6_exposed_cycles);
					t0_result.finalized_duplicate_packets = static_cast<long long>(t0_eval.arch_ctrs.digital_stream_duplicate_packets);
					t0_result.finalized_invalid_packets = static_cast<long long>(t0_eval.arch_ctrs.digital_stream_invalid_packets);
					t0_result.finalized_timeouts = static_cast<long long>(t0_eval.arch_ctrs.digital_stream_timeouts);
					t0_result.dcim_activation_quantized_elements = static_cast<long long>(t0_eval.arch_ctrs.digital_stream_activation_quantized_elements);
					t0_result.dcim_dequantized_elements = static_cast<long long>(t0_eval.arch_ctrs.digital_stream_dequantized_elements);
					t0_result.dcim_comparator_tree_passes = static_cast<long long>(t0_eval.arch_ctrs.digital_stream_comparator_tree_passes);
					t0_result.cim_rows = static_cast<int>(t0_result.finalized_rows);
					t0_result.cim_exposed_cycles = t0_result.dcim_compute_only_exposed_cycles;

					// Run incremental timesteps t=1..N-1
				const bool export_z_per_timestep =
				    this->gcp->arg_parse->get_export_z_per_timestep();
				const std::set<int> export_z_timestep_set =
				    parse_timestep_set(this->gcp->arg_parse->get_export_z_timestep_list());
				auto maybe_export_timestep_e2e =
				    [&](int timestep, TimestepResult& ts_out) {
						if (!export_z_per_timestep || num_layers_runtime < 2) return;
						if (!export_z_timestep_set.empty() &&
						    export_z_timestep_set.find(timestep) == export_z_timestep_set.end()) {
							return;
						}
							const std::string rel_dir = "per_timestep_z/t" + std::to_string(timestep);
							const std::string abs_dir = this->gcp->get_results_dir() + "/" + rel_dir;
							if (!ensure_dir_recursive(abs_dir)) {
								throw RuntimeExportError("runtime export directory creation failed: " + abs_dir);
							}
#if DESPOINA_V3_FUNCTIONAL_DCIM
							// Export the already-computed closed-loop runtime states.  Every
							// file is checked before the completion manifest is written.
							std::ostringstream bundle_meta;
								bundle_meta << "{\n  \"schema\": \"despoina-r5-runtime-v1\",\n"
							            << "  \"timestep\": " << timestep << ",\n"
							            << "  \"layers\": [\n";
							for (int layer = 1; layer <= num_layers_runtime; ++layer) {
								const std::string tag = "l" + std::to_string(layer);
								const std::vector<float>& raw_y = layer == 1
								    ? this->dygnn_controller->get_y_current()
								    : this->dygnn_controller->get_y_layer(layer);
								const std::vector<float>& z = layer == 1
								    ? this->dygnn_controller->get_z_current()
								    : this->dygnn_controller->get_z_layer(layer);
								const size_t expected_state =
								    static_cast<size_t>(a_new.n) * static_cast<size_t>(this->pe_array_d);
								if (raw_y.size() != expected_state || z.size() != expected_state) {
									throw RuntimeExportError("runtime export state shape mismatch for " + tag);
								}

								const DualModeSpMM::RuntimeLayerTrace* qtrace =
								    this->dygnn_controller->get_runtime_layer_trace(layer);
								const DualModeSpMM::RuntimeActivationTrace* atrace =
								    this->dygnn_controller->get_runtime_activation_trace(layer);
								const std::vector<int> empty_rows;
								const std::vector<int8_t> empty_i8;
								const std::vector<float> empty_f32;
								const std::vector<int>& q_rows = qtrace ? qtrace->graph_rows : empty_rows;
								const std::vector<int8_t>& q_inputs = qtrace ? qtrace->quantized_inputs : empty_i8;
								const std::vector<float>& q_scales = qtrace ? qtrace->input_scales : empty_f32;
								const std::vector<int8_t>& q_weights = qtrace ? qtrace->quantized_weights : empty_i8;
									const std::vector<float>& persisted_h =
									    this->dygnn_controller->get_h_layer(layer);
									const std::vector<float>& norm_relu = atrace
									    ? atrace->norm_relu_output : persisted_h;
									if (norm_relu.size() != expected_state) {
										throw RuntimeExportError("R5 missing paper H state for " + tag);
									}
								const std::vector<float>& next_delta = atrace
								    ? atrace->next_layer_delta : empty_f32;
								const std::vector<int>& tau_rows = atrace
								    ? atrace->next_layer_tau_rows : empty_rows;

								write_vector_checked(abs_dir + "/raw_activation_" + tag + ".f32", raw_y);
								write_vector_checked(abs_dir + "/quantized_input_" + tag + ".i8", q_inputs);
								write_vector_checked(abs_dir + "/quantized_input_scales_" + tag + ".f32", q_scales);
								write_vector_checked(abs_dir + "/quantized_input_rows_" + tag + ".i32", q_rows);
								write_vector_checked(abs_dir + "/quantized_weight_" + tag + ".i8", q_weights);
								write_vector_checked(abs_dir + "/dequantized_z_" + tag + ".f32", z);
								write_vector_checked(abs_dir + "/norm_relu_output_" + tag + ".f32", norm_relu);
								write_vector_checked(abs_dir + "/next_layer_delta_" + tag + ".f32", next_delta);
								write_vector_checked(abs_dir + "/next_layer_tau_rows_" + tag + ".i32", tau_rows);

								std::ostringstream layer_meta;
								layer_meta << "{\n"
								           << "  \"layer\": " << layer << ",\n"
								           << "  \"executed\": " << (qtrace ? "true" : "false") << ",\n"
								           << "  \"weight_scale\": " << std::setprecision(9)
								           << (qtrace ? qtrace->weight_scale : 1.0f) << ",\n"
								           << "  \"raw_activation_bytes\": " << raw_y.size() * sizeof(float) << ",\n"
								           << "  \"quantized_input_bytes\": " << q_inputs.size() * sizeof(int8_t) << ",\n"
								           << "  \"quantized_weight_bytes\": " << q_weights.size() * sizeof(int8_t) << ",\n"
								           << "  \"dequantized_z_bytes\": " << z.size() * sizeof(float) << ",\n"
								           << "  \"next_layer_delta_fnv1a64\": \"" << std::hex
								           << vector_hash(next_delta) << "\",\n"
								           << "  \"next_layer_tau_rows_fnv1a64\": \""
								           << vector_hash(tau_rows) << "\"\n}"
								           << std::dec;
								write_text_checked(abs_dir + "/layer_" + tag + ".json", layer_meta.str());
								bundle_meta << "    {\"layer\": " << layer
								            << ", \"metadata\": \"layer_" << tag << ".json\"}"
								            << (layer == num_layers_runtime ? "\n" : ",\n");
							}
							bundle_meta << "  ],\n  \"complete\": true\n}\n";
							write_text_checked(abs_dir + "/RUNTIME_EXPORT_MANIFEST.json", bundle_meta.str());
#else
							DualModeSpMM::MultiLayerResult step_ml =
						    this->dygnn_controller->run_multi_layer_gcn(num_layers_runtime, a_new, true, abs_dir);
						ts_out.e2e_accuracy = step_ml.final_accuracy;
#endif
						ts_out.e2e_exported = true;
						ts_out.e2e_export_dir = rel_dir;
						};
					maybe_export_timestep_e2e(0, t0_result);
					this->capture_exact_hbm_snapshot(t0_result);
					t0_result.row_mean_enabled = t0_eval.row_mean_enabled;
					t0_result.row_mean_scaled_rows = t0_eval.row_mean_scaled_rows;
					t0_result.row_mean_scale_elements = t0_eval.row_mean_scale_elements;
					t0_result.row_mean_scalar_ops = t0_eval.row_mean_scalar_ops;
					t0_result.row_mean_degree_read_events = t0_eval.row_mean_degree_read_events;
					t0_result.row_mean_degree_read_bytes = t0_eval.row_mean_degree_read_bytes;
					t0_result.row_mean_reciprocal_events = t0_eval.row_mean_reciprocal_events;
					t0_result.factorized_native_pe_cycles = t0_eval.factorized_native_pe_cycles;
					t0_result.factorized_native_owner_cycles = t0_eval.factorized_native_owner_cycles;
					t0_result.factorized_state_capacity_bytes = t0_eval.factorized_state_capacity_bytes;
					t0_result.factorized_state_sram_bytes = t0_eval.factorized_state_sram_bytes;
					t0_result.factorized_state_hbm_bytes = t0_eval.factorized_state_hbm_bytes;
					t0_result.factorized_state_hbm_read_bytes = t0_eval.factorized_state_hbm_read_bytes;
					t0_result.factorized_state_hbm_write_bytes = t0_eval.factorized_state_hbm_write_bytes;
					t0_result.factorized_state_hbm_bursts = t0_eval.factorized_state_hbm_bursts;
					t0_result.factorized_state_hbm_cycles = t0_eval.factorized_state_hbm_cycles;
					t0_result.factorized_old_y_snapshot_bytes = t0_eval.factorized_old_y_snapshot_bytes;
					t0_result.factorized_old_y_sram_bytes = t0_eval.factorized_old_y_sram_bytes;
					t0_result.factorized_old_y_hbm_write_bytes = t0_eval.factorized_old_y_hbm_write_bytes;
					t0_result.factorized_old_y_hbm_read_bytes = t0_eval.factorized_old_y_hbm_read_bytes;
					t0_result.factorized_old_y_hbm_bursts = t0_eval.factorized_old_y_hbm_bursts;
					t0_result.factorized_old_y_hbm_cycles = t0_eval.factorized_old_y_hbm_cycles;
					t0_result.factorized_pe_dynamic_energy_pj = t0_eval.factorized_pe_dynamic_energy_pj;
					this->multi_ts_results.push_back(t0_result);
					this->write_multi_ts_stats();
					for (int t = 1; t < total_ts; ++t) {
					// Load delta files for this timestep
					std::string delta_a_file, delta_f_file;
					if (!delta_dir.empty()) {
						delta_a_file = delta_dir + "/delta_a_t" + std::to_string(t) + ".txt";
						delta_f_file = delta_dir + "/delta_f_t" + std::to_string(t) + ".txt";
					} else {
						// Fallback to single delta files (repeated application)
						delta_a_file = this->gcp->arg_parse->get_delta_a_path();
						delta_f_file = this->gcp->arg_parse->get_delta_f_path();
					}

					try {
						auto delta_a = DeltaConverter::parse_delta_a_file(delta_a_file);
						auto delta_f = DeltaConverter::parse_delta_f_file(delta_f_file, this->pe_array_d);

						cout << "[Multi-TS] t=" << t << " loading: " << delta_a_file << endl;

						// Apply delta_a to update A_new (A_t) so that subsequent timesteps see the evolved graph.
						for (const auto& u : delta_a) {
							if (u.src < 0 || u.src >= a_new.n) continue;
							if (u.dst < 0 || u.dst >= a_new.n) continue;
							auto& s = adj_sets[static_cast<size_t>(u.src)];
							if (u.is_add) {
								s.insert(u.dst);
							} else {
								s.erase(u.dst);
							}
						}
						rebuild_csr_from_adj_sets();

							// Snapshot the exported/persistent paper H state.  R5 forbids
							// reconstructing old H by privately normalizing old Z.
							std::vector<float> h_before_l1;
							if (num_layers_runtime >= 2) {
								h_before_l1 = this->dygnn_controller->get_h_layer(1);
								if (h_before_l1.empty()) {
									throw std::runtime_error("R5 missing predecessor H1 state");
								}
							}
							std::map<int, std::vector<float>> h_before_layer;
							for (int l = 2; l <= num_layers_runtime; l++) {
								h_before_layer[l] = this->dygnn_controller->get_h_layer(l);
								if (h_before_layer[l].empty()) {
									throw std::runtime_error("R5 missing higher-layer predecessor H state");
								}
								}

							this->dygnn_controller->begin_runtime_timestep_trace();
							DualModeSpMM::StepResult res = this->dygnn_controller->run_once_incremental(a_new, delta_a, delta_f);

						// Generic N-layer cascade
						const int num_higher_layers = std::max(0, num_layers_runtime - 1);
						std::vector<long long> higher_layer_cycles_vec(num_higher_layers, 0);
						// Accumulate higher-layer stats for energy accounting
						PEArraySpMM::Stats hl_pe_accum{};
						DualModeSpMM::StepResult::PerPhaseInfo hl_phase1{}, hl_phase2{}, hl_phase3{};
						long long hl_cim_cyc = 0, hl_cim_stream = 0, hl_cim_exp = 0, hl_combine = 0;
						long long hl_cim_bp = 0, hl_cim_if = 0, hl_cim_al = 0, hl_cim_as = 0;
						long long hl_y_r = 0, hl_y_w = 0, hl_z_rw = 0;
						long long hl_rm_scaled = 0, hl_rm_changed = 0, hl_rm_scale_elems = 0;
						long long hl_rm_residual_elems = 0, hl_rm_ops = 0, hl_rm_delay = 0;
						long long hl_rm_degree_reads = 0, hl_rm_degree_read_bytes = 0;
						long long hl_rm_reciprocals = 0;
						long long hl_fn_pe_cycles = 0, hl_fn_owner_cycles = 0;
						long long hl_fn_state_capacity = 0, hl_fn_state_sram = 0, hl_fn_state_hbm = 0;
						long long hl_fn_state_hbm_read = 0, hl_fn_state_hbm_write = 0;
						long long hl_fn_state_hbm_bursts = 0, hl_fn_state_hbm_cycles = 0;
						long long hl_fn_old_y_bytes = 0, hl_fn_old_y_sram = 0;
						long long hl_fn_old_y_hbm_write = 0, hl_fn_old_y_hbm_read = 0;
						long long hl_fn_old_y_hbm_bursts = 0, hl_fn_old_y_hbm_cycles = 0;
						double hl_fn_pe_energy = 0.0;
						long long hl_z_hbm_rw_bytes = 0;
						// NPF H-state traffic: H_prev is read only for changed rows whose
						// distance bound may exceed tau, H' is written for propagating rows,
						// both in HBM and charged serially.
						DualModeSpMM::NpfStateAccounting npf_total{};
						long long npf_charged_cycles = 0;
						// Distance-bound compute, counted from the code executed per changed
						// row of a gated layer (d = 128, no zero skipping): NormReLU(Z_old) =
						// ReLU 128 compares + sum of squares 128 mul + 128 add + sqrt and
						// reciprocal 2 + scale 128 mul; delta max-reduction = 128 sub + 128
						// max (fabs is a sign-bit clear); bound update and compare 4.  Total
						// 774 raw scalar ops.  Energy follows the PE convention (two raw ops =
						// one FP32 fused MAC): 387 fused MACs, rounded up to 392 so the legacy
						// eight-wide identity stays exact, booked on the row-mean fused-MAC
						// owner.  Cycles: 774 raw ops on the 2,048-op/cycle PE lanes, serial.
						constexpr long long kBoundRawOpsPerRow = 774;
						constexpr long long kBoundFusedMacsPerRow = 392;
						constexpr long long kPeOpsPerCycle = 256LL * 8LL;
						long long npf_bound_ops = 0;
						long long npf_bound_compute_cycles = 0;
						auto commit_npf_layer = [&](int layer,
						                            const std::vector<int>& changed_rows,
						                            const std::vector<int>& propagating_rows,
						                            const std::vector<float>& h_old,
						                            const std::vector<float>& h_new_full,
						                            int dim,
						                            float tau,
						                            std::vector<float>* h_committed,
						                            std::vector<float>* scoped_delta) {
							// Rows below tau keep H_prev; only propagating rows take H'.
							*h_committed = h_old;
							scoped_delta->assign(h_old.size(), 0.0f);
							for (int row : propagating_rows) {
								const size_t base = static_cast<size_t>(row) * static_cast<size_t>(dim);
								for (int j = 0; j < dim; ++j) {
									(*h_committed)[base + j] = h_new_full[base + j];
									(*scoped_delta)[base + j] = h_new_full[base + j] - h_old[base + j];
								}
							}
							if (!this->dygnn_controller->set_h_layer(layer, *h_committed)) {
								throw std::runtime_error("NPF H commit failed closed");
							}
							// Distance bound: b >= ||NormReLU(Z) - H_prev||_inf per row.  It grows
							// by ||NormReLU(Z_new) - NormReLU(Z_old)||_inf (Z_old is already read
							// for the Z update), resets to 0 on propagation and to the exact
							// distance after a checked drop.  A changed row with b <= tau cannot
							// pass the NPF test, so it is dropped without reading H_prev.
							auto& g_last = npf_g_last_[layer];
							auto& bound = npf_bound_[layer];
							const size_t rows = h_old.size() / static_cast<size_t>(dim);
							if (g_last.size() != h_new_full.size()) {
								// First visit: the previous NormReLU(Z) is the bootstrap H.
								g_last = h_old;
								bound.assign(rows, 0.0f);
							}
							std::vector<char> propagates(rows, 0);
							for (int row : propagating_rows) propagates[static_cast<size_t>(row)] = 1;
							std::vector<int> read_rows;
							long long wrong = 0;
							for (int row : changed_rows) {
								const size_t base = static_cast<size_t>(row) * static_cast<size_t>(dim);
								float inc = 0.0f, actual = 0.0f;
								for (int j = 0; j < dim; ++j) {
									inc = std::max(inc, std::fabs(h_new_full[base + j] - g_last[base + j]));
									actual = std::max(actual, std::fabs(h_new_full[base + j] - h_old[base + j]));
									g_last[base + j] = h_new_full[base + j];
								}
								// Round the bound up so FP32 summation never under-estimates it.
								const float b = (bound[row] + inc) * (1.0f + 1e-6f) + 1e-9f;
								if (b <= tau) {
									bound[row] = b;
									if (actual > tau || propagates[static_cast<size_t>(row)]) ++wrong;
								} else {
									read_rows.push_back(row);
									bound[row] = propagates[static_cast<size_t>(row)] ? 0.0f : actual;
								}
							}
							if (wrong != 0) {
								throw std::runtime_error("NPF distance bound dropped a propagating row");
							}
							// Bound metadata (one FP32 per row) is read and written in HBM.
							const auto bacc = this->dygnn_controller->account_npf_bound_state(
								layer, changed_rows);
							npf_total.h_hbm_read_bytes += bacc.h_hbm_read_bytes;
							npf_total.h_hbm_write_bytes += bacc.h_hbm_write_bytes;
							npf_total.h_hbm_cycles += bacc.h_hbm_cycles;
							const long long rows_n = static_cast<long long>(changed_rows.size());
							npf_bound_ops += kBoundFusedMacsPerRow * rows_n;
							npf_bound_compute_cycles +=
								(kBoundRawOpsPerRow * rows_n + kPeOpsPerCycle - 1) / kPeOpsPerCycle;
							const auto acc = this->dygnn_controller->account_npf_h_state(
								layer, read_rows, propagating_rows);
							npf_total.h_read_rows += acc.h_read_rows;
							npf_total.h_write_rows += acc.h_write_rows;
							npf_total.h_hbm_read_bytes += acc.h_hbm_read_bytes;
							npf_total.h_hbm_write_bytes += acc.h_hbm_write_bytes;
							npf_total.h_hbm_cycles += acc.h_hbm_cycles;
							return bacc.h_hbm_cycles + acc.h_hbm_cycles;
						};
							DualModeSpMM::StepResult::ArchAblationCounters hl_arch_accum{};
							DualModeSpMM::StepResult::AccuracyMetrics hl_final_accuracy{};
						// Depth-selectivity ablation: per-layer propagation tracking
						std::vector<int> ablation_plpr_b;
						std::vector<long long> ablation_plmo_b;

						if (num_layers_runtime >= 2 && !res.affected_rows.empty()) {
							const int n_nodes = a_new.n;
							// Isolation mode from CLI
							const std::string iso_mode = this->gcp->arg_parse->get_isolation_mode();
							const double nl_override = this->gcp->arg_parse->get_nl_threshold_override();
							const double prop_tau = this->gcp->arg_parse->get_propagation_tau();
							const bool scope_enabled = this->gcp->arg_parse->get_propagation_scope_enabled();
							const bool use_layer_norm = this->gcp->arg_parse->get_effective_layer_norm_enabled();
							// Unify: if propagation_tau is set and no explicit nl-threshold override,
							// use propagation_tau as the nonlinear isolation threshold
							const float nl_threshold = (!scope_enabled || iso_mode == "none") ? std::numeric_limits<float>::max()
							                         : (iso_mode == "weak") ? 0.0f
							                         : (nl_override >= 0.0) ? static_cast<float>(nl_override)
							                         : (prop_tau > 0.0) ? static_cast<float>(prop_tau)
							                         : 1e-6f;

							// Record L1 affected rows
							ablation_plpr_b.push_back(static_cast<int>(res.affected_rows.size()));
							ablation_plmo_b.push_back(res.pe_stats.total_mac_ops);

#if DESPOINA_V3_FUNCTIONAL_DCIM
							const auto& y_after_l1 = this->dygnn_controller->get_z_current();
#else
							const auto& y_after_l1 = this->dygnn_controller->get_y_current();
#endif
								const int d_l1 = static_cast<int>(y_after_l1.size()) / std::max(1, n_nodes);

								std::vector<float> h_prev_new;
								if (!DualModeSpMM::build_norm_relu(
								        y_after_l1, n_nodes, use_layer_norm, &h_prev_new) ||
								    h_before_l1.size() != h_prev_new.size()) {
									throw std::runtime_error("R5 L1 NormReLU continuity failed closed");
								}

							std::vector<float> delta_h_prev(y_after_l1.size(), 0.0f);
							for (size_t i = 0; i < y_after_l1.size(); i++)
									delta_h_prev[i] = h_prev_new[i] - h_before_l1[i];

							std::vector<int> prev_propagating;
							if (!scope_enabled || iso_mode == "none") {
								prev_propagating = res.affected_rows;
							} else {
								for (int row : res.affected_rows) {
									if (row < 0 || row >= n_nodes) continue;
									float max_abs = 0.0f;
									const size_t base = static_cast<size_t>(row) * d_l1;
									for (int j = 0; j < d_l1; j++) {
										const float v = std::fabs(delta_h_prev[base + j]);
										if (v > max_abs) max_abs = v;
									}
										if (max_abs > nl_threshold) prev_propagating.push_back(row);
									}
								}
								{
									const std::vector<float> h_new_full = h_prev_new;
									npf_charged_cycles += commit_npf_layer(1, res.affected_rows, prev_propagating,
									                 h_before_l1, h_new_full, d_l1, nl_threshold,
									                 &h_prev_new, &delta_h_prev);
								}
								this->dygnn_controller->set_runtime_activation_trace(
									1, h_prev_new, delta_h_prev, prev_propagating);

							// No inter-layer PE overlap: L(l+1) depends on L(l) CiM output
#if !DESPOINA_V3_FUNCTIONAL_DCIM
							long long prev_overlap_budget = 0;
#endif
							// Preserve structural P1 across a tau-empty predecessor.  The
							// lower-level routine then skips only scoped P2/P3.
							const bool has_structural_delta_a_work =
								DeltaConverter::delta_a_to_csr(delta_a, n_nodes).nnz > 0;

						for (int l = 2; l <= num_layers_runtime; l++) {
								if (prev_propagating.empty() && !has_structural_delta_a_work) break;

								// Record propagating rows entering this layer
								ablation_plpr_b.push_back(static_cast<int>(prev_propagating.size()));

								cout << "[Layer-" << l << "] Nonlinear isolation: "
								     << (l == 2 ? (int)res.affected_rows.size() : (int)prev_propagating.size())
								     << " → " << prev_propagating.size() << " propagating" << endl;

									auto l_res = this->dygnn_controller->run_higher_layer_spmm(
										l, a_new, delta_a, h_prev_new, delta_h_prev, prev_propagating);
									hl_final_accuracy = l_res.accuracy;
									const long long overlap =
#if DESPOINA_V3_FUNCTIONAL_DCIM
										0;
#else
										std::min(prev_overlap_budget, l_res.total_cycles);
#endif
								higher_layer_cycles_vec[l - 2] = std::max(0LL, l_res.total_cycles - overlap);

								// Record per-layer MAC ops
								ablation_plmo_b.push_back(l_res.pe_stats.total_mac_ops);

								cout << "[Multi-TS] t=" << t << " Layer-" << l << " SpMM: "
								     << l_res.total_cycles << " cycles (overlap=" << overlap
								     << ", net=" << higher_layer_cycles_vec[l - 2] << ")" << endl;

								// Accumulate higher-layer counters
								hl_pe_accum.total_mac_ops += l_res.pe_stats.total_mac_ops;
								hl_pe_accum.f_cache_hits += l_res.pe_stats.f_cache_hits;
								hl_pe_accum.f_cache_misses += l_res.pe_stats.f_cache_misses;
								hl_pe_accum.l1_f_cache_hits += l_res.pe_stats.l1_f_cache_hits;
								hl_pe_accum.l2_shared_hits += l_res.pe_stats.l2_shared_hits;
								hl_pe_accum.l1_bank_conflict_stalls += l_res.pe_stats.l1_bank_conflict_stalls;
								hl_pe_accum.l2_bank_conflict_stalls += l_res.pe_stats.l2_bank_conflict_stalls;
								hl_pe_accum.dram_fills += l_res.pe_stats.dram_fills;
								hl_pe_accum.dram_requests += l_res.pe_stats.dram_requests;
								hl_pe_accum.a_load_dram_requests += l_res.pe_stats.a_load_dram_requests;
								hl_pe_accum.a_load_dram_latency_sum += l_res.pe_stats.a_load_dram_latency_sum;
								hl_pe_accum.dram_row_buffer_hits += l_res.pe_stats.dram_row_buffer_hits;
								hl_pe_accum.noc_steal_count += l_res.pe_stats.noc_steal_count;
								hl_pe_accum.noc_steal_latency_total_cycles += l_res.pe_stats.noc_steal_latency_total_cycles;
								hl_pe_accum.output_egress_bytes += l_res.pe_stats.output_egress_bytes;
								hl_pe_accum.output_egress_rows += l_res.pe_stats.output_egress_rows;
								hl_phase1.nnz += l_res.phase1.nnz;
								hl_phase1.mac_ops += l_res.phase1.mac_ops;
								hl_phase1.cycles += l_res.phase1.cycles;
								hl_phase2.nnz += l_res.phase2.nnz;
								hl_phase2.mac_ops += l_res.phase2.mac_ops;
								hl_phase2.cycles += l_res.phase2.cycles;
								hl_phase3.nnz += l_res.phase3.nnz;
								hl_phase3.mac_ops += l_res.phase3.mac_ops;
								hl_phase3.cycles += l_res.phase3.cycles;
								hl_rm_scaled += l_res.row_mean_scaled_rows;
								hl_rm_changed += l_res.row_mean_changed_degree_rows;
								hl_rm_scale_elems += l_res.row_mean_scale_elements;
								hl_rm_residual_elems += l_res.row_mean_residual_correction_elements;
								hl_rm_ops += l_res.row_mean_scalar_ops;
								hl_rm_delay += l_res.row_mean_stream_delay_cycles;
								hl_rm_degree_reads += l_res.row_mean_degree_read_events;
								hl_rm_degree_read_bytes += l_res.row_mean_degree_read_bytes;
								hl_rm_reciprocals += l_res.row_mean_reciprocal_events;
								hl_fn_pe_cycles += l_res.factorized_native_pe_cycles;
								hl_fn_owner_cycles += l_res.factorized_native_owner_cycles;
								hl_fn_state_capacity += l_res.factorized_state_capacity_bytes;
								hl_fn_state_sram += l_res.factorized_state_sram_bytes;
								hl_fn_state_hbm += l_res.factorized_state_hbm_bytes;
								hl_fn_state_hbm_read += l_res.factorized_state_hbm_read_bytes;
								hl_fn_state_hbm_write += l_res.factorized_state_hbm_write_bytes;
								hl_fn_state_hbm_bursts += l_res.factorized_state_hbm_bursts;
								hl_fn_state_hbm_cycles += l_res.factorized_state_hbm_cycles;
								hl_fn_old_y_bytes += l_res.factorized_old_y_snapshot_bytes;
								hl_fn_old_y_sram += l_res.factorized_old_y_sram_bytes;
								hl_fn_old_y_hbm_write += l_res.factorized_old_y_hbm_write_bytes;
								hl_fn_old_y_hbm_read += l_res.factorized_old_y_hbm_read_bytes;
								hl_fn_old_y_hbm_bursts += l_res.factorized_old_y_hbm_bursts;
								hl_fn_old_y_hbm_cycles += l_res.factorized_old_y_hbm_cycles;
								hl_fn_pe_energy += l_res.factorized_pe_dynamic_energy_pj;
								hl_cim_cyc += l_res.cim_cycles;
								hl_cim_stream += l_res.cim_stream_cycles;
								hl_cim_exp += l_res.cim_exposed_cycles;
								hl_combine += l_res.combine_cycles;
								hl_cim_bp += l_res.cim_pe_backpressure_cycles;
								hl_cim_if += l_res.cim_input_fifo_full_cycles;
								hl_cim_al += l_res.cim_active_lane_cycles;
								hl_cim_as += l_res.cim_active_slot_cycles;
								hl_y_r += l_res.y_dram_read_cycles;
								hl_y_w += l_res.y_dram_write_cycles;
								hl_z_rw += l_res.z_dram_rw_cycles;
								hl_z_hbm_rw_bytes += l_res.z_hbm_rw_bytes;
								hl_arch_accum.add(l_res.arch_ctrs);

								if (l < num_layers_runtime && !l_res.affected_rows.empty()) {
#if DESPOINA_V3_FUNCTIONAL_DCIM
									const auto& yl_after = this->dygnn_controller->get_z_layer(l);
#else
									const auto& yl_after = this->dygnn_controller->get_y_layer(l);
#endif
									const auto& h_old_tmp = h_before_layer[l];
									const int dl = static_cast<int>(yl_after.size()) / std::max(1, n_nodes);

									if (!DualModeSpMM::build_norm_relu(
									        yl_after, n_nodes, use_layer_norm, &h_prev_new) ||
									    h_old_tmp.size() != h_prev_new.size()) {
										throw std::runtime_error("R5 higher-layer NormReLU continuity failed closed");
									}

									// Delta is formed solely from the exported predecessor H.
									delta_h_prev.assign(yl_after.size(), 0.0f);
									for (size_t i = 0; i < yl_after.size(); i++)
										delta_h_prev[i] = h_prev_new[i] - h_old_tmp[i];

									prev_propagating.clear();
									if (!scope_enabled || iso_mode == "none") {
										for (int row : l_res.affected_rows) {
											if (row >= 0 && row < n_nodes) prev_propagating.push_back(row);
										}
									} else {
										for (int row : l_res.affected_rows) {
											if (row < 0 || row >= n_nodes) continue;
											float max_abs = 0.0f;
											const size_t base = static_cast<size_t>(row) * dl;
											for (int j = 0; j < dl; j++) {
												const float v = std::fabs(delta_h_prev[base + j]);
												if (v > max_abs) max_abs = v;
											}
										if (max_abs > nl_threshold) prev_propagating.push_back(row);
									}
								}
								{
									const std::vector<float> h_new_full = h_prev_new;
									npf_charged_cycles += commit_npf_layer(l, l_res.affected_rows, prev_propagating,
									                 h_old_tmp, h_new_full, dl, nl_threshold,
									                 &h_prev_new, &delta_h_prev);
								}
								this->dygnn_controller->set_runtime_activation_trace(
									l, h_prev_new, delta_h_prev, prev_propagating);

#if !DESPOINA_V3_FUNCTIONAL_DCIM
									prev_overlap_budget = l_res.combine_cycles;
#endif
									} else if (l == num_layers_runtime && !l_res.affected_rows.empty()) {
#if DESPOINA_V3_FUNCTIONAL_DCIM
										const auto& terminal_z = this->dygnn_controller->get_z_layer(l);
#else
										const auto& terminal_z = this->dygnn_controller->get_y_layer(l);
#endif
										const auto& terminal_old_h = h_before_layer[l];
										std::vector<float> terminal_h;
										if (!DualModeSpMM::build_norm_relu(
										        terminal_z, n_nodes, use_layer_norm, &terminal_h) ||
										    terminal_old_h.size() != terminal_h.size()) {
											throw std::runtime_error("R5 terminal NormReLU continuity failed closed");
										}
										std::vector<float> terminal_delta(terminal_h.size(), 0.0f);
										for (size_t i = 0; i < terminal_h.size(); ++i) {
											terminal_delta[i] = terminal_h[i] - terminal_old_h[i];
										}
										const int terminal_d = static_cast<int>(terminal_h.size()) /
										    std::max(1, n_nodes);
										std::vector<int> terminal_tau_rows;
										for (int row : l_res.affected_rows) {
											if (row < 0 || row >= n_nodes) continue;
											float max_abs = 0.0f;
											const size_t base = static_cast<size_t>(row) * terminal_d;
											for (int j = 0; j < terminal_d; ++j) {
												max_abs = std::max(max_abs, std::fabs(terminal_delta[base + j]));
											}
											if (!scope_enabled || iso_mode == "none" || max_abs > nl_threshold) {
												terminal_tau_rows.push_back(row);
											}
										}
										if (!this->dygnn_controller->set_h_layer(l, terminal_h)) {
											throw std::runtime_error("R5 terminal H commit failed closed");
										}
										{
											// The output layer is not tau-gated: every changed row writes
											// its new embedding.  Its dH only feeds the activation trace,
											// so no H_prev read is charged.
											const auto acc = this->dygnn_controller->account_npf_h_state(
												l, std::vector<int>{}, l_res.affected_rows);
											npf_total.h_read_rows += acc.h_read_rows;
											npf_total.h_write_rows += acc.h_write_rows;
											npf_total.h_hbm_read_bytes += acc.h_hbm_read_bytes;
											npf_total.h_hbm_write_bytes += acc.h_hbm_write_bytes;
											npf_total.h_hbm_cycles += acc.h_hbm_cycles;
											npf_charged_cycles += acc.h_hbm_cycles;
										}
										this->dygnn_controller->set_runtime_activation_trace(
										    l, terminal_h, terminal_delta, terminal_tau_rows);
										break;
									} else {
									break;
								}
							}
						}

						// Store result
						TimestepResult ts_result;
						ts_result.timestep = t;
						ts_result.mode_used = res.mode_used;
						ts_result.phase_mode_used = res.phase_mode_used;
						ts_result.higher_layer_cycles = higher_layer_cycles_vec;
							ts_result.per_layer_propagating_rows = ablation_plpr_b;
							ts_result.per_layer_mac_ops = ablation_plmo_b;
							ts_result.e2e_accuracy = hl_final_accuracy;
							ts_result.e2e_exported = hl_final_accuracy.executed;
						long long sum_higher = 0;
						for (auto c : higher_layer_cycles_vec) sum_higher += c;
						// NPF H_prev/H' HBM traffic is charged serially (conservative).
						ts_result.cycles = res.total_cycles + sum_higher + npf_charged_cycles +
							npf_bound_compute_cycles;
						ts_result.npf_h_read_rows = npf_total.h_read_rows;
						ts_result.npf_h_write_rows = npf_total.h_write_rows;
						ts_result.npf_h_hbm_read_bytes = npf_total.h_hbm_read_bytes;
						ts_result.npf_h_hbm_write_bytes = npf_total.h_hbm_write_bytes;
						ts_result.npf_h_hbm_cycles = npf_total.h_hbm_cycles;
						ts_result.z_hbm_rw_bytes = res.z_hbm_rw_bytes + hl_z_hbm_rw_bytes;
						// Merge L1 pe_stats with accumulated higher-layer stats
						ts_result.pe_stats = res.pe_stats;
						ts_result.pe_stats.total_mac_ops += hl_pe_accum.total_mac_ops;
						ts_result.pe_stats.f_cache_hits += hl_pe_accum.f_cache_hits;
						ts_result.pe_stats.f_cache_misses += hl_pe_accum.f_cache_misses;
						ts_result.pe_stats.l1_f_cache_hits += hl_pe_accum.l1_f_cache_hits;
						ts_result.pe_stats.l2_shared_hits += hl_pe_accum.l2_shared_hits;
						ts_result.pe_stats.l1_bank_conflict_stalls += hl_pe_accum.l1_bank_conflict_stalls;
						ts_result.pe_stats.l2_bank_conflict_stalls += hl_pe_accum.l2_bank_conflict_stalls;
						ts_result.pe_stats.dram_fills += hl_pe_accum.dram_fills;
						ts_result.pe_stats.dram_requests += hl_pe_accum.dram_requests;
						ts_result.pe_stats.a_load_dram_requests += hl_pe_accum.a_load_dram_requests;
						ts_result.pe_stats.a_load_dram_latency_sum += hl_pe_accum.a_load_dram_latency_sum;
						ts_result.pe_stats.dram_row_buffer_hits += hl_pe_accum.dram_row_buffer_hits;
						ts_result.pe_stats.noc_steal_count += hl_pe_accum.noc_steal_count;
						ts_result.pe_stats.noc_steal_latency_total_cycles += hl_pe_accum.noc_steal_latency_total_cycles;
						ts_result.pe_stats.output_egress_bytes += hl_pe_accum.output_egress_bytes;
						ts_result.pe_stats.output_egress_rows += hl_pe_accum.output_egress_rows;
						ts_result.phase1 = res.phase1;
						ts_result.phase2 = res.phase2;
						ts_result.phase3 = res.phase3;
						ts_result.phase1.nnz += hl_phase1.nnz;
						ts_result.phase1.mac_ops += hl_phase1.mac_ops;
						ts_result.phase1.cycles += hl_phase1.cycles;
						ts_result.phase2.nnz += hl_phase2.nnz;
						ts_result.phase2.mac_ops += hl_phase2.mac_ops;
						ts_result.phase2.cycles += hl_phase2.cycles;
						ts_result.phase3.nnz += hl_phase3.nnz;
						ts_result.phase3.mac_ops += hl_phase3.mac_ops;
						ts_result.phase3.cycles += hl_phase3.cycles;
						ts_result.row_mean_enabled = res.row_mean_enabled;
						ts_result.row_mean_scaled_rows = res.row_mean_scaled_rows + hl_rm_scaled;
						ts_result.row_mean_graph_changed_degree_rows = res.row_mean_graph_changed_degree_rows;
						ts_result.row_mean_changed_degree_rows = res.row_mean_changed_degree_rows + hl_rm_changed;
						ts_result.row_mean_scale_elements = res.row_mean_scale_elements + hl_rm_scale_elems;
						ts_result.row_mean_residual_correction_elements = res.row_mean_residual_correction_elements + hl_rm_residual_elems;
						ts_result.row_mean_scalar_ops = res.row_mean_scalar_ops + hl_rm_ops + npf_bound_ops;
						ts_result.row_mean_stream_delay_cycles = res.row_mean_stream_delay_cycles + hl_rm_delay;
						ts_result.row_mean_degree_read_events = res.row_mean_degree_read_events + hl_rm_degree_reads;
						ts_result.row_mean_degree_write_events = res.row_mean_degree_write_events;
						ts_result.row_mean_degree_read_bytes = res.row_mean_degree_read_bytes + hl_rm_degree_read_bytes;
						ts_result.row_mean_degree_write_bytes = res.row_mean_degree_write_bytes;
						ts_result.row_mean_reciprocal_events = res.row_mean_reciprocal_events + hl_rm_reciprocals;
						ts_result.factorized_native_pe_cycles = res.factorized_native_pe_cycles + hl_fn_pe_cycles +
							npf_bound_compute_cycles;
						ts_result.factorized_native_owner_cycles = res.factorized_native_owner_cycles + hl_fn_owner_cycles;
						ts_result.factorized_state_capacity_bytes = res.factorized_state_capacity_bytes + hl_fn_state_capacity;
						ts_result.factorized_state_sram_bytes = res.factorized_state_sram_bytes + hl_fn_state_sram;
						ts_result.factorized_state_hbm_bytes = res.factorized_state_hbm_bytes + hl_fn_state_hbm;
						ts_result.factorized_state_hbm_read_bytes = res.factorized_state_hbm_read_bytes + hl_fn_state_hbm_read;
						ts_result.factorized_state_hbm_write_bytes = res.factorized_state_hbm_write_bytes + hl_fn_state_hbm_write;
						ts_result.factorized_state_hbm_bursts = res.factorized_state_hbm_bursts + hl_fn_state_hbm_bursts;
						ts_result.factorized_state_hbm_cycles = res.factorized_state_hbm_cycles + hl_fn_state_hbm_cycles;
						ts_result.factorized_old_y_snapshot_bytes = res.factorized_old_y_snapshot_bytes + hl_fn_old_y_bytes;
						ts_result.factorized_old_y_sram_bytes = res.factorized_old_y_sram_bytes + hl_fn_old_y_sram;
						ts_result.factorized_old_y_hbm_write_bytes = res.factorized_old_y_hbm_write_bytes + hl_fn_old_y_hbm_write;
						ts_result.factorized_old_y_hbm_read_bytes = res.factorized_old_y_hbm_read_bytes + hl_fn_old_y_hbm_read;
						ts_result.factorized_old_y_hbm_bursts = res.factorized_old_y_hbm_bursts + hl_fn_old_y_hbm_bursts;
						ts_result.factorized_old_y_hbm_cycles = res.factorized_old_y_hbm_cycles + hl_fn_old_y_hbm_cycles;
						ts_result.factorized_pe_dynamic_energy_pj = res.factorized_pe_dynamic_energy_pj + hl_fn_pe_energy +
							static_cast<double>(npf_bound_ops / 8) * energy::pe_array::SIMD_MAC_ENERGY_PJ;
						ts_result.indexer_ctrs = res.indexer_ctrs;
						ts_result.cim_mode = res.cim_mode;
						ts_result.combine_cycles = res.combine_cycles + hl_combine;
						ts_result.cim_cycles = res.cim_cycles + hl_cim_cyc;
						ts_result.cim_stream_cycles = res.cim_stream_cycles + hl_cim_stream;
						ts_result.cim_exposed_cycles = res.cim_exposed_cycles + hl_cim_exp;
						ts_result.cim_pe_backpressure_cycles = res.cim_pe_backpressure_cycles + hl_cim_bp;
						ts_result.cim_input_fifo_full_rejections = res.cim_input_fifo_full_rejections;
						ts_result.cim_input_fifo_full_cycles = res.cim_input_fifo_full_cycles + hl_cim_if;
						ts_result.cim_output_fifo_full_cycles = res.cim_output_fifo_full_cycles;
						ts_result.cim_output_fifo_full_stall_cycles = res.cim_output_fifo_full_stall_cycles;
						ts_result.cim_stage3_output_block_cycles = res.cim_stage3_output_block_cycles;
						ts_result.cim_inject_no_empty_lane_cycles = res.cim_inject_no_empty_lane_cycles;
						ts_result.cim_active_lane_cycles = res.cim_active_lane_cycles + hl_cim_al;
						ts_result.cim_active_slot_cycles = res.cim_active_slot_cycles + hl_cim_as;
						ts_result.cim_input_fifo_occupancy_cycles = res.cim_input_fifo_occupancy_cycles;
						ts_result.cim_output_fifo_occupancy_cycles = res.cim_output_fifo_occupancy_cycles;
						ts_result.cim_max_active_lanes = res.cim_max_active_lanes;
						ts_result.cim_rows = res.cim_rows;
						ts_result.y_dram_read_cycles = res.y_dram_read_cycles + hl_y_r;
						ts_result.y_dram_write_cycles = res.y_dram_write_cycles + hl_y_w;
						ts_result.z_dram_rw_cycles = res.z_dram_rw_cycles + hl_z_rw;
						ts_result.z_sram_rw_bytes = res.z_sram_rw_bytes;
						ts_result.row_detect_cycles = res.row_detect_cycles;
						ts_result.row_detect_checks = res.row_detect_checks;
						ts_result.row_detect_sram_reads = res.row_detect_sram_reads;
						ts_result.arch_ctrs = res.arch_ctrs;
						ts_result.arch_ctrs.add(hl_arch_accum);
#if DESPOINA_V3_FUNCTIONAL_DCIM
						ts_result.finalized_rows = static_cast<long long>(ts_result.arch_ctrs.digital_stream_rows);
						ts_result.finalized_packets = static_cast<long long>(ts_result.arch_ctrs.digital_stream_packets);
						ts_result.dcim_scalar_macs = static_cast<long long>(ts_result.arch_ctrs.digital_stream_mac_ops);
						ts_result.dcim_compute_cycles = static_cast<long long>(ts_result.arch_ctrs.digital_stream_compute_cycles);
						ts_result.tile_busy_work_cycles = static_cast<long long>(ts_result.arch_ctrs.tile_busy_work_cycles);
						ts_result.causal_stream_elapsed_cycles = static_cast<long long>(ts_result.arch_ctrs.causal_stream_elapsed_cycles);
						ts_result.exposed_tail_cycles = static_cast<long long>(ts_result.arch_ctrs.exposed_tail_cycles);
						ts_result.dcim_streams = static_cast<long long>(ts_result.arch_ctrs.digital_stream_count);
						ts_result.dcim_drain_sensitivity_cycles = static_cast<long long>(ts_result.arch_ctrs.digital_stream_drain_sensitivity_cycles);
						ts_result.dcim_compute_only_exposed_cycles = static_cast<long long>(ts_result.arch_ctrs.dcim_exposed_cycles);
						ts_result.dcim_plus6_exposed_cycles = static_cast<long long>(ts_result.arch_ctrs.digital_stream_plus6_exposed_cycles);
						ts_result.finalized_duplicate_packets = static_cast<long long>(ts_result.arch_ctrs.digital_stream_duplicate_packets);
						ts_result.finalized_invalid_packets = static_cast<long long>(ts_result.arch_ctrs.digital_stream_invalid_packets);
						ts_result.finalized_timeouts = static_cast<long long>(ts_result.arch_ctrs.digital_stream_timeouts);
						ts_result.dcim_activation_quantized_elements = static_cast<long long>(ts_result.arch_ctrs.digital_stream_activation_quantized_elements);
						ts_result.dcim_dequantized_elements = static_cast<long long>(ts_result.arch_ctrs.digital_stream_dequantized_elements);
						ts_result.dcim_comparator_tree_passes = static_cast<long long>(ts_result.arch_ctrs.digital_stream_comparator_tree_passes);
						ts_result.cim_rows = static_cast<int>(ts_result.finalized_rows);
#endif
							ts_result.accuracy = res.accuracy;
							ts_result.y_accuracy = res.y_accuracy;
							maybe_export_timestep_e2e(t, ts_result);
						this->capture_exact_hbm_snapshot(ts_result);
						this->multi_ts_results.push_back(ts_result);
						this->write_multi_ts_stats();

						const long long total_step_cycles = res.total_cycles + sum_higher;

						cout << "[Multi-TS] t=" << t << " done: " << total_step_cycles << " cycles ("
						     << ((res.mode_used == DualModeSpMM::Mode::INCREMENTAL) ? "incremental" : "full");
						for (int l = 0; l < (int)higher_layer_cycles_vec.size(); l++) {
							if (higher_layer_cycles_vec[l] > 0)
								cout << ", L" << (l + 2) << "=" << higher_layer_cycles_vec[l];
						}
							cout << ")" << endl;

						} catch (const RuntimeExportError&) {
							throw;
						} catch (const std::exception& e) {
							if (this->gcp != nullptr &&
							    this->gcp->arg_parse != nullptr &&
							    this->gcp->arg_parse->is_official_profile()) {
								throw;
							}
							cerr << "[Multi-TS] WARNING t=" << t << ": " << e.what() << endl;
						// Continue with next timestep
					}
				}

				// =============================================================
				// Multi-layer GCN (N-layer end-to-end) after last timestep
				// =============================================================
#if !DESPOINA_V3_FUNCTIONAL_DCIM
				const int num_layers_done = this->gcp->arg_parse->get_num_layers();
				const bool final_e2e_has_cim =
					this->dygnn_controller && this->dygnn_controller->get_cim_mode() != CiMMode::NONE;
				if (num_layers_done >= 2 && final_e2e_has_cim) {
					cout << "\n[Multi-Layer] Running " << num_layers_done << "-layer GCN end-to-end..." << endl;
					const bool do_export_z = this->gcp->arg_parse->get_export_z();
					const std::string export_dir = this->gcp->get_results_dir();
					const int final_num_layers = this->gcp->arg_parse->get_num_layers();
					DualModeSpMM::MultiLayerResult ml_result =
						this->dygnn_controller->run_multi_layer_gcn(final_num_layers, a_new, do_export_z, export_dir);
					this->multi_ts_runtime_cim_energy_pj = ml_result.runtime_cim_energy_pj;
					this->multi_ts_eval_cim_energy_pj = ml_result.eval_cim_energy_pj;
					this->multi_ts_eval_energy_valid = true;

					// Write multi-layer results JSON
						{
							const std::string ml_path = this->gcp->get_results_dir() + "/multi_layer_results.json";
							std::ofstream mofs(ml_path);
							if (mofs.is_open()) {
								const bool dcim_on = this->gcp->arg_parse->get_dcim_accuracy();
								const bool dcim_dig = this->gcp->arg_parse->get_dcim_digital();
								const std::string compute_mode_str =
								    dcim_dig ? "DCIM_DIGITAL" :
								    dcim_on  ? "ACIM_FAITHFUL" : "IDEAL_FP32";
								const bool has_dcim_cfg = (this->cim_unit != nullptr);
								const int dcim_bpc = has_dcim_cfg ? this->cim_unit->get_bits_per_cycle() : 0;
								const int dcim_adc_bits = has_dcim_cfg ? this->cim_unit->get_adc_bits() : 0;
									const double dcim_sigma_mv = has_dcim_cfg ? this->cim_unit->get_resolved_thermal_sigma_mv() : 0.0;
									const std::string dcim_thermal_model = has_dcim_cfg ? this->cim_unit->get_thermal_model() : "legacy";
									const double dcim_thermal_bl_cap_ff = has_dcim_cfg ? this->cim_unit->get_thermal_bl_cap_ff() : 0.0;
									const double dcim_thermal_adc_comp_sigma_mv =
									    has_dcim_cfg ? this->cim_unit->get_thermal_adc_comp_sigma_mv() : 0.0;
									const std::string dcim_act_scale_mode = has_dcim_cfg ? this->cim_unit->get_activation_scale_mode_str() : "fixed";
								const int dcim_act_calib_rows = has_dcim_cfg ? this->cim_unit->get_act_calib_rows() : 0;
								const bool dcim_pv_on = has_dcim_cfg ? this->cim_unit->get_enable_pv() : false;
								const bool dcim_thermal_on = has_dcim_cfg ? this->cim_unit->get_enable_thermal() : false;
								const bool dcim_fp_act = has_dcim_cfg ? this->cim_unit->get_fp_activations() : false;
								mofs << "{\n";
								mofs << "  \"compute_mode\": \"" << compute_mode_str << "\",\n";
								mofs << "  \"dcim_mode\": " << (dcim_on ? "true" : "false") << ",\n";
								mofs << "  \"dcim_digital\": " << (dcim_dig ? "true" : "false") << ",\n";
								mofs << "  \"variation_seed\": " << this->gcp->arg_parse->get_variation_seed() << ",\n";
							mofs << "  \"num_layers\": " << ml_result.num_layers << ",\n";
							mofs << "  \"total_cycles\": " << ml_result.total_cycles << ",\n";
							mofs << "  \"total_energy_pj\": " << ml_result.total_energy_pj << ",\n";
							mofs << "  \"runtime_cim_energy_pj\": " << ml_result.runtime_cim_energy_pj << ",\n";
							mofs << "  \"eval_cim_energy_pj\": " << ml_result.eval_cim_energy_pj << ",\n";
							mofs << "  \"cim_total_energy_pj\": "
							     << (ml_result.runtime_cim_energy_pj + ml_result.eval_cim_energy_pj) << ",\n";
							mofs << "  \"layers\": [\n";
							for (size_t li = 0; li < ml_result.layers.size(); li++) {
								const auto& lr = ml_result.layers[li];
								mofs << "    {\n";
								mofs << "      \"layer_id\": " << lr.layer_id << ",\n";
								mofs << "      \"spmm_cycles\": " << lr.spmm_cycles << ",\n";
									mofs << "      \"cim_cycles\": " << lr.cim_cycles << ",\n";
									mofs << "      \"activation_cycles\": " << lr.activation_cycles << ",\n";
									mofs << "      \"cim_energy_pj\": " << lr.cim_energy_pj << ",\n";
									mofs << "      \"noise_config\": {\n";
										mofs << "        \"pv_on\": " << (dcim_pv_on ? "true" : "false") << ",\n";
										mofs << "        \"thermal_on\": " << (dcim_thermal_on ? "true" : "false") << ",\n";
										mofs << "        \"resolved_thermal_sigma_mv\": " << dcim_sigma_mv << ",\n";
										mofs << "        \"thermal_model\": \"" << dcim_thermal_model << "\",\n";
										mofs << "        \"thermal_bl_cap_ff\": " << dcim_thermal_bl_cap_ff << ",\n";
										mofs << "        \"thermal_adc_comp_sigma_mv\": " << dcim_thermal_adc_comp_sigma_mv << "\n";
										mofs << "      },\n";
										mofs << "      \"quant_config\": {\n";
										mofs << "        \"bits_per_cycle\": " << dcim_bpc << ",\n";
										mofs << "        \"adc_bits\": " << dcim_adc_bits << ",\n";
									mofs << "        \"fp_activations\": " << (dcim_fp_act ? "true" : "false") << ",\n";
									mofs << "        \"act_scale_mode\": \"" << dcim_act_scale_mode << "\",\n";
									mofs << "        \"act_calib_rows\": " << dcim_act_calib_rows << "\n";
									mofs << "      },\n";
									mofs << "      \"accuracy\": {\n";
								mofs << "        \"rmse\": " << lr.accuracy.rmse << ",\n";
								mofs << "        \"cosine_similarity\": " << lr.accuracy.cosine_similarity << ",\n";
								mofs << "        \"max_abs_error\": " << lr.accuracy.max_abs_error << ",\n";
								mofs << "        \"mean_abs_error\": " << lr.accuracy.mean_abs_error << ",\n";
								mofs << "        \"rel_error_mean\": " << lr.accuracy.rel_error_mean << ",\n";
								mofs << "        \"hit_rate_rel_1pct\": " << lr.accuracy.hit_rate_rel_1pct << ",\n";
								mofs << "        \"hit_rate_rel_5pct\": " << lr.accuracy.hit_rate_rel_5pct << "\n";
								mofs << "      }\n";
								mofs << "    }" << (li + 1 < ml_result.layers.size() ? "," : "") << "\n";
							}
							mofs << "  ],\n";
							mofs << "  \"end_to_end_accuracy\": {\n";
							mofs << "    \"rmse\": " << ml_result.final_accuracy.rmse << ",\n";
							mofs << "    \"cosine_similarity\": " << ml_result.final_accuracy.cosine_similarity << ",\n";
							mofs << "    \"max_abs_error\": " << ml_result.final_accuracy.max_abs_error << ",\n";
							mofs << "    \"mean_abs_error\": " << ml_result.final_accuracy.mean_abs_error << ",\n";
							mofs << "    \"rel_error_mean\": " << ml_result.final_accuracy.rel_error_mean << ",\n";
							mofs << "    \"hit_rate_rel_1pct\": " << ml_result.final_accuracy.hit_rate_rel_1pct << ",\n";
							mofs << "    \"hit_rate_rel_5pct\": " << ml_result.final_accuracy.hit_rate_rel_5pct << "\n";
							mofs << "  }";
							if (do_export_z) {
								mofs << ",\n  \"exports\": {\n";
								mofs << "    \"z2_hw\": \"z2_hw.bin\",\n";
								mofs << "    \"z2_golden\": \"z2_golden.bin\",\n";
								mofs << "    \"h1_hw\": \"h1_hw.bin\"\n";
								mofs << "  }";
							}
							mofs << "\n}\n";
							cout << "[Multi-Layer] Results written to " << ml_path << endl;
						}
							cout << "[Multi-Layer] Total cycles: " << ml_result.total_cycles << endl;
						cout << "[Multi-Layer] E2E RMSE: " << ml_result.final_accuracy.rmse
						     << ", Cosine: " << ml_result.final_accuracy.cosine_similarity << endl;
					}
					if (num_layers_done >= 2 && !final_e2e_has_cim) {
						cout << "\n[Multi-Layer] Skipping final end-to-end CiM report because cim-mode=none." << endl;
					}
#else
				// v3 validation consumes the exported resident runtime Z states.
				// A post-run run_multi_layer_gcn() would be a duplicate dense pass,
				// reset direct component counters, and is therefore forbidden.
				this->multi_ts_eval_energy_valid = false;
#endif

					// Output combined stats
					this->write_multi_ts_stats();
				this->multi_ts_done = true;
				return true;
			}
		}

#ifdef ENABLE_CIM
		{
			auto dump_pe_array_stats_json = [&]() {
				const std::string out_path = this->gcp->get_results_dir() + "/pe_array_stats.json";
				std::ofstream ofs(out_path);
				if (!ofs.is_open()) {
					return;
				}
				PEArraySpMM::Stats st = this->pe_array_spmm->get_stats();
				std::string mode = (this->gcp && this->gcp->arg_parse) ? this->gcp->arg_parse->get_spmm_mode() : "full";
				std::string cim_mode = (this->gcp && this->gcp->arg_parse) ? this->gcp->arg_parse->get_cim_mode() : "none";
				const std::string arch_variant =
					(this->dygnn_controller != nullptr)
						? this->dygnn_controller->get_arch_variant_string()
						: ((this->gcp && this->gcp->arg_parse)
						       ? this->gcp->arg_parse->get_arch_variant()
						       : default_arch_variant_string());
				const std::string dataset =
					(this->gcp && this->gcp->arg_parse) ? this->gcp->arg_parse->get_dataset() : "";
				const std::string program_root =
					(this->gcp && this->gcp->arg_parse) ? this->gcp->arg_parse->get_program_root() : "";
				const std::string dataset_root =
					(this->gcp && this->gcp->arg_parse) ? this->gcp->arg_parse->get_dataset_root() : "";
				const int timesteps_requested =
					(this->gcp && this->gcp->arg_parse) ? this->gcp->arg_parse->get_timesteps() : 1;
				const int num_layers =
					(this->gcp && this->gcp->arg_parse) ? this->gcp->arg_parse->get_num_layers() : 1;
				const double propagation_tau =
					(this->gcp && this->gcp->arg_parse) ? this->gcp->arg_parse->get_propagation_tau() : 0.0;
				const std::string scope_mode =
					(this->gcp && this->gcp->arg_parse) ? this->gcp->arg_parse->get_scope_mode() : "tau";
				const std::string layer_norm_mode =
					(this->gcp && this->gcp->arg_parse) ? this->gcp->arg_parse->get_layer_norm_mode() : "auto";
				const bool layer_norm_effective =
					(this->gcp && this->gcp->arg_parse) ? this->gcp->arg_parse->get_effective_layer_norm_enabled() : false;
				const std::string memory_type = this->gcp ? this->gcp->get_memory_type() : "hbm2";
				const std::string dram_config = this->gcp ? this->gcp->get_dram_config_path() : "./ktb_dram_configs/ktb_HBM2_8Gb_x128.ini";
				const double memory_bw_gbps = this->gcp ? this->gcp->get_memory_bw_gbps() : 256.0;
				const int memory_latency_cycles = this->gcp ? this->gcp->get_memory_latency_cycles() : 70;
				const double memory_hit_pj_per_txn = this->gcp ? this->gcp->get_memory_hit_pj_per_txn() : 358.0;
				const double memory_miss_pj_per_txn = this->gcp ? this->gcp->get_memory_miss_pj_per_txn() : 1894.0;
				const bool cxl_mode = this->gcp ? this->gcp->get_cxl_mode() : false;
				const double cxl_link_bw_gbps = this->gcp ? this->gcp->get_cxl_link_bw_gbps() : 0.0;
				const int cxl_extra_latency_cycles = this->gcp ? this->gcp->get_cxl_extra_latency_cycles() : 0;
					double thr = (this->gcp && this->gcp->arg_parse) ? this->gcp->arg_parse->get_mode_threshold() : 0.0;
					const long long cim_weight_cycles = (this->cim_unit != nullptr)
						? static_cast<long long>(this->cim_unit->get_weight_load_cycles()) : 0LL;
						const long long total_cycles_with_weight = static_cast<long long>(st.total_cycles) + cim_weight_cycles;
						const bool has_dcim_cfg = (this->cim_unit != nullptr);
						const int dcim_bpc = has_dcim_cfg ? this->cim_unit->get_bits_per_cycle() : 0;
						const int dcim_fifo_depth = has_dcim_cfg ? this->cim_unit->get_fifo_depth() : 0;
						const int dcim_adc_bits = has_dcim_cfg ? this->cim_unit->get_adc_bits() : 0;
						const double dcim_sigma_mv = has_dcim_cfg ? this->cim_unit->get_resolved_thermal_sigma_mv() : 0.0;
						const std::string dcim_thermal_model = has_dcim_cfg ? this->cim_unit->get_thermal_model() : "legacy";
						const double dcim_thermal_bl_cap_ff = has_dcim_cfg ? this->cim_unit->get_thermal_bl_cap_ff() : 0.0;
						const double dcim_thermal_adc_comp_sigma_mv =
							has_dcim_cfg ? this->cim_unit->get_thermal_adc_comp_sigma_mv() : 0.0;
						const std::string dcim_act_scale_mode = has_dcim_cfg ? this->cim_unit->get_activation_scale_mode_str() : "fixed";
					const int dcim_act_calib_rows = has_dcim_cfg ? this->cim_unit->get_act_calib_rows() : 0;
					const bool dcim_pv_on = has_dcim_cfg ? this->cim_unit->get_enable_pv() : false;
					const bool dcim_thermal_on = has_dcim_cfg ? this->cim_unit->get_enable_thermal() : false;
					ofs << "{\n";
					ofs << "  \"spmm_mode\": \"" << mode << "\",\n";
					ofs << "  \"cim_mode\": \"" << cim_mode << "\",\n";
					ofs << "  \"arch_variant\": \"" << arch_variant << "\",\n";
					ofs << "  \"workload_settings\": {\n";
					ofs << "    \"dataset\": \"" << dataset << "\",\n";
					ofs << "    \"program_root\": \"" << program_root << "\",\n";
					ofs << "    \"dataset_root\": \"" << dataset_root << "\",\n";
					ofs << "    \"timesteps_requested\": " << timesteps_requested << ",\n";
					ofs << "    \"num_layers\": " << num_layers << ",\n";
					ofs << "    \"propagation_tau\": " << propagation_tau << ",\n";
					ofs << "    \"scope_mode\": \"" << scope_mode << "\",\n";
					ofs << "    \"layer_norm_mode\": \"" << layer_norm_mode << "\",\n";
					ofs << "    \"layer_norm_effective\": " << (layer_norm_effective ? "true" : "false") << ",\n";
					ofs << "    \"aggregation_operator\": "
					    << "\"D_dst^-1 B_no_implicit_self_loop\"\n";
					ofs << "  },\n";
					ofs << "  \"memory_type\": \"" << memory_type << "\",\n";
					ofs << "  \"dram_config\": \"" << dram_config << "\",\n";
					ofs << "  \"memory_bw_gbps\": " << memory_bw_gbps << ",\n";
					ofs << "  \"memory_latency_cycles\": " << memory_latency_cycles << ",\n";
					ofs << "  \"memory_hit_pj_per_txn\": " << memory_hit_pj_per_txn << ",\n";
					ofs << "  \"memory_miss_pj_per_txn\": " << memory_miss_pj_per_txn << ",\n";
					ofs << "  \"cxl_mode\": " << (cxl_mode ? "true" : "false") << ",\n";
					ofs << "  \"cxl_link_bw_gbps\": " << cxl_link_bw_gbps << ",\n";
					ofs << "  \"cxl_extra_latency_cycles\": " << cxl_extra_latency_cycles << ",\n";
					ofs << "  \"mode_threshold\": " << thr << ",\n";
				if (this->gcp && this->gcp->arg_parse) {
					ofs << "  \"indexer_mode\": \"" << this->gcp->arg_parse->get_indexer_mode() << "\",\n";
					ofs << "  \"force_incremental_after_warmup\": "
					    << (this->gcp->arg_parse->get_force_incremental_after_warmup() ? "true" : "false") << ",\n";
				} else {
					ofs << "  \"indexer_mode\": \"indexstore\",\n";
					ofs << "  \"force_incremental_after_warmup\": false,\n";
				}
					ofs << "  \"n\": " << this->pe_array_n << ",\n";
					ofs << "  \"nnz\": " << this->pe_array_nnz << ",\n";
					ofs << "  \"d\": " << this->pe_array_d << ",\n";
					ofs << "  \"cim_fifo_depth\": " << dcim_fifo_depth << ",\n";
						ofs << "  \"dcim_config\": {\n";
						ofs << "    \"bits_per_cycle\": " << dcim_bpc << ",\n";
						ofs << "    \"fifo_depth\": " << dcim_fifo_depth << ",\n";
						ofs << "    \"adc_bits\": " << dcim_adc_bits << ",\n";
						ofs << "    \"resolved_thermal_sigma_mv\": " << dcim_sigma_mv << ",\n";
						ofs << "    \"thermal_model\": \"" << dcim_thermal_model << "\",\n";
						ofs << "    \"thermal_bl_cap_ff\": " << dcim_thermal_bl_cap_ff << ",\n";
						ofs << "    \"thermal_adc_comp_sigma_mv\": " << dcim_thermal_adc_comp_sigma_mv << ",\n";
						ofs << "    \"act_scale_mode\": \"" << dcim_act_scale_mode << "\",\n";
					ofs << "    \"act_calib_rows\": " << dcim_act_calib_rows << ",\n";
					ofs << "    \"pv_on\": " << (dcim_pv_on ? "true" : "false") << ",\n";
					ofs << "    \"thermal_on\": " << (dcim_thermal_on ? "true" : "false") << "\n";
					ofs << "  },\n";
					ofs << "  \"total_cycles\": " << total_cycles_with_weight << ",\n";
				ofs << "  \"compute_cycles\": " << st.compute_cycles << ",\n";
				ofs << "  \"stall_cycles\": " << st.stall_cycles << ",\n";
				ofs << "  \"memory_stall_cycles\": " << st.memory_stall_cycles << ",\n";
				ofs << "  \"total_mac_ops\": " << st.total_mac_ops << ",\n";
				ofs << "  \"cache_hits\": " << st.f_cache_hits << ",\n";
				ofs << "  \"cache_misses\": " << st.f_cache_misses << ",\n";
				ofs << "  \"l1_f_cache_hits\": " << st.l1_f_cache_hits << ",\n";
				ofs << "  \"l2_shared_hits\": " << st.l2_shared_hits << ",\n";
				ofs << "  \"l1_bank_conflict_stalls\": " << st.l1_bank_conflict_stalls << ",\n";
				ofs << "  \"l2_bank_conflict_stalls\": " << st.l2_bank_conflict_stalls << ",\n";
				ofs << "  \"noc_steal_count\": " << st.noc_steal_count << ",\n";
				ofs << "  \"noc_steal_latency_total_cycles\": " << st.noc_steal_latency_total_cycles << ",\n";
				ofs << "  \"noc_steal_avg_hops\": " << st.noc_steal_avg_hops << ",\n";
				ofs << "  \"a_load_dram_requests\": " << st.a_load_dram_requests << ",\n";
				ofs << "  \"a_load_dram_latency_sum\": " << st.a_load_dram_latency_sum << ",\n";
				ofs << "  \"dram_requests\": " << st.dram_requests << ",\n";
				ofs << "  \"row_buffer_hits\": " << st.dram_row_buffer_hits << ",\n";
				ofs << "  \"dram_fills\": " << st.dram_fills << ",\n";
				ofs << "  \"shared_f_cache_hits\": " << st.shared_f_cache_hits << ",\n";
				ofs << "  \"shared_f_cache_misses\": " << st.shared_f_cache_misses << ",\n";
				ofs << "  \"dramsim_output_dir\": \"" << (this->gcp->get_results_dir() + "/dramsim3_pe_output") << "\",\n";
				ofs << "  \"dramsim_a_load_output_dir\": \"" << (this->gcp->get_results_dir() + "/dramsim3_a_load_output") << "\",\n";
				ofs << "  \"dramsim_yz_output_dir\": \"" << (this->gcp->get_results_dir() + "/dramsim3_yz_output") << "\",\n";
				ofs << "  \"dramsim_cim_weight_output_dir\": \"" << (this->gcp->get_results_dir() + "/dramsim3_cim_weight_output") << "\",\n";
				// Indexer/FeatureStore overhead (filled when incremental driver is wired)
				ofs << "  \"index_bytes_read\": " << 0 << ",\n";
				ofs << "  \"index_bytes_write\": " << 0 << ",\n";
				ofs << "  \"feature_bytes_write\": " << 0 << ",\n";
				ofs << "  \"indexer_cam_searches\": " << 0 << ",\n";
				ofs << "  \"indexer_mshr_writes\": " << 0 << ",\n";
				ofs << "  \"indexer_mshr_reads\": " << 0 << ",\n";
				ofs << "  \"index_sram_hits\": " << 0 << ",\n";
				ofs << "  \"index_sram_misses\": " << 0 << ",\n";
				ofs << "  \"index_sram_fills\": " << 0 << ",\n";
				ofs << "  \"index_sram_evictions\": " << 0 << ",\n";
				ofs << "  \"index_lookup_cycles\": " << 0 << ",\n";
				ofs << "  \"rows_streamed\": " << this->streaming_rows_completed << ",\n";
				ofs << "  \"rows_returned\": " << this->streaming_outputs_completed << ",\n";
				ofs << "  \"pe_cim_noc_packets\": " << this->pe_cim_noc_packets << ",\n";
				ofs << "  \"pe_cim_noc_flits\": " << this->pe_cim_noc_flits << ",\n";
				ofs << "  \"pe_cim_noc_latency_cycles\": " << this->pe_cim_noc_latency_cycles << ",\n";
				ofs << "  \"pe_cim_noc_output_packets\": " << this->pe_cim_noc_output_packets << ",\n";
				ofs << "  \"cim_col_tile_passes\": " << (this->cim_unit ? this->cim_unit->get_total_col_tile_passes() : 0) << ",\n";
				ofs << "  \"cim_col_tile_pipeline_restarts\": " << (this->cim_unit ? this->cim_unit->get_col_tile_pipeline_restarts() : 0) << ",\n";
				const double cim_total_energy_pj = (this->cim_unit ? this->cim_unit->get_total_energy_pj() : 0.0);
				ofs << "  \"cim_runtime_energy_pj\": " << cim_total_energy_pj << ",\n";
				ofs << "  \"cim_eval_energy_pj\": " << 0.0 << ",\n";
				ofs << "  \"cim_total_energy_split_pj\": " << cim_total_energy_pj << ",\n";
				ofs << "  \"cim_energy_stage_split_valid\": false,\n";
				ofs << "  \"cim_total_energy_pj\": " << cim_total_energy_pj << ",\n";
				ofs << "  \"cim_input_dac_energy_pj\": " << (this->cim_unit ? this->cim_unit->get_input_dac_energy_pj() : 0.0) << ",\n";
				ofs << "  \"cim_input_dac_cycles\": " << (this->cim_unit ? this->cim_unit->get_cycles_input_dac() : 0) << ",\n";
				ofs << "  \"cim_weight_load_cycles\": " << (this->cim_unit ? this->cim_unit->get_weight_load_cycles() : 0) << ",\n";
				ofs << "  \"cim_weight_load_dram_requests\": " << (this->cim_unit ? this->cim_unit->get_weight_load_dram_requests() : 0) << ",\n";
				ofs << "  \"cim_weight_load_bytes\": " << (this->cim_unit ? this->cim_unit->get_weight_load_bytes() : 0) << ",\n";
				ofs << "  \"cim_weight_source_fp32_bytes\": " << (this->cim_unit ? this->cim_unit->get_weight_source_fp32_bytes() : 0) << ",\n";
				ofs << "  \"cim_weight_resident_int8_bytes\": " << (this->cim_unit ? this->cim_unit->get_weight_resident_int8_bytes() : 0) << ",\n";
				ofs << "  \"cim_weight_load_events\": " << (this->cim_unit ? this->cim_unit->get_weight_load_events() : 0) << ",\n";
				ofs << "  \"cim_weight_program_tile_events\": " << (this->cim_unit ? this->cim_unit->get_weight_program_tile_events() : 0) << "\n";
				ofs << "}\n";
			};

			bool streaming_enabled_for_mode = true;
			if (this->gcp && this->gcp->arg_parse) {
				const std::string mode = this->gcp->arg_parse->get_spmm_mode();
				if (mode == "incremental" || mode == "scheduled") {
					streaming_enabled_for_mode = false;
				}
			}
			if (!streaming_enabled_for_mode) {
				cout << "[Streaming] Disabled for mode; finalizing after PE-Array completion." << endl;
				this->print_pe_array_stats();
				dump_pe_array_stats_json();
				return true;
			}

			// STREAMING MODE: PE Array -> NoC -> CiM -> NoC -> PE sink
			const bool rows_streamed_done = this->streaming_rows_completed >= this->pe_array_expected_rows;
			const bool cim_compute_done = (this->cim_unit != nullptr) ? this->cim_unit->is_done() : true;
			const bool outputs_drained = this->streaming_outputs_completed >= this->pe_array_expected_rows;
			const bool noc_quiesced = this->streaming_noc_quiesced();
			if (!(rows_streamed_done && cim_compute_done && outputs_drained && noc_quiesced)) {
				this->tick_streaming_cim();
				this->gcp->tick();
				return false;  // Continue simulation
			}
			
			// All rows streamed - output results
			{
				// All done - output results
				long long total_cycles = this->gcp->get_cycle() - this->pe_array_start_cycle;
				
				// Calculate what traditional method would have taken
				// In traditional: PE ??DRAM write ??DRAM read ??CiM (sequential)
				const int DRAM_LATENCY = 100;
				long long pe_cycles = this->pe_array_spmm->get_total_cycles();
				long long memory_overhead = this->pe_array_n * DRAM_LATENCY * 2;  // Y write + read
				
				// For traditional CiM cycles, use analytical model
				const int CIM_TILES = 16;
				const int CIM_PIPELINE_LATENCY = 10;
				long long cim_cycles_analytical = (this->pe_array_n + CIM_TILES - 1) / CIM_TILES + CIM_PIPELINE_LATENCY;
				long long traditional_total = pe_cycles + memory_overhead + cim_cycles_analytical;
				
				// Streaming: actual total cycles from simulation
				long long streaming_total = total_cycles;
				cout << "\n[Streaming] End-to-end results" << endl;
				cout << "[Traditional] total cycles: " << traditional_total << endl;
				cout << "[Streaming] total cycles: " << streaming_total << endl;
				cout << "[Streaming] rows streamed: " << this->streaming_rows_completed
				     << ", rows returned: " << this->streaming_outputs_completed << endl;
				cout << "[Streaming] NoC flits: " << this->pe_cim_noc_flits
				     << ", latency sum cycles: " << this->pe_cim_noc_latency_cycles << endl;
				double speedup = (double)traditional_total / streaming_total;
				long long cycles_saved = traditional_total - streaming_total;
				
				cout << "[Streaming] cycles saved: " << cycles_saved << endl;
				cout << "[Streaming] speedup: " << fixed << setprecision(2) << speedup << "x" << endl;
				cout << "[Streaming] latency @1GHz: " << fixed << setprecision(2) << (streaming_total / 1000.0) << " us" << endl;

				
				this->print_pe_array_stats();

				dump_pe_array_stats_json();
				return true;
			}
			// PE Array done but CiM still processing - continue CiM tick
			this->tick_streaming_cim();
			this->gcp->tick();
			return false;
		}
		
#else
		// No CiM, just output SpMM results
		long long pe_cycles = this->gcp->get_cycle() - this->pe_array_start_cycle;
		cout << "\n========================================" << endl;
		cout << "  PE ARRAY SpMM COMPLETED" << endl;
		cout << "========================================" << endl;
		cout << "SpMM cycles: " << pe_cycles << endl;
		this->print_pe_array_stats();
#endif
		return true;  // End simulation
	}
	
	// Hard stop check
	if (this->gcp->get_cycle() > this->gcp->HARD_STOP) {
		cout << "[PE Array] Hard stop reached!" << endl;
		this->print_pe_array_stats();
		return true;
	}
	
	return false;  // Continue simulation
}

bool NeuraTile::is_router_extra_port_in_use(int router_x, int router_y)
{
	queue<InstX*>* router_in_extra = this->router_array[router_x][router_y].get_in_port_ref(RTR_EXTRA_PORT);
	queue<InstX*>* router_out_extra = this->router_array[router_x][router_y].get_out_port_ref(RTR_EXTRA_PORT);
	for (Conn* conn : this->conn_array) {
		if (conn->link_0_out_port == router_in_extra ||
			conn->link_1_out_port == router_in_extra ||
			conn->link_0_in_port == router_out_extra ||
			conn->link_1_in_port == router_out_extra) {
			return true;
		}
	}
	return false;
}

bool NeuraTile::setup_streaming_noc_endpoints()
{
	if (this->streaming_pe_endpoint != nullptr && this->streaming_cim_endpoint != nullptr) {
		return true;
	}

	std::vector<std::pair<int, int>> free_extra_routers;
	for (int x = 0; x < this->router_x_count; ++x) {
		for (int y = 0; y < this->router_y_count; ++y) {
			if (!this->is_router_extra_port_in_use(x, y)) {
				free_extra_routers.emplace_back(x, y);
			}
		}
	}
	if (free_extra_routers.size() < 2) {
		return false;
	}

	const auto pe_router = free_extra_routers.front();
	auto cim_router = free_extra_routers[1];
	int best_dist = -1;
	for (const auto& cand : free_extra_routers) {
		if (cand == pe_router) {
			continue;
		}
		const int dist = std::abs(cand.first - pe_router.first) + std::abs(cand.second - pe_router.second);
		if (dist > best_dist) {
			best_dist = dist;
			cim_router = cand;
		}
	}

	this->streaming_pe_router_x = pe_router.first;
	this->streaming_pe_router_y = pe_router.second;
	this->streaming_cim_router_x = cim_router.first;
	this->streaming_cim_router_y = cim_router.second;

	const string pe_endpoint_name = "P_STREAM_" + to_string(this->id);
	const string cim_endpoint_name = "Q_STREAM_" + to_string(this->id);
	this->streaming_pe_endpoint = new StreamingNoCEndpoint(this->gcp, pe_endpoint_name);
	this->streaming_cim_endpoint = new StreamingNoCEndpoint(this->gcp, cim_endpoint_name);

	NeuraRouter* pe_router_ptr = &this->router_array[this->streaming_pe_router_x][this->streaming_pe_router_y];
	NeuraRouter* cim_router_ptr = &this->router_array[this->streaming_cim_router_x][this->streaming_cim_router_y];
	this->gcp->add_edge_to_arch_graph(this->streaming_pe_endpoint->get_vertex_name(), pe_router_ptr->get_vertex_name(), STANDARD_WEIGHT, STANDARD_WEIGHT);
	this->gcp->add_edge_to_arch_graph(this->streaming_cim_endpoint->get_vertex_name(), cim_router_ptr->get_vertex_name(), STANDARD_WEIGHT, STANDARD_WEIGHT);

	this->conn_array.push_back(create_link(this->gcp, this->streaming_pe_endpoint, kStreamingEndpointPort, pe_router_ptr, RTR_EXTRA_PORT));
	this->conn_array.push_back(create_link(this->gcp, this->streaming_cim_endpoint, kStreamingEndpointPort, cim_router_ptr, RTR_EXTRA_PORT));
	pe_router_ptr->uid_to_port_map[this->streaming_pe_endpoint->get_uid()] = RTR_EXTRA_PORT;
	cim_router_ptr->uid_to_port_map[this->streaming_cim_endpoint->get_uid()] = RTR_EXTRA_PORT;

	cout << "[Streaming-NoC] PE endpoint " << this->streaming_pe_endpoint->get_vertex_name()
	     << " <-> " << pe_router_ptr->get_vertex_name()
	     << ", CiM endpoint " << this->streaming_cim_endpoint->get_vertex_name()
	     << " <-> " << cim_router_ptr->get_vertex_name() << endl;
	return true;
}

void NeuraTile::tick_streaming_noc_fabric()
{
	for (int conn_tick = 0; conn_tick < MULTI_TICK_CONN; ++conn_tick) {
		for (Conn* conn : this->conn_array) {
			conn->tick();
		}
	}
	for (int r_tick = 0; r_tick < MULTI_TICK_ROUTER; ++r_tick) {
		for (int x = 0; x < this->router_x_count; ++x) {
			for (int y = 0; y < this->router_y_count; ++y) {
				this->router_array[x][y].tick();
			}
		}
	}
}

bool NeuraTile::can_inject_streaming_row(int feature_dim)
{
	if (this->streaming_pe_endpoint == nullptr || this->streaming_cim_endpoint == nullptr) {
		return false;
	}
	const int flits = std::max(1, (feature_dim + kStreamingFlitFloats - 1) / kStreamingFlitFloats);
	const size_t max_endpoint_queue = std::max<size_t>(32, this->gcp->NEURA_ROUTER_BUFFER_SIZE * 4);
	const queue<InstX*>* out_q = this->streaming_pe_endpoint->get_out_port_ref(kStreamingEndpointPort);
	return (out_q->size() + static_cast<size_t>(flits)) <= max_endpoint_queue;
}

bool NeuraTile::inject_pe_to_cim_packet(int row_id, const float* row_data, int feature_dim)
{
	if (!this->can_inject_streaming_row(feature_dim)) {
		return false;
	}
	const int total_flits = std::max(1, (feature_dim + kStreamingFlitFloats - 1) / kStreamingFlitFloats);
	const long long now = this->gcp->get_cycle();
	for (int flit_idx = 0; flit_idx < total_flits; ++flit_idx) {
		const int feature_offset = flit_idx * kStreamingFlitFloats;
		const int feature_count = std::max(0, std::min(kStreamingFlitFloats, feature_dim - feature_offset));
		std::vector<float> payload;
		payload.reserve(static_cast<size_t>(feature_count));
		for (int i = 0; i < feature_count; ++i) {
			payload.push_back(row_data[feature_offset + i]);
		}
		const int payload_bytes = feature_count * static_cast<int>(sizeof(float));
		auto* packet = new StreamingNoCPacket(
			this->streaming_pe_endpoint->get_uid(),
			this->streaming_cim_endpoint->get_uid(),
			row_id,
			true,
			flit_idx,
			total_flits,
			feature_offset,
			std::move(payload),
			payload_bytes,
			now);
		this->streaming_pe_endpoint->get_out_port_ref(kStreamingEndpointPort)->push(packet);
		this->pe_cim_noc_packets++;
		this->pe_cim_noc_flits++;
#ifdef ENABLE_HAGG_BUFFER
		const double noc_energy_pj_per_byte = energy::noc::F_ROW_TRANSFER_PJ / 256.0;
		this->hagg_noc_energy_pj += payload_bytes * noc_energy_pj_per_byte;
#endif
	}
	return true;
}

bool NeuraTile::can_inject_cim_output(const CiMOutputPacket& output)
{
	if (this->streaming_cim_endpoint == nullptr || this->streaming_pe_endpoint == nullptr) {
		return false;
	}
	const int out_dim = static_cast<int>(output.result.size());
	const int flits = std::max(1, (out_dim + kStreamingFlitFloats - 1) / kStreamingFlitFloats);
	const size_t max_endpoint_queue = std::max<size_t>(32, this->gcp->NEURA_ROUTER_BUFFER_SIZE * 4);
	const queue<InstX*>* out_q = this->streaming_cim_endpoint->get_out_port_ref(kStreamingEndpointPort);
	return (out_q->size() + static_cast<size_t>(flits)) <= max_endpoint_queue;
}

bool NeuraTile::inject_cim_to_pe_packet(const CiMOutputPacket& output)
{
	if (!this->can_inject_cim_output(output)) {
		return false;
	}
	const int out_dim = static_cast<int>(output.result.size());
	const int total_flits = std::max(1, (out_dim + kStreamingFlitFloats - 1) / kStreamingFlitFloats);
	const long long now = this->gcp->get_cycle();
	for (int flit_idx = 0; flit_idx < total_flits; ++flit_idx) {
		const int feature_offset = flit_idx * kStreamingFlitFloats;
		const int feature_count = std::max(0, std::min(kStreamingFlitFloats, out_dim - feature_offset));
		std::vector<float> payload;
		payload.reserve(static_cast<size_t>(feature_count));
		for (int i = 0; i < feature_count; ++i) {
			payload.push_back(output.result[feature_offset + i]);
		}
		const int payload_bytes = feature_count * static_cast<int>(sizeof(float));
		auto* packet = new StreamingNoCPacket(
			this->streaming_cim_endpoint->get_uid(),
			this->streaming_pe_endpoint->get_uid(),
			output.row_id,
			false,
			flit_idx,
			total_flits,
			feature_offset,
			std::move(payload),
			payload_bytes,
			now);
		this->streaming_cim_endpoint->get_out_port_ref(kStreamingEndpointPort)->push(packet);
		this->pe_cim_noc_packets++;
		this->pe_cim_noc_flits++;
	}
	this->pe_cim_noc_output_packets++;
	return true;
}

void NeuraTile::drain_streaming_cim_endpoint()
{
	if (this->streaming_cim_endpoint == nullptr || this->cim_unit == nullptr) {
		return;
	}

	const long long now = this->gcp->get_cycle();
	queue<InstX*>* in_q = this->streaming_cim_endpoint->get_in_port_ref(kStreamingEndpointPort);
	while (!in_q->empty()) {
		InstX* inst = in_q->front();
		in_q->pop();
		if (inst == nullptr || inst->type != INSTX_TYPE_CIM_PACKET) {
			continue;
		}
		auto* packet = static_cast<StreamingNoCPacket*>(inst);
		if (!packet->pe_to_cim || packet->dest_uid != this->streaming_cim_endpoint->get_uid()) {
			delete packet;
			continue;
		}
		this->pe_cim_noc_latency_cycles += std::max<long long>(0, now - packet->inject_cycle);
#ifdef ENABLE_HAGG_BUFFER
		this->hagg_noc_cycles += std::max<long long>(0, now - packet->inject_cycle);
#endif
		auto& partial = this->streaming_cim_partial_rows[packet->row_id];
		if (partial.empty()) {
			partial.resize(static_cast<size_t>(this->pe_array_d), 0.0f);
		}
		for (size_t i = 0; i < packet->payload.size(); ++i) {
			const int idx = packet->feature_offset + static_cast<int>(i);
			if (idx >= 0 && idx < static_cast<int>(partial.size())) {
				partial[static_cast<size_t>(idx)] = packet->payload[i];
			}
		}
		const int seen = ++this->streaming_cim_partial_flits[packet->row_id];
		this->streaming_cim_expected_flits[packet->row_id] = packet->total_flits;
		if (seen == packet->total_flits) {
			this->streaming_cim_ready_rows.push_back(packet->row_id);
		}
		delete packet;
	}

	const int cim_throughput = std::max(1, this->cim_unit->get_num_tiles());
	int delivered = 0;
	while (delivered < cim_throughput && !this->streaming_cim_ready_rows.empty()) {
		if (!this->cim_unit->can_accept_input()) {
#ifdef ENABLE_HAGG_BUFFER
			this->hagg_buffer_stall_cycles++;
#endif
			break;
		}
		const int row_id = this->streaming_cim_ready_rows.front();
		auto row_it = this->streaming_cim_partial_rows.find(row_id);
		if (row_it == this->streaming_cim_partial_rows.end()) {
			this->streaming_cim_ready_rows.pop_front();
			continue;
		}
		CiMInputPacket packet;
		packet.row_id = row_id;
		packet.features = std::move(row_it->second);
		packet.is_incremental = false;
		if (!this->cim_unit->push_input(packet)) {
#ifdef ENABLE_HAGG_BUFFER
			this->hagg_buffer_stall_cycles++;
#endif
			break;
		}
		this->streaming_rows_completed++;
		this->streaming_cim_ready_rows.pop_front();
		this->streaming_cim_partial_rows.erase(row_id);
		this->streaming_cim_partial_flits.erase(row_id);
		this->streaming_cim_expected_flits.erase(row_id);
		delivered++;
	}
}

void NeuraTile::drain_streaming_pe_endpoint()
{
	if (this->streaming_pe_endpoint == nullptr) {
		return;
	}
	queue<InstX*>* in_q = this->streaming_pe_endpoint->get_in_port_ref(kStreamingEndpointPort);
	while (!in_q->empty()) {
		InstX* inst = in_q->front();
		in_q->pop();
		if (inst == nullptr || inst->type != INSTX_TYPE_CIM_PACKET) {
			continue;
		}
		auto* packet = static_cast<StreamingNoCPacket*>(inst);
		if (packet->pe_to_cim || packet->dest_uid != this->streaming_pe_endpoint->get_uid()) {
			delete packet;
			continue;
		}
		this->pe_cim_noc_latency_cycles += std::max<long long>(0, this->gcp->get_cycle() - packet->inject_cycle);
		auto& partial = this->streaming_pe_partial_rows[packet->row_id];
		if (partial.empty()) {
			partial.resize(static_cast<size_t>(this->cim_d_out), 0.0f);
		}
		for (size_t i = 0; i < packet->payload.size(); ++i) {
			const int idx = packet->feature_offset + static_cast<int>(i);
			if (idx >= 0 && idx < static_cast<int>(partial.size())) {
				partial[static_cast<size_t>(idx)] = packet->payload[i];
			}
		}
		const int seen = ++this->streaming_pe_partial_flits[packet->row_id];
		this->streaming_pe_expected_flits[packet->row_id] = packet->total_flits;
		if (seen == packet->total_flits) {
			this->streaming_pe_ready_rows.push_back(packet->row_id);
		}
		delete packet;
	}

	while (!this->streaming_pe_ready_rows.empty()) {
		const int row_id = this->streaming_pe_ready_rows.front();
		this->streaming_pe_ready_rows.pop_front();
		this->streaming_outputs_completed++;
		this->streaming_pe_partial_rows.erase(row_id);
		this->streaming_pe_partial_flits.erase(row_id);
		this->streaming_pe_expected_flits.erase(row_id);
	}
}

bool NeuraTile::streaming_noc_quiesced()
{
	if (this->streaming_pe_endpoint != nullptr) {
		if (!this->streaming_pe_endpoint->get_in_port_ref(kStreamingEndpointPort)->empty()) return false;
		if (!this->streaming_pe_endpoint->get_out_port_ref(kStreamingEndpointPort)->empty()) return false;
	}
	if (this->streaming_cim_endpoint != nullptr) {
		if (!this->streaming_cim_endpoint->get_in_port_ref(kStreamingEndpointPort)->empty()) return false;
		if (!this->streaming_cim_endpoint->get_out_port_ref(kStreamingEndpointPort)->empty()) return false;
	}
	if (!this->streaming_cim_partial_rows.empty() ||
		!this->streaming_pe_partial_rows.empty() ||
		!this->streaming_cim_ready_rows.empty() ||
		!this->streaming_pe_ready_rows.empty() ||
		!this->streaming_pending_outputs.empty()) {
		return false;
	}
	for (Conn* conn : this->conn_array) {
		if (!conn->link_0_in_port->empty() ||
			!conn->link_0_out_port->empty() ||
			!conn->link_1_in_port->empty() ||
			!conn->link_1_out_port->empty()) {
			return false;
		}
	}
	for (int x = 0; x < this->router_x_count; ++x) {
		for (int y = 0; y < this->router_y_count; ++y) {
			for (int p = 0; p < this->gcp->NEURA_ROUTER_PORT_COUNT; ++p) {
				if (!this->router_array[x][y].get_in_port_ref(p)->empty() ||
					!this->router_array[x][y].get_out_port_ref(p)->empty()) {
					return false;
				}
			}
		}
	}
	return true;
}

// Tick CiM in streaming mode: PE rows are packetized and routed through real
// NeuraRouter/Conn links (both PE->CiM input and CiM->PE return path).
void NeuraTile::tick_streaming_cim()
{
#ifdef ENABLE_CIM
	static vector<float> tmp_row(2048);
	constexpr int HAGG_SRAM_ACCESS_CYCLES = energy::sram::shared_buffer::ACCESS_CYCLES;
	constexpr int HAGG_SRAM_LINE_BYTES = 256;

	if (this->streaming_pe_endpoint == nullptr || this->streaming_cim_endpoint == nullptr) {
		if (this->pe_array_spmm->has_completed_row() && this->cim_unit && this->cim_unit->can_accept_input()) {
			const int row_id = this->pe_array_spmm->pop_completed_row(tmp_row.data(), this->pe_array_d);
			if (row_id >= 0) {
				CiMInputPacket packet;
				packet.row_id = row_id;
				packet.features.assign(tmp_row.begin(), tmp_row.begin() + this->pe_array_d);
				packet.is_incremental = false;
				if (this->cim_unit->push_input(packet)) {
					this->streaming_rows_completed++;
				}
			}
		}
		if (this->cim_unit) {
			this->cim_unit->tick();
			this->streaming_cim_cycles++;
			CiMOutputPacket out;
			while (this->cim_unit->pop_output(out)) {
				this->streaming_outputs_completed++;
			}
		}
		return;
	}

	this->tick_streaming_noc_fabric();
	this->drain_streaming_cim_endpoint();
	this->drain_streaming_pe_endpoint();

	const int cim_throughput = (this->cim_unit != nullptr) ? std::max(1, this->cim_unit->get_num_tiles()) : 1;
	int injected_rows = 0;
	while (injected_rows < cim_throughput && this->pe_array_spmm->has_completed_row()) {
		if (!this->can_inject_streaming_row(this->pe_array_d)) {
#ifdef ENABLE_HAGG_BUFFER
			this->hagg_buffer_stall_cycles++;
#endif
			break;
		}
		const int row_id = this->pe_array_spmm->pop_completed_row(tmp_row.data(), this->pe_array_d);
		if (row_id < 0) {
			break;
		}
		const int row_bytes = this->pe_array_d * static_cast<int>(sizeof(float));
		const int sram_accesses = std::max(1, (row_bytes + HAGG_SRAM_LINE_BYTES - 1) / HAGG_SRAM_LINE_BYTES);
#ifdef ENABLE_HAGG_BUFFER
		this->hagg_buffer_write_cycles += HAGG_SRAM_ACCESS_CYCLES;
		this->hagg_buffer_read_cycles += HAGG_SRAM_ACCESS_CYCLES;
		this->hagg_sram_energy_pj += sram_accesses * energy::sram::shared_buffer::WRITE_ENERGY_PJ;
		this->hagg_sram_energy_pj += sram_accesses * energy::sram::shared_buffer::READ_ENERGY_PJ;
#endif
		if (!this->inject_pe_to_cim_packet(row_id, tmp_row.data(), this->pe_array_d)) {
#ifdef ENABLE_HAGG_BUFFER
			this->hagg_buffer_stall_cycles++;
#endif
			break;
		}
		injected_rows++;
	}

	if (this->cim_unit) {
		this->cim_unit->tick();
		this->streaming_cim_cycles++;

		while (!this->streaming_pending_outputs.empty()) {
			if (!this->can_inject_cim_output(this->streaming_pending_outputs.front())) {
#ifdef ENABLE_HAGG_BUFFER
				this->hagg_buffer_stall_cycles++;
#endif
				break;
			}
			this->inject_cim_to_pe_packet(this->streaming_pending_outputs.front());
			this->streaming_pending_outputs.pop_front();
		}

		CiMOutputPacket output;
		while (this->cim_unit->pop_output(output)) {
			if (!this->can_inject_cim_output(output)) {
				this->streaming_pending_outputs.push_back(output);
#ifdef ENABLE_HAGG_BUFFER
				this->hagg_buffer_stall_cycles++;
#endif
				break;
			}
			this->inject_cim_to_pe_packet(output);
		}
	}
#endif
}
void NeuraTile::capture_exact_hbm_snapshot(TimestepResult& result)
{
	if (result.arch_ctrs.z_sram_read_bytes + result.arch_ctrs.z_sram_write_bytes !=
	    result.arch_ctrs.z_sram_rw_bytes) {
		throw std::runtime_error("exact Z-SRAM read+write checksum mismatch");
	}
	result.z_sram_read_bytes = static_cast<long long>(result.arch_ctrs.z_sram_read_bytes);
	result.z_sram_write_bytes = static_cast<long long>(result.arch_ctrs.z_sram_write_bytes);
	result.z_sram_rw_bytes = static_cast<long long>(result.arch_ctrs.z_sram_rw_bytes);
#ifdef ENABLE_DRAMSIM3_PE
	if (this->pe_array_spmm != nullptr) {
		result.hbm_pe_cumulative =
			this->pe_array_spmm->get_pe_hbm_command_energy();
	}
	if (this->dygnn_controller != nullptr) {
		result.hbm_index_cumulative =
			this->dygnn_controller->get_index_hbm_command_energy();
		result.hbm_feature_cumulative =
			this->dygnn_controller->get_feature_hbm_command_energy();
		result.hbm_yz_cumulative =
			this->dygnn_controller->get_yz_hbm_command_energy();
	}
	if (this->cim_unit != nullptr) {
		result.hbm_weight_cumulative =
			this->cim_unit->get_weight_hbm_command_energy();
	}
#else
	(void)result;
#endif
}

void NeuraTile::write_exact_hbm_sidecar()
{
	if (this->gcp == nullptr || this->multi_ts_results.empty()) return;
	const std::string out_path = this->gcp->get_results_dir() +
		"/supported_dynamic_hbm_per_timestep.json";
	std::ofstream ofs(out_path);
	if (!ofs.is_open()) {
		throw std::runtime_error("failed to open exact HBM sidecar: " + out_path);
	}
	ofs << std::setprecision(17);
	ofs << "{\n"
	    << "  \"schema\": \"despoina.hbm-command-energy.v2\",\n"
	    << "  \"allocation_method\": \"direct_per_timestep_cumulative_snapshot_delta\",\n"
	    << "  \"energy_scope\": \"DRAMsim3_command_dynamic_only\",\n"
	    << "  \"energy_unit\": \"pJ\",\n"
	    << "  \"counter_unit\": \"issued_commands\",\n"
	    << "  \"epoch_boundaries\": \"post_previous_timestep_to_post_current_timestep; t0 starts_at_memory_system_lifetime_zero\",\n"
	    << "  \"t0_included_in_sidecar\": true,\n"
	    << "  \"t0_reported_in_figure8\": false,\n"
	    << "  \"owners_disjoint\": true,\n"
	    << "  \"per_timestep\": [\n";
#ifdef ENABLE_DRAMSIM3_PE
	dramsim3::CommandEnergySnapshot previous[5];
	const char* names[5] = {"pe", "index", "feature", "yz", "weight"};
	const char* owners[5] = {
		"pe_feature_hbm_memory_system",
		"delta_index_hbm_memory_system",
		"feature_update_hbm_memory_system",
		"y_z_materialization_hbm_memory_system",
		"cim_weight_hbm_memory_system"
	};
	auto write_snapshot = [&ofs](const dramsim3::CommandEnergySnapshot& s) {
		ofs << "{\"act\": " << s.act_commands
		    << ", \"read\": " << s.read_commands
		    << ", \"write\": " << s.write_commands
		    << ", \"refresh\": " << s.refresh_commands
		    << ", \"refresh_bank\": " << s.refresh_bank_commands
		    << ", \"act_energy_pj\": " << s.act_energy_pj
		    << ", \"read_energy_pj\": " << s.read_energy_pj
		    << ", \"write_energy_pj\": " << s.write_energy_pj
		    << ", \"refresh_energy_pj\": " << s.refresh_energy_pj
		    << ", \"refresh_bank_energy_pj\": " << s.refresh_bank_energy_pj
		    << ", \"total_energy_pj\": " << s.total_command_energy_pj() << "}";
	};
	auto subtract = [](const dramsim3::CommandEnergySnapshot& end,
	                   const dramsim3::CommandEnergySnapshot& start) {
		if (end.act_commands < start.act_commands ||
		    end.read_commands < start.read_commands ||
		    end.write_commands < start.write_commands ||
		    end.refresh_commands < start.refresh_commands ||
		    end.refresh_bank_commands < start.refresh_bank_commands ||
		    end.act_energy_pj + 1e-9 < start.act_energy_pj ||
		    end.read_energy_pj + 1e-9 < start.read_energy_pj ||
		    end.write_energy_pj + 1e-9 < start.write_energy_pj ||
		    end.refresh_energy_pj + 1e-9 < start.refresh_energy_pj ||
		    end.refresh_bank_energy_pj + 1e-9 < start.refresh_bank_energy_pj) {
			throw std::runtime_error("non-monotonic HBM command-energy snapshot");
		}
		dramsim3::CommandEnergySnapshot d;
		d.act_commands = end.act_commands - start.act_commands;
		d.read_commands = end.read_commands - start.read_commands;
		d.write_commands = end.write_commands - start.write_commands;
		d.refresh_commands = end.refresh_commands - start.refresh_commands;
		d.refresh_bank_commands = end.refresh_bank_commands - start.refresh_bank_commands;
		d.act_energy_pj = std::max(0.0, end.act_energy_pj - start.act_energy_pj);
		d.read_energy_pj = std::max(0.0, end.read_energy_pj - start.read_energy_pj);
		d.write_energy_pj = std::max(0.0, end.write_energy_pj - start.write_energy_pj);
		d.refresh_energy_pj = std::max(0.0, end.refresh_energy_pj - start.refresh_energy_pj);
		d.refresh_bank_energy_pj = std::max(0.0, end.refresh_bank_energy_pj - start.refresh_bank_energy_pj);
		return d;
	};
	for (size_t i = 0; i < this->multi_ts_results.size(); ++i) {
		const TimestepResult& row = this->multi_ts_results[i];
		const dramsim3::CommandEnergySnapshot current[5] = {
			row.hbm_pe_cumulative, row.hbm_index_cumulative,
			row.hbm_feature_cumulative, row.hbm_yz_cumulative,
			row.hbm_weight_cumulative
		};
		ofs << "    {\n"
		    << "      \"timestep\": " << row.timestep << ",\n"
		    << "      \"allocation_method\": \"direct_per_timestep_cumulative_snapshot_delta\",\n"
		    << "      \"owners_disjoint\": true,\n"
		    << "      \"epoch\": {\"start\": \""
		    << (i == 0 ? "memory_system_lifetime_zero" :
		        ("post_t" + std::to_string(this->multi_ts_results[i - 1].timestep)))
		    << "\", \"end\": \"post_t" << row.timestep << "\"},\n"
		    << "      \"streams\": {\n";
		for (int owner = 0; owner < 5; ++owner) {
			const dramsim3::CommandEnergySnapshot delta =
				subtract(current[owner], previous[owner]);
			ofs << "        \"" << names[owner] << "\": {\"status\": \"direct_measured\", "
			    << "\"owner\": \"" << owners[owner] << "\", "
			    << "\"measurement\": \"cumulative_snapshot_delta\", "
			    << "\"start_snapshot\": ";
			write_snapshot(previous[owner]);
			ofs << ", \"end_snapshot\": ";
			write_snapshot(current[owner]);
			ofs << ", \"delta\": ";
			write_snapshot(delta);
			ofs << "},\n";
			previous[owner] = current[owner];
		}
		ofs << "        \"a_load\": {\"status\": \"unavailable\", "
		    << "\"owner\": \"adjacency_warmup_hbm_memory_system\", "
		    << "\"availability\": false, \"energy_pj\": null, "
		    << "\"reason\": \"owner API intentionally excluded from priced Figure 8 energy\"}\n"
		    << "      }\n"
		    << "    }" << (i + 1 < this->multi_ts_results.size() ? "," : "") << "\n";
	}
#else
	(void)ofs;
#endif
	ofs << "  ]\n}\n";
}

// Write multi-timestep stats to JSON
void NeuraTile::write_multi_ts_stats()
{
	if (this->multi_ts_results.empty()) return;

	const std::string out_path = this->gcp->get_results_dir() + "/pe_array_stats.json";
	std::ofstream ofs(out_path);
	if (!ofs.is_open()) {
		cerr << "[Multi-TS] Failed to open " << out_path << endl;
		return;
	}

	// Aggregate stats
	long long total_cycles = 0;
	long long total_mac_ops = 0;
	long long total_cache_hits = 0;
	long long total_cache_misses = 0;
	long long total_l1_f_cache_hits = 0;
	long long total_l2_shared_hits = 0;
	long long total_l1_bank_conflict_stalls = 0;
	long long total_l2_bank_conflict_stalls = 0;
	long long total_noc_steal_count = 0;
	long long total_noc_steal_latency_cycles = 0;
	double total_noc_steal_weighted_hops = 0.0;
	long long total_a_load_dram_requests = 0;
	long long total_a_load_dram_latency_sum = 0;
	long long total_dram_requests = 0;
	long long total_row_buffer_hits = 0;
	long long total_dram_fills = 0;
	long long total_combine_cycles = 0;
	long long total_cim_cycles = 0;
	long long total_cim_exposed_cycles = 0;
	long long total_cim_pe_backpressure_cycles = 0;
	long long total_cim_input_fifo_full_rejections = 0;
	long long total_cim_input_fifo_full_cycles = 0;
	long long total_cim_output_fifo_full_cycles = 0;
	long long total_cim_output_fifo_full_stall_cycles = 0;
	long long total_cim_stage3_output_block_cycles = 0;
	long long total_cim_inject_no_empty_lane_cycles = 0;
	long long total_cim_active_lane_cycles = 0;
	long long total_cim_active_slot_cycles = 0;
	long long total_cim_input_fifo_occupancy_cycles = 0;
	long long total_cim_output_fifo_occupancy_cycles = 0;
	long long total_cim_stream_cycles = 0;
	int max_cim_active_lanes = 0;
	long long total_cim_rows = 0;
	long long total_finalized_rows = 0;
	long long total_finalized_packets = 0;
	long long total_dcim_scalar_macs = 0;
	long long total_dcim_compute_cycles = 0;
	long long total_tile_busy_work_cycles = 0;
	long long total_causal_stream_elapsed_cycles = 0;
	long long total_exposed_tail_cycles = 0;
	long long total_dcim_streams = 0;
	long long total_dcim_drain_sensitivity_cycles = 0;
	long long total_dcim_compute_only_exposed_cycles = 0;
	long long total_dcim_plus6_exposed_cycles = 0;
	long long total_finalized_duplicate_packets = 0;
	long long total_finalized_invalid_packets = 0;
	long long total_finalized_timeouts = 0;
	long long total_dcim_activation_quantized_elements = 0;
	long long total_dcim_dequantized_elements = 0;
	long long total_dcim_comparator_tree_passes = 0;
	long long total_y_dram_read_cycles = 0;
	long long total_y_dram_write_cycles = 0;
	long long total_z_dram_rw_cycles = 0;
	long long total_z_sram_read_bytes = 0;
	long long total_z_sram_write_bytes = 0;
	long long total_z_sram_rw_bytes = 0;
	long long total_row_detect_cycles = 0;
	long long total_row_detect_checks = 0;
	long long total_row_detect_sram_reads = 0;
	uint64_t total_index_bytes_read = 0;
	uint64_t total_index_bytes_write = 0;
	uint64_t total_index_update_ops = 0;
	uint64_t total_index_update_payload_bytes = 0;
	long long total_index_update_dram_cycles = 0;
	long long total_index_dram_cycles = 0;
	uint64_t total_feature_bytes_write = 0;
	uint64_t total_indexer_cam_searches = 0;
	uint64_t total_indexer_mshr_writes = 0;
	uint64_t total_indexer_mshr_reads = 0;
	uint64_t total_index_sram_hits = 0;
	uint64_t total_index_sram_misses = 0;
	uint64_t total_index_sram_fills = 0;
	uint64_t total_index_sram_evictions = 0;
	long long total_index_lookup_cycles = 0;
	// Accumulate per-layer cycles dynamically
	std::map<int, long long> total_higher_layer_cycles; // key = layer index (0=L2, 1=L3, ...)

	for (const auto& res : this->multi_ts_results) {
		total_cycles += res.cycles;
		for (int l = 0; l < (int)res.higher_layer_cycles.size(); l++) {
			total_higher_layer_cycles[l] += res.higher_layer_cycles[l];
		}
		total_mac_ops += res.pe_stats.total_mac_ops;
		total_cache_hits += res.pe_stats.f_cache_hits;
		total_cache_misses += res.pe_stats.f_cache_misses;
		total_l1_f_cache_hits += res.pe_stats.l1_f_cache_hits;
		total_l2_shared_hits += res.pe_stats.l2_shared_hits;
		total_l1_bank_conflict_stalls += res.pe_stats.l1_bank_conflict_stalls;
		total_l2_bank_conflict_stalls += res.pe_stats.l2_bank_conflict_stalls;
		total_noc_steal_count += res.pe_stats.noc_steal_count;
		total_noc_steal_latency_cycles += res.pe_stats.noc_steal_latency_total_cycles;
		total_noc_steal_weighted_hops +=
			static_cast<double>(res.pe_stats.noc_steal_avg_hops) * static_cast<double>(res.pe_stats.noc_steal_count);
		total_a_load_dram_requests += res.pe_stats.a_load_dram_requests;
		total_a_load_dram_latency_sum += res.pe_stats.a_load_dram_latency_sum;
		total_dram_requests += res.pe_stats.dram_requests;
		total_row_buffer_hits += res.pe_stats.dram_row_buffer_hits;
		total_dram_fills += res.pe_stats.dram_fills;
		total_combine_cycles += res.combine_cycles;
		total_cim_cycles += res.cim_cycles;
		total_cim_exposed_cycles += res.cim_exposed_cycles;
		total_cim_pe_backpressure_cycles += res.cim_pe_backpressure_cycles;
		total_cim_input_fifo_full_rejections += res.cim_input_fifo_full_rejections;
		total_cim_input_fifo_full_cycles += res.cim_input_fifo_full_cycles;
		total_cim_output_fifo_full_cycles += res.cim_output_fifo_full_cycles;
		total_cim_output_fifo_full_stall_cycles += res.cim_output_fifo_full_stall_cycles;
		total_cim_stage3_output_block_cycles += res.cim_stage3_output_block_cycles;
		total_cim_inject_no_empty_lane_cycles += res.cim_inject_no_empty_lane_cycles;
		total_cim_active_lane_cycles += res.cim_active_lane_cycles;
		total_cim_active_slot_cycles += res.cim_active_slot_cycles;
		total_cim_input_fifo_occupancy_cycles += res.cim_input_fifo_occupancy_cycles;
		total_cim_output_fifo_occupancy_cycles += res.cim_output_fifo_occupancy_cycles;
		total_cim_stream_cycles += res.cim_stream_cycles;
		max_cim_active_lanes = std::max(max_cim_active_lanes, res.cim_max_active_lanes);
		total_cim_rows += res.cim_rows;
		total_finalized_rows += res.finalized_rows;
		total_finalized_packets += res.finalized_packets;
		total_dcim_scalar_macs += res.dcim_scalar_macs;
		total_dcim_compute_cycles += res.dcim_compute_cycles;
		total_tile_busy_work_cycles += res.tile_busy_work_cycles;
		total_causal_stream_elapsed_cycles += res.causal_stream_elapsed_cycles;
		total_exposed_tail_cycles += res.exposed_tail_cycles;
		total_dcim_streams += res.dcim_streams;
		total_dcim_drain_sensitivity_cycles += res.dcim_drain_sensitivity_cycles;
		total_dcim_compute_only_exposed_cycles += res.dcim_compute_only_exposed_cycles;
		total_dcim_plus6_exposed_cycles += res.dcim_plus6_exposed_cycles;
		total_finalized_duplicate_packets += res.finalized_duplicate_packets;
		total_finalized_invalid_packets += res.finalized_invalid_packets;
		total_finalized_timeouts += res.finalized_timeouts;
		total_dcim_activation_quantized_elements += res.dcim_activation_quantized_elements;
		total_dcim_dequantized_elements += res.dcim_dequantized_elements;
		total_dcim_comparator_tree_passes += res.dcim_comparator_tree_passes;
		total_y_dram_read_cycles += res.y_dram_read_cycles;
		total_y_dram_write_cycles += res.y_dram_write_cycles;
		total_z_dram_rw_cycles += res.z_dram_rw_cycles;
		total_z_sram_read_bytes += res.z_sram_read_bytes;
		total_z_sram_write_bytes += res.z_sram_write_bytes;
		total_z_sram_rw_bytes += res.z_sram_rw_bytes;
		total_row_detect_cycles += res.row_detect_cycles;
		total_row_detect_checks += res.row_detect_checks;
		total_row_detect_sram_reads += res.row_detect_sram_reads;
		total_index_bytes_read += res.indexer_ctrs.index_bytes_read;
		total_index_bytes_write += res.indexer_ctrs.index_bytes_write;
		total_index_update_ops += res.indexer_ctrs.index_update_ops;
		total_index_update_payload_bytes += res.indexer_ctrs.index_update_payload_bytes;
		total_index_update_dram_cycles += res.indexer_ctrs.index_update_dram_cycles;
		total_index_dram_cycles += res.indexer_ctrs.index_dram_cycles;
		total_feature_bytes_write += res.indexer_ctrs.feature_bytes_write;
		total_indexer_cam_searches += res.indexer_ctrs.cam_searches;
		total_indexer_mshr_writes += res.indexer_ctrs.mshr_writes;
		total_indexer_mshr_reads += res.indexer_ctrs.mshr_reads;
		total_index_sram_hits += res.indexer_ctrs.index_sram_hits;
		total_index_sram_misses += res.indexer_ctrs.index_sram_misses;
		total_index_sram_fills += res.indexer_ctrs.index_sram_fills;
		total_index_sram_evictions += res.indexer_ctrs.index_sram_evictions;
		total_index_lookup_cycles += res.indexer_ctrs.index_lookup_cycles;
	}

	std::string mode = (this->gcp && this->gcp->arg_parse) ? this->gcp->arg_parse->get_spmm_mode() : "full";
	std::string cim_mode = (this->gcp && this->gcp->arg_parse) ? this->gcp->arg_parse->get_cim_mode() : "none";
	const std::string arch_variant =
		(this->dygnn_controller != nullptr)
			? this->dygnn_controller->get_arch_variant_string()
			: ((this->gcp && this->gcp->arg_parse)
			       ? this->gcp->arg_parse->get_arch_variant()
			       : default_arch_variant_string());
	const std::string dataset =
		(this->gcp && this->gcp->arg_parse) ? this->gcp->arg_parse->get_dataset() : "";
	const std::string program_root =
		(this->gcp && this->gcp->arg_parse) ? this->gcp->arg_parse->get_program_root() : "";
	const std::string dataset_root =
		(this->gcp && this->gcp->arg_parse) ? this->gcp->arg_parse->get_dataset_root() : "";
	const int timesteps_requested =
		(this->gcp && this->gcp->arg_parse) ? this->gcp->arg_parse->get_timesteps() : 1;
	const int num_layers =
		(this->gcp && this->gcp->arg_parse) ? this->gcp->arg_parse->get_num_layers() : 1;
	const double propagation_tau =
		(this->gcp && this->gcp->arg_parse) ? this->gcp->arg_parse->get_propagation_tau() : 0.0;
	const std::string scope_mode =
		(this->gcp && this->gcp->arg_parse) ? this->gcp->arg_parse->get_scope_mode() : "tau";
	const std::string layer_norm_mode =
		(this->gcp && this->gcp->arg_parse) ? this->gcp->arg_parse->get_layer_norm_mode() : "auto";
	const bool layer_norm_effective =
		(this->gcp && this->gcp->arg_parse) ? this->gcp->arg_parse->get_effective_layer_norm_enabled() : false;
	const std::string memory_type = this->gcp ? this->gcp->get_memory_type() : "hbm2";
	const std::string dram_config = this->gcp ? this->gcp->get_dram_config_path() : "./ktb_dram_configs/ktb_HBM2_8Gb_x128.ini";
	const double memory_bw_gbps = this->gcp ? this->gcp->get_memory_bw_gbps() : 256.0;
	const int memory_latency_cycles = this->gcp ? this->gcp->get_memory_latency_cycles() : 70;
	const double memory_hit_pj_per_txn = this->gcp ? this->gcp->get_memory_hit_pj_per_txn() : 358.0;
	const double memory_miss_pj_per_txn = this->gcp ? this->gcp->get_memory_miss_pj_per_txn() : 1894.0;
	const bool cxl_mode = this->gcp ? this->gcp->get_cxl_mode() : false;
	const double cxl_link_bw_gbps = this->gcp ? this->gcp->get_cxl_link_bw_gbps() : 0.0;
	const int cxl_extra_latency_cycles = this->gcp ? this->gcp->get_cxl_extra_latency_cycles() : 0;
	std::string phase_mode = (this->gcp && this->gcp->arg_parse) ? this->gcp->arg_parse->get_phase_mode() : "three_phase";
	double thr = (this->gcp && this->gcp->arg_parse) ? this->gcp->arg_parse->get_mode_threshold() : 0.0;
	std::string indexer_mode = (this->gcp && this->gcp->arg_parse) ? this->gcp->arg_parse->get_indexer_mode() : "indexstore";
	auto phase_mode_to_string = [](PhaseMode mode_enum) -> const char* {
		switch (mode_enum) {
			case PhaseMode::THREE_PHASE:
				return "three_phase";
			case PhaseMode::EDGE_ONLY:
				return "edge_only";
			case PhaseMode::TWO_PHASE:
				return "two_phase";
			case PhaseMode::ANEW_FULL:
				return "anew_full";
		}
		return "three_phase";
	};
	// Memory placement assumptions for energy accounting.
	// - index_store: on-chip SRAM (or partial hybrid SRAM+DRAM when misses occur)
	// - feature_store: off-chip (HBM) modeled by DRAMsim3 feature output
	std::string index_mem = "dram";
	if (indexer_mode == "indexstore") {
		index_mem = (total_index_sram_misses > 0) ? "hybrid" : "sram";
	}
	std::string feature_mem = "dram";
	const long long cim_weight_load_cycles = (this->cim_unit != nullptr)
	    ? static_cast<long long>(this->cim_unit->get_weight_load_cycles()) : 0LL;
	const long long cim_weight_load_dram_requests = (this->cim_unit != nullptr)
	    ? static_cast<long long>(this->cim_unit->get_weight_load_dram_requests()) : 0LL;
	const long long cim_weight_load_bytes = (this->cim_unit != nullptr)
	    ? static_cast<long long>(this->cim_unit->get_weight_load_bytes()) : 0LL;
	const bool has_dcim_cfg = (this->cim_unit != nullptr);
	const int dcim_bpc = has_dcim_cfg ? this->cim_unit->get_bits_per_cycle() : 0;
	const int dcim_fifo_depth = has_dcim_cfg ? this->cim_unit->get_fifo_depth() : 0;
	const int dcim_adc_bits = has_dcim_cfg ? this->cim_unit->get_adc_bits() : 0;
		const double dcim_sigma_mv = has_dcim_cfg ? this->cim_unit->get_resolved_thermal_sigma_mv() : 0.0;
		const std::string dcim_thermal_model = has_dcim_cfg ? this->cim_unit->get_thermal_model() : "legacy";
		const double dcim_thermal_bl_cap_ff = has_dcim_cfg ? this->cim_unit->get_thermal_bl_cap_ff() : 0.0;
		const double dcim_thermal_adc_comp_sigma_mv =
		    has_dcim_cfg ? this->cim_unit->get_thermal_adc_comp_sigma_mv() : 0.0;
		const std::string dcim_act_scale_mode = has_dcim_cfg ? this->cim_unit->get_activation_scale_mode_str() : "fixed";
	const int dcim_act_calib_rows = has_dcim_cfg ? this->cim_unit->get_act_calib_rows() : 0;
	const bool dcim_pv_on = has_dcim_cfg ? this->cim_unit->get_enable_pv() : false;
	const bool dcim_thermal_on = has_dcim_cfg ? this->cim_unit->get_enable_thermal() : false;
	const double cim_energy_snapshot_pj = has_dcim_cfg ? this->cim_unit->get_total_energy_pj() : 0.0;
	double cim_runtime_energy_pj = cim_energy_snapshot_pj;
	double cim_eval_energy_pj = 0.0;
	if (this->multi_ts_eval_energy_valid) {
		cim_runtime_energy_pj = this->multi_ts_runtime_cim_energy_pj;
		cim_eval_energy_pj = this->multi_ts_eval_cim_energy_pj;
	}
	const double cim_total_energy_split_pj = cim_runtime_energy_pj + cim_eval_energy_pj;

	ofs << "{\n";
	ofs << "  \"spmm_mode\": \"" << mode << "\",\n";
	ofs << "  \"cim_mode\": \"" << cim_mode << "\",\n";
	ofs << "  \"arch_variant\": \"" << arch_variant << "\",\n";
	ofs << "  \"hagg_staging\": {\n";
	ofs << "    \"model\": " << pe_hagg_staging::kModel << ",\n";
	ofs << "    \"bounded\": " << (pe_hagg_staging::kBounded ? "true" : "false") << ",\n";
	ofs << "    \"bytes\": " << pe_hagg_staging::kBytes << ",\n";
	ofs << "    \"line_bytes\": " << pe_hagg_staging::kLineBytes << ",\n";
	ofs << "    \"capacity_rows\": " << pe_hagg_staging::kCapacityRows << ",\n";
	ofs << "    \"current_occupancy_rows\": " << this->pe_array_spmm->get_staging_current_occupancy_rows() << ",\n";
	ofs << "    \"max_occupancy_rows\": " << this->pe_array_spmm->get_staging_max_occupancy_rows() << ",\n";
	ofs << "    \"max_occupancy_bytes\": "
	    << pe_hagg_staging::occupancy_bytes(this->pe_array_spmm->get_staging_max_occupancy_rows()) << ",\n";
	ofs << "    \"staging_full_stall_cycles\": " << this->pe_array_spmm->get_staging_full_stall_cycles() << ",\n";
	ofs << "    \"functional_shadow\": \"fp32\",\n";
	ofs << "    \"physical_packet\": \"one_aligned_256B_int8_row_scale_metadata\"\n";
	ofs << "  },\n";
	ofs << "  \"workload_settings\": {\n";
	ofs << "    \"dataset\": \"" << dataset << "\",\n";
	ofs << "    \"program_root\": \"" << program_root << "\",\n";
	ofs << "    \"dataset_root\": \"" << dataset_root << "\",\n";
	ofs << "    \"timesteps_requested\": " << timesteps_requested << ",\n";
	ofs << "    \"num_layers\": " << num_layers << ",\n";
	ofs << "    \"propagation_tau\": " << propagation_tau << ",\n";
	ofs << "    \"scope_mode\": \"" << scope_mode << "\",\n";
	ofs << "    \"layer_norm_mode\": \"" << layer_norm_mode << "\",\n";
	ofs << "    \"layer_norm_effective\": " << (layer_norm_effective ? "true" : "false") << "\n";
	ofs << "  },\n";
	ofs << "  \"dcim_digital\": "
	    << ((this->gcp && this->gcp->arg_parse && this->gcp->arg_parse->get_dcim_digital()) ? "true" : "false")
	    << ",\n";
	ofs << "  \"memory_type\": \"" << memory_type << "\",\n";
	ofs << "  \"dram_config\": \"" << dram_config << "\",\n";
	ofs << "  \"memory_bw_gbps\": " << memory_bw_gbps << ",\n";
	ofs << "  \"memory_latency_cycles\": " << memory_latency_cycles << ",\n";
	ofs << "  \"memory_hit_pj_per_txn\": " << memory_hit_pj_per_txn << ",\n";
	ofs << "  \"memory_miss_pj_per_txn\": " << memory_miss_pj_per_txn << ",\n";
	ofs << "  \"cxl_mode\": " << (cxl_mode ? "true" : "false") << ",\n";
	ofs << "  \"cxl_link_bw_gbps\": " << cxl_link_bw_gbps << ",\n";
	ofs << "  \"cxl_extra_latency_cycles\": " << cxl_extra_latency_cycles << ",\n";
	ofs << "  \"phase_mode\": \"" << phase_mode << "\",\n";
	ofs << "  \"mode_threshold\": " << thr << ",\n";
	ofs << "  \"indexer_mode\": \"" << indexer_mode << "\",\n";
	ofs << "  \"force_incremental_after_warmup\": "
	    << ((this->gcp && this->gcp->arg_parse && this->gcp->arg_parse->get_force_incremental_after_warmup()) ? "true" : "false") << ",\n";
	ofs << "  \"index_mem\": \"" << index_mem << "\",\n";
	ofs << "  \"feature_mem\": \"" << feature_mem << "\",\n";
	ofs << "  \"n\": " << this->pe_array_n << ",\n";
	ofs << "  \"nnz\": " << this->pe_array_nnz << ",\n";
	ofs << "  \"d\": " << this->pe_array_d << ",\n";
	ofs << "  \"cim_fifo_depth\": " << dcim_fifo_depth << ",\n";
		ofs << "  \"dcim_config\": {\n";
		ofs << "    \"bits_per_cycle\": " << dcim_bpc << ",\n";
		ofs << "    \"fifo_depth\": " << dcim_fifo_depth << ",\n";
		ofs << "    \"adc_bits\": " << dcim_adc_bits << ",\n";
		ofs << "    \"resolved_thermal_sigma_mv\": " << dcim_sigma_mv << ",\n";
		ofs << "    \"thermal_model\": \"" << dcim_thermal_model << "\",\n";
		ofs << "    \"thermal_bl_cap_ff\": " << dcim_thermal_bl_cap_ff << ",\n";
		ofs << "    \"thermal_adc_comp_sigma_mv\": " << dcim_thermal_adc_comp_sigma_mv << ",\n";
		ofs << "    \"act_scale_mode\": \"" << dcim_act_scale_mode << "\",\n";
	ofs << "    \"act_calib_rows\": " << dcim_act_calib_rows << ",\n";
	ofs << "    \"pv_on\": " << (dcim_pv_on ? "true" : "false") << ",\n";
	ofs << "    \"thermal_on\": " << (dcim_thermal_on ? "true" : "false") << "\n";
	ofs << "  },\n";
	ofs << "  \"isolation_mode\": \"" << this->gcp->arg_parse->get_isolation_mode() << "\",\n";
	ofs << "  \"timesteps\": " << this->multi_ts_results.size() << ",\n";
	// Keep reported performance cycles aligned with per-timestep runtime cycles.
	// CiM weight-load counters are exported separately and should not shift latency
	// when accuracy validation is enabled.
	ofs << "  \"total_cycles\": " << total_cycles << ",\n";
	for (const auto& kv : total_higher_layer_cycles) {
		ofs << "  \"layer" << (kv.first + 2) << "_cycles\": " << kv.second << ",\n";
	}
	ofs << "  \"total_mac_ops\": " << total_mac_ops << ",\n";
	ofs << "  \"cache_hits\": " << total_cache_hits << ",\n";
	ofs << "  \"cache_misses\": " << total_cache_misses << ",\n";
	ofs << "  \"l1_f_cache_hits\": " << total_l1_f_cache_hits << ",\n";
	ofs << "  \"l2_shared_hits\": " << total_l2_shared_hits << ",\n";
	ofs << "  \"l1_bank_conflict_stalls\": " << total_l1_bank_conflict_stalls << ",\n";
	ofs << "  \"l2_bank_conflict_stalls\": " << total_l2_bank_conflict_stalls << ",\n";
	ofs << "  \"noc_steal_count\": " << total_noc_steal_count << ",\n";
	ofs << "  \"noc_steal_latency_total_cycles\": " << total_noc_steal_latency_cycles << ",\n";
	ofs << "  \"noc_steal_avg_hops\": "
	    << (total_noc_steal_count > 0 ? (total_noc_steal_weighted_hops / static_cast<double>(total_noc_steal_count)) : 0.0)
	    << ",\n";
	ofs << "  \"a_load_dram_requests\": " << total_a_load_dram_requests << ",\n";
	ofs << "  \"a_load_dram_latency_sum\": " << total_a_load_dram_latency_sum << ",\n";
	ofs << "  \"dram_requests\": " << total_dram_requests << ",\n";
	ofs << "  \"row_buffer_hits\": " << total_row_buffer_hits << ",\n";
	ofs << "  \"dram_fills\": " << total_dram_fills << ",\n";
	ofs << "  \"combine_cycles\": " << total_combine_cycles << ",\n";
	ofs << "  \"cim_cycles\": " << total_cim_cycles << ",\n";
	ofs << "  \"cim_exposed_cycles\": " << total_cim_exposed_cycles << ",\n";
	ofs << "  \"cim_stream_cycles\": " << total_cim_stream_cycles << ",\n";
	ofs << "  \"cim_pe_backpressure_cycles\": " << total_cim_pe_backpressure_cycles << ",\n";
	ofs << "  \"cim_input_fifo_full_rejections\": " << total_cim_input_fifo_full_rejections << ",\n";
	ofs << "  \"cim_input_fifo_full_cycles\": " << total_cim_input_fifo_full_cycles << ",\n";
	ofs << "  \"cim_output_fifo_full_cycles\": " << total_cim_output_fifo_full_cycles << ",\n";
	ofs << "  \"cim_output_fifo_full_stall_cycles\": " << total_cim_output_fifo_full_stall_cycles << ",\n";
	ofs << "  \"cim_stage3_output_block_cycles\": " << total_cim_stage3_output_block_cycles << ",\n";
	ofs << "  \"cim_inject_no_empty_lane_cycles\": " << total_cim_inject_no_empty_lane_cycles << ",\n";
	ofs << "  \"cim_active_lane_cycles\": " << total_cim_active_lane_cycles << ",\n";
	ofs << "  \"cim_active_slot_cycles\": " << total_cim_active_slot_cycles << ",\n";
	ofs << "  \"cim_input_fifo_occupancy_cycles\": " << total_cim_input_fifo_occupancy_cycles << ",\n";
	ofs << "  \"cim_output_fifo_occupancy_cycles\": " << total_cim_output_fifo_occupancy_cycles << ",\n";
	ofs << "  \"cim_max_active_lanes\": " << max_cim_active_lanes << ",\n";
	ofs << "  \"cim_avg_active_lanes\": "
	    << (total_cim_stream_cycles > 0
	            ? static_cast<double>(total_cim_active_lane_cycles) / static_cast<double>(total_cim_stream_cycles)
	            : 0.0)
	    << ",\n";
	ofs << "  \"cim_avg_active_slots\": "
	    << (total_cim_stream_cycles > 0
	            ? static_cast<double>(total_cim_active_slot_cycles) / static_cast<double>(total_cim_stream_cycles)
	            : 0.0)
	    << ",\n";
	ofs << "  \"cim_avg_input_fifo_occupancy\": "
	    << (total_cim_stream_cycles > 0
	            ? static_cast<double>(total_cim_input_fifo_occupancy_cycles) / static_cast<double>(total_cim_stream_cycles)
	            : 0.0)
	    << ",\n";
	ofs << "  \"cim_avg_output_fifo_occupancy\": "
	    << (total_cim_stream_cycles > 0
	            ? static_cast<double>(total_cim_output_fifo_occupancy_cycles) / static_cast<double>(total_cim_stream_cycles)
	            : 0.0)
	    << ",\n";
	ofs << "  \"cim_rows\": " << total_cim_rows << ",\n";
	ofs << "  \"dcim_publication_cycle_shape\": \"128x128-only\",\n";
	ofs << "  \"finalized_rows\": " << total_finalized_rows << ",\n";
	ofs << "  \"finalized_packets\": " << total_finalized_packets << ",\n";
	ofs << "  \"dcim_scalar_macs_exact_once\": " << total_dcim_scalar_macs << ",\n";
	ofs << "  \"dcim_tile_busy_cycles_from_events\": " << total_dcim_compute_cycles << ",\n";
	ofs << "  \"tile_busy_work_cycles\": " << total_tile_busy_work_cycles << ",\n";
	ofs << "  \"causal_stream_elapsed_cycles\": " << total_causal_stream_elapsed_cycles << ",\n";
	ofs << "  \"exposed_tail_cycles\": " << total_exposed_tail_cycles << ",\n";
	ofs << "  \"dcim_streams\": " << total_dcim_streams << ",\n";
	ofs << "  \"dcim_drain_cycles_new_sensitivity\": " << total_dcim_drain_sensitivity_cycles << ",\n";
	ofs << "  \"dcim_compute_only_exposed_cycles\": " << total_dcim_compute_only_exposed_cycles << ",\n";
	ofs << "  \"dcim_plus6_exposed_cycles_new_sensitivity\": " << total_dcim_plus6_exposed_cycles << ",\n";
	ofs << "  \"finalized_duplicate_packets\": " << total_finalized_duplicate_packets << ",\n";
	ofs << "  \"finalized_invalid_packets\": " << total_finalized_invalid_packets << ",\n";
	ofs << "  \"finalized_timeouts\": " << total_finalized_timeouts << ",\n";
	ofs << "  \"dcim_activation_quantized_elements_exact_once\": " << total_dcim_activation_quantized_elements << ",\n";
	ofs << "  \"dcim_dequantized_elements_exact_once\": " << total_dcim_dequantized_elements << ",\n";
	ofs << "  \"dcim_comparator_tree_passes_exact_once\": " << total_dcim_comparator_tree_passes << ",\n";
	ofs << "  \"y_dram_read_cycles\": " << total_y_dram_read_cycles << ",\n";
	ofs << "  \"y_dram_write_cycles\": " << total_y_dram_write_cycles << ",\n";
	ofs << "  \"z_dram_rw_cycles\": " << total_z_dram_rw_cycles << ",\n";
	ofs << "  \"z_sram_read_bytes\": " << total_z_sram_read_bytes << ",\n";
	ofs << "  \"z_sram_write_bytes\": " << total_z_sram_write_bytes << ",\n";
	ofs << "  \"z_sram_rw_bytes\": " << total_z_sram_rw_bytes << ",\n";
	ofs << "  \"z_sram_checksum_assertion\": "
	    << ((total_z_sram_read_bytes + total_z_sram_write_bytes == total_z_sram_rw_bytes)
	        ? "true" : "false") << ",\n";
	ofs << "  \"pe_cim_noc_contract\": {\"flit_payload_bytes\": 32, "
	    << "\"path\": \"pe_output_staging_to_dcim_input_fifo_direct_boundary\", "
	    << "\"hop_count\": null, \"hop_inference\": false},\n";
	ofs << "  \"row_detect_cycles\": " << total_row_detect_cycles << ",\n";
	ofs << "  \"row_detect_checks\": " << total_row_detect_checks << ",\n";
	ofs << "  \"row_detect_sram_reads\": " << total_row_detect_sram_reads << ",\n";
	ofs << "  \"index_bytes_read\": " << total_index_bytes_read << ",\n";
	ofs << "  \"index_bytes_write\": " << total_index_bytes_write << ",\n";
	ofs << "  \"index_update_ops\": " << total_index_update_ops << ",\n";
	ofs << "  \"index_update_payload_bytes\": " << total_index_update_payload_bytes << ",\n";
	ofs << "  \"index_update_dram_cycles\": " << total_index_update_dram_cycles << ",\n";
	ofs << "  \"index_dram_cycles\": " << total_index_dram_cycles << ",\n";
	ofs << "  \"weighted_operator_maintenance_owner\": "
	    << "\"indexer_apply_delta_a_once; payload=8B/record; cycles subset of index_dram_cycles\",\n";
	ofs << "  \"feature_bytes_write\": " << total_feature_bytes_write << ",\n";
	ofs << "  \"indexer_cam_searches\": " << total_indexer_cam_searches << ",\n";
	ofs << "  \"indexer_mshr_writes\": " << total_indexer_mshr_writes << ",\n";
	ofs << "  \"indexer_mshr_reads\": " << total_indexer_mshr_reads << ",\n";
	ofs << "  \"index_sram_hits\": " << total_index_sram_hits << ",\n";
	ofs << "  \"index_sram_misses\": " << total_index_sram_misses << ",\n";
	ofs << "  \"index_sram_fills\": " << total_index_sram_fills << ",\n";
	ofs << "  \"index_sram_evictions\": " << total_index_sram_evictions << ",\n";
	ofs << "  \"index_lookup_cycles\": " << total_index_lookup_cycles << ",\n";
	ofs << "  \"dramsim_output_dir\": \"" << (this->gcp->get_results_dir() + "/dramsim3_pe_output") << "\",\n";
	ofs << "  \"dramsim_a_load_output_dir\": \"" << (this->gcp->get_results_dir() + "/dramsim3_a_load_output") << "\",\n";
	ofs << "  \"dramsim_index_output_dir\": \"" << (this->gcp->get_results_dir() + "/dramsim3_indexer_output") << "\",\n";
	ofs << "  \"dramsim_feature_output_dir\": \"" << (this->gcp->get_results_dir() + "/dramsim3_feature_output") << "\",\n";
	ofs << "  \"dramsim_yz_output_dir\": \"" << (this->gcp->get_results_dir() + "/dramsim3_yz_output") << "\",\n";
	ofs << "  \"dramsim_cim_weight_output_dir\": \"" << (this->gcp->get_results_dir() + "/dramsim3_cim_weight_output") << "\",\n";
	ofs << "  \"rows_streamed\": " << this->streaming_rows_completed << ",\n";
	ofs << "  \"rows_returned\": " << this->streaming_outputs_completed << ",\n";
	ofs << "  \"pe_cim_noc_packets\": " << this->pe_cim_noc_packets << ",\n";
	ofs << "  \"pe_cim_noc_flits\": " << this->pe_cim_noc_flits << ",\n";
	ofs << "  \"pe_cim_noc_latency_cycles\": " << this->pe_cim_noc_latency_cycles << ",\n";
	ofs << "  \"pe_cim_noc_output_packets\": " << this->pe_cim_noc_output_packets << ",\n";
	ofs << "  \"cim_col_tile_passes\": "
	    << (this->cim_unit ? this->cim_unit->get_total_col_tile_passes() : 0) << ",\n";
	ofs << "  \"cim_col_tile_pipeline_restarts\": "
	    << (this->cim_unit ? this->cim_unit->get_col_tile_pipeline_restarts() : 0) << ",\n";
	ofs << "  \"cim_runtime_energy_pj\": " << cim_runtime_energy_pj << ",\n";
	ofs << "  \"cim_eval_energy_pj\": " << cim_eval_energy_pj << ",\n";
	ofs << "  \"cim_total_energy_split_pj\": " << cim_total_energy_split_pj << ",\n";
	ofs << "  \"cim_energy_stage_split_valid\": " << (this->multi_ts_eval_energy_valid ? "true" : "false") << ",\n";
	ofs << "  \"cim_total_energy_pj\": "
	    << cim_total_energy_split_pj << ",\n";
	ofs << "  \"cim_input_dac_energy_pj\": "
	    << (this->cim_unit ? this->cim_unit->get_input_dac_energy_pj() : 0.0) << ",\n";
	ofs << "  \"cim_multiplier_energy_pj\": "
	    << (this->cim_unit ? this->cim_unit->get_multiplier_energy_pj() : 0.0) << ",\n";
	ofs << "  \"cim_adder_tree_energy_pj\": "
	    << (this->cim_unit ? this->cim_unit->get_adder_tree_energy_pj() : 0.0) << ",\n";
	ofs << "  \"cim_comparator_energy_pj\": "
	    << (this->cim_unit ? this->cim_unit->get_comparator_energy_pj() : 0.0) << ",\n";
	ofs << "  \"cim_quantizer_energy_pj\": "
	    << (this->cim_unit ? this->cim_unit->get_quantizer_energy_pj() : 0.0) << ",\n";
	ofs << "  \"cim_dequantizer_energy_pj\": "
	    << (this->cim_unit ? this->cim_unit->get_dequantizer_energy_pj() : 0.0) << ",\n";
	ofs << "  \"cim_array_energy_pj\": "
	    << (this->cim_unit ? this->cim_unit->get_array_energy_pj() : 0.0) << ",\n";
	ofs << "  \"cim_fifo_energy_pj\": "
	    << (this->cim_unit ? this->cim_unit->get_fifo_energy_pj() : 0.0) << ",\n";
		ofs << "  \"official_profile\": \"despoina-h6-dcim-final\",\n";
		ofs << "  \"cim_runtime_architecture\": "
		    << "\"despoina-official-dcim-factorized-row-mean\",\n";
		ofs << "  \"cim_energy_scope\": \"steady_state_resident_weight_inference\",\n";
	ofs << "  \"cim_sram_energy_owner\": \"NeuroSim_SRAM_sense_precharge_array_plus_wordline_decoder\",\n";
	ofs << "  \"cim_custom_energy_owner\": \"existing_energy_dcim_Accelergy_plus_CACTI_FIFO\",\n";
		ofs << "  \"model_deployment_setup_in_official_scope\": false,\n";
		ofs << "  \"cim_energy_status\": \"OFFICIAL_SCOPE_COMPLETE\",\n";
	ofs << "  \"cim_input_dac_cycles\": "
	    << (this->cim_unit ? this->cim_unit->get_cycles_input_dac() : 0) << ",\n";
	ofs << "  \"cim_weight_load_cycles\": " << cim_weight_load_cycles << ",\n";
	ofs << "  \"cim_weight_load_dram_requests\": " << cim_weight_load_dram_requests << ",\n";
	ofs << "  \"cim_weight_load_bytes\": " << cim_weight_load_bytes << ",\n";
	ofs << "  \"cim_weight_source_fp32_bytes\": " << (this->cim_unit ? this->cim_unit->get_weight_source_fp32_bytes() : 0) << ",\n";
	ofs << "  \"cim_weight_resident_int8_bytes\": " << (this->cim_unit ? this->cim_unit->get_weight_resident_int8_bytes() : 0) << ",\n";
	ofs << "  \"cim_weight_load_events\": " << (this->cim_unit ? this->cim_unit->get_weight_load_events() : 0) << ",\n";
	ofs << "  \"cim_weight_program_tile_events\": " << (this->cim_unit ? this->cim_unit->get_weight_program_tile_events() : 0) << ",\n";

	// Per-timestep breakdown
	ofs << "  \"per_timestep\": [\n";
		for (size_t i = 0; i < this->multi_ts_results.size(); ++i) {
			const auto& res = this->multi_ts_results[i];
			const long long pe_raw_scalar_ops = res.pe_stats.total_mac_ops;
			const long long pe_fused_mac_events =
				despoina::official_energy::fused_mac_events_from_raw_scalar_ops(
					pe_raw_scalar_ops);
			const double pe_fused_mac_energy_pj =
				despoina::official_energy::fused_mac_energy_pj(pe_raw_scalar_ops);
			const double row_mean_fp32_energy_pj =
				despoina::official_energy::row_mean_scalar_energy_pj(
					res.row_mean_scalar_ops);
			std::string mode_used_str = (res.mode_used == DualModeSpMM::Mode::FULL) ? "full" :
		                            (res.mode_used == DualModeSpMM::Mode::INCREMENTAL) ? "incremental" : "scheduled";
		std::string phase_mode_used_str = phase_mode_to_string(res.phase_mode_used);
		std::string cim_mode_str = "none";
		if (res.cim_mode == CiMMode::DELTA_ONLY) {
			cim_mode_str = "delta";
		} else if (res.cim_mode == CiMMode::FULL_YNEW) {
			cim_mode_str = "full";
		} else if (res.cim_mode == CiMMode::PER_PHASE_LINEAR) {
			cim_mode_str = "linear";
		}
		ofs << "    {\n";
		ofs << "      \"timestep\": " << res.timestep << ",\n";
		ofs << "      \"mode_used\": \"" << mode_used_str << "\",\n";
		ofs << "      \"phase_mode\": \"" << phase_mode_used_str << "\",\n";
		ofs << "      \"cim_mode\": \"" << cim_mode_str << "\",\n";
		ofs << "      \"cycles\": " << res.cycles << ",\n";
		ofs << "      \"combine_cycles\": " << res.combine_cycles << ",\n";
		ofs << "      \"cim_cycles\": " << res.cim_cycles << ",\n";
		ofs << "      \"cim_stream_cycles\": " << res.cim_stream_cycles << ",\n";
		ofs << "      \"cim_exposed_cycles\": " << res.cim_exposed_cycles << ",\n";
		ofs << "      \"cim_pe_backpressure_cycles\": " << res.cim_pe_backpressure_cycles << ",\n";
		ofs << "      \"cim_input_fifo_full_rejections\": " << res.cim_input_fifo_full_rejections << ",\n";
		ofs << "      \"cim_input_fifo_full_cycles\": " << res.cim_input_fifo_full_cycles << ",\n";
		ofs << "      \"cim_output_fifo_full_cycles\": " << res.cim_output_fifo_full_cycles << ",\n";
		ofs << "      \"cim_output_fifo_full_stall_cycles\": " << res.cim_output_fifo_full_stall_cycles << ",\n";
		ofs << "      \"cim_stage3_output_block_cycles\": " << res.cim_stage3_output_block_cycles << ",\n";
		ofs << "      \"cim_inject_no_empty_lane_cycles\": " << res.cim_inject_no_empty_lane_cycles << ",\n";
		ofs << "      \"cim_active_lane_cycles\": " << res.cim_active_lane_cycles << ",\n";
		ofs << "      \"cim_active_slot_cycles\": " << res.cim_active_slot_cycles << ",\n";
		ofs << "      \"cim_input_fifo_occupancy_cycles\": " << res.cim_input_fifo_occupancy_cycles << ",\n";
		ofs << "      \"cim_output_fifo_occupancy_cycles\": " << res.cim_output_fifo_occupancy_cycles << ",\n";
		ofs << "      \"cim_max_active_lanes\": " << res.cim_max_active_lanes << ",\n";
		ofs << "      \"cim_avg_active_lanes\": "
		    << (res.cim_stream_cycles > 0
		            ? static_cast<double>(res.cim_active_lane_cycles) / static_cast<double>(res.cim_stream_cycles)
		            : 0.0)
		    << ",\n";
		ofs << "      \"cim_avg_input_fifo_occupancy\": "
		    << (res.cim_stream_cycles > 0
		            ? static_cast<double>(res.cim_input_fifo_occupancy_cycles) / static_cast<double>(res.cim_stream_cycles)
		            : 0.0)
		    << ",\n";
		ofs << "      \"cim_avg_output_fifo_occupancy\": "
		    << (res.cim_stream_cycles > 0
		            ? static_cast<double>(res.cim_output_fifo_occupancy_cycles) / static_cast<double>(res.cim_stream_cycles)
		            : 0.0)
		    << ",\n";
		ofs << "      \"cim_rows\": " << res.cim_rows << ",\n";
		ofs << "      \"finalized_rows\": " << res.finalized_rows << ",\n";
		ofs << "      \"finalized_packets\": " << res.finalized_packets << ",\n";
		ofs << "      \"dcim_scalar_macs_exact_once\": " << res.dcim_scalar_macs << ",\n";
		ofs << "      \"dcim_tile_busy_cycles_from_events\": " << res.dcim_compute_cycles << ",\n";
		ofs << "      \"tile_busy_work_cycles\": " << res.tile_busy_work_cycles << ",\n";
		ofs << "      \"causal_stream_elapsed_cycles\": " << res.causal_stream_elapsed_cycles << ",\n";
		ofs << "      \"exposed_tail_cycles\": " << res.exposed_tail_cycles << ",\n";
		ofs << "      \"dcim_streams\": " << res.dcim_streams << ",\n";
		ofs << "      \"dcim_drain_cycles_new_sensitivity\": " << res.dcim_drain_sensitivity_cycles << ",\n";
		ofs << "      \"dcim_compute_only_exposed_cycles\": " << res.dcim_compute_only_exposed_cycles << ",\n";
		ofs << "      \"dcim_plus6_exposed_cycles_new_sensitivity\": " << res.dcim_plus6_exposed_cycles << ",\n";
		ofs << "      \"finalized_duplicate_packets\": " << res.finalized_duplicate_packets << ",\n";
		ofs << "      \"finalized_invalid_packets\": " << res.finalized_invalid_packets << ",\n";
		ofs << "      \"finalized_timeouts\": " << res.finalized_timeouts << ",\n";
		ofs << "      \"dcim_activation_quantized_elements_exact_once\": " << res.dcim_activation_quantized_elements << ",\n";
		ofs << "      \"dcim_dequantized_elements_exact_once\": " << res.dcim_dequantized_elements << ",\n";
		ofs << "      \"dcim_comparator_tree_passes_exact_once\": " << res.dcim_comparator_tree_passes << ",\n";
		ofs << "      \"y_dram_read_cycles\": " << res.y_dram_read_cycles << ",\n";
		ofs << "      \"y_dram_write_cycles\": " << res.y_dram_write_cycles << ",\n";
		ofs << "      \"z_dram_rw_cycles\": " << res.z_dram_rw_cycles << ",\n";
		ofs << "      \"z_sram_read_bytes\": " << res.z_sram_read_bytes << ",\n";
		ofs << "      \"z_sram_write_bytes\": " << res.z_sram_write_bytes << ",\n";
		ofs << "      \"z_sram_rw_bytes\": " << res.z_sram_rw_bytes << ",\n";
		ofs << "      \"pe_cim_noc_flits\": " << res.arch_ctrs.pe_cim_noc_flits << ",\n";
		ofs << "      \"pe_cim_noc_bytes\": " << res.arch_ctrs.pe_cim_noc_bytes << ",\n";
		ofs << "      \"phase1\": {\"nnz\": " << res.phase1.nnz
		    << ", \"mac_ops\": " << res.phase1.mac_ops
		    << ", \"cycles\": " << res.phase1.cycles << "},\n";
		ofs << "      \"phase2\": {\"nnz\": " << res.phase2.nnz
		    << ", \"mac_ops\": " << res.phase2.mac_ops
		    << ", \"cycles\": " << res.phase2.cycles << "},\n";
			ofs << "      \"phase3\": {\"nnz\": " << res.phase3.nnz
			    << ", \"mac_ops\": " << res.phase3.mac_ops
			    << ", \"cycles\": " << res.phase3.cycles << "},\n";
			ofs << "      \"official_fp32_22nm_energy\": {"
			    << "\"pJ_per_action\": "
			    << despoina::official_energy::kFp32ActionEnergy22nmPj
			    << ", \"main_pe_raw_scalar_ops\": " << pe_raw_scalar_ops
			    << ", \"main_pe_fused_mac_events\": " << pe_fused_mac_events
			    << ", \"main_pe_fused_mac_energy_pj\": " << pe_fused_mac_energy_pj
			    << ", \"row_mean_scalar_ops\": " << res.row_mean_scalar_ops
			    << ", \"row_mean_scalar_energy_pj\": " << row_mean_fp32_energy_pj
			    << ", \"total_pj\": "
			    << (pe_fused_mac_energy_pj + row_mean_fp32_energy_pj)
			    << "},\n";
		ofs << "      \"dynamic_row_mean\": {"
		    << "\"enabled\": " << (res.row_mean_enabled ? "true" : "false")
		    << ", \"self_loop_convention\": \"none_reject_explicit\""
		    << ", \"degree_axis\": \"destination_csr_row\""
		    << ", \"scaled_rows\": " << res.row_mean_scaled_rows
		    << ", \"graph_changed_degree_rows\": " << res.row_mean_graph_changed_degree_rows
		    << ", \"changed_degree_scaled_rows\": " << res.row_mean_changed_degree_rows
		    << ", \"scale_elements\": " << res.row_mean_scale_elements
		    << ", \"residual_correction_elements\": " << res.row_mean_residual_correction_elements
		    << ", \"scalar_ops\": " << res.row_mean_scalar_ops
		    << ", \"stream_delay_work_cycles\": " << res.row_mean_stream_delay_cycles
		    << ", \"degree_read_events\": " << res.row_mean_degree_read_events
		    << ", \"degree_write_events\": " << res.row_mean_degree_write_events
		    << ", \"degree_read_bytes\": " << res.row_mean_degree_read_bytes
		    << ", \"degree_write_bytes\": " << res.row_mean_degree_write_bytes
		    << ", \"reciprocal_events\": " << res.row_mean_reciprocal_events
		    << ", \"native_pe_cycles\": " << res.factorized_native_pe_cycles
		    << ", \"owner_cycles_no_overlap\": " << res.factorized_native_owner_cycles
		    << ", \"state_capacity_bytes\": " << res.factorized_state_capacity_bytes
		    << ", \"state_sram_bytes\": " << res.factorized_state_sram_bytes
		    << ", \"state_hbm_bytes\": " << res.factorized_state_hbm_bytes
		    << ", \"state_hbm_read_bytes\": " << res.factorized_state_hbm_read_bytes
		    << ", \"state_hbm_write_bytes\": " << res.factorized_state_hbm_write_bytes
		    << ", \"state_hbm_bursts_64b\": " << res.factorized_state_hbm_bursts
		    << ", \"state_hbm_cycles\": " << res.factorized_state_hbm_cycles
		    << ", \"old_y_snapshot_bytes\": " << res.factorized_old_y_snapshot_bytes
		    << ", \"old_y_sram_bytes\": " << res.factorized_old_y_sram_bytes
		    << ", \"old_y_hbm_write_bytes\": " << res.factorized_old_y_hbm_write_bytes
		    << ", \"old_y_hbm_read_bytes\": " << res.factorized_old_y_hbm_read_bytes
		    << ", \"old_y_hbm_bursts_64b\": " << res.factorized_old_y_hbm_bursts
		    << ", \"old_y_hbm_cycles\": " << res.factorized_old_y_hbm_cycles
			    << ", \"fp32_scalar_action_energy_pj\": "
			    << despoina::official_energy::kFp32ActionEnergy22nmPj
			    << ", \"fp32_scalar_energy_pj\": " << row_mean_fp32_energy_pj
		    << "},\n";
		ofs << "      \"paper_faithful_state\": {\"npf_h_read_rows\": " << res.npf_h_read_rows
		    << ", \"npf_h_write_rows\": " << res.npf_h_write_rows
		    << ", \"npf_h_hbm_read_bytes\": " << res.npf_h_hbm_read_bytes
		    << ", \"npf_h_hbm_write_bytes\": " << res.npf_h_hbm_write_bytes
		    << ", \"npf_h_hbm_cycles\": " << res.npf_h_hbm_cycles
		    << ", \"z_hbm_rw_bytes\": " << res.z_hbm_rw_bytes
		    << ", \"z_dram_rw_cycles\": " << res.z_dram_rw_cycles << "},\n";
		ofs << "      \"row_detect_cycles\": " << res.row_detect_cycles << ",\n";
		// Depth-selectivity ablation: per-layer propagation tracking
		if (!res.per_layer_propagating_rows.empty()) {
			ofs << "      \"per_layer_propagating_rows\": [";
			for (size_t li = 0; li < res.per_layer_propagating_rows.size(); li++) {
				ofs << res.per_layer_propagating_rows[li];
				if (li + 1 < res.per_layer_propagating_rows.size()) ofs << ", ";
			}
			ofs << "],\n";
		}
		if (!res.per_layer_mac_ops.empty()) {
			ofs << "      \"per_layer_mac_ops\": [";
			for (size_t li = 0; li < res.per_layer_mac_ops.size(); li++) {
				ofs << res.per_layer_mac_ops[li];
				if (li + 1 < res.per_layer_mac_ops.size()) ofs << ", ";
			}
			ofs << "],\n";
		}
		ofs << "      \"row_detect_checks\": " << res.row_detect_checks << ",\n";
		ofs << "      \"row_detect_sram_reads\": " << res.row_detect_sram_reads << ",\n";
		ofs << "      \"accuracy\": {\n";
		ofs << "        \"executed\": " << (res.accuracy.executed ? "true" : "false") << ",\n";
		ofs << "        \"valid\": " << (res.accuracy.valid ? "true" : "false") << ",\n";
		ofs << "        \"pass\": " << (res.accuracy.pass ? "true" : "false") << ",\n";
		ofs << "        \"pass_hitrate\": " << (res.accuracy.pass_hitrate ? "true" : "false") << ",\n";
		ofs << "        \"rmse\": " << res.accuracy.rmse << ",\n";
		ofs << "        \"cosine_similarity\": " << res.accuracy.cosine_similarity << ",\n";
		ofs << "        \"max_abs_error\": " << res.accuracy.max_abs_error << ",\n";
		ofs << "        \"mean_abs_error\": " << res.accuracy.mean_abs_error << ",\n";
		ofs << "        \"rel_error_mean\": " << res.accuracy.rel_error_mean << ",\n";
		ofs << "        \"rel_error_max\": " << res.accuracy.rel_error_max << ",\n";
		ofs << "        \"hit_rate_rel_1pct\": " << res.accuracy.hit_rate_rel_1pct << ",\n";
		ofs << "        \"hit_rate_rel_2pct\": " << res.accuracy.hit_rate_rel_2pct << ",\n";
		ofs << "        \"hit_rate_rel_5pct\": " << res.accuracy.hit_rate_rel_5pct << ",\n";
		ofs << "        \"rows_compared\": " << res.accuracy.rows_compared << ",\n";
		ofs << "        \"elements_compared\": " << res.accuracy.elements_compared << ",\n";
		ofs << "        \"hw_complete\": " << (res.accuracy.hw_complete ? "true" : "false") << ",\n";
		ofs << "        \"hw_timeout\": " << (res.accuracy.hw_timeout ? "true" : "false") << ",\n";
		ofs << "        \"hw_expected_rows\": " << res.accuracy.hw_expected_rows << ",\n";
		ofs << "        \"hw_received_unique_rows\": " << res.accuracy.hw_received_unique_rows << ",\n";
		ofs << "        \"hw_duplicate_packets\": " << res.accuracy.hw_duplicate_packets << ",\n";
		ofs << "        \"hw_invalid_packets\": " << res.accuracy.hw_invalid_packets << ",\n";
		ofs << "        \"rmse_tolerance\": " << res.accuracy.rmse_tolerance << ",\n";
		ofs << "        \"cosine_tolerance\": " << res.accuracy.cosine_tolerance << ",\n";
		ofs << "        \"rel_error_epsilon\": " << res.accuracy.rel_error_epsilon << ",\n";
		ofs << "        \"abs_error_tolerance\": " << res.accuracy.abs_error_tolerance << ",\n";
		ofs << "        \"hitrate_pass_threshold\": " << res.accuracy.hitrate_pass_threshold << "\n";
		ofs << "      },\n";
		ofs << "      \"y_accuracy\": {\n";
		ofs << "        \"valid\": " << (res.y_accuracy.valid ? "true" : "false") << ",\n";
		ofs << "        \"exact\": " << (res.y_accuracy.exact ? "true" : "false") << ",\n";
		ofs << "        \"rmse\": " << res.y_accuracy.rmse << ",\n";
		ofs << "        \"cosine_similarity\": " << res.y_accuracy.cosine_similarity << "\n";
		ofs << "      },\n";
		ofs << "      \"e2e_accuracy\": {\n";
		ofs << "        \"executed\": " << (res.e2e_exported ? "true" : "false") << ",\n";
		ofs << "        \"valid\": " << (res.e2e_accuracy.valid ? "true" : "false") << ",\n";
		ofs << "        \"pass\": " << (res.e2e_accuracy.pass ? "true" : "false") << ",\n";
		ofs << "        \"pass_hitrate\": " << (res.e2e_accuracy.pass_hitrate ? "true" : "false") << ",\n";
		ofs << "        \"rmse\": " << res.e2e_accuracy.rmse << ",\n";
		ofs << "        \"cosine_similarity\": " << res.e2e_accuracy.cosine_similarity << ",\n";
		ofs << "        \"max_abs_error\": " << res.e2e_accuracy.max_abs_error << ",\n";
		ofs << "        \"mean_abs_error\": " << res.e2e_accuracy.mean_abs_error << ",\n";
		ofs << "        \"rel_error_mean\": " << res.e2e_accuracy.rel_error_mean << ",\n";
		ofs << "        \"rel_error_max\": " << res.e2e_accuracy.rel_error_max << ",\n";
		ofs << "        \"hit_rate_rel_1pct\": " << res.e2e_accuracy.hit_rate_rel_1pct << ",\n";
		ofs << "        \"hit_rate_rel_2pct\": " << res.e2e_accuracy.hit_rate_rel_2pct << ",\n";
		ofs << "        \"hit_rate_rel_5pct\": " << res.e2e_accuracy.hit_rate_rel_5pct << ",\n";
		ofs << "        \"rows_compared\": " << res.e2e_accuracy.rows_compared << ",\n";
		ofs << "        \"elements_compared\": " << res.e2e_accuracy.elements_compared << ",\n";
		ofs << "        \"hw_complete\": " << (res.e2e_accuracy.hw_complete ? "true" : "false") << ",\n";
		ofs << "        \"hw_timeout\": " << (res.e2e_accuracy.hw_timeout ? "true" : "false") << "\n";
		ofs << "      },\n";
		ofs << "      \"e2e_exported\": " << (res.e2e_exported ? "true" : "false") << ",\n";
		ofs << "      \"e2e_export_dir\": \"" << res.e2e_export_dir << "\",\n";
		ofs << "      \"mac_ops\": " << res.pe_stats.total_mac_ops << ",\n";
		ofs << "      \"cache_hits\": " << res.pe_stats.f_cache_hits << ",\n";
		ofs << "      \"cache_misses\": " << res.pe_stats.f_cache_misses << ",\n";
		ofs << "      \"l1_f_cache_hits\": " << res.pe_stats.l1_f_cache_hits << ",\n";
		ofs << "      \"l2_shared_hits\": " << res.pe_stats.l2_shared_hits << ",\n";
		ofs << "      \"l1_bank_conflict_stalls\": " << res.pe_stats.l1_bank_conflict_stalls << ",\n";
		ofs << "      \"l2_bank_conflict_stalls\": " << res.pe_stats.l2_bank_conflict_stalls << ",\n";
		ofs << "      \"noc_steal_count\": " << res.pe_stats.noc_steal_count << ",\n";
		ofs << "      \"noc_steal_latency_total_cycles\": " << res.pe_stats.noc_steal_latency_total_cycles << ",\n";
		ofs << "      \"noc_steal_avg_hops\": " << res.pe_stats.noc_steal_avg_hops << ",\n";
		ofs << "      \"a_load_dram_requests\": " << res.pe_stats.a_load_dram_requests << ",\n";
		ofs << "      \"a_load_dram_latency_sum\": " << res.pe_stats.a_load_dram_latency_sum << ",\n";
		ofs << "      \"dram_requests\": " << res.pe_stats.dram_requests << ",\n";
		ofs << "      \"row_buffer_hits\": " << res.pe_stats.dram_row_buffer_hits << ",\n";
		ofs << "      \"dram_fills\": " << res.pe_stats.dram_fills << ",\n";
		ofs << "      \"index_bytes_read\": " << res.indexer_ctrs.index_bytes_read << ",\n";
		ofs << "      \"index_bytes_write\": " << res.indexer_ctrs.index_bytes_write << ",\n";
		ofs << "      \"index_update_ops\": " << res.indexer_ctrs.index_update_ops << ",\n";
		ofs << "      \"index_update_payload_bytes\": " << res.indexer_ctrs.index_update_payload_bytes << ",\n";
		ofs << "      \"index_update_dram_cycles\": " << res.indexer_ctrs.index_update_dram_cycles << ",\n";
		ofs << "      \"index_dram_cycles\": " << res.indexer_ctrs.index_dram_cycles << ",\n";
		ofs << "      \"feature_bytes_write\": " << res.indexer_ctrs.feature_bytes_write << ",\n";
		ofs << "      \"indexer_cam_searches\": " << res.indexer_ctrs.cam_searches << ",\n";
		ofs << "      \"indexer_mshr_writes\": " << res.indexer_ctrs.mshr_writes << ",\n";
		ofs << "      \"indexer_mshr_reads\": " << res.indexer_ctrs.mshr_reads << ",\n";
		ofs << "      \"index_sram_hits\": " << res.indexer_ctrs.index_sram_hits << ",\n";
		ofs << "      \"index_sram_misses\": " << res.indexer_ctrs.index_sram_misses << ",\n";
		ofs << "      \"index_sram_fills\": " << res.indexer_ctrs.index_sram_fills << ",\n";
		ofs << "      \"index_sram_evictions\": " << res.indexer_ctrs.index_sram_evictions << ",\n";
		ofs << "      \"index_lookup_cycles\": " << res.indexer_ctrs.index_lookup_cycles << "\n";
		ofs << "    }" << (i < this->multi_ts_results.size() - 1 ? "," : "") << "\n";
	}
	ofs << "  ]\n";
	ofs << "}\n";

	// Console output
	cout << "\n[Multi-TS] Results: " << this->multi_ts_results.size() << " timesteps" << endl;
	for (const auto& res : this->multi_ts_results) {
		std::string mode_str = (res.mode_used == DualModeSpMM::Mode::FULL) ? "Full" :
		                       (res.mode_used == DualModeSpMM::Mode::INCREMENTAL) ? "Incr" : "Sched";
		cout << "[Multi-TS] t=" << res.timestep
		     << " mode=" << mode_str
		     << " cycles=" << res.cycles
		     << " macs=" << res.pe_stats.total_mac_ops << endl;
	}
	cout << "[Multi-TS] Total cycles: " << total_cycles << endl;
	cout << "[Multi-TS] Stats written to " << out_path << endl;
	this->write_exact_hbm_sidecar();
	this->write_arch_ablation_stats();
}

void NeuraTile::write_arch_ablation_stats()
{
	if (this->multi_ts_results.empty()) return;
	if (this->gcp == nullptr) return;

	const std::string out_path = this->gcp->get_results_dir() + "/arch_ablation_stats.json";
	std::ofstream ofs(out_path);
	if (!ofs.is_open()) {
		cerr << "[Despoina] Failed to open " << out_path << endl;
		return;
	}

	DualModeSpMM::StepResult::ArchAblationCounters total_arch{};
	long long total_cycles = 0;
	double cim_total_energy_pj = 0.0;
	for (const auto& res : this->multi_ts_results) {
		total_cycles += res.cycles;
		total_arch.add(res.arch_ctrs);
	}
	if (this->multi_ts_eval_energy_valid) {
		cim_total_energy_pj = this->multi_ts_runtime_cim_energy_pj + this->multi_ts_eval_cim_energy_pj;
	} else if (this->cim_unit != nullptr) {
		cim_total_energy_pj = this->cim_unit->get_total_energy_pj();
	}

	const std::string arch_variant =
		(this->dygnn_controller != nullptr)
			? this->dygnn_controller->get_arch_variant_string()
			: ((this->gcp->arg_parse != nullptr)
			       ? this->gcp->arg_parse->get_arch_variant()
			       : default_arch_variant_string());
	const std::string effective_backend_policy =
		(this->dygnn_controller != nullptr)
			? this->dygnn_controller->get_effective_backend_policy_string()
			: "full_despoina_legacy_reference";
	const std::string effective_indexer_mode =
		(this->dygnn_controller != nullptr)
			? this->dygnn_controller->get_effective_indexer_mode_string()
			: ((this->gcp->arg_parse != nullptr) ? this->gcp->arg_parse->get_indexer_mode() : "indexstore");
	const std::string dataset =
		(this->gcp->arg_parse != nullptr) ? this->gcp->arg_parse->get_dataset() : "";
	const std::string program_root =
		(this->gcp->arg_parse != nullptr) ? this->gcp->arg_parse->get_program_root() : "";
	const std::string dataset_root =
		(this->gcp->arg_parse != nullptr) ? this->gcp->arg_parse->get_dataset_root() : "";
	const int timesteps_requested =
		(this->gcp->arg_parse != nullptr) ? this->gcp->arg_parse->get_timesteps() : 1;
	const int num_layers =
		(this->gcp->arg_parse != nullptr) ? this->gcp->arg_parse->get_num_layers() : 1;
	const double propagation_tau =
		(this->gcp->arg_parse != nullptr) ? this->gcp->arg_parse->get_propagation_tau() : 0.0;
	const std::string scope_mode =
		(this->gcp->arg_parse != nullptr) ? this->gcp->arg_parse->get_scope_mode() : "tau";
	const std::string layer_norm_mode =
		(this->gcp->arg_parse != nullptr) ? this->gcp->arg_parse->get_layer_norm_mode() : "auto";
	const bool layer_norm_effective =
		(this->gcp->arg_parse != nullptr) ? this->gcp->arg_parse->get_effective_layer_norm_enabled() : false;
	const int cim_fifo_depth =
		(this->cim_unit != nullptr) ? this->cim_unit->get_fifo_depth()
		                            : ((this->gcp->arg_parse != nullptr) ? this->gcp->arg_parse->get_cim_fifo_depth() : 16);

	auto write_arch_fields = [](std::ostream& out,
	                            const DualModeSpMM::StepResult::ArchAblationCounters& c,
	                            const std::string& indent,
	                            bool trailing_comma) {
		out << indent << "\"behavior_forced\": " << (c.behavior_forced ? "true" : "false") << ",\n";
		out << indent << "\"dcim_enabled\": " << (c.dcim_enabled ? "true" : "false") << ",\n";
		out << indent << "\"streaming_enabled\": " << (c.streaming_enabled ? "true" : "false") << ",\n";
		out << indent << "\"fullscan_enabled\": " << (c.fullscan_enabled ? "true" : "false") << ",\n";
		out << indent << "\"indexstore_enabled\": " << (c.indexstore_enabled ? "true" : "false") << ",\n";
		out << indent << "\"y_materialize_read_bytes\": " << c.y_materialize_read_bytes << ",\n";
		out << indent << "\"y_materialize_write_bytes\": " << c.y_materialize_write_bytes << ",\n";
		out << indent << "\"z_materialize_read_bytes\": " << c.z_materialize_read_bytes << ",\n";
		out << indent << "\"z_materialize_write_bytes\": " << c.z_materialize_write_bytes << ",\n";
		out << indent << "\"materialization_eliminated_bytes\": " << c.materialization_eliminated_bytes << ",\n";
		out << indent << "\"fullscan_bytes_read\": " << c.fullscan_bytes_read << ",\n";
		out << indent << "\"indexstore_bytes_read\": " << c.indexstore_bytes_read << ",\n";
		out << indent << "\"indexstore_bytes_write\": " << c.indexstore_bytes_write << ",\n";
		out << indent << "\"index_lookup_ops\": " << c.index_lookup_ops << ",\n";
		out << indent << "\"index_lookup_cycles\": " << c.index_lookup_cycles << ",\n";
		out << indent << "\"index_bytes_read\": " << c.index_bytes_read << ",\n";
		out << indent << "\"index_bytes_write\": " << c.index_bytes_write << ",\n";
		out << indent << "\"crb_bytes_read\": " << c.crb_bytes_read << ",\n";
		out << indent << "\"crb_bytes_write\": " << c.crb_bytes_write << ",\n";
		out << indent << "\"crb_popcount_cycles\": " << c.crb_popcount_cycles << ",\n";
		out << indent << "\"crb_bytes_read_estimated\": " << c.crb_bytes_read_estimated << ",\n";
		out << indent << "\"crb_bytes_write_estimated\": " << c.crb_bytes_write_estimated << ",\n";
		out << indent << "\"crb_popcount_cycles_estimated\": " << c.crb_popcount_cycles_estimated << ",\n";
		out << indent << "\"row_detect_cycles\": " << c.row_detect_cycles << ",\n";
		out << indent << "\"row_detect_checks\": " << c.row_detect_checks << ",\n";
		out << indent << "\"row_detect_sram_reads\": " << c.row_detect_sram_reads << ",\n";
		out << indent << "\"dcim_cycles\": " << c.dcim_cycles << ",\n";
		out << indent << "\"dcim_exposed_cycles\": " << c.dcim_exposed_cycles << ",\n";
		out << indent << "\"dcim_stream_cycles\": " << c.dcim_stream_cycles << ",\n";
		out << indent << "\"tile_busy_work_cycles\": " << c.tile_busy_work_cycles << ",\n";
		out << indent << "\"causal_stream_elapsed_cycles\": " << c.causal_stream_elapsed_cycles << ",\n";
		out << indent << "\"exposed_tail_cycles\": " << c.exposed_tail_cycles << ",\n";
		out << indent << "\"cim_rows\": " << c.cim_rows << ",\n";
		out << indent << "\"cim_active_lane_cycles\": " << c.cim_active_lane_cycles << ",\n";
		out << indent << "\"cim_active_slot_cycles\": " << c.cim_active_slot_cycles << ",\n";
		out << indent << "\"cim_avg_active_lanes\": "
		    << (c.dcim_stream_cycles > 0
		            ? static_cast<double>(c.cim_active_lane_cycles) / static_cast<double>(c.dcim_stream_cycles)
		            : 0.0)
		    << ",\n";
		out << indent << "\"cim_avg_active_slots\": "
		    << (c.dcim_stream_cycles > 0
		            ? static_cast<double>(c.cim_active_slot_cycles) / static_cast<double>(c.dcim_stream_cycles)
		            : 0.0)
		    << ",\n";
		out << indent << "\"cim_input_fifo_occupancy_cycles\": " << c.cim_input_fifo_occupancy_cycles << ",\n";
		out << indent << "\"cim_output_fifo_occupancy_cycles\": " << c.cim_output_fifo_occupancy_cycles << ",\n";
		out << indent << "\"cim_avg_input_fifo_occupancy\": "
		    << (c.dcim_stream_cycles > 0
		            ? static_cast<double>(c.cim_input_fifo_occupancy_cycles) / static_cast<double>(c.dcim_stream_cycles)
		            : 0.0)
		    << ",\n";
		out << indent << "\"cim_avg_output_fifo_occupancy\": "
		    << (c.dcim_stream_cycles > 0
		            ? static_cast<double>(c.cim_output_fifo_occupancy_cycles) / static_cast<double>(c.dcim_stream_cycles)
		            : 0.0)
		    << ",\n";
		out << indent << "\"cim_pe_backpressure_cycles\": " << c.cim_pe_backpressure_cycles << ",\n";
		out << indent << "\"stream_fifo_bytes\": " << c.stream_fifo_bytes << ",\n";
		out << indent << "\"stream_fifo_bytes_estimated\": " << c.stream_fifo_bytes_estimated << ",\n";
		out << indent << "\"digital_stream_estimated_cycles\": " << c.digital_stream_estimated_cycles << ",\n";
		out << indent << "\"digital_stream_saved_cycles\": " << c.digital_stream_saved_cycles << ",\n";
		out << indent << "\"digital_stream_fifo_bytes_estimated\": " << c.digital_stream_fifo_bytes_estimated << ",\n";
		out << indent << "\"digital_stream_measured_cycles\": " << c.digital_stream_measured_cycles << ",\n";
		out << indent << "\"digital_stream_compute_cycles\": " << c.digital_stream_compute_cycles << ",\n";
		out << indent << "\"digital_stream_overlap_cycles\": " << c.digital_stream_overlap_cycles << ",\n";
		out << indent << "\"digital_stream_fifo_bytes\": " << c.digital_stream_fifo_bytes << ",\n";
		out << indent << "\"digital_stream_input_fifo_occupancy_cycles\": " << c.digital_stream_input_fifo_occupancy_cycles << ",\n";
		out << indent << "\"digital_stream_output_fifo_occupancy_cycles\": " << c.digital_stream_output_fifo_occupancy_cycles << ",\n";
		out << indent << "\"digital_stream_backpressure_cycles\": " << c.digital_stream_backpressure_cycles << ",\n";
		out << indent << "\"digital_stream_rows\": " << c.digital_stream_rows << ",\n";
		out << indent << "\"digital_stream_mac_ops\": " << c.digital_stream_mac_ops << ",\n";
		out << indent << "\"digital_stream_tile_rows\": " << c.digital_stream_tile_rows << ",\n";
		out << indent << "\"digital_stream_tile_cols\": " << c.digital_stream_tile_cols << ",\n";
		out << indent << "\"digital_stream_lanes\": " << c.digital_stream_lanes << ",\n";
		out << indent << "\"digital_stream_bit_step_issue_events\": " << c.digital_stream_bit_step_issue_events << ",\n";
		out << indent << "\"digital_stream_bit_step_completion_events\": " << c.digital_stream_bit_step_completion_events << ",\n";
		out << indent << "\"digital_stream_full_int8_energy_charge_events\": " << c.digital_stream_full_int8_energy_charge_events << ",\n";
		out << indent << "\"digital_stream_tile_dispatch_events\": " << c.digital_stream_tile_dispatch_events << ",\n";
		out << indent << "\"digital_stream_weight_program_cycles\": " << c.digital_stream_weight_program_cycles << ",\n";
		out << indent << "\"digital_stream_weight_program_tile_events\": " << c.digital_stream_weight_program_tile_events << ",\n";
		out << indent << "\"digital_stream_weight_source_fp32_bytes\": " << c.digital_stream_weight_source_fp32_bytes << ",\n";
		out << indent << "\"digital_stream_weight_programmed_int8_bytes\": " << c.digital_stream_weight_programmed_int8_bytes << ",\n";
		out << indent << "\"digital_stream_weight_bank_prefetch_events\": " << c.digital_stream_weight_bank_prefetch_events << ",\n";
		out << indent << "\"digital_stream_weight_bank_swap_events\": " << c.digital_stream_weight_bank_swap_events << ",\n";
		out << indent << "\"digital_stream_weight_bank_exposed_stall_cycles\": " << c.digital_stream_weight_bank_exposed_stall_cycles << ",\n";
		out << indent << "\"digital_stream_overlap_sensitivity_finish_cycles\": " << c.digital_stream_overlap_sensitivity_finish_cycles << ",\n";
		out << indent << "\"digital_stream_overlap_sensitivity_exposed_cycles\": " << c.digital_stream_overlap_sensitivity_exposed_cycles << ",\n";
		out << indent << "\"digital_stream_event_trace_xor\": " << c.digital_stream_event_trace_xor << ",\n";
		out << indent << "\"digital_stream_first_bit_step_issue_cycle\": " << c.digital_stream_first_bit_step_issue_cycle << ",\n";
		out << indent << "\"digital_stream_last_bit_step_completion_cycle\": " << c.digital_stream_last_bit_step_completion_cycle << ",\n";
		out << indent << "\"pe_cim_noc_flits\": " << c.pe_cim_noc_flits << ",\n";
		out << indent << "\"pe_cim_noc_bytes\": " << c.pe_cim_noc_bytes << ",\n";
		out << indent << "\"pe_cim_noc_flit_payload_bytes\": 32,\n";
		out << indent << "\"pe_cim_noc_path\": \"pe_output_staging_to_dcim_input_fifo_direct_boundary\",\n";
		out << indent << "\"pe_cim_noc_hop_inference\": false,\n";
		out << indent << "\"z_sram_read_bytes\": " << c.z_sram_read_bytes << ",\n";
		out << indent << "\"z_sram_write_bytes\": " << c.z_sram_write_bytes << ",\n";
		out << indent << "\"z_sram_rw_bytes\": " << c.z_sram_rw_bytes << ",\n";
		out << indent << "\"systolic_cycles\": " << c.systolic_cycles << ",\n";
		out << indent << "\"systolic_compute_cycles\": " << c.systolic_compute_cycles << ",\n";
		out << indent << "\"systolic_pack_cycles\": " << c.systolic_pack_cycles << ",\n";
		out << indent << "\"systolic_materialize_cycles\": " << c.systolic_materialize_cycles << ",\n";
		out << indent << "\"systolic_active_rows\": " << c.systolic_active_rows << ",\n";
		out << indent << "\"systolic_mac_ops\": " << c.systolic_mac_ops << ",\n";
		out << indent << "\"systolic_pack_bytes\": " << c.systolic_pack_bytes << ",\n";
		out << indent << "\"systolic_materialize_bytes\": " << c.systolic_materialize_bytes << ",\n";
		out << indent << "\"systolic_tile_rows\": " << c.systolic_tile_rows << ",\n";
		out << indent << "\"systolic_tile_cols\": " << c.systolic_tile_cols
		    << (trailing_comma ? ",\n" : "\n");
	};

	const bool h0_estimator = (arch_variant == "H0_systolic_scoped");
	const bool h3_digital_stream_measured =
		(arch_variant == "H3_sparsefe_index_digital_stream");
	const long long reported_total_cycles =
		(h0_estimator && total_arch.systolic_cycles > 0)
			? static_cast<long long>(total_arch.systolic_cycles)
			: total_cycles;

	ofs << "{\n";
	ofs << "  \"arch_variant\": \"" << arch_variant << "\",\n";
	ofs << "  \"effective_backend_policy\": \"" << effective_backend_policy << "\",\n";
	ofs << "  \"effective_indexer_mode\": \"" << effective_indexer_mode << "\",\n";
	if (h0_estimator) {
		ofs << "  \"estimator_label\": \"analytical_scoped_systolic_estimator\",\n";
		ofs << "  \"estimator_assumptions\": {\n";
		ofs << "    \"measured_path\": false,\n";
		ofs << "    \"tile_rows\": " << total_arch.systolic_tile_rows << ",\n";
		ofs << "    \"tile_cols\": " << total_arch.systolic_tile_cols << ",\n";
		ofs << "    \"dataflow\": \"weight_stationary\",\n";
		ofs << "    \"active_rows_source\": \"affected_rows_or_full_rows_per_timestep; higher_layers_use_propagated_rows\",\n";
		ofs << "    \"mac_ops_formula\": \"active_rows * feature_dim * hidden_dim\",\n";
		ofs << "    \"compute_cycles_formula\": \"ceil(active_rows/tile_rows) * ceil(hidden_dim/tile_cols) * (feature_dim + tile_rows + tile_cols - 2)\",\n";
		ofs << "    \"pack_cycles_source\": \"analytical_dram_transactions_for_active_rows_feature_dim_read_write\",\n";
		ofs << "    \"materialize_cycles_source\": \"analytical_dram_transactions_for_active_rows_hidden_dim_write\"\n";
		ofs << "  },\n";
	}
	if (h3_digital_stream_measured) {
		ofs << "  \"estimator_label\": \"measured_digital_stream_backend\",\n";
		ofs << "  \"estimator_assumptions\": {\n";
		ofs << "    \"measured_path\": true,\n";
		ofs << "    \"simulator_path\": \"SparseFE + IndexStore producer feeding measured digital streaming backend counters\",\n";
		ofs << "    \"dcim_backend\": false,\n";
		ofs << "    \"streaming_model\": \"finite input/output FIFO with producer-consumer overlap and backpressure\",\n";
		ofs << "    \"digital_backend\": \"16x16 weight-stationary digital tile model, no analog/DCIM\",\n";
		ofs << "    \"reported_cycles_formula\": \"sum of simulator StepResult total_cycles; H3 is not post-hoc replaced in this writer\",\n";
		ofs << "    \"remaining_approximation\": \"digital dense backend is an internal cycle micro-model driven by measured SparseFE/IndexStore row counts and PE producer cycles\"\n";
		ofs << "  },\n";
	}
	ofs << "  \"counter_sources\": {\n";
	ofs << "    \"index_and_row_detect\": \"measured_existing_step_counters\",\n";
	ofs << "    \"y_z_materialization_bytes\": \"tensor_size_accounting\",\n";
	ofs << "    \"dcim_pipeline\": \"measured_existing_cim_step_counters_when_path_active\",\n";
	ofs << "    \"stream_fifo_bytes\": \"measured_pe_output_egress_or_estimated_row_tensor_bytes\",\n";
	ofs << "    \"digital_stream_backend\": \"simulator_internal_measured_cycle_accounted_no_dcim_backend_for_H3\",\n";
	ofs << "    \"pe_cim_noc\": \"measured_neuratile_streaming_noc_counters\",\n";
	ofs << "    \"crb_estimates\": \"row_bitmap_size_per_active_phase\"\n";
	ofs << "  },\n";
	ofs << "  \"workload_settings\": {\n";
	ofs << "    \"dataset\": \"" << dataset << "\",\n";
	ofs << "    \"program_root\": \"" << program_root << "\",\n";
	ofs << "    \"dataset_root\": \"" << dataset_root << "\",\n";
	ofs << "    \"timesteps_requested\": " << timesteps_requested << ",\n";
	ofs << "    \"num_layers\": " << num_layers << ",\n";
	ofs << "    \"propagation_tau\": " << propagation_tau << ",\n";
	ofs << "    \"scope_mode\": \"" << scope_mode << "\",\n";
	ofs << "    \"layer_norm_mode\": \"" << layer_norm_mode << "\",\n";
	ofs << "    \"layer_norm_effective\": " << (layer_norm_effective ? "true" : "false") << "\n";
	ofs << "  },\n";
	ofs << "  \"timesteps\": " << this->multi_ts_results.size() << ",\n";
	ofs << "  \"total_cycles\": " << reported_total_cycles << ",\n";
	ofs << "  \"simulator_total_cycles\": " << total_cycles << ",\n";
	ofs << "  \"cim_fifo_depth\": " << cim_fifo_depth << ",\n";
	ofs << "  \"hagg_staging_model\": " << pe_hagg_staging::kModel << ",\n";
	ofs << "  \"hagg_staging_bytes\": " << pe_hagg_staging::kBytes << ",\n";
	ofs << "  \"hagg_staging_line_bytes\": " << pe_hagg_staging::kLineBytes << ",\n";
	ofs << "  \"hagg_staging_capacity_rows\": " << pe_hagg_staging::kCapacityRows << ",\n";
	ofs << "  \"hagg_staging_max_occupancy_rows\": " << this->pe_array_spmm->get_staging_max_occupancy_rows() << ",\n";
	ofs << "  \"hagg_staging_full_stall_cycles\": " << this->pe_array_spmm->get_staging_full_stall_cycles() << ",\n";
	ofs << "  \"cim_total_energy_pj\": " << cim_total_energy_pj << ",\n";
	write_arch_fields(ofs, total_arch, "  ", true);
	ofs << "  \"pe_cim_noc_packets\": " << this->pe_cim_noc_packets << ",\n";
	ofs << "  \"pe_cim_noc_flits\": " << this->pe_cim_noc_flits << ",\n";
	ofs << "  \"pe_cim_noc_latency_cycles\": " << this->pe_cim_noc_latency_cycles << ",\n";
	ofs << "  \"rows_streamed\": " << this->streaming_rows_completed << ",\n";
	ofs << "  \"rows_returned\": " << this->streaming_outputs_completed << ",\n";
	ofs << "  \"per_timestep\": [\n";
	for (size_t i = 0; i < this->multi_ts_results.size(); ++i) {
		const auto& res = this->multi_ts_results[i];
		ofs << "    {\n";
		ofs << "      \"timestep\": " << res.timestep << ",\n";
		ofs << "      \"cycles\": " << res.cycles << ",\n";
		write_arch_fields(ofs, res.arch_ctrs, "      ", false);
		ofs << "    }" << (i + 1 < this->multi_ts_results.size() ? "," : "") << "\n";
	}
	ofs << "  ]\n";
	ofs << "}\n";
	cout << "[Despoina] Architecture ablation stats written to " << out_path << endl;
}

#endif // USE_PE_ARRAY_MODE

#endif // ENABLE_PE_ARRAY
