// SPDX-License-Identifier: CC-BY-4.0
// Derived from NeuraChip/NeuraSim (DOI: 10.5281/zenodo.10896280); modified for Despoina and this standalone release in 2026.
// Created by Kaustubh Shivdikar
//

#include "../lib/functions.h"
#include "../lib/debugger/debugger.h"
#include "../lib/neuracore/neuracore.h"
#include "../lib/counters/globalcounters.h"
#include "../lib/neuratile/neuratile.h"
#include "../lib/config.h"
#include "../lib/argparse/argparse.h"

using namespace std;

#ifndef PE_SHARED_F_CACHE_ENTRIES
#define PE_SHARED_F_CACHE_ENTRIES 8192
#endif
#ifndef PE_L1_MRU_ENTRIES
#define PE_L1_MRU_ENTRIES 32
#endif
#ifndef PE_SHARED_F_CACHE_BANKS
#define PE_SHARED_F_CACHE_BANKS 64
#endif

// Global Variables

// ************ MAIN FUNCTION ************

int main(int argc, char *argv[])
{
	ArgParse arg_parse(argc, argv);
	cerr << "[CACHE_CONFIG_JSON] {\"pe_shared_f_cache_entries\":" << PE_SHARED_F_CACHE_ENTRIES
	     << ",\"pe_l1_mru_entries\":" << PE_L1_MRU_ENTRIES
	     << ",\"pe_shared_f_cache_banks\":" << PE_SHARED_F_CACHE_BANKS << "}" << endl;
	cerr << "[HAGG_STAGING_JSON] {\"model\":" << pe_hagg_staging::kModel
	     << ",\"bounded\":" << (pe_hagg_staging::kBounded ? "true" : "false")
	     << ",\"bytes\":" << pe_hagg_staging::kBytes
	     << ",\"line_bytes\":" << pe_hagg_staging::kLineBytes
	     << ",\"capacity_rows\":" << pe_hagg_staging::kCapacityRows
	     << ",\"functional_shadow\":\"fp32\",\"physical_packet\":\"aligned_int8_row_scale_metadata\"}" << endl;

	// For SPAM protection, comment out to disable SPAM protection.
	// host_notify(&arg_parse);

	greetings();
	// cout << "Experiment Name: " << arg_parse.get_exp_name() << endl;
	// return 0;

	// arg_parse.print();

	// Set omp threads
	// omp_set_num_threads(32);

	NeuraTile *neura_tile;
	neura_tile = sim_init(&arg_parse);

	// launch_visualizer_script(neura_tile);
	// return 0;

	sanity_check(neura_tile);

	// return 0;
	// statistics(neura_tile);
	
	run_sim(neura_tile);

	// cout << "Main Statistics" << endl;
	statistics(neura_tile);

	// cout << "Post Sim" << endl;
	post_sim(neura_tile);

	// cout << "Launch Visualizer" << endl;
	// launch_visualizer_script(neura_tile);

	// cout << "Cleanup" << endl;
	sim_cleanup(neura_tile);



	cout << "\n\e[1;31mProgram End\e[0m\n\n\n";
	return SUCCESS;
}
