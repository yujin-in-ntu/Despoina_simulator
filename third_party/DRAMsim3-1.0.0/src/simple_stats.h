#ifndef __SIMPLE_STATS_
#define __SIMPLE_STATS_

#include <fstream>
#include <string>
#include <unordered_map>
#include <vector>

#include "command_energy_snapshot.h"
#include "configuration.h"
#include "json.hpp"

namespace dramsim3 {

class SimpleStats {
   public:
    SimpleStats(const Config& config, int channel_id);
    // incrementing counter
    void Increment(const std::string name) {
        epoch_counters_[name] += 1;
        // Keep a lifetime monotonic counter at the actual command issue
        // owner.  This intentionally survives reporting resets so callers can
        // form exact adjacent snapshot deltas without perturbing timing.
        if (name == "num_act_cmds") {
            command_energy_.act_commands += 1;
            command_energy_.act_energy_pj += config_.act_energy_inc;
        } else if (name == "num_read_cmds") {
            command_energy_.read_commands += 1;
            command_energy_.read_energy_pj += config_.read_energy_inc;
        } else if (name == "num_write_cmds") {
            command_energy_.write_commands += 1;
            command_energy_.write_energy_pj += config_.write_energy_inc;
        } else if (name == "num_ref_cmds") {
            command_energy_.refresh_commands += 1;
            command_energy_.refresh_energy_pj += config_.ref_energy_inc;
        } else if (name == "num_refb_cmds") {
            command_energy_.refresh_bank_commands += 1;
            command_energy_.refresh_bank_energy_pj += config_.refb_energy_inc;
        }
    }

    CommandEnergySnapshot GetCommandEnergySnapshot() const {
        return command_energy_;
    }

    // incrementing for vec counter
    void IncrementVec(const std::string name, int pos) {
        epoch_vec_counters_[name][pos] += 1;
    }

    // increment vec counter by number
    void IncrementVecBy(const std::string name, int pos, int num) {
        epoch_vec_counters_[name][pos] += num;
    }

    // add historgram value
    void AddValue(const std::string name, const int value);

    // return per rank background energy
    double RankBackgroundEnergy(const int r) const;

    // Epoch update
    void PrintEpochStats();

    // Final statas output
    void PrintFinalStats();

    // Reset (usually after one phase of simulation)
    void Reset();

   private:
    using VecStat = std::unordered_map<std::string, std::vector<uint64_t> >;
    using HistoCount = std::unordered_map<int, uint64_t>;
    using Json = nlohmann::json;
    void InitStat(std::string name, std::string stat_type,
                  std::string description);
    void InitVecStat(std::string name, std::string stat_type,
                     std::string description, std::string part_name,
                     int vec_len);
    void InitHistoStat(std::string name, std::string description, int start_val,
                       int end_val, int num_bins);

    void UpdateCounters();
    void UpdateHistoBins();
    void UpdatePrints(bool epoch);
    double GetHistoAvg(const HistoCount& histo_counts) const;
    std::string GetTextHeader(bool is_final) const;
    void UpdateEpochStats();
    void UpdateFinalStats();

    const Config& config_;
    int channel_id_;

    // map names to descriptions
    std::unordered_map<std::string, std::string> header_descs_;

    // counter stats, indexed by their name
    std::unordered_map<std::string, uint64_t> counters_;
    std::unordered_map<std::string, uint64_t> epoch_counters_;

    // vectored counter stats, first indexed by name then by index
    VecStat vec_counters_;
    VecStat epoch_vec_counters_;

    // NOTE: doubles_ vec_doubles_ and calculated_ are basically one time
    // placeholders after each epoch they store the value for that epoch
    // (different from the counters) and in the end updated to the overall value
    std::unordered_map<std::string, double> doubles_;

    std::unordered_map<std::string, std::vector<double> > vec_doubles_;

    // calculated stats, similar to double, but not the same
    std::unordered_map<std::string, double> calculated_;

    // histogram stats
    std::unordered_map<std::string, std::vector<std::string> > histo_headers_;

    std::unordered_map<std::string, std::pair<int, int> > histo_bounds_;
    std::unordered_map<std::string, int> bin_widths_;
    std::unordered_map<std::string, HistoCount> histo_counts_;
    std::unordered_map<std::string, HistoCount> epoch_histo_counts_;
    VecStat histo_bins_;
    VecStat epoch_histo_bins_;

    // outputs
    Json j_data_;
    std::vector<std::pair<std::string, std::string> > print_pairs_;
    CommandEnergySnapshot command_energy_;
};

}  // namespace dramsim3
#endif
