#ifndef __COMMAND_ENERGY_SNAPSHOT_H
#define __COMMAND_ENERGY_SNAPSHOT_H

#include <cstdint>

namespace dramsim3 {

// Monotonic, side-effect-free command-energy view.  Snapshot reads neither
// advance an epoch nor mutate simulator state.
struct CommandEnergySnapshot {
    uint64_t act_commands = 0;
    uint64_t read_commands = 0;
    uint64_t write_commands = 0;
    uint64_t refresh_commands = 0;
    uint64_t refresh_bank_commands = 0;
    double act_energy_pj = 0.0;
    double read_energy_pj = 0.0;
    double write_energy_pj = 0.0;
    double refresh_energy_pj = 0.0;
    double refresh_bank_energy_pj = 0.0;

    double total_command_energy_pj() const {
        return act_energy_pj + read_energy_pj + write_energy_pj +
               refresh_energy_pj + refresh_bank_energy_pj;
    }
};

}  // namespace dramsim3

#endif
