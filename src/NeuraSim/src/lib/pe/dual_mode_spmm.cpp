// SPDX-License-Identifier: Apache-2.0
#include "dual_mode_spmm.h"
#include "l1_p3_support.h"
#include "../official_energy_contract.h"
#include "../official_accuracy_gate.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <deque>
#include <fstream>
#include <functional>
#include <iostream>
#include <limits>
#include <random>
#include <set>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

#include "../counters/globalcounters.h"
#include "../neuracim/neuracim.h"

#ifndef PE_PHASE3_KEEP_SHARED_F_CACHE
#define PE_PHASE3_KEEP_SHARED_F_CACHE 1
#endif

#ifndef PE_PHASE3_KEEP_MAX_ROWS
#define PE_PHASE3_KEEP_MAX_ROWS 32
#endif

#ifndef PE_PHASE3_KEEP_MAX_NNZ
#define PE_PHASE3_KEEP_MAX_NNZ 128
#endif

#ifdef _WIN32
#include <direct.h>
#define CREATE_DIR(path) _mkdir((path))
#else
#include <sys/stat.h>
#define CREATE_DIR(path) mkdir((path), 0777)
#endif

namespace {

constexpr long long kCombineThroughputPerCycle = 256LL * 8LL;
constexpr int kCombineReadLatencyCycles = 1;
constexpr int kCombineComputeLatencyCycles = 1;
constexpr int kCombineWriteLatencyCycles = 1;
constexpr long long kBytesPerElement = 4LL;
constexpr long long kDramTransactionBytes = 64LL;
constexpr long long kDramAvgCyclesPerTransaction = 20LL;
constexpr int kFallbackCimTiles = 16;
constexpr int kFallbackCimPipelineLatency = 8;
constexpr uint64_t kYDramBaseAddr = 0x30000000ULL;
constexpr uint64_t kZDramBaseAddr = 0x40000000ULL;
constexpr uint64_t kFactorizedStateDramBaseAddr = 0x50000000ULL;
constexpr uint64_t kFactorizedOldYDramBaseAddr = 0x60000000ULL;
constexpr uint64_t kZStateDramBaseAddr = 0x80000000ULL;  // + (layer-1) * n * d_out * 4
constexpr uint64_t kHStateDramBaseAddr = 0xC0000000ULL;  // + (layer-1) * n * d * 4
constexpr uint64_t kBoundStateDramBaseAddr = 0x200000000ULL;  // + (layer-1) * n * 4
constexpr int kYZDramMaxRequestsPerCycle = 8;
constexpr int kYZDramMshrSize = 64;
constexpr long long kAccuracyWatchdogScale = 64LL;
constexpr long long kAccuracyWatchdogBase = 4096LL;
constexpr int kH0SystolicTileRows = 16;
constexpr int kH0SystolicTileCols = 16;
constexpr int kDigitalStreamTileRows = 16;
constexpr int kDigitalStreamTileCols = 16;
constexpr int kDigitalStreamInputFifoRows = 64;
constexpr int kDigitalStreamOutputFifoRows = 64;
constexpr int kDigitalStreamFifoBytesPerCycle = 512;
// F-Cache capacity computed at runtime: PE_SHARED_F_CACHE_ENTRIES × d × 4 bytes.
// Do NOT hardcode — entry count is set via compile-time -DPE_SHARED_F_CACHE_ENTRIES.
#ifndef PE_SHARED_F_CACHE_ENTRIES
#define PE_SHARED_F_CACHE_ENTRIES 8192
#endif

void load_required_fp32_weights(const std::string& path,
                                int d_in,
                                int d_out,
                                const char* layer,
                                std::vector<float>* destination) {
    if (destination == nullptr || d_in <= 0 || d_out <= 0) {
        throw std::invalid_argument(std::string(layer) + " clean-weight shape is invalid");
    }
    if (path.empty()) {
        throw std::runtime_error(std::string(layer) + " clean-weight path is required; random fallback is forbidden");
    }
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input.is_open()) {
        throw std::runtime_error(std::string(layer) + " clean-weight file cannot be opened: " + path);
    }
    const size_t elements = static_cast<size_t>(d_in) * static_cast<size_t>(d_out);
    const std::streamoff expected = static_cast<std::streamoff>(elements * sizeof(float));
    const std::streamoff actual = input.tellg();
    if (actual != expected) {
        throw std::runtime_error(std::string(layer) + " clean-weight size mismatch: " + path);
    }
    input.seekg(0, std::ios::beg);
    destination->assign(elements, 0.0f);
    input.read(reinterpret_cast<char*>(destination->data()), expected);
    if (input.gcount() != expected || input.fail()) {
        throw std::runtime_error(std::string(layer) + " clean-weight read failed: " + path);
    }
    if (std::any_of(destination->begin(), destination->end(),
                    [](float value) { return !std::isfinite(value); })) {
        throw std::runtime_error(std::string(layer) + " clean-weight payload contains non-finite float32: " + path);
    }
    std::cout << "[DyGNN] " << layer << " clean weights loaded from " << path
              << " (" << d_in << "x" << d_out << ")\n";
}

// ---------------------------------------------------------------------------
// Row Detection Scanner (RDS) hardware parameters.
// Scans CSR row_ptr array stored in on-chip SRAM (IndexStore or A-load buffer)
// to identify non-zero output rows for CiM feeding.
//
// Hardware: 8-wide integer comparator array (matches PE SIMD width).
// Each cycle reads row_ptr entries from SRAM and compares pairs.
// row_ptr stored as int32 (4B each); one 64B cache line = 16 entries.
// ---------------------------------------------------------------------------
constexpr int kRowDetectScanWidth = 8;         // Comparisons per cycle
constexpr int kRowDetectPipelineLatency = 1;   // SRAM read fill latency (cycles)
constexpr int kRowPtrEntriesPerCacheLine = 16; // 64B / 4B = 16 int32 entries

long long estimate_row_detect_cycles(int n) {
    if (n <= 0) return 0;
    return static_cast<long long>((n + kRowDetectScanWidth - 1) / kRowDetectScanWidth)
         + kRowDetectPipelineLatency;
}

int estimate_row_detect_sram_reads(int n) {
    if (n <= 0) return 0;
    // Need n+1 row_ptr entries; each 64B cache line holds 16 entries
    return (n + 1 + kRowPtrEntriesPerCacheLine - 1) / kRowPtrEntriesPerCacheLine;
}

// ---------------------------------------------------------------------------
// CSR-based non-zero row detection (hardware-friendly, O(n) integer checks).
// Replaces the previous O(n*d) dense-output scanning approach.
// A row i has non-zero SpMM output iff its input CSR row is non-empty
// (row_ptr[i] != row_ptr[i+1]).  This is a conservative superset -- exact
// cancellations are theoretically possible but negligible in practice.
// ---------------------------------------------------------------------------

#if !DESPOINA_V3_FUNCTIONAL_DCIM
int count_csr_nonzero_rows(const WorkStealingScheduler::CSRMatrix& csr) {
    int count = 0;
    for (int i = 0; i < csr.n; i++) {
        if (csr.row_ptr[i] != csr.row_ptr[i + 1]) {
            count++;
        }
    }
    return count;
}
#endif

std::vector<int> collect_csr_affected_rows(
    const WorkStealingScheduler::CSRMatrix& delta_a_csr,
    const WorkStealingScheduler::CSRMatrix& a_sub) {
    const auto valid_row_ptr = [](const WorkStealingScheduler::CSRMatrix& csr) {
        return csr.n >= 0 &&
            csr.row_ptr.size() == static_cast<size_t>(csr.n) + 1U;
    };
    if (!valid_row_ptr(delta_a_csr) || !valid_row_ptr(a_sub) ||
        delta_a_csr.n != a_sub.n) {
        throw std::runtime_error(
            "affected-row owner requires equal, initialized CSR row domains");
    }
    std::vector<int> rows;
    const int n = delta_a_csr.n;
    rows.reserve(static_cast<size_t>(n));
    for (int i = 0; i < n; i++) {
        if (delta_a_csr.row_ptr[i] != delta_a_csr.row_ptr[i + 1] ||
            a_sub.row_ptr[i] != a_sub.row_ptr[i + 1]) {
            rows.push_back(i);
        }
    }
    return rows;
}

long long estimate_dram_rw_cycles(int rows, int d_out) {
    if (rows <= 0 || d_out <= 0) {
        return 0;
    }
    const long long bytes =
        static_cast<long long>(rows) * static_cast<long long>(d_out) * kBytesPerElement * 2LL;
    const long long txns = (bytes + kDramTransactionBytes - 1LL) / kDramTransactionBytes;
    return txns * kDramAvgCyclesPerTransaction;
}

long long estimate_dram_read_cycles(int rows, int d_out) {
    if (rows <= 0 || d_out <= 0) {
        return 0;
    }
    const long long bytes =
        static_cast<long long>(rows) * static_cast<long long>(d_out) * kBytesPerElement;
    const long long txns = (bytes + kDramTransactionBytes - 1LL) / kDramTransactionBytes;
    return txns * kDramAvgCyclesPerTransaction;
}

long long estimate_dram_write_cycles(int rows, int d_out) {
    if (rows <= 0 || d_out <= 0) {
        return 0;
    }
    const long long bytes =
        static_cast<long long>(rows) * static_cast<long long>(d_out) * kBytesPerElement;
    const long long txns = (bytes + kDramTransactionBytes - 1LL) / kDramTransactionBytes;
    return txns * kDramAvgCyclesPerTransaction;
}

long long estimate_cim_cycles(NeuraCiM* cim, int rows) {
    if (rows <= 0) {
        return 0;
    }
    if (cim != nullptr) {
        const int d_in = cim->get_d_in();
        const int d_out = cim->get_d_out();
        if (d_in > 0 && d_out > 0) {
            const long long analytical_fallback =
                static_cast<long long>(cim->compute_cycles_analytical(rows));

            if (!cim->start_matmul(rows, d_in, d_out, true)) {
                return analytical_fallback;
            }

            long long cycles = 0;
            int next_row = 0;
            int outputs = 0;
            const std::vector<float> dummy_features(static_cast<size_t>(d_in), 1.0f);
            const long long max_cycles =
                std::max(analytical_fallback * 32LL + 1024LL, static_cast<long long>(rows) * 8LL + 1024LL);

            while (outputs < rows && cycles < max_cycles) {
                while (next_row < rows && cim->can_accept_input()) {
                    CiMInputPacket pkt;
                    pkt.row_id = next_row;
                    pkt.features = dummy_features;
                    pkt.is_incremental = true;
                    if (!cim->push_input(pkt)) {
                        break;
                    }
                    next_row++;
                }

                cim->tick();
                cycles++;

                CiMOutputPacket out;
                while (cim->pop_output(out)) {
                    outputs++;
                }
            }

            if (outputs >= rows) {
                return cycles;
            }
            return analytical_fallback;
        }
    }
    const long long rows_per_tile =
        (static_cast<long long>(rows) + kFallbackCimTiles - 1LL) / kFallbackCimTiles;
    return rows_per_tile + kFallbackCimPipelineLatency - 1LL;
}

NeuraCiM::PipelineCounters diff_pipeline_counters(
    const NeuraCiM::PipelineCounters& after,
    const NeuraCiM::PipelineCounters& before) {
    NeuraCiM::PipelineCounters delta;
    delta.input_fifo_full_rejections =
        after.input_fifo_full_rejections - before.input_fifo_full_rejections;
    delta.input_fifo_full_cycles =
        after.input_fifo_full_cycles - before.input_fifo_full_cycles;
    delta.output_fifo_full_cycles =
        after.output_fifo_full_cycles - before.output_fifo_full_cycles;
    delta.output_fifo_full_stall_cycles =
        after.output_fifo_full_stall_cycles - before.output_fifo_full_stall_cycles;
    delta.stage3_output_block_cycles =
        after.stage3_output_block_cycles - before.stage3_output_block_cycles;
    delta.inject_no_empty_lane_cycles =
        after.inject_no_empty_lane_cycles - before.inject_no_empty_lane_cycles;
    delta.active_lane_cycles =
        after.active_lane_cycles - before.active_lane_cycles;
    delta.active_slot_cycles =
        after.active_slot_cycles - before.active_slot_cycles;
    delta.input_fifo_occupancy_cycles =
        after.input_fifo_occupancy_cycles - before.input_fifo_occupancy_cycles;
    delta.output_fifo_occupancy_cycles =
        after.output_fifo_occupancy_cycles - before.output_fifo_occupancy_cycles;
    delta.max_active_lanes = after.max_active_lanes;
    return delta;
}

long long simulate_combine_cycles(long long work_items) {
    if (work_items <= 0) {
        return 0;
    }

    std::deque<long long> read_pipe(kCombineReadLatencyCycles, 0LL);
    std::deque<long long> compute_pipe(kCombineComputeLatencyCycles, 0LL);
    std::deque<long long> write_pipe(kCombineWriteLatencyCycles, 0LL);

    long long issued = 0;
    long long retired = 0;
    long long cycles = 0;

    while (retired < work_items) {
        // Retire completed writeback work.
        retired += write_pipe.back();
        write_pipe.pop_back();
        write_pipe.push_front(0LL);

        // Advance compute -> write stage.
        const long long ready_from_compute = compute_pipe.back();
        compute_pipe.pop_back();
        compute_pipe.push_front(0LL);
        write_pipe.front() += ready_from_compute;

        // Advance read -> compute stage.
        const long long ready_from_read = read_pipe.back();
        read_pipe.pop_back();
        read_pipe.push_front(0LL);
        compute_pipe.front() += ready_from_read;

        // Issue new combine work into read stage.
        if (issued < work_items) {
            const long long to_issue =
                std::min(kCombineThroughputPerCycle, work_items - issued);
            read_pipe.front() += to_issue;
            issued += to_issue;
        }

        cycles++;
    }

    return cycles;
}

#if !DESPOINA_V3_FUNCTIONAL_DCIM
long long finalize_full_step_cycles(const PEArraySpMM::Stats& pe_stats,
                                    const DeltaFAwareIndexer::Counters& indexer_ctrs,
                                    long long y_dram_read_cycles,
                                    long long y_dram_write_cycles,
                                    long long cim_cycles,
                                    long long z_dram_rw_cycles,
                                    long long* cim_exposed_cycles) {
    const long long pe_path =
        static_cast<long long>(pe_stats.total_cycles) +
        indexer_ctrs.index_dram_cycles +
        indexer_ctrs.feature_dram_cycles +
        y_dram_read_cycles +
        y_dram_write_cycles;

    long long exposed = 0;
    if (cim_cycles > 0) {
        // Full-mode CiM can start consuming rows as PE produces Y rows.
        // Only the CiM tail that outlives PE remains on the critical path.
        exposed = std::max(0LL, cim_cycles - static_cast<long long>(pe_stats.total_cycles));
    }
    if (cim_exposed_cycles != nullptr) {
        *cim_exposed_cycles = exposed;
    }

    return std::max(pe_path + exposed, z_dram_rw_cycles);
}
#endif

long long finalize_causal_dcim_step_cycles(
    const PEArraySpMM::Stats& pe_stats,
    const DeltaFAwareIndexer::Counters& indexer_ctrs,
    long long y_dram_read_cycles,
    long long y_dram_write_cycles,
    despoina_v3::ExposedTailCycles exposed_tail,
    long long z_dram_rw_cycles) {
    const long long signed_prefix =
        static_cast<long long>(pe_stats.total_cycles) +
        indexer_ctrs.index_dram_cycles +
        indexer_ctrs.feature_dram_cycles +
        y_dram_read_cycles +
        y_dram_write_cycles;
    if (signed_prefix < 0 || z_dram_rw_cycles < 0) {
        throw std::runtime_error("negative causal cycle component");
    }
    const uint64_t critical = despoina_v3::add_exposed_tail_once(
        static_cast<uint64_t>(signed_prefix), exposed_tail);
    return static_cast<long long>(std::max<uint64_t>(
        critical, static_cast<uint64_t>(z_dram_rw_cycles)));
}

}  // namespace

WorkStealingScheduler::CSRMatrix
DualModeSpMM::fullscan_extract(const WorkStealingScheduler::CSRMatrix& a_new,
                               const std::vector<int>& changed_nodes) const {
    WorkStealingScheduler::CSRMatrix out;
    out.n = a_new.n;
    out.row_ptr.assign(a_new.n + 1, 0);

    std::vector<char> is_changed(static_cast<size_t>(a_new.n), 0);
    for (size_t i = 0; i < changed_nodes.size(); i++) {
        int j = changed_nodes[i];
        if (j >= 0 && j < a_new.n) is_changed[static_cast<size_t>(j)] = 1;
    }

    // First pass: count nnz per row
    int nnz = 0;
    for (int i = 0; i < a_new.n; i++) {
        out.row_ptr[i] = nnz;
        int start = a_new.row_ptr[i];
        int end = a_new.row_ptr[i + 1];
        for (int p = start; p < end; p++) {
            int j = a_new.col_idx[p];
            if (j >= 0 && j < a_new.n && is_changed[static_cast<size_t>(j)]) {
                nnz++;
            }
        }
    }
    out.row_ptr[a_new.n] = nnz;
    out.nnz = nnz;
    out.col_idx.resize(nnz);
    out.values.resize(nnz);

    // Second pass: fill
    int ptr = 0;
    for (int i = 0; i < a_new.n; i++) {
        int start = a_new.row_ptr[i];
        int end = a_new.row_ptr[i + 1];
        for (int p = start; p < end; p++) {
            int j = a_new.col_idx[p];
            if (j >= 0 && j < a_new.n && is_changed[static_cast<size_t>(j)]) {
                out.col_idx[ptr] = j;
                out.values[ptr] = a_new.values[p];
                ptr++;
            }
        }
    }
    return out;
}

DualModeSpMM::DualModeSpMM(GlobalCounters* gcp, PEArraySpMM* pe_array)
    : gcp_(gcp), pe_(pe_array) {
}

DualModeSpMM::~DualModeSpMM() {
#ifdef ENABLE_DRAMSIM3_PE
    if (yz_dram_ != nullptr) {
        delete yz_dram_;
        yz_dram_ = nullptr;
    }
#endif
}

#ifdef ENABLE_DRAMSIM3_PE
void DualModeSpMM::ensure_yz_dramsim3_enabled() {
    if (yz_dram_ != nullptr) {
        return;
    }
    if (gcp_ == nullptr) {
        return;
    }

    std::string cfg = "./ktb_dram_configs/ktb_HBM2_8Gb_x128.ini";
    if (gcp_ != nullptr) {
        cfg = gcp_->get_dram_config_path();
    }
    yz_dram_output_dir_ = gcp_->get_results_dir() + "/dramsim3_yz_output";
    CREATE_DIR(yz_dram_output_dir_.c_str());

    yz_dram_ = dramsim3::GetMemorySystem(
        cfg,
        yz_dram_output_dir_,
        std::bind(&DualModeSpMM::yz_dram_read_cb, this, std::placeholders::_1),
        std::bind(&DualModeSpMM::yz_dram_write_cb, this, std::placeholders::_1));
    
    // Calculate clock ratio: NC clock (1.0 ns) / DRAM tCK
    constexpr double NC_CLOCK_PERIOD_NS = 1.0;
    double dram_tck_ns = (gcp_ != nullptr) ? gcp_->get_memory_tck_ns() : 1.0;
    yz_dram_clock_ratio_ = NC_CLOCK_PERIOD_NS / dram_tck_ns;
    yz_dram_tick_accumulator_ = 0.0;
}

void DualModeSpMM::flush_yz_dramsim3_stats() {
    if (yz_dram_ != nullptr) {
        yz_dram_->PrintStats();
    }
}

void DualModeSpMM::yz_dram_read_cb(uint64_t addr) {
    std::unordered_map<uint64_t, int>::iterator it = yz_inflight_.find(addr);
    if (it == yz_inflight_.end()) {
        return;
    }
    it->second -= 1;
    if (it->second <= 0) {
        yz_inflight_.erase(it);
    }
}

void DualModeSpMM::yz_dram_write_cb(uint64_t addr) {
    yz_dram_read_cb(addr);
}

long long DualModeSpMM::issue_yz_cacheline_batch(const std::vector<uint64_t>& cachelines, bool is_write) {
    if (cachelines.empty()) {
        return 0;
    }
    if (yz_dram_ == nullptr) {
        return 0;
    }

    long long nc_cycles = 0;  // NC cycles (not DRAMsim3 cycles)
    size_t next_to_issue = 0;
    yz_inflight_.clear();
    
    // Helper lambda for clock-ratio-aware tick
    auto tick_yz_dram = [this, &nc_cycles]() {
        yz_dram_tick_accumulator_ += yz_dram_clock_ratio_;
        while (yz_dram_tick_accumulator_ >= 1.0) {
            yz_dram_->ClockTick();
            yz_dram_tick_accumulator_ -= 1.0;
        }
        nc_cycles++;
    };

    while (next_to_issue < cachelines.size() || !yz_inflight_.empty()) {
        tick_yz_dram();

        int issued_this_cycle = 0;
        while (next_to_issue < cachelines.size() &&
               issued_this_cycle < kYZDramMaxRequestsPerCycle) {
            if (yz_inflight_.size() >= static_cast<size_t>(kYZDramMshrSize)) {
                break;
            }

            const uint64_t addr = cachelines[next_to_issue];
            if (!yz_dram_->WillAcceptTransaction(addr, is_write)) {
                break;
            }
            yz_dram_->AddTransaction(addr, is_write);
            yz_inflight_[addr] += 1;
            issued_this_cycle++;
            next_to_issue++;
        }
    }
    return nc_cycles;
}

long long DualModeSpMM::issue_yz_all_rows(uint64_t base_addr, int rows, int dim, bool is_write) {
    if (rows <= 0 || dim <= 0) {
        return 0;
    }
    const uint64_t total_bytes =
        static_cast<uint64_t>(rows) * static_cast<uint64_t>(dim) * static_cast<uint64_t>(kBytesPerElement);
    const uint64_t line = static_cast<uint64_t>(kDramTransactionBytes);
    const uint64_t first = (base_addr / line) * line;
    const uint64_t last = ((base_addr + total_bytes - 1ULL) / line) * line;

    std::vector<uint64_t> cachelines;
    cachelines.reserve(static_cast<size_t>((last - first) / line + 1ULL));
    for (uint64_t addr = first; addr <= last; addr += line) {
        cachelines.push_back(addr);
    }
    return issue_yz_cacheline_batch(cachelines, is_write);
}

long long DualModeSpMM::issue_yz_selected_rows(uint64_t base_addr,
                                               const std::vector<int>& rows,
                                               int dim,
                                               bool is_write) {
    if (rows.empty() || dim <= 0) {
        return 0;
    }
    const uint64_t line = static_cast<uint64_t>(kDramTransactionBytes);
    const uint64_t row_bytes =
        static_cast<uint64_t>(dim) * static_cast<uint64_t>(kBytesPerElement);

    std::set<uint64_t> unique_lines;
    for (size_t i = 0; i < rows.size(); i++) {
        const int row = rows[i];
        if (row < 0) {
            continue;
        }
        const uint64_t row_addr = base_addr + static_cast<uint64_t>(row) * row_bytes;
        const uint64_t first = (row_addr / line) * line;
        const uint64_t last = ((row_addr + row_bytes - 1ULL) / line) * line;
        for (uint64_t addr = first; addr <= last; addr += line) {
            unique_lines.insert(addr);
        }
    }

    std::vector<uint64_t> cachelines(unique_lines.begin(), unique_lines.end());
    return issue_yz_cacheline_batch(cachelines, is_write);
}
#endif

DualModeSpMM::Mode DualModeSpMM::parse_mode(const std::string& s) {
    if (s == "full") return Mode::FULL;
    if (s == "incremental") return Mode::INCREMENTAL;
    if (s == "scheduled") return Mode::SCHEDULED;
    return Mode::FULL;
}

CiMMode DualModeSpMM::parse_cim_mode(const std::string& s) {
    if (s == "none") return CiMMode::NONE;
    if (s == "delta") return CiMMode::DELTA_ONLY;
    if (s == "full") return CiMMode::FULL_YNEW;
    if (s == "linear") return CiMMode::PER_PHASE_LINEAR;
    return CiMMode::NONE;
}

PhaseMode DualModeSpMM::parse_phase_mode(const std::string& s) {
    if (s == "three_phase") return PhaseMode::THREE_PHASE;
    if (s == "edge_only") return PhaseMode::EDGE_ONLY;
    if (s == "two_phase") return PhaseMode::TWO_PHASE;
    if (s == "anew_full") return PhaseMode::ANEW_FULL;
    return PhaseMode::THREE_PHASE;
}

const char* DualModeSpMM::phase_mode_to_string(PhaseMode mode) {
    switch (mode) {
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
}

bool DualModeSpMM::arch_forces_sparse_materialized() const {
    return arch_variant_ == ArchVariant::H1_SPARSEFE_DIGITALBE ||
           arch_variant_ == ArchVariant::H2_SPARSEFE_INDEX_DIGITALBE;
}

bool DualModeSpMM::arch_uses_systolic_estimator() const {
    return arch_variant_ == ArchVariant::H0_SYSTOLIC_SCOPED;
}

bool DualModeSpMM::arch_uses_digital_stream_backend() const {
    return arch_variant_ == ArchVariant::H3_SPARSEFE_INDEX_DIGITAL_STREAM;
}

bool DualModeSpMM::arch_collects_dcim_counters() const {
    return arch_variant_ == ArchVariant::H4_SPARSEFE_INDEX_DCIM ||
           arch_variant_ == ArchVariant::H5_SPARSEFE_INDEX_DCIM_STREAM ||
           arch_variant_ == ArchVariant::H6_FULL_DESPOINA;
}

CiMMode DualModeSpMM::effective_cim_mode_for_arch() const {
    if (arch_forces_sparse_materialized() || arch_uses_digital_stream_backend()) {
        return CiMMode::NONE;
    }
    if (arch_variant_ == ArchVariant::H4_SPARSEFE_INDEX_DCIM) {
        return CiMMode::DELTA_ONLY;
    }
    if (arch_variant_ == ArchVariant::H5_SPARSEFE_INDEX_DCIM_STREAM) {
        return CiMMode::PER_PHASE_LINEAR;
    }
    return cim_mode_;
}

std::string DualModeSpMM::effective_indexer_mode_for_arch() const {
    if (arch_variant_ == ArchVariant::H1_SPARSEFE_DIGITALBE) {
        return "fullscan";
    }
    if (arch_variant_ == ArchVariant::H2_SPARSEFE_INDEX_DIGITALBE) {
        return "indexstore";
    }
    if (arch_variant_ == ArchVariant::H3_SPARSEFE_INDEX_DIGITAL_STREAM) {
        return "indexstore";
    }
    if (arch_variant_ == ArchVariant::H4_SPARSEFE_INDEX_DCIM ||
        arch_variant_ == ArchVariant::H5_SPARSEFE_INDEX_DCIM_STREAM) {
        return "indexstore";
    }
    return indexer_mode_;
}

const char* DualModeSpMM::get_effective_indexer_mode_string() const {
    if (arch_variant_ == ArchVariant::H1_SPARSEFE_DIGITALBE) {
        return "fullscan";
    }
    if (arch_variant_ == ArchVariant::H2_SPARSEFE_INDEX_DIGITALBE) {
        return "indexstore";
    }
    if (arch_variant_ == ArchVariant::H3_SPARSEFE_INDEX_DIGITAL_STREAM) {
        return "indexstore";
    }
    if (arch_variant_ == ArchVariant::H4_SPARSEFE_INDEX_DCIM ||
        arch_variant_ == ArchVariant::H5_SPARSEFE_INDEX_DCIM_STREAM) {
        return "indexstore";
    }
    return indexer_mode_.c_str();
}

const char* DualModeSpMM::get_effective_backend_policy_string() const {
    switch (arch_variant_) {
        case ArchVariant::H1_SPARSEFE_DIGITALBE:
            return "sparse_pe_fullscan_materialized_no_dcim";
        case ArchVariant::H2_SPARSEFE_INDEX_DIGITALBE:
            return "sparse_pe_indexstore_materialized_no_dcim";
        case ArchVariant::H3_SPARSEFE_INDEX_DIGITAL_STREAM:
            return "sparse_pe_indexstore_digital_stream_measured_no_dcim";
        case ArchVariant::H6_FULL_DESPOINA:
            return "full_despoina_legacy_reference";
        case ArchVariant::H0_SYSTOLIC_SCOPED:
            return "analytical_scoped_systolic_estimator";
        case ArchVariant::H4_SPARSEFE_INDEX_DCIM:
            return "dcim_no_stream_indexstore_active_y_materialized";
        case ArchVariant::H5_SPARSEFE_INDEX_DCIM_STREAM:
            return "dcim_stream_indexstore_phase_linear";
    }
    return "full_despoina_legacy_reference";
}

DualModeSpMM::DigitalStreamStats DualModeSpMM::simulate_digital_stream_backend(
    long long producer_cycles,
    int rows,
    int d_in,
    int d_out,
    const DigitalBackendConfig& input_cfg) const {
    DigitalBackendConfig cfg = input_cfg;
    if (cfg.tile_rows <= 0) cfg.tile_rows = kDigitalStreamTileRows;
    if (cfg.tile_cols <= 0) cfg.tile_cols = kDigitalStreamTileCols;
    if (cfg.input_fifo_rows <= 0) cfg.input_fifo_rows = kDigitalStreamInputFifoRows;
    if (cfg.output_fifo_rows <= 0) cfg.output_fifo_rows = kDigitalStreamOutputFifoRows;
    if (cfg.ingress_bytes_per_cycle <= 0) cfg.ingress_bytes_per_cycle = kDigitalStreamFifoBytesPerCycle;
    if (cfg.output_bytes_per_cycle <= 0) cfg.output_bytes_per_cycle = kDigitalStreamFifoBytesPerCycle;

    DigitalStreamStats s;
    s.tile_rows = cfg.tile_rows;
    s.tile_cols = cfg.tile_cols;
    s.lanes = cfg.tile_rows * cfg.tile_cols;
    s.rows = std::max(0, rows);
    if (rows <= 0 || d_in <= 0 || d_out <= 0) {
        s.measured_cycles = std::max(0LL, producer_cycles);
        return s;
    }

    const long long row_bytes = static_cast<long long>(d_in) * kBytesPerElement;
    const long long out_row_bytes = static_cast<long long>(d_out) * kBytesPerElement;
    const long long row_tiles =
        (static_cast<long long>(rows) + cfg.tile_rows - 1LL) / cfg.tile_rows;
    const long long col_tiles =
        (static_cast<long long>(d_out) + cfg.tile_cols - 1LL) / cfg.tile_cols;
    const long long pipeline_cycles =
        static_cast<long long>(d_in) + cfg.tile_rows + cfg.tile_cols - 2LL;
    s.compute_cycles = row_tiles * col_tiles * pipeline_cycles;
    s.mac_ops = static_cast<long long>(rows) *
                static_cast<long long>(d_in) *
                static_cast<long long>(d_out);
    s.fifo_bytes = static_cast<long long>(rows) * row_bytes;

    const long long producer_interval =
        std::max(1LL, (std::max(0LL, producer_cycles) + rows - 1LL) / rows);
    const long long dense_row_service =
        std::max(1LL, (s.compute_cycles + rows - 1LL) / rows);
    const long long ingress_row_cycles =
        std::max(1LL, (row_bytes + cfg.ingress_bytes_per_cycle - 1LL) /
                       cfg.ingress_bytes_per_cycle);
    const long long output_row_cycles =
        std::max(1LL, (out_row_bytes + cfg.output_bytes_per_cycle - 1LL) /
                       cfg.output_bytes_per_cycle);

    std::deque<long long> input_consume_times;
    std::deque<long long> output_drain_times;
    long long producer_delay = 0;
    long long consumer_available = 0;
    long long output_drain_available = 0;
    long long last_produced = 0;

    for (int row = 0; row < rows; row++) {
        const long long nominal_ready =
            static_cast<long long>(row + 1) * producer_interval + ingress_row_cycles - 1LL;
        long long ready = nominal_ready + producer_delay;

        while (!input_consume_times.empty() && input_consume_times.front() <= ready) {
            input_consume_times.pop_front();
        }
        if (static_cast<int>(input_consume_times.size()) >= cfg.input_fifo_rows) {
            const long long release = input_consume_times.front();
            const long long wait = std::max(0LL, release - ready);
            s.backpressure_cycles += wait;
            producer_delay += wait;
            ready += wait;
            while (!input_consume_times.empty() && input_consume_times.front() <= ready) {
                input_consume_times.pop_front();
            }
        }

        const long long start = std::max(consumer_available, ready);
        s.input_fifo_occupancy_cycles += std::max(0LL, start - ready);
        input_consume_times.push_back(start);

        long long output_ready = start + dense_row_service;
        while (!output_drain_times.empty() && output_drain_times.front() <= output_ready) {
            output_drain_times.pop_front();
        }
        if (static_cast<int>(output_drain_times.size()) >= cfg.output_fifo_rows) {
            const long long release = output_drain_times.front();
            const long long wait = std::max(0LL, release - output_ready);
            s.backpressure_cycles += wait;
            output_ready += wait;
            while (!output_drain_times.empty() && output_drain_times.front() <= output_ready) {
                output_drain_times.pop_front();
            }
        }

        const long long drain_start = std::max(output_ready, output_drain_available);
        const long long drain_done = drain_start + output_row_cycles;
        s.output_fifo_occupancy_cycles += std::max(0LL, drain_done - output_ready);
        output_drain_available = drain_done;
        output_drain_times.push_back(drain_done);
        consumer_available = output_ready;
        last_produced = ready;
    }

    s.measured_cycles = std::max(std::max(last_produced, consumer_available), output_drain_available);
    const long long serial_no_overlap =
        std::max(0LL, producer_cycles) + s.compute_cycles +
        static_cast<long long>(rows) * output_row_cycles;
    s.overlap_cycles = std::max(0LL, serial_no_overlap - s.measured_cycles);
    return s;
}

void DualModeSpMM::apply_digital_stream_backend_accounting(
    StepResult& r,
    long long producer_cycles,
    int rows,
    int d_in,
    int d_out,
    long long z_materialize_cycles) {
    DigitalBackendConfig cfg;
    cfg.tile_rows = kDigitalStreamTileRows;
    cfg.tile_cols = kDigitalStreamTileCols;
    cfg.input_fifo_rows = kDigitalStreamInputFifoRows;
    cfg.output_fifo_rows = kDigitalStreamOutputFifoRows;
    cfg.ingress_bytes_per_cycle = kDigitalStreamFifoBytesPerCycle;
    cfg.output_bytes_per_cycle = kDigitalStreamFifoBytesPerCycle;

    DigitalStreamStats stats = simulate_digital_stream_backend(
        producer_cycles, rows, d_in, d_out, cfg);
    r.combine_cycles = stats.compute_cycles;
    r.z_dram_rw_cycles = std::max(0LL, z_materialize_cycles);
    r.total_cycles = std::max(stats.measured_cycles, r.z_dram_rw_cycles);

    auto& c = r.arch_ctrs;
    c.digital_stream_measured_cycles += static_cast<uint64_t>(std::max(0LL, stats.measured_cycles));
    c.digital_stream_compute_cycles += static_cast<uint64_t>(std::max(0LL, stats.compute_cycles));
    c.digital_stream_overlap_cycles += static_cast<uint64_t>(std::max(0LL, stats.overlap_cycles));
    c.digital_stream_fifo_bytes += static_cast<uint64_t>(std::max(0LL, stats.fifo_bytes));
    c.stream_fifo_bytes += static_cast<uint64_t>(std::max(0LL, stats.fifo_bytes));
    c.digital_stream_input_fifo_occupancy_cycles +=
        static_cast<uint64_t>(std::max(0LL, stats.input_fifo_occupancy_cycles));
    c.digital_stream_output_fifo_occupancy_cycles +=
        static_cast<uint64_t>(std::max(0LL, stats.output_fifo_occupancy_cycles));
    c.digital_stream_backpressure_cycles +=
        static_cast<uint64_t>(std::max(0LL, stats.backpressure_cycles));
    c.digital_stream_rows += static_cast<uint64_t>(std::max(0LL, stats.rows));
    c.digital_stream_mac_ops += static_cast<uint64_t>(std::max(0LL, stats.mac_ops));
    c.digital_stream_tile_rows = stats.tile_rows;
    c.digital_stream_tile_cols = stats.tile_cols;
    c.digital_stream_lanes = stats.lanes;
}

void DualModeSpMM::populate_arch_ablation_counters(
    StepResult& r,
    bool y_materialized,
    bool z_materialized,
    bool /*higher_layer_materialization_estimate*/,
    int active_phase_count,
    int y_materialized_rows,
    int materialization_eliminated_rows) const {
    const bool sparse_materialized = arch_forces_sparse_materialized();
    const bool systolic_estimator = arch_uses_systolic_estimator();
    const bool digital_stream_backend = arch_uses_digital_stream_backend();
    const bool dcim_arch = arch_collects_dcim_counters();
    if (!sparse_materialized && !systolic_estimator && !digital_stream_backend && !dcim_arch) {
        return;
    }

    auto& c = r.arch_ctrs;
    c.behavior_forced =
        sparse_materialized ||
        digital_stream_backend ||
        arch_variant_ == ArchVariant::H4_SPARSEFE_INDEX_DCIM ||
        arch_variant_ == ArchVariant::H5_SPARSEFE_INDEX_DCIM_STREAM;
    c.dcim_enabled = (r.cim_mode != CiMMode::NONE);
    c.streaming_enabled = digital_stream_backend || (r.cim_mode == CiMMode::PER_PHASE_LINEAR);
    c.fullscan_enabled = (arch_variant_ == ArchVariant::H1_SPARSEFE_DIGITALBE);
    c.indexstore_enabled = (arch_variant_ == ArchVariant::H2_SPARSEFE_INDEX_DIGITALBE ||
                            arch_variant_ == ArchVariant::H3_SPARSEFE_INDEX_DIGITAL_STREAM ||
                            arch_variant_ == ArchVariant::H4_SPARSEFE_INDEX_DCIM ||
                            arch_variant_ == ArchVariant::H5_SPARSEFE_INDEX_DCIM_STREAM ||
                            (arch_variant_ == ArchVariant::H6_FULL_DESPOINA &&
                             effective_indexer_mode_for_arch() == "indexstore"));

    if (systolic_estimator) {
        int active_rows = 0;
        if (!r.affected_rows.empty()) {
            active_rows = static_cast<int>(r.affected_rows.size());
        } else if (r.mode_used == Mode::FULL) {
            active_rows = n_;
        } else if (y_materialized_rows >= 0) {
            active_rows = y_materialized_rows;
        }
        active_rows = std::max(0, active_rows);

        const uint64_t active = static_cast<uint64_t>(active_rows);
        const uint64_t d_in = static_cast<uint64_t>(std::max(0, d_));
        const uint64_t d_out = static_cast<uint64_t>(std::max(0, d_out_));
        const uint64_t row_tiles =
            (active + static_cast<uint64_t>(kH0SystolicTileRows) - 1ULL) /
            static_cast<uint64_t>(kH0SystolicTileRows);
        const uint64_t col_tiles =
            (d_out + static_cast<uint64_t>(kH0SystolicTileCols) - 1ULL) /
            static_cast<uint64_t>(kH0SystolicTileCols);
        const uint64_t pipeline_cycles =
            d_in + static_cast<uint64_t>(kH0SystolicTileRows + kH0SystolicTileCols - 2);

        c.systolic_tile_rows = kH0SystolicTileRows;
        c.systolic_tile_cols = kH0SystolicTileCols;
        c.systolic_active_rows += active;
        c.systolic_mac_ops += active * d_in * d_out;
        c.systolic_compute_cycles += row_tiles * col_tiles * pipeline_cycles;
        c.systolic_pack_bytes += active * d_in * static_cast<uint64_t>(kBytesPerElement) * 2ULL;
        c.systolic_materialize_bytes += active * d_out * static_cast<uint64_t>(kBytesPerElement);
        c.systolic_pack_cycles +=
            static_cast<uint64_t>(estimate_dram_read_cycles(active_rows, d_)) +
            static_cast<uint64_t>(estimate_dram_write_cycles(active_rows, d_));
        c.systolic_materialize_cycles +=
            static_cast<uint64_t>(estimate_dram_write_cycles(active_rows, d_out_));
        c.systolic_cycles += c.systolic_compute_cycles +
                             c.systolic_pack_cycles +
                             c.systolic_materialize_cycles;
    }

    const int y_rows = (y_materialized_rows >= 0) ? y_materialized_rows : n_;
    const uint64_t y_bytes =
        static_cast<uint64_t>(std::max(0, y_rows)) *
        static_cast<uint64_t>(std::max(0, d_)) *
        static_cast<uint64_t>(kBytesPerElement);
    const uint64_t z_bytes =
        static_cast<uint64_t>(std::max(0, n_)) *
        static_cast<uint64_t>(std::max(0, d_out_)) *
        static_cast<uint64_t>(kBytesPerElement);

    if (y_materialized) {
        c.y_materialize_read_bytes += y_bytes;
        c.y_materialize_write_bytes += y_bytes;
    }
    if (z_materialized) {
        c.z_materialize_read_bytes += z_bytes;
        c.z_materialize_write_bytes += z_bytes;
    }
    if (materialization_eliminated_rows > 0) {
        const uint64_t eliminated_bytes =
            static_cast<uint64_t>(materialization_eliminated_rows) *
            static_cast<uint64_t>(std::max(0, d_)) *
            static_cast<uint64_t>(kBytesPerElement) * 2ULL;
        c.materialization_eliminated_bytes += eliminated_bytes;
    }

    c.index_lookup_ops += r.indexer_ctrs.index_lookup_ops;
    c.index_lookup_cycles += static_cast<uint64_t>(std::max(0LL, r.indexer_ctrs.index_lookup_cycles));
    c.index_bytes_read += r.indexer_ctrs.index_bytes_read;
    c.index_bytes_write += r.indexer_ctrs.index_bytes_write;
    if (arch_variant_ == ArchVariant::H1_SPARSEFE_DIGITALBE) {
        c.fullscan_bytes_read += r.indexer_ctrs.index_bytes_read;
    } else if (arch_variant_ == ArchVariant::H2_SPARSEFE_INDEX_DIGITALBE) {
        c.indexstore_bytes_read += r.indexer_ctrs.index_bytes_read;
        c.indexstore_bytes_write += r.indexer_ctrs.index_bytes_write;
    } else if (c.indexstore_enabled) {
        c.indexstore_bytes_read += r.indexer_ctrs.index_bytes_read;
        c.indexstore_bytes_write += r.indexer_ctrs.index_bytes_write;
    }

    c.row_detect_cycles += static_cast<uint64_t>(std::max(0LL, r.row_detect_cycles));
    c.row_detect_checks += static_cast<uint64_t>(std::max(0, r.row_detect_checks));
    c.row_detect_sram_reads += static_cast<uint64_t>(std::max(0, r.row_detect_sram_reads));

    if ((arch_variant_ == ArchVariant::H2_SPARSEFE_INDEX_DIGITALBE ||
         arch_variant_ == ArchVariant::H3_SPARSEFE_INDEX_DIGITAL_STREAM ||
         arch_variant_ == ArchVariant::H4_SPARSEFE_INDEX_DCIM ||
         arch_variant_ == ArchVariant::H5_SPARSEFE_INDEX_DCIM_STREAM ||
         arch_variant_ == ArchVariant::H6_FULL_DESPOINA) &&
        active_phase_count > 0) {
        const uint64_t row_bitmap_bytes =
            (static_cast<uint64_t>(std::max(0, n_)) + 7ULL) / 8ULL;
        const uint64_t popcount_words =
            (static_cast<uint64_t>(std::max(0, n_)) + 63ULL) / 64ULL;
        c.crb_bytes_read_estimated += row_bitmap_bytes * static_cast<uint64_t>(active_phase_count);
        c.crb_bytes_write_estimated += row_bitmap_bytes * static_cast<uint64_t>(active_phase_count);
        c.crb_popcount_cycles_estimated += popcount_words * static_cast<uint64_t>(active_phase_count);
    }

    if (digital_stream_backend) {
        const int stream_rows = (materialization_eliminated_rows > 0)
                                    ? materialization_eliminated_rows
                                    : static_cast<int>(c.digital_stream_rows);
        const uint64_t stream_bytes =
            static_cast<uint64_t>(stream_rows) *
            static_cast<uint64_t>(std::max(0, d_)) *
            static_cast<uint64_t>(kBytesPerElement);
        c.materialization_eliminated_bytes += stream_bytes * 2ULL;
    }

    if (dcim_arch) {
        c.dcim_cycles += static_cast<uint64_t>(std::max(0LL, r.cim_cycles));
#if !DESPOINA_V3_FUNCTIONAL_DCIM
        c.dcim_exposed_cycles += static_cast<uint64_t>(std::max(0LL, r.cim_exposed_cycles));
#endif
        c.dcim_stream_cycles += static_cast<uint64_t>(std::max(0LL, r.cim_stream_cycles));
        c.cim_rows += static_cast<uint64_t>(std::max(0, r.cim_rows));
        c.cim_active_lane_cycles += static_cast<uint64_t>(std::max(0LL, r.cim_active_lane_cycles));
        c.cim_active_slot_cycles += static_cast<uint64_t>(std::max(0LL, r.cim_active_slot_cycles));
        c.cim_input_fifo_occupancy_cycles +=
            static_cast<uint64_t>(std::max(0LL, r.cim_input_fifo_occupancy_cycles));
        c.cim_output_fifo_occupancy_cycles +=
            static_cast<uint64_t>(std::max(0LL, r.cim_output_fifo_occupancy_cycles));
        c.cim_pe_backpressure_cycles +=
            static_cast<uint64_t>(std::max(0LL, r.cim_pe_backpressure_cycles));
        const uint64_t measured_stream_bytes =
            static_cast<uint64_t>(std::max(0LL, r.pe_stats.output_egress_bytes));
        if (measured_stream_bytes > 0) {
            c.stream_fifo_bytes += measured_stream_bytes;
        } else if (r.cim_mode == CiMMode::PER_PHASE_LINEAR && r.cim_rows > 0) {
            c.stream_fifo_bytes_estimated +=
                static_cast<uint64_t>(r.cim_rows) *
                static_cast<uint64_t>(std::max(0, d_)) *
                static_cast<uint64_t>(kBytesPerElement);
        }
    }
}

void DualModeSpMM::set_mode_from_string(const std::string& mode_str) {
    mode_ = parse_mode(mode_str);
}

void DualModeSpMM::set_cim_mode_from_string(const std::string& mode_str) {
    cim_mode_ = parse_cim_mode(mode_str);
}

void DualModeSpMM::set_phase_mode_from_string(const std::string& mode_str) {
    phase_mode_ = parse_phase_mode(mode_str);
}

void DualModeSpMM::set_arch_variant_from_string(const std::string& variant_str) {
    arch_variant_ = parse_arch_variant(variant_str);
}

void DualModeSpMM::set_cim(NeuraCiM* cim, int d_out, CiMMode mode) {
    cim_ = cim;
    if (d_out > 0) {
        d_out_ = d_out;
    }
    cim_mode_ = (cim_ == nullptr) ? CiMMode::NONE : mode;

    if (n_ > 0 && d_out_ > 0) {
        const size_t z_size = static_cast<size_t>(n_) * static_cast<size_t>(d_out_);
        z_current_.assign(z_size, 0.0f);
        z_prev_.assign(z_size, 0.0f);
    } else {
        z_current_.clear();
        z_prev_.clear();
    }
}

void DualModeSpMM::init_from_loaded_graph(const PEArraySpMM::Stats& /*unused*/, int n, int nnz, int d) {
    n_ = n;
    nnz_ = nnz;
    d_ = d;
    y_layer_state_.clear();
    z_layer_state_.clear();
    h_layer_state_.clear();
    runtime_layer_trace_.clear();
    runtime_activation_trace_.clear();
    if (n_ <= 0 || d_ <= 0) {
        f_current_.clear();
        y_current_.clear();
        z_current_.clear();
        z_prev_.clear();
        return;
    }
    if (f_current_.empty()) {
        f_current_.assign(static_cast<size_t>(n_) * static_cast<size_t>(d_), 1.0f);
    }
    y_current_.assign(static_cast<size_t>(n_) * static_cast<size_t>(d_), 0.0f);
    const size_t z_size = static_cast<size_t>(n_) * static_cast<size_t>(d_out_);
    z_current_.assign(z_size, 0.0f);
    z_prev_.assign(z_size, 0.0f);
}

void DualModeSpMM::load_feature_init(const std::string& f_init_path, int n, int d) {
    n_ = n;
    d_ = d;
    if (f_init_path.empty()) {
        throw std::runtime_error("f-init is required");
    }
    std::ifstream fin(f_init_path, std::ios::binary);
    if (!fin.is_open()) {
        throw std::runtime_error("cannot open f-init: " + f_init_path);
    }
    f_current_.assign(static_cast<size_t>(n_) * static_cast<size_t>(d_), 0.0f);
    fin.read(reinterpret_cast<char*>(f_current_.data()),
             static_cast<std::streamsize>(f_current_.size() * sizeof(float)));
    if (!fin || fin.peek() != std::ifstream::traits_type::eof()) {
        throw std::runtime_error("f-init must contain exactly N*128 FP32 values");
    }
}

void DualModeSpMM::load_weights(const std::string& path, int d_in, int d_out) {
    d_out_ = d_out;
    load_required_fp32_weights(path, d_in, d_out, "L1", &weights_);
}

DualModeSpMM::StepResult DualModeSpMM::run_full_from_existing_y(
    const WorkStealingScheduler::CSRMatrix& a_new,
    const std::vector<float>& y_full,
    const PEArraySpMM::Stats& pe_stats) {
    StepResult r;
    r.mode_used = Mode::FULL;
    r.phase_mode_used = phase_mode_;
    r.pe_stats = pe_stats;
    // Seed the maintained raw-CSR IndexStore from A_0.  Incremental t1+
    // applies only the structural delta and never materializes normalized M/dM.
    indexer_.reset_counters();
    indexer_.build_from_csr(a_new);
    r.indexer_ctrs = indexer_.get_counters();

    const size_t expected_y = static_cast<size_t>(n_) * static_cast<size_t>(d_);
    if (expected_y > 0 && y_full.size() == expected_y) {
        y_current_ = y_full;
    } else {
        y_current_.assign(expected_y, 0.0f);
    }
    row_mean_operator_.reset(a_new);
    DynamicRowMeanOperator::apply_full_row_mean(
        &y_current_, row_mean_operator_.degree(), d_);
    r.row_mean_enabled = true;
    r.row_mean_scaled_rows = n_;
    r.row_mean_scale_elements = static_cast<long long>(n_) * d_;
    r.row_mean_scalar_ops = r.row_mean_scale_elements;
    r.row_mean_degree_read_events = n_;
    r.row_mean_degree_read_bytes = static_cast<long long>(n_) * sizeof(std::int32_t);
    r.row_mean_reciprocal_events = std::count_if(
        row_mean_operator_.degree().begin(), row_mean_operator_.degree().end(),
        [](int degree) { return degree > 0; });
    r.y_accuracy = compute_y_accuracy_metrics(y_current_, golden_full_spmm(a_new));

    r.y_dram_read_cycles = estimate_dram_read_cycles(n_, d_);
    r.y_dram_write_cycles = estimate_dram_write_cycles(n_, d_);

    const CiMMode effective_cim_mode = effective_cim_mode_for_arch();
    if (effective_cim_mode != CiMMode::NONE) {
        r.cim_mode = CiMMode::FULL_YNEW;
        r.cim_rows = n_;
        if (compute_accuracy_) {
            run_full_cim_validation(a_new, r);
        } else {
            r.cim_cycles = estimate_cim_cycles(cim_, r.cim_rows);
        }
        const long long z_bytes_all =
            static_cast<long long>(n_) * static_cast<long long>(d_out_) * kBytesPerElement;
        if (dedicated_z_sram_can_hold(z_bytes_all)) {
            r.z_dram_rw_cycles = 0;
            r.z_sram_accum_used = true;
            r.account_z_sram_bytes(z_bytes_all, z_bytes_all);
        } else {
#ifdef ENABLE_DRAMSIM3_PE
            if (yz_dram_ != nullptr) {
                r.z_dram_rw_cycles = issue_yz_all_rows(kZDramBaseAddr, n_, d_out_, true);
            } else {
                r.z_dram_rw_cycles = estimate_dram_write_cycles(n_, d_out_);
            }
#else
            r.z_dram_rw_cycles = estimate_dram_write_cycles(n_, d_out_);
#endif
        }
    } else {
        r.cim_mode = CiMMode::NONE;
    }
    if (arch_uses_digital_stream_backend()) {
        r.y_dram_read_cycles = 0;
        r.y_dram_write_cycles = 0;
        const long long producer_cycles =
            static_cast<long long>(r.pe_stats.total_cycles) +
            r.indexer_ctrs.index_dram_cycles +
            r.indexer_ctrs.feature_dram_cycles;
        const long long z_write_cycles = estimate_dram_write_cycles(n_, d_out_);
        apply_digital_stream_backend_accounting(r,
                                                producer_cycles,
                                                n_,
                                                d_,
                                                d_out_,
                                                z_write_cycles);
        populate_arch_ablation_counters(r,
                                        false,
                                        r.z_dram_rw_cycles > 0,
                                        false,
                                        1,
                                        -1,
                                        n_);
        return r;
    }
#if DESPOINA_V3_FUNCTIONAL_DCIM
    r.total_cycles = finalize_causal_dcim_step_cycles(
        r.pe_stats, r.indexer_ctrs,
        r.y_dram_read_cycles, r.y_dram_write_cycles,
        r.exposed_tail_cycles, r.z_dram_rw_cycles);
    r.cim_cycles = static_cast<long long>(r.causal_stream_elapsed_cycles.value);
    r.cim_stream_cycles = r.cim_cycles;
    r.cim_exposed_cycles = static_cast<long long>(r.exposed_tail_cycles.value);
#else
    r.total_cycles = finalize_full_step_cycles(r.pe_stats,
                                               r.indexer_ctrs,
                                               r.y_dram_read_cycles,
                                               r.y_dram_write_cycles,
                                               r.cim_cycles,
                                               r.z_dram_rw_cycles,
                                               &r.cim_exposed_cycles);
#endif
    populate_arch_ablation_counters(r,
                                    r.y_dram_read_cycles > 0 || r.y_dram_write_cycles > 0,
                                    r.z_dram_rw_cycles > 0,
                                    false,
                                    1);
    return r;
}

std::vector<int> DualModeSpMM::collect_nonzero_rows(
    const WorkStealingScheduler::CSRMatrix& csr) const {
    std::vector<int> rows;
    rows.reserve(static_cast<size_t>(csr.n));
    for (int i = 0; i < csr.n; i++) {
        if (csr.row_ptr[i] != csr.row_ptr[i + 1]) {
            rows.push_back(i);
        }
    }
    return rows;
}

void DualModeSpMM::run_full_cim_validation(const WorkStealingScheduler::CSRMatrix& a_new,
                                           StepResult& r) {
    if (n_ <= 0 || d_out_ <= 0) {
        return;
    }
    std::vector<int> all_rows;
    all_rows.reserve(static_cast<size_t>(n_));
    for (int row = 0; row < n_; row++) {
        all_rows.push_back(row);
    }

    // R5.1 cold start: validation is a consumer of the normal layer-entry
    // resident-W transition, never a bypass around it.  The load establishes
    // source/hash/version/shape, programs every full-W replica, and waits for
    // every tile's program-ready completion before the first start_matmul.
    load_resident_weights_for_layer(1);
    if (!cim_->all_resident_tiles_ready()) {
        throw std::runtime_error("R5.1 t0 L1 resident replicas not ready");
    }
    const uint64_t load_events_before_validation = cim_->get_weight_load_events();
    const uint64_t tile_program_events_before_validation =
        cim_->get_weight_program_tile_events();
    const uint64_t programmed_bytes_before_validation =
        cim_->get_weight_resident_int8_bytes();

    r.cim_rows = n_;
    CimHardwareRunResult hw = run_cim_hardware_rows(y_current_, all_rows);
    if (cim_->get_weight_load_events() != load_events_before_validation ||
        cim_->get_weight_program_tile_events() != tile_program_events_before_validation ||
        cim_->get_weight_resident_int8_bytes() != programmed_bytes_before_validation) {
        throw std::runtime_error("R5.1 validation mutated resident-W program counters");
    }
#if DESPOINA_V3_FUNCTIONAL_DCIM
    const std::vector<uint64_t> t0_ready(static_cast<size_t>(n_), 0);
    if (!hw.ok || !capture_runtime_layer_trace(1, all_rows) ||
        !record_direct_dcim_accounting(t0_ready, 0, hw, r)) {
        throw std::runtime_error("R4 t0 layer-1 DCIM accounting/trace failed closed");
    }
#endif
    std::vector<float> cim_z = hw.output;
    if (cim_noise_ && (!cim_ || cim_->get_compute_mode() == CiMComputeMode::IDEAL_FP32)) {
        apply_cim_noise(cim_z, static_cast<int>(all_rows.size()), d_out_);
    }
    std::vector<float> golden_z = golden_full_forward_rows(a_new, all_rows);
    r.accuracy = compute_accuracy_metrics(cim_z, golden_z, hw);
#if DESPOINA_V3_FUNCTIONAL_DCIM
    r.cim_cycles = static_cast<long long>(r.causal_stream_elapsed_cycles.value);
    r.cim_stream_cycles = r.cim_cycles;
    r.cim_exposed_cycles = static_cast<long long>(r.exposed_tail_cycles.value);
#else
    r.cim_cycles = hw.cycles;
#endif

    const size_t z_size = static_cast<size_t>(n_) * static_cast<size_t>(d_out_);
    if (z_current_.size() != z_size) {
        z_current_.assign(z_size, 0.0f);
    }
    if (z_prev_.size() != z_size) {
        z_prev_.assign(z_size, 0.0f);
    }

    const bool cim_shape_ok = (cim_z.size() == z_size);
    const bool golden_shape_ok = (golden_z.size() == z_size);
    if (hw.ok && cim_shape_ok) {
        std::copy(cim_z.begin(), cim_z.end(), z_current_.begin());
    } else if (golden_shape_ok) {
        std::copy(golden_z.begin(), golden_z.end(), z_current_.begin());
    } else {
        std::fill(z_current_.begin(), z_current_.end(), 0.0f);
    }
    z_prev_ = z_current_;
}

DualModeSpMM::CimHardwareRunResult DualModeSpMM::run_cim_hardware_rows(
    const std::vector<float>& Y,
    const std::vector<int>& rows) {

    CimHardwareRunResult result;
    result.expected_rows = static_cast<int>(rows.size());
    result.output.assign(rows.size() * static_cast<size_t>(d_out_), 0.0f);

    if (rows.empty()) {
        result.ok = true;
        result.received_unique_rows = 0;
        return result;
    }
    if (cim_ == nullptr || d_ <= 0 || d_out_ <= 0 ||
        Y.size() < static_cast<size_t>(n_) * static_cast<size_t>(d_)) {
        result.ok = false;
        return result;
    }
    for (size_t i = 0; i < rows.size(); i++) {
        if (rows[i] < 0 || rows[i] >= n_) {
            result.invalid_packets += 1;
        }
    }
    if (result.invalid_packets > 0) {
        result.ok = false;
        return result;
    }

    const NeuraCiM::PipelineCounters cim_counters_before = cim_->get_pipeline_counters();
    if (!cim_->start_matmul(static_cast<int>(rows.size()), d_, d_out_, true)) {
        result.ok = false;
        return result;
    }

    std::vector<uint8_t> seen(rows.size(), 0);
    int unique_received = 0;
    int pushed = 0;

    const long long analytical =
        static_cast<long long>(cim_->compute_cycles_analytical(static_cast<int>(rows.size())));
    const long long max_cycles =
        std::max(analytical * kAccuracyWatchdogScale + kAccuracyWatchdogBase,
                 static_cast<long long>(rows.size()) * 32LL + kAccuracyWatchdogBase);

    while (unique_received < static_cast<int>(rows.size()) && result.cycles < max_cycles) {
        while (pushed < static_cast<int>(rows.size()) && cim_->can_accept_input()) {
            const int row = rows[static_cast<size_t>(pushed)];
            CiMInputPacket pkt;
            pkt.row_id = pushed;
            pkt.is_incremental = true;
            pkt.features.assign(
                Y.begin() + static_cast<size_t>(row) * static_cast<size_t>(d_),
                Y.begin() + static_cast<size_t>(row + 1) * static_cast<size_t>(d_));
            if (!cim_->push_input(pkt)) {
                break;
            }
            constexpr uint64_t kPeCimFlitBytes = 32;
            const uint64_t payload_bytes =
                static_cast<uint64_t>(pkt.features.size()) * sizeof(float);
            result.pe_cim_noc_bytes += payload_bytes;
            result.pe_cim_noc_flits +=
                (payload_bytes + kPeCimFlitBytes - 1) / kPeCimFlitBytes;
            pushed++;
        }

        cim_->tick();
        result.cycles++;

        CiMOutputPacket packet;
        while (cim_->pop_output(packet)) {
            if (packet.row_id < 0 ||
                packet.row_id >= static_cast<int>(rows.size()) ||
                static_cast<int>(packet.result.size()) != d_out_) {
                result.invalid_packets++;
                continue;
            }
            const size_t idx = static_cast<size_t>(packet.row_id);
            if (seen[idx] != 0) {
                result.duplicate_packets++;
                continue;
            }
            seen[idx] = 1;
            std::copy(packet.result.begin(),
                      packet.result.end(),
                      result.output.begin() + idx * static_cast<size_t>(d_out_));
            unique_received++;
        }
    }

    result.timeout = (unique_received < static_cast<int>(rows.size()));
    result.received_unique_rows = unique_received;
    result.ok = !result.timeout &&
                unique_received == static_cast<int>(rows.size()) &&
                result.duplicate_packets == 0 &&
                result.invalid_packets == 0;
    result.cim_pipeline =
        diff_pipeline_counters(cim_->get_pipeline_counters(), cim_counters_before);
    return result;
}

DualModeSpMM::CimHardwareRunResult DualModeSpMM::run_cim_hardware_rows_causal(
    const std::vector<float>& Y,
    const std::vector<int>& rows,
    const std::vector<uint64_t>& producer_ready_cycles) {
    CimHardwareRunResult result;
    result.expected_rows = static_cast<int>(rows.size());
    result.output.assign(rows.size() * static_cast<size_t>(d_out_), 0.0f);
    if (rows.empty()) { result.ok = true; return result; }
    if (cim_ == nullptr || rows.size() != producer_ready_cycles.size() ||
        d_ != 128 || d_out_ != 128 ||
        Y.size() < static_cast<size_t>(n_) * static_cast<size_t>(d_)) {
        result.invalid_packets++;
        return result;
    }
    struct Arrival { uint64_t ready; int ordinal; };
    std::vector<Arrival> arrivals;
    arrivals.reserve(rows.size());
    for (size_t i = 0; i < rows.size(); ++i) {
        if (rows[i] < 0 || rows[i] >= n_) { result.invalid_packets++; return result; }
        arrivals.push_back({producer_ready_cycles[i], static_cast<int>(i)});
    }
    std::stable_sort(arrivals.begin(), arrivals.end(),
        [](const Arrival& a, const Arrival& b) { return a.ready < b.ready; });
    result.producer_first_ready_cycle = static_cast<long long>(arrivals.front().ready);
    result.producer_last_ready_cycle = static_cast<long long>(arrivals.back().ready);
    // 256 KiB finalized-row staging, 128 FP32 elements/row.  The following
    // native input FIFO remains the separately finite FIFO16 contract.
    const size_t row_bytes = static_cast<size_t>(d_) * sizeof(float);
    const size_t staging_capacity = (256U * 1024U) / row_bytes;
    if (staging_capacity == 0) { result.invalid_packets++; return result; }
    result.staging_capacity_rows = static_cast<long long>(staging_capacity);
    const NeuraCiM::PipelineCounters before = cim_->get_pipeline_counters();
    if (!cim_->start_matmul(static_cast<int>(rows.size()), d_, d_out_, true)) return result;
    std::deque<int> staging;
    std::vector<uint8_t> seen(rows.size(), 0);
    size_t next = 0;
    int received = 0;
    uint64_t cycle = 0;
    const long long analytical = static_cast<long long>(cim_->compute_cycles_analytical(
        static_cast<int>(rows.size())));
    const uint64_t watchdog = static_cast<uint64_t>(std::max(
        analytical * kAccuracyWatchdogScale + kAccuracyWatchdogBase,
        static_cast<long long>(arrivals.back().ready) + analytical * kAccuracyWatchdogScale +
            kAccuracyWatchdogBase));
    while (received < static_cast<int>(rows.size()) && cycle < watchdog) {
        // The PE producer completion event owns the transition into staging.
        // If staging is full, this exact producer event is credit-stalled.
        while (next < arrivals.size() && arrivals[next].ready <= cycle &&
               staging.size() < staging_capacity) {
            staging.push_back(arrivals[next].ordinal);
            result.producer_last_enqueue_cycle = static_cast<long long>(cycle);
            ++next;
        }
        if (next < arrivals.size() && arrivals[next].ready <= cycle &&
            staging.size() >= staging_capacity) {
            result.pe_to_cim_backpressure_cycles++;
        }
        result.max_staging_rows = std::max(result.max_staging_rows,
            static_cast<long long>(staging.size()));
        // DCIM issues immediately whenever the finite FIFO has credit; it
        // never waits for producer completion of the whole phase.
        while (!staging.empty() && cim_->can_accept_input()) {
            const int ordinal = staging.front(); staging.pop_front();
            const int graph_row = rows[static_cast<size_t>(ordinal)];
            CiMInputPacket pkt;
            pkt.row_id = ordinal;
            pkt.is_incremental = true;
            pkt.features.assign(Y.begin() + static_cast<size_t>(graph_row) * static_cast<size_t>(d_),
                                Y.begin() + static_cast<size_t>(graph_row + 1) * static_cast<size_t>(d_));
            if (!cim_->push_input(pkt)) { result.invalid_packets++; return result; }
            constexpr uint64_t kPeCimFlitBytes = 32;
            const uint64_t payload_bytes =
                static_cast<uint64_t>(pkt.features.size()) * sizeof(float);
            result.pe_cim_noc_bytes += payload_bytes;
            result.pe_cim_noc_flits +=
                (payload_bytes + kPeCimFlitBytes - 1) / kPeCimFlitBytes;
        }
        cim_->tick();
        ++cycle;
        CiMOutputPacket packet;
        while (cim_->pop_output(packet)) {
            if (packet.row_id < 0 || packet.row_id >= static_cast<int>(rows.size()) ||
                packet.result.size() != static_cast<size_t>(d_out_)) { result.invalid_packets++; continue; }
            const size_t id = static_cast<size_t>(packet.row_id);
            if (seen[id] != 0) { result.duplicate_packets++; continue; }
            seen[id] = 1;
            std::copy(packet.result.begin(), packet.result.end(),
                      result.output.begin() + id * static_cast<size_t>(d_out_));
            ++received;
        }
    }
    result.cycles = static_cast<long long>(cycle);
    result.timeout = received != static_cast<int>(rows.size()) || next != arrivals.size();
    result.received_unique_rows = received;
    result.cim_pipeline = diff_pipeline_counters(cim_->get_pipeline_counters(), before);
    const NeuraCiM::NativeTraceStats& native = cim_->get_native_trace_stats();
    result.native_first_issue_cycle = 0;
    result.native_last_completion_cycle = 0;
    for (size_t i = 0; i < rows.size(); ++i) {
        const uint64_t issue = native.row_issue_cycles[i];
        const uint64_t done = native.row_completion_cycles[i];
        if (issue == 0 || done == 0) { result.invalid_packets++; continue; }
        if (result.native_first_issue_cycle == 0 ||
            static_cast<long long>(issue) < result.native_first_issue_cycle) {
            result.native_first_issue_cycle = static_cast<long long>(issue);
        }
        result.native_last_completion_cycle = std::max(result.native_last_completion_cycle,
            static_cast<long long>(done));
    }
    const long long overlap_lo = std::max(result.producer_first_ready_cycle,
        result.native_first_issue_cycle);
    const long long overlap_hi = std::min(result.producer_last_enqueue_cycle,
        result.native_last_completion_cycle);
    result.overlap_cycles = std::max(0LL, overlap_hi - overlap_lo + 1);
    result.exposed_tail_cycles = std::max(0LL, result.native_last_completion_cycle -
        result.producer_last_enqueue_cycle);
    result.ok = !result.timeout && received == static_cast<int>(rows.size()) &&
        result.duplicate_packets == 0 && result.invalid_packets == 0 &&
        native.row_issue_events == rows.size() && native.row_completion_events == rows.size() &&
        native.functional_apply_events == rows.size();
#if defined(DESPOINA_ENABLE_TEST_FAULT_INJECTION) && \
    DESPOINA_ENABLE_TEST_FAULT_INJECTION
    const char* inject = std::getenv("DESPOINA_INTERNAL_TEST_CORRUPT_DCIM_OUTPUT");
    if (result.ok && inject != nullptr && std::string(inject) == "1" &&
        !result.output.empty()) {
        const size_t corrupt_elements = std::min<size_t>(8, result.output.size());
        for (size_t i = 0; i < corrupt_elements; ++i) {
            result.output[i] += 1024.0f;
        }
    }
#endif
    return result;
}

DualModeSpMM::CimHardwareRunResult DualModeSpMM::run_cim_hardware_streaming_phases(
    const std::vector<LinearStreamPhase>& phases,
    const std::vector<int>& unique_rows,
    std::vector<float>* y_state,
    long long* cim_cycles,
    std::vector<PEArraySpMM::Stats>* phase_stats) {
    CimHardwareRunResult result;
    result.output.assign(unique_rows.size() * static_cast<size_t>(d_out_), 0.0f);
    if (cim_cycles) *cim_cycles = 0;
    if (phase_stats) phase_stats->assign(phases.size(), PEArraySpMM::Stats{});
    result.expected_rows = static_cast<int>(unique_rows.size());
    if (unique_rows.empty()) { result.ok = true; return result; }
    if (cim_ == nullptr || pe_ == nullptr || y_state == nullptr || d_ != 128 ||
        d_out_ != 128 || y_state->size() != static_cast<size_t>(n_) * d_) {
        result.invalid_packets++;
        return result;
    }
    std::unordered_map<int, size_t> unique_pos;
    unique_pos.reserve(unique_rows.size());
    std::vector<int> last_phase(unique_rows.size(), -1);
    for (size_t i = 0; i < unique_rows.size(); ++i) {
        if (unique_rows[i] < 0 || unique_rows[i] >= n_ ||
            !unique_pos.emplace(unique_rows[i], i).second) {
            result.invalid_packets++;
            return result;
        }
    }
    for (size_t p = 0; p < phases.size(); ++p) {
        if (phases[p].csr == nullptr || phases[p].f_matrix == nullptr) continue;
        for (size_t i = 0; i < unique_rows.size(); ++i) {
            const int row = unique_rows[i];
            if (phases[p].csr->row_ptr[static_cast<size_t>(row)] !=
                phases[p].csr->row_ptr[static_cast<size_t>(row + 1)]) {
                last_phase[i] = static_cast<int>(p);
            }
        }
    }
    for (size_t i = 0; i < last_phase.size(); ++i) {
        if (last_phase[i] < 0) {
            std::cerr << "[DCIM_STREAM_INVARIANT] reason=row_has_no_owner_phase"
                      << " graph_row=" << unique_rows[i]
                      << " ordinal=" << i
                      << " unique_rows=" << unique_rows.size() << std::endl;
            result.invalid_packets++;
            return result;
        }
    }
    // y_state is a verification-only FP32 shadow (exact recurrence) used by the
    // Y-accuracy check.  It is not modeled
    // hardware state: the DCIM input below is formed from the PE accumulator
    // alone, which starts at zero and holds Ph1 + Ph2 - Ph3 of the row.
    std::vector<float> row_mean_old_state;
    if (!row_mean_operator_.initialized()) {
        result.invalid_packets++;
        return result;
    }
    row_mean_old_state.resize(unique_rows.size() * static_cast<std::size_t>(d_));
    for (std::size_t ordinal = 0; ordinal < unique_rows.size(); ++ordinal) {
        const float* src = y_state->data() +
            static_cast<std::size_t>(unique_rows[ordinal]) * static_cast<std::size_t>(d_);
        std::copy(src, src + d_,
                  row_mean_old_state.begin() +
                      static_cast<std::ptrdiff_t>(ordinal * static_cast<std::size_t>(d_)));
    }

    const NeuraCiM::PipelineCounters before = cim_->get_pipeline_counters();
    if (!cim_->start_matmul(result.expected_rows, d_, d_out_, true)) return result;
    pe_->set_streaming_output_enabled(true);
    struct StagedRow {
        int ordinal;
        long long ready_cycle;
        std::vector<float> features;
    };
    std::deque<StagedRow> staging;
    constexpr size_t kStagingRows = (256U * 1024U) / (128U * sizeof(float));
    result.staging_capacity_rows = static_cast<long long>(kStagingRows);
    std::vector<uint8_t> finalized(unique_rows.size(), 0);
    std::vector<uint8_t> seen(unique_rows.size(), 0);
    std::vector<float> tmp_row(static_cast<size_t>(d_), 0.0f);
    // PE accumulator register file: raw three-phase delta of each changed row.
    std::vector<float> raw_delta_acc(unique_rows.size() * static_cast<size_t>(d_), 0.0f);
    result.z_old_scale.assign(unique_rows.size(), 1.0f);
    size_t phase_index = 0;
    bool phase_loaded = false;
    int finalized_rows = 0;
    int received = 0;
    long long cycles = 0;
    const long long max_cycles = std::max<long long>(
        static_cast<long long>(unique_rows.size()) * 16384LL +
            static_cast<long long>(n_) * 4096LL + kAccuracyWatchdogBase,
        1000000LL);

    while (cycles < max_cycles) {
        while (!phase_loaded && phase_index < phases.size()) {
            const LinearStreamPhase& phase = phases[phase_index];
            if (phase.csr == nullptr || phase.f_matrix == nullptr ||
                phase.expected_rows <= 0) {
                ++phase_index;
                continue;
            }
            pe_->load_spmm(phase.csr->row_ptr.data(), phase.csr->col_idx.data(),
                           phase.csr->values.data(), phase.csr->n, phase.csr->nnz,
                           phase.f_matrix, d_, phase.keep_shared_f_cache,
                           phase.matrix_epoch);
            phase_loaded = true;
        }
        if (phase_loaded && !pe_->is_done() && staging.size() < kStagingRows) {
            pe_->tick();
        } else if (phase_loaded && staging.size() >= kStagingRows) {
            ++result.pe_to_cim_backpressure_cycles;
        }

        while (phase_loaded && pe_->has_completed_row() &&
               staging.size() < kStagingRows) {
            const int row = pe_->pop_completed_row(tmp_row.data(), d_);
            const auto pos_it = unique_pos.find(row);
            if (row < 0 || pos_it == unique_pos.end()) {
                result.invalid_packets++;
                break;
            }
            const size_t ordinal = pos_it->second;
            float* state = y_state->data() + static_cast<size_t>(row) * d_;
            float* acc = raw_delta_acc.data() + ordinal * static_cast<size_t>(d_);
            for (int j = 0; j < d_; ++j) {
                acc[j] += phases[phase_index].y_scale * tmp_row[static_cast<size_t>(j)];
                state[j] += phases[phase_index].y_scale * tmp_row[static_cast<size_t>(j)];
            }
            if (last_phase[ordinal] == static_cast<int>(phase_index)) {
                if (finalized[ordinal] != 0) {
                    result.duplicate_packets++;
                    break;
                }
                finalized[ordinal] = 1;
                ++finalized_rows;
                long long row_mean_latency = 0;
                const int d_old = row_mean_operator_.old_degree(row, row_mean_transition_);
                const int d_new = row_mean_operator_.new_degree(row);
                const bool degree_changed = d_old != d_new;
                const float* old_row = row_mean_old_state.data() +
                    ordinal * static_cast<std::size_t>(d_);
                DynamicRowMeanOperator::apply_scoped_recurrence_row(
                    state, old_row, d_old, d_new,
                    row_mean_operator_.old_over_new(row, row_mean_transition_),
                    row_mean_operator_.new_reciprocal(row), d_);
                ++result.row_mean_scaled_rows;
                result.row_mean_changed_degree_rows += degree_changed ? 1 : 0;
                result.row_mean_scale_elements += d_;
                result.row_mean_residual_correction_elements += degree_changed ? d_ : 0;
                // PE: dY = raw_delta * (1/d_new).  Post-combination buffer:
                // Z' = Z_old * (d_old/d_new) + Q(dY)W (scale only when the
                // degree changed).  An empty destination row writes zero.
                const int row_mean_ops = d_new == 0 ? 1 : (degree_changed ? 3 : 2);
                result.row_mean_scalar_ops += static_cast<long long>(row_mean_ops) * d_;
                result.row_mean_degree_read_events += degree_changed ? 2 : 1;
                result.row_mean_degree_read_bytes +=
                    static_cast<long long>(degree_changed ? 2 : 1) * sizeof(std::int32_t);
                result.row_mean_reciprocal_events += d_new > 0 ? 1 : 0;
                // Existing PE lanes are priced after all sparse phases as an
                // owner-disjoint, non-overlapped serial phase.
                row_mean_latency = 0;
                StagedRow staged;
                staged.ordinal = static_cast<int>(ordinal);
                staged.ready_cycle = cycles + 1 + row_mean_latency;
                staged.features.resize(static_cast<size_t>(d_));
                if (d_new == 0) {
                    std::fill(staged.features.begin(), staged.features.end(), 0.0f);
                    result.z_old_scale[ordinal] = 0.0f;
                } else {
                    const float inv_new = row_mean_operator_.new_reciprocal(row);
                    for (int j = 0; j < d_; ++j) {
                        staged.features[static_cast<size_t>(j)] =
                            static_cast<float>(acc[j] * inv_new);
                    }
                    result.z_old_scale[ordinal] =
                        row_mean_operator_.old_over_new(row, row_mean_transition_);
                }
                staging.push_back(std::move(staged));
                if (result.producer_first_ready_cycle == 0) {
                    result.producer_first_ready_cycle = cycles + 1 + row_mean_latency;
                }
                result.producer_last_ready_cycle = cycles + 1 + row_mean_latency;
            }
        }

        result.max_staging_rows = std::max(
            result.max_staging_rows, static_cast<long long>(staging.size()));
        while (!staging.empty() && staging.front().ready_cycle <= cycles + 1 &&
               cim_->can_accept_input()) {
            CiMInputPacket packet;
            packet.row_id = staging.front().ordinal;
            packet.features.swap(staging.front().features);
            packet.is_incremental = true;
            staging.pop_front();
            if (!cim_->push_input(packet)) {
                result.invalid_packets++;
                break;
            }
            // This is the actual transfer owner: a completed FP32 PE dY row
            // crosses the direct staging-to-DCIM input boundary here.
            constexpr uint64_t kPeCimFlitBytes = 32;
            const uint64_t payload_bytes =
                static_cast<uint64_t>(packet.features.size()) * sizeof(float);
            result.pe_cim_noc_bytes += payload_bytes;
            result.pe_cim_noc_flits +=
                (payload_bytes + kPeCimFlitBytes - 1) / kPeCimFlitBytes;
            result.producer_last_enqueue_cycle = cycles + 1;
        }

        cim_->tick();
        ++cycles;
        CiMOutputPacket packet;
        while (cim_->pop_output(packet)) {
            if (packet.row_id < 0 || packet.row_id >= result.expected_rows ||
                packet.result.size() != static_cast<size_t>(d_out_)) {
                result.invalid_packets++;
                continue;
            }
            const size_t ordinal = static_cast<size_t>(packet.row_id);
            if (seen[ordinal] != 0) { result.duplicate_packets++; continue; }
            seen[ordinal] = 1;
            std::copy(packet.result.begin(), packet.result.end(),
                      result.output.begin() + ordinal * static_cast<size_t>(d_out_));
            ++received;
        }
        if (gcp_) gcp_->tick();
        if (phase_loaded && pe_->is_done() && !pe_->has_completed_row()) {
            if (phase_stats) (*phase_stats)[phase_index] = pe_->get_stats();
            phase_loaded = false;
            ++phase_index;
        }
        const bool phases_done = !phase_loaded && phase_index >= phases.size();
        if (phases_done && staging.empty() && received == result.expected_rows &&
            cim_->is_done()) break;
    }

    pe_->set_streaming_output_enabled(false);
    result.cycles = cycles;
    result.received_unique_rows = received;
    result.timeout = received != result.expected_rows ||
        finalized_rows != result.expected_rows || !staging.empty() ||
        phase_loaded || phase_index < phases.size();
    result.cim_pipeline = diff_pipeline_counters(cim_->get_pipeline_counters(), before);
    const NeuraCiM::NativeTraceStats& native = cim_->get_native_trace_stats();
    result.native_first_issue_cycle = static_cast<long long>(native.first_bit_step_issue_cycle);
    result.native_last_completion_cycle = static_cast<long long>(native.last_bit_step_completion_cycle);
    const long long overlap_lo = std::max(result.producer_first_ready_cycle,
                                           result.native_first_issue_cycle);
    const long long overlap_hi = std::min(result.producer_last_enqueue_cycle,
                                           result.native_last_completion_cycle);
    result.overlap_cycles = std::max(0LL, overlap_hi - overlap_lo + 1);
    result.exposed_tail_cycles = std::max(
        0LL, result.native_last_completion_cycle - result.producer_last_enqueue_cycle);
    result.ok = !result.timeout && result.duplicate_packets == 0 &&
        result.invalid_packets == 0 && native.row_issue_events == unique_rows.size() &&
        native.row_completion_events == unique_rows.size() &&
        native.functional_apply_events == unique_rows.size() &&
        native.bit_step_issue_events == unique_rows.size() * 8ULL &&
        native.bit_step_completion_events == unique_rows.size() * 8ULL &&
        native.bit_step_temporal_order_violations == 0;
    if (!result.ok) {
        std::cerr << "[DCIM_STREAM_INVARIANT] expected=" << result.expected_rows
                  << " pe_finalized=" << finalized_rows
                  << " dcim_received=" << received
                  << " phase_index=" << phase_index
                  << " phase_loaded=" << phase_loaded
                  << " staging_rows=" << staging.size()
                  << " duplicate_packets=" << result.duplicate_packets
                  << " invalid_packets=" << result.invalid_packets
                  << " issue=" << native.row_issue_events
                  << " completion=" << native.row_completion_events
                  << " apply=" << native.functional_apply_events
                  << " bit_issue=" << native.bit_step_issue_events
                  << " bit_completion=" << native.bit_step_completion_events
                  << " energy=" << native.full_int8_energy_charge_events
                  << " elapsed=" << native.elapsed_cycles
                  << " loop_cycles=" << cycles;
        int missing_finalized = 0;
        int missing_seen = 0;
        std::cerr << " missing_finalized_graph_rows=";
        for (size_t i = 0; i < unique_rows.size(); ++i) {
            if (finalized[i] == 0) {
                if (missing_finalized++ < 8) std::cerr << unique_rows[i] << ',';
            }
        }
        std::cerr << " missing_seen_graph_rows=";
        for (size_t i = 0; i < unique_rows.size(); ++i) {
            if (seen[i] == 0) {
                if (missing_seen++ < 8) std::cerr << unique_rows[i] << ',';
            }
        }
        std::cerr << " missing_finalized_count=" << missing_finalized
                  << " missing_seen_count=" << missing_seen << std::endl;
    }
    if (cim_cycles) *cim_cycles = cycles;
#if defined(DESPOINA_ENABLE_TEST_FAULT_INJECTION) && \
    DESPOINA_ENABLE_TEST_FAULT_INJECTION
    const char* inject = std::getenv("DESPOINA_INTERNAL_TEST_CORRUPT_DCIM_OUTPUT");
    if (result.ok && inject != nullptr && std::string(inject) == "1" &&
        !result.output.empty()) {
        const size_t corrupt_elements = std::min<size_t>(8, result.output.size());
        for (size_t i = 0; i < corrupt_elements; ++i) {
            result.output[i] += 1024.0f;
        }
    }
#endif
    return result;
}

bool DualModeSpMM::z_layer_resident_in_post_combination_buffer(int layer_id) const {
    if (!dedicated_z_sram_ || layer_id < 1) return false;
    if (dedicated_z_sram_capacity_bytes_ <= 0) return true;
    const long long layer_bytes = static_cast<long long>(n_) *
        static_cast<long long>(d_out_) * static_cast<long long>(sizeof(float));
    return static_cast<long long>(layer_id) * layer_bytes <= dedicated_z_sram_capacity_bytes_;
}

void DualModeSpMM::account_z_state_rmw(int layer_id,
                                       const std::vector<int>& rows,
                                       StepResult& result) {
    if (rows.empty()) return;
    // Z' = Z_old * r + Q(dY)W reads and writes each committed row once.
    const long long row_bytes = static_cast<long long>(d_out_) *
        static_cast<long long>(sizeof(float));
    const long long bytes = static_cast<long long>(rows.size()) * row_bytes;
    if (z_layer_resident_in_post_combination_buffer(layer_id)) {
        result.z_sram_accum_used = true;
        result.account_z_sram_bytes(bytes, bytes);
        return;
    }
    const uint64_t base = kZStateDramBaseAddr +
        static_cast<uint64_t>(layer_id - 1) * static_cast<uint64_t>(n_) *
        static_cast<uint64_t>(row_bytes);
    long long cycles = 0;
#ifdef ENABLE_DRAMSIM3_PE
    if (yz_dram_ != nullptr) {
        cycles += issue_yz_selected_rows(base, rows, d_out_, false);
        cycles += issue_yz_selected_rows(base, rows, d_out_, true);
    } else
#endif
    {
        (void)base;
        cycles = estimate_dram_rw_cycles(static_cast<int>(rows.size()), d_out_);
    }
    result.z_hbm_rw_bytes += 2 * bytes;
    result.z_dram_rw_cycles += cycles;
}

DualModeSpMM::NpfStateAccounting DualModeSpMM::account_npf_h_state(
    int layer_id,
    const std::vector<int>& read_rows,
    const std::vector<int>& write_rows) {
    NpfStateAccounting out;
    if (layer_id < 1) return out;
    const long long row_bytes = static_cast<long long>(d_out_) *
        static_cast<long long>(sizeof(float));
    const uint64_t base = kHStateDramBaseAddr +
        static_cast<uint64_t>(layer_id - 1) * static_cast<uint64_t>(n_) *
        static_cast<uint64_t>(row_bytes);
    out.h_read_rows = static_cast<long long>(read_rows.size());
    out.h_write_rows = static_cast<long long>(write_rows.size());
    out.h_hbm_read_bytes = out.h_read_rows * row_bytes;
    out.h_hbm_write_bytes = out.h_write_rows * row_bytes;
#ifdef ENABLE_DRAMSIM3_PE
    if (yz_dram_ != nullptr) {
        out.h_hbm_cycles += issue_yz_selected_rows(base, read_rows, d_out_, false);
        out.h_hbm_cycles += issue_yz_selected_rows(base, write_rows, d_out_, true);
        flush_yz_dramsim3_stats();
        return out;
    }
#endif
    (void)base;
    out.h_hbm_cycles =
        estimate_dram_read_cycles(static_cast<int>(read_rows.size()), d_out_) +
        estimate_dram_write_cycles(static_cast<int>(write_rows.size()), d_out_);
    return out;
}

DualModeSpMM::NpfStateAccounting DualModeSpMM::account_npf_bound_state(
    int layer_id,
    const std::vector<int>& rows) {
    NpfStateAccounting out;
    if (layer_id < 1) return out;
    const uint64_t base = kBoundStateDramBaseAddr +
        static_cast<uint64_t>(layer_id - 1) * static_cast<uint64_t>(n_) *
        static_cast<uint64_t>(kBytesPerElement);
    out.h_hbm_read_bytes = static_cast<long long>(rows.size()) * kBytesPerElement;
    out.h_hbm_write_bytes = static_cast<long long>(rows.size()) * kBytesPerElement;
#ifdef ENABLE_DRAMSIM3_PE
    if (yz_dram_ != nullptr) {
        out.h_hbm_cycles += issue_yz_selected_rows(base, rows, 1, false);
        out.h_hbm_cycles += issue_yz_selected_rows(base, rows, 1, true);
        flush_yz_dramsim3_stats();
        return out;
    }
#endif
    (void)base;
    out.h_hbm_cycles = estimate_dram_read_cycles(static_cast<int>(rows.size()), 1) +
        estimate_dram_write_cycles(static_cast<int>(rows.size()), 1);
    return out;
}

void DualModeSpMM::account_row_mean_stream_result(
    const CimHardwareRunResult& hw,
    StepResult* result) const {
    if (result == nullptr) return;
    result->row_mean_enabled = true;
    result->row_mean_scaled_rows += hw.row_mean_scaled_rows;
    result->row_mean_changed_degree_rows += hw.row_mean_changed_degree_rows;
    result->row_mean_scale_elements += hw.row_mean_scale_elements;
    result->row_mean_residual_correction_elements +=
        hw.row_mean_residual_correction_elements;
    result->row_mean_scalar_ops += hw.row_mean_scalar_ops;
    result->row_mean_stream_delay_cycles += hw.row_mean_stream_delay_cycles;
    result->row_mean_degree_read_events += hw.row_mean_degree_read_events;
    result->row_mean_degree_read_bytes += hw.row_mean_degree_read_bytes;
    result->row_mean_reciprocal_events += hw.row_mean_reciprocal_events;
}

void DualModeSpMM::account_factorized_native_owners(
    const std::vector<int>& rows,
    StepResult* result) {
    if (result == nullptr) return;

    // No new SRAM macro is introduced.  The dedicated Z-SRAM holds persistent
    // Z, so degree/reciprocal state uses the already-modeled HBM hierarchy.
    // There is no old-Y preservation: the DCIM input is dY and Z accumulates.
    // This conservative allocation is explicit and fail-closed: zero borrowed
    // SRAM bytes, exact 64-byte HBM cachelines, and no overlap with PE work.
    // The conservative 16-byte existing-HBM record contains degree,
    // reciprocal, the transition's bit-exact old/new ratio, and padding.  The
    // ratio is transition metadata rather than a divider/correction lane; it
    // is charged here so its storage and transport can never become free.
    result->factorized_state_capacity_bytes = static_cast<long long>(n_) * 16LL;
    result->factorized_state_sram_bytes = 0;
    result->factorized_state_hbm_bytes = result->factorized_state_capacity_bytes;

    std::vector<int> valid_rows;
    valid_rows.reserve(rows.size());
    for (int row : rows) {
        if (row < 0 || row >= n_) {
            throw std::runtime_error("factorized native owner received invalid row");
        }
        valid_rows.push_back(row);
    }
    std::sort(valid_rows.begin(), valid_rows.end());
    valid_rows.erase(std::unique(valid_rows.begin(), valid_rows.end()), valid_rows.end());

    std::vector<int> changed_rows;
    if (result->row_mean_graph_changed_degree_rows > 0) {
        changed_rows = row_mean_transition_.changed_rows;
    }
    std::sort(changed_rows.begin(), changed_rows.end());
    changed_rows.erase(std::unique(changed_rows.begin(), changed_rows.end()),
                       changed_rows.end());

    result->factorized_state_hbm_read_bytes +=
        static_cast<long long>(valid_rows.size()) * 16LL;
    result->factorized_state_hbm_write_bytes +=
        static_cast<long long>(changed_rows.size()) * 16LL;
    const auto selected_state_bursts = [](const std::vector<int>& selected) {
        std::set<uint64_t> cachelines;
        for (size_t i = 0; i < selected.size(); i++) {
            const uint64_t row_addr = kFactorizedStateDramBaseAddr +
                static_cast<uint64_t>(selected[i]) * 16ULL;
            cachelines.insert(row_addr / 64ULL);
        }
        return static_cast<long long>(cachelines.size());
    };
    const long long state_read_bursts = selected_state_bursts(valid_rows);
    const long long state_write_bursts = selected_state_bursts(changed_rows);
    result->factorized_state_hbm_bursts += state_read_bursts + state_write_bursts;

    // No old-Y snapshot: the DCIM input is dY and Z accumulates, so Y_prev is
    // never stored.  The counters stay in the schema and remain zero.
    const long long row_bytes = 0;
    result->factorized_old_y_snapshot_bytes +=
        static_cast<long long>(valid_rows.size()) * row_bytes;
    result->factorized_old_y_sram_bytes = 0;
    result->factorized_old_y_hbm_write_bytes +=
        static_cast<long long>(valid_rows.size()) * row_bytes;
    result->factorized_old_y_hbm_read_bytes +=
        static_cast<long long>(valid_rows.size()) * row_bytes;
    const long long row_bursts = (row_bytes + 63LL) / 64LL;
    result->factorized_old_y_hbm_bursts +=
        static_cast<long long>(valid_rows.size()) * row_bursts * 2LL;

    long long state_cycles = 0;
    long long snapshot_cycles = 0;
#ifdef ENABLE_DRAMSIM3_PE
    if (yz_dram_ != nullptr) {
        state_cycles += issue_yz_selected_rows(
            kFactorizedStateDramBaseAddr, valid_rows, 4, false);
        state_cycles += issue_yz_selected_rows(
            kFactorizedStateDramBaseAddr, changed_rows, 4, true);
        (void)kFactorizedOldYDramBaseAddr;
    } else
#endif
    {
        state_cycles = (state_read_bursts + state_write_bursts) *
            kDramAvgCyclesPerTransaction;
        snapshot_cycles = result->factorized_old_y_hbm_bursts *
            kDramAvgCyclesPerTransaction;
    }
    result->factorized_state_hbm_cycles += state_cycles;
    result->factorized_old_y_hbm_cycles += snapshot_cycles;

    result->factorized_native_pe_cycles +=
        (result->row_mean_scalar_ops + kCombineThroughputPerCycle - 1LL) /
        kCombineThroughputPerCycle;
    result->factorized_pe_dynamic_energy_pj +=
        despoina::official_energy::row_mean_scalar_energy_pj(
            result->row_mean_scalar_ops);
    result->factorized_native_owner_cycles +=
        result->factorized_native_pe_cycles + state_cycles + snapshot_cycles;
}


std::vector<float> DualModeSpMM::golden_dense_matmul_rows(
    const std::vector<float>& Y,
    const std::vector<int>& rows) const {

    const size_t expected_w = static_cast<size_t>(d_) * static_cast<size_t>(d_out_);
    if (d_ <= 0 || d_out_ <= 0 || weights_.size() != expected_w ||
        Y.size() < static_cast<size_t>(n_) * static_cast<size_t>(d_)) {
        return std::vector<float>();
    }

    std::vector<float> out(rows.size() * static_cast<size_t>(d_out_), 0.0f);
    for (size_t k = 0; k < rows.size(); k++) {
        const int row = rows[k];
        if (row < 0 || row >= n_) {
            return std::vector<float>();
        }
        const float* y = Y.data() + static_cast<size_t>(row) * static_cast<size_t>(d_);
        float* z = out.data() + k * static_cast<size_t>(d_out_);
        for (int j = 0; j < d_out_; j++) {
            float sum = 0.0f;
            for (int i = 0; i < d_; i++) {
                sum += y[i] *
                       weights_[static_cast<size_t>(i) * static_cast<size_t>(d_out_) +
                                static_cast<size_t>(j)];
            }
            z[j] = sum;
        }
    }
    return out;
}

std::vector<float> DualModeSpMM::golden_full_forward_rows(
    const WorkStealingScheduler::CSRMatrix& a_new,
    const std::vector<int>& rows) const {

    const size_t expected_w = static_cast<size_t>(d_) * static_cast<size_t>(d_out_);
    const size_t expected_f = static_cast<size_t>(n_) * static_cast<size_t>(d_);
    if (n_ <= 0 || d_ <= 0 || d_out_ <= 0 || a_new.n != n_ ||
        a_new.row_ptr.size() < static_cast<size_t>(n_ + 1) ||
        a_new.col_idx.size() != a_new.values.size() ||
        weights_.size() != expected_w || f_current_.size() < expected_f) {
        return std::vector<float>();
    }

    std::vector<float> out(rows.size() * static_cast<size_t>(d_out_), 0.0f);
    std::vector<float> y_row(static_cast<size_t>(d_), 0.0f);
    for (size_t k = 0; k < rows.size(); k++) {
        const int row = rows[k];
        if (row < 0 || row >= n_) {
            return std::vector<float>();
        }
        const int start = a_new.row_ptr[static_cast<size_t>(row)];
        const int end = a_new.row_ptr[static_cast<size_t>(row + 1)];
        if (start < 0 || end < start ||
            end > static_cast<int>(a_new.col_idx.size()) ||
            end > static_cast<int>(a_new.values.size())) {
            return std::vector<float>();
        }

        std::fill(y_row.begin(), y_row.end(), 0.0f);
        for (int p = start; p < end; p++) {
            const int col = a_new.col_idx[static_cast<size_t>(p)];
            if (col < 0 || col >= n_) {
                return std::vector<float>();
            }
            const float a_val = a_new.values[static_cast<size_t>(p)];
            const float* f =
                f_current_.data() + static_cast<size_t>(col) * static_cast<size_t>(d_);
            for (int i = 0; i < d_; i++) {
                y_row[static_cast<size_t>(i)] += a_val * f[i];
            }
        }
        if (end > start) {
            const float inv_degree = 1.0f / static_cast<float>(end - start);
            for (int i = 0; i < d_; ++i) {
                y_row[static_cast<size_t>(i)] *= inv_degree;
            }
        }

        float* z = out.data() + k * static_cast<size_t>(d_out_);
        for (int j = 0; j < d_out_; j++) {
            float sum = 0.0f;
            for (int i = 0; i < d_; i++) {
                sum += y_row[static_cast<size_t>(i)] *
                       weights_[static_cast<size_t>(i) * static_cast<size_t>(d_out_) +
                                static_cast<size_t>(j)];
            }
            z[j] = sum;
        }
    }
    return out;
}

std::vector<float> DualModeSpMM::golden_full_spmm(
    const WorkStealingScheduler::CSRMatrix& a_new) const {

    const size_t expected_f = static_cast<size_t>(n_) * static_cast<size_t>(d_);
    if (n_ <= 0 || d_ <= 0 || a_new.n != n_ ||
        a_new.row_ptr.size() < static_cast<size_t>(n_ + 1) ||
        a_new.col_idx.size() != a_new.values.size() ||
        f_current_.size() < expected_f) {
        return std::vector<float>();
    }

    std::vector<float> out(static_cast<size_t>(n_) * static_cast<size_t>(d_), 0.0f);
    for (int row = 0; row < n_; row++) {
        const int start = a_new.row_ptr[static_cast<size_t>(row)];
        const int end = a_new.row_ptr[static_cast<size_t>(row + 1)];
        if (start < 0 || end < start ||
            end > static_cast<int>(a_new.col_idx.size()) ||
            end > static_cast<int>(a_new.values.size())) {
            return std::vector<float>();
        }
        float* y_row = out.data() + static_cast<size_t>(row) * static_cast<size_t>(d_);
        for (int p = start; p < end; p++) {
            const int col = a_new.col_idx[static_cast<size_t>(p)];
            if (col < 0 || col >= n_) {
                return std::vector<float>();
            }
            const float a_val = a_new.values[static_cast<size_t>(p)];
            const float* f =
                f_current_.data() + static_cast<size_t>(col) * static_cast<size_t>(d_);
            for (int j = 0; j < d_; j++) {
                y_row[static_cast<size_t>(j)] += a_val * f[j];
            }
        }
    }
    DynamicRowMeanOperator::apply_full_row_mean(
        &out, DynamicRowMeanOperator::degrees(a_new), d_);
    return out;
}

std::vector<float> DualModeSpMM::golden_full_spmm_rows(
    const WorkStealingScheduler::CSRMatrix& a_new,
    const std::vector<int>& rows) const {

    const size_t expected_f = static_cast<size_t>(n_) * static_cast<size_t>(d_);
    if (n_ <= 0 || d_ <= 0 || a_new.n != n_ ||
        a_new.row_ptr.size() < static_cast<size_t>(n_ + 1) ||
        a_new.col_idx.size() != a_new.values.size() ||
        f_current_.size() < expected_f) {
        return std::vector<float>();
    }

    std::vector<float> out(rows.size() * static_cast<size_t>(d_), 0.0f);
    for (size_t idx = 0; idx < rows.size(); idx++) {
        const int row = rows[idx];
        if (row < 0 || row >= n_) {
            return std::vector<float>();
        }
        const int start = a_new.row_ptr[static_cast<size_t>(row)];
        const int end = a_new.row_ptr[static_cast<size_t>(row + 1)];
        if (start < 0 || end < start ||
            end > static_cast<int>(a_new.col_idx.size()) ||
            end > static_cast<int>(a_new.values.size())) {
            return std::vector<float>();
        }
        float* y_row = out.data() + idx * static_cast<size_t>(d_);
        for (int p = start; p < end; p++) {
            const int col = a_new.col_idx[static_cast<size_t>(p)];
            if (col < 0 || col >= n_) {
                return std::vector<float>();
            }
            const float a_val = a_new.values[static_cast<size_t>(p)];
            const float* f =
                f_current_.data() + static_cast<size_t>(col) * static_cast<size_t>(d_);
            for (int j = 0; j < d_; j++) {
                y_row[static_cast<size_t>(j)] += a_val * f[j];
            }
        }
        if (end > start) {
            const float inv = 1.0f / static_cast<float>(end - start);
            for (int j = 0; j < d_; ++j) {
                y_row[static_cast<size_t>(j)] = static_cast<float>(
                    y_row[static_cast<size_t>(j)] * inv);
            }
        }
    }
    return out;
}

void DualModeSpMM::apply_cim_noise(std::vector<float>& output, int rows, int cols) const {
    if (!cim_noise_ || rows <= 0 || cols <= 0 || output.empty()) {
        return;
    }
    static std::mt19937 rng(12345);
    const float sigma_rel = static_cast<float>(neurosim::noise::THERMAL_NOISE_MV) / 500.0f;
    std::normal_distribution<float> dist(0.0f, sigma_rel);
    for (size_t i = 0; i < output.size(); i++) {
        output[i] *= (1.0f + dist(rng));
    }
}

DualModeSpMM::StepResult::AccuracyMetrics DualModeSpMM::compute_accuracy_metrics(
    const std::vector<float>& cim_output,
    const std::vector<float>& golden,
    const CimHardwareRunResult& hw) const {

    StepResult::AccuracyMetrics m;
    m.executed = true;
    m.hw_expected_rows = hw.expected_rows;
    m.hw_received_unique_rows = hw.received_unique_rows;
    m.hw_duplicate_packets = hw.duplicate_packets;
    m.hw_invalid_packets = hw.invalid_packets;
    m.hw_timeout = hw.timeout;
    m.hw_complete = hw.ok;
    const size_t compared_elements = std::min(cim_output.size(), golden.size());
    m.elements_compared = static_cast<int>(compared_elements);
    if (d_out_ > 0) {
        m.rows_compared = static_cast<int>(compared_elements / static_cast<size_t>(d_out_));
    }

    const bool shape_ok = (cim_output.size() == golden.size());
    m.valid = hw.ok && shape_ok;
    if (!m.valid) {
        m.pass_hitrate = false;
        m.pass = false;
        if (fail_closed_accuracy_) {
            throw std::runtime_error("official DCIM accuracy result is invalid");
        }
        return m;
    }

    if (cim_output.empty()) {
        m.rmse = 0.0;
        m.mean_abs_error = 0.0;
        m.max_abs_error = 0.0;
        m.cosine_similarity = 1.0;
        m.rel_error_mean = 0.0;
        m.rel_error_max = 0.0;
        m.hit_rate_rel_1pct = 1.0;
        m.hit_rate_rel_2pct = 1.0;
        m.hit_rate_rel_5pct = 1.0;
        m.pass_hitrate = true;
        m.pass = true;
        return m;
    }

    double sum_sq = 0.0;
    double sum_abs = 0.0;
    double sum_rel = 0.0;
    double max_abs = 0.0;
    double max_rel = 0.0;
    double dot = 0.0;
    double norm_a = 0.0;
    double norm_b = 0.0;
    long long hit1 = 0;
    long long hit2 = 0;
    long long hit5 = 0;
    for (size_t i = 0; i < cim_output.size(); i++) {
        const double a = static_cast<double>(cim_output[i]);
        const double b = static_cast<double>(golden[i]);
        const double e = a - b;
        const double abs_e = std::abs(e);
        const double rel_e = abs_e / std::max(std::abs(b), m.rel_error_epsilon);
        const bool abs_ok = (abs_e <= m.abs_error_tolerance);
        sum_sq += e * e;
        sum_abs += abs_e;
        sum_rel += rel_e;
        max_abs = std::max(max_abs, abs_e);
        max_rel = std::max(max_rel, rel_e);
        if (abs_ok || rel_e <= 0.01) hit1++;
        if (abs_ok || rel_e <= 0.02) hit2++;
        if (abs_ok || rel_e <= 0.05) hit5++;
        dot += a * b;
        norm_a += a * a;
        norm_b += b * b;
    }

    const double total = static_cast<double>(cim_output.size());
    m.rmse = std::sqrt(sum_sq / total);
    m.mean_abs_error = sum_abs / total;
    m.max_abs_error = max_abs;
    m.rel_error_mean = sum_rel / total;
    m.rel_error_max = max_rel;
    m.hit_rate_rel_1pct = static_cast<double>(hit1) / total;
    m.hit_rate_rel_2pct = static_cast<double>(hit2) / total;
    m.hit_rate_rel_5pct = static_cast<double>(hit5) / total;

    if (norm_a == 0.0 && norm_b == 0.0) {
        m.cosine_similarity = 1.0;
    } else if (norm_a == 0.0 || norm_b == 0.0) {
        m.cosine_similarity = 0.0;
    } else {
        const double raw_cosine = dot / (std::sqrt(norm_a) * std::sqrt(norm_b));
        m.cosine_similarity = std::max(-1.0, std::min(1.0, raw_cosine));
    }

    m.pass_hitrate =
        despoina::official_accuracy::same_domain_pass(
            cim_output, golden, 0.05, m.abs_error_tolerance,
            m.hitrate_pass_threshold);
    m.pass = m.pass_hitrate;
    if (fail_closed_accuracy_ && !m.pass) {
        throw std::runtime_error("official DCIM accuracy comparison failed");
    }
    return m;
}

DualModeSpMM::StepResult::YAccuracy DualModeSpMM::compute_y_accuracy_metrics(
    const std::vector<float>& y_eval,
    const std::vector<float>& y_golden) const {

    StepResult::YAccuracy m;
    if (y_eval.size() != y_golden.size()) {
        m.valid = false;
        m.exact = false;
        m.cosine_similarity = 0.0;
        return m;
    }

    m.valid = true;
    if (y_eval.empty()) {
        m.exact = true;
        m.rmse = 0.0;
        m.cosine_similarity = 1.0;
        return m;
    }

    double sum_sq = 0.0;
    double dot = 0.0;
    double norm_a = 0.0;
    double norm_b = 0.0;
    for (size_t i = 0; i < y_eval.size(); i++) {
        const double a = static_cast<double>(y_eval[i]);
        const double b = static_cast<double>(y_golden[i]);
        const double e = a - b;
        sum_sq += e * e;
        dot += a * b;
        norm_a += a * a;
        norm_b += b * b;
    }

    m.rmse = std::sqrt(sum_sq / static_cast<double>(y_eval.size()));
    if (norm_a == 0.0 && norm_b == 0.0) {
        m.cosine_similarity = 1.0;
    } else if (norm_a == 0.0 || norm_b == 0.0) {
        m.cosine_similarity = 0.0;
    } else {
        const double raw_cos = dot / (std::sqrt(norm_a) * std::sqrt(norm_b));
        m.cosine_similarity = std::max(-1.0, std::min(1.0, raw_cos));
    }
    m.exact = (m.rmse <= 1e-6) && (m.cosine_similarity >= 1.0 - 1e-6);
    return m;
}

void DualModeSpMM::apply_delta_f(const DeltaConverter::DeltaFParsed& df) {
    if (df.feature_dim != d_) return;
    for (size_t idx = 0; idx < df.nodes.size(); idx++) {
        int node = df.nodes[idx];
        if (node < 0 || node >= n_) continue;
        const float* dv = df.deltas.data() + idx * d_;
        float* row = f_current_.data() + static_cast<size_t>(node) * static_cast<size_t>(d_);
        for (int j = 0; j < d_; j++) {
            row[j] += dv[j];
        }
    }
    // FeatureStore update traffic + (optional) DRAMsim3 transactions for accuracy
    indexer_.apply_feature_store_updates(df.nodes, d_);

    // Cache coherence: invalidate cached rows for updated nodes so subsequent SpMMs
    // fetch the updated feature values.
    if (pe_ != nullptr) {
        pe_->invalidate_f_rows(df.nodes);
    }
}

bool DualModeSpMM::should_incremental(int k, int l, int nnz, double avg_deg, double threshold) {
    if (nnz <= 0) return false;
    double rho_e = static_cast<double>(k) / static_cast<double>(nnz);
    double rho_w = static_cast<double>(l) * avg_deg / static_cast<double>(nnz);
    return (rho_e + rho_w) < threshold;
}

PEArraySpMM::Stats DualModeSpMM::run_pe_job(const WorkStealingScheduler::CSRMatrix& csr,
                                            const float* f_matrix,
                                            bool keep_shared_f_cache,
                                            uint64_t matrix_epoch) {
    pe_->set_streaming_output_enabled(false);
    pe_->load_spmm(csr.row_ptr.data(), csr.col_idx.data(), csr.values.data(),
                   csr.n, csr.nnz, f_matrix, d_, keep_shared_f_cache, matrix_epoch);
    while (!pe_->is_done()) {
        pe_->tick();
        if (gcp_ != nullptr) {
            gcp_->tick();
        }
    }
    return pe_->get_stats();
}

DualModeSpMM::PhaseCapture DualModeSpMM::run_pe_job_capture(
    const WorkStealingScheduler::CSRMatrix& csr,
    const float* f_matrix,
    bool keep_shared_f_cache,
    uint64_t matrix_epoch) {
    PhaseCapture capture;
    pe_->set_streaming_output_enabled(true);
    pe_->load_spmm(csr.row_ptr.data(), csr.col_idx.data(), csr.values.data(),
                   csr.n, csr.nnz, f_matrix, d_, keep_shared_f_cache, matrix_epoch);
    std::vector<float> row(static_cast<size_t>(d_), 0.0f);
    while (!pe_->is_done()) {
        pe_->tick();
        while (pe_->has_completed_row()) {
            const int row_id = pe_->pop_completed_row(row.data(), d_);
            if (row_id >= 0) {
                capture.row_ready_cycle[row_id] = pe_->get_total_cycles();
            }
        }
        if (gcp_ != nullptr) {
            gcp_->tick();
        }
    }
    while (pe_->has_completed_row()) {
        const int row_id = pe_->pop_completed_row(row.data(), d_);
        if (row_id >= 0) {
            capture.row_ready_cycle[row_id] = pe_->get_total_cycles();
        }
    }
    pe_->set_streaming_output_enabled(false);
    capture.stats = pe_->get_stats();
    return capture;
}

const std::vector<float>& DualModeSpMM::weights_for_layer(int layer_id) const {
    if (layer_id <= 1) return weights_;
    if (layer_id == 2) return weights_l2_;
    if (layer_id == 3) return weights_l3_;
    if (layer_id == 4) return weights_l4_;
    if (layer_id == 5) return weights_l5_;
    throw std::out_of_range("no resident W for requested layer");
}

OfficialDcimWeightBankKey DualModeSpMM::official_dcim_weight_key_for_layer(int layer_id) {
    const std::vector<float>& w = weights_for_layer(layer_id);
    const size_t expected = static_cast<size_t>(d_) * static_cast<size_t>(d_out_);
    if (w.size() != expected) {
        throw std::runtime_error("Despoina official DCIM bank key shape mismatch");
    }
    const uint64_t source_hash = despoina_r5::fnv1a64(
        w.data(), w.size() * sizeof(float));
    if (source_hash == 0) throw std::runtime_error("Despoina official DCIM source hash is invalid");
    uint64_t& observed_hash = observed_weight_hash_by_layer_[layer_id];
    uint64_t& version = weight_version_by_layer_[layer_id];
    if (observed_hash == 0) {
        observed_hash = source_hash;
        version = 1;
    } else if (observed_hash != source_hash) {
        observed_hash = source_hash;
        ++version;
    }
    OfficialDcimWeightBankKey key;
    key.layer_id = layer_id;
    key.version = version;
    key.source_hash = source_hash;
    key.rows = d_;
    key.cols = d_out_;
    return key;
}

void DualModeSpMM::official_dcim_schedule_layer_ahead_prefetch(int current_layer_id) {
#if DESPOINA_V3_FUNCTIONAL_DCIM
    if (cim_ == nullptr || cim_->get_compute_mode() != CiMComputeMode::DCIM_DIGITAL ||
        gcp_ == nullptr || gcp_->arg_parse == nullptr) {
        return;
    }
    const int num_layers = std::max(1, gcp_->arg_parse->get_num_layers());
    const int next_layer = current_layer_id >= num_layers ? 1 : current_layer_id + 1;
    const OfficialDcimWeightBankKey next_key = official_dcim_weight_key_for_layer(next_layer);
    constexpr uint64_t kCentralProxyProgramCycles = 128;  // 64 x 256B, 2 cyc/access
    if (cim_->official_dcim_prefetch_inactive_bank(
            weights_for_layer(next_layer), next_key, kCentralProxyProgramCycles)) {
        pending_weight_program_event_ = true;
        pending_weight_program_cycles_ += kCentralProxyProgramCycles;
        pending_weight_program_tile_events_ += static_cast<uint64_t>(cim_->get_num_tiles());
        pending_weight_source_fp32_bytes_ +=
            static_cast<uint64_t>(weights_for_layer(next_layer).size()) * sizeof(float);
        pending_weight_programmed_int8_bytes_ +=
            static_cast<uint64_t>(weights_for_layer(next_layer).size()) *
            sizeof(int8_t) * static_cast<uint64_t>(cim_->get_num_tiles());
        ++pending_weight_bank_prefetch_events_;
    }
#else
    (void)current_layer_id;
#endif
}

bool DualModeSpMM::load_resident_weights_for_layer(int layer_id) {
    if (cim_ == nullptr) {
        throw std::runtime_error("DCIM unit is unavailable");
    }
    const std::vector<float>& w = weights_for_layer(layer_id);
    const size_t expected = static_cast<size_t>(d_) * static_cast<size_t>(d_out_);
    if (w.size() != expected) {
        throw std::runtime_error("resident layer weight shape mismatch");
    }
    const OfficialDcimWeightBankKey bank_key = official_dcim_weight_key_for_layer(layer_id);
    const despoina_r5::ResidentWeightKey key{
        bank_key.layer_id, bank_key.version, bank_key.source_hash,
        bank_key.rows, bank_key.cols};
    const bool needs_load = resident_weight_contract_.requires_load(key);
    if (needs_load) {
        bool bank_swap = false;
#if DESPOINA_V3_FUNCTIONAL_DCIM
        if (cim_->get_compute_mode() == CiMComputeMode::DCIM_DIGITAL &&
            cim_->official_dcim_inactive_bank_matches(bank_key)) {
            const uint64_t available = cim_->get_native_trace_stats().elapsed_cycles;
            const uint64_t stall =
                cim_->official_dcim_swap_to_prefetched(bank_key, available);
            ++pending_weight_bank_swap_events_;
            pending_weight_bank_exposed_stall_cycles_ += stall;
            bank_swap = true;
        }
#endif
        if (!bank_swap) {
            // Cold/miss path. The central CACTI proxy is explicit and bounded;
            // subsequent layer transitions must use the prefetched bank.
            cim_->load_weights(w);
#if DESPOINA_V3_FUNCTIONAL_DCIM
            if (cim_->get_compute_mode() == CiMComputeMode::DCIM_DIGITAL) {
                cim_->official_dcim_bind_active_bank(bank_key);
                constexpr uint64_t kCentralProxyProgramCycles = 128;
                pending_weight_program_cycles_ += kCentralProxyProgramCycles;
                pending_weight_program_tile_events_ +=
                    static_cast<uint64_t>(cim_->get_num_tiles());
                pending_weight_source_fp32_bytes_ +=
                    static_cast<uint64_t>(w.size()) * sizeof(float);
                pending_weight_programmed_int8_bytes_ +=
                    static_cast<uint64_t>(w.size()) * sizeof(int8_t) *
                    static_cast<uint64_t>(cim_->get_num_tiles());
            }
#endif
        }
        const std::vector<int8_t>& q = cim_->get_quantized_weights();
        if (q.size() != expected) {
            throw std::runtime_error("resident INT8 weight shape mismatch after load");
        }
        const uint64_t q_hash = despoina_r5::fnv1a64(q.data(), q.size());
        resident_weight_contract_.commit_load(key, q_hash);
        pending_weight_program_event_ = true;
        if (pending_weight_source_fp32_bytes_ == 0) {
            pending_weight_program_cycles_ += cim_->get_last_weight_program_cycles();
            pending_weight_program_tile_events_ += static_cast<uint64_t>(cim_->get_num_tiles());
            pending_weight_source_fp32_bytes_ += cim_->get_last_weight_source_fp32_bytes();
            pending_weight_programmed_int8_bytes_ +=
                cim_->get_last_weight_programmed_int8_bytes();
        }
        pending_weight_tile_program_done_cycles_ =
            cim_->get_last_weight_tile_program_done_cycles();
    }
    const std::vector<int8_t>& resident_q = cim_->get_quantized_weights();
    const uint64_t resident_q_hash = resident_q.empty()
        ? 0 : despoina_r5::fnv1a64(resident_q.data(), resident_q.size());
    const std::vector<uint64_t>& per_tile_hash =
        cim_->get_resident_weight_hash_per_tile();
    const std::vector<uint8_t>& per_tile_loaded =
        cim_->get_resident_weight_loaded_per_tile();
    const bool every_tile_matches =
        per_tile_hash.size() == static_cast<size_t>(cim_->get_num_tiles()) &&
        per_tile_loaded.size() == static_cast<size_t>(cim_->get_num_tiles()) &&
        std::all_of(per_tile_hash.begin(), per_tile_hash.end(),
                    [resident_q_hash](uint64_t h) { return h == resident_q_hash; }) &&
        std::all_of(per_tile_loaded.begin(), per_tile_loaded.end(),
                    [](uint8_t loaded) { return loaded == 1; }) &&
        cim_->all_resident_tiles_ready();
    if (resident_q.size() != expected ||
        !every_tile_matches ||
        !resident_weight_contract_.validate(key, resident_q_hash)) {
        throw std::runtime_error("stale or wrong resident W detected");
    }
    official_dcim_schedule_layer_ahead_prefetch(layer_id);
    return needs_load;
}

void DualModeSpMM::begin_runtime_timestep_trace() {
    runtime_layer_trace_.clear();
    runtime_activation_trace_.clear();
}

void DualModeSpMM::set_runtime_activation_trace(
    int layer_id,
    const std::vector<float>& norm_relu_output,
    const std::vector<float>& next_layer_delta,
    const std::vector<int>& next_layer_tau_rows) {
    RuntimeActivationTrace& trace = runtime_activation_trace_[layer_id];
    trace.norm_relu_output = norm_relu_output;
    trace.next_layer_delta = next_layer_delta;
    trace.next_layer_tau_rows = next_layer_tau_rows;
}

const DualModeSpMM::RuntimeLayerTrace*
DualModeSpMM::get_runtime_layer_trace(int layer_id) const {
    const auto it = runtime_layer_trace_.find(layer_id);
    return it == runtime_layer_trace_.end() ? nullptr : &it->second;
}

const DualModeSpMM::RuntimeActivationTrace*
DualModeSpMM::get_runtime_activation_trace(int layer_id) const {
    const auto it = runtime_activation_trace_.find(layer_id);
    return it == runtime_activation_trace_.end() ? nullptr : &it->second;
}

bool DualModeSpMM::capture_runtime_layer_trace(
    int layer_id,
    const std::vector<int>& graph_rows) {
    if (cim_ == nullptr) return false;
    const size_t rows = graph_rows.size();
    const size_t expected_inputs = rows * static_cast<size_t>(d_);
    const size_t expected_weights =
        static_cast<size_t>(d_) * static_cast<size_t>(d_out_);
    const auto& q_inputs = cim_->get_last_quantized_inputs();
    const auto& scales = cim_->get_last_input_scales();
    const auto& valid = cim_->get_last_input_trace_valid();
    const auto& q_weights = cim_->get_quantized_weights();
    if (q_inputs.size() != expected_inputs || scales.size() != rows ||
        valid.size() != rows || q_weights.size() != expected_weights) {
        return false;
    }
    for (uint8_t v : valid) {
        if (v != 1) return false;
    }
    RuntimeLayerTrace trace;
    trace.executed = true;
    trace.graph_rows = graph_rows;
    trace.quantized_inputs = q_inputs;
    trace.input_scales = scales;
    trace.quantized_weights = q_weights;
    trace.weight_scale = cim_->get_weight_scale();
    runtime_layer_trace_[layer_id] = std::move(trace);
    return true;
}

bool DualModeSpMM::record_direct_dcim_accounting(
    const std::vector<uint64_t>& row_ready_cycles,
    long long producer_cycles,
    const CimHardwareRunResult& hw,
    StepResult& result) {
    despoina_v3::TimingStats timing;
    if (cim_ == nullptr || !hw.ok) {
        result.finalized_invalid_packets++;
        return false;
    }
    const NeuraCiM::NativeTraceStats& native = cim_->get_native_trace_stats();
    const uint64_t rows = static_cast<uint64_t>(hw.received_unique_rows);
    if (native.row_issue_events != rows || native.row_completion_events != rows ||
        native.functional_apply_events != rows ||
        native.bit_step_issue_events != rows * 8ULL ||
        native.bit_step_completion_events != rows * 8ULL ||
        native.full_int8_energy_charge_events != rows ||
        native.bit_step_temporal_order_violations != 0 ||
        native.elapsed_cycles != static_cast<uint64_t>(std::max(0LL, hw.cycles))) {
        result.finalized_invalid_packets++;
        return false;
    }
    timing.finalized_rows = rows;
    timing.packets = rows;
    timing.scalar_macs = despoina_v3::checked_scalar_macs(
        rows, static_cast<uint64_t>(d_), static_cast<uint64_t>(d_out_));
    timing.tile_busy_work_cycles = despoina_v3::TileBusyWorkCycles(rows * 1024ULL);
    timing.causal_stream_elapsed_cycles = despoina_v3::CausalStreamElapsedCycles(native.elapsed_cycles);
    const long long exposed_tail = (hw.native_last_completion_cycle > 0 ||
                                    hw.producer_last_enqueue_cycle > 0)
        ? hw.exposed_tail_cycles
        : static_cast<long long>(native.elapsed_cycles);  // t0/baseline direct path
    timing.exposed_tail_cycles = despoina_v3::ExposedTailCycles(
        static_cast<uint64_t>(std::max(0LL, exposed_tail)));
    timing.implementation_plus6_finish_cycle = native.elapsed_cycles;
    timing.implementation_plus6_exposed_cycles = native.elapsed_cycles;
    timing.overlap_cycles = static_cast<uint64_t>(std::max(0LL, hw.overlap_cycles));
    timing.backpressure_cycles = hw.cim_pipeline.output_fifo_full_stall_cycles +
                                 hw.cim_pipeline.inject_no_empty_lane_cycles;
    timing.max_staging_rows = static_cast<uint64_t>(std::max(0LL, hw.max_staging_rows));
    timing.max_fifo_rows = std::max(native.max_input_fifo_rows,
                                    native.max_output_fifo_rows);
    timing.bit_serial_steps = 8;
    timing.implementation_drain_cycles = 0;
    timing.physical_tiles = static_cast<uint64_t>(cim_->get_num_tiles());
    timing.per_tile_raw_bit_ops_per_cycle = 128;
    timing.cycles_per_bit_step = 128;
    timing.per_tile_row_service_cycles = 1024;
    timing.bit_step_issue_events = native.row_issue_events * 8ULL;
    timing.bit_step_completion_events = native.row_completion_events * 8ULL;
    timing.first_bit_step_issue_cycle = native.first_bit_step_issue_cycle;
    timing.last_bit_step_completion_cycle = native.last_bit_step_completion_cycle;
    timing.bit_step_trace_fnv1a64 = native.bit_step_trace_fnv1a64;
    timing.tile_dispatch_events = native.row_issue_events;
    timing.weight_program_cycles = pending_weight_program_cycles_;
    timing.weight_program_tile_events = pending_weight_program_tile_events_;
    timing.weight_source_fp32_bytes = pending_weight_program_event_
        ? pending_weight_source_fp32_bytes_ : 0;
    timing.weight_programmed_int8_bytes = pending_weight_program_event_
        ? pending_weight_programmed_int8_bytes_ : 0;
    timing.overlap_sensitivity_finish_cycle = native.elapsed_cycles;
    timing.overlap_sensitivity_exposed_cycles = timing.exposed_tail_cycles.value;
    const despoina_v3::DirectEnergyEvents events =
        despoina_v3::direct_energy_events_once(
            timing.finalized_rows, static_cast<uint64_t>(d_),
            static_cast<uint64_t>(d_out_));

    result.finalized_rows += static_cast<long long>(timing.finalized_rows);
    result.finalized_packets += static_cast<long long>(timing.packets);
    result.dcim_scalar_macs += static_cast<long long>(timing.scalar_macs);
    despoina_v3::accumulate(
        result.tile_busy_work_cycles, timing.tile_busy_work_cycles);
    result.dcim_bit_serial_steps = static_cast<long long>(timing.bit_serial_steps);
    result.dcim_drain_cycles +=
        static_cast<long long>(timing.implementation_drain_cycles);
    result.dcim_streams += 1;
    despoina_v3::accumulate(
        result.causal_stream_elapsed_cycles,
        timing.causal_stream_elapsed_cycles);
    result.finalized_stream_cycles_plus6 +=
        static_cast<long long>(timing.implementation_plus6_finish_cycle);
    despoina_v3::accumulate(
        result.exposed_tail_cycles, timing.exposed_tail_cycles);
    result.finalized_exposed_cycles_plus6 +=
        static_cast<long long>(timing.implementation_plus6_exposed_cycles);
    result.finalized_backpressure_cycles +=
        static_cast<long long>(timing.backpressure_cycles);
    result.finalized_max_staging_rows = std::max(
        result.finalized_max_staging_rows,
        static_cast<long long>(timing.max_staging_rows));
    result.finalized_max_fifo_rows = std::max(
        result.finalized_max_fifo_rows,
        static_cast<long long>(timing.max_fifo_rows));
    result.finalized_timeout += hw.timeout ? 1 : 0;
    result.finalized_duplicate_packets += hw.duplicate_packets;
    result.finalized_invalid_packets += hw.invalid_packets;
    result.dcim_activation_quantized_elements +=
        static_cast<long long>(events.activation_quantized_elements);
    result.dcim_dequantized_elements +=
        static_cast<long long>(events.dequantized_elements);
    result.dcim_comparator_tree_passes +=
        static_cast<long long>(events.comparator_tree_passes);

    pending_weight_program_event_ = false;
    pending_weight_program_cycles_ = 0;
    pending_weight_source_fp32_bytes_ = 0;
    pending_weight_programmed_int8_bytes_ = 0;
    pending_weight_program_tile_events_ = 0;
    pending_weight_tile_program_done_cycles_.clear();

    auto& c = result.arch_ctrs;
    c.dcim_enabled = true;
    c.streaming_enabled = true;
    // R4 authoritative owner for compute-only exposed cycles.  The generic
    // population pass is forbidden from adding this field in functional DCIM.
    c.dcim_exposed_cycles += timing.exposed_tail_cycles.value;
    c.tile_busy_work_cycles += timing.tile_busy_work_cycles.value;
    c.causal_stream_elapsed_cycles += timing.causal_stream_elapsed_cycles.value;
    c.exposed_tail_cycles += timing.exposed_tail_cycles.value;
    c.digital_stream_rows += timing.finalized_rows;
    c.digital_stream_mac_ops += timing.scalar_macs;
    c.digital_stream_count += 1;
    c.digital_stream_drain_sensitivity_cycles +=
        timing.implementation_drain_cycles;
    c.digital_stream_plus6_exposed_cycles +=
        timing.implementation_plus6_exposed_cycles;
    c.digital_stream_packets += timing.packets;
    c.digital_stream_duplicate_packets +=
        static_cast<uint64_t>(std::max(0, hw.duplicate_packets));
    c.digital_stream_invalid_packets +=
        static_cast<uint64_t>(std::max(0, hw.invalid_packets));
    c.digital_stream_timeouts += hw.timeout ? 1U : 0U;
    c.digital_stream_activation_quantized_elements +=
        events.activation_quantized_elements;
    c.digital_stream_dequantized_elements += events.dequantized_elements;
    c.digital_stream_comparator_tree_passes += events.comparator_tree_passes;
    c.digital_stream_bit_step_issue_events += timing.bit_step_issue_events;
    c.digital_stream_bit_step_completion_events += timing.bit_step_completion_events;
    c.digital_stream_full_int8_energy_charge_events +=
        native.full_int8_energy_charge_events;
    c.digital_stream_tile_dispatch_events += timing.tile_dispatch_events;
    c.digital_stream_weight_program_cycles += timing.weight_program_cycles;
    c.digital_stream_weight_program_tile_events += timing.weight_program_tile_events;
    c.digital_stream_weight_source_fp32_bytes += timing.weight_source_fp32_bytes;
    c.digital_stream_weight_programmed_int8_bytes += timing.weight_programmed_int8_bytes;
    c.digital_stream_weight_bank_prefetch_events += pending_weight_bank_prefetch_events_;
    c.digital_stream_weight_bank_swap_events += pending_weight_bank_swap_events_;
    c.digital_stream_weight_bank_exposed_stall_cycles +=
        pending_weight_bank_exposed_stall_cycles_;
    pending_weight_bank_prefetch_events_ = 0;
    pending_weight_bank_swap_events_ = 0;
    pending_weight_bank_exposed_stall_cycles_ = 0;
    c.digital_stream_overlap_sensitivity_finish_cycles +=
        timing.overlap_sensitivity_finish_cycle;
    c.digital_stream_overlap_sensitivity_exposed_cycles +=
        timing.overlap_sensitivity_exposed_cycles;
    c.digital_stream_event_trace_xor ^= timing.bit_step_trace_fnv1a64;
    if (c.digital_stream_first_bit_step_issue_cycle == 0 ||
        (timing.first_bit_step_issue_cycle != 0 &&
         timing.first_bit_step_issue_cycle < c.digital_stream_first_bit_step_issue_cycle)) {
        c.digital_stream_first_bit_step_issue_cycle = timing.first_bit_step_issue_cycle;
    }
    c.digital_stream_last_bit_step_completion_cycle = std::max(
        c.digital_stream_last_bit_step_completion_cycle,
        timing.last_bit_step_completion_cycle);
    c.pe_cim_noc_flits += hw.pe_cim_noc_flits;
    c.pe_cim_noc_bytes += hw.pe_cim_noc_bytes;
    c.digital_stream_compute_cycles += timing.tile_busy_work_cycles.value;
    c.digital_stream_measured_cycles += timing.causal_stream_elapsed_cycles.value;
    c.digital_stream_overlap_cycles += timing.overlap_cycles;
    c.digital_stream_backpressure_cycles += timing.backpressure_cycles;
    c.digital_stream_tile_rows = 128;
    c.digital_stream_tile_cols = 128;
    c.digital_stream_lanes = static_cast<int>(timing.physical_tiles);
    return true;
}

bool DualModeSpMM::commit_finalized_dcim_rows(
    int layer_id,
    const std::vector<int>& input_rows,
    const std::vector<uint64_t>& input_ready_cycles,
    long long producer_cycles,
    StepResult& result,
    const CimHardwareRunResult* concurrent_hw) {
    std::vector<int> rows;
    try {
        rows = despoina_v3::coalesce_rows_once(input_rows, n_);
    } catch (const std::exception&) {
        std::cerr << "[DCIM_COMMIT_INVARIANT] layer=" << layer_id
                  << " reason=row_id_domain_or_duplicate"
                  << " input_rows=" << input_rows.size() << std::endl;
        result.finalized_invalid_packets++;
        return false;
    }
    if (rows.size() != input_rows.size() || input_ready_cycles.size() != rows.size()) {
        std::cerr << "[DCIM_COMMIT_INVARIANT] layer=" << layer_id
                  << " reason=row_or_ready_cardinality"
                  << " input_rows=" << input_rows.size()
                  << " coalesced_rows=" << rows.size()
                  << " ready_cycles=" << input_ready_cycles.size() << std::endl;
        result.finalized_duplicate_packets +=
            static_cast<long long>(input_rows.size() - std::min(input_rows.size(), rows.size()));
        return false;
    }
    if (rows.empty()) {
        return true;
    }

    // Publication timing and the closed-loop hardware path are intentionally
    // fail-closed: Figure 8 is 128x128 and uses real DCIM_DIGITAL INT8
    // activation/weight quantization.  Variable dimensions are tested only by
    // the side-effect-free CPU functional reference.
    try {
        despoina_v3::assert_publication_cycle_shape(
            static_cast<uint64_t>(d_), static_cast<uint64_t>(d_out_));
    } catch (const std::exception&) {
        std::cerr << "[DCIM_COMMIT_INVARIANT] layer=" << layer_id
                  << " reason=publication_shape"
                  << " d_in=" << d_ << " d_out=" << d_out_ << std::endl;
        result.finalized_invalid_packets++;
        return false;
    }
    if (cim_ == nullptr ||
        cim_->get_compute_mode() != CiMComputeMode::DCIM_DIGITAL ||
        cim_->get_fp_activations()) {
        std::cerr << "[DCIM_COMMIT_INVARIANT] layer=" << layer_id
                  << " reason=dcim_mode"
                  << " cim_present=" << (cim_ != nullptr)
                  << " fp_activations=" << (cim_ != nullptr && cim_->get_fp_activations())
                  << std::endl;
        result.finalized_invalid_packets++;
        return false;
    }

    const std::vector<float>* y_state = nullptr;
    if (layer_id <= 1) {
        y_state = &y_current_;
    } else {
        const auto it = y_layer_state_.find(layer_id);
        if (it == y_layer_state_.end()) {
            std::cerr << "[DCIM_COMMIT_INVARIANT] layer=" << layer_id
                      << " reason=missing_y_layer_state" << std::endl;
            return false;
        }
        y_state = &it->second;
    }

    load_resident_weights_for_layer(layer_id);
    CimHardwareRunResult hw = concurrent_hw != nullptr
        ? *concurrent_hw
        : run_cim_hardware_rows_causal(*y_state, rows, input_ready_cycles);
    result.finalized_timeout += hw.timeout ? 1 : 0;
    result.finalized_duplicate_packets += hw.duplicate_packets;
    result.finalized_invalid_packets += hw.invalid_packets;
    if (!hw.ok || hw.received_unique_rows != static_cast<int>(rows.size()) ||
        hw.output.size() != rows.size() * static_cast<size_t>(d_out_)) {
        std::cerr << "[DCIM_COMMIT_INVARIANT] layer=" << layer_id
                  << " reason=hardware_result"
                  << " hw_ok=" << hw.ok
                  << " expected_rows=" << rows.size()
                  << " received_rows=" << hw.received_unique_rows
                  << " output_elements=" << hw.output.size()
                  << " expected_output_elements="
                  << rows.size() * static_cast<size_t>(d_out_)
                  << " duplicate_packets=" << hw.duplicate_packets
                  << " invalid_packets=" << hw.invalid_packets
                  << " timeout=" << hw.timeout << std::endl;
        return false;
    }

    if (!capture_runtime_layer_trace(layer_id, rows)) {
        std::cerr << "[DCIM_COMMIT_INVARIANT] layer=" << layer_id
                  << " reason=runtime_trace_capture"
                  << " rows=" << rows.size()
                  << " q_inputs=" << cim_->get_last_quantized_inputs().size()
                  << " scales=" << cim_->get_last_input_scales().size()
                  << " valid=" << cim_->get_last_input_trace_valid().size()
                  << " q_weights=" << cim_->get_quantized_weights().size()
                  << std::endl;
        return false;
    }
    if (!record_direct_dcim_accounting(
            input_ready_cycles, producer_cycles, hw, result)) {
        const NeuraCiM::NativeTraceStats& native = cim_->get_native_trace_stats();
        std::cerr << "[DCIM_COMMIT_INVARIANT] layer=" << layer_id
                  << " reason=direct_accounting"
                  << " rows=" << rows.size()
                  << " issue=" << native.row_issue_events
                  << " completion=" << native.row_completion_events
                  << " apply=" << native.functional_apply_events
                  << " bit_issue=" << native.bit_step_issue_events
                  << " bit_completion=" << native.bit_step_completion_events
                  << " energy=" << native.full_int8_energy_charge_events
                  << " order_violations=" << native.bit_step_temporal_order_violations
                  << " elapsed=" << native.elapsed_cycles
                  << " hw_cycles=" << hw.cycles << std::endl;
        return false;
    }
    // Z becomes architecturally visible only after the causal event engine has
    // emitted all eight bit-step completions and the dequant completion.
    std::vector<float>* z_state = layer_id <= 1
        ? &z_current_ : &z_layer_state_[layer_id];
    const size_t z_size = static_cast<size_t>(n_) * static_cast<size_t>(d_out_);
    if (z_state->size() != z_size) z_state->assign(z_size, 0.0f);
    // Streaming results carry Q(dY)·W and accumulate into the persistent Z;
    // the causal full-row path (no z_old_scale) still writes complete rows.
    const bool accumulate_z = hw.z_old_scale.size() == rows.size();
    for (size_t p = 0; p < rows.size(); ++p) {
        float* z_row = z_state->data() + static_cast<size_t>(rows[p]) * static_cast<size_t>(d_out_);
        const float* out_row = hw.output.data() + p * static_cast<size_t>(d_out_);
        if (accumulate_z) {
            const float scale = hw.z_old_scale[p];
            for (int j = 0; j < d_out_; ++j) {
                z_row[j] = static_cast<float>(
                    static_cast<float>(z_row[j] * scale) + out_row[j]);
            }
        } else {
            std::copy(out_row, out_row + d_out_, z_row);
        }
    }
    account_z_state_rmw(layer_id, rows, result);
    result.cim_rows = static_cast<int>(result.finalized_rows);
    // Compatibility counters now carry latency quantities.  Busy work remains
    // available only through tile_busy_work_cycles and can never masquerade as
    // combine latency or enter total_cycles.
    result.cim_cycles = static_cast<long long>(
        result.causal_stream_elapsed_cycles.value);
    result.cim_stream_cycles = static_cast<long long>(
        result.causal_stream_elapsed_cycles.value);
    result.cim_exposed_cycles = static_cast<long long>(
        result.exposed_tail_cycles.value);
    result.cim_pe_backpressure_cycles = result.finalized_backpressure_cycles;
    result.combine_cycles = 0;
    return true;
}

DualModeSpMM::StepResult DualModeSpMM::run_once_full(const WorkStealingScheduler::CSRMatrix& a_new) {
    StepResult r;
    r.mode_used = Mode::FULL;
    r.phase_mode_used = phase_mode_;
    indexer_.reset_counters();
    indexer_.build_from_csr(a_new);

    PEArraySpMM::Stats st = run_pe_job(a_new, f_current_.data());
    y_current_.assign(pe_->get_output(), pe_->get_output() + static_cast<size_t>(n_) * static_cast<size_t>(d_));
    row_mean_operator_.reset(a_new);
    DynamicRowMeanOperator::apply_full_row_mean(
        &y_current_, row_mean_operator_.degree(), d_);
    r.row_mean_enabled = true;
    r.row_mean_scaled_rows = n_;
    r.row_mean_scale_elements = static_cast<long long>(n_) * d_;
    r.row_mean_scalar_ops = r.row_mean_scale_elements;
    r.row_mean_degree_read_events = n_;
    r.row_mean_degree_read_bytes = static_cast<long long>(n_) * sizeof(std::int32_t);
    r.row_mean_reciprocal_events = std::count_if(
        row_mean_operator_.degree().begin(), row_mean_operator_.degree().end(),
        [](int degree) { return degree > 0; });
    r.y_accuracy = compute_y_accuracy_metrics(y_current_, golden_full_spmm(a_new));
    r.pe_stats = st;
    r.indexer_ctrs = indexer_.get_counters();
    r.y_dram_read_cycles = estimate_dram_read_cycles(n_, d_);
    r.y_dram_write_cycles = estimate_dram_write_cycles(n_, d_);
    const CiMMode effective_cim_mode = effective_cim_mode_for_arch();
    if (effective_cim_mode != CiMMode::NONE) {
        r.cim_mode = CiMMode::FULL_YNEW;
        r.cim_rows = n_;
        if (compute_accuracy_) {
            run_full_cim_validation(a_new, r);
        } else {
            r.cim_cycles = estimate_cim_cycles(cim_, r.cim_rows);
        }
        const long long z_bytes_all =
            static_cast<long long>(n_) * static_cast<long long>(d_out_) * kBytesPerElement;
        if (dedicated_z_sram_can_hold(z_bytes_all)) {
            r.z_dram_rw_cycles = 0;
            r.z_sram_accum_used = true;
            r.account_z_sram_bytes(z_bytes_all, z_bytes_all);
        } else {
#ifdef ENABLE_DRAMSIM3_PE
            if (yz_dram_ != nullptr) {
                r.z_dram_rw_cycles = issue_yz_all_rows(kZDramBaseAddr, n_, d_out_, true);
            } else {
                r.z_dram_rw_cycles = estimate_dram_write_cycles(n_, d_out_);
            }
#else
            r.z_dram_rw_cycles = estimate_dram_write_cycles(n_, d_out_);
#endif
        }
    } else {
        r.cim_mode = CiMMode::NONE;
    }
    if (arch_uses_digital_stream_backend()) {
        r.y_dram_read_cycles = 0;
        r.y_dram_write_cycles = 0;
        const long long producer_cycles =
            static_cast<long long>(r.pe_stats.total_cycles) +
            r.indexer_ctrs.index_dram_cycles +
            r.indexer_ctrs.feature_dram_cycles;
        const long long z_write_cycles = estimate_dram_write_cycles(n_, d_out_);
        apply_digital_stream_backend_accounting(r,
                                                producer_cycles,
                                                n_,
                                                d_,
                                                d_out_,
                                                z_write_cycles);
        populate_arch_ablation_counters(r,
                                        false,
                                        r.z_dram_rw_cycles > 0,
                                        false,
                                        1,
                                        -1,
                                        n_);
        return r;
    }
#if DESPOINA_V3_FUNCTIONAL_DCIM
    r.total_cycles = finalize_causal_dcim_step_cycles(
        r.pe_stats, r.indexer_ctrs,
        r.y_dram_read_cycles, r.y_dram_write_cycles,
        r.exposed_tail_cycles, r.z_dram_rw_cycles);
    r.cim_cycles = static_cast<long long>(r.causal_stream_elapsed_cycles.value);
    r.cim_stream_cycles = r.cim_cycles;
    r.cim_exposed_cycles = static_cast<long long>(r.exposed_tail_cycles.value);
#else
    r.total_cycles = finalize_full_step_cycles(r.pe_stats,
                                               r.indexer_ctrs,
                                               r.y_dram_read_cycles,
                                               r.y_dram_write_cycles,
                                               r.cim_cycles,
                                               r.z_dram_rw_cycles,
                                               &r.cim_exposed_cycles);
#endif
    populate_arch_ablation_counters(r,
                                    r.y_dram_read_cycles > 0 || r.y_dram_write_cycles > 0,
                                    r.z_dram_rw_cycles > 0,
                                    false,
                                    1);
    return r;
}

DualModeSpMM::StepResult DualModeSpMM::run_once_incremental(
    const WorkStealingScheduler::CSRMatrix& a_new,
    const std::vector<DeltaConverter::DeltaAUpdate>& delta_a,
    const DeltaConverter::DeltaFParsed& delta_f) {

    StepResult r;
    indexer_.reset_counters();

    // Enable DRAMsim3 for index/feature traffic (exact accounting)
#ifdef ENABLE_DRAMSIM3_PE
    if (gcp_ != nullptr) {
        std::string cfg = "./ktb_dram_configs/ktb_HBM2_8Gb_x128.ini";
        cfg = gcp_->get_dram_config_path();
        const std::string base = gcp_->get_results_dir();
        const double tck_ns = gcp_->get_memory_tck_ns();
        if (!indexer_.dramsim3_enabled()) {
            indexer_.enable_dramsim3(cfg,
                                     base + "/dramsim3_indexer_output",
                                     base + "/dramsim3_feature_output",
                                     tck_ns);
        }
    }
    ensure_yz_dramsim3_enabled();
#endif

    // Factorized keeps raw CSR/IndexStore state, but never materializes
    // normalized M or dM.  The host raw-CSR preparation is functional-only;
    // this IndexStore delta is the one charged topology update owner.
    if (indexer_.n() != n_ || a_new.n != n_) {
        throw std::runtime_error(
            "incremental operator index state is not initialized for graph domain");
    }
    indexer_.apply_delta_a(delta_a);
    const auto& maintenance = indexer_.get_counters();
    const uint64_t expected_update_payload_bytes =
        static_cast<uint64_t>(delta_a.size()) * 8ULL;
    if (maintenance.index_update_ops != delta_a.size() ||
        maintenance.index_update_payload_bytes != expected_update_payload_bytes ||
        (!delta_a.empty() && maintenance.index_update_dram_cycles <= 0) ||
        maintenance.index_update_dram_cycles > maintenance.index_dram_cycles) {
        throw std::runtime_error(
            "incremental operator maintenance ownership/counter gate failed");
    }
    row_mean_transition_ = row_mean_operator_.prepare(a_new, delta_a);
    r.row_mean_enabled = true;
    r.row_mean_graph_changed_degree_rows = static_cast<long long>(
        row_mean_transition_.changed_rows.size());
    r.row_mean_degree_read_events = static_cast<long long>(
        row_mean_transition_.degree_read_events);
    r.row_mean_degree_write_events = static_cast<long long>(
        row_mean_transition_.degree_write_events);
    r.row_mean_degree_read_bytes = static_cast<long long>(
        row_mean_transition_.degree_read_bytes);
    r.row_mean_degree_write_bytes = static_cast<long long>(
        row_mean_transition_.degree_write_bytes);
    r.row_mean_reciprocal_events = static_cast<long long>(
        row_mean_transition_.reciprocal_events);

    // Canonicalize before any L1 phase consumes ΔF.  A duplicate row that
    // cancels numerically is not active P3 source support.
    const l1_p3::CanonicalDeltaF l1_delta_f =
        l1_p3::canonicalize_delta_f(delta_f, n_, d_);
    const DeltaConverter::DeltaFParsed& delta_f_payload = l1_delta_f.payload;
    const std::vector<int>& delta_f_support = l1_delta_f.nonzero_support;

    // P1 sees F_new.  P2 and P3 below consume this one immutable payload.
    apply_delta_f(delta_f_payload);

    // Scheduled policy can select FULL here (we keep function name but honor mode_)
    int k = static_cast<int>(delta_a.size());
    int l = static_cast<int>(delta_f_support.size());
    double avg_deg = (n_ > 0) ? (static_cast<double>(a_new.nnz) / static_cast<double>(n_)) : 0.0;
    Mode chosen = mode_;
    if (mode_ == Mode::SCHEDULED) {
        chosen = should_incremental(k, l, a_new.nnz, avg_deg, threshold_) ? Mode::INCREMENTAL : Mode::FULL;
    }
    r.mode_used = chosen;
    r.phase_mode_used = phase_mode_;
    const CiMMode effective_cim_mode = effective_cim_mode_for_arch();
    r.cim_mode = (chosen == Mode::INCREMENTAL)
        ? effective_cim_mode
        : (effective_cim_mode != CiMMode::NONE ? CiMMode::FULL_YNEW : CiMMode::NONE);

    if (chosen == Mode::FULL) {
        PEArraySpMM::Stats st = run_pe_job(a_new, f_current_.data());
        y_current_.assign(pe_->get_output(), pe_->get_output() + static_cast<size_t>(n_) * static_cast<size_t>(d_));
        DynamicRowMeanOperator::apply_full_row_mean(
            &y_current_, row_mean_operator_.degree(), d_);
        r.row_mean_scaled_rows += n_;
        r.row_mean_scale_elements += static_cast<long long>(n_) * d_;
        r.row_mean_scalar_ops += static_cast<long long>(n_) * d_;
        r.row_mean_degree_read_events += n_;
        r.row_mean_degree_read_bytes += static_cast<long long>(n_) * sizeof(std::int32_t);
        r.y_accuracy = compute_y_accuracy_metrics(y_current_, golden_full_spmm(a_new));
        r.pe_stats = st;
        r.indexer_ctrs = indexer_.get_counters();

        // Scheduled -> FULL fallback must still account Phase-E Y-domain DRAM traffic.
#ifdef ENABLE_DRAMSIM3_PE
        if (yz_dram_ != nullptr) {
            r.y_dram_read_cycles = issue_yz_all_rows(kYDramBaseAddr, n_, d_, false);
            r.y_dram_write_cycles = issue_yz_all_rows(kYDramBaseAddr, n_, d_, true);
        } else {
            r.y_dram_read_cycles = estimate_dram_read_cycles(n_, d_);
            r.y_dram_write_cycles = estimate_dram_write_cycles(n_, d_);
        }
#else
        r.y_dram_read_cycles = estimate_dram_read_cycles(n_, d_);
        r.y_dram_write_cycles = estimate_dram_write_cycles(n_, d_);
#endif

        if (r.cim_mode == CiMMode::FULL_YNEW) {
            r.cim_rows = n_;
            if (compute_accuracy_) {
                run_full_cim_validation(a_new, r);
            } else {
                r.cim_cycles = estimate_cim_cycles(cim_, r.cim_rows);
            }
            const long long z_bytes_all =
                static_cast<long long>(n_) * static_cast<long long>(d_out_) * kBytesPerElement;
            if (dedicated_z_sram_can_hold(z_bytes_all)) {
                r.z_dram_rw_cycles = 0;
                r.z_sram_accum_used = true;
                r.account_z_sram_bytes(z_bytes_all, z_bytes_all);
            } else {
#ifdef ENABLE_DRAMSIM3_PE
                if (yz_dram_ != nullptr) {
                    r.z_dram_rw_cycles = issue_yz_all_rows(kZDramBaseAddr, n_, d_out_, true);
                } else {
                    r.z_dram_rw_cycles = estimate_dram_write_cycles(n_, d_out_);
                }
#else
                r.z_dram_rw_cycles = estimate_dram_write_cycles(n_, d_out_);
#endif
            }
        }
        // When scheduled mode falls back to FULL, all rows are affected
        // so that the higher-layer cascade in neuratile.cpp executes properly.
        r.affected_rows.reserve(n_);
        for (int i = 0; i < n_; i++) {
            r.affected_rows.push_back(i);
        }

        if (arch_uses_digital_stream_backend()) {
            r.y_dram_read_cycles = 0;
            r.y_dram_write_cycles = 0;
            const long long producer_cycles =
                static_cast<long long>(r.pe_stats.total_cycles) +
                r.indexer_ctrs.index_dram_cycles +
                r.indexer_ctrs.feature_dram_cycles;
            const long long z_write_cycles = estimate_dram_write_cycles(n_, d_out_);
            apply_digital_stream_backend_accounting(r,
                                                    producer_cycles,
                                                    n_,
                                                    d_,
                                                    d_out_,
                                                    z_write_cycles);
            populate_arch_ablation_counters(r,
                                            false,
                                            r.z_dram_rw_cycles > 0,
                                            false,
                                            1,
                                            -1,
                                            n_);
        } else {
#if DESPOINA_V3_FUNCTIONAL_DCIM
            r.total_cycles = finalize_causal_dcim_step_cycles(
                r.pe_stats, r.indexer_ctrs,
                r.y_dram_read_cycles, r.y_dram_write_cycles,
                r.exposed_tail_cycles, r.z_dram_rw_cycles);
            r.cim_cycles = static_cast<long long>(r.causal_stream_elapsed_cycles.value);
            r.cim_stream_cycles = r.cim_cycles;
            r.cim_exposed_cycles = static_cast<long long>(r.exposed_tail_cycles.value);
#else
            r.total_cycles = finalize_full_step_cycles(r.pe_stats,
                                                       r.indexer_ctrs,
                                                       r.y_dram_read_cycles,
                                                       r.y_dram_write_cycles,
                                                       r.cim_cycles,
                                                       r.z_dram_rw_cycles,
                                                       &r.cim_exposed_cycles);
#endif
            populate_arch_ablation_counters(r,
                                            r.y_dram_read_cycles > 0 || r.y_dram_write_cycles > 0,
                                            r.z_dram_rw_cycles > 0,
                                            false,
                                            1);
        }

        account_factorized_native_owners(r.affected_rows, &r);
        r.total_cycles += r.factorized_native_owner_cycles;

#ifdef ENABLE_DRAMSIM3_PE
        indexer_.flush_dramsim3_stats();
        flush_yz_dramsim3_stats();
#endif
        return r;
    }

    // Phase 1: ΔA × F_new
    auto delta_a_csr = DeltaConverter::delta_a_to_csr(delta_a, n_);
    auto delta_f_dense = DeltaConverter::delta_f_to_dense(delta_f_payload, n_, d_);
    // CSR rows are destinations, columns are source feature rows.  P3 is
    // -ΔA×ΔF, therefore filter ΔA *columns* by exact nonzero ΔF support.
    const WorkStealingScheduler::CSRMatrix delta_a_p3_csr =
        fullscan_extract(delta_a_csr, delta_f_support);
    WorkStealingScheduler::CSRMatrix a_sub;
    const std::string effective_indexer_mode = effective_indexer_mode_for_arch();
    if (effective_indexer_mode == "fullscan") {
        a_sub = fullscan_extract(a_new, delta_f_support);
        indexer_.account_fullscan_dram_cost(a_new.n, a_new.nnz);
    } else {
        a_sub = indexer_.extract_a_sub_for_delta_f(delta_f_support);
    }
    const std::vector<int> affected_rows = collect_csr_affected_rows(delta_a_csr, a_sub);

    const bool phase2_active = (phase_mode_ != PhaseMode::EDGE_ONLY);
    const bool phase3_active = (phase_mode_ == PhaseMode::THREE_PHASE ||
                                phase_mode_ == PhaseMode::ANEW_FULL);
    const WorkStealingScheduler::CSRMatrix& phase2_csr =
        (phase_mode_ == PhaseMode::ANEW_FULL) ? a_new : a_sub;
    const bool phase2_has_payload_work = phase2_active && phase2_csr.nnz > 0;

    const size_t y_size = static_cast<size_t>(n_) * static_cast<size_t>(d_);
    PEArraySpMM::Stats st1{};
    PEArraySpMM::Stats st2{};
    PEArraySpMM::Stats st3{};
    CimHardwareRunResult linear_stream_hw{};

    if (y_current_.empty()) {
        y_current_.assign(y_size, 0.0f);
    }
    std::vector<float> y_prev_rows;
    y_prev_rows.reserve(affected_rows.size() * static_cast<size_t>(d_));
    for (size_t idx = 0; idx < affected_rows.size(); idx++) {
        const int row = affected_rows[idx];
        const float* src = y_current_.data() + static_cast<size_t>(row) * static_cast<size_t>(d_);
        y_prev_rows.insert(y_prev_rows.end(), src, src + d_);
    }

    if (r.cim_mode == CiMMode::PER_PHASE_LINEAR) {
#if DESPOINA_V3_FUNCTIONAL_DCIM
        const std::vector<int> phase1_rows = collect_nonzero_rows(delta_a_csr);
        const std::vector<int> phase2_rows = phase2_active
            ? collect_nonzero_rows(phase2_csr) : std::vector<int>();
        const std::vector<int> phase3_rows = phase3_active
            ? collect_nonzero_rows(delta_a_p3_csr) : std::vector<int>();
        const uint64_t epoch_phase1 = next_cache_epoch_++;
        const uint64_t epoch_delta = next_cache_epoch_++;
        std::vector<LinearStreamPhase> phases(3);
        phases[0].csr = &delta_a_csr;
        phases[0].f_matrix = f_current_.data();
        phases[0].y_scale = 1.0f;
        phases[0].expected_rows = static_cast<int>(phase1_rows.size());
        phases[0].keep_shared_f_cache = false;
        phases[0].matrix_epoch = epoch_phase1;
        phases[1].csr = phase2_active ? &phase2_csr : nullptr;
        phases[1].f_matrix = phase2_active ? delta_f_dense.data() : nullptr;
        phases[1].y_scale = 1.0f;
        phases[1].expected_rows = static_cast<int>(phase2_rows.size());
        // P1's F_new epoch is distinct: P1->P2 invalidates feature state.
        phases[1].keep_shared_f_cache = false;
        phases[1].matrix_epoch = epoch_delta;
        phases[2].csr = phase3_active ? &delta_a_p3_csr : nullptr;
        phases[2].f_matrix = phase3_active ? delta_f_dense.data() : nullptr;
        phases[2].y_scale = -1.0f;
        phases[2].expected_rows = static_cast<int>(phase3_rows.size());
        // P2->P3 alone may preserve the exact immutable ΔF feature payload.
        phases[2].keep_shared_f_cache = phase2_has_payload_work;
        phases[2].matrix_epoch = epoch_delta;
        load_resident_weights_for_layer(1);
        std::vector<PEArraySpMM::Stats> phase_stats;
        linear_stream_hw = run_cim_hardware_streaming_phases(
            phases, affected_rows, &y_current_, nullptr, &phase_stats);
        if (phase_stats.size() > 0) st1 = phase_stats[0];
        if (phase_stats.size() > 1) st2 = phase_stats[1];
        if (phase_stats.size() > 2) st3 = phase_stats[2];
        std::vector<uint64_t> ready_cycles(affected_rows.size(), 0);
        const long long producer_cycles = st1.total_cycles + st2.total_cycles + st3.total_cycles;
        if (!commit_finalized_dcim_rows(1, affected_rows, ready_cycles,
                                        producer_cycles, r, &linear_stream_hw)) {
            throw std::runtime_error("v3 finalized L1 DCIM commit failed closed");
        }
#else
        const std::vector<int> phase1_rows = collect_nonzero_rows(delta_a_csr);
        const std::vector<int> phase2_rows = phase2_active ? collect_nonzero_rows(phase2_csr) : std::vector<int>();
        const std::vector<int> phase3_rows = phase3_active ? collect_nonzero_rows(delta_a_p3_csr) : std::vector<int>();

        std::vector<LinearStreamPhase> phases;
        phases.reserve(3);

        LinearStreamPhase phase1;
        phase1.csr = &delta_a_csr;
        phase1.f_matrix = f_current_.data();
        phase1.y_scale = 1.0f;
        phase1.expected_rows = static_cast<int>(phase1_rows.size());
        phase1.keep_shared_f_cache = false;
        phase1.matrix_epoch = next_cache_epoch_++;
        phases.push_back(phase1);

        LinearStreamPhase phase2;
        phase2.csr = phase2_active ? &phase2_csr : nullptr;
        phase2.f_matrix = phase2_active ? delta_f_dense.data() : nullptr;
        phase2.y_scale = 1.0f;
        phase2.expected_rows = static_cast<int>(phase2_rows.size());
        phase2.keep_shared_f_cache = false;
        phase2.matrix_epoch = next_cache_epoch_++;
        phases.push_back(phase2);

        LinearStreamPhase phase3;
        phase3.csr = phase3_active ? &delta_a_p3_csr : nullptr;
        phase3.f_matrix = phase3_active ? delta_f_dense.data() : nullptr;
        phase3.y_scale = -1.0f;
        phase3.expected_rows = static_cast<int>(phase3_rows.size());
        phase3.keep_shared_f_cache = phase2_has_payload_work;
        phase3.matrix_epoch = phase2.matrix_epoch;
        phases.push_back(phase3);

        std::vector<PEArraySpMM::Stats> phase_stats;
        linear_stream_hw = run_cim_hardware_streaming_phases(
            phases, affected_rows, &y_current_, nullptr, &phase_stats);
        if (phase_stats.size() > 0) st1 = phase_stats[0];
        if (phase_stats.size() > 1) st2 = phase_stats[1];
        if (phase_stats.size() > 2) st3 = phase_stats[2];
#endif
    } else {
        auto zero_output = [&]() {
            return std::vector<float>(y_size, 0.0f);
        };

        std::vector<float> out1 = zero_output();
        std::vector<float> out2 = zero_output();
        std::vector<float> out3 = zero_output();

        const uint64_t epoch_phase1 = next_cache_epoch_++;
        const uint64_t epoch_delta = next_cache_epoch_++;
        st1 = run_pe_job(delta_a_csr, f_current_.data(),
                         /*keep_shared_f_cache=*/false, epoch_phase1);
        out1.assign(pe_->get_output(), pe_->get_output() + y_size);

        if (phase2_active) {
            st2 = run_pe_job(phase2_csr, delta_f_dense.data(),
                             /*keep_shared_f_cache=*/false, epoch_delta);
            out2.assign(pe_->get_output(), pe_->get_output() + y_size);
        }

        if (phase3_active) {
            st3 = run_pe_job(delta_a_p3_csr, delta_f_dense.data(),
                             /*keep_shared_f_cache=*/phase2_has_payload_work, epoch_delta);
            out3.assign(pe_->get_output(), pe_->get_output() + y_size);
        }

        // Combine: Y_new = Y_prev + out1 + out2 - out3
        if (y_current_.empty()) {
            y_current_.assign(y_size, 0.0f);
        }
        for (size_t i = 0; i < y_current_.size(); i++) {
            float next = y_current_[i] + out1[i];
            if (phase2_active) {
                next += out2[i];
            }
            if (phase3_active) {
                next -= out3[i];
            }
            y_current_[i] = next;
        }
    }

    if (r.cim_mode == CiMMode::PER_PHASE_LINEAR) {
            account_row_mean_stream_result(linear_stream_hw, &r);
    } else {
            for (std::size_t ordinal = 0; ordinal < affected_rows.size(); ++ordinal) {
                const int row = affected_rows[ordinal];
                float* state = y_current_.data() +
                    static_cast<std::size_t>(row) * static_cast<std::size_t>(d_);
                const float* old_row = y_prev_rows.data() +
                    ordinal * static_cast<std::size_t>(d_);
                const int d_old = row_mean_operator_.old_degree(row, row_mean_transition_);
                const int d_new = row_mean_operator_.new_degree(row);
                const bool changed = d_old != d_new;
                DynamicRowMeanOperator::apply_scoped_recurrence_row(
                    state, old_row, d_old, d_new,
                    row_mean_operator_.old_over_new(row, row_mean_transition_),
                    row_mean_operator_.new_reciprocal(row), d_);
                ++r.row_mean_scaled_rows;
                r.row_mean_changed_degree_rows += changed ? 1 : 0;
                r.row_mean_scale_elements += d_;
                r.row_mean_residual_correction_elements += changed ? d_ : 0;
                const int row_mean_ops = d_new == 0 ? 1 : (changed ? 4 : 3);
                r.row_mean_scalar_ops += static_cast<long long>(row_mean_ops) * d_;
                r.row_mean_degree_read_events += changed ? 2 : 1;
                r.row_mean_degree_read_bytes +=
                    static_cast<long long>(changed ? 2 : 1) * sizeof(std::int32_t);
                r.row_mean_reciprocal_events += d_new > 0 ? 1 : 0;
            }
    }

    if (affected_rows.empty()) {
        r.y_accuracy.valid = true;
        r.y_accuracy.exact = true;
        r.y_accuracy.rmse = 0.0;
        r.y_accuracy.cosine_similarity = 1.0;
    } else {
        std::vector<float> y_delta_eval_rows;
        y_delta_eval_rows.reserve(affected_rows.size() * static_cast<size_t>(d_));
        for (size_t idx = 0; idx < affected_rows.size(); idx++) {
            const int row = affected_rows[idx];
            const float* src = y_current_.data() + static_cast<size_t>(row) * static_cast<size_t>(d_);
            const float* prev = y_prev_rows.data() + idx * static_cast<size_t>(d_);
            for (int j = 0; j < d_; j++) {
                y_delta_eval_rows.push_back(src[j] - prev[j]);
            }
        }
        std::vector<float> y_delta_golden = golden_full_spmm_rows(a_new, affected_rows);
        for (size_t i = 0; i < y_delta_golden.size(); i++) {
            y_delta_golden[i] -= y_prev_rows[i];
        }
        r.y_accuracy = compute_y_accuracy_metrics(y_delta_eval_rows, y_delta_golden);
    }

    // Charge explicit Y-domain combine cycles for NONE/DELTA/FULL modes.
    // PER_PHASE_LINEAR uses Z-domain combine after phase-wise CiM accumulation.
    if (r.cim_mode != CiMMode::PER_PHASE_LINEAR) {
        const long long combine_ops = static_cast<long long>(n_) * static_cast<long long>(d_);
        r.combine_cycles = simulate_combine_cycles(combine_ops);
    }

    // Aggregate stats (simple sum; will be refined in stats_energy_integration)
    PEArraySpMM::Stats total = st1;
    total.total_cycles = st1.total_cycles + st2.total_cycles + st3.total_cycles;
    total.compute_cycles = st1.compute_cycles + st2.compute_cycles + st3.compute_cycles;
    total.stall_cycles = st1.stall_cycles + st2.stall_cycles + st3.stall_cycles;
    total.memory_stall_cycles = st1.memory_stall_cycles + st2.memory_stall_cycles + st3.memory_stall_cycles;
    total.total_mac_ops = st1.total_mac_ops + st2.total_mac_ops + st3.total_mac_ops;
    total.f_cache_hits = st1.f_cache_hits + st2.f_cache_hits + st3.f_cache_hits;
    total.f_cache_misses = st1.f_cache_misses + st2.f_cache_misses + st3.f_cache_misses;
    total.l1_f_cache_hits = st1.l1_f_cache_hits + st2.l1_f_cache_hits + st3.l1_f_cache_hits;
    total.l2_shared_hits = st1.l2_shared_hits + st2.l2_shared_hits + st3.l2_shared_hits;
    total.l2_shared_hits_pre_submit = st1.l2_shared_hits_pre_submit + st2.l2_shared_hits_pre_submit + st3.l2_shared_hits_pre_submit;
    total.l1_bank_conflict_stalls = st1.l1_bank_conflict_stalls + st2.l1_bank_conflict_stalls + st3.l1_bank_conflict_stalls;
    total.l2_bank_conflict_stalls = st1.l2_bank_conflict_stalls + st2.l2_bank_conflict_stalls + st3.l2_bank_conflict_stalls;
    total.dram_fills = st1.dram_fills + st2.dram_fills + st3.dram_fills;
    total.dram_requests = st1.dram_requests + st2.dram_requests + st3.dram_requests;
    total.a_load_dram_requests = st1.a_load_dram_requests + st2.a_load_dram_requests + st3.a_load_dram_requests;
    total.a_load_dram_latency_sum = st1.a_load_dram_latency_sum + st2.a_load_dram_latency_sum + st3.a_load_dram_latency_sum;
    total.dram_row_buffer_hits = st1.dram_row_buffer_hits + st2.dram_row_buffer_hits + st3.dram_row_buffer_hits;
    total.shared_f_cache_hits = st1.shared_f_cache_hits + st2.shared_f_cache_hits + st3.shared_f_cache_hits;
    total.shared_f_cache_misses = st1.shared_f_cache_misses + st2.shared_f_cache_misses + st3.shared_f_cache_misses;
    total.output_egress_bytes = st1.output_egress_bytes + st2.output_egress_bytes + st3.output_egress_bytes;
    total.output_egress_rows = st1.output_egress_rows + st2.output_egress_rows + st3.output_egress_rows;
    total.output_egress_stall_cycles = st1.output_egress_stall_cycles + st2.output_egress_stall_cycles + st3.output_egress_stall_cycles;
    total.output_cluster_backpressure_stalls =
        st1.output_cluster_backpressure_stalls + st2.output_cluster_backpressure_stalls + st3.output_cluster_backpressure_stalls;
    total.work_steals = st1.work_steals + st2.work_steals + st3.work_steals;
    total.noc_steal_latency_total_cycles =
        st1.noc_steal_latency_total_cycles + st2.noc_steal_latency_total_cycles + st3.noc_steal_latency_total_cycles;
    total.noc_steal_count = st1.noc_steal_count + st2.noc_steal_count + st3.noc_steal_count;
    if (total.noc_steal_count > 0) {
        const float weighted_hops =
            st1.noc_steal_avg_hops * static_cast<float>(st1.noc_steal_count) +
            st2.noc_steal_avg_hops * static_cast<float>(st2.noc_steal_count) +
            st3.noc_steal_avg_hops * static_cast<float>(st3.noc_steal_count);
        total.noc_steal_avg_hops = weighted_hops / static_cast<float>(total.noc_steal_count);
    } else {
        total.noc_steal_avg_hops = 0.0f;
    }
    // utilization/load_imbalance will be recomputed later; keep max for now.
    total.pe_utilization = std::max(st1.pe_utilization, std::max(st2.pe_utilization, st3.pe_utilization));
    total.load_imbalance = std::max(st1.load_imbalance, std::max(st2.load_imbalance, st3.load_imbalance));

    r.pe_stats = total;
    r.phase1.nnz = delta_a_csr.nnz;
    r.phase1.mac_ops = st1.total_mac_ops;
    r.phase1.cycles = st1.total_cycles;
    r.phase2.nnz = phase2_active ? phase2_csr.nnz : 0;
    r.phase2.mac_ops = st2.total_mac_ops;
    r.phase2.cycles = st2.total_cycles;
    r.phase3.nnz = phase3_active ? delta_a_p3_csr.nnz : 0;
    r.phase3.mac_ops = st3.total_mac_ops;
    r.phase3.cycles = st3.total_cycles;
    r.indexer_ctrs = indexer_.get_counters();

    // Prepare Z buffers for subsequent incremental timesteps.
    if (n_ > 0 && d_out_ > 0) {
        const size_t z_size = static_cast<size_t>(n_) * static_cast<size_t>(d_out_);
        if (z_current_.size() != z_size) {
            z_current_.assign(z_size, 0.0f);
        }
        if (z_prev_.size() != z_size) {
            z_prev_.assign(z_size, 0.0f);
        }
        z_prev_ = z_current_;
    }

    auto calc_y_read_all_cycles = [&]() -> long long {
#ifdef ENABLE_DRAMSIM3_PE
        if (yz_dram_ != nullptr) {
            return issue_yz_all_rows(kYDramBaseAddr, n_, d_, false);
        }
#endif
        return estimate_dram_read_cycles(n_, d_);
    };

    auto calc_y_write_all_cycles = [&]() -> long long {
#ifdef ENABLE_DRAMSIM3_PE
        if (yz_dram_ != nullptr) {
            return issue_yz_all_rows(kYDramBaseAddr, n_, d_, true);
        }
#endif
        return estimate_dram_write_cycles(n_, d_);
    };

    auto calc_z_rw_cycles = [&](const std::vector<int>& rows) -> long long {
        const long long z_bytes =
            static_cast<long long>(rows.size()) * static_cast<long long>(d_out_) * kBytesPerElement;

        // Dedicated Z-SRAM: if it fits, Z never leaves SRAM.
        if (dedicated_z_sram_can_hold(z_bytes)) {
            r.z_sram_accum_used = true;
            const int num_phases =
#if DESPOINA_V3_FUNCTIONAL_DCIM
                1;
#else
                (r.cim_mode == CiMMode::PER_PHASE_LINEAR) ? 3 : 1;
#endif
            r.account_z_sram_bytes(z_bytes * num_phases, z_bytes * num_phases);
            return 0;
        }

        // F-Cache SRAM accumulation: during CiM combination phase, F-Cache is
        // idle (aggregation done). Z partial sums are stored in F-Cache SRAM
        // instead of DRAM. Only the final result is written back to DRAM once.
        const long long f_cache_capacity_bytes =
            static_cast<long long>(PE_SHARED_F_CACHE_ENTRIES) *
            static_cast<long long>(d_) * kBytesPerElement;
if (z_sram_accum_ && z_bytes <= f_cache_capacity_bytes) {
            r.z_sram_accum_used = true;
            const int num_phases =
#if DESPOINA_V3_FUNCTIONAL_DCIM
                1;
#else
                (r.cim_mode == CiMMode::PER_PHASE_LINEAR) ? 3 : 1;
#endif
            r.account_z_sram_bytes(z_bytes * num_phases, z_bytes * num_phases);

#ifdef ENABLE_DRAMSIM3_PE
            if (yz_dram_ != nullptr) {
                return issue_yz_selected_rows(kZDramBaseAddr, rows, d_out_, true);
            }
#endif
            return estimate_dram_write_cycles(static_cast<int>(rows.size()), d_out_);
        }

        // Original: full DRAM R+W path
#ifdef ENABLE_DRAMSIM3_PE
        if (yz_dram_ != nullptr) {
            const long long read_cycles = issue_yz_selected_rows(kZDramBaseAddr, rows, d_out_, false);
            const long long write_cycles = issue_yz_selected_rows(kZDramBaseAddr, rows, d_out_, true);
            return read_cycles + write_cycles;
        }
#endif
        return estimate_dram_rw_cycles(static_cast<int>(rows.size()), d_out_);
    };

    auto calc_z_write_all_cycles = [&]() -> long long {
        // Dedicated Z-SRAM: if it fits, Z stays in SRAM.
        if (dedicated_z_sram_can_hold(
                static_cast<long long>(n_) * static_cast<long long>(d_out_) * kBytesPerElement)) {
            const long long z_bytes_all =
                static_cast<long long>(n_) * static_cast<long long>(d_out_) * kBytesPerElement;
            r.z_sram_accum_used = true;
            r.account_z_sram_bytes(z_bytes_all, z_bytes_all);
            return 0;
        }
#ifdef ENABLE_DRAMSIM3_PE
        if (yz_dram_ != nullptr) {
            return issue_yz_all_rows(kZDramBaseAddr, n_, d_out_, true);
        }
#endif
        return estimate_dram_write_cycles(n_, d_out_);
    };

    int y_materialized_rows_for_arch = -1;
    int materialization_eliminated_rows_for_arch = 0;

    // Phase B: CiM mode-specific incremental combination modeling.
    switch (r.cim_mode) {
        case CiMMode::NONE:
            if (!arch_uses_digital_stream_backend()) {
                r.y_dram_read_cycles = calc_y_read_all_cycles();
                r.y_dram_write_cycles = calc_y_write_all_cycles();
            }
            break;
        case CiMMode::DELTA_ONLY: {
            WorkStealingScheduler::CSRMatrix phase2_rows_csr;
            if (phase2_active) {
                phase2_rows_csr = a_sub;
            } else {
                phase2_rows_csr.n = n_;
                phase2_rows_csr.nnz = 0;
                phase2_rows_csr.row_ptr.assign(static_cast<size_t>(n_) + 1, 0);
            }
            const std::vector<int> nz_rows = collect_csr_affected_rows(delta_a_csr, phase2_rows_csr);
            r.cim_rows = static_cast<int>(nz_rows.size());
            if (compute_accuracy_) {
                CimHardwareRunResult hw = run_cim_hardware_rows(y_current_, nz_rows);
                std::vector<float> cim_z = hw.output;
                const size_t expected_elements =
                    nz_rows.size() * static_cast<size_t>(d_out_);
                const bool shape_ok = (cim_z.size() == expected_elements);
                if (cim_noise_ && (!cim_ || cim_->get_compute_mode() == CiMComputeMode::IDEAL_FP32)) {
                    apply_cim_noise(cim_z, static_cast<int>(nz_rows.size()), d_out_);
                }
                std::vector<float> golden_z = golden_full_forward_rows(a_new, nz_rows);
                if (hw.ok &&
                    shape_ok &&
                    z_current_.size() >= static_cast<size_t>(n_) * static_cast<size_t>(d_out_)) {
                    for (size_t idx = 0; idx < nz_rows.size(); idx++) {
                        const int row = nz_rows[idx];
                        std::copy(cim_z.data() + idx * static_cast<size_t>(d_out_),
                                  cim_z.data() + (idx + 1) * static_cast<size_t>(d_out_),
                                  z_current_.data() + static_cast<size_t>(row) * static_cast<size_t>(d_out_));
                    }
                }
                r.accuracy = compute_accuracy_metrics(cim_z, golden_z, hw);
                // Keep performance accounting invariant to accuracy toggle.
                r.cim_cycles = estimate_cim_cycles(cim_, r.cim_rows);
            } else {
                r.cim_cycles = estimate_cim_cycles(cim_, r.cim_rows);
            }
            if (arch_variant_ == ArchVariant::H4_SPARSEFE_INDEX_DCIM) {
                y_materialized_rows_for_arch = static_cast<int>(nz_rows.size());
                r.y_dram_read_cycles += estimate_dram_read_cycles(y_materialized_rows_for_arch, d_);
                r.y_dram_write_cycles += estimate_dram_write_cycles(y_materialized_rows_for_arch, d_);
                r.cim_exposed_cycles = r.cim_cycles;
            }
            r.z_dram_rw_cycles = calc_z_rw_cycles(nz_rows);

            // RDS: scan row_ptr of delta_a_csr + a_sub (serial, pre-CiM)
            r.row_detect_cycles = estimate_row_detect_cycles(n_) * 2;
            r.row_detect_checks = 2 * n_;
            r.row_detect_sram_reads = estimate_row_detect_sram_reads(n_) * 2;
            break;
        }
        case CiMMode::FULL_YNEW: {
            r.y_dram_read_cycles = calc_y_read_all_cycles();
            r.y_dram_write_cycles = calc_y_write_all_cycles();
            r.cim_rows = n_;
            if (compute_accuracy_) {
                std::vector<int> all_rows;
                all_rows.reserve(static_cast<size_t>(n_));
                for (int row = 0; row < n_; row++) {
                    all_rows.push_back(row);
                }
                CimHardwareRunResult hw = run_cim_hardware_rows(y_current_, all_rows);
                std::vector<float> cim_z = hw.output;
                const size_t expected_elements =
                    all_rows.size() * static_cast<size_t>(d_out_);
                const bool shape_ok = (cim_z.size() == expected_elements);
                if (cim_noise_ && (!cim_ || cim_->get_compute_mode() == CiMComputeMode::IDEAL_FP32)) {
                    apply_cim_noise(cim_z, static_cast<int>(all_rows.size()), d_out_);
                }
                std::vector<float> golden_z = golden_full_forward_rows(a_new, all_rows);
                if (hw.ok &&
                    shape_ok &&
                    z_current_.size() >= static_cast<size_t>(n_) * static_cast<size_t>(d_out_)) {
                    for (size_t idx = 0; idx < all_rows.size(); idx++) {
                        const int row = all_rows[idx];
                        std::copy(cim_z.data() + idx * static_cast<size_t>(d_out_),
                                  cim_z.data() + (idx + 1) * static_cast<size_t>(d_out_),
                                  z_current_.data() + static_cast<size_t>(row) * static_cast<size_t>(d_out_));
                    }
                }
                r.accuracy = compute_accuracy_metrics(cim_z, golden_z, hw);
                // Keep performance accounting invariant to accuracy toggle.
                r.cim_cycles = estimate_cim_cycles(cim_, r.cim_rows);
            } else {
                r.cim_cycles = estimate_cim_cycles(cim_, r.cim_rows);
            }
            r.z_dram_rw_cycles = calc_z_write_all_cycles();
            break;
        }
        case CiMMode::PER_PHASE_LINEAR: {
#if DESPOINA_V3_FUNCTIONAL_DCIM
            const std::vector<int>& unique_rows = affected_rows;
            materialization_eliminated_rows_for_arch = static_cast<int>(unique_rows.size());
            // Z read-modify-write traffic is owned by commit_finalized_dcim_rows.
            r.row_detect_cycles = estimate_row_detect_cycles(n_);
            r.row_detect_checks = n_;
            r.row_detect_sram_reads = estimate_row_detect_sram_reads(n_);
            if (compute_accuracy_) {
                std::vector<float> eval;
                eval.reserve(unique_rows.size() * static_cast<size_t>(d_out_));
                for (int row : unique_rows) {
                    const float* src = z_current_.data() +
                        static_cast<size_t>(row) * static_cast<size_t>(d_out_);
                    eval.insert(eval.end(), src, src + d_out_);
                }
                std::vector<float> golden_z = golden_full_forward_rows(a_new, unique_rows);
                CimHardwareRunResult ok_hw;
                ok_hw.ok = true;
                ok_hw.expected_rows = static_cast<int>(unique_rows.size());
                ok_hw.received_unique_rows = static_cast<int>(unique_rows.size());
                r.accuracy = compute_accuracy_metrics(eval, golden_z, ok_hw);
            }
#else
            const int rows1 = count_csr_nonzero_rows(delta_a_csr);
            const int rows2 = phase2_active ? count_csr_nonzero_rows(phase2_csr) : 0;
            const int rows3 = phase3_active ? count_csr_nonzero_rows(delta_a_csr) : 0;
            r.cim_rows = rows1 + rows2 + rows3;

            const std::vector<int>& unique_rows = affected_rows;
            const int unique_changed = static_cast<int>(unique_rows.size());
            materialization_eliminated_rows_for_arch = unique_changed;
            r.z_dram_rw_cycles = calc_z_rw_cycles(unique_rows);

            const long long z_combine_ops =
                static_cast<long long>(unique_changed) * static_cast<long long>(d_out_) *
                static_cast<long long>(1 + (phase2_active ? 1 : 0) + (phase3_active ? 1 : 0));
            r.combine_cycles = simulate_combine_cycles(z_combine_ops);

            if (compute_accuracy_) {
                std::vector<float> golden_z =
                    golden_full_forward_rows(a_new, unique_rows);
                r.accuracy = compute_accuracy_metrics(linear_stream_hw.output, golden_z, linear_stream_hw);
            }
            const long long cim1 = estimate_cim_cycles(cim_, rows1);
            const long long cim2 = estimate_cim_cycles(cim_, rows2);
            const long long cim3 = estimate_cim_cycles(cim_, rows3);
            r.cim_cycles = cim1 + cim2 + cim3;
            r.cim_stream_cycles = linear_stream_hw.cycles;
            r.cim_pe_backpressure_cycles = linear_stream_hw.pe_to_cim_backpressure_cycles;
            r.cim_input_fifo_full_rejections =
                static_cast<long long>(linear_stream_hw.cim_pipeline.input_fifo_full_rejections);
            r.cim_input_fifo_full_cycles =
                static_cast<long long>(linear_stream_hw.cim_pipeline.input_fifo_full_cycles);
            r.cim_output_fifo_full_cycles =
                static_cast<long long>(linear_stream_hw.cim_pipeline.output_fifo_full_cycles);
            r.cim_output_fifo_full_stall_cycles =
                static_cast<long long>(linear_stream_hw.cim_pipeline.output_fifo_full_stall_cycles);
            r.cim_stage3_output_block_cycles =
                static_cast<long long>(linear_stream_hw.cim_pipeline.stage3_output_block_cycles);
            r.cim_inject_no_empty_lane_cycles =
                static_cast<long long>(linear_stream_hw.cim_pipeline.inject_no_empty_lane_cycles);
            r.cim_active_lane_cycles =
                static_cast<long long>(linear_stream_hw.cim_pipeline.active_lane_cycles);
            r.cim_active_slot_cycles =
                static_cast<long long>(linear_stream_hw.cim_pipeline.active_slot_cycles);
            r.cim_input_fifo_occupancy_cycles =
                static_cast<long long>(linear_stream_hw.cim_pipeline.input_fifo_occupancy_cycles);
            r.cim_output_fifo_occupancy_cycles =
                static_cast<long long>(linear_stream_hw.cim_pipeline.output_fifo_occupancy_cycles);
            r.cim_max_active_lanes =
                static_cast<int>(linear_stream_hw.cim_pipeline.max_active_lanes);

            // ================================================================
            // Actual PE->CiM streaming is preserved for the data path, but
            // performance accounting remains analytical so reported latency
            // follows the original overlap model.
            // ================================================================
            const long long pe_total_3phase =
                static_cast<long long>(st1.total_cycles)
                + static_cast<long long>(st2.total_cycles)
                + static_cast<long long>(st3.total_cycles);
            r.cim_exposed_cycles = std::max(0LL, r.cim_cycles - pe_total_3phase);

            // Row Detection Scanner (RDS):
            // For incremental timesteps the affected-row set is already known
            // from delta_a / delta_f inputs — no post-hoc CSR row_ptr scan
            // needed.  We keep the counters for energy accounting but remove
            // the serial cycle cost from the critical path.
            const long long rd_per_phase = estimate_row_detect_cycles(n_);
            const int active_phase_count = 1 + (phase2_active ? 1 : 0) + (phase3_active ? 1 : 0);
            r.row_detect_cycles = active_phase_count * rd_per_phase;          // energy only
            r.row_detect_checks = active_phase_count * n_;
            r.row_detect_sram_reads = estimate_row_detect_sram_reads(n_) * active_phase_count;
#endif
            break;
        }
    }

    const long long cim_added = (r.cim_mode == CiMMode::PER_PHASE_LINEAR)
        ? r.cim_exposed_cycles
        : r.cim_cycles;

    account_factorized_native_owners(affected_rows, &r);

    // Row detection serial cost:
    //   PER_PHASE_LINEAR: affected rows are known from delta inputs at
    //   incremental time — no serial CSR scan on the critical path.
    //   DELTA_ONLY / FULL: all scans serial (kept for those modes).
    const long long row_detect_serial = (r.cim_mode == CiMMode::PER_PHASE_LINEAR)
        ? 0LL                              // known from delta inputs
        : r.row_detect_cycles;             // all scans serial

    if (arch_uses_digital_stream_backend()) {
#if DESPOINA_V3_FUNCTIONAL_DCIM
        r.total_cycles = finalize_causal_dcim_step_cycles(
            total, r.indexer_ctrs, 0, 0,
            r.exposed_tail_cycles, r.z_dram_rw_cycles);
        r.total_cycles += r.factorized_native_owner_cycles;
        r.affected_rows = affected_rows;
#ifdef ENABLE_DRAMSIM3_PE
        indexer_.flush_dramsim3_stats();
        flush_yz_dramsim3_stats();
#endif
        return r;
#else
        const int active_phase_count = 1 + (phase2_active ? 1 : 0) + (phase3_active ? 1 : 0);
        if (r.row_detect_cycles == 0 && active_phase_count > 0) {
            r.row_detect_cycles = active_phase_count * estimate_row_detect_cycles(n_);
            r.row_detect_checks = active_phase_count * n_;
            r.row_detect_sram_reads = estimate_row_detect_sram_reads(n_) * active_phase_count;
        }
        const int stream_rows = static_cast<int>(affected_rows.size());
        const long long producer_cycles = total.total_cycles
            + r.indexer_ctrs.index_dram_cycles
            + r.indexer_ctrs.feature_dram_cycles
            + r.row_detect_cycles;
        const long long z_write_cycles = estimate_dram_write_cycles(stream_rows, d_out_);
        apply_digital_stream_backend_accounting(r,
                                                producer_cycles,
                                                stream_rows,
                                                d_,
                                                d_out_,
                                                z_write_cycles);
        r.total_cycles += r.factorized_native_owner_cycles;
        r.affected_rows = affected_rows;
        populate_arch_ablation_counters(r,
                                        false,
                                        r.z_dram_rw_cycles > 0,
                                        false,
                                        active_phase_count,
                                        -1,
                                        stream_rows);

#ifdef ENABLE_DRAMSIM3_PE
        indexer_.flush_dramsim3_stats();
        flush_yz_dramsim3_stats();
#endif
        return r;
#endif
    }

    // ====================================================================
    // Final cycle accounting with compute-memory overlap:
    //
    //  compute_path = PE + index + feature + combine + CiM_exposed + RDS
    //  z_dram_path  = Z write-back via DMA (runs in background)
    //
    //  Inter-timestep pipelining: step k's Z write-back overlaps with
    //  step k+1's compute.  Steady-state per-step cost is:
    //      max(compute_path, z_dram_previous)
    //  We approximate this within a single step as:
    //      compute_path + max(0, z_dram − compute_path)
    //    = max(compute_path, z_dram)
    // ====================================================================
    const long long compute_path = total.total_cycles
        + r.indexer_ctrs.index_dram_cycles
        + r.indexer_ctrs.feature_dram_cycles
        + r.combine_cycles
        + cim_added
        + row_detect_serial
        + r.y_dram_read_cycles
        + r.y_dram_write_cycles
        + r.factorized_native_owner_cycles;

    r.total_cycles = std::max(compute_path, r.z_dram_rw_cycles);
    r.affected_rows = affected_rows;
    populate_arch_ablation_counters(r,
                                    r.y_dram_read_cycles > 0 || r.y_dram_write_cycles > 0,
                                    r.z_dram_rw_cycles > 0,
                                    false,
                                    1 + (phase2_active ? 1 : 0) + (phase3_active ? 1 : 0),
                                    y_materialized_rows_for_arch,
                                    materialization_eliminated_rows_for_arch);

#ifdef ENABLE_DRAMSIM3_PE
    indexer_.flush_dramsim3_stats();
    flush_yz_dramsim3_stats();
#endif
    return r;
}

// =============================================================================
// Layer-2 SpMM for incremental cycle counting
// =============================================================================

// =============================================================================
// Generic per-layer Y state accessors
// =============================================================================
static const std::vector<float> empty_y_vec_;

const std::vector<float>& DualModeSpMM::get_y_layer(int layer_id) const {
    auto it = y_layer_state_.find(layer_id);
    if (it != y_layer_state_.end()) return it->second;
    return empty_y_vec_;
}

void DualModeSpMM::set_y_layer(int layer_id, const std::vector<float>& y) {
    y_layer_state_[layer_id] = y;
}

const std::vector<float>& DualModeSpMM::get_h_layer(int layer_id) const {
    const auto it = h_layer_state_.find(layer_id);
    return it == h_layer_state_.end() ? empty_y_vec_ : it->second;
}

bool DualModeSpMM::set_h_layer(int layer_id, const std::vector<float>& h) {
    const size_t expected = static_cast<size_t>(n_) * static_cast<size_t>(d_out_);
    if (layer_id < 1 || h.size() != expected) return false;
    h_layer_state_[layer_id] = h;
    return true;
}

bool DualModeSpMM::build_norm_relu(const std::vector<float>& z,
                                   int n,
                                   bool apply_l2_norm,
                                   std::vector<float>* h) {
    if (h == nullptr || n <= 0 || z.empty() || z.size() % static_cast<size_t>(n) != 0) {
        return false;
    }
    const int d = static_cast<int>(z.size() / static_cast<size_t>(n));
    if (d <= 0) return false;
    h->resize(z.size());
    for (size_t i = 0; i < z.size(); ++i) {
        (*h)[i] = std::max(0.0f, z[i]);
    }
    if (apply_l2_norm) {
        for (int row = 0; row < n; ++row) {
            const size_t base = static_cast<size_t>(row) * static_cast<size_t>(d);
            float norm_sq = 0.0f;
            for (int k = 0; k < d; ++k) {
                const float v = (*h)[base + static_cast<size_t>(k)];
                norm_sq += v * v;
            }
            const float inv_norm = 1.0f / (std::sqrt(norm_sq) + 1e-12f);
            for (int k = 0; k < d; ++k) {
                (*h)[base + static_cast<size_t>(k)] *= inv_norm;
            }
        }
    }
    return true;
}

const std::vector<float>& DualModeSpMM::get_z_layer(int layer_id) const {
    auto it = z_layer_state_.find(layer_id);
    if (it != z_layer_state_.end()) return it->second;
    return empty_y_vec_;
}

bool DualModeSpMM::initialize_higher_layer_state(
    int layer_id,
    const WorkStealingScheduler::CSRMatrix& a_new,
    const std::vector<float>& h_prev,
    StepResult* t0_direct_accounting) {
    const size_t state_size = static_cast<size_t>(n_) * static_cast<size_t>(d_);
    if (layer_id < 2 || a_new.n != n_ || h_prev.size() != state_size) {
        return false;
    }
#if DESPOINA_V3_FUNCTIONAL_DCIM
    try {
        despoina_v3::assert_publication_cycle_shape(
            static_cast<uint64_t>(d_), static_cast<uint64_t>(d_out_));
    } catch (const std::exception&) {
        return false;
    }
    if (cim_ == nullptr ||
        cim_->get_compute_mode() != CiMComputeMode::DCIM_DIGITAL ||
        cim_->get_fp_activations()) {
        return false;
    }
#endif
    std::vector<float> y(state_size, 0.0f);
    for (int row = 0; row < n_; ++row) {
        for (int p = a_new.row_ptr[row]; p < a_new.row_ptr[row + 1]; ++p) {
            const int col = a_new.col_idx[p];
            if (col < 0 || col >= n_) return false;
            const float av = a_new.values.empty() ? 1.0f : a_new.values[p];
            for (int k = 0; k < d_; ++k) {
                y[static_cast<size_t>(row) * d_ + k] +=
                    av * h_prev[static_cast<size_t>(col) * d_ + k];
            }
        }
    }
    DynamicRowMeanOperator::apply_full_row_mean(
        &y, row_mean_operator_.degree(), d_);
    if (t0_direct_accounting != nullptr) {
            t0_direct_accounting->row_mean_enabled = true;
            t0_direct_accounting->row_mean_scaled_rows += n_;
            t0_direct_accounting->row_mean_scale_elements +=
                static_cast<long long>(n_) * d_;
            t0_direct_accounting->row_mean_scalar_ops +=
                static_cast<long long>(n_) * d_;
            t0_direct_accounting->row_mean_degree_read_events += n_;
            t0_direct_accounting->row_mean_degree_read_bytes +=
                static_cast<long long>(n_) * sizeof(std::int32_t);
            t0_direct_accounting->row_mean_reciprocal_events += std::count_if(
                row_mean_operator_.degree().begin(), row_mean_operator_.degree().end(),
                [](int degree) { return degree > 0; });
    }
    y_layer_state_[layer_id] = y;
    std::vector<int> all_rows(static_cast<size_t>(n_));
    for (int row = 0; row < n_; ++row) all_rows[static_cast<size_t>(row)] = row;
    load_resident_weights_for_layer(layer_id);
    CimHardwareRunResult hw = run_cim_hardware_rows(y_layer_state_[layer_id], all_rows);
    if (!hw.ok || hw.output.size() != static_cast<size_t>(n_) * static_cast<size_t>(d_out_)) {
        return false;
    }
    if (!capture_runtime_layer_trace(layer_id, all_rows)) {
        return false;
    }
    if (t0_direct_accounting != nullptr) {
        const std::vector<uint64_t> t0_ready(static_cast<size_t>(n_), 0);
        if (!record_direct_dcim_accounting(
                t0_ready, 0, hw, *t0_direct_accounting)) {
            return false;
        }
    }
    z_layer_state_[layer_id] = hw.output;
    return true;
}

// =============================================================================
// Generic higher-layer incremental SpMM (layers 2, 3, 4, 5, ...)
// =============================================================================

DualModeSpMM::StepResult DualModeSpMM::run_higher_layer_spmm(
    int layer_id,
    const WorkStealingScheduler::CSRMatrix& a_new,
    const std::vector<DeltaConverter::DeltaAUpdate>& delta_a,
    const std::vector<float>& h_prev_new,
    const std::vector<float>& delta_h_prev,
    const std::vector<int>& prev_affected_rows) {

    const std::string tag = "[Layer-" + std::to_string(layer_id) + "]";
    StepResult r;
    r.mode_used = Mode::INCREMENTAL;
    r.phase_mode_used = phase_mode_;

    // P1 is structural work and independent of prior tau-propagated support.
    // An empty previous support must not erase ΔA×H_new.
    const WorkStealingScheduler::CSRMatrix delta_a_csr =
        DeltaConverter::delta_a_to_csr(delta_a, n_);
    const size_t expected_h_elements =
        static_cast<size_t>(n_) * static_cast<size_t>(d_);
    if (delta_a_csr.nnz > 0 && h_prev_new.size() != expected_h_elements) {
        throw std::invalid_argument("higher-layer P1 H_new shape mismatch");
    }
    const bool has_structural_p1_work = delta_a_csr.nnz > 0;

    // Apply tau filtering: only keep rows where ||delta_H[row]||_inf > tau.
    // scope-mode=none disables this row pruning while preserving the same
    // higher-layer execution path and backend accounting.
    std::vector<int> filtered_affected_rows;
    if (propagation_scope_enabled_ && propagation_tau_ > 0 && !delta_h_prev.empty() && d_ > 0) {
        filtered_affected_rows.reserve(prev_affected_rows.size());
        for (int row : prev_affected_rows) {
            if (row < 0 || row >= n_) continue;
            const float* row_data = delta_h_prev.data() + static_cast<size_t>(row) * d_;
            float linf_norm = 0.0f;
            for (int j = 0; j < d_; j++) {
                float abs_val = std::fabs(row_data[j]);
                if (abs_val > linf_norm) linf_norm = abs_val;
            }
            if (linf_norm > propagation_tau_) {
                filtered_affected_rows.push_back(row);
            }
        }
        std::cout << tag << " Tau filtering: tau=" << propagation_tau_
                  << " prev_affected=" << prev_affected_rows.size()
                  << " -> filtered=" << filtered_affected_rows.size()
                  << " (removed " << (prev_affected_rows.size() - filtered_affected_rows.size()) << ")"
                  << std::endl;
    } else {
        filtered_affected_rows = prev_affected_rows;
    }
    const std::vector<int>& active_rows = filtered_affected_rows;

    // Only a true no-op can return early.  Tau-empty propagated support skips
    // P2/P3, but nonempty ΔA still executes P1 and drives the next layer.
    if (!has_structural_p1_work && active_rows.empty()) {
        std::cout << tag << " No structural ΔA or scoped propagated work; early exit."
                  << std::endl;
        return r;
    }

    // P(l-1) defines the effective dH support for the next layer.  Materialize
    // that exact payload once and feed it to both P2 and P3.  Using the dense,
    // pre-threshold dH in P3 breaks the paper's scoped three-phase identity.
    const size_t y_size = static_cast<size_t>(n_) * static_cast<size_t>(d_);
    std::vector<float> scoped_delta_h_prev(y_size, 0.0f);
    for (int row : active_rows) {
        if (row < 0 || row >= n_) return r;
        const size_t base = static_cast<size_t>(row) * static_cast<size_t>(d_);
        std::copy(delta_h_prev.begin() + static_cast<std::ptrdiff_t>(base),
                  delta_h_prev.begin() + static_cast<std::ptrdiff_t>(base + d_),
                  scoped_delta_h_prev.begin() + static_cast<std::ptrdiff_t>(base));
    }

    // --- Phase 1: ΔA × H_{l-1}_new ---
    const WorkStealingScheduler::CSRMatrix delta_a_p3_csr =
        fullscan_extract(delta_a_csr, active_rows);
    std::vector<float> out1(y_size, 0.0f), out2(y_size, 0.0f), out3(y_size, 0.0f);
    PEArraySpMM::Stats st1{};
    PhaseCapture cap1, cap2, cap3;
    const uint64_t epoch_phase1 = next_cache_epoch_++;
    const uint64_t epoch_delta = next_cache_epoch_++;
    if (delta_a_csr.nnz > 0 && !h_prev_new.empty()) {
#if DESPOINA_V3_FUNCTIONAL_DCIM
        // Executed below in the shared PE/DCIM tick loop.
#else
        st1 = run_pe_job(delta_a_csr, h_prev_new.data(),
                         /*keep_shared_f_cache=*/false, epoch_phase1);
#endif
#if !DESPOINA_V3_FUNCTIONAL_DCIM
        const float* pe_out = pe_->get_output();
        if (pe_out) std::copy(pe_out, pe_out + y_size, out1.begin());
#endif
        std::cout << tag << " Phase1 ΔA×H: nnz=" << delta_a_csr.nnz
                  << " cycles=" << st1.total_cycles << std::endl;
    }

    // --- Phase 2: A_sub × ΔH_{l-1} ---
    indexer_.reset_counters();
    WorkStealingScheduler::CSRMatrix a_sub;
    const std::string effective_indexer_mode = effective_indexer_mode_for_arch();
    if (effective_indexer_mode == "fullscan") {
        a_sub = fullscan_extract(a_new, active_rows);
        indexer_.account_fullscan_dram_cost(a_new.n, a_new.nnz);
    } else {
        a_sub = indexer_.extract_a_sub_for_delta_f(active_rows);
    }

    PEArraySpMM::Stats st2{};
    if (a_sub.nnz > 0) {
        std::cout << tag << " Phase2 A_sub×ΔH: active_rows=" << active_rows.size()
                  << " a_sub.nnz=" << a_sub.nnz
                  << "/" << a_new.nnz << " ("
                  << (100.0 * a_sub.nnz / std::max(1, a_new.nnz)) << "%)"
                  << std::endl;
#if DESPOINA_V3_FUNCTIONAL_DCIM
        // Executed below in the shared PE/DCIM tick loop.
#else
        st2 = run_pe_job(a_sub, scoped_delta_h_prev.data(), /*keep_shared_f_cache=*/false,
                         epoch_delta);
#endif
#if !DESPOINA_V3_FUNCTIONAL_DCIM
        const float* pe_out = pe_->get_output();
        if (pe_out) std::copy(pe_out, pe_out + y_size, out2.begin());
#endif
    }

    const std::vector<int> row_mean_phase_rows =
        collect_csr_affected_rows(delta_a_csr, a_sub);

    // --- Phase 3: -ΔA × ΔH_{l-1} ---
    PEArraySpMM::Stats st3{};
    if (delta_a_p3_csr.nnz > 0) {
#if DESPOINA_V3_FUNCTIONAL_DCIM
        // Executed below in the shared PE/DCIM tick loop.
#else
        st3 = run_pe_job(delta_a_p3_csr, scoped_delta_h_prev.data(),
                         /*keep_shared_f_cache=*/a_sub.nnz > 0, epoch_delta);
#endif
#if !DESPOINA_V3_FUNCTIONAL_DCIM
        const float* pe_out = pe_->get_output();
        if (pe_out) std::copy(pe_out, pe_out + y_size, out3.begin());
#endif
        std::cout << tag << " Phase3 -ΔA×scoped(ΔH): nnz=" << delta_a_p3_csr.nnz
                  << " cycles=" << st3.total_cycles << std::endl;
    }

#if DESPOINA_V3_FUNCTIONAL_DCIM
    r.affected_rows = row_mean_phase_rows;
    auto& concurrent_y_l = y_layer_state_[layer_id];
    if (concurrent_y_l.size() != y_size) concurrent_y_l.assign(y_size, 0.0f);
    const std::vector<int> p1_rows = collect_nonzero_rows(delta_a_csr);
    const std::vector<int> p2_rows = collect_nonzero_rows(a_sub);
    const std::vector<int> p3_rows = collect_nonzero_rows(delta_a_p3_csr);
    std::vector<LinearStreamPhase> phases(3);
    phases[0].csr = delta_a_csr.nnz > 0 ? &delta_a_csr : nullptr;
    phases[0].f_matrix = delta_a_csr.nnz > 0 ? h_prev_new.data() : nullptr;
    phases[0].y_scale = 1.0f;
    phases[0].expected_rows = static_cast<int>(p1_rows.size());
    phases[0].keep_shared_f_cache = false;
    phases[0].matrix_epoch = epoch_phase1;
    phases[1].csr = a_sub.nnz > 0 ? &a_sub : nullptr;
    phases[1].f_matrix = a_sub.nnz > 0 ? scoped_delta_h_prev.data() : nullptr;
    phases[1].y_scale = 1.0f;
    phases[1].expected_rows = static_cast<int>(p2_rows.size());
    // P1->P2 changes both payload and epoch: invalidate feature state.
    phases[1].keep_shared_f_cache = false;
    phases[1].matrix_epoch = epoch_delta;
    phases[2].csr = delta_a_p3_csr.nnz > 0 ? &delta_a_p3_csr : nullptr;
    phases[2].f_matrix = delta_a_p3_csr.nnz > 0 ? scoped_delta_h_prev.data() : nullptr;
    phases[2].y_scale = -1.0f;
    phases[2].expected_rows = static_cast<int>(p3_rows.size());
    // P2->P3 may keep only the identical scoped payload cache.
    phases[2].keep_shared_f_cache = a_sub.nnz > 0;
    phases[2].matrix_epoch = epoch_delta;
    load_resident_weights_for_layer(layer_id);
    std::vector<PEArraySpMM::Stats> concurrent_stats;
    CimHardwareRunResult concurrent_hw = run_cim_hardware_streaming_phases(
        phases, r.affected_rows, &concurrent_y_l, nullptr, &concurrent_stats);
    account_row_mean_stream_result(concurrent_hw, &r);
    if (concurrent_stats.size() > 0) st1 = concurrent_stats[0];
    if (concurrent_stats.size() > 1) st2 = concurrent_stats[1];
    if (concurrent_stats.size() > 2) st3 = concurrent_stats[2];
    std::vector<uint64_t> concurrent_ready(r.affected_rows.size(), 0);
    if (!commit_finalized_dcim_rows(layer_id, r.affected_rows, concurrent_ready,
                                    st1.total_cycles + st2.total_cycles + st3.total_cycles,
                                    r, &concurrent_hw)) {
        throw std::runtime_error("Despoina official DCIM concurrent higher-layer commit failed closed");
    }
    if (compute_accuracy_) {
        const std::vector<float>& layer_weights = weights_for_layer(layer_id);
        std::vector<float> golden(
            r.affected_rows.size() * static_cast<size_t>(d_out_), 0.0f);
        for (size_t ordinal = 0; ordinal < r.affected_rows.size(); ++ordinal) {
            const int row = r.affected_rows[ordinal];
            const float* y = concurrent_y_l.data() +
                static_cast<size_t>(row) * static_cast<size_t>(d_);
            float* z = golden.data() +
                ordinal * static_cast<size_t>(d_out_);
            for (int j = 0; j < d_out_; ++j) {
                for (int k = 0; k < d_; ++k) {
                    z[j] += y[k] * layer_weights[
                        static_cast<size_t>(k) * static_cast<size_t>(d_out_) +
                        static_cast<size_t>(j)];
                }
            }
        }
        // DCIM emits Q(dY)W; the architecturally visible result is the
        // accumulated Z row, so the official comparison reads committed Z.
        const std::vector<float>& z_layer = z_layer_state_[layer_id];
        std::vector<float> committed(
            r.affected_rows.size() * static_cast<size_t>(d_out_), 0.0f);
        for (size_t ordinal = 0; ordinal < r.affected_rows.size(); ++ordinal) {
            const float* src = z_layer.data() +
                static_cast<size_t>(r.affected_rows[ordinal]) * static_cast<size_t>(d_out_);
            std::copy(src, src + d_out_,
                      committed.begin() + static_cast<std::ptrdiff_t>(
                          ordinal * static_cast<size_t>(d_out_)));
        }
        r.accuracy = compute_accuracy_metrics(committed, golden, concurrent_hw);
    }
#endif

    r.indexer_ctrs = indexer_.get_counters();

    // --- Combine ---
#if !DESPOINA_V3_FUNCTIONAL_DCIM
    const long long combine_ops = static_cast<long long>(n_) * static_cast<long long>(d_);
    r.combine_cycles = simulate_combine_cycles(combine_ops);
#endif

    // Aggregate PE stats
    PEArraySpMM::Stats total = st1;
    total.total_cycles = st1.total_cycles + st2.total_cycles + st3.total_cycles;
    total.compute_cycles = st1.compute_cycles + st2.compute_cycles + st3.compute_cycles;
    total.stall_cycles = st1.stall_cycles + st2.stall_cycles + st3.stall_cycles;
    total.memory_stall_cycles = st1.memory_stall_cycles + st2.memory_stall_cycles + st3.memory_stall_cycles;
    total.total_mac_ops = st1.total_mac_ops + st2.total_mac_ops + st3.total_mac_ops;
    total.f_cache_hits = st1.f_cache_hits + st2.f_cache_hits + st3.f_cache_hits;
    total.f_cache_misses = st1.f_cache_misses + st2.f_cache_misses + st3.f_cache_misses;
    total.dram_requests = st1.dram_requests + st2.dram_requests + st3.dram_requests;
    r.pe_stats = total;
    r.phase1.nnz = delta_a_csr.nnz;
    r.phase1.mac_ops = st1.total_mac_ops;
    r.phase1.cycles = st1.total_cycles;
    r.phase2.nnz = a_sub.nnz;
    r.phase2.mac_ops = st2.total_mac_ops;
    r.phase2.cycles = st2.total_cycles;
    r.phase3.nnz = delta_a_p3_csr.nnz;
    r.phase3.mac_ops = st3.total_mac_ops;
    r.phase3.cycles = st3.total_cycles;

    r.total_cycles = total.total_cycles + r.combine_cycles;

    // Materialize Y_l for downstream propagation.
#if !DESPOINA_V3_FUNCTIONAL_DCIM
    auto& y_l = y_layer_state_[layer_id];
    if (y_l.size() != y_size) {
        y_l.assign(y_size, 0.0f);
    }
    std::vector<float> row_mean_old_rows;
    row_mean_old_rows.reserve(
        row_mean_phase_rows.size() * static_cast<std::size_t>(d_));
    for (int row : row_mean_phase_rows) {
        const float* src = y_l.data() +
            static_cast<std::size_t>(row) * static_cast<std::size_t>(d_);
        row_mean_old_rows.insert(row_mean_old_rows.end(), src, src + d_);
    }
    for (size_t i = 0; i < y_size; i++) {
        y_l[i] += out1[i] + out2[i] - out3[i];
    }
    r.row_mean_enabled = true;
        for (std::size_t ordinal = 0; ordinal < row_mean_phase_rows.size(); ++ordinal) {
            const int row = row_mean_phase_rows[ordinal];
            float* state = y_l.data() +
                static_cast<std::size_t>(row) * static_cast<std::size_t>(d_);
            const float* old_row = row_mean_old_rows.data() +
                ordinal * static_cast<std::size_t>(d_);
            const int d_old = row_mean_operator_.old_degree(row, row_mean_transition_);
            const int d_new = row_mean_operator_.new_degree(row);
            const bool changed = d_old != d_new;
            DynamicRowMeanOperator::apply_scoped_recurrence_row(
                state, old_row, d_old, d_new,
                row_mean_operator_.old_over_new(row, row_mean_transition_),
                row_mean_operator_.new_reciprocal(row), d_);
            ++r.row_mean_scaled_rows;
            r.row_mean_changed_degree_rows += changed ? 1 : 0;
            r.row_mean_scale_elements += d_;
            r.row_mean_residual_correction_elements += changed ? d_ : 0;
            const int row_mean_ops = d_new == 0 ? 1 : (changed ? 4 : 3);
            r.row_mean_scalar_ops += static_cast<long long>(row_mean_ops) * d_;
            r.row_mean_degree_read_events += changed ? 2 : 1;
            r.row_mean_degree_read_bytes +=
                static_cast<long long>(changed ? 2 : 1) * sizeof(std::int32_t);
            r.row_mean_reciprocal_events += d_new > 0 ? 1 : 0;
    }
#endif

    // Mark affected rows (Phase 1 edge changes + Phase 2 neighbor expansion)
#if DESPOINA_V3_FUNCTIONAL_DCIM
    r.affected_rows = collect_csr_affected_rows(delta_a_csr, a_sub);
#else
    {
        std::unordered_set<int> affected_set;
        for (const auto& u : delta_a) {
            if (u.src >= 0 && u.src < n_) affected_set.insert(u.src);
        }
        std::unordered_set<int> prev_set(prev_affected_rows.begin(), prev_affected_rows.end());
        for (int row = 0; row < n_; row++) {
            if (affected_set.count(row)) continue;
            for (int ptr = a_new.row_ptr[row]; ptr < a_new.row_ptr[row + 1]; ptr++) {
                if (prev_set.count(a_new.col_idx[ptr])) {
                    affected_set.insert(row);
                    break;
                }
            }
        }
        r.affected_rows.assign(affected_set.begin(), affected_set.end());
        std::sort(r.affected_rows.begin(), r.affected_rows.end());
    }
#endif

#if DESPOINA_V3_FUNCTIONAL_DCIM
    r.row_detect_cycles = estimate_row_detect_cycles(n_);
    r.row_detect_checks = n_;
    r.row_detect_sram_reads = estimate_row_detect_sram_reads(n_);
    r.total_cycles = finalize_causal_dcim_step_cycles(
        total, r.indexer_ctrs, 0, 0,
        r.exposed_tail_cycles, r.z_dram_rw_cycles);
#endif

    if (arch_uses_digital_stream_backend()) {
#if DESPOINA_V3_FUNCTIONAL_DCIM
        // The functional commit above already owns the single dense application
        // and its finalized-row timing/energy counters.
#else
        const int active_phase_count = 3;
        r.row_detect_cycles = active_phase_count * estimate_row_detect_cycles(n_);
        r.row_detect_checks = active_phase_count * n_;
        r.row_detect_sram_reads = estimate_row_detect_sram_reads(n_) * active_phase_count;

        const int stream_rows = static_cast<int>(r.affected_rows.size());
        const long long producer_cycles = total.total_cycles
            + r.indexer_ctrs.index_dram_cycles
            + r.indexer_ctrs.feature_dram_cycles
            + r.row_detect_cycles;
        const long long z_write_cycles = estimate_dram_write_cycles(stream_rows, d_out_);
        apply_digital_stream_backend_accounting(r,
                                                producer_cycles,
                                                stream_rows,
                                                d_,
                                                d_out_,
                                                z_write_cycles);
        populate_arch_ablation_counters(r,
                                        false,
                                        r.z_dram_rw_cycles > 0,
                                        true,
                                        active_phase_count,
                                        -1,
                                        stream_rows);
#endif
    } else {
        populate_arch_ablation_counters(r,
                                        true,
                                        false,
                                        true,
                                        3);
    }

    account_factorized_native_owners(r.affected_rows, &r);
    r.total_cycles += r.factorized_native_owner_cycles;

    std::cout << tag << " Total: PE=" << total.total_cycles
              << " (P1=" << st1.total_cycles << " P2=" << st2.total_cycles
              << " P3=" << st3.total_cycles << ") combine=" << r.combine_cycles
              << " total=" << r.total_cycles
              << " affected=" << r.affected_rows.size() << std::endl;

    return r;
}

// =============================================================================
// Multi-Layer GCN (2-Layer End-to-End)
// =============================================================================

void DualModeSpMM::load_weights_l2(const std::string& path, int d_in, int d_out) {
    d_out_l2_ = d_out;
    load_required_fp32_weights(path, d_in, d_out, "L2", &weights_l2_);
}

DualModeSpMM::MultiLayerResult DualModeSpMM::run_two_layer_gcn(
    const WorkStealingScheduler::CSRMatrix& a_new,
    bool do_export_z,
    const std::string& export_dir) {

    MultiLayerResult result;
    result.num_layers = 2;
    if (cim_) {
        // Snapshot multi-timestep runtime energy before 2-layer eval starts.
        result.runtime_cim_energy_pj = cim_->get_total_energy_pj();
    }

    std::vector<int> all_rows;
    all_rows.reserve(static_cast<size_t>(n_));
    for (int row = 0; row < n_; row++) {
        all_rows.push_back(row);
    }

    // =========================================================================
    // Layer 1: Y1 = existing y_current_ (A × F), Z1 = CiM(Y1 × W1)
    // =========================================================================
    LayerResult l1;
    l1.layer_id = 1;

    // Reuse mode-specific incremental Z when available so delta/linear behavior
    // propagates into 2-layer end-to-end accuracy.
    CimHardwareRunResult hw1;
    std::vector<float> z1_hw;
    const size_t expected_z1 = static_cast<size_t>(n_) * static_cast<size_t>(d_out_);
    const bool use_cached_z1 =
        compute_accuracy_ &&
        cim_mode_ != CiMMode::NONE &&
        z_current_.size() == expected_z1;
    if (use_cached_z1) {
        z1_hw = z_current_;
        hw1.ok = true;
        hw1.expected_rows = n_;
        hw1.received_unique_rows = n_;
        hw1.cycles = 0;
        l1.cim_cycles = 0;
        l1.cim_energy_pj = 0.0;
        std::cout << "[Multi-Layer] Layer1 HW source: cached incremental Z\n";
    } else {
        if (cim_) {
            // Isolate L1 eval-stage energy from previously accumulated runtime energy.
            cim_->reset_counters();
            if (!weights_.empty()) load_resident_weights_for_layer(1);
        }
        // CiM with layer 1 weights (already loaded via load_weights + set_cim)
        hw1 = run_cim_hardware_rows(y_current_, all_rows);
        z1_hw = hw1.output;
        l1.cim_cycles = hw1.cycles;
        if (cim_) l1.cim_energy_pj = cim_->get_total_energy_pj();
    }

    // Golden for layer 1
    std::vector<float> z1_golden = golden_dense_matmul_rows(y_current_, all_rows);
    l1.accuracy = compute_accuracy_metrics(z1_hw, z1_golden, hw1);

    // ReLU activation on HW output
    const size_t z1_size = z1_hw.size();
    h1_current_.resize(z1_size);
    for (size_t i = 0; i < z1_size; i++) {
        h1_current_[i] = std::max(0.0f, z1_hw[i]);
    }
    // ReLU is fused into CiM output port (comparator gate) → 0 additional cycles
    l1.activation_cycles = 0;

    result.layers.push_back(l1);

    // =========================================================================
    // Layer 2: Y2 = A × H1, Z2 = CiM(Y2 × W2)
    // =========================================================================
    LayerResult l2;
    l2.layer_id = 2;

    // SpMM for layer 2: Y2 = A × H1_hw
    PEArraySpMM::Stats pe2_stats = run_pe_job(a_new, h1_current_.data());
    const size_t y2_size = static_cast<size_t>(n_) * static_cast<size_t>(d_out_);
    std::vector<float> y2_current(y2_size, 0.0f);
    const float* pe_out = pe_->get_output();
    if (pe_out) {
        std::copy(pe_out, pe_out + y2_size, y2_current.begin());
    }
    l2.spmm_cycles = pe2_stats.total_cycles;

    // Reload CiM with layer 2 weights
    if (cim_) {
        cim_->reset_counters();
        load_resident_weights_for_layer(2);
    }

    CimHardwareRunResult hw2 = run_cim_hardware_rows(y2_current, all_rows);
    z2_current_ = hw2.output;
    l2.cim_cycles = hw2.cycles;
    if (cim_) l2.cim_energy_pj = cim_->get_total_energy_pj();

    // =========================================================================
    // Golden end-to-end for layer 2
    // =========================================================================
    // Golden H1 = ReLU(Y1 × W1_fp32)
    std::vector<float> h1_golden(z1_golden.size());
    for (size_t i = 0; i < z1_golden.size(); i++) {
        h1_golden[i] = std::max(0.0f, z1_golden[i]);
    }
    // Golden Y2 = A × H1_golden (dense matmul via CSR)
    const int d_hidden = d_out_;
    std::vector<float> y2_golden(static_cast<size_t>(n_) * d_hidden, 0.0f);
    for (int row = 0; row < n_; row++) {
        for (int ptr = a_new.row_ptr[row]; ptr < a_new.row_ptr[row + 1]; ptr++) {
            const int col = a_new.col_idx[ptr];
            const float val = a_new.values.empty() ? 1.0f : a_new.values[ptr];
            for (int k = 0; k < d_hidden; k++) {
                y2_golden[static_cast<size_t>(row) * d_hidden + k] +=
                    val * h1_golden[static_cast<size_t>(col) * d_hidden + k];
            }
        }
    }
    // Golden Z2 = Y2_golden × W2
    z2_golden_.resize(static_cast<size_t>(n_) * d_out_l2_, 0.0f);
    for (int row = 0; row < n_; row++) {
        for (int j = 0; j < d_out_l2_; j++) {
            float sum = 0.0f;
            for (int k = 0; k < d_hidden; k++) {
                sum += y2_golden[static_cast<size_t>(row) * d_hidden + k] *
                       weights_l2_[static_cast<size_t>(k) * d_out_l2_ + j];
            }
            z2_golden_[static_cast<size_t>(row) * d_out_l2_ + j] = sum;
        }
    }

    l2.accuracy = compute_accuracy_metrics(z2_current_, z2_golden_, hw2);
    result.layers.push_back(l2);

    // =========================================================================
    // End-to-end accuracy
    // =========================================================================
    CimHardwareRunResult hw_e2e;
    hw_e2e.ok = hw1.ok && hw2.ok;
    hw_e2e.expected_rows = n_;
    hw_e2e.received_unique_rows = n_;
    result.final_accuracy = compute_accuracy_metrics(z2_current_, z2_golden_, hw_e2e);
    result.total_cycles = l1.cim_cycles + l1.activation_cycles +
                          l2.spmm_cycles + l2.cim_cycles;
    result.total_energy_pj = l1.cim_energy_pj + l2.cim_energy_pj;
    result.eval_cim_energy_pj = result.total_energy_pj;

    // =========================================================================
    // Export binary files
    // =========================================================================
    if (do_export_z && !export_dir.empty()) {
        export_binary(export_dir + "/z2_hw.bin", z2_current_);
        export_binary(export_dir + "/z2_golden.bin", z2_golden_);
        export_binary(export_dir + "/h1_hw.bin", h1_current_);
        std::cout << "[DyGNN] Exported z2_hw.bin (" << z2_current_.size()
                  << " floats), z2_golden.bin, h1_hw.bin to " << export_dir << "\n";
    }

    // Do not speculatively restore L1. The later L2->L1 operation owns that
    // transition and its single explicit load event.

    return result;
}

// =============================================================================
// Multi-Layer GCN (N-Layer End-to-End, N >= 3)
// =============================================================================

void DualModeSpMM::load_weights_l3(const std::string& path, int d_in, int d_out) {
    d_out_l3_ = d_out;
    load_required_fp32_weights(path, d_in, d_out, "L3", &weights_l3_);
}

void DualModeSpMM::load_weights_l4(const std::string& path, int d_in, int d_out) {
    d_out_l4_ = d_out;
    load_required_fp32_weights(path, d_in, d_out, "L4", &weights_l4_);
}

void DualModeSpMM::load_weights_l5(const std::string& path, int d_in, int d_out) {
    d_out_l5_ = d_out;
    load_required_fp32_weights(path, d_in, d_out, "L5", &weights_l5_);
}

DualModeSpMM::MultiLayerResult DualModeSpMM::run_multi_layer_gcn(
    int num_layers,
    const WorkStealingScheduler::CSRMatrix& a_new,
    bool do_export_z,
    const std::string& export_dir) {

    if (num_layers <= 2) {
        return run_two_layer_gcn(a_new, do_export_z, export_dir);
    }

    MultiLayerResult result;
    result.num_layers = num_layers;
    if (cim_) {
        result.runtime_cim_energy_pj = cim_->get_total_energy_pj();
    }

    std::vector<int> all_rows;
    all_rows.reserve(static_cast<size_t>(n_));
    for (int row = 0; row < n_; row++) all_rows.push_back(row);

    // Collect weights for each layer
    std::vector<std::vector<float>*> layer_weights = {&weights_, &weights_l2_, &weights_l3_, &weights_l4_, &weights_l5_};
    std::vector<int> layer_d_out = {d_out_, d_out_l2_, d_out_l3_, d_out_l4_, d_out_l5_};

    // Fresh-recompute Y = A × F to avoid incremental drift in multi-timestep runs.
    // y_current_ accumulates FP32 rounding over 99 timesteps, causing magnitude
    // explosion. golden_full_spmm recomputes from scratch for correct evaluation.
    std::vector<float> y_fresh = golden_full_spmm(a_new);
    if (y_fresh.empty()) {
        y_fresh = y_current_;  // fallback if golden_full_spmm fails
    }
    std::vector<float> h_hw = y_fresh;
    std::vector<float> h_golden = y_fresh;
    int current_dim = d_out_;

    long long total_spmm_cycles = 0;
    long long total_cim_cycles = 0;
    double total_cim_energy = 0.0;

    // Process each layer
    for (int layer = 1; layer <= num_layers; layer++) {
        LayerResult lr;
        lr.layer_id = layer;

        // Get weights for this layer
        std::vector<float>* w_ptr = layer_weights[std::min(layer - 1, 4)];
        int next_dim = layer_d_out[std::min(layer - 1, 4)];

        // CiM: Z = H × W
        if (cim_ && !w_ptr->empty()) {
            cim_->reset_counters();
            load_resident_weights_for_layer(layer);
        }

        CimHardwareRunResult hw = run_cim_hardware_rows(h_hw, all_rows);
        std::vector<float> z_hw = hw.output;
        lr.cim_cycles = hw.cycles;
        if (cim_) lr.cim_energy_pj = cim_->get_total_energy_pj();
        total_cim_cycles += lr.cim_cycles;
        total_cim_energy += lr.cim_energy_pj;

        // Golden: Z_golden = H_golden × W
        std::vector<float> z_golden(static_cast<size_t>(n_) * next_dim, 0.0f);
        for (int row = 0; row < n_; row++) {
            for (int j = 0; j < next_dim; j++) {
                float sum = 0.0f;
                for (int k = 0; k < current_dim; k++) {
                    sum += h_golden[static_cast<size_t>(row) * current_dim + k] *
                           (*w_ptr)[static_cast<size_t>(k) * next_dim + j];
                }
                z_golden[static_cast<size_t>(row) * next_dim + j] = sum;
            }
        }

        // Accuracy for this layer
        lr.accuracy = compute_accuracy_metrics(z_hw, z_golden, hw);
        result.layers.push_back(lr);

        // Last layer: no ReLU, no SpMM
        if (layer == num_layers) {
            z_final_ = z_hw;
            z_final_golden_ = z_golden;
            result.final_accuracy = lr.accuracy;
            break;
        }

        // ReLU activation + per-row L2 normalization (standard in deep GCN)
        const size_t z_size = z_hw.size();
        std::vector<float> h_next_hw(z_size);
        std::vector<float> h_next_golden(z_size);
        for (size_t i = 0; i < z_size; i++) {
            h_next_hw[i] = std::max(0.0f, z_hw[i]);
            h_next_golden[i] = std::max(0.0f, z_golden[i]);
        }
        // Per-row L2 normalize to stabilize magnitude across layers
        for (int row = 0; row < n_; row++) {
            float norm_hw = 0.0f, norm_gd = 0.0f;
            const size_t base = static_cast<size_t>(row) * next_dim;
            for (int k = 0; k < next_dim; k++) {
                norm_hw += h_next_hw[base + k] * h_next_hw[base + k];
                norm_gd += h_next_golden[base + k] * h_next_golden[base + k];
            }
            norm_hw = std::sqrt(norm_hw) + 1e-8f;
            norm_gd = std::sqrt(norm_gd) + 1e-8f;
            for (int k = 0; k < next_dim; k++) {
                h_next_hw[base + k] /= norm_hw;
                h_next_golden[base + k] /= norm_gd;
            }
        }
        lr.activation_cycles = 0;

        // SpMM for next layer: Y = A × H
        PEArraySpMM::Stats pe_stats = run_pe_job(a_new, h_next_hw.data());
        lr.spmm_cycles = pe_stats.total_cycles;
        total_spmm_cycles += lr.spmm_cycles;

        // Update hidden state
        const size_t y_size = static_cast<size_t>(n_) * next_dim;
        h_hw.resize(y_size);
        const float* pe_out = pe_->get_output();
        if (pe_out) std::copy(pe_out, pe_out + y_size, h_hw.begin());

        // Golden SpMM: Y_golden = A × H_golden
        h_golden.resize(y_size, 0.0f);
        std::fill(h_golden.begin(), h_golden.end(), 0.0f);
        for (int row = 0; row < n_; row++) {
            for (int ptr = a_new.row_ptr[row]; ptr < a_new.row_ptr[row + 1]; ptr++) {
                const int col = a_new.col_idx[ptr];
                const float val = a_new.values.empty() ? 1.0f : a_new.values[ptr];
                for (int k = 0; k < next_dim; k++) {
                    h_golden[static_cast<size_t>(row) * next_dim + k] +=
                        val * h_next_golden[static_cast<size_t>(col) * next_dim + k];
                }
            }
        }

        current_dim = next_dim;
    }

    result.total_cycles = total_spmm_cycles + total_cim_cycles;
    result.total_energy_pj = total_cim_energy;
    result.eval_cim_energy_pj = total_cim_energy;

    // Export if requested
    if (do_export_z && !export_dir.empty()) {
        export_binary(export_dir + "/z_final_hw.bin", z_final_);
        if (!z_final_golden_.empty()) {
            export_binary(export_dir + "/z_final_golden.bin", z_final_golden_);
        }
        std::cout << "[DyGNN] Exported z_final_hw.bin + z_final_golden.bin ("
                  << z_final_.size() << " floats) for " << num_layers << "-layer GCN\n";
    }

    // No speculative L1 restore; the next operation performs the transition.

    std::cout << "[Multi-Layer] " << num_layers << "-layer GCN completed. "
              << "Final CosSim=" << result.final_accuracy.cosine_similarity
              << ", RMSE=" << result.final_accuracy.rmse << "\n";

    return result;
}
