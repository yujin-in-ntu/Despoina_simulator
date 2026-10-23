// SPDX-License-Identifier: Apache-2.0
#ifndef FINALIZED_DCIM_V3_H
#define FINALIZED_DCIM_V3_H

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <deque>
#include <limits>
#include <stdexcept>
#include <unordered_set>
#include <vector>

namespace despoina_v3 {

inline bool cache_epoch_can_reuse(bool caller_requests_reuse,
                                  bool current_epoch_valid,
                                  uint64_t current_epoch,
                                  uint64_t requested_epoch) {
    return caller_requests_reuse && current_epoch_valid &&
           current_epoch == requested_epoch;
}

// Paper-provable publication contract. Figure 8 fixes Din=Dout=128.  The
// hardware timing model below deliberately rejects every other shape; smaller
// and unequal shapes are permitted only by the functional INT8 reference.
struct PhysicalContract {
    static constexpr int kPublicationDim = 128;
    static constexpr int kPaperTiles = 16;
    static constexpr int kPaperAggregateRawBitOpsPerCycle = 2048;
    // NEW_EXPLICIT_ARCHITECTURE_DECISION: full-W replication and row-level
    // parallelism.  Paper aggregate width / 16 identical tiles = 128 signed
    // INT8 bit operations per tile-cycle.
    static constexpr int kPerTileRawBitOpsPerCycle =
        kPaperAggregateRawBitOpsPerCycle / kPaperTiles;
    static constexpr int kBitSerialSteps = 8;
    static constexpr int kScalarMacsPerFinalizedRow =
        kPublicationDim * kPublicationDim;
    static constexpr int kCyclesPerBitStep =
        kScalarMacsPerFinalizedRow / kPerTileRawBitOpsPerCycle;
    static constexpr int kPerTileRowServiceCycles =
        kBitSerialSteps * kCyclesPerBitStep;
};

// Existing implementation-decision sensitivity.  This is not folded into the
// causal compute-event count and is always reported as a separate field.
struct ImplementationDecision {
    static constexpr int kDrainCyclesPerStream = 6;
};

// Cycle quantities with intentionally incompatible types.  Busy work is a
// resource-usage sum over tiles; elapsed and exposed-tail cycles are latency
// quantities on one causal clock.  No implicit integer conversion is provided,
// so busy work cannot accidentally enter a top-level latency expression.
struct TileBusyWorkCycles {
    explicit TileBusyWorkCycles(uint64_t v = 0) : value(v) {}
    uint64_t value;
};

struct CausalStreamElapsedCycles {
    explicit CausalStreamElapsedCycles(uint64_t v = 0) : value(v) {}
    uint64_t value;
};

struct ExposedTailCycles {
    explicit ExposedTailCycles(uint64_t v = 0) : value(v) {}
    uint64_t value;
};

inline void accumulate(TileBusyWorkCycles& dst, TileBusyWorkCycles src) {
    if (src.value > std::numeric_limits<uint64_t>::max() - dst.value) {
        throw std::overflow_error("tile busy-work cycle overflow");
    }
    dst.value += src.value;
}

inline void accumulate(CausalStreamElapsedCycles& dst,
                       CausalStreamElapsedCycles src) {
    if (src.value > std::numeric_limits<uint64_t>::max() - dst.value) {
        throw std::overflow_error("causal elapsed cycle overflow");
    }
    dst.value += src.value;
}

inline void accumulate(ExposedTailCycles& dst, ExposedTailCycles src) {
    if (src.value > std::numeric_limits<uint64_t>::max() - dst.value) {
        throw std::overflow_error("exposed-tail cycle overflow");
    }
    dst.value += src.value;
}

inline uint64_t add_exposed_tail_once(uint64_t causal_prefix_cycles,
                                      ExposedTailCycles exposed_tail) {
    if (exposed_tail.value >
        std::numeric_limits<uint64_t>::max() - causal_prefix_cycles) {
        throw std::overflow_error("top-level causal cycle overflow");
    }
    return causal_prefix_cycles + exposed_tail.value;
}

inline void assert_publication_cycle_shape(uint64_t input_dim,
                                           uint64_t output_dim) {
    if (input_dim != PhysicalContract::kPublicationDim ||
        output_dim != PhysicalContract::kPublicationDim) {
        throw std::invalid_argument(
            "publication DCIM cycle model requires Din=Dout=128");
    }
}

inline uint64_t checked_scalar_macs(uint64_t rows,
                                    uint64_t input_dim,
                                    uint64_t output_dim) {
    if (input_dim == 0 || output_dim == 0) {
        throw std::invalid_argument("DCIM dimensions must be positive");
    }
    if (rows > std::numeric_limits<uint64_t>::max() / input_dim) {
        throw std::overflow_error("rows*input_dim overflow");
    }
    const uint64_t partial = rows * input_dim;
    if (partial > std::numeric_limits<uint64_t>::max() / output_dim) {
        throw std::overflow_error("rows*input_dim*output_dim overflow");
    }
    return partial * output_dim;
}

inline std::vector<int> coalesce_rows_once(const std::vector<int>& candidates,
                                           int nrows) {
    if (nrows < 0) {
        throw std::invalid_argument("negative row count");
    }
    std::vector<int> out;
    out.reserve(candidates.size());
    std::unordered_set<int> seen;
    for (int row : candidates) {
        if (row < 0 || row >= nrows) {
            throw std::out_of_range("invalid finalized row");
        }
        if (seen.insert(row).second) {
            out.push_back(row);
        }
    }
    std::sort(out.begin(), out.end());
    return out;
}

struct TimingStats {
    uint64_t finalized_rows = 0;
    uint64_t packets = 0;
    uint64_t scalar_macs = 0;
    TileBusyWorkCycles tile_busy_work_cycles{};
    CausalStreamElapsedCycles causal_stream_elapsed_cycles{};
    ExposedTailCycles exposed_tail_cycles{};
    uint64_t implementation_plus6_finish_cycle = 0;
    uint64_t implementation_plus6_exposed_cycles = 0;
    uint64_t overlap_cycles = 0;
    uint64_t backpressure_cycles = 0;
    uint64_t max_staging_rows = 0;
    uint64_t max_fifo_rows = 0;
    uint64_t bit_serial_steps = PhysicalContract::kBitSerialSteps;
    uint64_t implementation_drain_cycles =
        ImplementationDecision::kDrainCyclesPerStream;
    bool timeout = false;
    uint64_t duplicate_packets = 0;
    uint64_t invalid_packets = 0;
    uint64_t physical_tiles = 0;
    uint64_t per_tile_raw_bit_ops_per_cycle =
        PhysicalContract::kPerTileRawBitOpsPerCycle;
    uint64_t cycles_per_bit_step = PhysicalContract::kCyclesPerBitStep;
    uint64_t per_tile_row_service_cycles =
        PhysicalContract::kPerTileRowServiceCycles;
    uint64_t bit_step_issue_events = 0;
    uint64_t bit_step_completion_events = 0;
    uint64_t tile_dispatch_events = 0;
    uint64_t weight_program_cycles = 0;
    uint64_t weight_program_tile_events = 0;
    uint64_t weight_source_fp32_bytes = 0;
    uint64_t weight_programmed_int8_bytes = 0;
    uint64_t overlap_sensitivity_finish_cycle = 0;
    uint64_t overlap_sensitivity_exposed_cycles = 0;
    std::vector<uint64_t> rows_per_tile;
    uint64_t bit_step_trace_fnv1a64 = 1469598103934665603ULL;
    uint64_t first_bit_step_issue_cycle = 0;
    uint64_t last_bit_step_completion_cycle = 0;
    struct BitStepEvent {
        BitStepEvent(uint64_t row_sequence_in,
                     uint64_t tile_in,
                     uint64_t step_in,
                     uint64_t issue_cycle_in,
                     uint64_t completion_cycle_in)
            : row_sequence(row_sequence_in),
              tile(tile_in),
              step(step_in),
              issue_cycle(issue_cycle_in),
              completion_cycle(completion_cycle_in) {}
        uint64_t row_sequence;
        uint64_t tile;
        uint64_t step;
        uint64_t issue_cycle;
        uint64_t completion_cycle;
    };
    std::vector<BitStepEvent> bit_step_events;
};

struct CausalScheduleCore {
    uint64_t finish = 0;
    uint64_t producer_end = 0;
    uint64_t backpressure = 0;
    uint64_t max_staging = 0;
    uint64_t max_fifo = 0;
    uint64_t bit_step_issues = 0;
    uint64_t bit_step_completions = 0;
    std::vector<uint64_t> rows_per_tile;
    uint64_t trace_hash = 1469598103934665603ULL;
    uint64_t first_step_issue = 0;
    uint64_t last_step_completion = 0;
    std::vector<TimingStats::BitStepEvent> bit_step_events;
};

inline void trace_mix_u64(uint64_t& h, uint64_t value) {
    for (int b = 0; b < 8; ++b) {
        h ^= (value >> (b * 8)) & 0xffULL;
        h *= 1099511628211ULL;
    }
}

inline CausalScheduleCore schedule_full_w_replicated_rows(
    std::vector<uint64_t> ready_cycles,
    uint64_t producer_end_cycle,
    const std::vector<uint64_t>& tile_program_ready,
    uint64_t staging_capacity_rows,
    uint64_t per_tile_fifo_depth) {
    if (tile_program_ready.empty() || staging_capacity_rows == 0 ||
        per_tile_fifo_depth == 0) {
        throw std::invalid_argument("physical tiles and bounded queues must be positive");
    }
    std::sort(ready_cycles.begin(), ready_cycles.end());
    const size_t tiles = tile_program_ready.size();
    std::vector<std::deque<uint64_t>> tile_retire(tiles);
    std::vector<uint64_t> tile_tail(tiles, 0);
    std::deque<uint64_t> staging_dispatch_times;
    CausalScheduleCore out;
    out.rows_per_tile.assign(tiles, 0);
    uint64_t producer_delay = 0;
    uint64_t ingress_free = 0;

    uint64_t row_sequence = 0;
    for (uint64_t nominal_ready : ready_cycles) {
        uint64_t ready = nominal_ready + producer_delay;
        while (!staging_dispatch_times.empty() &&
               staging_dispatch_times.front() <= ready) {
            staging_dispatch_times.pop_front();
        }
        if (staging_dispatch_times.size() >= staging_capacity_rows) {
            const uint64_t release = staging_dispatch_times.front();
            const uint64_t wait = release > ready ? release - ready : 0;
            producer_delay += wait;
            out.backpressure += wait;
            ready += wait;
            while (!staging_dispatch_times.empty() &&
                   staging_dispatch_times.front() <= ready) {
                staging_dispatch_times.pop_front();
            }
        }
        out.max_staging = std::max<uint64_t>(
            out.max_staging, staging_dispatch_times.size() + 1);

        uint64_t probe = std::max(ready, ingress_free);
        size_t chosen = 0;
        uint64_t chosen_accept = std::numeric_limits<uint64_t>::max();
        uint64_t chosen_service = std::numeric_limits<uint64_t>::max();
        for (size_t t = 0; t < tiles; ++t) {
            while (!tile_retire[t].empty() && tile_retire[t].front() <= probe) {
                tile_retire[t].pop_front();
            }
            uint64_t accept = probe;
            if (tile_retire[t].size() >= per_tile_fifo_depth) {
                accept = std::max(accept, tile_retire[t].front());
            }
            const uint64_t projected_service = std::max(
                std::max(accept + 1, tile_tail[t]), tile_program_ready[t]);
            if (projected_service < chosen_service ||
                (projected_service == chosen_service && t < chosen)) {
                chosen = t;
                chosen_accept = accept;
                chosen_service = projected_service;
            }
        }
        const uint64_t dispatch = chosen_accept;
        ingress_free = dispatch + 1;  // one 512-B FP32 row packet/cycle
        staging_dispatch_times.push_back(dispatch);

        while (!tile_retire[chosen].empty() &&
               tile_retire[chosen].front() <= dispatch) {
            tile_retire[chosen].pop_front();
        }
        const uint64_t quant_done = dispatch + 1;
        uint64_t service = std::max(
            std::max(quant_done, tile_tail[chosen]), tile_program_ready[chosen]);
        for (int step = 0; step < PhysicalContract::kBitSerialSteps; ++step) {
            const uint64_t issue = service;
            ++out.bit_step_issues;
            service += PhysicalContract::kCyclesPerBitStep;
            const uint64_t completion = service;
            ++out.bit_step_completions;
            out.bit_step_events.push_back(TimingStats::BitStepEvent{
                row_sequence,
                static_cast<uint64_t>(chosen),
                static_cast<uint64_t>(step),
                issue,
                completion});
            if (out.first_step_issue == 0) out.first_step_issue = issue;
            out.last_step_completion = std::max(out.last_step_completion, completion);
            trace_mix_u64(out.trace_hash, row_sequence);
            trace_mix_u64(out.trace_hash, static_cast<uint64_t>(chosen));
            trace_mix_u64(out.trace_hash, static_cast<uint64_t>(step));
            trace_mix_u64(out.trace_hash, issue);
            trace_mix_u64(out.trace_hash, completion);
        }
        const uint64_t done = service + 1;  // INT32->FP32 dequant completion
        tile_tail[chosen] = done;
        tile_retire[chosen].push_back(done);
        out.max_fifo = std::max<uint64_t>(out.max_fifo, tile_retire[chosen].size());
        ++out.rows_per_tile[chosen];
        out.finish = std::max(out.finish, done);
        ++row_sequence;
    }
    out.producer_end = producer_end_cycle + producer_delay;
    return out;
}

// Causal NEW_EXPLICIT_ARCHITECTURE_DECISION model.  Primary timing does not
// overlap layer-entry W programming with the PE producer.  A separately labeled
// sensitivity permits producer/program overlap using per-tile DRAM completion
// timestamps; neither path inserts rows*64 or post-processes a tile count.
inline TimingStats simulate_causal_full_w_replicated(
    const std::vector<uint64_t>& ready_cycles,
    uint64_t producer_end_cycle,
    uint64_t input_dim,
    uint64_t output_dim,
    uint64_t staging_capacity_rows,
    uint64_t per_tile_fifo_depth,
    uint64_t physical_tiles,
    uint64_t weight_program_total_cycles,
    const std::vector<uint64_t>& weight_tile_program_done,
    uint64_t weight_source_fp32_bytes,
    uint64_t weight_programmed_int8_bytes) {
    assert_publication_cycle_shape(input_dim, output_dim);
    if (physical_tiles != 16 && physical_tiles != 32 && physical_tiles != 128) {
        throw std::invalid_argument("R5 physical tiles must be 16, 32, or 128");
    }
    if (weight_program_total_cycles > 0 &&
        weight_tile_program_done.size() != physical_tiles) {
        throw std::invalid_argument("per-tile W programming completion missing");
    }
    TimingStats s;
    s.physical_tiles = physical_tiles;
    s.finalized_rows = ready_cycles.size();
    s.packets = ready_cycles.size();
    s.scalar_macs = checked_scalar_macs(ready_cycles.size(), input_dim, output_dim);
    s.weight_program_cycles = weight_program_total_cycles;
    s.weight_program_tile_events = weight_program_total_cycles > 0 ? physical_tiles : 0;
    s.weight_source_fp32_bytes = weight_source_fp32_bytes;
    s.weight_programmed_int8_bytes = weight_programmed_int8_bytes;

    std::vector<uint64_t> primary_ready = ready_cycles;
    for (uint64_t& v : primary_ready) v += weight_program_total_cycles;
    std::vector<uint64_t> primary_tile_ready(physical_tiles, weight_program_total_cycles);
    CausalScheduleCore primary = schedule_full_w_replicated_rows(
        primary_ready, producer_end_cycle + weight_program_total_cycles,
        primary_tile_ready, staging_capacity_rows, per_tile_fifo_depth);
    s.causal_stream_elapsed_cycles = CausalStreamElapsedCycles{primary.finish};
    s.implementation_plus6_finish_cycle = primary.finish +
        (ready_cycles.empty() ? 0 : ImplementationDecision::kDrainCyclesPerStream);
    // Added to the original PE producer duration, so the primary exposed term
    // includes the non-overlapped layer-entry programming interval.
    s.exposed_tail_cycles = ExposedTailCycles{
        primary.finish > producer_end_cycle
            ? primary.finish - producer_end_cycle : 0};
    s.implementation_plus6_exposed_cycles =
        s.implementation_plus6_finish_cycle > producer_end_cycle
        ? s.implementation_plus6_finish_cycle - producer_end_cycle : 0;
    s.backpressure_cycles = primary.backpressure;
    s.max_staging_rows = primary.max_staging;
    s.max_fifo_rows = primary.max_fifo;
    s.bit_step_issue_events = primary.bit_step_issues;
    s.bit_step_completion_events = primary.bit_step_completions;
    s.tile_dispatch_events = ready_cycles.size();
    s.rows_per_tile = primary.rows_per_tile;
    s.bit_step_trace_fnv1a64 = primary.trace_hash;
    s.first_bit_step_issue_cycle = primary.first_step_issue;
    s.last_bit_step_completion_cycle = primary.last_step_completion;
    s.bit_step_events = primary.bit_step_events;
    // Busy cycles emerge by summing eight emitted step durations per row.
    s.tile_busy_work_cycles = TileBusyWorkCycles{
        primary.bit_step_completions *
            PhysicalContract::kCyclesPerBitStep};
    const uint64_t serial = primary.producer_end + s.tile_busy_work_cycles.value;
    s.overlap_cycles = serial > primary.finish ? serial - primary.finish : 0;

    std::vector<uint64_t> overlap_tile_ready = weight_program_total_cycles > 0
        ? weight_tile_program_done : std::vector<uint64_t>(physical_tiles, 0);
    CausalScheduleCore overlap = schedule_full_w_replicated_rows(
        ready_cycles, producer_end_cycle, overlap_tile_ready,
        staging_capacity_rows, per_tile_fifo_depth);
    s.overlap_sensitivity_finish_cycle = overlap.finish;
    s.overlap_sensitivity_exposed_cycles = overlap.finish > overlap.producer_end
        ? overlap.finish - overlap.producer_end : 0;
    return s;
}

struct DirectEnergyEvents {
    uint64_t finalized_rows = 0;
    uint64_t scalar_int8_macs = 0;
    uint64_t activation_quantized_elements = 0;
    uint64_t dequantized_elements = 0;
    uint64_t comparator_tree_passes = 0;
};

inline DirectEnergyEvents direct_energy_events_once(uint64_t finalized_rows,
                                                    uint64_t input_dim,
                                                    uint64_t output_dim) {
    DirectEnergyEvents e;
    e.finalized_rows = finalized_rows;
    e.scalar_int8_macs = checked_scalar_macs(
        finalized_rows, input_dim, output_dim);
    e.activation_quantized_elements = finalized_rows * input_dim;
    e.dequantized_elements = finalized_rows * output_dim;
    e.comparator_tree_passes = finalized_rows;
    return e;
}

struct QuantizedDenseResult {
    std::vector<float> output;
    std::vector<float> activation_scales;
    float weight_scale = 1.0f;
    DirectEnergyEvents events{};
};

// CPU functional reference for the exact DCIM_DIGITAL arithmetic:
// global symmetric INT8 weight quantization, per-row dynamic symmetric INT8
// activation quantization, INT32 accumulation, then FP32 dequantization.
// Timing is intentionally absent so variable dimensions remain functional-only.
inline QuantizedDenseResult quantized_dense_rows_once(
                                          const std::vector<float>& y,
                                          const std::vector<int>& rows,
                                          int nrows,
                                          int input_dim,
                                          const std::vector<float>& weights,
                                          int output_dim) {
    if (nrows < 0 || input_dim <= 0 || output_dim <= 0 ||
        y.size() != static_cast<size_t>(nrows) * static_cast<size_t>(input_dim) ||
        weights.size() != static_cast<size_t>(input_dim) * static_cast<size_t>(output_dim)) {
        throw std::invalid_argument("quantized_dense_rows_once shape mismatch");
    }
    const std::vector<int> unique_rows = coalesce_rows_once(rows, nrows);
    if (unique_rows.size() != rows.size()) {
        throw std::invalid_argument("duplicate dense-row application");
    }
    QuantizedDenseResult result;
    result.output.assign(unique_rows.size() * static_cast<size_t>(output_dim), 0.0f);
    result.activation_scales.assign(unique_rows.size(), 1.0f);
    result.events = direct_energy_events_once(
        unique_rows.size(), input_dim, output_dim);

    float w_abs_max = 0.0f;
    for (float w : weights) w_abs_max = std::max(w_abs_max, std::fabs(w));
    result.weight_scale = w_abs_max > 1e-10f ? w_abs_max / 127.0f : 1.0f;
    std::vector<int8_t> wq(weights.size(), 0);
    for (size_t i = 0; i < weights.size(); ++i) {
        float q = std::round(weights[i] / result.weight_scale);
        q = std::max(-128.0f, std::min(127.0f, q));
        wq[i] = static_cast<int8_t>(q);
    }

    for (size_t p = 0; p < unique_rows.size(); ++p) {
        const float* in = y.data() + static_cast<size_t>(unique_rows[p]) * input_dim;
        float x_abs_max = 0.0f;
        for (int k = 0; k < input_dim; ++k) {
            x_abs_max = std::max(x_abs_max, std::fabs(in[k]));
        }
        const float x_scale = x_abs_max > 1e-10f ? x_abs_max / 127.0f : 1.0f;
        result.activation_scales[p] = x_scale;
        std::vector<int8_t> xq(static_cast<size_t>(input_dim), 0);
        for (int k = 0; k < input_dim; ++k) {
            float q = std::round(in[k] / x_scale);
            q = std::max(-128.0f, std::min(127.0f, q));
            xq[static_cast<size_t>(k)] = static_cast<int8_t>(q);
        }
        float* dst = result.output.data() + p * static_cast<size_t>(output_dim);
        for (int j = 0; j < output_dim; ++j) {
            int32_t acc = 0;
            for (int k = 0; k < input_dim; ++k) {
                acc += static_cast<int32_t>(xq[static_cast<size_t>(k)]) *
                       static_cast<int32_t>(wq[static_cast<size_t>(k) * output_dim + j]);
            }
            dst[j] = static_cast<float>(acc) * x_scale * result.weight_scale;
        }
    }
    return result;
}

inline void apply_signed_finalized_rows(std::vector<float>& y_state,
                                        const std::vector<int>& rows,
                                        int nrows,
                                        int dim,
                                        const std::vector<float>& phase1,
                                        const std::vector<float>& phase2,
                                        const std::vector<float>& phase3) {
    if (nrows < 0 || dim <= 0 ||
        y_state.size() != static_cast<size_t>(nrows) * static_cast<size_t>(dim) ||
        phase1.size() != y_state.size() || phase2.size() != y_state.size() ||
        phase3.size() != y_state.size()) {
        throw std::invalid_argument("signed phase shape mismatch");
    }
    const std::vector<int> unique_rows = coalesce_rows_once(rows, nrows);
    if (unique_rows.size() != rows.size()) {
        throw std::invalid_argument("duplicate finalized row ownership");
    }
    for (int row : unique_rows) {
        const size_t base = static_cast<size_t>(row) * static_cast<size_t>(dim);
        for (int j = 0; j < dim; ++j) {
            const size_t idx = base + static_cast<size_t>(j);
            y_state[idx] += phase1[idx] + phase2[idx] - phase3[idx];
        }
    }
}

}  // namespace despoina_v3

#endif  // FINALIZED_DCIM_V3_H
