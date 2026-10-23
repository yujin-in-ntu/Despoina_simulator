// SPDX-License-Identifier: Apache-2.0
#ifndef DYNAMIC_ROW_MEAN_OPERATOR_H
#define DYNAMIC_ROW_MEAN_OPERATOR_H

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <map>
#include <stdexcept>
#include <vector>

#include "delta_converter.h"
#include "work_stealing_scheduler.h"

// Exact paper-aligned mean aggregation owner:
//     M_t = D_t^{-1} B_t,
// where D_t is the destination/CSR-row degree.  The current raw Figure-9
// lineage has no self loops, so the fixed operator adds none and rejects an
// explicit diagonal entry rather than silently changing the convention.
//
// The implementation never materializes dM.  Given the existing normalized
// state y_old and the raw three-phase numerator delta
//     q = dB H_new + B_new scoped(dH) - dB scoped(dH),
// it applies, row by row,
//     y_new = (d_old / d_new) y_old + q / d_new.
// This is the exact scoped recurrence.  Scaling y_old + q by 1/d_new is wrong
// whenever d_old != 1 and is deliberately not provided.
class DynamicRowMeanOperator {
public:
    using CSRMatrix = WorkStealingScheduler::CSRMatrix;
    using DeltaAUpdate = DeltaConverter::DeltaAUpdate;

    struct Transition {
        struct RowChange {
            int row = -1;
            int old_degree = 0;
            int new_degree = 0;
            float old_over_new = 0.0f;
            float inv_new = 0.0f;

            RowChange() = default;
            RowChange(int row_value, int old_value, int new_value,
                      float ratio_value, float reciprocal_value)
                : row(row_value), old_degree(old_value), new_degree(new_value),
                  old_over_new(ratio_value), inv_new(reciprocal_value) {}
        };
        std::vector<RowChange> changes;
        std::vector<int> changed_rows;
        std::uint64_t requested_updates = 0;
        std::uint64_t effective_updates = 0;
        std::uint64_t no_ops = 0;
        std::uint64_t degree_read_events = 0;
        std::uint64_t degree_write_events = 0;
        std::uint64_t degree_read_bytes = 0;
        std::uint64_t degree_write_bytes = 0;
        std::uint64_t reciprocal_events = 0;

        const RowChange* find(int row) const {
            const auto it = std::lower_bound(
                changes.begin(), changes.end(), row,
                [](const RowChange& change, int value) { return change.row < value; });
            return it != changes.end() && it->row == row ? &*it : nullptr;
        }
    };

    void reset(const CSRMatrix& graph) {
        validate_binary_no_self_loop(graph);
        n_ = graph.n;
        degree_ = degrees(graph);
        reciprocal_.resize(degree_.size());
        for (std::size_t row = 0; row < degree_.size(); ++row) {
            reciprocal_[row] = deterministic_reciprocal(degree_[row]);
        }
    }

    bool initialized() const { return n_ > 0 && degree_.size() == static_cast<std::size_t>(n_); }
    const std::vector<int>& degree() const { return degree_; }
    const std::vector<float>& reciprocal() const { return reciprocal_; }

    int new_degree(int row) const {
        if (row < 0 || row >= n_) throw std::out_of_range("row-mean degree row");
        return degree_[static_cast<std::size_t>(row)];
    }

    float new_reciprocal(int row) const {
        if (row < 0 || row >= n_) throw std::out_of_range("row-mean reciprocal row");
        return reciprocal_[static_cast<std::size_t>(row)];
    }

    int old_degree(int row, const Transition& transition) const {
        const auto* change = transition.find(row);
        return change == nullptr ? new_degree(row) : change->old_degree;
    }

    float old_over_new(int row, const Transition& transition) const {
        const auto* change = transition.find(row);
        if (change != nullptr) return change->old_over_new;
        return new_degree(row) == 0 ? 0.0f : 1.0f;
    }

    Transition prepare(const CSRMatrix& graph_new,
                       const std::vector<DeltaAUpdate>& requested) {
        if (!initialized()) throw std::logic_error("row-mean owner is not initialized");
        if (graph_new.n != n_ || graph_new.row_ptr.size() != static_cast<std::size_t>(n_ + 1)) {
            throw std::invalid_argument("row-mean graph dimension changed");
        }

        Transition t;
        const auto updates = DeltaConverter::coalesce_delta_a_updates(requested);
        t.requested_updates = updates.size();
        std::map<int, int> degree_delta;
        for (const auto& u : updates) {
            if (u.src < 0 || u.src >= n_ || u.dst < 0 || u.dst >= n_ ||
                u.src == u.dst || !std::isfinite(u.w) || u.w != 1.0f) {
                throw std::invalid_argument("row-mean delta violates binary/no-self-loop contract");
            }
            degree_delta[u.src] += u.is_add ? 1 : -1;
            ++t.effective_updates;
        }
        t.changes.reserve(degree_delta.size());
        t.changed_rows.reserve(degree_delta.size());
        for (const auto& entry : degree_delta) {
            const int row = entry.first;
            const int old_value = degree_[static_cast<std::size_t>(row)];
            const int new_value = old_value + entry.second;
            const int csr_value = graph_new.row_ptr[static_cast<std::size_t>(row + 1)] -
                                  graph_new.row_ptr[static_cast<std::size_t>(row)];
            if (new_value < 0 || new_value != csr_value) {
                throw std::runtime_error("row-mean stale old/new graph or non-effective delta");
            }
            if (new_value == old_value) continue;
            const float inv_new = deterministic_reciprocal(new_value);
            const float old_over_new = new_value == 0 ? 0.0f :
                static_cast<float>(static_cast<float>(old_value) /
                                   static_cast<float>(new_value));
            t.changes.emplace_back(row, old_value, new_value, old_over_new, inv_new);
            t.changed_rows.push_back(row);
            degree_[static_cast<std::size_t>(row)] = new_value;
            reciprocal_[static_cast<std::size_t>(row)] = inv_new;
        }
        t.degree_read_events = 2ULL * t.changed_rows.size();
        t.degree_write_events = t.changed_rows.size();
        t.degree_read_bytes = t.degree_read_events * sizeof(std::int32_t);
        t.degree_write_bytes = t.degree_write_events * sizeof(std::int32_t);
        t.reciprocal_events = static_cast<std::uint64_t>(std::count_if(
            t.changes.begin(), t.changes.end(), [](const Transition::RowChange& change) {
                return change.new_degree > 0;
            }));
        return t;
    }

    static void apply_full_row_mean(std::vector<float>* values,
                                    const std::vector<int>& degree,
                                    int d) {
        if (values == nullptr || d <= 0 ||
            values->size() != degree.size() * static_cast<std::size_t>(d)) {
            throw std::invalid_argument("row-mean full scaling shape mismatch");
        }
        for (std::size_t row = 0; row < degree.size(); ++row) {
            if (degree[row] < 0) throw std::invalid_argument("row-mean negative destination degree");
            if (degree[row] == 0) {
                for (int k = 0; k < d; ++k) {
                    (*values)[row * static_cast<std::size_t>(d) +
                              static_cast<std::size_t>(k)] = 0.0f;
                }
                continue;
            }
            const float inv = 1.0f / static_cast<float>(degree[row]);
            for (int k = 0; k < d; ++k) {
                const std::size_t p = row * static_cast<std::size_t>(d) +
                                      static_cast<std::size_t>(k);
                (*values)[p] = static_cast<float>((*values)[p] * inv);
            }
        }
    }

    static void apply_scoped_recurrence_row(float* state_after_raw_delta,
                                            const float* old_normalized,
                                            int d_old,
                                            int d_new,
                                            float old_over_new,
                                            float inv_new,
                                            int d) {
        if (state_after_raw_delta == nullptr || old_normalized == nullptr ||
            d <= 0 || d_old < 0 || d_new < 0) {
            throw std::invalid_argument("row-mean recurrence arguments invalid");
        }
        // An empty destination row has mean zero.  This also covers deletion
        // of the final edge without inventing an implicit self loop.
        if (d_new == 0) {
            std::fill(state_after_raw_delta, state_after_raw_delta + d, 0.0f);
            return;
        }
        const float expected_ratio = static_cast<float>(
            static_cast<float>(d_old) / static_cast<float>(d_new));
        const float expected_inv = deterministic_reciprocal(d_new);
        if (old_over_new != expected_ratio || inv_new != expected_inv) {
            throw std::runtime_error("row-mean deterministic reciprocal state mismatch");
        }
        for (int k = 0; k < d; ++k) {
            const float old_y = old_normalized[k];
            const float raw_delta = static_cast<float>(state_after_raw_delta[k] - old_y);
            state_after_raw_delta[k] = static_cast<float>(
                static_cast<float>(old_y * old_over_new) +
                static_cast<float>(raw_delta * inv_new));
        }
    }

    static float deterministic_reciprocal(int degree) {
        if (degree < 0) throw std::invalid_argument("row-mean negative degree reciprocal");
        return degree == 0 ? 0.0f :
            static_cast<float>(1.0f / static_cast<float>(degree));
    }

    static std::vector<int> degrees(const CSRMatrix& graph) {
        if (graph.n <= 0 || graph.row_ptr.size() != static_cast<std::size_t>(graph.n + 1) ||
            graph.row_ptr.front() != 0 || graph.row_ptr.back() != graph.nnz) {
            throw std::invalid_argument("row-mean invalid CSR");
        }
        std::vector<int> out(static_cast<std::size_t>(graph.n), 0);
        for (int row = 0; row < graph.n; ++row) {
            const int d = graph.row_ptr[static_cast<std::size_t>(row + 1)] -
                          graph.row_ptr[static_cast<std::size_t>(row)];
            if (d < 0) throw std::invalid_argument("row-mean negative destination degree");
            out[static_cast<std::size_t>(row)] = d;
        }
        return out;
    }

    static void validate_binary_no_self_loop(const CSRMatrix& graph) {
        if (graph.n <= 0 || graph.nnz < 0 ||
            graph.row_ptr.size() != static_cast<std::size_t>(graph.n + 1) ||
            graph.col_idx.size() != static_cast<std::size_t>(graph.nnz) ||
            (!graph.values.empty() && graph.values.size() != static_cast<std::size_t>(graph.nnz)) ||
            graph.row_ptr.front() != 0 || graph.row_ptr.back() != graph.nnz) {
            throw std::invalid_argument("row-mean invalid CSR shape");
        }
        for (int row = 0; row < graph.n; ++row) {
            int previous = -1;
            for (int p = graph.row_ptr[static_cast<std::size_t>(row)];
                 p < graph.row_ptr[static_cast<std::size_t>(row + 1)]; ++p) {
                const int col = graph.col_idx[static_cast<std::size_t>(p)];
                const float value = graph.values.empty() ? 1.0f :
                    graph.values[static_cast<std::size_t>(p)];
                if (col < 0 || col >= graph.n || col <= previous || col == row ||
                    !std::isfinite(value) || value != 1.0f) {
                    throw std::invalid_argument(
                        "row-mean requires sorted unique binary CSR without self loops");
                }
                previous = col;
            }
        }
        (void)degrees(graph);
    }

private:
    int n_ = 0;
    std::vector<int> degree_;
    std::vector<float> reciprocal_;
};

#endif
