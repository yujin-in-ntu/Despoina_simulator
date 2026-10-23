// SPDX-License-Identifier: Apache-2.0
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "../src/lib/pe/delta_f_aware_indexer.h"
#include "../src/lib/pe/dynamic_row_mean_operator.h"

namespace {
using CSR = WorkStealingScheduler::CSRMatrix;
using Update = DeltaConverter::DeltaAUpdate;

CSR csr(const std::vector<std::vector<int>>& rows) {
    CSR x;
    x.n = static_cast<int>(rows.size());
    x.row_ptr.assign(rows.size() + 1, 0);
    for (std::size_t r = 0; r < rows.size(); ++r) {
        x.row_ptr[r] = static_cast<int>(x.col_idx.size());
        for (int c : rows[r]) {
            x.col_idx.push_back(c);
            x.values.push_back(1.0f);
        }
    }
    x.row_ptr[rows.size()] = static_cast<int>(x.col_idx.size());
    x.nnz = static_cast<int>(x.col_idx.size());
    return x;
}

void require(bool ok, const char* message) {
    if (!ok) throw std::runtime_error(message);
}

bool close(float a, float b, float tol = 1e-6f) {
    return std::fabs(a - b) <= tol;
}
}

int main() {
    try {
        // No implicit self loops. Every row nevertheless has positive degree.
        const CSR old_graph = csr({{1, 2}, {0}, {0}, {0}});
        const CSR inserted = csr({{1, 2, 3}, {0}, {0}, {0}});
        DynamicRowMeanOperator op;
        op.reset(old_graph);
        Update add;
        add.src = 0; add.dst = 3; add.w = 1.0f; add.is_add = true;
        const auto ins = op.prepare(inserted, {add});
        require(ins.changes.size() == 1 && ins.changes[0].old_degree == 2 &&
                    ins.changes[0].new_degree == 3,
                "insertion degree transition mismatch");
        require(ins.changed_rows == std::vector<int>({0}),
                "insertion changed-row mismatch");

        // H_old=[-, 2, 4, 8], scoped dH at node 1 is +3.  For row 0:
        // P1=dB*H_new=8, P2=B_new*dH=3, P3=dB*dH=0.
        const float old_y = 3.0f;  // (2+4)/2
        const float p1 = 8.0f;
        const float p2 = 3.0f;
        const float p3 = 0.0f;
        float exact_state = old_y + p1 + p2 - p3;
        DynamicRowMeanOperator::apply_scoped_recurrence_row(
            &exact_state, &old_y, 2, 3, ins.changes[0].old_over_new,
            ins.changes[0].inv_new, 1);
        const float fresh = (5.0f + 4.0f + 8.0f) / 3.0f;
        require(close(exact_state, fresh),
                "P1+P2-P3 scoped insertion recurrence mismatch");

        // Counterexample requested by the audit: simply scaling the raw
        // accumulator (old normalized Y + raw phase delta) by 1/d_new loses
        // the old numerator's d_old ownership.
        const float naive = (old_y + p1 + p2 - p3) / 3.0f;
        require(std::fabs(naive - fresh) > 0.5f,
                "naive post-row scaling counterexample did not separate");

        // Tau-empty still executes structural P1.  Here P2=P3=0 and the
        // recurrence must equal the new fresh row mean.
        float tau_empty = old_y + p1;
        DynamicRowMeanOperator::apply_scoped_recurrence_row(
            &tau_empty, &old_y, 2, 3, ins.changes[0].old_over_new,
            ins.changes[0].inv_new, 1);
        require(close(tau_empty, (2.0f + 4.0f + 8.0f) / 3.0f),
                "tau-empty P1 recurrence mismatch");

        Update del = add;
        del.is_add = false;
        const auto deletion = op.prepare(old_graph, {del});
        require(deletion.changes.size() == 1 && deletion.changes[0].old_degree == 3 &&
                    deletion.changes[0].new_degree == 2,
                "deletion degree transition mismatch");

        // No implicit self loop means isolated rows are legal and aggregate
        // to zero.  Exercise both the first insertion and final deletion.
        const CSR isolated = csr({{}, {0}});
        const CSR one_edge = csr({{1}, {0}});
        DynamicRowMeanOperator zero_degree;
        zero_degree.reset(isolated);
        Update first; first.src = 0; first.dst = 1; first.w = 1.0f; first.is_add = true;
        const auto first_transition = zero_degree.prepare(one_edge, {first});
        float first_state = 7.0f;
        const float old_zero = 0.0f;
        DynamicRowMeanOperator::apply_scoped_recurrence_row(
            &first_state, &old_zero, 0, 1, first_transition.changes[0].old_over_new,
            first_transition.changes[0].inv_new, 1);
        require(close(first_state, 7.0f), "zero-to-one degree insertion mismatch");
        first.is_add = false;
        const auto final_transition = zero_degree.prepare(isolated, {first});
        float deleted_state = 7.0f - 7.0f;
        const float old_seven = 7.0f;
        DynamicRowMeanOperator::apply_scoped_recurrence_row(
            &deleted_state, &old_seven, 1, 0, final_transition.changes[0].old_over_new,
            final_transition.changes[0].inv_new, 1);
        require(close(deleted_state, 0.0f), "one-to-zero degree deletion mismatch");
        require(first_transition.changes[0].old_degree == 0 &&
                    first_transition.changes[0].new_degree == 1,
                "zero-to-one degree metadata mismatch");
        require(final_transition.changes[0].old_degree == 1 &&
                    final_transition.changes[0].new_degree == 0,
                "one-to-zero degree metadata mismatch");

        // Multiple simultaneous updates on distinct destination rows.
        const CSR simultaneous = csr({{2}, {0, 2}, {0}, {0}});
        DynamicRowMeanOperator op2;
        op2.reset(old_graph);
        Update del01; del01.src = 0; del01.dst = 1; del01.w = 1.0f; del01.is_add = false;
        Update add12; add12.src = 1; add12.dst = 2; add12.w = 1.0f; add12.is_add = true;
        const auto multi = op2.prepare(simultaneous, {del01, add12});
        require(multi.changed_rows == std::vector<int>({0, 1}),
                "simultaneous changed-row mismatch");

        // Sparse topology maintenance has one owner.  Logical payload is one
        // 8-byte edge record per raw delta; cacheline-rounded write traffic is
        // serialized separately, and update latency is a subset of the total
        // IndexStore latency.  A per-step reset prevents cross-timestep carry.
        DeltaFAwareIndexer indexer;
        indexer.build_from_csr(old_graph);
        indexer.reset_counters();
        indexer.apply_delta_a({del01, add12});
        const auto first_owner = indexer.get_counters();
        require(first_owner.index_update_ops == 2,
                "IndexStore update operation ownership mismatch");
        require(first_owner.index_update_payload_bytes == 16,
                "IndexStore update payload byte units mismatch");
        require(first_owner.index_bytes_write == 128,
                "IndexStore cacheline-rounded write byte units mismatch");
        require(first_owner.index_update_dram_cycles == 1 &&
                    first_owner.index_dram_cycles == 1,
                "IndexStore update latency subset mismatch");
        indexer.reset_counters();
        const auto reset_owner = indexer.get_counters();
        require(reset_owner.index_update_ops == 0 &&
                    reset_owner.index_update_payload_bytes == 0 &&
                    reset_owner.index_update_dram_cycles == 0 &&
                    reset_owner.index_dram_cycles == 0,
                "IndexStore per-step owner reset mismatch");
        Update add03; add03.src = 0; add03.dst = 3; add03.w = 1.0f; add03.is_add = true;
        indexer.apply_delta_a({add03});
        const auto next_owner = indexer.get_counters();
        require(next_owner.index_update_ops == 1 &&
                    next_owner.index_update_payload_bytes == 8 &&
                    next_owner.index_update_dram_cycles == 1 &&
                    next_owner.index_dram_cycles == 1,
                "IndexStore exactly-once next-step ownership mismatch");

        bool self_loop_rejected = false;
        try {
            DynamicRowMeanOperator bad;
            bad.reset(csr({{0}, {0}}));
        } catch (const std::invalid_argument&) {
            self_loop_rejected = true;
        }
        require(self_loop_rejected, "self-loop convention did not fail closed");

        bool stale_rejected = false;
        try {
            DynamicRowMeanOperator stale;
            stale.reset(old_graph);
            (void)stale.prepare(old_graph, {add});
        } catch (const std::runtime_error&) {
            stale_rejected = true;
        }
        require(stale_rejected, "stale/no-op delta did not fail closed");

        std::cout << "DYNAMIC_ROW_MEAN_OPERATOR_UNIT_PASS\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "DYNAMIC_ROW_MEAN_OPERATOR_UNIT_FAIL: " << e.what() << "\n";
        return 1;
    }
}
