// SPDX-License-Identifier: Apache-2.0
#ifndef L1_P3_SUPPORT_H
#define L1_P3_SUPPORT_H

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <vector>

#include "delta_converter.h"

// L1 P3 is -DeltaA * DeltaF.  Its DeltaA columns are therefore restricted to
// the *effective* (canonical, nonzero) DeltaF source support.  This helper is
// intentionally separate from the indexer: a CSR row is a destination/output
// row while its column is the source feature row.
namespace l1_p3 {

struct CanonicalDeltaF {
    DeltaConverter::DeltaFParsed payload;
    std::vector<int> nonzero_support;
    uint64_t payload_hash = 1469598103934665603ULL;
};

inline uint64_t fnv1a64_update(uint64_t h, const void* ptr, size_t bytes) {
    const unsigned char* p = static_cast<const unsigned char*>(ptr);
    for (size_t i = 0; i < bytes; ++i) {
        h ^= static_cast<uint64_t>(p[i]);
        h *= 1099511628211ULL;
    }
    return h;
}

inline CanonicalDeltaF canonicalize_delta_f(
    const DeltaConverter::DeltaFParsed& raw, int n, int d) {
    if (raw.feature_dim != d) {
        throw std::invalid_argument("L1 P3 DeltaF dimension mismatch");
    }
    CanonicalDeltaF out;
    out.payload = DeltaConverter::coalesce_delta_f_updates(raw);
    for (size_t i = 0; i < out.payload.nodes.size(); ++i) {
        const int node = out.payload.nodes[i];
        if (node < 0 || node >= n) {
            continue;  // matches dense DeltaF's valid-node domain.
        }
        bool nonzero = false;
        const float* row = out.payload.deltas.data() +
            i * static_cast<size_t>(d);
        for (int j = 0; j < d; ++j) {
            if (!std::isfinite(row[j])) {
                throw std::invalid_argument("L1 P3 DeltaF contains non-finite value");
            }
            if (row[j] != 0.0f) nonzero = true;
        }
        if (nonzero) out.nonzero_support.push_back(node);
    }
    // Hash the canonical immutable payload, including zero rows.  P2 and P3
    // must use this identical payload and matrix epoch; support is only the
    // sparse traversal filter, never a rewritten feature payload.
    const int feature_dim = out.payload.feature_dim;
    out.payload_hash = fnv1a64_update(out.payload_hash, &feature_dim,
                                      sizeof(feature_dim));
    for (size_t i = 0; i < out.payload.nodes.size(); ++i) {
        out.payload_hash = fnv1a64_update(out.payload_hash,
                                          &out.payload.nodes[i],
                                          sizeof(out.payload.nodes[i]));
        out.payload_hash = fnv1a64_update(
            out.payload_hash,
            out.payload.deltas.data() + i * static_cast<size_t>(d),
            static_cast<size_t>(d) * sizeof(float));
    }
    return out;
}

inline WorkStealingScheduler::CSRMatrix filter_csr_columns(
    const WorkStealingScheduler::CSRMatrix& input,
    const std::vector<int>& source_support) {
    WorkStealingScheduler::CSRMatrix out;
    out.n = input.n;
    out.row_ptr.assign(static_cast<size_t>(input.n) + 1, 0);
    std::vector<unsigned char> selected(static_cast<size_t>(input.n), 0);
    for (size_t i = 0; i < source_support.size(); ++i) {
        const int col = source_support[i];
        if (col >= 0 && col < input.n) selected[static_cast<size_t>(col)] = 1;
    }
    for (int row = 0; row < input.n; ++row) {
        out.row_ptr[static_cast<size_t>(row)] = out.nnz;
        const int begin = input.row_ptr[static_cast<size_t>(row)];
        const int end = input.row_ptr[static_cast<size_t>(row + 1)];
        if (begin < 0 || end < begin ||
            end > static_cast<int>(input.col_idx.size()) ||
            end > static_cast<int>(input.values.size())) {
            throw std::invalid_argument("invalid CSR while filtering L1 P3");
        }
        for (int p = begin; p < end; ++p) {
            const int col = input.col_idx[static_cast<size_t>(p)];
            if (col >= 0 && col < input.n && selected[static_cast<size_t>(col)]) {
                ++out.nnz;
            }
        }
    }
    out.row_ptr[static_cast<size_t>(input.n)] = out.nnz;
    out.col_idx.reserve(static_cast<size_t>(out.nnz));
    out.values.reserve(static_cast<size_t>(out.nnz));
    for (int row = 0; row < input.n; ++row) {
        const int begin = input.row_ptr[static_cast<size_t>(row)];
        const int end = input.row_ptr[static_cast<size_t>(row + 1)];
        for (int p = begin; p < end; ++p) {
            const int col = input.col_idx[static_cast<size_t>(p)];
            if (col >= 0 && col < input.n && selected[static_cast<size_t>(col)]) {
                out.col_idx.push_back(col);
                out.values.push_back(input.values[static_cast<size_t>(p)]);
            }
        }
    }
    return out;
}

}  // namespace l1_p3

#endif  // L1_P3_SUPPORT_H
