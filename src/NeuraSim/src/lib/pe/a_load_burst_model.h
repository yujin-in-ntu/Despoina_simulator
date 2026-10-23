// SPDX-License-Identifier: Apache-2.0
#ifndef A_LOAD_BURST_MODEL_H
#define A_LOAD_BURST_MODEL_H

#include <cstdint>
#include <stdexcept>

struct ALoadBurstSpan {
    std::uint64_t begin = 0;
    std::uint64_t end = 0;
    std::uint64_t payload_bytes = 0;

    std::uint64_t bursts() const { return (end - begin) / 64ULL; }
};

inline ALoadBurstSpan make_a_load_burst_span(std::uint64_t base,
                                             int csr_start_ptr,
                                             int nnz) {
    if (csr_start_ptr < 0 || nnz < 0) {
        throw std::invalid_argument("negative CSR A-load span");
    }
    ALoadBurstSpan span;
    span.payload_bytes = static_cast<std::uint64_t>(nnz) * 8ULL;
    const std::uint64_t raw_begin = base +
        static_cast<std::uint64_t>(csr_start_ptr) * 8ULL;
    span.begin = raw_begin & ~std::uint64_t{63};
    if (span.payload_bytes == 0) {
        span.end = span.begin;
        return span;
    }
    const std::uint64_t raw_end = raw_begin + span.payload_bytes;
    span.end = (raw_end + 63ULL) & ~std::uint64_t{63};
    return span;
}

#endif
