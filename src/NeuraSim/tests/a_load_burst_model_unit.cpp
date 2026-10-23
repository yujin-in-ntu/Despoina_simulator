// SPDX-License-Identifier: Apache-2.0
#include <iostream>
#include <stdexcept>

#include "../src/lib/pe/a_load_burst_model.h"

void require(bool ok, const char* message) {
    if (!ok) throw std::runtime_error(message);
}

int main() {
    try {
        constexpr std::uint64_t base = 0x30000000ULL;
        const auto empty = make_a_load_burst_span(base, 0, 0);
        require(empty.payload_bytes == 0 && empty.bursts() == 0,
                "empty row issued a burst");
        const auto one = make_a_load_burst_span(base, 0, 1);
        require(one.payload_bytes == 8 && one.bursts() == 1,
                "one record must occupy one line");
        const auto aligned = make_a_load_burst_span(base, 0, 8);
        require(aligned.payload_bytes == 64 && aligned.bursts() == 1,
                "aligned 64B payload mismatch");
        const auto crossing = make_a_load_burst_span(base, 7, 2);
        require(crossing.payload_bytes == 16 && crossing.bursts() == 2,
                "unaligned boundary crossing mismatch");
        const auto long_row = make_a_load_burst_span(base, 3, 20);
        require(long_row.payload_bytes == 160 && long_row.bursts() == 3,
                "multi-line row mismatch");
        std::cout << "A_LOAD_BURST_MODEL_UNIT_PASS\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "A_LOAD_BURST_MODEL_UNIT_FAIL: " << error.what() << "\n";
        return 1;
    }
}
