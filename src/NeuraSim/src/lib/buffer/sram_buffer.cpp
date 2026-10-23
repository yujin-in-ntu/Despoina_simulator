// SPDX-License-Identifier: Apache-2.0
/**
 * sram_buffer.cpp
 * 
 * Implementation of SRAM Buffer for DyGNN Accelerator.
 */

#include "sram_buffer.h"
#include <iostream>
#include <cstring>
#include <algorithm>

SRAMBuffer::SRAMBuffer() 
    : capacity_bytes(0)
    , capacity_floats(0)
    , double_buffer_enabled(false)
    , state_a(BufferState::EMPTY)
    , state_b(BufferState::EMPTY)
    , active_write_buffer(0)
    , active_read_buffer(0)
    , rows_written(0)
    , rows_read(0)
    , current_feature_dim(0)
    , current_tile_start(0)
    , current_tile_rows(0)
    , write_stall_cycles(0)
    , read_stall_cycles(0)
    , bank_conflict_stalls(0)
    , read_bank_conflict_stalls(0)
    , write_bank_conflict_stalls(0)
    , total_bytes_written(0)
    , total_bytes_read(0)
    , logical_cycle(0)
{
    bank_next_free_cycle.fill(0);
}

SRAMBuffer::SRAMBuffer(int capacity_kb, bool double_buffer_mode)
    : SRAMBuffer()
{
    init(capacity_kb, double_buffer_mode);
}

SRAMBuffer::~SRAMBuffer() {}

void SRAMBuffer::init(int capacity_kb, bool double_buffer_mode) {
    capacity_bytes = capacity_kb * 1024;
    capacity_floats = capacity_bytes / sizeof(float);
    double_buffer_enabled = double_buffer_mode;
    
    if (double_buffer_enabled) {
        // Split capacity between two buffers
        int half_floats = capacity_floats / 2;
        buffer_a.resize(half_floats);
        buffer_b.resize(half_floats);
        capacity_floats = half_floats;  // Per-buffer capacity
    } else {
        buffer_a.resize(capacity_floats);
    }
    
    reset();
}

void SRAMBuffer::reset() {
    state_a = BufferState::EMPTY;
    state_b = BufferState::EMPTY;
    active_write_buffer = 0;
    active_read_buffer = 0;
    rows_written = 0;
    rows_read = 0;
    current_feature_dim = 0;
    current_tile_start = 0;
    current_tile_rows = 0;
    write_stall_cycles = 0;
    read_stall_cycles = 0;
    bank_conflict_stalls = 0;
    read_bank_conflict_stalls = 0;
    write_bank_conflict_stalls = 0;
    total_bytes_written = 0;
    total_bytes_read = 0;
    logical_cycle = 0;
    bank_next_free_cycle.fill(0);
    
    while (!pending_tiles.empty()) {
        pending_tiles.pop();
    }
}

// =========================================================================
// Write Interface
// =========================================================================

bool SRAMBuffer::can_write() const {
    if (double_buffer_enabled) {
        // Can write if write buffer is not READY (being consumed)
        BufferState ws = (active_write_buffer == 0) ? state_a : state_b;
        return ws != BufferState::READY && ws != BufferState::DRAINING;
    } else {
        // Single buffer: can write if not full
        return rows_written * current_feature_dim < capacity_floats;
    }
}

int SRAMBuffer::write_row(int row_id, const float* data, int dim) {
    int cycles = WRITE_LATENCY;
    const int bank_stall = reserve_bank(row_id);
    cycles += bank_stall;
    if (bank_stall > 0) {
        bank_conflict_stalls += static_cast<uint64_t>(bank_stall);
        write_bank_conflict_stalls += static_cast<uint64_t>(bank_stall);
    }
    
    // Set feature dimension on first write
    if (current_feature_dim == 0) {
        current_feature_dim = dim;
        current_tile_start = row_id;
    }
    
    // Check if buffer is full
    int required_floats = (rows_written + 1) * dim;
    if (required_floats > capacity_floats) {
        // Buffer full - need to stall or complete tile
        if (double_buffer_enabled) {
            // In double buffer mode, swap and continue
            swap_buffers();
            rows_written = 0;
            current_tile_start = row_id;
        } else {
            // Single buffer - stall until data is consumed
            write_stall_cycles++;
            logical_cycle += static_cast<uint64_t>(cycles + 1);
            return cycles + 1;  // Extra stall cycle
        }
    }
    
    // Write data to buffer
    std::vector<float>& buf = get_write_buffer();
    int offset = rows_written * dim;
    std::memcpy(buf.data() + offset, data, dim * sizeof(float));
    
    rows_written++;
    current_tile_rows++;
    total_bytes_written += dim * sizeof(float);
    
    // Update state
    BufferState& ws = get_write_state();
    if (ws == BufferState::EMPTY) {
        ws = BufferState::FILLING;
    }
    
    logical_cycle += static_cast<uint64_t>(cycles);
    return cycles;
}

void SRAMBuffer::complete_write_tile() {
    if (rows_written > 0) {
        // Create tile descriptor
        BufferTile tile(current_tile_start, current_tile_rows, current_feature_dim);
        pending_tiles.push(tile);
        
        // Mark buffer as ready
        BufferState& ws = get_write_state();
        ws = BufferState::READY;
        
        // Reset tile tracking
        current_tile_rows = 0;
    }
}

// =========================================================================
// Read Interface
// =========================================================================

bool SRAMBuffer::can_read() const {
    if (double_buffer_enabled) {
        BufferState rs = (active_read_buffer == 0) ? state_a : state_b;
        return rs == BufferState::READY || rs == BufferState::DRAINING;
    } else {
        return rows_written > rows_read || !pending_tiles.empty();
    }
}

int SRAMBuffer::read_row(int row_id, float* data, int dim) {
    int cycles = READ_LATENCY;
    const int bank_stall = reserve_bank(row_id);
    cycles += bank_stall;
    if (bank_stall > 0) {
        bank_conflict_stalls += static_cast<uint64_t>(bank_stall);
        read_bank_conflict_stalls += static_cast<uint64_t>(bank_stall);
    }
    
    if (!can_read()) {
        read_stall_cycles++;
        logical_cycle += static_cast<uint64_t>(cycles + 1);
        return cycles + 1;  // Stall
    }
    
    // Read data from buffer
    std::vector<float>& buf = get_read_buffer();
    int offset = rows_read * dim;
    std::memcpy(data, buf.data() + offset, dim * sizeof(float));
    
    rows_read++;
    total_bytes_read += dim * sizeof(float);
    
    // Update state
    BufferState& rs = get_read_state();
    if (rs == BufferState::READY) {
        rs = BufferState::DRAINING;
    }
    
    logical_cycle += static_cast<uint64_t>(cycles);
    return cycles;
}

bool SRAMBuffer::get_next_read_tile(BufferTile* tile) {
    if (pending_tiles.empty()) {
        return false;
    }
    
    *tile = pending_tiles.front();
    return true;
}

void SRAMBuffer::complete_read_tile() {
    if (!pending_tiles.empty()) {
        pending_tiles.pop();
        
        // Mark buffer as empty if no more tiles
        if (pending_tiles.empty()) {
            BufferState& rs = get_read_state();
            rs = BufferState::EMPTY;
            rows_read = 0;
        }
    }
}

// =========================================================================
// Double Buffering Control
// =========================================================================

void SRAMBuffer::swap_buffers() {
    if (!double_buffer_enabled) return;
    
    // Swap write buffer
    active_write_buffer = 1 - active_write_buffer;
    
    // If read buffer is empty, swap it too
    BufferState& rs = get_read_state();
    if (rs == BufferState::EMPTY) {
        active_read_buffer = 1 - active_read_buffer;
    }
}

bool SRAMBuffer::should_swap() const {
    if (!double_buffer_enabled) return false;
    
    // Should swap if write buffer is READY and read buffer is EMPTY
    BufferState ws = (active_write_buffer == 0) ? state_a : state_b;
    BufferState rs = (active_read_buffer == 0) ? state_a : state_b;
    
    return ws == BufferState::READY && rs == BufferState::EMPTY;
}

// =========================================================================
// Statistics
// =========================================================================

float SRAMBuffer::get_utilization() const {
    if (capacity_floats == 0 || current_feature_dim == 0) return 0.0f;
    return static_cast<float>(rows_written * current_feature_dim) / capacity_floats;
}

BufferState SRAMBuffer::get_write_buffer_state() const {
    return (active_write_buffer == 0) ? state_a : state_b;
}

BufferState SRAMBuffer::get_read_buffer_state() const {
    return (active_read_buffer == 0) ? state_a : state_b;
}

// =========================================================================
// Helper Functions
// =========================================================================

std::vector<float>& SRAMBuffer::get_write_buffer() {
    return (active_write_buffer == 0) ? buffer_a : buffer_b;
}

std::vector<float>& SRAMBuffer::get_read_buffer() {
    return (active_read_buffer == 0) ? buffer_a : buffer_b;
}

BufferState& SRAMBuffer::get_write_state() {
    return (active_write_buffer == 0) ? state_a : state_b;
}

BufferState& SRAMBuffer::get_read_state() {
    return (active_read_buffer == 0) ? state_a : state_b;
}

int SRAMBuffer::reserve_bank(int row_id) {
    int bank = row_id % BANK_COUNT;
    if (bank < 0) {
        bank += BANK_COUNT;
    }

    const uint64_t next_free = bank_next_free_cycle[bank];
    int stall_cycles = 0;
    if (next_free > logical_cycle) {
        stall_cycles = static_cast<int>(next_free - logical_cycle);
    }

    const uint64_t start_cycle = logical_cycle + static_cast<uint64_t>(stall_cycles);
    bank_next_free_cycle[bank] = start_cycle + 1;
    return stall_cycles;
}
