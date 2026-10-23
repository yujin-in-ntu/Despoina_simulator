// SPDX-License-Identifier: Apache-2.0
/**
 * sram_buffer.h
 * 
 * SRAM Buffer for stage-to-stage data transfer in DyGNN Accelerator.
 * 
 * Primary use: H_agg Buffer (128KB)
 * - Stores SpMM output (Aggregation result)
 * - Feeds CiM input (Combination step)
 * 
 * Features:
 * - Configurable capacity
 * - Double buffering mode for compute/memory overlap
 * - Stall modeling when buffer is full/empty
 * - Tiled access for large matrices
 */

#ifndef SRAM_BUFFER_H
#define SRAM_BUFFER_H

#include <vector>
#include <cstdint>
#include <queue>
#include <array>

/**
 * Buffer state for double buffering
 */
enum class BufferState {
    EMPTY,      // No data
    FILLING,    // Being written to
    READY,      // Data ready for consumption
    DRAINING    // Being read from
};

/**
 * Tile descriptor for tiled buffer access
 */
struct BufferTile {
    int start_row;      // First row in tile
    int num_rows;       // Number of rows in tile
    int feature_dim;    // Feature dimension
    bool valid;         // Tile contains valid data
    
    BufferTile() : start_row(0), num_rows(0), feature_dim(0), valid(false) {}
    BufferTile(int sr, int nr, int fd) 
        : start_row(sr), num_rows(nr), feature_dim(fd), valid(true) {}
};

/**
 * SRAM Buffer with optional double buffering
 */
class SRAMBuffer {
public:
    // Configuration
    static constexpr int DEFAULT_CAPACITY_KB = 128;  // 128KB default
    static constexpr int WRITE_LATENCY = 1;          // 1 cycle per write
    static constexpr int READ_LATENCY = 1;           // 1 cycle per read
    static constexpr int BANK_COUNT = 8;             // 8 banks for parallel access
    
    // Constructor
    SRAMBuffer();
    SRAMBuffer(int capacity_kb, bool double_buffer_mode = false);
    ~SRAMBuffer();
    
    // Initialize buffer
    void init(int capacity_kb, bool double_buffer_mode = false);
    
    // Reset buffer state
    void reset();
    
    // =========================================================================
    // Write Interface (from SpMM/PE Array)
    // =========================================================================
    
    // Check if buffer can accept more data
    bool can_write() const;
    
    // Write a row of data to buffer
    // Returns cycles taken (including potential stall)
    int write_row(int row_id, const float* data, int dim);
    
    // Notify that a tile is complete (for tiled mode)
    void complete_write_tile();
    
    // =========================================================================
    // Read Interface (to CiM)
    // =========================================================================
    
    // Check if buffer has data to read
    bool can_read() const;
    
    // Read a row of data from buffer
    // Returns cycles taken
    int read_row(int row_id, float* data, int dim);
    
    // Get next tile ready for reading
    bool get_next_read_tile(BufferTile* tile);
    
    // Notify that a tile has been consumed
    void complete_read_tile();
    
    // =========================================================================
    // Double Buffering Control
    // =========================================================================
    
    // Swap buffers (for double buffering mode)
    void swap_buffers();
    
    // Check if swap is needed/possible
    bool should_swap() const;
    
    // =========================================================================
    // Statistics and State
    // =========================================================================
    
    // Get buffer utilization (0.0 - 1.0)
    float get_utilization() const;
    
    // Get total stall cycles
    uint64_t get_write_stall_cycles() const { return write_stall_cycles; }
    uint64_t get_read_stall_cycles() const { return read_stall_cycles; }
    uint64_t get_total_stall_cycles() const { return write_stall_cycles + read_stall_cycles; }
    uint64_t get_bank_conflict_stalls() const { return bank_conflict_stalls; }
    uint64_t get_read_bank_conflict_stalls() const { return read_bank_conflict_stalls; }
    uint64_t get_write_bank_conflict_stalls() const { return write_bank_conflict_stalls; }
    
    // Get buffer capacity info
    int get_capacity_bytes() const { return capacity_bytes; }
    int get_capacity_floats() const { return capacity_floats; }
    int get_max_rows(int feature_dim) const { return capacity_floats / feature_dim; }
    
    // Tile info
    int get_rows_written() const { return rows_written; }
    int get_rows_read() const { return rows_read; }
    
    // State queries
    bool is_double_buffer_mode() const { return double_buffer_enabled; }
    BufferState get_write_buffer_state() const;
    BufferState get_read_buffer_state() const;

private:
    // Configuration
    int capacity_bytes;
    int capacity_floats;
    bool double_buffer_enabled;
    
    // Double buffer storage
    std::vector<float> buffer_a;
    std::vector<float> buffer_b;
    
    // Buffer states
    BufferState state_a;
    BufferState state_b;
    int active_write_buffer;  // 0 = A, 1 = B
    int active_read_buffer;   // 0 = A, 1 = B
    
    // Current tile tracking
    int rows_written;
    int rows_read;
    int current_feature_dim;
    int current_tile_start;
    int current_tile_rows;
    
    // Tile queue for tiled access
    std::queue<BufferTile> pending_tiles;
    
    // Statistics
    uint64_t write_stall_cycles;
    uint64_t read_stall_cycles;
    uint64_t bank_conflict_stalls;
    uint64_t read_bank_conflict_stalls;
    uint64_t write_bank_conflict_stalls;
    uint64_t total_bytes_written;
    uint64_t total_bytes_read;
    uint64_t logical_cycle;
    std::array<uint64_t, BANK_COUNT> bank_next_free_cycle;
    
    // Helper functions
    std::vector<float>& get_write_buffer();
    std::vector<float>& get_read_buffer();
    BufferState& get_write_state();
    BufferState& get_read_state();
    int reserve_bank(int row_id);
};

#endif // SRAM_BUFFER_H
