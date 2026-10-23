// SPDX-License-Identifier: Apache-2.0
#ifndef PE_H
#define PE_H

#include <iostream>
#include <vector>
#include <cstring>

// Forward declaration - GlobalCounters is optional for standalone tests
class GlobalCounters;

/**
 * PE (Processing Element) for PE Array-based SpMM
 * 
 * Based on DyGNN Accelerator design from config.py:
 * - 8-wide SIMD FMA unit with 2-stage pipeline
 * - 1 MAC per cycle per lane (pipelined)
 * - Registers: a_reg(4), f_reg(2048), psum_reg(2048), col_idx_fifo(16)
 * 
 * 2-Stage Pipeline Architecture:
 *   Stage 1 (MUL): product = a_val * f_row[i]
 *   Stage 2 (ACC): psum[i] += product
 * 
 * Benefits:
 * - Higher clock frequency (shorter critical path)
 * - Overlapped execution for back-to-back operations
 * - 1 cycle initiation interval, 2 cycle latency
 * 
 * Dataflow:
 * 1. Load A values into a_reg
 * 2. For each A value, load corresponding F row
 * 3. Perform pipelined SIMD MAC: psum += a_val * f_row
 * 4. Output accumulated psum when row is complete
 */
class PE {
public:
    // Configuration constants (from config.py PEArrayConfig)
    // 8 MACs/PE with 2-stage pipeline for higher frequency
    static constexpr int SIMD_WIDTH = 8;       // 8 parallel lanes
    static constexpr int MAC_PER_CYCLE = 1;    // 1 MAC per lane per cycle
    static constexpr int PIPELINE_STAGES = 2;  // 2-stage: MUL -> ACC
    static constexpr int A_REG_SIZE = 4;
    static constexpr int F_REG_SIZE = 128;      // Matches JODIE dataset feature_dim
    static constexpr int PSUM_REG_SIZE = 128;   // Matches F_REG_SIZE
    static constexpr int COL_IDX_FIFO_DEPTH = 16;

    // Constructor
    PE();
    PE(GlobalCounters* gcp, int pe_id, int cluster_id);
    ~PE();
    
    void init(GlobalCounters* gcp, int pe_id, int cluster_id);
    
    // Main simulation tick - returns true if work was done
    bool tick();
    
    // Load A values for current row segment
    // @param vals: Array of A matrix non-zero values
    // @param col_indices: Corresponding column indices
    // @param count: Number of values to load
    void load_a_values(const float* vals, const int* col_indices, int count);
    
    // Load F row into f_reg (called by PECluster for shared F)
    // @param row_data: Pointer to F row data
    // @param dim: Feature dimension
    void load_f_row(const float* row_data, int dim);
    
    // Check if F row is already in local cache
    bool has_f_row(int col_idx) const;
    
    // Set output row ID for accumulation
    void set_output_row(int row_id);
    
    // Get accumulated partial sums
    const float* get_psum() const { return psum_reg; }
    
    // Clear partial sums for new row
    void clear_psum();
    
    // State queries
    bool is_busy() const { return busy; }
    bool is_done() const { return done; }
    bool needs_f_row() const { return waiting_for_f; }
    int get_needed_f_col() const { return needed_f_col; }
    
    // Statistics
    long long get_mac_count() const { return total_mac_ops; }
    long long get_busy_cycles() const { return busy_cycles; }
    long long get_idle_cycles() const { return idle_cycles; }
    long long get_pipeline_stalls() const { return pipeline_stalls; }
    long long get_wait_f_stalls() const { return wait_f_stalls; }
    long long get_no_a_stalls() const { return no_a_stalls; }

    // Reset the complete per-SpMM execution state.  The feature register may
    // only survive when the caller proves that the new job uses the exact same
    // feature-payload epoch (Phase-2 -> Phase-3 reuse).
    void reset_for_new_spmm(bool keep_feature_register = false);

    // Reset per-job counters (does not touch functional state).
    void reset_stats();
    
    // Pipeline info (for debugging/analysis)
    int get_pipeline_stages() const { return PIPELINE_STAGES; }
    bool is_pipeline_active() const { return pipe_mul.valid || pipe_acc.valid; }
    
    // PE identification
    int get_pe_id() const { return pe_id; }
    int get_cluster_id() const { return cluster_id; }

private:
    GlobalCounters* gcp;
    int pe_id;
    int cluster_id;
    
    // Registers (small, stack-allocated)
    // A operand staging registers (loaded by cluster in batches, then enqueued gradually)
    float a_reg[A_REG_SIZE];               // A values (staging)
    int a_col_idx[A_REG_SIZE];             // Column indices (staging)
    int a_reg_count;                       // Number of valid staging entries
    int a_reg_enqueue_ptr;                 // Next staging entry to enqueue into FIFO (1/cycle)
    
    // Large buffers (heap-allocated to avoid stack overflow)
    float* f_reg;                          // F row cache [F_REG_SIZE]
    int f_reg_col_idx;                     // Which F row is loaded
    int f_reg_dim;                         // Dimension of loaded F row
    
    float* psum_reg;                       // Partial sums accumulator [PSUM_REG_SIZE]
    int output_row_id;                     // Which output row we're computing
    
    // A operand FIFO (decouple fetch/issue from compute)
    // One enqueue per cycle, one dequeue per operand-start.
    float a_val_fifo[COL_IDX_FIFO_DEPTH];
    int col_idx_fifo[COL_IDX_FIFO_DEPTH];
    int fifo_head;
    int fifo_tail;
    int fifo_count;
    
    // State machine / active operand
    bool busy;
    bool done;
    bool waiting_for_f;
    int needed_f_col;
    int remaining_cycles;

    bool active_operand_valid;
    float active_a_val;
    int active_col_idx;
    
    // 2-Stage Pipeline State
    // Stage 1: MUL - computes product = a_val * f_row[chunk]
    // Stage 2: ACC - accumulates psum[chunk] += product
    struct PipelineStage {
        bool valid;              // Stage has valid data
        float a_val;             // A value being processed
        int chunk_start;         // Start index in f_row
        int chunk_end;           // End index in f_row
        float products[SIMD_WIDTH];  // Intermediate products (MUL stage output)
        
        PipelineStage() : valid(false), a_val(0.0f), chunk_start(0), chunk_end(0) {
            for (int i = 0; i < SIMD_WIDTH; i++) {
                products[i] = 0.0f;
            }
        }
    };
    PipelineStage pipe_mul;      // MUL stage (Stage 1)
    PipelineStage pipe_acc;      // ACC stage (Stage 2)
    int current_chunk;           // Current chunk being processed
    int total_chunks;            // Total chunks for current f_row
    
    // Statistics
    long long total_mac_ops;
    long long busy_cycles;
    long long idle_cycles;
    long long pipeline_stalls;   // Stalls due to pipeline hazards
    long long wait_f_stalls;     // Stalls waiting for F row to arrive
    long long no_a_stalls;       // Stalls with no A work but not fully done (draining)
    
    // Internal methods
    void simd_mac();                       // Perform SIMD MAC operation (legacy)
    void simd_mac_pipelined(float a_val);  // 2-stage pipelined MAC (issue 1 chunk)
    void advance_pipeline();               // Advance pipeline stages
    void maybe_enqueue_a_operand();        // Enqueue at most 1 A operand per cycle
    bool pop_a_operand(float* a_val, int* col_idx); // Dequeue 1 operand
};

#endif // PE_H
