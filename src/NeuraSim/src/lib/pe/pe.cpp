// SPDX-License-Identifier: Apache-2.0
/**
 * PE (Processing Element) Implementation
 * 
 * Cycle-accurate simulation of a single PE in the PE Array.
 * Each PE performs SIMD MAC operations for SpMM computation.
 * 
 * 2-Stage Pipeline Architecture:
 *   Cycle N:   [MUL: chunk_i] [ACC: chunk_(i-1)]
 *   Cycle N+1: [MUL: chunk_(i+1)] [ACC: chunk_i]
 *   ...
 * 
 * This allows overlapped execution with 1-cycle initiation interval.
 */

#include "pe.h"
#include <cmath>
#include <algorithm>

PE::PE() : gcp(nullptr), pe_id(-1), cluster_id(-1), f_reg(nullptr), psum_reg(nullptr) {
    // Allocate large buffers on heap
    f_reg = new float[F_REG_SIZE]();      // () for zero-initialization
    psum_reg = new float[PSUM_REG_SIZE]();
    
    // Zero-initialize small arrays
    std::memset(a_reg, 0, sizeof(a_reg));
    std::memset(a_col_idx, 0, sizeof(a_col_idx));
    std::memset(col_idx_fifo, 0, sizeof(col_idx_fifo));
    std::memset(a_val_fifo, 0, sizeof(a_val_fifo));
    
    a_reg_count = 0;
    a_reg_enqueue_ptr = 0;
    f_reg_col_idx = -1;
    f_reg_dim = 0;
    output_row_id = -1;
    
    fifo_head = 0;
    fifo_tail = 0;
    fifo_count = 0;
    
    busy = false;
    done = true;
    waiting_for_f = false;
    needed_f_col = -1;
    remaining_cycles = 0;
    active_operand_valid = false;
    active_a_val = 0.0f;
    active_col_idx = -1;
    
    // Initialize 2-stage pipeline
    pipe_mul = PipelineStage();
    pipe_acc = PipelineStage();
    current_chunk = 0;
    total_chunks = 0;
    
    total_mac_ops = 0;
    busy_cycles = 0;
    idle_cycles = 0;
    pipeline_stalls = 0;
    wait_f_stalls = 0;
    no_a_stalls = 0;
}

PE::PE(GlobalCounters* gcp, int pe_id, int cluster_id) : PE() {
    init(gcp, pe_id, cluster_id);
}

PE::~PE() {
    // Free heap-allocated buffers
    delete[] f_reg;
    delete[] psum_reg;
    f_reg = nullptr;
    psum_reg = nullptr;
}

void PE::init(GlobalCounters* gcp, int pe_id, int cluster_id) {
    this->gcp = gcp;
    this->pe_id = pe_id;
    this->cluster_id = cluster_id;
    
    // Reset state
    clear_psum();
    a_reg_count = 0;
    a_reg_enqueue_ptr = 0;
    busy = false;
    done = true;
    waiting_for_f = false;
    needed_f_col = -1;
    fifo_head = 0;
    fifo_tail = 0;
    fifo_count = 0;
    active_operand_valid = false;
    active_a_val = 0.0f;
    active_col_idx = -1;
    
    // Reset 2-stage pipeline
    pipe_mul = PipelineStage();
    pipe_acc = PipelineStage();
    current_chunk = 0;
    total_chunks = 0;
    pipeline_stalls = 0;
}

bool PE::tick() {
    // If we're done and have no work and pipeline is empty, idle
    if (done && a_reg_count == 0 && !pipe_mul.valid && !pipe_acc.valid) {
        idle_cycles++;
        return false;
    }
    
    // Always advance pipeline first (cycle-accurate)
    advance_pipeline();

    // Enqueue at most 1 new A operand per cycle from staging regs into FIFO.
    maybe_enqueue_a_operand();

    // If waiting for F row from cluster, stall (but pipeline can still drain)
    if (waiting_for_f) {
        wait_f_stalls++;
        if (pipe_acc.valid || pipe_mul.valid) {
            busy_cycles++;
            return true;
        }
        idle_cycles++;
        return false;
    }
    
    // =========================================================================
    // 2-Stage Pipelined Execution
    // =========================================================================
    // Each cycle:
    //   1. ACC stage commits results from previous MUL
    //   2. MUL stage receives new data from ACC stage
    //   3. New work enters MUL stage (if available)
    // =========================================================================
    
    // Start a new active operand if none is active (dequeue 1/cycle at most)
    if (!active_operand_valid) {
        float a_val = 0.0f;
        int col_idx = -1;
        if (pop_a_operand(&a_val, &col_idx)) {
            active_operand_valid = true;
            active_a_val = a_val;
            active_col_idx = col_idx;
            current_chunk = 0;
            total_chunks = 0;
        }
    }

    // Process active operand if present
    if (active_operand_valid) {
        // Need matching F row before issuing MAC chunks
        if (f_reg_col_idx != active_col_idx) {
            waiting_for_f = true;
            needed_f_col = active_col_idx;
            if (pipe_mul.valid || pipe_acc.valid) {
                busy_cycles++;
                return true;
            }
            idle_cycles++;
            return false;
        }

        if (total_chunks == 0) {
            total_chunks = (f_reg_dim + SIMD_WIDTH - 1) / SIMD_WIDTH;
        }

        // Issue exactly one chunk per cycle into MUL stage
        if (!pipe_mul.valid && current_chunk < total_chunks) {
            simd_mac_pipelined(active_a_val);
        } else if (pipe_mul.valid && current_chunk < total_chunks) {
            pipeline_stalls++;
        }

        // Operand done when all chunks issued and pipeline drained
        if (current_chunk >= total_chunks && !pipe_mul.valid && !pipe_acc.valid) {
            active_operand_valid = false;
            active_a_val = 0.0f;
            active_col_idx = -1;
            current_chunk = 0;
            total_chunks = 0;
        }

        busy_cycles++;
        return true;
    }
    
    // No more A values, but drain remaining pipeline
    if (pipe_mul.valid || pipe_acc.valid) {
        // We are effectively stalled on new work while draining
        no_a_stalls++;
        busy_cycles++;
        return true;
    }
    
    // All done when no staging, no FIFO, no active operand, pipeline empty
    if (a_reg_count == 0 && fifo_count == 0 && !active_operand_valid &&
        !pipe_mul.valid && !pipe_acc.valid) {
        done = true;
        busy = false;
        idle_cycles++;
        return false;
    }

    // Otherwise, we're idle this cycle but not globally done (should be rare)
    idle_cycles++;
    return false;
}

void PE::load_a_values(const float* vals, const int* col_indices, int count) {
    // Load up to A_REG_SIZE values
    a_reg_count = std::min(count, A_REG_SIZE);
    
    for (int i = 0; i < a_reg_count; i++) {
        a_reg[i] = vals[i];
        a_col_idx[i] = col_indices[i];
    }
    
    a_reg_enqueue_ptr = 0;
    done = false;
    busy = true;
}

void PE::load_f_row(const float* row_data, int dim) {
    if (!f_reg || !row_data) return;
    
    // Copy F row into local register
    int copy_dim = std::min(dim, F_REG_SIZE);
    
    for (int i = 0; i < copy_dim; i++) {
        f_reg[i] = row_data[i];
    }
    
    // Zero-pad if necessary
    for (int i = copy_dim; i < F_REG_SIZE; i++) {
        f_reg[i] = 0.0f;
    }
    
    f_reg_dim = dim;
    f_reg_col_idx = needed_f_col;
    
    // Clear waiting state
    waiting_for_f = false;
    needed_f_col = -1;
}

bool PE::has_f_row(int col_idx) const {
    return f_reg_col_idx == col_idx;
}

void PE::reset_for_new_spmm(bool keep_feature_register) {
    // A completed PE still retains f_reg_col_idx/f_reg.  Reusing that register
    // across a different payload epoch aliases F' from Phase 1 as delta-F in
    // Phase 2 whenever the next job's first column happens to match.  LastFM
    // t1 L1 row 540 exposed exactly this case as an all-ones +1 contribution.
    // Preserve the register only for an explicitly proven same-epoch job.
    if (!keep_feature_register) {
        f_reg_col_idx = -1;
        f_reg_dim = 0;
    }

    a_reg_count = 0;
    a_reg_enqueue_ptr = 0;
    fifo_head = 0;
    fifo_tail = 0;
    fifo_count = 0;
    busy = false;
    done = true;
    waiting_for_f = false;
    needed_f_col = -1;
    remaining_cycles = 0;
    active_operand_valid = false;
    active_a_val = 0.0f;
    active_col_idx = -1;
    pipe_mul = PipelineStage();
    pipe_acc = PipelineStage();
    current_chunk = 0;
    total_chunks = 0;
    output_row_id = -1;
    clear_psum();
    reset_stats();
}

void PE::reset_stats() {
    total_mac_ops = 0;
    busy_cycles = 0;
    idle_cycles = 0;
    pipeline_stalls = 0;
    wait_f_stalls = 0;
    no_a_stalls = 0;
}

void PE::set_output_row(int row_id) {
    output_row_id = row_id;
}

void PE::clear_psum() {
    if (psum_reg) {
        std::memset(psum_reg, 0, PSUM_REG_SIZE * sizeof(float));
    }
}

void PE::simd_mac() {
    // Legacy single-cycle MAC (kept for compatibility)
    if (!f_reg || !psum_reg) return;
    
    // NOTE: legacy path kept for compatibility; active operand model uses pipelined path.
    // Use the first staging entry if present.
    float a_val = (a_reg_count > 0) ? a_reg[0] : 0.0f;
    
    // SIMD MAC: psum[i] += a_val * f_reg[i] for all i
    for (int i = 0; i < f_reg_dim && i < PSUM_REG_SIZE; i++) {
        psum_reg[i] += a_val * f_reg[i];
    }
    
    // Count MAC operations (multiply + add = 2 ops per element)
    total_mac_ops += f_reg_dim * 2;
}

void PE::simd_mac_pipelined(float a_val) {
    // 2-Stage Pipelined MAC
    // Stage 1 (MUL): Compute products for current chunk
    // This stage fills pipe_mul with intermediate results
    
    // Null check
    if (!f_reg) return;
    
    // a_val provided by active operand
    int chunk_start = current_chunk * SIMD_WIDTH;
    int chunk_end = std::min(chunk_start + SIMD_WIDTH, f_reg_dim);
    
    // Bounds check
    if (chunk_start < 0 || chunk_end > F_REG_SIZE) {
        return;
    }
    
    // Fill MUL stage
    pipe_mul.valid = true;
    pipe_mul.a_val = a_val;
    pipe_mul.chunk_start = chunk_start;
    pipe_mul.chunk_end = chunk_end;
    
    // Compute products (Stage 1 work)
    for (int i = 0; i < SIMD_WIDTH; i++) {
        int idx = chunk_start + i;
        if (idx >= 0 && idx < chunk_end && idx < f_reg_dim && idx < F_REG_SIZE) {
            pipe_mul.products[i] = a_val * f_reg[idx];
        } else {
            pipe_mul.products[i] = 0.0f;
        }
    }
    
    current_chunk++;
}

void PE::advance_pipeline() {
    // =========================================================================
    // Stage 2 (ACC): Commit results from pipe_acc to psum_reg
    // =========================================================================
    if (pipe_acc.valid && psum_reg) {
        int chunk_start = pipe_acc.chunk_start;
        int chunk_end = pipe_acc.chunk_end;
        
        // Bounds check
        if (chunk_start >= 0 && chunk_end <= PSUM_REG_SIZE) {
            // Accumulate products into psum
            for (int i = 0; i < SIMD_WIDTH; i++) {
                int idx = chunk_start + i;
                if (idx < chunk_end && idx < PSUM_REG_SIZE) {
                    psum_reg[idx] += pipe_acc.products[i];
                }
            }
            
            // Count MAC operations (each product is 1 mul + 1 add = 2 ops)
            int ops = chunk_end - chunk_start;
            total_mac_ops += ops * 2;
        }
        
        pipe_acc = PipelineStage();  // Reset stage
    }
    
    // =========================================================================
    // Move MUL stage -> ACC stage
    // =========================================================================
    if (pipe_mul.valid) {
        // Transfer MUL results to ACC stage
        pipe_acc = pipe_mul;
        pipe_mul = PipelineStage();  // Reset stage
    }
}

void PE::maybe_enqueue_a_operand() {
    // Enqueue at most 1 operand per cycle from staging regs into FIFO.
    if (a_reg_enqueue_ptr >= a_reg_count) return;
    if (fifo_count >= COL_IDX_FIFO_DEPTH) {
        pipeline_stalls++;  // front-end FIFO full
        return;
    }
    a_val_fifo[fifo_tail] = a_reg[a_reg_enqueue_ptr];
    col_idx_fifo[fifo_tail] = a_col_idx[a_reg_enqueue_ptr];
    fifo_tail = (fifo_tail + 1) % COL_IDX_FIFO_DEPTH;
    fifo_count++;
    a_reg_enqueue_ptr++;
    if (a_reg_enqueue_ptr >= a_reg_count) {
        // staging drained
        a_reg_count = 0;
        a_reg_enqueue_ptr = 0;
    }
}

bool PE::pop_a_operand(float* a_val, int* col_idx) {
    if (fifo_count <= 0) return false;
    if (!a_val || !col_idx) return false;
    *a_val = a_val_fifo[fifo_head];
    *col_idx = col_idx_fifo[fifo_head];
    fifo_head = (fifo_head + 1) % COL_IDX_FIFO_DEPTH;
    fifo_count--;
    return true;
}
