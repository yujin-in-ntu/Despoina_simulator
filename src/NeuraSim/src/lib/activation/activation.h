// SPDX-License-Identifier: Apache-2.0
/**
 * activation.h
 * 
 * Activation Unit for DyGNN Accelerator.
 * Performs element-wise activation functions on CiM output.
 * 
 * Supported activations:
 * - ReLU: max(0, x) - 1 cycle (comparator + mux)
 * - Sigmoid: LUT-based - 3 cycles (quantize + lookup + interpolate)
 * - GELU: LUT-based - 3 cycles
 * 
 * Architecture:
 * - 16 parallel lanes (matches CiM output width)
 * - Pipelined: can accept new input every cycle
 * - Throughput: 16 elements/cycle
 */

#ifndef ACTIVATION_H
#define ACTIVATION_H

#include <vector>
#include <cstdint>
#include <cmath>
#include <algorithm>

enum class ActivationType {
    RELU,
    LEAKY_RELU,
    SIGMOID,
    TANH,
    GELU,
    NONE  // Pass-through
};

/**
 * Activation Unit - Pipelined element-wise activation
 */
class ActivationUnit {
public:
    // Configuration
    static constexpr int NUM_LANES = 16;        // Parallel processing lanes
    static constexpr int RELU_LATENCY = 1;      // ReLU: compare + mux
    static constexpr int LUT_LATENCY = 3;       // LUT-based: quantize + lookup + interpolate
    static constexpr int LUT_SIZE = 256;        // 8-bit LUT entries
    static constexpr float LEAKY_ALPHA = 0.01f; // Leaky ReLU slope

    // Constructor
    ActivationUnit();
    ~ActivationUnit();
    
    // Initialize with activation type
    void init(ActivationType type = ActivationType::RELU);
    
    // Main simulation tick - returns true if output ready
    bool tick();
    
    // Submit data for activation (n elements, d features per element)
    // Returns cycles needed
    void start(int n, int d);
    
    // Apply activation to a single row (for integration with CiM pipeline)
    // Returns cycles taken
    int apply_row(float* data, int dim);
    
    // Apply activation to entire matrix (batch mode)
    // Returns total cycles
    uint64_t apply_matrix(float* data, int n, int d);
    
    // Analytical cycle calculation (for fast mode)
    uint64_t compute_cycles_analytical(int n, int d) const;
    
    // State queries
    bool is_done() const { return state == State::DONE || state == State::IDLE; }
    bool is_idle() const { return state == State::IDLE; }
    
    // Statistics
    uint64_t get_total_cycles() const { return total_cycles; }
    uint64_t get_elements_processed() const { return elements_processed; }
    
    // Get activation type name
    static const char* type_name(ActivationType type);

private:
    // State machine
    enum class State {
        IDLE,
        RUNNING,
        DONE
    };
    
    // Internal state
    State state;
    ActivationType act_type;
    uint64_t total_cycles;
    uint64_t remaining_cycles;
    uint64_t elements_processed;
    
    // LUT for non-linear functions
    float sigmoid_lut[LUT_SIZE];
    float tanh_lut[LUT_SIZE];
    float gelu_lut[LUT_SIZE];
    
    // Initialize LUTs
    void init_luts();
    
    // Activation implementations
    float apply_relu(float x) const;
    float apply_leaky_relu(float x) const;
    float apply_sigmoid(float x) const;
    float apply_tanh(float x) const;
    float apply_gelu(float x) const;
    
    // LUT lookup with interpolation
    float lut_lookup(const float* lut, float x, float min_x, float max_x) const;
    
    // Get latency for current activation type
    int get_latency() const;
};

#endif // ACTIVATION_H
