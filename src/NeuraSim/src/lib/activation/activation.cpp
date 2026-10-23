// SPDX-License-Identifier: Apache-2.0
/**
 * activation.cpp
 * 
 * Implementation of Activation Unit for DyGNN Accelerator.
 */

#include "activation.h"
#include <iostream>
#include <cstring>

ActivationUnit::ActivationUnit() 
    : state(State::IDLE)
    , act_type(ActivationType::RELU)
    , total_cycles(0)
    , remaining_cycles(0)
    , elements_processed(0)
{
    init_luts();
}

ActivationUnit::~ActivationUnit() {}

void ActivationUnit::init(ActivationType type) {
    act_type = type;
    state = State::IDLE;
    total_cycles = 0;
    remaining_cycles = 0;
    elements_processed = 0;
}

void ActivationUnit::init_luts() {
    const float kPi = 3.14159265358979323846f;

    // Initialize sigmoid LUT: sigmoid(x) = 1 / (1 + exp(-x))
    // Range: [-8, 8] mapped to [0, 255]
    for (int i = 0; i < LUT_SIZE; i++) {
        float x = -8.0f + (16.0f * i / (LUT_SIZE - 1));
        sigmoid_lut[i] = 1.0f / (1.0f + std::exp(-x));
    }
    
    // Initialize tanh LUT: tanh(x)
    // Range: [-4, 4] mapped to [0, 255]
    for (int i = 0; i < LUT_SIZE; i++) {
        float x = -4.0f + (8.0f * i / (LUT_SIZE - 1));
        tanh_lut[i] = std::tanh(x);
    }
    
    // Initialize GELU LUT: GELU(x) = x * Φ(x)
    // Range: [-4, 4] mapped to [0, 255]
    for (int i = 0; i < LUT_SIZE; i++) {
        float x = -4.0f + (8.0f * i / (LUT_SIZE - 1));
        // Approximate GELU
        gelu_lut[i] = 0.5f * x * (1.0f + std::tanh(std::sqrt(2.0f / kPi) * (x + 0.044715f * x * x * x)));
    }
}

bool ActivationUnit::tick() {
    if (state == State::IDLE || state == State::DONE) {
        return false;
    }
    
    total_cycles++;
    
    if (remaining_cycles > 0) {
        remaining_cycles--;
        if (remaining_cycles == 0) {
            state = State::DONE;
        }
        return true;
    }
    
    return false;
}

void ActivationUnit::start(int n, int d) {
    // Calculate total cycles needed
    remaining_cycles = compute_cycles_analytical(n, d);
    state = State::RUNNING;
    elements_processed = static_cast<uint64_t>(n) * d;
}

int ActivationUnit::apply_row(float* data, int dim) {
    int latency = get_latency();
    
    // Apply activation function element-wise
    switch (act_type) {
        case ActivationType::RELU:
            for (int i = 0; i < dim; i++) {
                data[i] = apply_relu(data[i]);
            }
            break;
        case ActivationType::LEAKY_RELU:
            for (int i = 0; i < dim; i++) {
                data[i] = apply_leaky_relu(data[i]);
            }
            break;
        case ActivationType::SIGMOID:
            for (int i = 0; i < dim; i++) {
                data[i] = apply_sigmoid(data[i]);
            }
            break;
        case ActivationType::TANH:
            for (int i = 0; i < dim; i++) {
                data[i] = apply_tanh(data[i]);
            }
            break;
        case ActivationType::GELU:
            for (int i = 0; i < dim; i++) {
                data[i] = apply_gelu(data[i]);
            }
            break;
        case ActivationType::NONE:
            // Pass-through
            break;
    }
    
    elements_processed += dim;
    
    // Return cycles: ceil(dim / NUM_LANES) * latency
    int batches = (dim + NUM_LANES - 1) / NUM_LANES;
    return batches * latency;
}

uint64_t ActivationUnit::apply_matrix(float* data, int n, int d) {
    uint64_t cycles = 0;
    
    for (int i = 0; i < n; i++) {
        cycles += apply_row(data + i * d, d);
    }
    
    total_cycles += cycles;
    return cycles;
}

uint64_t ActivationUnit::compute_cycles_analytical(int n, int d) const {
    int latency = get_latency();
    
    // Each row: ceil(d / NUM_LANES) batches
    int batches_per_row = (d + NUM_LANES - 1) / NUM_LANES;
    
    // Pipeline: first row takes full latency, subsequent rows overlap
    // Total = (n - 1) * 1 + batches_per_row * latency (for last row)
    // Simplified: with good pipelining, throughput is ~1 row/cycle
    
    // Conservative estimate: assume minimal pipelining
    // Total = n * batches_per_row * latency / pipeline_depth
    int pipeline_depth = NUM_LANES;  // Full pipeline
    
    uint64_t total = static_cast<uint64_t>(n) * batches_per_row * latency;
    total = (total + pipeline_depth - 1) / pipeline_depth;
    
    // Minimum latency
    return std::max(total, static_cast<uint64_t>(latency));
}

int ActivationUnit::get_latency() const {
    switch (act_type) {
        case ActivationType::RELU:
        case ActivationType::LEAKY_RELU:
            return RELU_LATENCY;
        case ActivationType::SIGMOID:
        case ActivationType::TANH:
        case ActivationType::GELU:
            return LUT_LATENCY;
        case ActivationType::NONE:
            return 0;
    }
    return 1;
}

float ActivationUnit::apply_relu(float x) const {
    return std::max(0.0f, x);
}

float ActivationUnit::apply_leaky_relu(float x) const {
    return x > 0 ? x : LEAKY_ALPHA * x;
}

float ActivationUnit::apply_sigmoid(float x) const {
    return lut_lookup(sigmoid_lut, x, -8.0f, 8.0f);
}

float ActivationUnit::apply_tanh(float x) const {
    return lut_lookup(tanh_lut, x, -4.0f, 4.0f);
}

float ActivationUnit::apply_gelu(float x) const {
    return lut_lookup(gelu_lut, x, -4.0f, 4.0f);
}

float ActivationUnit::lut_lookup(const float* lut, float x, float min_x, float max_x) const {
    // Clamp to range
    x = std::max(min_x, std::min(max_x, x));
    
    // Map to LUT index
    float normalized = (x - min_x) / (max_x - min_x);
    float idx_f = normalized * (LUT_SIZE - 1);
    int idx = static_cast<int>(idx_f);
    float frac = idx_f - idx;
    
    // Linear interpolation
    if (idx >= LUT_SIZE - 1) {
        return lut[LUT_SIZE - 1];
    }
    
    return lut[idx] * (1.0f - frac) + lut[idx + 1] * frac;
}

const char* ActivationUnit::type_name(ActivationType type) {
    switch (type) {
        case ActivationType::RELU: return "ReLU";
        case ActivationType::LEAKY_RELU: return "LeakyReLU";
        case ActivationType::SIGMOID: return "Sigmoid";
        case ActivationType::TANH: return "Tanh";
        case ActivationType::GELU: return "GELU";
        case ActivationType::NONE: return "None";
    }
    return "Unknown";
}
