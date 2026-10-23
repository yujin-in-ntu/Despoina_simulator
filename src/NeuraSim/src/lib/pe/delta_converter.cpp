// SPDX-License-Identifier: Apache-2.0
#include "delta_converter.h"

#include <fstream>
#include <cmath>
#include <map>
#include <sstream>
#include <stdexcept>

std::vector<DeltaConverter::DeltaAUpdate>
DeltaConverter::parse_delta_a_file(const std::string& path) {
    std::vector<DeltaAUpdate> out;
    if (path.empty()) return out;

    std::ifstream fin(path);
    if (!fin.is_open()) {
        throw std::runtime_error("Cannot open delta-A file: " + path);
    }

    std::string line;
    while (std::getline(fin, line)) {
        if (line.empty()) continue;
        if (line[0] == '#') continue;
        std::istringstream iss(line);
        std::string op;
        int src, dst;
        float w = 1.0f;
        if (!(iss >> op >> src >> dst)) continue;
        if (iss >> w) { /* ok */ }

        DeltaAUpdate u;
        u.src = src;
        u.dst = dst;
        u.w = w;
        u.is_add = (op == "add" || op == "ADD" || op == "+");
        if (op == "del" || op == "DEL" || op == "-") u.is_add = false;
        out.push_back(u);
    }
    return coalesce_delta_a_updates(out);
}

std::vector<DeltaConverter::DeltaAUpdate>
DeltaConverter::coalesce_delta_a_updates(
    const std::vector<DeltaAUpdate>& updates) {
    std::map<std::pair<int, int>, DeltaAUpdate> unique;
    for (const auto& u : updates) {
        const std::pair<int, int> key{u.src, u.dst};
        auto it = unique.find(key);
        if (it == unique.end()) {
            unique.emplace(key, u);
            continue;
        }
        // Exact repeated set-style graph operations are idempotent and collapse
        // to one update. Mixed add/delete or conflicting weights are ambiguous
        // without the pre-update adjacency state, so fail closed.
        if (it->second.is_add != u.is_add ||
            std::fabs(it->second.w - u.w) > 1e-7f) {
            throw std::invalid_argument(
                "conflicting duplicate delta-A update for one edge");
        }
    }
    std::vector<DeltaAUpdate> out;
    out.reserve(unique.size());
    for (const auto& kv : unique) out.push_back(kv.second);
    return out;
}

DeltaConverter::DeltaFParsed
DeltaConverter::parse_delta_f_file(const std::string& path, int d) {
    DeltaFParsed df;
    df.feature_dim = d;
    if (path.empty()) return df;

    std::ifstream fin(path);
    if (!fin.is_open()) {
        throw std::runtime_error("Cannot open delta-F file: " + path);
    }

    std::string line;
    while (std::getline(fin, line)) {
        if (line.empty()) continue;
        if (line[0] == '#') continue;
        std::istringstream iss(line);
        int node;
        if (!(iss >> node)) continue;
        df.nodes.push_back(node);
        for (int i = 0; i < d; i++) {
            float v = 0.0f;
            if (!(iss >> v)) {
                // If missing values, pad with zeros.
                v = 0.0f;
            }
            df.deltas.push_back(v);
        }
    }
    return coalesce_delta_f_updates(df);
}

DeltaConverter::DeltaFParsed
DeltaConverter::coalesce_delta_f_updates(const DeltaFParsed& updates) {
    if (updates.feature_dim <= 0 ||
        updates.deltas.size() != updates.nodes.size() *
            static_cast<size_t>(updates.feature_dim)) {
        throw std::invalid_argument("invalid delta-F shape");
    }
    std::map<int, std::vector<float>> sums;
    for (size_t p = 0; p < updates.nodes.size(); ++p) {
        auto& row = sums[updates.nodes[p]];
        if (row.empty()) row.assign(static_cast<size_t>(updates.feature_dim), 0.0f);
        for (int j = 0; j < updates.feature_dim; ++j) {
            row[static_cast<size_t>(j)] +=
                updates.deltas[p * static_cast<size_t>(updates.feature_dim) + j];
        }
    }
    DeltaFParsed out;
    out.feature_dim = updates.feature_dim;
    out.nodes.reserve(sums.size());
    out.deltas.reserve(sums.size() * static_cast<size_t>(updates.feature_dim));
    for (const auto& kv : sums) {
        out.nodes.push_back(kv.first);
        out.deltas.insert(out.deltas.end(), kv.second.begin(), kv.second.end());
    }
    return out;
}

DeltaConverter::CSRMatrix
DeltaConverter::delta_a_to_csr(const std::vector<DeltaAUpdate>& delta_a, int n) {
    const std::vector<DeltaAUpdate> canonical = coalesce_delta_a_updates(delta_a);
    // Build per-row edge list then CSR.
    std::vector<std::vector<std::pair<int, float>>> per_row(n);
    int nnz = 0;
    for (const auto& u : canonical) {
        if (u.src < 0 || u.src >= n) continue;
        if (u.dst < 0 || u.dst >= n) continue;
        float w = u.is_add ? u.w : -u.w;
        per_row[u.src].push_back({ u.dst, w });
        nnz++;
    }

    CSRMatrix csr;
    csr.n = n;
    csr.nnz = nnz;
    csr.row_ptr.assign(n + 1, 0);
    csr.col_idx.resize(nnz);
    csr.values.resize(nnz);

    int ptr = 0;
    for (int i = 0; i < n; i++) {
        csr.row_ptr[i] = ptr;
        auto& row = per_row[i];
        std::sort(row.begin(), row.end(),
                  [](const std::pair<int, float>& a, const std::pair<int, float>& b) {
                      return a.first < b.first;
                  });
        for (const auto& kv : row) {
            csr.col_idx[ptr] = kv.first;
            csr.values[ptr] = kv.second;
            ptr++;
        }
    }
    csr.row_ptr[n] = ptr;
    csr.nnz = ptr;
    csr.col_idx.resize(ptr);
    csr.values.resize(ptr);
    return csr;
}

std::vector<float>
DeltaConverter::delta_f_to_dense(const DeltaFParsed& df, int n, int d) {
    std::vector<float> dense(static_cast<size_t>(n) * static_cast<size_t>(d), 0.0f);
    for (size_t idx = 0; idx < df.nodes.size(); idx++) {
        int node = df.nodes[idx];
        if (node < 0 || node >= n) continue;
        const float* src = df.deltas.data() + idx * d;
        float* dst = dense.data() + static_cast<size_t>(node) * static_cast<size_t>(d);
        for (int j = 0; j < d; j++) {
            dst[j] += src[j];
        }
    }
    return dense;
}
