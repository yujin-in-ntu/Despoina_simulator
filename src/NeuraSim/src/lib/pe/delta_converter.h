// SPDX-License-Identifier: Apache-2.0
#ifndef DELTA_CONVERTER_H
#define DELTA_CONVERTER_H

#include <string>
#include <vector>

#include "work_stealing_scheduler.h"
#include "delta_f_aware_indexer.h"

/**
 * DeltaConverter
 *
 * - ΔA/ΔF 입력 파일을 파싱하고, PEArraySpMM에 넣을 형태로 변환한다.
 * - FeatureStore 업데이트(ΔF 적용)는 DualModeSpMM이 수행하되,
 *   이 클래스는 파싱 결과(변경 노드 리스트, delta 벡터)를 제공한다.
 */
class DeltaConverter {
public:
    using CSRMatrix = WorkStealingScheduler::CSRMatrix;
    using DeltaAUpdate = DeltaFAwareIndexer::DeltaAUpdate;

    struct DeltaFParsed {
        int feature_dim = 0;
        std::vector<int> nodes;          // changed node ids
        std::vector<float> deltas;       // row-major [nodes.size() × d]
    };

    // Parse ΔA file: lines like:
    //   add <src> <dst> <w>
    //   del <src> <dst> <w>
    // If <w> omitted, defaults to 1.0.
    static std::vector<DeltaAUpdate> parse_delta_a_file(const std::string& path);
    static std::vector<DeltaAUpdate> coalesce_delta_a_updates(
        const std::vector<DeltaAUpdate>& updates);

    // Parse ΔF file: lines like:
    //   <node> <d0> <d1> ... <d(d-1)>
    // Each row is a delta vector (F_new - F_old).
    static DeltaFParsed parse_delta_f_file(const std::string& path, int d);
    static DeltaFParsed coalesce_delta_f_updates(const DeltaFParsed& updates);

    // Convert ΔA updates to CSR (row-centric) with del represented as negative weight.
    static CSRMatrix delta_a_to_csr(const std::vector<DeltaAUpdate>& delta_a, int n);

    // Build dense ΔF matrix (n×d) with zeros except changed nodes rows.
    static std::vector<float> delta_f_to_dense(const DeltaFParsed& df, int n, int d);
};

#endif  // DELTA_CONVERTER_H
