// SPDX-License-Identifier: Apache-2.0
#ifndef ARCH_VARIANT_H
#define ARCH_VARIANT_H

#include <string>

enum class ArchVariant {
    H0_SYSTOLIC_SCOPED,
    H1_SPARSEFE_DIGITALBE,
    H2_SPARSEFE_INDEX_DIGITALBE,
    H3_SPARSEFE_INDEX_DIGITAL_STREAM,
    H4_SPARSEFE_INDEX_DCIM,
    H5_SPARSEFE_INDEX_DCIM_STREAM,
    H6_FULL_DESPOINA
};

inline const char* default_arch_variant_string() {
    return "H6_full_despoina";
}

inline bool is_valid_arch_variant(const std::string& s) {
    return s == "H0_systolic_scoped" ||
           s == "H1_sparsefe_digitalbe" ||
           s == "H1_sparse_pe" ||
           s == "H2_sparsefe_index_digitalbe" ||
           s == "H2_metadata_indexstore" ||
           s == "H3_sparsefe_index_digital_stream" ||
           s == "H3_dcim_no_stream" ||
           s == "H4_sparsefe_index_dcim" ||
           s == "H4_dcim_stream" ||
           s == "H5_sparsefe_index_dcim_stream" ||
           s == "H5_full_despoina" ||
           s == "H6_full_despoina";
}

inline ArchVariant parse_arch_variant(const std::string& s) {
    if (s == "H0_systolic_scoped") return ArchVariant::H0_SYSTOLIC_SCOPED;
    if (s == "H1_sparsefe_digitalbe" || s == "H1_sparse_pe") {
        return ArchVariant::H1_SPARSEFE_DIGITALBE;
    }
    if (s == "H2_sparsefe_index_digitalbe" || s == "H2_metadata_indexstore") {
        return ArchVariant::H2_SPARSEFE_INDEX_DIGITALBE;
    }
    if (s == "H3_sparsefe_index_digital_stream") {
        return ArchVariant::H3_SPARSEFE_INDEX_DIGITAL_STREAM;
    }
    if (s == "H4_sparsefe_index_dcim" || s == "H3_dcim_no_stream") {
        return ArchVariant::H4_SPARSEFE_INDEX_DCIM;
    }
    if (s == "H5_sparsefe_index_dcim_stream" || s == "H4_dcim_stream") {
        return ArchVariant::H5_SPARSEFE_INDEX_DCIM_STREAM;
    }
    return ArchVariant::H6_FULL_DESPOINA;
}

inline const char* arch_variant_to_string(ArchVariant variant) {
    switch (variant) {
        case ArchVariant::H0_SYSTOLIC_SCOPED:
            return "H0_systolic_scoped";
        case ArchVariant::H1_SPARSEFE_DIGITALBE:
            return "H1_sparsefe_digitalbe";
        case ArchVariant::H2_SPARSEFE_INDEX_DIGITALBE:
            return "H2_sparsefe_index_digitalbe";
        case ArchVariant::H3_SPARSEFE_INDEX_DIGITAL_STREAM:
            return "H3_sparsefe_index_digital_stream";
        case ArchVariant::H4_SPARSEFE_INDEX_DCIM:
            return "H4_sparsefe_index_dcim";
        case ArchVariant::H5_SPARSEFE_INDEX_DCIM_STREAM:
            return "H5_sparsefe_index_dcim_stream";
        case ArchVariant::H6_FULL_DESPOINA:
            return "H6_full_despoina";
    }
    return default_arch_variant_string();
}

#endif  // ARCH_VARIANT_H
