// SPDX-License-Identifier: Apache-2.0
#ifndef RUNTIME_EXPORT_R4_H
#define RUNTIME_EXPORT_R4_H

#include <cerrno>
#include <cstdint>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <sys/stat.h>
#include <vector>

namespace despoina_r4_export {

class RuntimeExportError : public std::runtime_error {
public:
    explicit RuntimeExportError(const std::string& what) : std::runtime_error(what) {}
};

inline bool ensure_dir_recursive(const std::string& path) {
    if (path.empty()) return true;
    std::string cur = path[0] == '/' ? "/" : "";
    std::stringstream ss(path);
    std::string part;
    while (std::getline(ss, part, '/')) {
        if (part.empty() || part == ".") continue;
        if (part == "..") return false;
        if (!cur.empty() && cur[cur.size() - 1] != '/') cur.push_back('/');
        cur += part;
        if (mkdir(cur.c_str(), 0777) != 0 && errno != EEXIST) return false;
        struct stat st;
        if (stat(cur.c_str(), &st) != 0 || !S_ISDIR(st.st_mode)) return false;
    }
    return true;
}

inline uint64_t fnv1a64(const void* data, size_t bytes) {
    const uint8_t* p = static_cast<const uint8_t*>(data);
    uint64_t h = 1469598103934665603ULL;
    for (size_t i = 0; i < bytes; ++i) {
        h ^= static_cast<uint64_t>(p[i]);
        h *= 1099511628211ULL;
    }
    return h;
}

template <typename T>
uint64_t vector_hash(const std::vector<T>& data) {
    return fnv1a64(data.empty() ? nullptr : data.data(), data.size() * sizeof(T));
}

inline void verify_file_size_or_throw(const std::string& path, size_t expected_bytes) {
    struct stat st;
    if (stat(path.c_str(), &st) != 0 || !S_ISREG(st.st_mode) ||
        static_cast<uint64_t>(st.st_size) != static_cast<uint64_t>(expected_bytes)) {
        throw RuntimeExportError("runtime export size mismatch: " + path);
    }
}

template <typename T>
void write_vector_checked(const std::string& path, const std::vector<T>& data) {
    const size_t bytes = data.size() * sizeof(T);
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out.is_open()) throw RuntimeExportError("runtime export open failed: " + path);
    if (bytes > 0) {
        out.write(reinterpret_cast<const char*>(data.data()),
                  static_cast<std::streamsize>(bytes));
    }
    if (!out.good()) throw RuntimeExportError("runtime export write failed: " + path);
    out.flush();
    if (!out.good()) throw RuntimeExportError("runtime export flush failed: " + path);
    out.close();
    if (out.fail()) throw RuntimeExportError("runtime export close failed: " + path);
    verify_file_size_or_throw(path, bytes);
}

inline void write_text_checked(const std::string& path, const std::string& data) {
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out.is_open()) throw RuntimeExportError("runtime export open failed: " + path);
    if (!data.empty()) out.write(data.data(), static_cast<std::streamsize>(data.size()));
    if (!out.good()) throw RuntimeExportError("runtime export write failed: " + path);
    out.flush();
    if (!out.good()) throw RuntimeExportError("runtime export flush failed: " + path);
    out.close();
    if (out.fail()) throw RuntimeExportError("runtime export close failed: " + path);
    verify_file_size_or_throw(path, data.size());
}

}  // namespace despoina_r4_export

#endif
