// SPDX-License-Identifier: Apache-2.0
#ifndef DESPOINA_RESULT_SINK_H
#define DESPOINA_RESULT_SINK_H

#include <memory>
#include <sstream>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

struct ResultField {
    std::string key;
    std::string json_value;
};

struct ResultRecord {
    std::vector<ResultField> fields;
};

std::string result_json_quote(const std::string& value);

inline std::string result_json_value(const std::string& value) {
    return result_json_quote(value);
}

inline std::string result_json_value(const char* value) {
    return result_json_quote(value ? std::string(value) : std::string());
}

inline std::string result_json_value(bool value) {
    return value ? "true" : "false";
}

template <typename T>
typename std::enable_if<std::is_arithmetic<T>::value &&
                        !std::is_same<T, bool>::value, std::string>::type
result_json_value(T value) {
    std::ostringstream stream;
    stream << value;
    return stream.str();
}

template <typename T>
std::string result_json_value(const std::vector<T>& values) {
    std::string encoded = "[";
    for (std::size_t index = 0; index < values.size(); ++index) {
        if (index) encoded += ",";
        encoded += result_json_value(values[index]);
    }
    return encoded + "]";
}

template <typename T>
ResultField result_field(const char* key, const T& value) {
    return ResultField{key, result_json_value(value)};
}

inline void append_result_fields(ResultRecord&) {}

template <typename... Rest>
void append_result_fields(ResultRecord& record, const ResultField& field,
                          const Rest&... rest) {
    record.fields.push_back(field);
    append_result_fields(record, rest...);
}

template <typename... Args>
ResultRecord make_result_record(const Args&... args) {
    ResultRecord record;
    append_result_fields(record, args...);
    return record;
}

class ResultSinkState;

class ResultCollection {
public:
    ResultCollection() = default;
    ResultCollection(std::shared_ptr<ResultSinkState> state, std::string component);
    void insert_one(const ResultRecord& record) const;

private:
    std::shared_ptr<ResultSinkState> state_;
    std::string component_;
};

class ResultSink {
public:
    explicit ResultSink(std::string experiment_name);
    bool sanity_check() const;
    ResultCollection get_component_collection(const std::string& component) const;
    void drop();
    void remove_document(const std::string&, const std::string&, const std::string&);

private:
    std::shared_ptr<ResultSinkState> state_;
};

#endif
