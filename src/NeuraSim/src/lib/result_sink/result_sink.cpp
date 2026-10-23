// SPDX-License-Identifier: Apache-2.0
#include "result_sink.h"

#include <cstdio>
#include <fstream>
#include <mutex>

class ResultSinkState {
public:
    explicit ResultSinkState(const std::string& experiment_name)
        : path("results_neurachip/" + experiment_name + "/component_metrics.jsonl") {}

    std::string path;
    mutable std::mutex mutex;
};

std::string result_json_quote(const std::string& value) {
    std::string encoded = "\"";
    for (std::string::const_iterator it = value.begin(); it != value.end(); ++it) {
        const unsigned char ch = static_cast<unsigned char>(*it);
        switch (ch) {
            case '\\': encoded += "\\\\"; break;
            case '"': encoded += "\\\""; break;
            case '\n': encoded += "\\n"; break;
            case '\r': encoded += "\\r"; break;
            case '\t': encoded += "\\t"; break;
            default:
                if (ch >= 0x20) encoded += static_cast<char>(ch);
        }
    }
    return encoded + "\"";
}

ResultCollection::ResultCollection(std::shared_ptr<ResultSinkState> state,
                                   std::string component)
    : state_(std::move(state)), component_(std::move(component)) {}

void ResultCollection::insert_one(const ResultRecord& record) const {
    if (!state_) return;
    std::lock_guard<std::mutex> lock(state_->mutex);
    std::ofstream output(state_->path.c_str(), std::ios::out | std::ios::app);
    if (!output) return;
    output << "{\"component\":" << result_json_quote(component_);
    for (std::vector<ResultField>::const_iterator it = record.fields.begin();
         it != record.fields.end(); ++it) {
        output << "," << result_json_quote(it->key) << ":" << it->json_value;
    }
    output << "}\n";
}

ResultSink::ResultSink(std::string experiment_name)
    : state_(new ResultSinkState(experiment_name)) {}

bool ResultSink::sanity_check() const {
    std::lock_guard<std::mutex> lock(state_->mutex);
    std::ofstream output(state_->path.c_str(), std::ios::out | std::ios::app);
    return static_cast<bool>(output);
}

ResultCollection ResultSink::get_component_collection(
    const std::string& component) const {
    return ResultCollection(state_, component);
}

void ResultSink::drop() {
    std::lock_guard<std::mutex> lock(state_->mutex);
    std::remove(state_->path.c_str());
}

void ResultSink::remove_document(
    const std::string&, const std::string&, const std::string&) {
    // The standalone sink is append-only; selective deletion is intentionally
    // unsupported because the release writes into a fresh result directory.
}
