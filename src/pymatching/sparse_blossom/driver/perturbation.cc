// Copyright 2026 PyMatching Contributors. Licensed under Apache-2.0.
#include "perturbation.h"
#include "user_graph.h"
#include "mwpm_decoding.h"
#include <algorithm>
#include <set>

uint64_t pm::PerturbationEnsemble::splitmix64(uint64_t x) {
    x += 0x9E3779B97F4A7C15ULL;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
    return x ^ (x >> 31);
}

uint64_t pm::PerturbationEnsemble::shot_seed(uint64_t shot) const {
    return splitmix64(seed ^ splitmix64(shot) ^ splitmix64(stream_id + 0xD1B54A32D192ED03ULL));
}

pm::PerturbationEnsemble::PerturbationEnsemble(
        UserGraph& graph, double alpha, uint64_t seed, size_t size, uint64_t stream_id)
    : alpha(alpha), seed(seed), stream_id(stream_id), size(size) {
    if (!std::isfinite(alpha) || alpha < 0 || alpha > 1 || size == 0)
        throw std::invalid_argument("alpha must be in [0, 1] and ensemble_size must be positive");
    std::set<std::pair<size_t, size_t>> seen;
    for (const auto& edge : graph.edges) {
        auto p = edge.error_probability;
        if (!std::isfinite(p) || p <= 0 || p > 0.5 || !std::isfinite(edge.weight) || edge.weight < 0)
            throw std::invalid_argument("Perturbation requires finite nonnegative weights and error_probabilities in (0, 0.5]");
        auto expected = std::log((1 - p) / p);
        if (std::abs(edge.weight - expected) > 1e-12 * std::max(1.0, std::abs(expected)))
            throw std::invalid_argument("Perturbation weights must equal log((1-p)/p)");
        if (size > 1 && alpha > 0 && p * (1 + alpha) > 0.5)
            throw std::invalid_argument("Perturbation could produce negative weights: p*(1+alpha) exceeds 0.5");
        size_t u = edge.node1, v = edge.node2;
        if (graph.is_boundary_node(u)) std::swap(u, v);
        if (graph.is_boundary_node(v)) v = SIZE_MAX;
        if (u == v || graph.is_boundary_node(u) || !seen.insert(std::minmax(u, v)).second)
            throw std::invalid_argument("Perturbation does not support self loops or parallel edges");
        probabilities.push_back(p);
        base_weights.push_back(edge.weight);
    }
    weights.resize(base_weights.size());
    graph.get_mwpm();  // Pristine solver, including its original integer weights.
    if (size == 1 || alpha == 0) return;
    work = graph.to_mwpm(NUM_DISTINCT_WEIGHTS, false);
    // Resolve both directed weight slots once, after boundary insertions are complete.
    for (const auto& edge : graph.edges) {
        size_t u = edge.node1, v = edge.node2;
        if (graph.is_boundary_node(u)) std::swap(u, v);
        if (graph.is_boundary_node(v)) v = SIZE_MAX;
        weight_slots.emplace_back();
        auto& slots = weight_slots.back();
        auto& matching = work.flooder.graph.nodes;
        auto target = v == SIZE_MAX ? nullptr : &matching[v];
        auto index = matching[u].index_of_neighbor(target);
        slots.push_back(&matching[u].neighbor_weights[index]);
        if (v != SIZE_MAX)
            slots.push_back(&matching[v].neighbor_weights[matching[v].index_of_neighbor(&matching[u])]);
        auto& search = work.search_flooder.graph.nodes;
        if (!search.empty()) {
            auto s_target = v == SIZE_MAX ? nullptr : &search[v];
            slots.push_back(&search[u].neighbor_weights[search[u].index_of_neighbor(s_target)]);
            if (v != SIZE_MAX)
                slots.push_back(&search[v].neighbor_weights[search[v].index_of_neighbor(&search[u])]);
        }
    }
}

void pm::PerturbationEnsemble::sample_weights(std::mt19937_64& generator, std::vector<double>& output) const {
    for (size_t edge = 0; edge < probabilities.size(); edge++) {
        double uniform = static_cast<double>(generator() >> 11) * 0x1.0p-53;
        double p = std::clamp(probabilities[edge] * (1 + alpha * (2 * uniform - 1)), 1e-14, 1 - 1e-14);
        output[edge] = std::log((1 - p) / p);
    }
}

void pm::PerturbationEnsemble::update_weights() {
    double max_weight = 0;
    bool all_integral = true;
    for (auto w : weights) {
        max_weight = std::max(max_weight, w);
        all_integral &= std::round(w) == w;
    }
    double scale = all_integral ? 1.0 : static_cast<double>(NUM_DISTINCT_WEIGHTS - 1) / max_weight;
    for (size_t edge = 0; edge < weights.size(); edge++) {
        auto w = static_cast<weight_int>(std::round(weights[edge] * scale)) * 2;
        for (auto slot : weight_slots[edge]) *slot = w;
    }
    work.flooder.graph.normalising_constant = 2 * scale;
}

void pm::PerturbationEnsemble::decode(
        UserGraph& graph, const std::vector<uint64_t>& detections, uint64_t shot,
        uint8_t* predictions, double* solution_weights) {
    auto& base = graph.get_mwpm();
    size_t width = graph.get_num_observables();
    rng.seed(shot_seed(shot));
    try {
        total_weight_int total = 0;
        decode_detection_events(base, detections, predictions, total);
        solution_weights[0] = static_cast<double>(total) / base.flooder.graph.normalising_constant;
        for (size_t member = 1; member < size; member++) {
            if (alpha == 0) {
                std::copy_n(predictions, width, predictions + member * width);
                solution_weights[member] = solution_weights[0];
            } else {
                sample_weights(rng, weights);
                work.reset();
                update_weights();
                total = 0;
                decode_detection_events(work, detections, predictions + member * width, total);
                solution_weights[member] = static_cast<double>(total) / work.flooder.graph.normalising_constant;
            }
        }
    } catch (...) {
        base.reset();
        work.reset();
        throw;
    }
}
