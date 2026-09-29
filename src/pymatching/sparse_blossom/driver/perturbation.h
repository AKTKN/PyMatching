// Copyright 2026 PyMatching Contributors. Licensed under Apache-2.0.
#ifndef PYMATCHING_PERTURBATION_H
#define PYMATCHING_PERTURBATION_H

#include <random>
#include "pymatching/sparse_blossom/matcher/mwpm.h"

namespace pm {
class UserGraph;

// Fixed topology, nonnegative weights only. No colour-code-specific data lives here.
struct PerturbationEnsemble {
    static constexpr uint64_t VERSION = 1;
    double alpha;
    bool clip_probabilities;
    uint64_t seed, stream_id, shot_position = 0;
    size_t size;
    Mwpm work;
    std::vector<double> probabilities, base_weights, weights;
    std::vector<std::vector<weight_int*>> weight_slots;
    std::mt19937_64 rng;

    PerturbationEnsemble(UserGraph& graph, double alpha, uint64_t seed, size_t size, uint64_t stream_id, bool clip_probabilities = false);
    static uint64_t splitmix64(uint64_t value);
    uint64_t shot_seed(uint64_t shot) const;
    void sample_weights(std::mt19937_64& generator, std::vector<double>& output) const;
    void update_weights();
    void decode(UserGraph& graph, const std::vector<uint64_t>& detections, uint64_t shot,
                uint8_t* predictions, double* solution_weights);
};
}
#endif
