// Copyright 2026 PyMatching Contributors. Licensed under Apache-2.0.
#include "perturbation.h"
#include "user_graph.h"
#include "mwpm_decoding.h"
#include <gtest/gtest.h>

TEST(Perturbation, RandomPrimitiveKnownValues) {
    ASSERT_EQ(pm::PerturbationEnsemble::splitmix64(0), 0xE220A8397B1DCDAFULL);
    std::mt19937_64 rng(5489);
    ASSERT_EQ(rng(), 14514284786278117030ULL);
}

TEST(Perturbation, FreshConstructionAndStableWorkspace) {
    for (size_t width : {size_t(8), size_t(72)}) {
        pm::UserGraph graph(7, width);
        for (size_t node = 0; node < 7; node++) {
            double p = 0.07 + node * 0.017;
            graph.add_or_merge_boundary_edge(node, {node, width - 1}, std::log((1-p)/p), p);
            p += 0.01;
            graph.add_or_merge_edge(node, (node+1)%7, {node}, std::log((1-p)/p), p);
        }
        graph.configure_perturbation(1, 1234, 5, 2);
        auto& ensemble = *graph.perturbation;
        auto nodes = ensemble.work.flooder.graph.nodes.data();
        auto search_nodes = ensemble.work.search_flooder.graph.nodes.data();
        ASSERT_EQ(graph.mwpm_build_count, 2);
        for (uint64_t shot = 0; shot < 40; shot++) {
            std::vector<uint64_t> detections;
            for (size_t node = 0; node < 7; node++)
                if ((shot >> node) & 1) detections.push_back(node);
            std::vector<uint8_t> output(5*width, 0);
            std::vector<double> solution_weights(5);
            ensemble.decode(graph, detections, shot, output.data(), solution_weights.data());
            std::mt19937_64 rng(ensemble.shot_seed(shot));
            std::vector<double> weights(ensemble.base_weights.size());
            for (size_t member = 0; member < 5; member++) {
                if (member == 0) weights = ensemble.base_weights;
                else ensemble.sample_weights(rng, weights);
                pm::UserGraph fresh(7, width);
                size_t index = 0;
                for (const auto& edge : graph.edges) {
                    if (edge.node2 == SIZE_MAX)
                        fresh.add_or_merge_boundary_edge(edge.node1, edge.observable_indices, weights[index], edge.error_probability);
                    else
                        fresh.add_or_merge_edge(edge.node1, edge.node2, edge.observable_indices, weights[index], edge.error_probability);
                    index++;
                }
                std::vector<uint8_t> reference(width, 0);
                pm::total_weight_int total = 0;
                auto& mwpm = fresh.get_mwpm();
                pm::decode_detection_events(mwpm, detections, reference.data(), total);
                ASSERT_EQ(reference, std::vector<uint8_t>(output.begin()+member*width, output.begin()+(member+1)*width));
                ASSERT_DOUBLE_EQ(solution_weights[member], static_cast<double>(total)/mwpm.flooder.graph.normalising_constant);
            }
            ASSERT_EQ(nodes, ensemble.work.flooder.graph.nodes.data());
            ASSERT_EQ(search_nodes, ensemble.work.search_flooder.graph.nodes.data());
            ASSERT_EQ(graph.mwpm_build_count, 2);
        }
    }
}

TEST(Perturbation, ResetAfterInfeasibleSyndrome) {
    pm::UserGraph graph(3, 70);
    double p = 0.1, weight = std::log((1-p)/p);
    graph.add_or_merge_edge(0, 1, {0, 69}, weight, p);
    graph.add_or_merge_edge(1, 2, {1}, weight, p);
    graph.configure_perturbation(1, 9, 3, 0);
    std::vector<uint8_t> output(210);
    std::vector<double> weights(3);
    ASSERT_THROW(graph.perturbation->decode(graph, {0}, 0, output.data(), weights.data()), std::invalid_argument);
    ASSERT_NO_THROW(graph.perturbation->decode(graph, {0, 1}, 0, output.data(), weights.data()));
    ASSERT_EQ(output[0], 1);
    ASSERT_EQ(output[69], 1);
    ASSERT_EQ(graph.mwpm_build_count, 2);
}

TEST(Perturbation, OriginalSolverAndUnsupportedInputs) {
    pm::UserGraph graph;
    graph.add_or_merge_boundary_edge(0, {0}, 0, 0.5);
    ASSERT_THROW(graph.configure_perturbation(1, 0, 2, 0), std::invalid_argument);
    graph.configure_perturbation(0, 0, 3, 0);
    ASSERT_EQ(graph.mwpm_build_count, 1);
    std::vector<uint8_t> output(3);
    std::vector<double> weights(3);
    graph.perturbation->decode(graph, {0}, 0, output.data(), weights.data());
    ASSERT_EQ(output, (std::vector<uint8_t>{1, 1, 1}));
    ASSERT_EQ(weights, (std::vector<double>{0, 0, 0}));
    ASSERT_THROW(graph.add_or_merge_boundary_edge(1, {1}, 1, 0.1), std::invalid_argument);
}
