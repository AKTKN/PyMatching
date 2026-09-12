#include "gtest/gtest.h"
#include "pymatching/sparse_blossom/gap_dijkstra/metric_graph.h"
#include "pymatching/sparse_blossom/driver/user_graph.h"
#include "pymatching/sparse_blossom/driver/mwpm_decoding.h"

TEST(SoftOutputMetric, PartialAndParallelEdges) {
    pm::MetricGraph g{{-1,-1}, {{10,0,1,5}, {11,0,1,9}}, {{0,1}}};
    g.validate(0);
    auto [dist, residual] = g.evaluate({1,2});
    ASSERT_EQ(dist, std::vector<double>({2}));
    ASSERT_EQ(residual, std::vector<double>({2,6}));
}

TEST(SoftOutputMetric, UniqueBoundaryHalfEdgeRestriction) {
    pm::MatchingGraph g(2,1);
    g.add_edge(0,1,4,{});
    g.add_boundary_edge(0,2,{});
    ASSERT_EQ(g.nodes[0].neighbors[0], nullptr);
    ASSERT_THROW(g.add_boundary_edge(0,6,{}), std::invalid_argument);
}

TEST(SoftOutputMetric, OrdinaryWeightAndPrediction) {
    pm::UserGraph g;
    g.add_or_merge_boundary_edge(0,{0},1,-1);
    g.add_or_merge_edge(0,1,{1},5,-1);
    g.add_or_merge_boundary_edge(1,{2},2,-1);
    pm::MetricGraph metric{{0,1,-1,-1}, {{0,2,0,1},{1,0,1,5},{2,1,3,2}}, {{2,3}}};
    g.configure_soft_output(metric);
    auto &mwpm = g.get_mwpm();
    for (auto shot : std::vector<std::vector<uint64_t>>{{0},{1},{0,1},{},{0}}) {
        uint8_t a[3] = {}, b[3] = {};
        pm::total_weight_int wa = 0, wb = 0;
        std::vector<double> soft, radii;
        pm::decode_detection_events(mwpm,shot,a,wa);
        pm::decode_detection_events_with_soft_output(mwpm,shot,b,wb,metric,soft,radii);
        ASSERT_EQ(wa,wb);
        for (size_t j=0;j<3;j++) ASSERT_EQ(a[j],b[j]);
        ASSERT_EQ(soft.size(),1);
        ASSERT_GE(soft[0],0);
    }
}
