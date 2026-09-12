#ifndef PYMATCHING_METRIC_GRAPH_H
#define PYMATCHING_METRIC_GRAPH_H

#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <queue>
#include <set>
#include <stdexcept>
#include <tuple>
#include <vector>

namespace pm {
// Analysis edges remain individually labelled even when the hard matcher merges
// mechanisms. No analysis terminal is inserted into the hard matching graph.
struct MetricEdge {
    size_t id, u, v;
    double weight;
};

struct MetricGraph {
    std::vector<int64_t> node_map;  // -1: analysis-only vertex; otherwise matcher node
    std::vector<MetricEdge> edges;
    std::vector<std::pair<size_t, size_t>> terminal_pairs;

    void validate(size_t matcher_nodes) const {
        if (node_map.empty() || terminal_pairs.empty())
            throw std::invalid_argument("Soft output requires vertices and terminal pairs");
        for (auto v : node_map)
            if (v < -1 || (v >= 0 && (size_t)v >= matcher_nodes))
                throw std::invalid_argument("Invalid analysis-to-matcher node map");
        std::set<size_t> labels;
        for (const auto &e : edges) {
            if (e.u >= node_map.size() || e.v >= node_map.size() || !labels.insert(e.id).second)
                throw std::invalid_argument("Invalid analysis edge endpoint or duplicate edge id");
            if (!std::isfinite(e.weight) || e.weight < 0)
                throw std::invalid_argument("Soft output requires finite nonnegative original weights");
        }
        for (auto [u, v] : terminal_pairs)
            if (u >= node_map.size() || v >= node_map.size())
                throw std::invalid_argument("Invalid soft-output terminal pair");
    }

    // Multi-source propagation of supplied metric-ball radii. This is a graph
    // geometry operation; it does not certify a matching dual.
    std::pair<std::vector<double>, std::vector<double>> evaluate(
        const std::vector<double> &radii) const {
        if (radii.size() != node_map.size())
            throw std::invalid_argument("One radius is required per analysis vertex");
        std::vector<std::vector<std::pair<size_t, double>>> adjacency(node_map.size());
        for (auto &e : edges) {
            adjacency[e.u].push_back({e.v, e.weight});
            adjacency[e.v].push_back({e.u, e.weight});
        }
        std::vector<double> h = radii;
        std::priority_queue<std::pair<double, size_t>> growth;
        for (size_t u = 0; u < h.size(); u++) {
            if (!std::isfinite(h[u]) || h[u] < 0)
                throw std::invalid_argument("Radii must be finite and nonnegative");
            if (h[u] > 0) growth.push({h[u], u});
        }
        while (!growth.empty()) {
            auto [r, u] = growth.top(); growth.pop();
            if (r != h[u]) continue;
            for (auto [v, w] : adjacency[u])
                if (r - w > h[v]) {
                    h[v] = r - w;
                    growth.push({h[v], v});
                }
        }
        for (auto &a : adjacency) a.clear();
        std::vector<double> residuals;
        for (auto &e : edges) {
            double w = std::max(0.0, e.weight - h[e.u] - h[e.v]);
            residuals.push_back(w);
            adjacency[e.u].push_back({e.v, w});
            adjacency[e.v].push_back({e.u, w});
        }
        std::vector<double> outputs;
        for (auto [source, target] : terminal_pairs) {
            std::vector<double> dist(node_map.size(), std::numeric_limits<double>::infinity());
            using Entry = std::pair<double, size_t>;
            std::priority_queue<Entry, std::vector<Entry>, std::greater<Entry>> queue;
            dist[source] = 0; queue.push({0, source});
            while (!queue.empty()) {
                auto [d, u] = queue.top(); queue.pop();
                if (d != dist[u]) continue;
                if (u == target) break;
                for (auto [v, w] : adjacency[u])
                    if (d + w < dist[v]) {
                        dist[v] = d + w;
                        queue.push({dist[v], v});
                    }
            }
            outputs.push_back(dist[target]);
        }
        return {outputs, residuals};
    }
};
}
#endif
