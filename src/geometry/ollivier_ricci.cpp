#include "geometry/ollivier_ricci.hpp"
#include "utils/logging.hpp"
#include <queue>
#include <set>
#include <unordered_set>
#include <algorithm>
#include <numeric>
#include <cmath>

namespace discretum {

namespace {

/**
 * @brief Compute shortest path distances from source using BFS
 * @param graph The graph
 * @param source Source node
 * @param max_distance Maximum distance to explore
 * @return Map from node to distance
 */
std::unordered_map<uint32_t, uint32_t> compute_distances_bfs(const DynamicGraph& graph, 
                                                             uint32_t source,
                                                             uint32_t max_distance = std::numeric_limits<uint32_t>::max()) {
    std::unordered_map<uint32_t, uint32_t> distances;
    std::queue<uint32_t> q;
    
    distances[source] = 0;
    q.push(source);
    
    while (!q.empty()) {
        uint32_t current = q.front();
        q.pop();
        
        uint32_t current_dist = distances[current];
        if (current_dist >= max_distance) continue;
        
        for (uint32_t neighbor : graph.neighbors(current)) {
            if (distances.find(neighbor) == distances.end()) {
                distances[neighbor] = current_dist + 1;
                q.push(neighbor);
            }
        }
    }
    
    return distances;
}

/**
 * @brief Compute W₁ distance via min-cost flow on a bipartite graph.
 *
 * We model the transportation problem as a min-cost flow problem and
 * solve it with the successive shortest paths algorithm (Bellman-Ford
 * for finding augmenting paths with minimum reduced cost).
 *
 * For our use-case (support sizes ≤ 25), this runs in microseconds.
 *
 * Network: source S → supply nodes i → demand nodes j → sink T
 * Edge (S, i): capacity = supply[i], cost = 0
 * Edge (i, j): capacity = ∞, cost = cost_matrix[i][j]
 * Edge (j, T): capacity = demand[j], cost = 0
 *
 * We use integer-scaled capacities (multiply by large N) to avoid
 * floating-point issues in flow algorithms, then convert back.
 */
float solve_transport(const std::vector<std::vector<float>>& cost_matrix,
                      const std::vector<float>& supply,
                      const std::vector<float>& demand) {
    const int m = static_cast<int>(supply.size());
    const int n = static_cast<int>(demand.size());
    
    if (m == 0 || n == 0) return 0.0f;
    
    // Scale to integer capacities for exact arithmetic
    // Use scale factor large enough for precision
    constexpr int64_t SCALE = 1000000;
    
    std::vector<int64_t> isupply(m), idemand(n);
    int64_t total_supply = 0, total_demand = 0;
    for (int i = 0; i < m; ++i) {
        isupply[i] = static_cast<int64_t>(supply[i] * SCALE + 0.5);
        total_supply += isupply[i];
    }
    for (int j = 0; j < n; ++j) {
        idemand[j] = static_cast<int64_t>(demand[j] * SCALE + 0.5);
        total_demand += idemand[j];
    }
    
    // Balance supply and demand (handle rounding)
    int64_t total_flow = std::min(total_supply, total_demand);
    if (total_flow == 0) return 0.0f;
    
    // Network nodes: S=0, supply nodes 1..m, demand nodes m+1..m+n, T=m+n+1
    const int S = 0;
    const int T = m + n + 1;
    const int V = T + 1;
    
    // Adjacency list representation for min-cost flow
    struct Edge {
        int to, rev;      // target node, index of reverse edge
        int64_t cap, flow; // capacity, current flow
        double cost;       // cost per unit flow
    };
    
    std::vector<std::vector<Edge>> graph(V);
    
    auto add_edge = [&](int from, int to, int64_t cap, double cost) {
        graph[from].push_back({to, static_cast<int>(graph[to].size()), cap, 0, cost});
        graph[to].push_back({from, static_cast<int>(graph[from].size()) - 1, 0, 0, -cost});
    };
    
    // S → supply nodes
    for (int i = 0; i < m; ++i) {
        if (isupply[i] > 0)
            add_edge(S, i + 1, isupply[i], 0.0);
    }
    
    // supply nodes → demand nodes
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            add_edge(i + 1, m + 1 + j, total_flow, static_cast<double>(cost_matrix[i][j]));
        }
    }
    
    // demand nodes → T
    for (int j = 0; j < n; ++j) {
        if (idemand[j] > 0)
            add_edge(m + 1 + j, T, idemand[j], 0.0);
    }
    
    // Successive shortest paths with Bellman-Ford (SPFA)
    double total_cost = 0.0;
    int64_t flow_remaining = total_flow;
    
    while (flow_remaining > 0) {
        // Bellman-Ford / SPFA to find shortest path S → T
        std::vector<double> dist(V, 1e18);
        std::vector<bool> in_queue(V, false);
        std::vector<int> prev_node(V, -1), prev_edge(V, -1);
        dist[S] = 0.0;
        
        std::queue<int> q;
        q.push(S);
        in_queue[S] = true;
        
        while (!q.empty()) {
            int u = q.front();
            q.pop();
            in_queue[u] = false;
            
            for (int idx = 0; idx < static_cast<int>(graph[u].size()); ++idx) {
                const Edge& e = graph[u][idx];
                if (e.cap - e.flow > 0 && dist[u] + e.cost < dist[e.to] - 1e-12) {
                    dist[e.to] = dist[u] + e.cost;
                    prev_node[e.to] = u;
                    prev_edge[e.to] = idx;
                    if (!in_queue[e.to]) {
                        q.push(e.to);
                        in_queue[e.to] = true;
                    }
                }
            }
        }
        
        if (dist[T] >= 1e17) break;  // No augmenting path
        
        // Find bottleneck capacity along the path
        int64_t path_flow = flow_remaining;
        for (int v = T; v != S; v = prev_node[v]) {
            Edge& e = graph[prev_node[v]][prev_edge[v]];
            path_flow = std::min(path_flow, e.cap - e.flow);
        }
        
        // Augment flow along the path
        for (int v = T; v != S; v = prev_node[v]) {
            Edge& e = graph[prev_node[v]][prev_edge[v]];
            e.flow += path_flow;
            graph[v][e.rev].flow -= path_flow;
        }
        
        total_cost += path_flow * dist[T];
        flow_remaining -= path_flow;
    }
    
    return static_cast<float>(total_cost / SCALE);
}

} // anonymous namespace

std::unordered_map<uint32_t, float> build_probability_measure(const DynamicGraph& graph,
                                                             uint32_t node,
                                                             float alpha) {
    std::unordered_map<uint32_t, float> measure;
    
    // Add self-loop probability
    if (alpha > 0) {
        measure[node] = alpha;
    }
    
    // Add neighbor probabilities
    auto neighbors = graph.neighbors(node);
    uint32_t degree = neighbors.size();
    
    if (degree > 0) {
        float neighbor_prob = (1.0f - alpha) / degree;
        for (uint32_t neighbor : neighbors) {
            measure[neighbor] = neighbor_prob;
        }
    } else {
        // Isolated node - all probability stays at node
        measure[node] = 1.0f;
    }
    
    return measure;
}

float wasserstein_1_distance(const std::vector<std::vector<float>>& dist_matrix,
                            const std::vector<float>& mu_x,
                            const std::vector<float>& mu_y) {
    return solve_transport(dist_matrix, mu_x, mu_y);
}

float compute_ollivier_ricci(const DynamicGraph& graph, uint32_t u, uint32_t v, float alpha) {
    // Validate input
    if (!graph.has_edge(u, v)) {
        spdlog::warn("compute_ollivier_ricci: No edge between {} and {}", u, v);
        return 0.0f;
    }
    
    if (alpha < 0.0f || alpha >= 1.0f) {
        spdlog::warn("compute_ollivier_ricci: Invalid alpha {}, using 0", alpha);
        alpha = 0.0f;
    }
    
    // Build probability measures
    auto mu_u = build_probability_measure(graph, u, alpha);
    auto mu_v = build_probability_measure(graph, v, alpha);
    
    // Get all nodes in the support of both measures
    std::unordered_set<uint32_t> support;
    for (const auto& [node, prob] : mu_u) support.insert(node);
    for (const auto& [node, prob] : mu_v) support.insert(node);
    
    std::vector<uint32_t> support_vec(support.begin(), support.end());
    std::sort(support_vec.begin(), support_vec.end());
    
    // Compute distances between all pairs in support
    std::unordered_map<uint32_t, std::unordered_map<uint32_t, uint32_t>> all_distances;
    for (uint32_t node : support_vec) {
        all_distances[node] = compute_distances_bfs(graph, node, support_vec.size());
    }
    
    // Build distance matrix for Wasserstein computation
    const size_t n = support_vec.size();
    std::vector<std::vector<float>> dist_matrix(n, std::vector<float>(n));
    
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            uint32_t node_i = support_vec[i];
            uint32_t node_j = support_vec[j];
            
            auto it = all_distances[node_i].find(node_j);
            if (it != all_distances[node_i].end()) {
                dist_matrix[i][j] = static_cast<float>(it->second);
            } else {
                // Nodes not connected - use large distance
                dist_matrix[i][j] = static_cast<float>(graph.num_nodes());
            }
        }
    }
    
    // Build probability vectors
    std::vector<float> p_u(n, 0.0f), p_v(n, 0.0f);
    for (size_t i = 0; i < n; ++i) {
        uint32_t node = support_vec[i];
        if (mu_u.find(node) != mu_u.end()) p_u[i] = mu_u[node];
        if (mu_v.find(node) != mu_v.end()) p_v[i] = mu_v[node];
    }
    
    // Compute Wasserstein distance
    float w1_dist = wasserstein_1_distance(dist_matrix, p_u, p_v);
    
    // Graph distance between u and v
    float graph_dist = 1.0f; // Since we know u and v are adjacent
    
    // Ollivier-Ricci curvature
    float curvature = 1.0f - w1_dist / graph_dist;
    
    return curvature;
}

std::map<std::pair<uint32_t, uint32_t>, float>
compute_all_ollivier_ricci(const DynamicGraph& graph, float alpha) {
    std::map<std::pair<uint32_t, uint32_t>, float> curvatures;
    
    for (uint32_t u = 0; u < graph.get_nodes().size(); ++u) {
        if (graph.degree(u) == 0) continue;
        
        for (uint32_t v : graph.neighbors(u)) {
            auto edge = std::minmax(u, v);
            if (curvatures.find(edge) == curvatures.end()) {
                float curvature = compute_ollivier_ricci(graph, u, v, alpha);
                curvatures[edge] = curvature;
            }
        }
    }
    
    return curvatures;
}

float compute_average_ollivier_ricci(const DynamicGraph& graph, float alpha) {
    auto all_curvatures = compute_all_ollivier_ricci(graph, alpha);
    
    if (all_curvatures.empty()) {
        return 0.0f;
    }
    
    float sum = 0.0f;
    for (const auto& [edge, curvature] : all_curvatures) {
        sum += curvature;
    }
    
    return sum / all_curvatures.size();
}

OllivierRicciStats compute_ollivier_ricci_stats(const DynamicGraph& graph, 
                                                float alpha,
                                                float zero_tolerance) {
    auto all_curvatures = compute_all_ollivier_ricci(graph, alpha);
    
    OllivierRicciStats stats{};
    stats.num_edges = all_curvatures.size();
    
    if (stats.num_edges == 0) {
        return stats;
    }
    
    // Collect all curvature values
    std::vector<float> values;
    values.reserve(all_curvatures.size());
    
    for (const auto& [edge, curvature] : all_curvatures) {
        values.push_back(curvature);
        
        if (std::abs(curvature) < zero_tolerance) {
            stats.num_zero++;
        } else if (curvature > 0) {
            stats.num_positive++;
        } else {
            stats.num_negative++;
        }
    }
    
    // Compute statistics
    stats.mean = std::accumulate(values.begin(), values.end(), 0.0f) / values.size();
    
    // Standard deviation
    float sq_sum = 0.0f;
    for (float val : values) {
        float diff = val - stats.mean;
        sq_sum += diff * diff;
    }
    stats.std_dev = std::sqrt(sq_sum / values.size());
    
    // Min and max
    auto [min_it, max_it] = std::minmax_element(values.begin(), values.end());
    stats.min = *min_it;
    stats.max = *max_it;
    
    // Median
    std::sort(values.begin(), values.end());
    if (values.size() % 2 == 0) {
        stats.median = (values[values.size()/2 - 1] + values[values.size()/2]) / 2.0f;
    } else {
        stats.median = values[values.size()/2];
    }
    
    return stats;
}

} // namespace discretum
