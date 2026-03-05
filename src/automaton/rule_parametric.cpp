#include "automaton/rule_parametric.hpp"
#include <cmath>
#include <algorithm>
#include <cstring>

namespace discretum {

float ParametricRule::sigmoid(float x) {
    return 1.0f / (1.0f + std::exp(-x));
}

ParametricRule::ParametricRule() {
    std::memset(theta_state, 0, sizeof(theta_state));
    std::memset(theta_topo, 0, sizeof(theta_topo));
}

ParametricRule::ParametricRule(const std::vector<float>& params) {
    set_params(params);
}

void ParametricRule::set_params(const std::vector<float>& params) {
    int ns = NUM_STATES * NUM_STATES;
    for (int i = 0; i < ns && i < static_cast<int>(params.size()); ++i)
        theta_state[i] = params[i];
    for (int i = 0; i < NUM_TOPO_PARAMS && ns + i < static_cast<int>(params.size()); ++i)
        theta_topo[i] = params[ns + i];
}

std::vector<float> ParametricRule::get_params() const {
    std::vector<float> p(TOTAL_PARAMS);
    int ns = NUM_STATES * NUM_STATES;
    for (int i = 0; i < ns; ++i) p[i] = theta_state[i];
    for (int i = 0; i < NUM_TOPO_PARAMS; ++i) p[ns + i] = theta_topo[i];
    return p;
}

void ParametricRule::apply(DynamicGraph& graph, PCG32& rng) {
    apply_state_update(graph);
    apply_topo_update(graph, rng);
}

void ParametricRule::apply_state_update(DynamicGraph& graph) {
    // For each node, compute new state based on:
    //   - Its current state s
    //   - The majority state m among its neighbours
    //   - theta_state[s][m] determines the transition weight
    //
    // New state = argmax_s'  theta_state[s'][m]
    // (deterministic update — all nodes updated synchronously)
    
    uint32_t n = graph.num_nodes();
    std::vector<uint8_t> new_states(n);
    
    for (uint32_t i = 0; i < n; ++i) {
        if (graph.degree(i) == 0) {
            new_states[i] = graph.get_state(i);
            continue;
        }
        
        // Count neighbour states
        int counts[NUM_STATES] = {};
        auto nbrs = graph.neighbors(i);
        for (uint32_t nbr : nbrs) {
            uint8_t s = graph.get_state(nbr);
            if (s < NUM_STATES) counts[s]++;
        }
        
        // Find majority state among neighbours
        int majority = 0;
        for (int s = 1; s < NUM_STATES; ++s) {
            if (counts[s] > counts[majority]) majority = s;
        }
        
        // Compute transition scores for each possible new state
        uint8_t cur = graph.get_state(i);
        float best_score = -1e30f;
        int best_state = cur;
        for (int s = 0; s < NUM_STATES; ++s) {
            float score = theta_state[s * NUM_STATES + majority];
            // Bias toward keeping current state (stability)
            if (s == cur) score += 0.5f;
            if (score > best_score) {
                best_score = score;
                best_state = s;
            }
        }
        new_states[i] = static_cast<uint8_t>(best_state);
    }
    
    // Apply synchronously
    for (uint32_t i = 0; i < n; ++i) {
        graph.set_state(i, new_states[i]);
    }
}

void ParametricRule::apply_topo_update(DynamicGraph& graph, PCG32& rng) {
    float p_add    = sigmoid(theta_topo[0]);
    float p_remove = sigmoid(theta_topo[1]);
    float p_rewire = sigmoid(theta_topo[2]);
    float p_split  = sigmoid(theta_topo[3]);
    float p_merge  = sigmoid(theta_topo[4]);
    
    uint32_t n = graph.num_nodes();
    if (n == 0) return;
    
    // Edge addition: for random node pairs at distance 2, add edge with probability p_add
    uint32_t add_attempts = static_cast<uint32_t>(n * p_add);
    for (uint32_t a = 0; a < add_attempts; ++a) {
        uint32_t u = rng.uniform(n);
        if (graph.degree(u) == 0) continue;
        auto nbrs_u = graph.neighbors(u);
        if (nbrs_u.empty()) continue;
        
        // Pick a random neighbour's neighbour (2-hop)
        uint32_t mid = nbrs_u[rng.uniform(nbrs_u.size())];
        auto nbrs_mid = graph.neighbors(mid);
        if (nbrs_mid.empty()) continue;
        uint32_t v = nbrs_mid[rng.uniform(nbrs_mid.size())];
        
        if (u != v && !graph.has_edge(u, v)) {
            graph.add_edge(u, v);
        }
    }
    
    // Edge removal: for random edges, remove with probability p_remove
    // Collect edges first to avoid modifying while iterating
    uint32_t remove_attempts = static_cast<uint32_t>(n * p_remove * 0.5f);
    for (uint32_t a = 0; a < remove_attempts; ++a) {
        uint32_t u = rng.uniform(n);
        auto nbrs = graph.neighbors(u);
        if (nbrs.size() <= 1) continue;  // Don't disconnect nodes
        uint32_t v = nbrs[rng.uniform(nbrs.size())];
        if (graph.degree(v) <= 1) continue;  // Don't disconnect v either
        graph.remove_edge(u, v);
    }
    
    // Edge rewiring: pick a random edge, rewire one endpoint
    uint32_t rewire_attempts = static_cast<uint32_t>(n * p_rewire * 0.3f);
    for (uint32_t a = 0; a < rewire_attempts; ++a) {
        uint32_t u = rng.uniform(n);
        auto nbrs = graph.neighbors(u);
        if (nbrs.empty()) continue;
        uint32_t old_v = nbrs[rng.uniform(nbrs.size())];
        uint32_t new_v = rng.uniform(n);
        if (new_v != u && new_v != old_v && !graph.has_edge(u, new_v)) {
            graph.rewire_edge(u, old_v, new_v);
        }
    }
    
    // Node splitting: split high-degree nodes
    uint32_t split_attempts = static_cast<uint32_t>(n * p_split * 0.1f);
    for (uint32_t a = 0; a < split_attempts; ++a) {
        uint32_t u = rng.uniform(n);
        if (graph.degree(u) >= 6) {
            SplitParams sp;
            sp.redistribution_ratio = 0.5f;
            graph.split_node(u, sp);
        }
    }
    
    // Node merging: merge pairs of low-degree connected nodes
    uint32_t merge_attempts = static_cast<uint32_t>(n * p_merge * 0.1f);
    for (uint32_t a = 0; a < merge_attempts; ++a) {
        uint32_t u = rng.uniform(graph.num_nodes());
        if (graph.degree(u) <= 2 && graph.degree(u) > 0) {
            auto nbrs = graph.neighbors(u);
            uint32_t v = nbrs[rng.uniform(nbrs.size())];
            if (graph.degree(v) <= 2 && u != v) {
                graph.merge_nodes(u, v);
            }
        }
    }
}

} // namespace discretum
