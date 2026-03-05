#pragma once

#include "core/graph.hpp"
#include "utils/random.hpp"
#include <vector>
#include <cstdint>

namespace discretum {

/**
 * @brief Parametric cellular automaton rule on a dynamic graph.
 *
 * Each step consists of two phases:
 *   1. **State update**: each node's state is updated based on its own
 *      state and its neighbours' states, controlled by theta_state.
 *   2. **Topological update**: edges may be added, removed, or rewired,
 *      and nodes may be split or merged, controlled by theta_topo.
 *
 * Parameter layout:
 *   theta_state (size NUM_STATES * NUM_STATES):
 *     transition_weight[current_state][neighbor_majority_state]
 *   theta_topo  (size 5):
 *     [0] p_edge_add    – probability of adding an edge between 2-hop neighbours
 *     [1] p_edge_remove – probability of removing an edge
 *     [2] p_rewire      – probability of rewiring an edge
 *     [3] p_split       – probability of splitting a high-degree node
 *     [4] p_merge       – probability of merging low-degree nodes
 *
 * All probabilities are passed through sigmoid: p = 1/(1+exp(-θ)).
 */
class ParametricRule {
public:
    static constexpr int NUM_STATES = 3;
    static constexpr int NUM_TOPO_PARAMS = 5;
    static constexpr int TOTAL_PARAMS = NUM_STATES * NUM_STATES + NUM_TOPO_PARAMS;
    
    ParametricRule();
    explicit ParametricRule(const std::vector<float>& params);
    
    /// Set all parameters from a flat vector
    void set_params(const std::vector<float>& params);
    
    /// Get all parameters as a flat vector
    std::vector<float> get_params() const;
    
    /// Apply one step of the automaton to the graph
    void apply(DynamicGraph& graph, PCG32& rng);
    
    /// Apply state update only
    void apply_state_update(DynamicGraph& graph);
    
    /// Apply topological update only
    void apply_topo_update(DynamicGraph& graph, PCG32& rng);
    
    /// Number of parameters
    static int num_params() { return TOTAL_PARAMS; }
    
private:
    float theta_state[NUM_STATES * NUM_STATES];
    float theta_topo[NUM_TOPO_PARAMS];
    
    static float sigmoid(float x);
};

} // namespace discretum
