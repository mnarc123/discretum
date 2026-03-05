#pragma once

#include "core/graph.hpp"
#include "automaton/rule_parametric.hpp"
#include "utils/random.hpp"
#include <vector>
#include <functional>

namespace discretum {

/**
 * @brief Evolves a graph under a parametric cellular automaton rule.
 *
 * Manages the simulation loop: applies the rule for a specified number
 * of steps, optionally recording snapshots and computing observables.
 */
struct EvolutionConfig {
    uint32_t num_steps = 100;
    uint32_t snapshot_interval = 0;  // 0 = no snapshots
    uint64_t seed = 42;
    uint32_t max_nodes = 0;  // 0 = unlimited; abort if node count exceeds this
};

struct EvolutionResult {
    uint32_t final_num_nodes;
    uint32_t final_num_edges;
    float    final_avg_degree;
    uint32_t steps_completed;
    std::vector<uint32_t> node_count_history;
    std::vector<uint32_t> edge_count_history;
};

class Evolution {
public:
    Evolution() = default;
    Evolution(DynamicGraph graph, ParametricRule rule, EvolutionConfig config = {});
    
    /// Run the full evolution
    EvolutionResult run();
    
    /// Run a single step
    void step();
    
    /// Access the current graph
    const DynamicGraph& get_graph() const { return graph_; }
    DynamicGraph& get_graph() { return graph_; }
    
    /// Access the rule
    const ParametricRule& get_rule() const { return rule_; }
    
    /// Current step counter
    uint32_t current_step() const { return step_count_; }
    
private:
    DynamicGraph graph_;
    ParametricRule rule_;
    EvolutionConfig config_;
    PCG32 rng_{42};
    uint32_t step_count_ = 0;
};

} // namespace discretum
