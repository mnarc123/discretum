#include "automaton/evolution.hpp"
#include "utils/logging.hpp"
#include "utils/timer.hpp"

namespace discretum {

Evolution::Evolution(DynamicGraph graph, ParametricRule rule, EvolutionConfig config)
    : graph_(std::move(graph))
    , rule_(std::move(rule))
    , config_(config)
    , rng_(config.seed)
    , step_count_(0)
{
}

void Evolution::step() {
    rule_.apply(graph_, rng_);
    step_count_++;
}

EvolutionResult Evolution::run() {
    TIMED_SCOPE("evolution_run");
    
    EvolutionResult result{};
    result.node_count_history.reserve(config_.num_steps + 1);
    result.edge_count_history.reserve(config_.num_steps + 1);
    
    // Record initial state
    result.node_count_history.push_back(graph_.num_nodes());
    result.edge_count_history.push_back(graph_.num_edges());
    
    for (uint32_t s = 0; s < config_.num_steps; ++s) {
        step();
        
        result.node_count_history.push_back(graph_.num_nodes());
        result.edge_count_history.push_back(graph_.num_edges());
        
        // Abort if graph has grown past the safety cap
        if (config_.max_nodes > 0 && graph_.num_nodes() > config_.max_nodes) {
            spdlog::warn("Evolution aborted at step {}: {} nodes exceeds max_nodes={}",
                         s + 1, graph_.num_nodes(), config_.max_nodes);
            break;
        }
        
        if (config_.snapshot_interval > 0 && (s + 1) % config_.snapshot_interval == 0) {
            spdlog::info("Evolution step {}/{}: nodes={}, edges={}", 
                         s + 1, config_.num_steps, 
                         graph_.num_nodes(), graph_.num_edges());
        }
    }
    
    result.final_num_nodes = graph_.num_nodes();
    result.final_num_edges = graph_.num_edges();
    result.steps_completed = step_count_;
    
    uint32_t total_degree = 0;
    for (uint32_t i = 0; i < graph_.num_nodes(); ++i)
        total_degree += graph_.degree(i);
    result.final_avg_degree = graph_.num_nodes() > 0 
        ? static_cast<float>(total_degree) / graph_.num_nodes() 
        : 0.0f;
    
    return result;
}

} // namespace discretum
