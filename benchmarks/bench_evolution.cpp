#include <iostream>
#include <chrono>
#include <fmt/core.h>
#include "core/graph.hpp"
#include "automaton/rule_parametric.hpp"
#include "automaton/evolution.hpp"

using namespace discretum;

static double bench_evolution(int side, int steps) {
    std::vector<float> params(ParametricRule::TOTAL_PARAMS, 0.1f);
    ParametricRule rule(params);
    auto graph = DynamicGraph::create_lattice_3d(side, side, side);
    
    EvolutionConfig cfg;
    cfg.num_steps = steps;
    cfg.seed = 42;
    
    auto t0 = std::chrono::high_resolution_clock::now();
    Evolution evo(std::move(graph), std::move(rule), cfg);
    evo.run();
    auto t1 = std::chrono::high_resolution_clock::now();
    
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    const auto& g = evo.get_graph();
    fmt::print("  {}^3 lattice ({} nodes), {} steps → {} nodes, {} edges: {:.1f} ms\n",
               side, side*side*side, steps, g.num_nodes(), g.num_edges(), ms);
    return ms;
}

int main() {
    fmt::print("=== Evolution Benchmark ===\n\n");
    
    bench_evolution(5, 50);
    bench_evolution(8, 50);
    bench_evolution(10, 50);
    bench_evolution(10, 100);
    bench_evolution(15, 50);
    
    return 0;
}
