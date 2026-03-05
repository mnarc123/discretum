#include <iostream>
#include <chrono>
#include <fmt/core.h>
#include "core/graph.hpp"
#include "geometry/ollivier_ricci.hpp"

using namespace discretum;

static void bench_ollivier(const std::string& name, const DynamicGraph& graph, float alpha) {
    auto t0 = std::chrono::high_resolution_clock::now();
    float avg = compute_average_ollivier_ricci(graph, alpha);
    auto t1 = std::chrono::high_resolution_clock::now();
    
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    fmt::print("  {} (N={}, E={}, α={:.1f}): avg κ = {:.4f}, {:.1f} ms\n",
               name, graph.num_nodes(), graph.num_edges(), alpha, avg, ms);
}

int main() {
    fmt::print("=== Ollivier-Ricci Benchmark ===\n\n");
    
    // Complete graphs
    for (int n : {10, 20, 30, 50}) {
        DynamicGraph g(n);
        for (int i = 0; i < n; ++i)
            for (int j = i + 1; j < n; ++j)
                g.add_edge(i, j);
        bench_ollivier(fmt::format("K_{}", n), g, 0.5f);
    }
    
    fmt::print("\n");
    
    // Lattices
    for (int s : {4, 6, 8}) {
        auto g = DynamicGraph::create_lattice_3d(s, s, s);
        bench_ollivier(fmt::format("{}^3 lattice", s), g, 0.5f);
    }
    
    fmt::print("\n");
    
    // Cycle graphs
    for (int n : {50, 100, 200}) {
        DynamicGraph g(n);
        for (int i = 0; i < n; ++i) g.add_edge(i, (i + 1) % n);
        bench_ollivier(fmt::format("C_{}", n), g, 0.0f);
    }
    
    return 0;
}
