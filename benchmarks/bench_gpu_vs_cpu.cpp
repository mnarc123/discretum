#include <iostream>
#include <chrono>
#include <vector>
#include <set>
#include <cstdint>
#include <fmt/core.h>
#include "core/graph.hpp"
#include "geometry/ollivier_ricci.hpp"

using namespace discretum;

// Forward declarations of GPU functions
namespace discretum {
void gpu_compute_ollivier_ricci(
    const uint32_t* csr_offsets, const uint32_t* csr_neighbors,
    uint32_t num_nodes, uint32_t num_csr_entries,
    const uint32_t* edge_u, const uint32_t* edge_v,
    uint32_t num_edges, float alpha, float* curvatures);

void launch_evolve_states(
    const uint32_t* row_ptr, const uint32_t* col_idx,
    const uint8_t* states, const float* theta_state,
    uint32_t num_nodes, uint32_t num_edges_csr,
    uint8_t* new_states_out);
}

static void graph_to_csr(const DynamicGraph& graph,
                          std::vector<uint32_t>& offsets,
                          std::vector<uint32_t>& neighbors)
{
    uint32_t n = graph.num_nodes();
    offsets.resize(n + 1, 0);
    neighbors.clear();
    for (uint32_t i = 0; i < n; ++i) {
        offsets[i] = static_cast<uint32_t>(neighbors.size());
        for (uint32_t nbr : graph.neighbors(i))
            neighbors.push_back(nbr);
    }
    offsets[n] = static_cast<uint32_t>(neighbors.size());
}

static void graph_edges(const DynamicGraph& graph,
                         std::vector<uint32_t>& eu, std::vector<uint32_t>& ev)
{
    eu.clear(); ev.clear();
    std::set<std::pair<uint32_t,uint32_t>> seen;
    for (uint32_t i = 0; i < graph.num_nodes(); ++i)
        for (uint32_t nbr : graph.neighbors(i)) {
            auto e = std::minmax(i, nbr);
            if (seen.insert(e).second) { eu.push_back(e.first); ev.push_back(e.second); }
        }
}

static void bench_ollivier_cpu_vs_gpu(const std::string& name, const DynamicGraph& graph) {
    float alpha = 0.5f;
    
    // CPU
    auto t0 = std::chrono::high_resolution_clock::now();
    float cpu_avg = compute_average_ollivier_ricci(graph, alpha);
    auto t1 = std::chrono::high_resolution_clock::now();
    double cpu_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    
    // GPU
    std::vector<uint32_t> offsets, nbrs, eu, ev;
    graph_to_csr(graph, offsets, nbrs);
    graph_edges(graph, eu, ev);
    uint32_t ne = static_cast<uint32_t>(eu.size());
    std::vector<float> curvatures(ne);
    
    auto t2 = std::chrono::high_resolution_clock::now();
    gpu_compute_ollivier_ricci(offsets.data(), nbrs.data(), graph.num_nodes(),
                                static_cast<uint32_t>(nbrs.size()),
                                eu.data(), ev.data(), ne, alpha, curvatures.data());
    auto t3 = std::chrono::high_resolution_clock::now();
    double gpu_ms = std::chrono::duration<double, std::milli>(t3 - t2).count();
    
    float gpu_avg = 0.0f;
    for (float c : curvatures) gpu_avg += c;
    gpu_avg /= ne;
    
    fmt::print("  {}: CPU {:.1f}ms (κ={:.4f}), GPU {:.1f}ms (κ={:.4f}), speedup {:.1f}x\n",
               name, cpu_ms, cpu_avg, gpu_ms, gpu_avg, cpu_ms / std::max(gpu_ms, 0.01));
}

static void bench_evolve_cpu_vs_gpu(const std::string& name, const DynamicGraph& graph) {
    std::vector<uint32_t> offsets, nbrs;
    graph_to_csr(graph, offsets, nbrs);
    
    uint32_t n = graph.num_nodes();
    std::vector<uint8_t> states(n, 0);
    for (uint32_t i = 0; i < n; ++i) states[i] = graph.get_state(i);
    
    float theta[9] = {0.1f, 0.2f, -0.1f, 0.3f, -0.2f, 0.1f, -0.3f, 0.2f, 0.4f};
    
    // CPU
    auto t0 = std::chrono::high_resolution_clock::now();
    std::vector<uint8_t> cpu_new(n);
    for (uint32_t i = 0; i < n; ++i) {
        uint32_t start = offsets[i], end = offsets[i + 1];
        if (start == end) { cpu_new[i] = states[i]; continue; }
        int counts[3] = {};
        for (uint32_t e = start; e < end; ++e) {
            uint8_t s = states[nbrs[e]];
            if (s < 3) counts[s]++;
        }
        int majority = 0;
        for (int s = 1; s < 3; ++s) if (counts[s] > counts[majority]) majority = s;
        float best = -1e30f; int best_s = states[i];
        for (int s = 0; s < 3; ++s) {
            float score = theta[s * 3 + majority];
            if (s == states[i]) score += 0.5f;
            if (score > best) { best = score; best_s = s; }
        }
        cpu_new[i] = static_cast<uint8_t>(best_s);
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    double cpu_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    
    // GPU
    std::vector<uint8_t> gpu_new(n);
    auto t2 = std::chrono::high_resolution_clock::now();
    launch_evolve_states(offsets.data(), nbrs.data(), states.data(), theta,
                          n, static_cast<uint32_t>(nbrs.size()), gpu_new.data());
    auto t3 = std::chrono::high_resolution_clock::now();
    double gpu_ms = std::chrono::duration<double, std::milli>(t3 - t2).count();
    
    // Verify match
    int mismatches = 0;
    for (uint32_t i = 0; i < n; ++i) if (cpu_new[i] != gpu_new[i]) mismatches++;
    
    fmt::print("  {}: CPU {:.2f}ms, GPU {:.2f}ms, speedup {:.1f}x, mismatches: {}\n",
               name, cpu_ms, gpu_ms, cpu_ms / std::max(gpu_ms, 0.01), mismatches);
}

int main() {
    fmt::print("=== GPU vs CPU Benchmark ===\n\n");
    
    fmt::print("Ollivier-Ricci curvature:\n");
    for (int n : {20, 30}) {
        DynamicGraph g(n);
        for (int i = 0; i < n; ++i)
            for (int j = i + 1; j < n; ++j)
                g.add_edge(i, j);
        bench_ollivier_cpu_vs_gpu(fmt::format("K_{}", n), g);
    }
    for (int s : {4, 6}) {
        auto g = DynamicGraph::create_lattice_3d(s, s, s);
        bench_ollivier_cpu_vs_gpu(fmt::format("{}^3 lattice", s), g);
    }
    
    fmt::print("\nState evolution:\n");
    for (int s : {10, 20, 30}) {
        auto g = DynamicGraph::create_lattice_3d(s, s, s);
        bench_evolve_cpu_vs_gpu(fmt::format("{}^3 lattice", s), g);
    }
    
    return 0;
}
