#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include "core/graph.hpp"
#include <vector>
#include <cstdint>
#include <cmath>
#include <set>

using Catch::Matchers::WithinAbs;

// Forward declarations of GPU functions
namespace discretum {
void gpu_compute_ollivier_ricci(
    const uint32_t* csr_offsets, const uint32_t* csr_neighbors,
    uint32_t num_nodes, uint32_t num_csr_entries,
    const uint32_t* edge_u, const uint32_t* edge_v,
    uint32_t num_edges, float alpha, float* curvatures);

void gpu_random_walk_trace(
    const uint32_t* csr_offsets, const uint32_t* csr_neighbors,
    uint32_t num_nodes, uint32_t num_csr_entries,
    const uint32_t* start_nodes, uint32_t num_walkers,
    uint32_t max_steps, uint64_t seed, uint32_t* return_counts);

void launch_evolve_states(
    const uint32_t* row_ptr, const uint32_t* col_idx,
    const uint8_t* states, const float* theta_state,
    uint32_t num_nodes, uint32_t num_edges_csr,
    uint8_t* new_states_out);
}

// Helper: convert DynamicGraph to CSR arrays
static void graph_to_csr(const discretum::DynamicGraph& graph,
                          std::vector<uint32_t>& offsets,
                          std::vector<uint32_t>& neighbors)
{
    uint32_t n = graph.num_nodes();
    offsets.resize(n + 1, 0);
    neighbors.clear();
    
    for (uint32_t i = 0; i < n; ++i) {
        offsets[i] = static_cast<uint32_t>(neighbors.size());
        auto nbrs = graph.neighbors(i);
        for (uint32_t nbr : nbrs)
            neighbors.push_back(nbr);
    }
    offsets[n] = static_cast<uint32_t>(neighbors.size());
}

// Helper: extract edge list from graph
static void graph_edges(const discretum::DynamicGraph& graph,
                         std::vector<uint32_t>& edge_u,
                         std::vector<uint32_t>& edge_v)
{
    edge_u.clear();
    edge_v.clear();
    std::set<std::pair<uint32_t,uint32_t>> seen;
    uint32_t n = graph.num_nodes();
    for (uint32_t i = 0; i < n; ++i) {
        for (uint32_t nbr : graph.neighbors(i)) {
            auto e = std::minmax(i, nbr);
            if (seen.insert(e).second) {
                edge_u.push_back(e.first);
                edge_v.push_back(e.second);
            }
        }
    }
}

TEST_CASE("GPU Ollivier-Ricci kernel", "[gpu][ollivier]") {
    using namespace discretum;
    
    SECTION("Complete graph K_5") {
        DynamicGraph graph(5);
        for (int i = 0; i < 5; ++i)
            for (int j = i + 1; j < 5; ++j)
                graph.add_edge(i, j);
        
        std::vector<uint32_t> offsets, nbrs, eu, ev;
        graph_to_csr(graph, offsets, nbrs);
        graph_edges(graph, eu, ev);
        
        uint32_t num_edges = static_cast<uint32_t>(eu.size());
        std::vector<float> curvatures(num_edges);
        
        gpu_compute_ollivier_ricci(
            offsets.data(), nbrs.data(), 5,
            static_cast<uint32_t>(nbrs.size()),
            eu.data(), ev.data(), num_edges, 0.0f,
            curvatures.data());
        
        // K_5 α=0: κ = (n-2)/(n-1) = 3/4
        float expected = 3.0f / 4.0f;
        for (uint32_t i = 0; i < num_edges; ++i) {
            REQUIRE_THAT(curvatures[i], WithinAbs(expected, 0.01f));
        }
    }
    
    SECTION("Cycle graph C_20 alpha=0") {
        DynamicGraph graph(20);
        for (int i = 0; i < 20; ++i)
            graph.add_edge(i, (i + 1) % 20);
        
        std::vector<uint32_t> offsets, nbrs, eu, ev;
        graph_to_csr(graph, offsets, nbrs);
        graph_edges(graph, eu, ev);
        
        uint32_t num_edges = static_cast<uint32_t>(eu.size());
        std::vector<float> curvatures(num_edges);
        
        gpu_compute_ollivier_ricci(
            offsets.data(), nbrs.data(), 20,
            static_cast<uint32_t>(nbrs.size()),
            eu.data(), ev.data(), num_edges, 0.0f,
            curvatures.data());
        
        for (uint32_t i = 0; i < num_edges; ++i) {
            REQUIRE_THAT(curvatures[i], WithinAbs(0.0f, 0.01f));
        }
    }
}

TEST_CASE("GPU random walk kernel", "[gpu][spectral]") {
    using namespace discretum;
    
    SECTION("Two-node graph return counts") {
        DynamicGraph graph(2);
        graph.add_edge(0, 1);
        
        std::vector<uint32_t> offsets, nbrs;
        graph_to_csr(graph, offsets, nbrs);
        
        uint32_t num_walkers = 10000;
        uint32_t max_steps = 100;
        std::vector<uint32_t> start_nodes(num_walkers, 0);
        std::vector<uint32_t> return_counts(max_steps + 1, 0);
        
        gpu_random_walk_trace(
            offsets.data(), nbrs.data(), 2,
            static_cast<uint32_t>(nbrs.size()),
            start_nodes.data(), num_walkers, max_steps, 42,
            return_counts.data());
        
        // On K_2 starting from 0: return at every even step.
        // return_counts[even] should be ~num_walkers, return_counts[odd] ~0
        for (uint32_t t = 2; t <= max_steps; t += 2) {
            REQUIRE(return_counts[t] > num_walkers * 0.9);
        }
        for (uint32_t t = 1; t <= max_steps; t += 2) {
            REQUIRE(return_counts[t] == 0);
        }
    }
}

TEST_CASE("GPU evolve states kernel", "[gpu][evolution]") {
    using namespace discretum;
    
    SECTION("State update matches CPU on path graph") {
        // Path P_5: 0-1-2-3-4
        // All nodes start in state 0, except node 2 which is in state 1
        DynamicGraph graph(5);
        for (int i = 0; i < 4; ++i) graph.add_edge(i, i + 1);
        graph.set_state(2, 1);
        
        std::vector<uint32_t> offsets, nbrs;
        graph_to_csr(graph, offsets, nbrs);
        
        // theta_state: 3x3 matrix, all zeros (so stability bias decides)
        float theta_state[9] = {};
        
        // Collect current states
        uint32_t n = graph.num_nodes();
        std::vector<uint8_t> states(n);
        for (uint32_t i = 0; i < n; ++i) states[i] = graph.get_state(i);
        
        // GPU result
        std::vector<uint8_t> gpu_new(n);
        launch_evolve_states(
            offsets.data(), nbrs.data(), states.data(), theta_state,
            n, static_cast<uint32_t>(nbrs.size()), gpu_new.data());
        
        // CPU result: manually compute expected states
        // With all theta_state=0 and stability bias +0.5 for current state,
        // every node keeps its current state (since all scores are 0 except current which is 0.5)
        for (uint32_t i = 0; i < n; ++i) {
            REQUIRE(gpu_new[i] == states[i]);
        }
    }
    
    SECTION("State update with non-zero weights") {
        // K_4 graph, all nodes in state 0
        DynamicGraph graph(4);
        for (int i = 0; i < 4; ++i)
            for (int j = i + 1; j < 4; ++j)
                graph.add_edge(i, j);
        
        std::vector<uint32_t> offsets, nbrs;
        graph_to_csr(graph, offsets, nbrs);
        
        uint32_t n = graph.num_nodes();
        std::vector<uint8_t> states(n, 0);
        
        // theta_state: make state 2 with majority 0 have very high weight
        // theta_state[s*3 + majority] = weight
        float theta_state[9] = {};
        theta_state[2 * 3 + 0] = 5.0f;  // state 2, majority 0 → score 5.0
        // state 0, majority 0 → score 0.0 + 0.5 (stability) = 0.5
        // So all nodes should transition to state 2
        
        std::vector<uint8_t> gpu_new(n);
        launch_evolve_states(
            offsets.data(), nbrs.data(), states.data(), theta_state,
            n, static_cast<uint32_t>(nbrs.size()), gpu_new.data());
        
        for (uint32_t i = 0; i < n; ++i) {
            REQUIRE(gpu_new[i] == 2);
        }
    }
}
