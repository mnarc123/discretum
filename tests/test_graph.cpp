#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include "core/graph.hpp"
#include <algorithm>
#include <cmath>

using namespace discretum;
using Catch::Matchers::WithinAbs;

TEST_CASE("DynamicGraph construction and basic operations", "[graph]") {
    SECTION("Empty graph") {
        DynamicGraph graph;
        REQUIRE(graph.num_nodes() == 0);
        REQUIRE(graph.num_edges() == 0);
        REQUIRE(graph.is_connected());
    }
    
    SECTION("Graph with nodes but no edges") {
        DynamicGraph graph(10);
        REQUIRE(graph.num_nodes() == 10);
        REQUIRE(graph.num_edges() == 0);
        REQUIRE_FALSE(graph.is_connected());
    }
    
    SECTION("Add and remove nodes") {
        DynamicGraph graph;
        uint32_t id1 = graph.add_node(1);
        uint32_t id2 = graph.add_node(2);
        
        REQUIRE(graph.num_nodes() == 2);
        REQUIRE(graph.get_state(id1) == 1);
        REQUIRE(graph.get_state(id2) == 2);
        
        graph.remove_node(id1);
        REQUIRE(graph.num_nodes() == 1);
        
        // Reuse ID
        uint32_t id3 = graph.add_node(3);
        REQUIRE(id3 == id1);  // ID should be reused
    }
    
    SECTION("Add and remove edges") {
        DynamicGraph graph(3);
        
        REQUIRE(graph.add_edge(0, 1));
        REQUIRE(graph.add_edge(1, 2));
        
        REQUIRE(graph.num_edges() == 2);
        REQUIRE(graph.has_edge(0, 1));
        REQUIRE(graph.has_edge(1, 0));  // Symmetric
        REQUIRE(graph.has_edge(1, 2));
        REQUIRE_FALSE(graph.has_edge(0, 2));
        
        REQUIRE(graph.degree(0) == 1);
        REQUIRE(graph.degree(1) == 2);
        REQUIRE(graph.degree(2) == 1);
        
        REQUIRE(graph.remove_edge(0, 1));
        REQUIRE(graph.num_edges() == 1);
        REQUIRE_FALSE(graph.has_edge(0, 1));
        REQUIRE(graph.has_edge(1, 2));
    }
    
    SECTION("Self-loops and duplicate edges") {
        DynamicGraph graph(2);
        
        REQUIRE_FALSE(graph.add_edge(0, 0));  // No self-loops
        REQUIRE(graph.add_edge(0, 1));
        REQUIRE(graph.add_edge(0, 1));  // Duplicate edge - should return true but not add
        
        REQUIRE(graph.num_edges() == 1);
    }
}

TEST_CASE("3D Lattice validation", "[graph][lattice]") {
    SECTION("Small 3D lattice") {
        auto graph = DynamicGraph::create_lattice_3d(10, 10, 10);
        
        REQUIRE(graph.num_nodes() == 1000);
        REQUIRE(graph.num_edges() == 2700);  // 3*10*10*10 - 3*10*10 = 3000 - 300 = 2700
        
        // Check interior node degree
        uint32_t interior_node = 5 + 10 * (5 + 10 * 5);  // (5,5,5)
        REQUIRE(graph.degree(interior_node) == 6);
        
        // Check corner node degree
        uint32_t corner_node = 0;  // (0,0,0)
        REQUIRE(graph.degree(corner_node) == 3);
        
        // Check edge node degree
        uint32_t edge_node = 5;  // (5,0,0)
        REQUIRE(graph.degree(edge_node) == 4);
        
        // Check face node degree
        uint32_t face_node = 5 + 10 * 5;  // (5,5,0)
        REQUIRE(graph.degree(face_node) == 5);
        
        // Check shortest path
        uint32_t dist = graph.shortest_path(0, 999);  // (0,0,0) to (9,9,9)
        REQUIRE(dist == 27);  // Manhattan distance
        
        REQUIRE(graph.is_connected());
        REQUIRE(graph.check_invariants());
    }
}

TEST_CASE("4D Lattice validation", "[graph][lattice]") {
    SECTION("Small 4D lattice") {
        auto graph = DynamicGraph::create_lattice_4d(5, 5, 5, 5);
        
        REQUIRE(graph.num_nodes() == 625);  // 5^4
        
        // Check interior node degree
        uint32_t interior_node = 2 + 5 * (2 + 5 * (2 + 5 * 2));  // (2,2,2,2)
        REQUIRE(graph.degree(interior_node) == 8);  // 4D hypercube has 8 neighbors
        
        // Check corner node degree
        uint32_t corner_node = 0;  // (0,0,0,0)
        REQUIRE(graph.degree(corner_node) == 4);
        
        REQUIRE(graph.is_connected());
        REQUIRE(graph.check_invariants());
    }
}

TEST_CASE("Random graphs", "[graph][random]") {
    SECTION("Erdős-Rényi graph") {
        auto graph = DynamicGraph::create_erdos_renyi(1000, 0.01f, 42);
        
        REQUIRE(graph.num_nodes() == 1000);
        
        // Expected edges: n(n-1)/2 * p = 1000*999/2 * 0.01 ≈ 4995
        // Check within reasonable bounds (binomial distribution)
        uint32_t num_edges = graph.num_edges();
        REQUIRE(num_edges > 4000);
        REQUIRE(num_edges < 6000);
        
        // Check degree distribution (should be approximately Poisson with λ = n*p = 10)
        std::vector<uint32_t> degree_hist(30, 0);
        for (uint32_t i = 0; i < 1000; ++i) {
            uint32_t deg = graph.degree(i);
            if (deg < degree_hist.size()) {
                degree_hist[deg]++;
            }
        }
        
        // Peak should be around degree 10
        auto max_it = std::max_element(degree_hist.begin(), degree_hist.end());
        size_t mode = std::distance(degree_hist.begin(), max_it);
        REQUIRE(mode >= 8);
        REQUIRE(mode <= 12);
        
        REQUIRE(graph.check_invariants());
    }
    
    SECTION("Random regular graph") {
        auto graph = DynamicGraph::create_random_regular(100, 4, 42);
        
        REQUIRE(graph.num_nodes() == 100);
        // Configuration model may reject some self-loops, so edges ≤ n*d/2
        REQUIRE(graph.num_edges() <= 200);
        REQUIRE(graph.num_edges() >= 180);  // Most edges should be present
        
        // Most nodes should have degree close to 4
        uint32_t deg4_count = 0;
        for (uint32_t i = 0; i < 100; ++i) {
            if (graph.degree(i) == 4) deg4_count++;
        }
        REQUIRE(deg4_count > 80);  // Most nodes are degree 4
        
        REQUIRE(graph.check_invariants());
    }
}

TEST_CASE("Graph algorithms", "[graph][algorithms]") {
    SECTION("Shortest path on a line graph") {
        DynamicGraph graph(10);
        for (uint32_t i = 0; i < 9; ++i) {
            graph.add_edge(i, i + 1);
        }
        
        REQUIRE(graph.shortest_path(0, 9) == 9);
        REQUIRE(graph.shortest_path(3, 7) == 4);
        REQUIRE(graph.shortest_path(5, 5) == 0);
        
        auto path = graph.shortest_path_with_path(0, 9);
        REQUIRE(path.size() == 10);
        for (uint32_t i = 0; i < 10; ++i) {
            REQUIRE(path[i] == i);
        }
    }
    
    SECTION("Connected components") {
        DynamicGraph graph(10);
        
        // Create three components
        graph.add_edge(0, 1);
        graph.add_edge(1, 2);
        
        graph.add_edge(3, 4);
        graph.add_edge(4, 5);
        graph.add_edge(5, 3);
        
        graph.add_edge(6, 7);
        
        // Nodes 8 and 9 are isolated
        
        auto components = graph.connected_components();
        REQUIRE(components.size() == 5);
        
        // Sort components by size for consistent testing
        std::sort(components.begin(), components.end(),
                  [](const auto& a, const auto& b) { return a.size() > b.size(); });
        
        REQUIRE(components[0].size() == 3);  // Component with nodes 0,1,2 or 3,4,5
        REQUIRE(components[1].size() == 3);
        REQUIRE(components[2].size() == 2);  // Component with nodes 6,7
        REQUIRE(components[3].size() == 1);  // Isolated node 8
        REQUIRE(components[4].size() == 1);  // Isolated node 9
    }
}

TEST_CASE("Topological operations", "[graph][topology]") {
    SECTION("Node splitting") {
        DynamicGraph graph(5);
        graph.add_edge(0, 1);
        graph.add_edge(0, 2);
        graph.add_edge(0, 3);
        graph.add_edge(0, 4);
        
        REQUIRE(graph.degree(0) == 4);
        
        SplitParams params;
        params.redistribution_ratio = 0.5f;
        params.preserve_state = true;
        
        auto [id1, id2] = graph.split_node(0, params);
        
        REQUIRE(graph.num_nodes() == 6);  // 5 - 1 + 2
        REQUIRE(graph.has_edge(id1, id2));  // New nodes connected
        
        // Check that edges were redistributed
        uint32_t total_degree = graph.degree(id1) + graph.degree(id2);
        REQUIRE(total_degree >= 5);  // Original 4 edges + 1 edge between new nodes
        
        REQUIRE(graph.check_invariants());
    }
    
    SECTION("Node merging") {
        DynamicGraph graph(5);
        graph.add_edge(0, 1);
        graph.add_edge(0, 2);
        graph.add_edge(1, 3);
        graph.add_edge(1, 4);
        
        graph.set_state(0, 2);
        graph.set_state(1, 4);
        
        uint32_t merged_id = graph.merge_nodes(0, 1);
        
        REQUIRE(merged_id != DynamicGraph::INVALID_ID);
        REQUIRE(graph.num_nodes() == 4);  // 5 - 2 + 1
        
        // Check merged node has all neighbors
        REQUIRE(graph.degree(merged_id) == 3);  // Connected to 2, 3, 4
        REQUIRE(graph.has_edge(merged_id, 2));
        REQUIRE(graph.has_edge(merged_id, 3));
        REQUIRE(graph.has_edge(merged_id, 4));
        
        // Check state averaging
        REQUIRE(graph.get_state(merged_id) == 3);  // (2 + 4) / 2
        
        REQUIRE(graph.check_invariants());
    }
    
    SECTION("Edge rewiring") {
        DynamicGraph graph(4);
        graph.add_edge(0, 1);
        graph.add_edge(1, 2);
        graph.add_edge(2, 3);
        
        REQUIRE(graph.rewire_edge(0, 1, 3));
        
        REQUIRE_FALSE(graph.has_edge(0, 1));
        REQUIRE(graph.has_edge(0, 3));
        REQUIRE(graph.has_edge(1, 2));
        REQUIRE(graph.has_edge(2, 3));
        
        REQUIRE(graph.check_invariants());
    }
}

TEST_CASE("Serialization", "[graph][io]") {
    SECTION("Save and load") {
        auto original = DynamicGraph::create_lattice_3d(5, 5, 5);
        
        // Set some states
        for (uint32_t i = 0; i < 125; ++i) {
            original.set_state(i, i % 3);
        }
        
        // Save
        std::string filename = "/tmp/test_graph.bin";
        original.save(filename);
        
        // Load
        auto loaded = DynamicGraph::load(filename);
        
        // Verify
        REQUIRE(loaded.num_nodes() == original.num_nodes());
        REQUIRE(loaded.num_edges() == original.num_edges());
        
        // Check states
        for (uint32_t i = 0; i < 125; ++i) {
            REQUIRE(loaded.get_state(i) == i % 3);
        }
        
        // Check structure
        for (uint32_t i = 0; i < 125; ++i) {
            REQUIRE(loaded.degree(i) == original.degree(i));
        }
        
        REQUIRE(loaded.check_invariants());
    }
}
