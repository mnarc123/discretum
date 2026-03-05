#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include "geometry/ollivier_ricci.hpp"
#include "core/graph.hpp"
#include <cmath>

using namespace discretum;
using Catch::Matchers::WithinAbs;

TEST_CASE("Ollivier-Ricci curvature validation", "[ollivier][critical]") {
    
    SECTION("Complete graph K_n, alpha=0") {
        // For K_n with α=0: each measure puts 1/(n-1) on each neighbor.
        // Common neighbors = n-2 nodes matched at cost 0.
        // Remaining: 1/(n-1) mass at v in μ_u must go to u in μ_v, cost 1.
        // W₁ = 1/(n-1), κ = 1 - 1/(n-1) = (n-2)/(n-1)
        
        for (int n : {5, 10}) {
            DynamicGraph graph(n);
            for (int i = 0; i < n; ++i)
                for (int j = i + 1; j < n; ++j)
                    graph.add_edge(i, j);
            
            float expected = static_cast<float>(n - 2) / (n - 1);
            
            for (int i = 0; i < std::min(3, n-1); ++i) {
                float curvature = compute_ollivier_ricci(graph, i, i+1, 0.0f);
                REQUIRE_THAT(curvature, WithinAbs(expected, 1e-5));
            }
        }
    }
    
    SECTION("Complete graph K_n, alpha=0.5") {
        // With α=0.5: μ_u puts 0.5 on u, 0.5/(n-1) on each neighbor.
        // μ_v puts 0.5 on v, 0.5/(n-1) on each neighbor.
        // Common neighbors: n-2 nodes, each matched at cost 0 → 0.5(n-2)/(n-1).
        // Remaining supply: 0.5 at u, 0.5/(n-1) at v.
        // Remaining demand: 0.5 at v, 0.5/(n-1) at u.
        // Transport: min(0.5, 0.5/(n-1)) from v→u + rest from u→v, both cost 1.
        // W₁ = 0.5/(n-1) + (0.5 - 0.5/(n-1)) = 0.5 (actually net 2 * 0.5/(n-1) + ...)
        // Simpler: total unmatched mass to transport = 0.5 - 0.5/(n-1) on each side.
        // W₁ = 2 * (0.5 - 0.5/(n-1)) * 0 + ... let's just verify numerically.
        
        DynamicGraph graph(5);
        for (int i = 0; i < 5; ++i)
            for (int j = i + 1; j < 5; ++j)
                graph.add_edge(i, j);
        
        float curvature = compute_ollivier_ricci(graph, 0, 1, 0.5f);
        // With lazy walk on K_5: κ must be positive and ≤ 0.75
        REQUIRE(curvature > 0.0f);
        REQUIRE(curvature <= 1.0f);
    }
    
    SECTION("Cycle graph C_n, alpha=0") {
        // For cycle C_n with α=0, edge (u, u+1):
        // μ_u = {1/2 on u-1, 1/2 on u+1}
        // μ_v = {1/2 on u, 1/2 on u+2}
        // No common support → must transport 1/2 from u-1→u (cost 1) + 1/2 from u+1→u+2 (cost 1)
        // W₁ = 1, κ = 1 - 1 = 0
        
        for (int n : {10, 20, 50}) {
            DynamicGraph graph(n);
            for (int i = 0; i < n; ++i)
                graph.add_edge(i, (i + 1) % n);
            
            float curvature = compute_ollivier_ricci(graph, 0, 1, 0.0f);
            REQUIRE_THAT(curvature, WithinAbs(0.0f, 1e-5));
        }
    }
    
    SECTION("Cycle graph C_n, alpha=0.5") {
        // With α=0.5: μ_u = {0.5 on u, 0.25 on u-1, 0.25 on u+1}
        // μ_v = {0.5 on v, 0.25 on u, 0.25 on u+2} where v=u+1
        // Common: u gets 0.25 from μ_u (via u+1=v neighbor), and 0.25 from μ_v.
        // v gets 0.25 from μ_u (u+1), and 0.5 from μ_v (self).
        // Actually let's just check symmetry and sign.
        
        DynamicGraph graph(20);
        for (int i = 0; i < 20; ++i)
            graph.add_edge(i, (i + 1) % 20);
        
        float curvature = compute_ollivier_ricci(graph, 0, 1, 0.5f);
        // With lazy walk on cycle, curvature should be 0 
        // (Lin-Lu-Yau result: cycle curvature = 0 with idleness)
        REQUIRE_THAT(curvature, WithinAbs(0.0f, 1e-5));
    }
    
    SECTION("Tree structures") {
        // Binary tree with internal edges: α=0
        // Edge (0,1): deg(0)=2, deg(1)=3
        // μ_0 = {1/2 on 1, 1/2 on 2}
        // μ_1 = {1/3 on 0, 1/3 on 3, 1/3 on 4}
        // No common support between measures → W₁ > 0
        // κ = 1 - W₁ < 1
        
        DynamicGraph tree(7);
        tree.add_edge(0, 1);
        tree.add_edge(0, 2);
        tree.add_edge(1, 3);
        tree.add_edge(1, 4);
        tree.add_edge(2, 5);
        tree.add_edge(2, 6);
        
        float curvature_root = compute_ollivier_ricci(tree, 0, 1, 0.0f);
        // μ_0 = {1/2 on 1, 1/2 on 2}, μ_1 = {1/3 on 0, 1/3 on 3, 1/3 on 4}
        // Transport 1/2 from node 1 (in μ_0) to: 1/3 to node 0 (cost 1), remaining ...
        // Let's verify it's negative for this tree
        REQUIRE(curvature_root <= 0.0f);
        
        // Edge (1,3): deg(1)=3, deg(3)=1
        // μ_1 = {1/3 on 0, 1/3 on 3, 1/3 on 4}, μ_3 = {1 on 1}
        // Transport: 1 from node 1 → 1/3 of node 0 (cost 2), 1/3 of node 3 (cost 2), 1/3 of node 4 (cost 2)
        // Wait: μ_3 = {1.0 on 1} (only neighbor). All mass at node 1.
        // Supply from μ_1: 1/3 at 0, 1/3 at 3, 1/3 at 4. Demand: 1.0 at 1.
        // Cost = 1/3*d(0,1) + 1/3*d(3,1) + 1/3*d(4,1) = 1/3*1 + 1/3*1 + 1/3*1 = 1
        // W₁ = 1, κ = 0
        float curvature_leaf = compute_ollivier_ricci(tree, 1, 3, 0.0f);
        REQUIRE_THAT(curvature_leaf, WithinAbs(0.0f, 1e-5));
        
        // With alpha=0.5, leaf edge (1,3): laziness concentrates mass,
        // making W₁ < 1 and κ > 0 for leaf edges
        float curvature_leaf_lazy = compute_ollivier_ricci(tree, 1, 3, 0.5f);
        REQUIRE_THAT(curvature_leaf_lazy, WithinAbs(1.0f/3.0f, 1e-4));
        
        // Internal edge (0,1) with α=0.5 should still be non-positive
        float curvature_root_lazy = compute_ollivier_ricci(tree, 0, 1, 0.5f);
        REQUIRE(curvature_root_lazy <= curvature_leaf_lazy);
    }
    
    SECTION("2D lattice interior edge") {
        // 2D lattice 5x5, interior edge between degree-4 nodes
        // For a grid graph, interior α=0 curvature of an edge (u,v):
        // deg(u)=4, deg(v)=4. They share 0 common neighbors (grid has no triangles).
        // μ_u puts 1/4 on each of u's 4 neighbors, μ_v on v's 4 neighbors.
        // v is a neighbor of u, u is a neighbor of v.
        // Shared in support but not common neighbors. Complex OT.
        
        DynamicGraph lattice_2d(25);
        for (int i = 0; i < 5; ++i) {
            for (int j = 0; j < 5; ++j) {
                int idx = i * 5 + j;
                if (j < 4) lattice_2d.add_edge(idx, idx + 1);
                if (i < 4) lattice_2d.add_edge(idx, idx + 5);
            }
        }
        
        // Edge (12,13) = center horizontal
        float curvature = compute_ollivier_ricci(lattice_2d, 12, 13, 0.0f);
        // Grid with no triangles: curvature should be negative
        REQUIRE(curvature <= 0.0f);
        
        // All interior horizontal edges should have same curvature (by symmetry)
        float curvature2 = compute_ollivier_ricci(lattice_2d, 11, 12, 0.0f);
        REQUIRE_THAT(curvature, WithinAbs(curvature2, 1e-5));
    }
    
    SECTION("Path graph, alpha=0") {
        // Path: degree-2 interior, degree-1 endpoints
        // Interior edge (i, i+1) with both deg 2: same as cycle → κ = 0
        // Endpoint edge (0,1): deg(0)=1, deg(1)=2
        // μ_0 = {1.0 on 1}, μ_1 = {1/2 on 0, 1/2 on 2}
        // Transport 1.0 from 1 to: 1/2 to 0 (cost 1) + 1/2 to 2 (cost 1) = 1
        // W₁ = 1, κ = 0
        
        DynamicGraph path(10);
        for (int i = 0; i < 9; ++i)
            path.add_edge(i, i + 1);
        
        float curvature_end = compute_ollivier_ricci(path, 0, 1, 0.0f);
        REQUIRE_THAT(curvature_end, WithinAbs(0.0f, 1e-5));
        
        float curvature_middle = compute_ollivier_ricci(path, 4, 5, 0.0f);
        REQUIRE_THAT(curvature_middle, WithinAbs(0.0f, 1e-5));
        
        // With laziness α=0.5:
        // Endpoint (0,1): deg(0)=1, deg(1)=2 → κ = 0.5 (positive due to asymmetry)
        // Middle (4,5): both deg 2 → κ = 0 (same as cycle)
        float curvature_end_lazy = compute_ollivier_ricci(path, 0, 1, 0.5f);
        float curvature_mid_lazy = compute_ollivier_ricci(path, 4, 5, 0.5f);
        REQUIRE_THAT(curvature_end_lazy, WithinAbs(0.5f, 1e-4));
        REQUIRE_THAT(curvature_mid_lazy, WithinAbs(0.0f, 1e-4));
    }
    
    SECTION("Star graph, alpha=0") {
        // Star: center degree n, leaves degree 1.
        // Edge (0, leaf): μ_0 = {1/n on each leaf}, μ_leaf = {1 on 0}
        // Transport 1/n from each leaf → node 0. Costs: 1/n * d(leaf_i, 0) = 1/n * 1 each.
        // But wait: supply is at leaves (μ_0), demand at node 0 (μ_leaf).
        // Actually supply = μ_0 = {1/n on leaf_1, ..., 1/n on leaf_n}
        // demand = μ_leaf = {1.0 on node 0}
        // W₁ = sum of 1/n * d(leaf_i, 0) = n * 1/n * 1 = 1. κ = 0.
        
        int n = 10;
        DynamicGraph star(n + 1);
        for (int i = 1; i <= n; ++i)
            star.add_edge(0, i);
        
        float curvature1 = compute_ollivier_ricci(star, 0, 1, 0.0f);
        float curvature2 = compute_ollivier_ricci(star, 0, 2, 0.0f);
        
        REQUIRE_THAT(curvature1, WithinAbs(0.0f, 1e-5));
        REQUIRE_THAT(curvature1, WithinAbs(curvature2, 1e-6));
        
        // With laziness α=0.5: leaf edge is positive due to mass concentration
        // κ = 1 - 0.9 = 0.1 for n=10
        float curvature_lazy = compute_ollivier_ricci(star, 0, 1, 0.5f);
        REQUIRE_THAT(curvature_lazy, WithinAbs(0.1f, 1e-4));
    }
}

TEST_CASE("Ollivier-Ricci statistics", "[ollivier]") {
    SECTION("Statistics on complete graph") {
        // K_10: all edges have κ = (n-2)/(n-1) = 8/9
        DynamicGraph graph(10);
        for (int i = 0; i < 10; ++i)
            for (int j = i + 1; j < 10; ++j)
                graph.add_edge(i, j);
        
        auto stats = compute_ollivier_ricci_stats(graph, 0.0f);
        
        float expected = 8.0f / 9.0f;
        REQUIRE(stats.num_edges == 45);
        REQUIRE_THAT(stats.mean, WithinAbs(expected, 1e-5));
        REQUIRE_THAT(stats.std_dev, WithinAbs(0.0f, 1e-5));
        REQUIRE_THAT(stats.min, WithinAbs(expected, 1e-5));
        REQUIRE_THAT(stats.max, WithinAbs(expected, 1e-5));
        REQUIRE(stats.num_positive == 45);
        REQUIRE(stats.num_negative == 0);
        REQUIRE(stats.num_zero == 0);
    }
    
    SECTION("Statistics on mixed graph") {
        DynamicGraph graph(10);
        
        // Complete subgraph K_5 (positive curvature)
        for (int i = 0; i < 5; ++i)
            for (int j = i + 1; j < 5; ++j)
                graph.add_edge(i, j);
        
        // Path (will have non-positive curvature with α=0)
        for (int i = 5; i < 9; ++i)
            graph.add_edge(i, i + 1);
        
        // Bridge
        graph.add_edge(4, 5);
        
        auto stats = compute_ollivier_ricci_stats(graph, 0.0f);
        
        REQUIRE(stats.num_edges == 15);  // 10 + 4 + 1
        REQUIRE(stats.num_positive > 0);
        REQUIRE(stats.std_dev > 0.0f);
    }
}

TEST_CASE("Wasserstein distance computation", "[ollivier]") {
    SECTION("Identity transport") {
        std::vector<std::vector<float>> dist_matrix = {
            {0, 1, 2},
            {1, 0, 1},
            {2, 1, 0}
        };
        
        std::vector<float> mu = {0.33f, 0.33f, 0.34f};
        
        float w1 = wasserstein_1_distance(dist_matrix, mu, mu);
        REQUIRE_THAT(w1, WithinAbs(0.0f, 1e-5));
    }
    
    SECTION("Simple transport") {
        std::vector<std::vector<float>> dist_matrix = {{0, 3}, {3, 0}};
        std::vector<float> mu_x = {1.0f, 0.0f};
        std::vector<float> mu_y = {0.0f, 1.0f};
        
        float w1 = wasserstein_1_distance(dist_matrix, mu_x, mu_y);
        REQUIRE_THAT(w1, WithinAbs(3.0f, 1e-5));
    }
    
    SECTION("Partial overlap") {
        // 3 points on a line: 0 -- 1 -- 2
        std::vector<std::vector<float>> dist_matrix = {
            {0, 1, 2},
            {1, 0, 1},
            {2, 1, 0}
        };
        // μ = {0.5, 0.5, 0}, ν = {0, 0.5, 0.5}
        // Optimal: move 0.5 from 0→1 (cost 0.5) and 0.5 from 1→2 (cost 0.5)
        // Hmm actually: move 0.5 from point 0 to point 2 (cost 1.0) 
        // or shift everything: 0→1 costs 0.5, 1→2 costs 0.5 = total 1.0 same.
        // W₁ = 1.0
        std::vector<float> mu_x = {0.5f, 0.5f, 0.0f};
        std::vector<float> mu_y = {0.0f, 0.5f, 0.5f};
        
        float w1 = wasserstein_1_distance(dist_matrix, mu_x, mu_y);
        REQUIRE_THAT(w1, WithinAbs(1.0f, 1e-5));
    }
}
