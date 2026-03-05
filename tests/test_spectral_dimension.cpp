#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include "geometry/spectral_dimension.hpp"
#include "utils/random.hpp"
#include "core/graph.hpp"
#include <cmath>

using namespace discretum;
using Catch::Matchers::WithinAbs;

TEST_CASE("Spectral dimension validation", "[spectral][critical]") {
    
    SECTION("1D chain") {
        // For a 1D chain, d_s should be 1.0
        // Use a long chain so walk doesn't hit boundary quickly
        DynamicGraph chain(500);
        for (int i = 0; i < 499; ++i)
            chain.add_edge(i, i + 1);
        
        // Many walkers, moderate steps (must be << chain length² to avoid boundary)
        auto result = compute_spectral_dimension_detailed(chain, 50000, 200, 0.05f, 0.4f, 42);
        
        REQUIRE_THAT(result.dimension, WithinAbs(1.0f, 0.3f));
    }
    
    SECTION("2D square lattice") {
        // d_s = 2.0. Use large lattice to avoid finite-size effects.
        // Walk length must be << L² where L=20, so steps << 400.
        DynamicGraph lattice_2d(400);
        for (int i = 0; i < 20; ++i) {
            for (int j = 0; j < 20; ++j) {
                int idx = i * 20 + j;
                if (j < 19) lattice_2d.add_edge(idx, idx + 1);
                if (i < 19) lattice_2d.add_edge(idx, idx + 20);
            }
        }
        
        auto result = compute_spectral_dimension_detailed(lattice_2d, 100000, 100, 0.05f, 0.5f, 42);
        
        REQUIRE_THAT(result.dimension, WithinAbs(2.0f, 0.4f));
    }
    
    SECTION("3D cubic lattice") {
        // d_s = 3.0. L=10 → L²=100, so steps should be < 100.
        auto graph = DynamicGraph::create_lattice_3d(10, 10, 10);
        
        auto result = compute_spectral_dimension_detailed(graph, 100000, 60, 0.05f, 0.5f, 42);
        
        REQUIRE_THAT(result.dimension, WithinAbs(3.0f, 1.0f));
    }
    
    SECTION("Complete graph") {
        // P(t) → 1/N exponentially fast, so no power-law decay.
        // Fitted dimension should be small / near zero.
        DynamicGraph graph(50);
        for (int i = 0; i < 50; ++i)
            for (int j = i + 1; j < 50; ++j)
                graph.add_edge(i, j);
        
        auto result = compute_spectral_dimension_detailed(graph, 5000, 50, 0.05f, 0.5f, 42);
        
        // Complete graph mixes instantly; d_s not well-defined but small
        REQUIRE(result.dimension < 1.0f);
    }
}

TEST_CASE("Random walk mechanics", "[spectral]") {
    SECTION("Isolated node") {
        DynamicGraph graph(1);
        PCG32 rng(42);
        
        auto returns = run_random_walk(graph, 0, 10, rng);
        
        // Isolated node always stays at origin
        REQUIRE(returns.size() == 10);
        for (uint32_t i = 0; i < 10; ++i) {
            REQUIRE(returns[i] == i + 1);
        }
    }
    
    SECTION("Two-node graph") {
        DynamicGraph graph(2);
        graph.add_edge(0, 1);
        PCG32 rng(42);
        
        auto returns = run_random_walk(graph, 0, 10000, rng);
        
        // On K_2, walker returns every 2 steps (even steps only)
        // Number of returns should be close to 5000
        REQUIRE(returns.size() > 4000);
        REQUIRE(returns.size() < 6000);
    }
    
    SECTION("Cycle graph return frequency") {
        // On C_n, the expected number of returns in T steps is ~ T/n for large T.
        // For n=10 and T=100000, expect ~10000 returns.
        DynamicGraph cycle(10);
        for (int i = 0; i < 10; ++i)
            cycle.add_edge(i, (i + 1) % 10);
        
        PCG32 rng(42);
        auto returns = run_random_walk(cycle, 0, 100000, rng);
        
        // Expected returns ≈ T/n = 10000. Allow wide margin.
        float num_returns = static_cast<float>(returns.size());
        REQUIRE(num_returns > 5000.0f);
        REQUIRE(num_returns < 15000.0f);
    }
}

TEST_CASE("Linear regression", "[spectral]") {
    SECTION("Perfect linear fit") {
        std::vector<float> x = {1, 2, 3, 4, 5};
        std::vector<float> y = {2, 4, 6, 8, 10};  // y = 2x
        
        auto [coeffs, r_squared] = linear_regression(x, y);
        auto [intercept, slope] = coeffs;
        
        REQUIRE_THAT(slope, WithinAbs(2.0f, 1e-6));
        REQUIRE_THAT(intercept, WithinAbs(0.0f, 1e-6));
        REQUIRE_THAT(r_squared, WithinAbs(1.0f, 1e-6));
    }
    
    SECTION("Noisy linear fit") {
        std::vector<float> x = {1, 2, 3, 4, 5};
        std::vector<float> y = {2.1f, 3.9f, 6.2f, 7.8f, 10.1f};
        
        auto [coeffs, r_squared] = linear_regression(x, y);
        auto [intercept, slope] = coeffs;
        
        REQUIRE_THAT(slope, WithinAbs(2.0f, 0.1f));
        REQUIRE(r_squared > 0.95f);
    }
    
    SECTION("Power law in log-log") {
        // y = x^(-1.5) => log(y) = -1.5 * log(x)
        std::vector<float> log_x, log_y;
        for (float x = 1.0f; x <= 100.0f; x += 1.0f) {
            log_x.push_back(std::log(x));
            log_y.push_back(std::log(std::pow(x, -1.5f)));
        }
        
        auto [coeffs, r_squared] = linear_regression(log_x, log_y);
        auto [intercept, slope] = coeffs;
        
        REQUIRE_THAT(slope, WithinAbs(-1.5f, 1e-6));
        REQUIRE_THAT(r_squared, WithinAbs(1.0f, 1e-6));
    }
}

TEST_CASE("Spectral dimension edge cases", "[spectral]") {
    SECTION("Empty graph") {
        DynamicGraph graph;
        auto result = compute_spectral_dimension(graph, 100, 100);
        
        REQUIRE(result.dimension == 0.0f);
        REQUIRE(result.time_points.empty());
    }
    
    SECTION("Disconnected graph") {
        // Two small paths — walkers trapped in small components
        DynamicGraph graph(10);
        graph.add_edge(0, 1);
        graph.add_edge(1, 2);
        graph.add_edge(2, 3);
        graph.add_edge(5, 6);
        graph.add_edge(6, 7);
        graph.add_edge(7, 8);
        
        auto result = compute_spectral_dimension(graph, 5000, 50);
        
        // Should produce some dimension estimate (finite components → d_s ~ 1)
        REQUIRE(result.dimension > 0.0f);
        REQUIRE(result.dimension < 3.0f);
    }
    
    SECTION("Very small graph") {
        DynamicGraph graph(3);
        graph.add_edge(0, 1);
        graph.add_edge(1, 2);
        
        auto result = compute_spectral_dimension(graph, 5000, 50);
        
        // Small graphs may not show clear scaling; allow noise
        REQUIRE(result.dimension >= -0.5f);
        REQUIRE(result.dimension <= 3.0f);
    }
}
