#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include "search/fitness.hpp"
#include "search/ensemble.hpp"
#include "core/graph.hpp"
#include "geometry/spectral_dimension.hpp"

using namespace discretum;

TEST_CASE("Fitness function", "[fitness]") {
    SECTION("Degenerate graph gets large penalty") {
        DynamicGraph graph(2);
        FitnessParams params;
        double fitness = compute_fitness(graph, params);
        REQUIRE(fitness <= -50.0);
    }
    
    SECTION("Connected graph gets finite fitness") {
        DynamicGraph graph(20);
        for (int i = 0; i < 19; ++i) graph.add_edge(i, i + 1);
        // Close the chain into a cycle
        graph.add_edge(0, 19);
        
        FitnessParams params;
        params.target_dimension = 1.0;
        params.target_size = 20.0;
        params.spectral_walkers = 500;
        params.spectral_steps = 50;
        
        double fitness = compute_fitness(graph, params);
        // Should be finite and negative (not perfect match)
        REQUIRE(fitness > -50.0);
        REQUIRE(fitness <= 0.0);
    }
    
    SECTION("Detailed breakdown sums to total") {
        DynamicGraph graph(10);
        for (int i = 0; i < 9; ++i) graph.add_edge(i, i + 1);
        graph.add_edge(0, 9);
        
        FitnessParams params;
        params.target_dimension = 1.0;
        params.target_size = 10.0;
        params.spectral_walkers = 200;
        params.spectral_steps = 30;
        
        auto fb = compute_fitness_detailed(graph, params);
        double sum = fb.ricci_term + fb.dimension_term + fb.connectivity_term
                   + fb.degree_reg_term + fb.size_term;
        REQUIRE_THAT(fb.total, Catch::Matchers::WithinAbs(sum, 1e-10));
        REQUIRE(fb.is_connected);
    }
    
    SECTION("Disconnected graph penalized") {
        DynamicGraph graph(10);
        // Two disconnected components
        for (int i = 0; i < 4; ++i) graph.add_edge(i, i + 1);
        for (int i = 5; i < 9; ++i) graph.add_edge(i, i + 1);
        
        FitnessParams params;
        params.spectral_walkers = 200;
        params.spectral_steps = 30;
        
        auto fb = compute_fitness_detailed(graph, params);
        REQUIRE(!fb.is_connected);
        REQUIRE(fb.connectivity_term < 0.0);
    }
}

// ════════════════════════════════════════════════════════════════
// V2 FITNESS FUNCTION TESTS
// ════════════════════════════════════════════════════════════════

TEST_CASE("V2 Fitness: degenerate graph", "[fitness][v2]") {
    DynamicGraph graph(2);
    FitnessParamsV2 params;
    auto r = compute_fitness_v2(graph, params);
    REQUIRE(r.total <= -50.0);
    REQUIRE(!r.is_connected);
}

TEST_CASE("V2 Fitness: breakdown sums to total", "[fitness][v2]") {
    DynamicGraph graph(20);
    for (int i = 0; i < 19; ++i) graph.add_edge(i, i + 1);
    graph.add_edge(0, 19);

    FitnessParamsV2 params;
    params.target_dim = 1.0;
    params.n_initial = 20.0;
    params.spectral_walkers = 200;
    params.spectral_steps = 30;
    params.hausdorff_sources = 10;

    auto r = compute_fitness_v2(graph, params);
    double sum = r.f_hausdorff + r.f_curvature + r.f_spectral
               + r.f_connectivity + r.f_stability + r.f_regularity;
    REQUIRE_THAT(r.total, Catch::Matchers::WithinAbs(sum, 1e-10));
    REQUIRE(r.is_connected);
}

TEST_CASE("V2 Fitness: disconnected graph penalized", "[fitness][v2]") {
    DynamicGraph graph(10);
    for (int i = 0; i < 4; ++i) graph.add_edge(i, i + 1);
    for (int i = 5; i < 9; ++i) graph.add_edge(i, i + 1);

    FitnessParamsV2 params;
    params.spectral_walkers = 100;
    params.spectral_steps = 20;

    auto r = compute_fitness_v2(graph, params);
    REQUIRE(!r.is_connected);
    REQUIRE(r.n_components == 2);
    REQUIRE(r.f_connectivity < 0.0);
}

TEST_CASE("V2 Fitness: 3D lattice scores well for target_dim=3", "[fitness][v2]") {
    auto g = DynamicGraph::create_lattice_3d(6, 6, 6); // 216 nodes
    FitnessParamsV2 params;
    params.target_dim = 3.0;
    params.n_initial = 216.0;
    params.spectral_walkers = 500;
    params.spectral_steps = 50;
    params.hausdorff_sources = 20;
    params.curvature_samples = 100;

    auto r = compute_fitness_v2(g, params);
    REQUIRE(r.is_connected);
    // d_H should be in the ballpark of 3
    REQUIRE(r.d_H > 1.5);
    REQUIRE(r.d_H < 5.0);
    // Curvature on a regular lattice should be near zero
    REQUIRE(std::abs(r.mean_curvature) < 0.3);
    // f_hausdorff should not be too bad
    REQUIRE(r.f_hausdorff > -10.0);
}

TEST_CASE("Lattice 4D structure", "[graph][4d]") {
    auto g = DynamicGraph::create_lattice_4d(5, 5, 5, 5);
    REQUIRE(g.num_nodes() == 625);
    // Internal node (2,2,2,2): index = 2 + 5*2 + 25*2 + 125*2 = 312, degree 8
    REQUIRE(g.degree(312) == 8);
    // Corner node (0,0,0,0): degree 4
    REQUIRE(g.degree(0) == 4);
    // Must be connected
    auto comp = g.connected_components();
    REQUIRE(comp.size() == 1);
    // Edge count: 4 directions, each has L^3*(L-1) edges = 4 * 125 * 4 = 2000
    REQUIRE(g.num_edges() == 2000);
}

TEST_CASE("V2 Fitness: 4D lattice scores well for target_dim=4", "[fitness][v2][4d]") {
    auto g = DynamicGraph::create_lattice_4d(5, 5, 5, 5); // 625 nodes
    FitnessParamsV2 params;
    params.target_dim = 4.0;
    params.n_initial = 625.0;
    params.spectral_walkers = 500;
    params.spectral_steps = 50;
    params.hausdorff_sources = 20;
    params.curvature_samples = 100;

    auto r = compute_fitness_v2(g, params);
    REQUIRE(r.is_connected);
    REQUIRE(r.d_H > 2.0);
    REQUIRE(r.d_H < 6.0);
    // Lattice curvature should be near zero
    REQUIRE(std::abs(r.mean_curvature) < 0.3);
    // Total fitness should be finite and reasonable
    REQUIRE(r.total > -50.0);
}

TEST_CASE("V2 Fitness: d_s != d_H penalised via f_spectral", "[fitness][v2]") {
    // A cycle has d_H ~ 1 but d_s may differ
    DynamicGraph g(30);
    for (int i = 0; i < 30; ++i) g.add_edge(i, (i + 1) % 30);

    FitnessParamsV2 params;
    params.target_dim = 1.0;
    params.n_initial = 30.0;
    params.spectral_walkers = 200;
    params.spectral_steps = 30;
    params.hausdorff_sources = 10;
    params.curvature_samples = 30;

    auto r = compute_fitness_v2(g, params);
    REQUIRE(r.is_connected);
    // f_spectral should be non-positive (penalises d_s != d_H)
    REQUIRE(r.f_spectral <= 0.0);
}

// ════════════════════════════════════════════════════════════════
// SPECTRAL DIMENSION V2 TESTS
// ════════════════════════════════════════════════════════════════

TEST_CASE("Spectral dimension v2: 3D lattice L=8", "[spectral][v2]") {
    auto g = DynamicGraph::create_lattice_3d(8, 8, 8); // N=512
    auto result = compute_spectral_dimension_v2(g, 10000, 500, 42);
    REQUIRE(result.P_t.size() > 0);
    REQUIRE(result.d_eff_t.size() > 0);
    REQUIRE(result.time_pts.size() == result.P_t.size());
    // d_s should be in the ballpark of 3 (may be lower on small lattice)
    REQUIRE(result.d_s > 1.0);
    REQUIRE(result.d_s < 6.0);
    // Global fit should also give something reasonable
    REQUIRE(result.d_s_global_fit > 0.5);
}

TEST_CASE("Spectral dimension v2: 4D lattice L=5", "[spectral][v2][4d]") {
    auto g = DynamicGraph::create_lattice_4d(5, 5, 5, 5); // N=625
    auto result = compute_spectral_dimension_v2(g, 10000, 500, 42);
    REQUIRE(result.P_t.size() > 0);
    REQUIRE(result.d_eff_t.size() > 0);
    // d_s should be positive and in a reasonable range
    REQUIRE(result.d_s > 1.0);
    REQUIRE(result.d_s < 8.0);
}

TEST_CASE("Spectral dimension v2: cycle graph", "[spectral][v2]") {
    DynamicGraph g(50);
    for (int i = 0; i < 50; ++i) g.add_edge(i, (i + 1) % 50);
    auto result = compute_spectral_dimension_v2(g, 5000, 200, 42);
    REQUIRE(result.P_t.size() > 0);
    // Cycle has d_s ≈ 1
    REQUIRE(result.d_s > 0.3);
    REQUIRE(result.d_s < 3.0);
}

TEST_CASE("Spectral dimension v2: empty graph", "[spectral][v2]") {
    DynamicGraph g(5);
    auto result = compute_spectral_dimension_v2(g, 100, 50, 42);
    // No connected nodes → d_s = 0
    REQUIRE(result.d_s == 0.0);
    REQUIRE(result.P_t.empty());
}

// ════════════════════════════════════════════════════════════════
// V3 FITNESS FUNCTION TESTS
// ════════════════════════════════════════════════════════════════

TEST_CASE("V3 Fitness: degenerate graph", "[fitness][v3]") {
    DynamicGraph graph(2);
    FitnessParamsV3 params;
    auto r = compute_fitness_v3(graph, params);
    REQUIRE(r.total <= -50.0);
    REQUIRE(!r.is_connected);
}

TEST_CASE("V3 Fitness: breakdown sums to total", "[fitness][v3]") {
    DynamicGraph graph(20);
    for (int i = 0; i < 19; ++i) graph.add_edge(i, i + 1);
    graph.add_edge(0, 19);

    FitnessParamsV3 params;
    params.target_dim = 1.0;
    params.n_initial = 20.0;
    params.spectral_walkers = 200;
    params.spectral_steps = 30;
    params.hausdorff_sources = 10;
    params.curvature_samples = 20;
    params.use_spectral_v2 = true;

    auto r = compute_fitness_v3(graph, params);
    double sum = r.f_hausdorff + r.f_curvature + r.f_spectral
               + r.f_connectivity + r.f_stability + r.f_regularity
               + r.f_density + r.f_degradation;
    REQUIRE_THAT(r.total, Catch::Matchers::WithinAbs(sum, 1e-10));
    REQUIRE(r.is_connected);
}

TEST_CASE("V3 Fitness: density penalty activates on dense graph", "[fitness][v3]") {
    // Create a nearly-complete graph (high degree)
    DynamicGraph graph(20);
    for (int i = 0; i < 20; ++i)
        for (int j = i + 1; j < 20; ++j)
            graph.add_edge(i, j);

    FitnessParamsV3 params;
    params.target_dim = 3.0;
    params.density_k_max_factor = 2.2;  // k_max = 6.6
    params.n_initial = 20.0;
    params.spectral_walkers = 100;
    params.spectral_steps = 20;

    auto r = compute_fitness_v3(graph, params);
    // Mean degree = 19, well above k_max = 6.6 → density penalty should be large
    // Hard safeguard triggers at 4*3=12, so we get early exit
    REQUIRE(r.f_density < -1.0);
}

TEST_CASE("V3 Fitness: non-degradation penalty", "[fitness][v3]") {
    // Create a cycle (d_H ≈ 1) with baseline d_H = 1.0
    DynamicGraph graph(30);
    for (int i = 0; i < 30; ++i) graph.add_edge(i, (i + 1) % 30);

    FitnessParamsV3 params;
    params.target_dim = 1.0;
    params.n_initial = 30.0;
    params.spectral_walkers = 200;
    params.spectral_steps = 30;
    params.hausdorff_sources = 10;
    params.curvature_samples = 20;

    // Set baseline with perfect d_H
    params.baseline.valid = true;
    params.baseline.d_H = 1.0;
    params.baseline.mean_curvature = 0.0;

    auto r = compute_fitness_v3(graph, params);
    REQUIRE(r.is_connected);
    // f_degradation should be <= 0 (either no degradation or negative penalty)
    REQUIRE(r.f_degradation <= 0.0);
}

TEST_CASE("V3 Fitness: baseline metrics computation", "[fitness][v3]") {
    auto g = DynamicGraph::create_lattice_3d(5, 5, 5); // N=125
    auto bm = compute_baseline_metrics(g, 3.0, 15, 5000, 200, 100, 42);
    REQUIRE(bm.valid);
    REQUIRE(bm.d_H > 1.0);
    REQUIRE(bm.d_H < 5.0);
    REQUIRE(bm.mean_degree > 4.0);  // 3D lattice internal degree = 6
    REQUIRE(std::abs(bm.mean_curvature) < 0.5);
}

TEST_CASE("Ensemble: basic statistics", "[ensemble]") {
    // Use a trivial identity-like rule on a small 3D lattice
    // with 3 runs to verify the ensemble machinery works
    std::vector<float> params(ParametricRule::TOTAL_PARAMS, 0.0f);
    // Set minimal activity so the graph doesn't change much
    params[9] = 0.01f;   // p_edge_add
    params[10] = 0.01f;  // p_edge_remove
    ParametricRule rule(params);

    EnsembleConfig cfg;
    cfg.num_runs = 3;
    cfg.evo_steps = 10;
    cfg.master_seed = 1234;
    cfg.graph_type = "lattice_3d";
    cfg.graph_size = 27;  // 3^3
    cfg.fitness_params.target_dim = 3.0;
    cfg.fitness_params.n_initial = 27.0;
    cfg.fitness_params.spectral_walkers = 100;
    cfg.fitness_params.spectral_steps = 20;
    cfg.fitness_params.hausdorff_sources = 5;
    cfg.fitness_params.curvature_samples = 20;

    auto result = run_ensemble(rule, cfg);

    REQUIRE(result.n_total == 3);
    REQUIRE(result.runs.size() == 3);
    // Each run should have different seeds
    REQUIRE(result.runs[0].seed != result.runs[1].seed);
    REQUIRE(result.runs[1].seed != result.runs[2].seed);
    // Statistics should have valid entries
    REQUIRE(result.d_H.n_valid >= 1);
    REQUIRE(result.fitness_total.n_valid >= 1);
    // Std error should be non-negative
    REQUIRE(result.d_H.std_err >= 0.0);

    // JSON serialization should work
    auto json_str = ensemble_result_to_json(result);
    REQUIRE(json_str.size() > 100);
}
