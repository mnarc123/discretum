#include <iostream>
#include <string>
#include <cmath>
#include <fmt/core.h>
#include <spdlog/spdlog.h>

#include "core/graph.hpp"
#include "geometry/ollivier_ricci.hpp"
#include "geometry/spectral_dimension.hpp"
#include "geometry/geodesic.hpp"
#include "geometry/metric_tensor.hpp"
#include "automaton/rule_parametric.hpp"
#include "automaton/evolution.hpp"
#include "search/fitness.hpp"

using namespace discretum;

struct ValidationResult {
    std::string name;
    bool passed;
    std::string detail;
};

static ValidationResult validate_graph_ops() {
    // Basic graph operations: create, add/remove edges, split, merge
    DynamicGraph g(10);
    for (int i = 0; i < 9; ++i) g.add_edge(i, i + 1);
    
    if (g.num_nodes() != 10 || g.num_edges() != 9)
        return {"Graph ops", false, "Wrong node/edge count"};
    
    g.remove_edge(0, 1);
    if (g.num_edges() != 8 || g.has_edge(0, 1))
        return {"Graph ops", false, "Edge removal failed"};
    
    g.add_edge(0, 1);
    SplitParams sp;
    auto [a, b] = g.split_node(5, sp);
    if (a == b)
        return {"Graph ops", false, "Split returned same node"};
    
    g.merge_nodes(a, b);
    
    return {"Graph ops", true, fmt::format("{} nodes, {} edges after ops", g.num_nodes(), g.num_edges())};
}

static ValidationResult validate_ollivier_ricci() {
    // K_4: exact curvature κ = 2/n = 2/4 = 0.5 for α=0 (Lin-Lu-Yau)
    // Our implementation uses Wasserstein-1 which gives κ = 2/3 for K_4
    DynamicGraph g(4);
    for (int i = 0; i < 4; ++i)
        for (int j = i + 1; j < 4; ++j)
            g.add_edge(i, j);
    
    float avg = compute_average_ollivier_ricci(g, 0.0f);
    if (avg < 0.4f || avg > 0.8f)
        return {"Ollivier-Ricci", false, fmt::format("K_4 avg curvature = {:.4f}, expected ~0.667", avg)};
    
    return {"Ollivier-Ricci", true, fmt::format("K_4 avg κ = {:.4f}", avg)};
}

static ValidationResult validate_spectral_dimension() {
    // C_50: spectral dimension ≈ 1.0
    DynamicGraph g(50);
    for (int i = 0; i < 50; ++i) g.add_edge(i, (i + 1) % 50);
    
    auto sd = compute_spectral_dimension_detailed(g, 5000, 100, 0.05f, 0.5f, 42);
    if (std::abs(sd.dimension - 1.0f) > 0.5f)
        return {"Spectral dim", false, fmt::format("C_50 d_s = {:.3f}, expected ~1.0", sd.dimension)};
    
    return {"Spectral dim", true, fmt::format("C_50 d_s = {:.3f}", sd.dimension)};
}

static ValidationResult validate_geodesic() {
    // Path P_10: diameter = 9
    DynamicGraph g(10);
    for (int i = 0; i < 9; ++i) g.add_edge(i, i + 1);
    
    uint32_t diam = compute_diameter(g);
    if (diam != 9)
        return {"Geodesic", false, fmt::format("P_10 diameter = {}, expected 9", diam)};
    
    double avg = compute_average_path_length(g);
    if (avg < 3.0 || avg > 4.0)
        return {"Geodesic", false, fmt::format("P_10 avg path = {:.3f}, expected ~3.67", avg)};
    
    return {"Geodesic", true, fmt::format("P_10 diam={}, avg_path={:.3f}", diam, avg)};
}

static ValidationResult validate_metric_tensor() {
    auto g = DynamicGraph::create_lattice_3d(5, 5, 1);
    auto G = compute_metric_tensor(g, 2);
    
    if (G.rows() != 2 || G.cols() != 2)
        return {"Metric tensor", false, "Wrong dimensions"};
    
    // Check symmetry
    if (std::abs(G(0, 1) - G(1, 0)) > 1e-10)
        return {"Metric tensor", false, "Not symmetric"};
    
    // Check positive semi-definite
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(G);
    if (es.eigenvalues()(0) < -1e-10)
        return {"Metric tensor", false, "Negative eigenvalue"};
    
    return {"Metric tensor", true, fmt::format("λ = [{:.4f}, {:.4f}]", es.eigenvalues()(0), es.eigenvalues()(1))};
}

static ValidationResult validate_evolution() {
    std::vector<float> params(ParametricRule::TOTAL_PARAMS, 0.0f);
    ParametricRule rule(params);
    DynamicGraph g = DynamicGraph::create_lattice_3d(4, 4, 4);
    uint32_t initial_nodes = g.num_nodes();
    
    EvolutionConfig cfg;
    cfg.num_steps = 20;
    cfg.seed = 42;
    
    Evolution evo(std::move(g), std::move(rule), cfg);
    auto result = evo.run();
    
    const auto& final_g = evo.get_graph();
    if (final_g.num_nodes() == 0)
        return {"Evolution", false, "Graph collapsed to 0 nodes"};
    
    return {"Evolution", true, fmt::format("{} → {} nodes, {} edges over {} steps",
            initial_nodes, final_g.num_nodes(), final_g.num_edges(), cfg.num_steps)};
}

static ValidationResult validate_fitness() {
    DynamicGraph g(20);
    for (int i = 0; i < 19; ++i) g.add_edge(i, i + 1);
    g.add_edge(0, 19);
    
    FitnessParams fp;
    fp.target_dimension = 1.0;
    fp.target_size = 20.0;
    fp.spectral_walkers = 500;
    fp.spectral_steps = 50;
    
    auto fb = compute_fitness_detailed(g, fp);
    
    double sum = fb.ricci_term + fb.dimension_term + fb.connectivity_term
               + fb.degree_reg_term + fb.size_term;
    if (std::abs(fb.total - sum) > 1e-10)
        return {"Fitness", false, "Breakdown doesn't sum to total"};
    
    if (!fb.is_connected)
        return {"Fitness", false, "C_20 reported as disconnected"};
    
    return {"Fitness", true, fmt::format("C_20 fitness = {:.4f}", fb.total)};
}

int main(int argc, char* argv[]) {
    (void)argc; (void)argv;
    spdlog::set_level(spdlog::level::warn);
    
    fmt::print("╔══════════════════════════════════════════════════╗\n");
    fmt::print("║     DISCRETUM — Validation Suite                ║\n");
    fmt::print("╚══════════════════════════════════════════════════╝\n\n");
    
    std::vector<ValidationResult> results;
    
    fmt::print("Running validations...\n\n");
    
    results.push_back(validate_graph_ops());
    results.push_back(validate_ollivier_ricci());
    results.push_back(validate_spectral_dimension());
    results.push_back(validate_geodesic());
    results.push_back(validate_metric_tensor());
    results.push_back(validate_evolution());
    results.push_back(validate_fitness());
    
    int passed = 0, failed = 0;
    for (auto& r : results) {
        fmt::print("  [{}] {}: {}\n", r.passed ? "PASS" : "FAIL", r.name, r.detail);
        if (r.passed) ++passed; else ++failed;
    }
    
    fmt::print("\n──────────────────────────────────────────────────\n");
    fmt::print("  Results: {} passed, {} failed, {} total\n", passed, failed, passed + failed);
    fmt::print("──────────────────────────────────────────────────\n");
    
    return failed > 0 ? 1 : 0;
}
