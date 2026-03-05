#include <iostream>
#include <string>
#include <cmath>
#include <fmt/core.h>
#include <spdlog/spdlog.h>
#include <nlohmann/json.hpp>
#include <fstream>

#include "core/graph.hpp"
#include "automaton/rule_parametric.hpp"
#include "automaton/evolution.hpp"
#include "search/fitness.hpp"
#include "search/cmaes.hpp"
#include "search/genetic.hpp"
#include "geometry/geodesic.hpp"
#include "geometry/metric_tensor.hpp"
#include "geometry/ollivier_ricci.hpp"
#include "geometry/spectral_dimension.hpp"

using namespace discretum;
using json = nlohmann::json;

static void print_usage() {
    fmt::print("DISCRETUM — Search for emergent gravity rules\n\n");
    fmt::print("Usage:\n");
    fmt::print("  discretum_search cmaes [config.json]   Run CMA-ES optimizer\n");
    fmt::print("  discretum_search ga    [config.json]   Run genetic algorithm\n");
    fmt::print("  discretum_search eval  <params.json>   Evaluate a single rule\n");
    fmt::print("  discretum_search analyze <graph.bin>   Analyze a saved graph\n");
}

static json default_config() {
    json cfg;
    cfg["optimizer"] = "cmaes";
    cfg["max_generations"] = 100;
    cfg["population_size"] = 0;  // auto
    cfg["evo_steps"] = 200;
    cfg["graph_size"] = 125;
    cfg["graph_type"] = "lattice_3d";
    cfg["fitness_version"] = 1;
    cfg["seed"] = 42;
    cfg["sigma0"] = 0.5;
    cfg["target_dimension"] = 3.0;
    cfg["target_curvature"] = 0.0;
    cfg["spectral_walkers"] = 500;
    cfg["spectral_steps"] = 50;
    cfg["checkpoint_dir"] = "checkpoints";
    cfg["checkpoint_interval"] = 1;
    return cfg;
}

static FitnessParams fitness_from_config(const json& cfg) {
    FitnessParams fp;
    fp.target_dimension = cfg.value("target_dimension", 3.0);
    fp.target_avg_curvature = cfg.value("target_curvature", 0.0);
    fp.spectral_walkers = cfg.value("spectral_walkers", 500);
    fp.spectral_steps = cfg.value("spectral_steps", 50);
    // Set target_size from graph_size so size penalty is meaningful
    int gs = cfg.value("graph_size", 125);
    std::string gt = cfg.value("graph_type", std::string("lattice_3d"));
    if (gt == "lattice_4d") {
        int side = std::max(3, static_cast<int>(std::pow(gs, 0.25)));
        fp.target_size = static_cast<double>(side * side * side * side);
    } else {
        int side = std::max(3, static_cast<int>(std::cbrt(gs)));
        fp.target_size = static_cast<double>(side * side * side);
    }
    return fp;
}

static FitnessParamsV2 fitness_v2_from_config(const json& cfg) {
    FitnessParamsV2 fp;
    fp.target_dim = cfg.value("target_dimension", 4.0);
    fp.w_hausdorff = cfg.value("w_hausdorff", 2.0);
    fp.w_curvature = cfg.value("w_curvature", 1.5);
    fp.w_spectral = cfg.value("w_spectral", 1.0);
    fp.w_connectivity = cfg.value("w_connectivity", 10.0);
    fp.w_stability = cfg.value("w_stability", 0.5);
    fp.w_regularity = cfg.value("w_regularity", 0.3);
    fp.curvature_fluct_beta = cfg.value("curvature_fluctuation_penalty", 0.5);
    fp.curvature_samples = cfg.value("curvature_samples", 300);
    fp.spectral_walkers = cfg.value("spectral_walkers", 5000);
    fp.spectral_steps = cfg.value("spectral_steps", 500);
    fp.hausdorff_sources = cfg.value("hausdorff_sources", 30);
    fp.seed = cfg.value("seed", 42);
    // Set n_initial from graph config
    int gs = cfg.value("graph_size", 625);
    std::string gt = cfg.value("graph_type", std::string("lattice_4d"));
    if (gt == "lattice_4d") {
        int side = std::max(3, static_cast<int>(std::pow(gs, 0.25)));
        fp.n_initial = static_cast<double>(side * side * side * side);
    } else {
        int side = std::max(3, static_cast<int>(std::cbrt(gs)));
        fp.n_initial = static_cast<double>(side * side * side);
    }
    return fp;
}

static FitnessParamsV3 fitness_v3_from_config(const json& cfg) {
    FitnessParamsV3 fp;
    fp.target_dim = cfg.value("target_dimension", 4.0);
    fp.w_hausdorff = cfg.value("w_hausdorff", 2.0);
    fp.w_curvature = cfg.value("w_curvature", 1.5);
    fp.w_spectral = cfg.value("w_spectral", 1.0);
    fp.w_connectivity = cfg.value("w_connectivity", 10.0);
    fp.w_stability = cfg.value("w_stability", 0.5);
    fp.w_regularity = cfg.value("w_regularity", 0.3);
    fp.w_density = cfg.value("w_density", 3.0);
    fp.w_degradation = cfg.value("w_degradation", 2.0);
    fp.density_k_max_factor = cfg.value("density_k_max_factor", 2.2);
    fp.curvature_fluct_beta = cfg.value("curvature_fluctuation_penalty", 0.5);
    fp.curvature_samples = cfg.value("curvature_samples", 300);
    fp.spectral_walkers = cfg.value("spectral_walkers", 20000);
    fp.spectral_steps = cfg.value("spectral_steps", 800);
    fp.use_spectral_v2 = cfg.value("use_spectral_v2", true);
    fp.hausdorff_sources = cfg.value("hausdorff_sources", 30);
    fp.seed = cfg.value("seed", 42);
    // Set n_initial from graph config
    int gs = cfg.value("graph_size", 625);
    std::string gt = cfg.value("graph_type", std::string("lattice_4d"));
    if (gt == "lattice_4d") {
        int side = std::max(3, static_cast<int>(std::pow(gs, 0.25)));
        fp.n_initial = static_cast<double>(side * side * side * side);
    } else {
        int side = std::max(3, static_cast<int>(std::cbrt(gs)));
        fp.n_initial = static_cast<double>(side * side * side);
    }
    return fp;
}

static DynamicGraph create_baseline_graph(const json& cfg) {
    std::string gt = cfg.value("graph_type", std::string("lattice_4d"));
    int gs = cfg.value("graph_size", 625);
    if (gt == "lattice_4d") {
        int side = std::max(3, static_cast<int>(std::pow(gs, 0.25)));
        return DynamicGraph::create_lattice_4d(side, side, side, side);
    } else {
        int side = std::max(3, static_cast<int>(std::cbrt(gs)));
        return DynamicGraph::create_lattice_3d(side, side, side);
    }
}

static int run_cmaes(const json& cfg) {
    CMAESConfig cc;
    cc.max_generations = cfg.value("max_generations", 100);
    cc.lambda = cfg.value("population_size", 0);
    cc.sigma0 = cfg.value("sigma0", 0.5);
    cc.evo_steps = cfg.value("evo_steps", 200);
    cc.graph_size = cfg.value("graph_size", 125);
    cc.seed = cfg.value("seed", 42);
    cc.graph_type = cfg.value("graph_type", std::string("lattice_3d"));
    cc.fitness_version = cfg.value("fitness_version", 1);
    cc.fitness_params = fitness_from_config(cfg);
    cc.fitness_params_v2 = fitness_v2_from_config(cfg);
    cc.fitness_params_v3 = fitness_v3_from_config(cfg);
    cc.checkpoint_dir = cfg.value("checkpoint_dir", std::string("checkpoints"));
    cc.checkpoint_interval = cfg.value("checkpoint_interval", 1);

    // Compute baseline for v3
    if (cc.fitness_version == 3) {
        spdlog::info("Computing baseline metrics for v3 non-degradation constraint...");
        auto baseline_graph = create_baseline_graph(cfg);
        cc.fitness_params_v3.baseline = compute_baseline_metrics(
            baseline_graph, cc.fitness_params_v3.target_dim,
            cc.fitness_params_v3.hausdorff_sources,
            std::min(cc.fitness_params_v3.spectral_walkers, 10000u),
            cc.fitness_params_v3.spectral_steps,
            cc.fitness_params_v3.curvature_samples,
            cc.fitness_params_v3.seed);
        spdlog::info("Baseline: d_H={:.3f}, d_s={:.3f}, <k>={:.1f}, <kappa>={:.4f}",
                     cc.fitness_params_v3.baseline.d_H, cc.fitness_params_v3.baseline.d_s,
                     cc.fitness_params_v3.baseline.mean_degree,
                     cc.fitness_params_v3.baseline.mean_curvature);
    }
    
    spdlog::info("Starting CMA-ES search (dim={}, max_gen={}, graph_type={}, fitness_v={}, checkpoint_dir={})",
                 cc.dim, cc.max_generations, cc.graph_type, cc.fitness_version, cc.checkpoint_dir);
    
    CMAES optimizer(cc);
    auto result = optimizer.optimize();
    
    spdlog::info("CMA-ES finished: best_fitness={:.6f} in {} generations",
                 result.best_fitness, result.generations_used);
    
    // Save final result
    json out;
    out["best_fitness"] = result.best_fitness;
    out["best_params"] = result.best_params;
    out["generations"] = result.generations_used;
    out["fitness_history"] = result.fitness_history;
    
    std::string result_path = cc.checkpoint_dir + "/cmaes_result.json";
    std::ofstream f(result_path);
    f << out.dump(2) << std::endl;
    spdlog::info("Results saved to {}", result_path);
    
    return 0;
}

static int run_ga(const json& cfg) {
    GAConfig gc;
    gc.max_generations = cfg.value("max_generations", 100);
    gc.pop_size = cfg.value("population_size", 50);
    if (gc.pop_size <= 0) gc.pop_size = 50;
    gc.evo_steps = cfg.value("evo_steps", 200);
    gc.graph_size = cfg.value("graph_size", 125);
    gc.seed = cfg.value("seed", 42);
    gc.graph_type = cfg.value("graph_type", std::string("lattice_3d"));
    gc.fitness_version = cfg.value("fitness_version", 1);
    gc.fitness_params = fitness_from_config(cfg);
    gc.fitness_params_v2 = fitness_v2_from_config(cfg);
    gc.fitness_params_v3 = fitness_v3_from_config(cfg);
    gc.checkpoint_dir = cfg.value("checkpoint_dir", std::string("checkpoints"));
    gc.checkpoint_interval = cfg.value("checkpoint_interval", 1);

    // Compute baseline for v3
    if (gc.fitness_version == 3) {
        spdlog::info("Computing baseline metrics for v3 non-degradation constraint...");
        auto baseline_graph = create_baseline_graph(cfg);
        gc.fitness_params_v3.baseline = compute_baseline_metrics(
            baseline_graph, gc.fitness_params_v3.target_dim,
            gc.fitness_params_v3.hausdorff_sources,
            std::min(gc.fitness_params_v3.spectral_walkers, 10000u),
            gc.fitness_params_v3.spectral_steps,
            gc.fitness_params_v3.curvature_samples,
            gc.fitness_params_v3.seed);
        spdlog::info("Baseline: d_H={:.3f}, d_s={:.3f}, <k>={:.1f}, <kappa>={:.4f}",
                     gc.fitness_params_v3.baseline.d_H, gc.fitness_params_v3.baseline.d_s,
                     gc.fitness_params_v3.baseline.mean_degree,
                     gc.fitness_params_v3.baseline.mean_curvature);
    }
    
    spdlog::info("Starting GA search (pop={}, max_gen={}, graph_type={}, fitness_v={}, checkpoint_dir={})",
                 gc.pop_size, gc.max_generations, gc.graph_type, gc.fitness_version, gc.checkpoint_dir);
    
    GeneticAlgorithm optimizer(gc);
    auto result = optimizer.evolve();
    
    spdlog::info("GA finished: best_fitness={:.6f} in {} generations",
                 result.best_fitness, result.generations_used);
    
    json out;
    out["best_fitness"] = result.best_fitness;
    out["best_params"] = result.best_params;
    out["generations"] = result.generations_used;
    out["fitness_history"] = result.fitness_history;
    
    std::string result_path = gc.checkpoint_dir + "/ga_result.json";
    std::ofstream f(result_path);
    f << out.dump(2) << std::endl;
    spdlog::info("Results saved to {}", result_path);
    
    return 0;
}

static int run_eval(const std::string& params_file) {
    std::ifstream f(params_file);
    if (!f) { fmt::print(stderr, "Cannot open {}\n", params_file); return 1; }
    json j; f >> j;
    
    std::vector<float> params;
    for (auto& v : j["params"]) params.push_back(v.get<float>());
    
    ParametricRule rule(params);
    int side = 5;
    DynamicGraph graph = DynamicGraph::create_lattice_3d(side, side, side);
    
    EvolutionConfig evo_cfg;
    evo_cfg.num_steps = j.value("evo_steps", 50);
    evo_cfg.seed = j.value("seed", 42);
    evo_cfg.snapshot_interval = 10;
    
    Evolution evo(std::move(graph), std::move(rule), evo_cfg);
    auto evo_result = evo.run();
    
    const auto& g = evo.get_graph();
    spdlog::info("Final graph: {} nodes, {} edges", g.num_nodes(), g.num_edges());
    
    FitnessParams fp;
    fp.spectral_walkers = 2000;
    fp.spectral_steps = 80;
    auto fb = compute_fitness_detailed(g, fp);
    
    fmt::print("\n=== Fitness Breakdown ===\n");
    fmt::print("  Total:        {:.6f}\n", fb.total);
    fmt::print("  Ricci:        {:.6f}  (measured avg κ = {:.4f})\n", fb.ricci_term, fb.measured_avg_curvature);
    fmt::print("  Dimension:    {:.6f}  (measured d_s = {:.4f})\n", fb.dimension_term, fb.measured_dimension);
    fmt::print("  Connectivity: {:.6f}  (connected: {})\n", fb.connectivity_term, fb.is_connected);
    fmt::print("  Degree reg:   {:.6f}\n", fb.degree_reg_term);
    fmt::print("  Size:         {:.6f}\n", fb.size_term);
    
    // Save evolved graph
    g.save("evolved_graph.bin");
    spdlog::info("Evolved graph saved to evolved_graph.bin");
    
    return 0;
}

static int run_analyze(const std::string& graph_file) {
    auto graph = DynamicGraph::load(graph_file);
    uint32_t n = graph.num_nodes();
    uint32_t m = graph.num_edges();
    
    fmt::print("\n=== Graph Analysis ===\n");
    fmt::print("Nodes: {}\n", n);
    fmt::print("Edges: {}\n", m);
    fmt::print("Avg degree: {:.2f}\n", n > 0 ? 2.0 * m / n : 0.0);
    
    auto components = graph.connected_components();
    fmt::print("Connected components: {}\n", components.size());
    
    if (n <= 500) {
        uint32_t diam = compute_diameter(graph);
        fmt::print("Diameter: {}\n", diam);
        
        double avg_path = compute_average_path_length(graph);
        fmt::print("Average path length: {:.3f}\n", avg_path);
        
        double d_H = estimate_hausdorff_dimension(graph);
        fmt::print("Hausdorff dimension: {:.3f}\n", d_H);
    }
    
    float avg_curv = compute_average_ollivier_ricci(graph, 0.5f);
    fmt::print("Average Ollivier-Ricci curvature (α=0.5): {:.4f}\n", avg_curv);
    
    auto sd = compute_spectral_dimension_detailed(graph, 5000, 100, 0.05f, 0.5f, 42);
    fmt::print("Spectral dimension: {:.3f} ± {:.3f}\n",
               sd.dimension, sd.fit_error);
    
    if (n <= 200) {
        double R = estimate_scalar_curvature(graph, 3);
        fmt::print("Scalar curvature proxy (anisotropy): {:.4f}\n", R);
    }
    
    return 0;
}

int main(int argc, char* argv[]) {
    spdlog::set_level(spdlog::level::info);
    
    if (argc < 2) { print_usage(); return 0; }
    
    std::string command = argv[1];
    
    if (command == "cmaes") {
        json cfg = default_config();
        if (argc >= 3) {
            std::ifstream f(argv[2]);
            if (f) { f >> cfg; }
        }
        return run_cmaes(cfg);
    }
    else if (command == "ga") {
        json cfg = default_config();
        if (argc >= 3) {
            std::ifstream f(argv[2]);
            if (f) { f >> cfg; }
        }
        return run_ga(cfg);
    }
    else if (command == "eval") {
        if (argc < 3) { fmt::print(stderr, "Usage: discretum_search eval <params.json>\n"); return 1; }
        return run_eval(argv[2]);
    }
    else if (command == "analyze") {
        if (argc < 3) { fmt::print(stderr, "Usage: discretum_search analyze <graph.bin>\n"); return 1; }
        return run_analyze(argv[2]);
    }
    else {
        print_usage();
        return 1;
    }
}
