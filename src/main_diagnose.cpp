#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <set>
#include <cmath>
#include <numeric>
#include <algorithm>
#include <chrono>

#include <fmt/core.h>
#include <spdlog/spdlog.h>
#include <nlohmann/json.hpp>

#include "core/graph.hpp"
#include "automaton/rule_parametric.hpp"
#include "automaton/evolution.hpp"
#include "search/fitness.hpp"
#include "search/ensemble.hpp"
#include "geometry/ollivier_ricci.hpp"
#include "geometry/spectral_dimension.hpp"
#include "geometry/geodesic.hpp"
#include "geometry/metric_tensor.hpp"
#include <Eigen/Eigenvalues>

using namespace discretum;
using json = nlohmann::json;

static void print_usage() {
    fmt::print("DISCRETUM — Detailed rule diagnostics\n\n");
    fmt::print("Usage:\n");
    fmt::print("  discretum_diagnose <params.json> [options]\n\n");
    fmt::print("The JSON file should contain either:\n");
    fmt::print("  {{\"best_params\": [...]}}   (output from search)\n");
    fmt::print("  {{\"params\": [...]}}         (raw params)\n\n");
    fmt::print("Options:\n");
    fmt::print("  --steps N         Total evolution steps (default: 200)\n");
    fmt::print("  --transient N     Transient steps before measurement (default: 50)\n");
    fmt::print("  --graph-size N    Initial lattice side length (default: 5, i.e. 5^3=125 nodes)\n");
    fmt::print("  --graph-type T    lattice_3d or lattice_4d (default: lattice_3d)\n");
    fmt::print("  --seed N          Random seed (default: 42)\n");
    fmt::print("  --output FILE     Save diagnostics JSON to file\n");
    fmt::print("  --target-dim D    Target spectral dimension (default: 3.0)\n");
    fmt::print("  --target-curv C   Target avg curvature (default: 0.0)\n");
    fmt::print("  --walkers N       Spectral dimension walkers (default: 5000)\n");
    fmt::print("  --walk-steps N    Spectral dimension max steps (default: 100)\n");
    fmt::print("  --ensemble N      Run N independent evolutions with error bars (default: 0 = off)\n");
    fmt::print("  --fitness-v N     Fitness version: 1, 2, or 3 (default: 1)\n");
    fmt::print("  --quiet           Suppress per-step logging\n");
    fmt::print("  --no-evolution    Measure bare lattice without evolving\n");
    fmt::print("  --output-spectral-detail FILE  Save P(t), d_eff(t) detail to JSON\n");
}

struct DiagnoseConfig {
    int total_steps = 200;
    int transient_steps = 50;
    int graph_side = 5;
    std::string graph_type = "lattice_3d";
    uint64_t seed = 42;
    std::string output_file;
    double target_dim = 3.0;
    double target_curv = 0.0;
    uint32_t walkers = 5000;
    uint32_t walk_steps = 100;
    int ensemble_runs = 0;
    int fitness_version = 1;
    bool quiet = false;
    bool no_evolution = false;
    std::string spectral_detail_file;
};

static DiagnoseConfig parse_args(int argc, char* argv[]) {
    DiagnoseConfig cfg;
    for (int i = 2; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--steps" && i + 1 < argc) cfg.total_steps = std::stoi(argv[++i]);
        else if (arg == "--transient" && i + 1 < argc) cfg.transient_steps = std::stoi(argv[++i]);
        else if (arg == "--graph-size" && i + 1 < argc) cfg.graph_side = std::stoi(argv[++i]);
        else if (arg == "--seed" && i + 1 < argc) cfg.seed = std::stoull(argv[++i]);
        else if (arg == "--output" && i + 1 < argc) cfg.output_file = argv[++i];
        else if (arg == "--target-dim" && i + 1 < argc) cfg.target_dim = std::stod(argv[++i]);
        else if (arg == "--target-curv" && i + 1 < argc) cfg.target_curv = std::stod(argv[++i]);
        else if (arg == "--walkers" && i + 1 < argc) cfg.walkers = std::stoul(argv[++i]);
        else if (arg == "--walk-steps" && i + 1 < argc) cfg.walk_steps = std::stoul(argv[++i]);
        else if (arg == "--graph-type" && i + 1 < argc) cfg.graph_type = argv[++i];
        else if (arg == "--ensemble" && i + 1 < argc) cfg.ensemble_runs = std::stoi(argv[++i]);
        else if (arg == "--fitness-v" && i + 1 < argc) cfg.fitness_version = std::stoi(argv[++i]);
        else if (arg == "--quiet") cfg.quiet = true;
        else if (arg == "--no-evolution") cfg.no_evolution = true;
        else if (arg == "--output-spectral-detail" && i + 1 < argc) cfg.spectral_detail_file = argv[++i];
    }
    return cfg;
}

int main(int argc, char* argv[]) {
    spdlog::set_level(spdlog::level::info);

    if (argc < 2 || std::string(argv[1]) == "--help" || std::string(argv[1]) == "-h") {
        print_usage();
        return 0;
    }

    // Parse params file
    std::string params_file = argv[1];
    std::ifstream f(params_file);
    if (!f) {
        fmt::print(stderr, "Cannot open {}\n", params_file);
        return 1;
    }
    json j;
    f >> j;
    f.close();

    std::vector<double> params_d;
    if (j.contains("best_params")) {
        params_d = j["best_params"].get<std::vector<double>>();
    } else if (j.contains("params")) {
        params_d = j["params"].get<std::vector<double>>();
    } else {
        fmt::print(stderr, "JSON must contain 'best_params' or 'params' array\n");
        return 1;
    }

    std::vector<float> params_f(params_d.begin(), params_d.end());

    DiagnoseConfig cfg = parse_args(argc, argv);

    if (cfg.quiet) spdlog::set_level(spdlog::level::warn);

    // ═══════════ ENSEMBLE MODE ═══════════
    if (cfg.ensemble_runs > 0) {
        spdlog::set_level(spdlog::level::warn);
        ParametricRule rule(params_f);

        EnsembleConfig ecfg;
        ecfg.num_runs = static_cast<uint32_t>(cfg.ensemble_runs);
        ecfg.evo_steps = cfg.no_evolution ? 0 : static_cast<uint32_t>(cfg.total_steps);
        ecfg.master_seed = cfg.seed;
        ecfg.graph_type = cfg.graph_type;
        {
            uint32_t s = static_cast<uint32_t>(cfg.graph_side);
            ecfg.graph_size = (cfg.graph_type == "lattice_4d")
                ? s * s * s * s
                : s * s * s;
        }
        ecfg.fitness_version = cfg.fitness_version;
        ecfg.fitness_params.target_dim = cfg.target_dim;
        ecfg.fitness_params.spectral_walkers = cfg.walkers;
        ecfg.fitness_params.spectral_steps = cfg.walk_steps;
        ecfg.fitness_params.hausdorff_sources = 30;
        ecfg.fitness_params.curvature_samples = 300;

        if (cfg.fitness_version == 3) {
            ecfg.fitness_params_v3.target_dim = cfg.target_dim;
            ecfg.fitness_params_v3.spectral_walkers = cfg.walkers;
            ecfg.fitness_params_v3.spectral_steps = cfg.walk_steps;
            ecfg.fitness_params_v3.hausdorff_sources = 30;
            ecfg.fitness_params_v3.curvature_samples = 300;
            ecfg.fitness_params_v3.use_spectral_v2 = true;
            // Compute baseline for non-degradation constraint
            DynamicGraph baseline_graph = (cfg.graph_type == "lattice_4d")
                ? DynamicGraph::create_lattice_4d(cfg.graph_side, cfg.graph_side, cfg.graph_side, cfg.graph_side)
                : DynamicGraph::create_lattice_3d(cfg.graph_side, cfg.graph_side, cfg.graph_side);
            ecfg.fitness_params_v3.baseline = compute_baseline_metrics(
                baseline_graph, cfg.target_dim, 30, cfg.walkers, cfg.walk_steps, 300, cfg.seed);
            fmt::print("  Baseline: d_H={:.3f}, d_s={:.3f}, <k>={:.1f}, <kappa>={:.4f}\n",
                       ecfg.fitness_params_v3.baseline.d_H, ecfg.fitness_params_v3.baseline.d_s,
                       ecfg.fitness_params_v3.baseline.mean_degree,
                       ecfg.fitness_params_v3.baseline.mean_curvature);
        }

        fmt::print("\n╔══════════════════════════════════════════════════════╗\n");
        fmt::print("║         DISCRETUM — Ensemble Diagnostics            ║\n");
        fmt::print("╚══════════════════════════════════════════════════════╝\n\n");
        fmt::print("  Runs: {}   Steps: {}   Graph: {} (side={})   Seed: {}\n\n",
                   ecfg.num_runs, ecfg.evo_steps, cfg.graph_type, cfg.graph_side, cfg.seed);

        auto t0 = std::chrono::high_resolution_clock::now();
        auto result = run_ensemble(rule, ecfg);
        auto t1 = std::chrono::high_resolution_clock::now();
        double elapsed = std::chrono::duration<double>(t1 - t0).count();

        fmt::print("  Completed in {:.1f}s ({} aborted, {}/{} connected)\n\n",
                   elapsed, result.n_aborted, result.n_connected, result.n_total);

        // Print summary table
        auto print_stat = [](const char* name, const ObservableStats& s) {
            fmt::print("  {:20s} = {:+10.4f} ± {:.4f}  (median={:+.4f}, [{:+.4f}, {:+.4f}])\n",
                       name, s.mean, s.std_err, s.median, s.q25, s.q75);
        };

        fmt::print("── Observable Summary (mean ± SE) ──\n");
        print_stat("fitness_total", result.fitness_total);
        print_stat("d_H", result.d_H);
        print_stat("d_s", result.d_s);
        print_stat("mean_curvature", result.mean_curvature);
        print_stat("std_curvature", result.std_curvature);
        print_stat("cv_degree", result.cv_degree);
        print_stat("mean_degree", result.mean_degree);
        print_stat("n_final", result.n_final);
        fmt::print("\n── Fitness Components ──\n");
        print_stat("f_hausdorff", result.f_hausdorff);
        print_stat("f_curvature", result.f_curvature);
        print_stat("f_spectral", result.f_spectral);
        print_stat("f_stability", result.f_stability);
        print_stat("f_regularity", result.f_regularity);

        // Shapiro-Wilk-like normality test (simplified: check skewness+kurtosis)
        auto normality_check = [](const char* name, const std::vector<double>& vals) {
            if (vals.size() < 8) return;
            int n = static_cast<int>(vals.size());
            double mean = 0;
            for (double v : vals) mean += v;
            mean /= n;
            double m2 = 0, m3 = 0, m4 = 0;
            for (double v : vals) {
                double d = v - mean;
                m2 += d * d;
                m3 += d * d * d;
                m4 += d * d * d * d;
            }
            m2 /= n; m3 /= n; m4 /= n;
            double sd = std::sqrt(m2);
            if (sd < 1e-15) return;
            double skew = m3 / (sd * sd * sd);
            double kurt = m4 / (sd * sd * sd * sd) - 3.0;  // excess kurtosis
            bool normal = (std::abs(skew) < 2.0 && std::abs(kurt) < 4.0);
            fmt::print("  {:20s}: skew={:+.3f}, kurt={:+.3f} → {}\n",
                       name, skew, kurt, normal ? "NORMAL" : "NON-NORMAL");
        };

        fmt::print("\n── Normality Check (|skew|<2, |excess kurt|<4) ──\n");
        {
            std::vector<double> v_dH, v_ds, v_kappa, v_fit;
            for (auto& rr : result.runs) {
                if (rr.aborted) continue;
                v_dH.push_back(rr.fitness.d_H);
                v_ds.push_back(rr.fitness.d_s);
                v_kappa.push_back(rr.fitness.mean_curvature);
                v_fit.push_back(rr.fitness.total);
            }
            normality_check("d_H", v_dH);
            normality_check("d_s", v_ds);
            normality_check("mean_curvature", v_kappa);
            normality_check("fitness_total", v_fit);
        }

        fmt::print("\n");

        // Save JSON output
        if (!cfg.output_file.empty()) {
            std::string json_str = ensemble_result_to_json(result);
            std::ofstream of(cfg.output_file);
            of << json_str << std::endl;
            fmt::print("Ensemble results saved to {}\n", cfg.output_file);
        }

        return 0;
    }

    // ═══════════ SINGLE-RUN MODE (original) ═══════════

    // Print parameter summary
    fmt::print("\n╔══════════════════════════════════════════════════════╗\n");
    fmt::print("║           DISCRETUM — Rule Diagnostics              ║\n");
    fmt::print("╚══════════════════════════════════════════════════════╝\n\n");

    fmt::print("── Parameters ({} total) ──\n", params_f.size());
    fmt::print("  θ_state ({}×{} = {}):", ParametricRule::NUM_STATES, ParametricRule::NUM_STATES,
               ParametricRule::NUM_STATES * ParametricRule::NUM_STATES);
    for (int i = 0; i < ParametricRule::NUM_STATES * ParametricRule::NUM_STATES; ++i)
        fmt::print(" {:.4f}", params_f[i]);
    fmt::print("\n");

    int topo_offset = ParametricRule::NUM_STATES * ParametricRule::NUM_STATES;
    const char* topo_names[] = {"p_edge_add", "p_edge_remove", "p_rewire", "p_split", "p_merge"};
    fmt::print("  θ_topo:\n");
    for (int i = 0; i < ParametricRule::NUM_TOPO_PARAMS; ++i) {
        float raw = params_f[topo_offset + i];
        float prob = 1.0f / (1.0f + std::exp(-raw));
        fmt::print("    {:15s} = {:.4f}  (raw={:.4f})\n", topo_names[i], prob, raw);
    }
    fmt::print("\n");

    // Create rule and initial graph
    ParametricRule rule(params_f);
    uint32_t N0;
    DynamicGraph graph = [&]() {
        if (cfg.graph_type == "lattice_4d") {
            int s = cfg.graph_side;
            N0 = s * s * s * s;
            return DynamicGraph::create_lattice_4d(s, s, s, s);
        } else {
            int s = cfg.graph_side;
            N0 = s * s * s;
            return DynamicGraph::create_lattice_3d(s, s, s);
        }
    }();
    int dim = (cfg.graph_type == "lattice_4d") ? 4 : 3;
    fmt::print("── Initial graph: {}^{} lattice = {} nodes, {} edges ──\n\n",
               cfg.graph_side, dim, graph.num_nodes(), graph.num_edges());

    std::vector<uint32_t> node_history;
    std::vector<uint32_t> edge_history;

    EvolutionConfig evo_cfg;
    evo_cfg.num_steps = cfg.transient_steps;
    evo_cfg.seed = cfg.seed;
    evo_cfg.snapshot_interval = 0;
    Evolution evo(std::move(graph), std::move(rule), evo_cfg);

    if (cfg.no_evolution) {
        fmt::print("── No evolution (bare lattice measurement) ──\n\n");
        node_history.push_back(evo.get_graph().num_nodes());
        edge_history.push_back(evo.get_graph().num_edges());
    } else {
        // Phase 1: Transient evolution
        fmt::print("── Phase 1: Transient ({} steps) ──\n", cfg.transient_steps);
        auto t0 = std::chrono::high_resolution_clock::now();

        auto trans_result = evo.run();

        auto t1 = std::chrono::high_resolution_clock::now();
        double trans_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        fmt::print("  After transient: {} nodes, {} edges ({:.1f} ms)\n\n",
                   evo.get_graph().num_nodes(), evo.get_graph().num_edges(), trans_ms);

        // Phase 2: Measurement evolution
        int measure_steps = cfg.total_steps - cfg.transient_steps;
        if (measure_steps <= 0) measure_steps = cfg.total_steps;

        fmt::print("── Phase 2: Measurement ({} steps) ──\n", measure_steps);

        auto t2 = std::chrono::high_resolution_clock::now();
        for (int s = 0; s < measure_steps; ++s) {
            evo.step();
            node_history.push_back(evo.get_graph().num_nodes());
            edge_history.push_back(evo.get_graph().num_edges());
        }
        auto t3 = std::chrono::high_resolution_clock::now();
        double meas_ms = std::chrono::duration<double, std::milli>(t3 - t2).count();
        fmt::print("  Measurement phase done ({:.1f} ms)\n\n", meas_ms);
    }

    // Now compute all observables on the final graph
    const DynamicGraph& g = evo.get_graph();
    uint32_t n_final = g.num_nodes();
    uint32_t m_final = g.num_edges();

    fmt::print("══════════════════════════════════════════════════════\n");
    fmt::print("                  DIAGNOSTIC REPORT                   \n");
    fmt::print("══════════════════════════════════════════════════════\n\n");

    // --- A) Graph Size ---
    fmt::print("── (a) Graph Size ──\n");
    fmt::print("  N_initial = {}\n", N0);
    fmt::print("  N_final   = {}\n", n_final);
    fmt::print("  E_final   = {}\n", m_final);
    fmt::print("  Ratio N_final/N_initial = {:.3f}\n\n", (double)n_final / N0);

    // --- B) Connectivity ---
    fmt::print("── (b) Connectivity ──\n");
    auto components = g.connected_components();
    bool connected = (components.size() == 1);
    fmt::print("  Connected: {}\n", connected ? "YES" : "NO");
    fmt::print("  Components: {}\n", components.size());
    if (!connected && components.size() <= 20) {
        fmt::print("  Component sizes:");
        std::vector<size_t> comp_sizes;
        for (const auto& c : components) comp_sizes.push_back(c.size());
        std::sort(comp_sizes.rbegin(), comp_sizes.rend());
        for (size_t s : comp_sizes) fmt::print(" {}", s);
        fmt::print("\n");
    }
    fmt::print("\n");

    // --- C) Degree distribution ---
    fmt::print("── (c) Degree Distribution ──\n");
    std::map<uint32_t, uint32_t> deg_hist;
    double mean_deg = 0.0;
    uint32_t max_deg_val = 0, min_deg_val = UINT32_MAX;
    for (uint32_t i = 0; i < g.get_nodes().size(); ++i) {
        uint32_t d = g.degree(i);
        if (d == 0 && g.get_nodes()[i].id == DynamicGraph::INVALID_ID) continue;
        deg_hist[d]++;
        mean_deg += d;
        max_deg_val = std::max(max_deg_val, d);
        min_deg_val = std::min(min_deg_val, d);
    }
    mean_deg /= n_final;
    double var_deg = 0.0;
    for (uint32_t i = 0; i < g.get_nodes().size(); ++i) {
        uint32_t d = g.degree(i);
        if (d == 0 && g.get_nodes()[i].id == DynamicGraph::INVALID_ID) continue;
        double diff = d - mean_deg;
        var_deg += diff * diff;
    }
    var_deg /= n_final;
    double cv_deg = (mean_deg > 0) ? std::sqrt(var_deg) / mean_deg : 0.0;

    fmt::print("  <degree> = {:.3f}\n", mean_deg);
    fmt::print("  σ(degree) = {:.3f}\n", std::sqrt(var_deg));
    fmt::print("  CV(degree) = {:.4f}\n", cv_deg);
    fmt::print("  min_degree = {}, max_degree = {}\n", min_deg_val, max_deg_val);
    fmt::print("  Degree histogram:\n");
    for (auto& [d, cnt] : deg_hist) {
        double frac = (double)cnt / n_final * 100.0;
        fmt::print("    deg {:2d}: {:5d} ({:5.1f}%)\n", d, cnt, frac);
    }
    fmt::print("\n");

    // --- D) Ollivier-Ricci curvature ---
    fmt::print("── (d) Ollivier-Ricci Curvature (α=0.5) ──\n");
    std::vector<float> curv_values;
    bool curv_sampled = false;
    bool curv_skipped = false;
    float curv_mean = 0.0f, curv_std = 0.0f, curv_min = 0.0f, curv_max = 0.0f, curv_median = 0.0f;
    size_t n_curv = 0, n_pos = 0, n_neg = 0, n_zero = 0;
    double curv_ms = 0.0;
    
    // Ollivier-Ricci uses BFS internally → O(V+E) per edge.
    // For graphs >50K nodes, even sampling is prohibitively slow.
    constexpr uint32_t MAX_NODES_FOR_CURV = 50000;
    constexpr uint32_t MAX_CURV_EDGES = 2000;
    
    if (n_final > MAX_NODES_FOR_CURV) {
        curv_skipped = true;
        fmt::print("  SKIPPED: graph too large ({} nodes > {} limit)\n", n_final, MAX_NODES_FOR_CURV);
        fmt::print("  Ollivier-Ricci uses BFS per edge → O(V+E) per call, infeasible at this scale.\n\n");
    } else {
        // Collect all edges
        std::vector<std::pair<uint32_t, uint32_t>> edge_list;
        {
            std::set<std::pair<uint32_t, uint32_t>> seen;
            const auto& nodes_vec = g.get_nodes();
            for (uint32_t i = 0; i < nodes_vec.size(); ++i) {
                if (g.degree(i) == 0) continue;
                for (uint32_t nb : g.neighbors(i)) {
                    auto e = std::minmax(i, nb);
                    if (seen.insert(e).second) edge_list.push_back(e);
                }
            }
        }
        
        // If too many edges, sample uniformly
        std::vector<std::pair<uint32_t, uint32_t>> edges_to_eval;
        if (edge_list.size() > MAX_CURV_EDGES) {
            curv_sampled = true;
            PCG32 sample_rng(cfg.seed + 999);
            std::vector<size_t> indices(edge_list.size());
            std::iota(indices.begin(), indices.end(), 0);
            for (size_t i = 0; i < MAX_CURV_EDGES; ++i) {
                size_t j = i + (sample_rng() % (indices.size() - i));
                std::swap(indices[i], indices[j]);
            }
            edges_to_eval.reserve(MAX_CURV_EDGES);
            for (size_t i = 0; i < MAX_CURV_EDGES; ++i)
                edges_to_eval.push_back(edge_list[indices[i]]);
            fmt::print("  Sampling {} of {} edges for curvature estimation...\n",
                       MAX_CURV_EDGES, edge_list.size());
        } else {
            edges_to_eval = edge_list;
            fmt::print("  Computing curvature for all {} edges...\n", edge_list.size());
        }
        std::cout.flush();
        
        auto t4 = std::chrono::high_resolution_clock::now();
        curv_values.reserve(edges_to_eval.size());
        for (size_t ei = 0; ei < edges_to_eval.size(); ++ei) {
            auto& [u, v] = edges_to_eval[ei];
            curv_values.push_back(compute_ollivier_ricci(g, u, v, 0.5f));
            if ((ei + 1) % 500 == 0) {
                fmt::print("  ... {}/{} edges done\n", ei + 1, edges_to_eval.size());
                std::cout.flush();
            }
        }
        auto t5 = std::chrono::high_resolution_clock::now();
        curv_ms = std::chrono::duration<double, std::milli>(t5 - t4).count();
    
        n_curv = curv_values.size();
        curv_min = 1e9f; curv_max = -1e9f;
        for (float k : curv_values) {
            curv_mean += k;
            curv_min = std::min(curv_min, k);
            curv_max = std::max(curv_max, k);
            if (k > 1e-6f) n_pos++;
            else if (k < -1e-6f) n_neg++;
            else n_zero++;
        }
        curv_mean /= std::max((size_t)1, n_curv);
        for (float k : curv_values) curv_std += (k - curv_mean) * (k - curv_mean);
        curv_std = std::sqrt(curv_std / std::max((size_t)1, n_curv));
        std::vector<float> sorted_curv = curv_values;
        std::sort(sorted_curv.begin(), sorted_curv.end());
        curv_median = n_curv > 0 ? sorted_curv[n_curv / 2] : 0.0f;
    
        fmt::print("  <κ>      = {:.6f}\n", curv_mean);
        fmt::print("  σ(κ)     = {:.6f}\n", curv_std);
        fmt::print("  min(κ)   = {:.6f}\n", curv_min);
        fmt::print("  max(κ)   = {:.6f}\n", curv_max);
        fmt::print("  median(κ)= {:.6f}\n", curv_median);
        fmt::print("  #edges   = {}{}\n", n_curv, curv_sampled ? " (SAMPLED)" : "");
        fmt::print("  #positive= {} ({:.1f}%)\n", n_pos, 100.0 * n_pos / std::max((size_t)1, n_curv));
        fmt::print("  #negative= {} ({:.1f}%)\n", n_neg, 100.0 * n_neg / std::max((size_t)1, n_curv));
        fmt::print("  #zero    = {} ({:.1f}%)\n", n_zero, 100.0 * n_zero / std::max((size_t)1, n_curv));
        fmt::print("  (computed in {:.1f} ms)\n\n", curv_ms);
    }

    // --- E) Spectral dimension ---
    fmt::print("── (e) Spectral Dimension ──\n");
    SpectralDimensionResult sd{};
    SpectralDimensionResultV2 sd_v2{};
    double sd_ms = 0.0;
    bool sd_skipped = false;
    constexpr uint32_t MAX_NODES_FOR_SD = 100000;
    if (n_final > MAX_NODES_FOR_SD) {
        sd_skipped = true;
        fmt::print("  SKIPPED: graph too large ({} nodes > {} limit)\n\n", n_final, MAX_NODES_FOR_SD);
    } else {
        auto t6 = std::chrono::high_resolution_clock::now();
        sd = compute_spectral_dimension_detailed(g, cfg.walkers, cfg.walk_steps, 0.05f, 0.5f, cfg.seed);
        auto t7 = std::chrono::high_resolution_clock::now();
        sd_ms = std::chrono::duration<double, std::milli>(t7 - t6).count();
        fmt::print("  d_s (v1)     = {:.4f} ± {:.4f}\n", sd.dimension, sd.fit_error);
        fmt::print("  scaling range: t ∈ [{:.0f}, {:.0f}]\n", sd.scaling_range_min, sd.scaling_range_max);
        fmt::print("  (v1 computed in {:.1f} ms)\n", sd_ms);

        // Also run v2 spectral estimator
        auto t8 = std::chrono::high_resolution_clock::now();
        sd_v2 = compute_spectral_dimension_v2(g, cfg.walkers, cfg.walk_steps, cfg.seed);
        auto t9 = std::chrono::high_resolution_clock::now();
        double sd_v2_ms = std::chrono::duration<double, std::milli>(t9 - t8).count();
        fmt::print("  d_s (v2)     = {:.4f} ± {:.4f}  (plateau={})\n",
                   sd_v2.d_s, sd_v2.d_s_error, sd_v2.has_plateau ? "yes" : "no");
        if (sd_v2.has_plateau)
            fmt::print("  plateau range: t ∈ [{:.0f}, {:.0f}]\n", sd_v2.plateau_t_min, sd_v2.plateau_t_max);
        fmt::print("  global fit   = {:.4f} (R²={:.4f})\n", sd_v2.d_s_global_fit, sd_v2.global_fit_r2);
        fmt::print("  (v2 computed in {:.1f} ms)\n", sd_v2_ms);

        // Save spectral detail if requested
        if (!cfg.spectral_detail_file.empty()) {
            json sd_json;
            sd_json["d_s"] = sd_v2.d_s;
            sd_json["d_s_error"] = sd_v2.d_s_error;
            sd_json["has_plateau"] = sd_v2.has_plateau;
            sd_json["plateau_t_min"] = sd_v2.plateau_t_min;
            sd_json["plateau_t_max"] = sd_v2.plateau_t_max;
            sd_json["d_s_global_fit"] = sd_v2.d_s_global_fit;
            sd_json["global_fit_r2"] = sd_v2.global_fit_r2;
            sd_json["t"] = sd_v2.time_pts;
            sd_json["P_t"] = sd_v2.P_t;
            sd_json["d_eff"] = sd_v2.d_eff_t;
            sd_json["n_nodes"] = n_final;
            sd_json["walkers"] = cfg.walkers;
            sd_json["walk_steps"] = cfg.walk_steps;
            std::ofstream sdf(cfg.spectral_detail_file);
            sdf << sd_json.dump(2) << std::endl;
            fmt::print("  Spectral detail saved to {}\n", cfg.spectral_detail_file);
        }
        fmt::print("\n");
    }

    // --- F) Geodesic analysis ---
    fmt::print("── (f) Geodesic Analysis ──\n");
    double avg_path = 0.0;
    uint32_t diameter = 0;
    std::vector<double> vol_growth;
    double hausdorff_dim = 0.0;
    std::vector<uint64_t> dist_dist;
    bool geodesic_skipped = false;
    double geodesic_ms = 0.0;
    constexpr uint32_t MAX_NODES_FOR_GEODESIC = 5000;

    if (n_final > MAX_NODES_FOR_GEODESIC) {
        geodesic_skipped = true;
        fmt::print("  SKIPPED: graph too large ({} nodes > {} limit for APSP)\n\n", n_final, MAX_NODES_FOR_GEODESIC);
    } else if (!connected) {
        geodesic_skipped = true;
        fmt::print("  SKIPPED: graph is disconnected\n\n");
    } else {
        auto tg0 = std::chrono::high_resolution_clock::now();
        avg_path = compute_average_path_length(g);
        vol_growth = compute_volume_growth(g);
        hausdorff_dim = estimate_hausdorff_dimension(g);
        dist_dist = compute_distance_distribution(g);
        diameter = vol_growth.size() > 0 ? static_cast<uint32_t>(vol_growth.size() - 1) : 0;
        auto tg1 = std::chrono::high_resolution_clock::now();
        geodesic_ms = std::chrono::duration<double, std::milli>(tg1 - tg0).count();

        fmt::print("  Diameter          = {}\n", diameter);
        fmt::print("  <path length>     = {:.3f}\n", avg_path);
        fmt::print("  Hausdorff dim d_H = {:.3f}\n", hausdorff_dim);
        fmt::print("  Volume growth N(r):\n");
        for (size_t r = 0; r < vol_growth.size() && r <= 10; ++r) {
            fmt::print("    r={:2d}: N(r)={:.1f}\n", r, vol_growth[r]);
        }
        if (vol_growth.size() > 11) fmt::print("    ... (up to r={})\n", vol_growth.size() - 1);
        fmt::print("  Distance distribution:\n");
        for (size_t d = 1; d < dist_dist.size() && d <= 10; ++d) {
            fmt::print("    d={:2d}: {} pairs\n", d, dist_dist[d]);
        }
        fmt::print("  (computed in {:.1f} ms)\n\n", geodesic_ms);
    }

    // --- G) Metric tensor & scalar curvature ---
    fmt::print("── (g) Metric Tensor & Scalar Curvature ──\n");
    double scalar_curv = 0.0;
    bool mds_skipped = false;
    double mds_ms = 0.0;
    constexpr uint32_t MAX_NODES_FOR_MDS = 3000;

    if (n_final > MAX_NODES_FOR_MDS) {
        mds_skipped = true;
        fmt::print("  SKIPPED: graph too large ({} nodes > {} limit for MDS)\n\n", n_final, MAX_NODES_FOR_MDS);
    } else if (!connected) {
        mds_skipped = true;
        fmt::print("  SKIPPED: graph is disconnected\n\n");
    } else {
        auto tm0 = std::chrono::high_resolution_clock::now();
        try {
            int embed_dim = std::min(static_cast<int>(cfg.target_dim), 4);
            auto metric = compute_metric_tensor(g, embed_dim);
            scalar_curv = estimate_scalar_curvature(g, embed_dim);
            auto tm1 = std::chrono::high_resolution_clock::now();
            mds_ms = std::chrono::duration<double, std::milli>(tm1 - tm0).count();

            int d = static_cast<int>(metric.rows());
            fmt::print("  Metric tensor ({}D embedding):\n", d);
            for (int i = 0; i < d; ++i) {
                fmt::print("    [");
                for (int j = 0; j < d; ++j)
                    fmt::print(" {:8.4f}", metric(i, j));
                fmt::print(" ]\n");
            }
            auto eigenvalues = Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd>(metric).eigenvalues();
            fmt::print("  Eigenvalues: [");
            for (int i = 0; i < d; ++i) {
                if (i > 0) fmt::print(", ");
                fmt::print("{:.4f}", eigenvalues(i));
            }
            fmt::print("]\n");
            fmt::print("  Scalar curvature R = {:.6f}\n", scalar_curv);
            fmt::print("  (computed in {:.1f} ms)\n\n", mds_ms);
        } catch (const std::exception& e) {
            mds_skipped = true;
            fmt::print("  ERROR: {}\n\n", e.what());
        }
    }

    // --- H) Size stability ---
    fmt::print("── (h) Size Stability ──\n");
    double mean_n = 0.0;
    for (uint32_t nn : node_history) mean_n += nn;
    mean_n /= node_history.size();
    double var_n = 0.0;
    for (uint32_t nn : node_history) {
        double diff = nn - mean_n;
        var_n += diff * diff;
    }
    var_n /= node_history.size();
    double stability = var_n / (mean_n * mean_n);

    fmt::print("  <N_t>             = {:.1f}\n", mean_n);
    fmt::print("  Var[N_t]          = {:.1f}\n", var_n);
    fmt::print("  Var[N_t]/<N_t>²   = {:.6f}\n", stability);
    fmt::print("  N_min during meas = {}\n", *std::min_element(node_history.begin(), node_history.end()));
    fmt::print("  N_max during meas = {}\n", *std::max_element(node_history.begin(), node_history.end()));
    fmt::print("\n");

    // --- I) FITNESS BREAKDOWN ---
    fmt::print("══════════════════════════════════════════════════════\n");
    fmt::print("              FITNESS COMPONENT BREAKDOWN             \n");
    fmt::print("══════════════════════════════════════════════════════\n\n");

    FitnessParams fp;
    fp.target_dimension = cfg.target_dim;
    fp.target_avg_curvature = cfg.target_curv;
    fp.target_size = static_cast<double>(N0);
    fp.spectral_walkers = cfg.walkers;
    fp.spectral_steps = cfg.walk_steps;

    // Compute fitness terms manually from already-measured values
    double F_ricci = curv_skipped ? -99.0 : -fp.weight_ricci * std::abs((double)curv_mean - fp.target_avg_curvature);
    double F_dim = sd_skipped ? -99.0 : -fp.weight_dim * std::abs(sd.dimension - fp.target_dimension);
    double F_conn = connected ? 0.0 : -fp.weight_connectivity * (double)components.size();
    double F_deg = -fp.weight_degree_reg * cv_deg;
    double size_ratio = (double)n_final / fp.target_size;
    double F_size = -fp.weight_size * std::abs(std::log(std::max(size_ratio, 0.01)));
    double F_total = F_ricci + F_dim + F_conn + F_deg + F_size;

    fmt::print("  Weights: w_ricci={:.1f}, w_dim={:.1f}, w_conn={:.1f}, w_deg={:.1f}, w_size={:.1f}\n",
               fp.weight_ricci, fp.weight_dim, fp.weight_connectivity, fp.weight_degree_reg, fp.weight_size);
    fmt::print("\n");
    if (curv_skipped)
        fmt::print("  F_ricci       = N/A     (graph too large for Ollivier-Ricci)\n");
    else
        fmt::print("  F_ricci       = {:.6f}  (|<κ> - target| = {:.4f})\n",
                   F_ricci, std::abs((double)curv_mean - fp.target_avg_curvature));
    if (sd_skipped)
        fmt::print("  F_dimension   = N/A     (graph too large for spectral dimension)\n");
    else
        fmt::print("  F_dimension   = {:.6f}  (|d_s - target| = {:.4f})\n",
                   F_dim, std::abs(sd.dimension - fp.target_dimension));
    fmt::print("  F_connectivity= {:.6f}  (connected={})\n", F_conn, connected);
    fmt::print("  F_degree_reg  = {:.6f}  (CV_deg = {:.4f})\n", F_deg, cv_deg);
    fmt::print("  F_size        = {:.6f}  (N={}, target={})\n", F_size, n_final, (int)fp.target_size);
    fmt::print("  ─────────────────────────────────────\n");
    fmt::print("  F_total       = {:.6f}\n\n", F_total);

    // Identify dominant component
    struct Component { std::string name; double value; };
    std::vector<Component> comps = {
        {"F_ricci", F_ricci}, {"F_dimension", F_dim},
        {"F_connectivity", F_conn}, {"F_degree_reg", F_deg},
        {"F_size", F_size}
    };
    std::sort(comps.begin(), comps.end(), [](const Component& a, const Component& b) {
        return a.value < b.value;
    });
    fmt::print("  Dominant penalty: {} = {:.6f} ({:.1f}% of total)\n",
               comps[0].name, comps[0].value,
               (F_total != 0.0) ? 100.0 * comps[0].value / F_total : 0.0);
    fmt::print("  Second penalty:   {} = {:.6f} ({:.1f}% of total)\n\n",
               comps[1].name, comps[1].value,
               (F_total != 0.0) ? 100.0 * comps[1].value / F_total : 0.0);

    // Histogram with 20 bins
    int nbins = 20;
    float cmin = curv_min, cmax = curv_max;
    if (cmin == cmax) { cmin -= 0.1f; cmax += 0.1f; }
    float bin_w = (cmax - cmin) / nbins;
    std::vector<int> hist(nbins, 0);
    for (float k : curv_values) {
        int bin = static_cast<int>((k - cmin) / bin_w);
        bin = std::clamp(bin, 0, nbins - 1);
        hist[bin]++;
    }
    fmt::print("── (h) Curvature Histogram ──\n");
    for (int i = 0; i < nbins; ++i) {
        float lo = cmin + i * bin_w;
        float hi = lo + bin_w;
        int bar_len = (curv_values.size() > 0) ? 40 * hist[i] / (int)curv_values.size() : 0;
        std::string bar(bar_len, '#');
        fmt::print("  [{:+.3f}, {:+.3f}): {:4d} {}\n", lo, hi, hist[i], bar);
    }
    fmt::print("\n");

    // --- Save JSON output ---
    if (!cfg.output_file.empty()) {
        json out;
        out["params"] = params_d;
        out["config"]["total_steps"] = cfg.total_steps;
        out["config"]["transient_steps"] = cfg.transient_steps;
        out["config"]["graph_side"] = cfg.graph_side;
        out["config"]["seed"] = cfg.seed;
        out["config"]["target_dim"] = cfg.target_dim;
        out["config"]["target_curv"] = cfg.target_curv;

        out["graph"]["N_initial"] = N0;
        out["graph"]["N_final"] = n_final;
        out["graph"]["E_final"] = m_final;
        out["graph"]["connected"] = connected;
        out["graph"]["num_components"] = components.size();
        out["graph"]["mean_degree"] = mean_deg;
        out["graph"]["std_degree"] = std::sqrt(var_deg);
        out["graph"]["cv_degree"] = cv_deg;

        out["curvature"]["mean"] = curv_mean;
        out["curvature"]["std"] = curv_std;
        out["curvature"]["min"] = curv_min;
        out["curvature"]["max"] = curv_max;
        out["curvature"]["median"] = curv_median;
        out["curvature"]["values"] = curv_values;

        out["spectral"]["dimension"] = sd.dimension;
        out["spectral"]["fit_error"] = sd.fit_error;
        out["spectral"]["log_time"] = sd.log_time;
        out["spectral"]["log_prob"] = sd.log_prob;

        out["stability"]["mean_N"] = mean_n;
        out["stability"]["var_N"] = var_n;
        out["stability"]["var_over_mean_sq"] = stability;
        out["stability"]["node_history"] = node_history;
        out["stability"]["edge_history"] = edge_history;

        out["fitness"]["total"] = F_total;
        out["fitness"]["ricci_term"] = F_ricci;
        out["fitness"]["dimension_term"] = F_dim;
        out["fitness"]["connectivity_term"] = F_conn;
        out["fitness"]["degree_reg_term"] = F_deg;
        out["fitness"]["size_term"] = F_size;

        // Geodesic data
        if (!geodesic_skipped) {
            out["geodesic"]["diameter"] = diameter;
            out["geodesic"]["avg_path_length"] = avg_path;
            out["geodesic"]["hausdorff_dimension"] = hausdorff_dim;
            out["geodesic"]["volume_growth"] = vol_growth;
            out["geodesic"]["distance_distribution"] = dist_dist;
        }

        // Metric tensor data
        if (!mds_skipped) {
            out["metric"]["scalar_curvature"] = scalar_curv;
        }

        // Degree histogram
        json deg_j;
        for (auto& [d, cnt] : deg_hist) deg_j[std::to_string(d)] = cnt;
        out["degree_histogram"] = deg_j;

        // Curvature histogram
        json curv_hist_j;
        for (int i = 0; i < nbins; ++i) {
            curv_hist_j.push_back({
                {"bin_lo", cmin + i * bin_w},
                {"bin_hi", cmin + (i + 1) * bin_w},
                {"count", hist[i]}
            });
        }
        out["curvature_histogram"] = curv_hist_j;

        std::ofstream of(cfg.output_file);
        of << out.dump(2) << std::endl;
        fmt::print("Diagnostics saved to {}\n", cfg.output_file);
    }

    fmt::print("══════════════════════════════════════════════════════\n");
    fmt::print("  Done.\n");
    fmt::print("══════════════════════════════════════════════════════\n");

    return 0;
}
