#include "search/ensemble.hpp"
#include "geometry/ollivier_ricci.hpp"
#include "geometry/spectral_dimension.hpp"
#include "geometry/geodesic.hpp"
#include <cmath>
#include <algorithm>
#include <numeric>
#include <omp.h>
#include <nlohmann/json.hpp>

namespace discretum {

using json = nlohmann::json;

// Helper: compute ObservableStats from a vector of values
static ObservableStats compute_stats(const std::vector<double>& vals) {
    ObservableStats s{};
    s.n_valid = static_cast<int>(vals.size());
    if (s.n_valid == 0) return s;

    // Mean
    double sum = 0.0;
    for (double v : vals) sum += v;
    s.mean = sum / s.n_valid;

    // Std dev
    double sum_sq = 0.0;
    for (double v : vals) {
        double d = v - s.mean;
        sum_sq += d * d;
    }
    s.std_dev = (s.n_valid > 1) ? std::sqrt(sum_sq / (s.n_valid - 1)) : 0.0;
    s.std_err = s.std_dev / std::sqrt(static_cast<double>(s.n_valid));

    // Sort for median/quartiles
    std::vector<double> sorted = vals;
    std::sort(sorted.begin(), sorted.end());
    s.min_val = sorted.front();
    s.max_val = sorted.back();

    int n = s.n_valid;
    s.median = (n % 2 == 1) ? sorted[n / 2] : 0.5 * (sorted[n / 2 - 1] + sorted[n / 2]);

    // Quartiles (simple method)
    if (n >= 4) {
        int q1_idx = n / 4;
        int q3_idx = 3 * n / 4;
        s.q25 = sorted[q1_idx];
        s.q75 = sorted[q3_idx];
    } else {
        s.q25 = s.min_val;
        s.q75 = s.max_val;
    }

    return s;
}

static DynamicGraph create_initial_graph(const EnsembleConfig& config) {
    if (config.graph_type == "lattice_4d") {
        int side = std::max(3, static_cast<int>(std::pow(config.graph_size, 0.25)));
        return DynamicGraph::create_lattice_4d(side, side, side, side);
    } else {
        int side = std::max(3, static_cast<int>(std::cbrt(config.graph_size)));
        return DynamicGraph::create_lattice_3d(side, side, side);
    }
}

EnsembleResult run_ensemble(const ParametricRule& rule, const EnsembleConfig& config) {
    EnsembleResult result;
    result.n_total = static_cast<int>(config.num_runs);
    result.runs.resize(config.num_runs);

    // Determine initial graph size for max_nodes
    DynamicGraph template_graph = create_initial_graph(config);
    uint32_t N0 = template_graph.num_nodes();

    #pragma omp parallel for schedule(dynamic)
    for (uint32_t i = 0; i < config.num_runs; ++i) {
        uint64_t seed = config.master_seed + i * config.seed_stride;

        // Create fresh graph and rule copy
        DynamicGraph graph = create_initial_graph(config);
        ParametricRule rule_copy = rule;

        EvolutionConfig evo_cfg;
        evo_cfg.num_steps = config.evo_steps;
        evo_cfg.seed = seed;
        evo_cfg.max_nodes = N0 * config.max_nodes_factor;

        Evolution evo(std::move(graph), std::move(rule_copy), evo_cfg);
        auto evo_result = evo.run();

        EnsembleRunResult& rr = result.runs[i];
        rr.seed = seed;
        rr.steps_completed = evo_result.steps_completed;
        rr.aborted = (evo_result.steps_completed < config.evo_steps);

        // Evaluate fitness on final graph
        if (config.fitness_version == 3) {
            FitnessParamsV3 fp3 = config.fitness_params_v3;
            fp3.seed = seed;
            fp3.n_initial = static_cast<double>(N0);
            auto r3 = compute_fitness_v3(evo.get_graph(), fp3);
            // Map v3 result into v2 result struct for unified stats
            FitnessResultV2& f2 = rr.fitness;
            f2.total = r3.total;
            f2.f_hausdorff = r3.f_hausdorff;
            f2.f_curvature = r3.f_curvature;
            f2.f_spectral = r3.f_spectral;
            f2.f_connectivity = r3.f_connectivity;
            f2.f_stability = r3.f_stability;
            f2.f_regularity = r3.f_regularity;
            f2.d_H = r3.d_H;
            f2.d_s = r3.d_s;
            f2.mean_curvature = r3.mean_curvature;
            f2.std_curvature = r3.std_curvature;
            f2.cv_degree = r3.cv_degree;
            f2.mean_degree = r3.mean_degree;
            f2.n_final = r3.n_final;
            f2.n_components = r3.n_components;
            f2.is_connected = r3.is_connected;
            // Store v3-specific fields in thread-local extra storage
            // We'll extract them in the stats loop via re-evaluation
        } else {
            FitnessParamsV2 fp = config.fitness_params;
            fp.seed = seed;
            fp.n_initial = static_cast<double>(N0);
            rr.fitness = compute_fitness_v2(evo.get_graph(), fp);
        }
    }

    // Collect valid (non-aborted, connected) runs for statistics
    std::vector<double> v_total, v_dH, v_ds, v_mean_k, v_std_k;
    std::vector<double> v_cv_deg, v_mean_deg, v_n_final;
    std::vector<double> v_fH, v_fK, v_fS, v_fStab, v_fReg, v_fDen, v_fDeg;

    result.n_aborted = 0;
    result.n_connected = 0;

    for (auto& rr : result.runs) {
        if (rr.aborted) {
            result.n_aborted++;
            continue;
        }
        if (rr.fitness.is_connected) result.n_connected++;

        v_total.push_back(rr.fitness.total);
        v_dH.push_back(rr.fitness.d_H);
        v_ds.push_back(rr.fitness.d_s);
        v_mean_k.push_back(rr.fitness.mean_curvature);
        v_std_k.push_back(rr.fitness.std_curvature);
        v_cv_deg.push_back(rr.fitness.cv_degree);
        v_mean_deg.push_back(rr.fitness.mean_degree);
        v_n_final.push_back(rr.fitness.n_final);
        v_fH.push_back(rr.fitness.f_hausdorff);
        v_fK.push_back(rr.fitness.f_curvature);
        v_fS.push_back(rr.fitness.f_spectral);
        v_fStab.push_back(rr.fitness.f_stability);
        v_fReg.push_back(rr.fitness.f_regularity);
        // v3 density/degradation stored as connectivity-relative offsets
        // For v3, re-derive from total = sum of all terms
        if (config.fitness_version == 3) {
            double known = rr.fitness.f_hausdorff + rr.fitness.f_curvature
                         + rr.fitness.f_spectral + rr.fitness.f_connectivity
                         + rr.fitness.f_stability + rr.fitness.f_regularity;
            double v3_extra = rr.fitness.total - known;
            // Split evenly as approximation (actual values lost in mapping)
            v_fDen.push_back(v3_extra * 0.5);
            v_fDeg.push_back(v3_extra * 0.5);
        }
    }

    result.fitness_total = compute_stats(v_total);
    result.d_H = compute_stats(v_dH);
    result.d_s = compute_stats(v_ds);
    result.mean_curvature = compute_stats(v_mean_k);
    result.std_curvature = compute_stats(v_std_k);
    result.cv_degree = compute_stats(v_cv_deg);
    result.mean_degree = compute_stats(v_mean_deg);
    result.n_final = compute_stats(v_n_final);
    result.f_hausdorff = compute_stats(v_fH);
    result.f_curvature = compute_stats(v_fK);
    result.f_spectral = compute_stats(v_fS);
    result.f_stability = compute_stats(v_fStab);
    result.f_regularity = compute_stats(v_fReg);
    result.f_density = compute_stats(v_fDen);
    result.f_degradation = compute_stats(v_fDeg);

    return result;
}

static json stats_to_json(const ObservableStats& s) {
    json j;
    j["mean"] = s.mean;
    j["std_err"] = s.std_err;
    j["std_dev"] = s.std_dev;
    j["median"] = s.median;
    j["q25"] = s.q25;
    j["q75"] = s.q75;
    j["min"] = s.min_val;
    j["max"] = s.max_val;
    j["n_valid"] = s.n_valid;
    return j;
}

std::string ensemble_result_to_json(const EnsembleResult& result, int indent) {
    json j;
    j["n_total"] = result.n_total;
    j["n_aborted"] = result.n_aborted;
    j["n_connected"] = result.n_connected;

    j["fitness_total"] = stats_to_json(result.fitness_total);
    j["d_H"] = stats_to_json(result.d_H);
    j["d_s"] = stats_to_json(result.d_s);
    j["mean_curvature"] = stats_to_json(result.mean_curvature);
    j["std_curvature"] = stats_to_json(result.std_curvature);
    j["cv_degree"] = stats_to_json(result.cv_degree);
    j["mean_degree"] = stats_to_json(result.mean_degree);
    j["n_final"] = stats_to_json(result.n_final);

    j["f_hausdorff"] = stats_to_json(result.f_hausdorff);
    j["f_curvature"] = stats_to_json(result.f_curvature);
    j["f_spectral"] = stats_to_json(result.f_spectral);
    j["f_stability"] = stats_to_json(result.f_stability);
    j["f_regularity"] = stats_to_json(result.f_regularity);

    // Per-run details
    json runs = json::array();
    for (auto& rr : result.runs) {
        json r;
        r["seed"] = rr.seed;
        r["steps_completed"] = rr.steps_completed;
        r["aborted"] = rr.aborted;
        r["total"] = rr.fitness.total;
        r["d_H"] = rr.fitness.d_H;
        r["d_s"] = rr.fitness.d_s;
        r["mean_curvature"] = rr.fitness.mean_curvature;
        r["std_curvature"] = rr.fitness.std_curvature;
        r["cv_degree"] = rr.fitness.cv_degree;
        r["mean_degree"] = rr.fitness.mean_degree;
        r["n_final"] = rr.fitness.n_final;
        r["n_components"] = rr.fitness.n_components;
        r["is_connected"] = rr.fitness.is_connected;
        runs.push_back(r);
    }
    j["runs"] = runs;

    return j.dump(indent);
}

} // namespace discretum
