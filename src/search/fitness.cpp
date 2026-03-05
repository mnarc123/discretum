#include "search/fitness.hpp"
#include "geometry/ollivier_ricci.hpp"
#include "geometry/spectral_dimension.hpp"
#include "geometry/geodesic.hpp"
#include "utils/random.hpp"
#include <cmath>
#include <algorithm>
#include <numeric>
#include <set>

namespace discretum {

FitnessBreakdown compute_fitness_detailed(const DynamicGraph& graph, const FitnessParams& params) {
    FitnessBreakdown fb{};
    
    uint32_t n = graph.num_nodes();
    uint32_t m = graph.num_edges();
    
    // Degenerate graph penalty
    if (n < 3 || m == 0) {
        fb.total = -100.0;
        fb.connectivity_term = -100.0;
        fb.is_connected = false;
        return fb;
    }
    
    // Runaway density penalty: if avg degree > 4*target_dim, skip expensive
    // computations and return a penalty proportional to excess density.
    // This prevents O(n²) Ollivier-Ricci on near-complete graphs.
    double avg_deg = 2.0 * m / n;
    double max_avg_deg = 4.0 * std::max(params.target_dimension, 3.0);
    if (avg_deg > max_avg_deg) {
        double excess = avg_deg / max_avg_deg - 1.0;
        fb.degree_reg_term = -params.weight_degree_reg * excess * 5.0;
        fb.total = -10.0 + fb.degree_reg_term;
        return fb;
    }
    
    // 4. Degree regularity (cheap — compute first)
    double mean_deg = 0.0;
    for (uint32_t i = 0; i < n; ++i) mean_deg += graph.degree(i);
    mean_deg /= n;
    
    double var_deg = 0.0;
    for (uint32_t i = 0; i < n; ++i) {
        double diff = graph.degree(i) - mean_deg;
        var_deg += diff * diff;
    }
    var_deg /= n;
    double cv_deg = (mean_deg > 0) ? std::sqrt(var_deg) / mean_deg : 0.0;
    fb.degree_reg_term = -params.weight_degree_reg * cv_deg;
    
    // 5. Size stability (cheap)
    double size_ratio = static_cast<double>(n) / params.target_size;
    double size_error = std::abs(std::log(std::max(size_ratio, 0.01)));
    fb.size_term = -params.weight_size * size_error;
    
    // 1. Connectivity
    auto components = graph.connected_components();
    fb.is_connected = (components.size() == 1);
    if (!fb.is_connected) {
        fb.connectivity_term = -params.weight_connectivity * static_cast<double>(components.size());
        // For disconnected graphs, skip expensive Ollivier-Ricci and spectral dim.
        // Use large penalties for unknown terms to guide search away.
        fb.ricci_term = -params.weight_ricci * 2.0;
        fb.dimension_term = -params.weight_dim * 5.0;
        fb.total = fb.ricci_term + fb.dimension_term + fb.connectivity_term
                 + fb.degree_reg_term + fb.size_term;
        return fb;
    }
    
    // 2. Ollivier-Ricci average curvature (sampled for large graphs)
    float avg_curv = 0.0f;
    if (params.max_curv_edges > 0 && m > params.max_curv_edges) {
        // Sample edges uniformly
        std::vector<std::pair<uint32_t, uint32_t>> edge_list;
        edge_list.reserve(m);
        std::set<std::pair<uint32_t, uint32_t>> seen;
        for (uint32_t i = 0; i < n; ++i) {
            if (graph.degree(i) == 0) continue;
            for (uint32_t nb : graph.neighbors(i)) {
                auto e = std::minmax(i, nb);
                if (seen.insert(e).second) edge_list.push_back(e);
            }
        }
        // Fisher-Yates partial shuffle
        PCG32 rng(42);
        uint32_t sample_n = std::min(params.max_curv_edges, static_cast<uint32_t>(edge_list.size()));
        for (uint32_t i = 0; i < sample_n; ++i) {
            uint32_t j = i + (rng() % (static_cast<uint32_t>(edge_list.size()) - i));
            std::swap(edge_list[i], edge_list[j]);
        }
        float sum_curv = 0.0f;
        for (uint32_t i = 0; i < sample_n; ++i) {
            sum_curv += compute_ollivier_ricci(graph, edge_list[i].first, edge_list[i].second, params.ollivier_alpha);
        }
        avg_curv = sum_curv / static_cast<float>(sample_n);
    } else {
        avg_curv = compute_average_ollivier_ricci(graph, params.ollivier_alpha);
    }
    fb.measured_avg_curvature = avg_curv;
    double curv_error = std::abs(avg_curv - params.target_avg_curvature);
    fb.ricci_term = -params.weight_ricci * curv_error;
    
    // 3. Spectral dimension
    auto sd_result = compute_spectral_dimension_detailed(
        graph, params.spectral_walkers, params.spectral_steps,
        0.05f, 0.5f, 42);
    fb.measured_dimension = sd_result.dimension;
    double dim_error = std::abs(sd_result.dimension - params.target_dimension);
    fb.dimension_term = -params.weight_dim * dim_error;
    
    // Total fitness (negative; closer to 0 is better)
    fb.total = fb.ricci_term + fb.dimension_term + fb.connectivity_term
             + fb.degree_reg_term + fb.size_term;
    
    return fb;
}

double compute_fitness(const DynamicGraph& graph, const FitnessParams& params) {
    return compute_fitness_detailed(graph, params).total;
}

// ════════════════════════════════════════════════════════════════
// V2 FITNESS FUNCTION
// ════════════════════════════════════════════════════════════════

FitnessResultV2 compute_fitness_v2(const DynamicGraph& graph, const FitnessParamsV2& params) {
    FitnessResultV2 r{};

    uint32_t n = graph.num_nodes();
    uint32_t m = graph.num_edges();
    r.n_final = static_cast<double>(n);

    // Degenerate graph penalty
    if (n < 3 || m == 0) {
        r.total = -100.0;
        r.f_connectivity = -100.0;
        r.n_components = (n < 3) ? n : n;  // each node isolated
        r.is_connected = false;
        return r;
    }

    // Density safeguard: abort if avg degree > 4*target_dim
    double avg_deg = 2.0 * m / n;
    r.mean_degree = avg_deg;
    double max_avg_deg = 4.0 * std::max(params.target_dim, 3.0);
    if (avg_deg > max_avg_deg) {
        double excess = avg_deg / max_avg_deg - 1.0;
        r.f_regularity = -params.w_regularity * excess * excess * 25.0;
        r.total = -10.0 + r.f_regularity;
        return r;
    }

    // 6. Degree regularity (cheap — compute first)
    double mean_deg = 0.0;
    for (uint32_t i = 0; i < n; ++i) mean_deg += graph.degree(i);
    mean_deg /= n;
    r.mean_degree = mean_deg;

    double var_deg = 0.0;
    for (uint32_t i = 0; i < n; ++i) {
        double diff = graph.degree(i) - mean_deg;
        var_deg += diff * diff;
    }
    var_deg /= n;
    r.cv_degree = (mean_deg > 0) ? std::sqrt(var_deg) / mean_deg : 0.0;
    r.f_regularity = -params.w_regularity * r.cv_degree * r.cv_degree;

    // 5. Size stability
    double log_ratio = std::log(std::max(static_cast<double>(n) / params.n_initial, 0.01));
    r.f_stability = -params.w_stability * log_ratio * log_ratio;

    // 4. Connectivity
    auto components = graph.connected_components();
    r.n_components = static_cast<int>(components.size());
    r.is_connected = (components.size() == 1);
    if (!r.is_connected) {
        r.f_connectivity = -params.w_connectivity * static_cast<double>(components.size() - 1);
        // Skip expensive computations for disconnected graphs
        r.f_hausdorff = -params.w_hausdorff * params.target_dim * params.target_dim;
        r.f_curvature = -params.w_curvature * 4.0;
        r.f_spectral = -params.w_spectral * params.target_dim * params.target_dim;
        r.total = r.f_hausdorff + r.f_curvature + r.f_spectral
                + r.f_connectivity + r.f_stability + r.f_regularity;
        return r;
    }

    // 1. Hausdorff dimension (sampled BFS — O(k*(V+E)))
    r.d_H = estimate_hausdorff_dimension_sampled(graph, params.hausdorff_sources, params.seed);
    double dH_err = r.d_H - params.target_dim;
    r.f_hausdorff = -params.w_hausdorff * dH_err * dH_err;

    // 2. Ollivier-Ricci curvature (sampled)
    {
        std::vector<std::pair<uint32_t, uint32_t>> edge_list;
        std::set<std::pair<uint32_t, uint32_t>> seen;
        for (uint32_t i = 0; i < n; ++i) {
            if (graph.degree(i) == 0) continue;
            for (uint32_t nb : graph.neighbors(i)) {
                auto e = std::minmax(i, nb);
                if (seen.insert(e).second) edge_list.push_back(e);
            }
        }

        uint32_t sample_n = std::min(params.curvature_samples,
                                     static_cast<uint32_t>(edge_list.size()));
        if (sample_n > 0 && edge_list.size() > sample_n) {
            PCG32 rng(params.seed + 777);
            for (uint32_t i = 0; i < sample_n; ++i) {
                uint32_t j = i + (rng() % (static_cast<uint32_t>(edge_list.size()) - i));
                std::swap(edge_list[i], edge_list[j]);
            }
        } else {
            sample_n = static_cast<uint32_t>(edge_list.size());
        }

        double sum_k = 0.0, sum_k2 = 0.0;
        for (uint32_t i = 0; i < sample_n; ++i) {
            float k = compute_ollivier_ricci(graph, edge_list[i].first,
                                              edge_list[i].second, params.ollivier_alpha);
            sum_k += k;
            sum_k2 += static_cast<double>(k) * k;
        }
        r.mean_curvature = sum_k / std::max(1u, sample_n);
        double var_k = sum_k2 / std::max(1u, sample_n) - r.mean_curvature * r.mean_curvature;
        r.std_curvature = std::sqrt(std::max(0.0, var_k));

        r.f_curvature = -params.w_curvature * (
            r.mean_curvature * r.mean_curvature
            + params.curvature_fluct_beta * var_k
        );
    }

    // 3. Spectral dimension → concordance with d_H
    {
        auto sd = compute_spectral_dimension_detailed(
            graph, params.spectral_walkers, params.spectral_steps,
            0.05f, 0.5f, params.seed);
        r.d_s = sd.dimension;
        double ds_dH = r.d_s - r.d_H;
        r.f_spectral = -params.w_spectral * ds_dH * ds_dH;
    }

    // Total
    r.total = r.f_hausdorff + r.f_curvature + r.f_spectral
            + r.f_connectivity + r.f_stability + r.f_regularity;

    return r;
}

double compute_fitness_v2_total(const DynamicGraph& graph, const FitnessParamsV2& params) {
    return compute_fitness_v2(graph, params).total;
}

// ════════════════════════════════════════════════════════════════
// V3 FITNESS FUNCTION — Geometry-preserving
// ════════════════════════════════════════════════════════════════

BaselineMetrics compute_baseline_metrics(const DynamicGraph& graph,
                                          double target_dim,
                                          uint32_t hausdorff_sources,
                                          uint32_t spectral_walkers,
                                          uint32_t spectral_steps,
                                          uint32_t curvature_samples,
                                          uint64_t seed) {
    BaselineMetrics bm;
    uint32_t n = graph.num_nodes();
    uint32_t m = graph.num_edges();
    if (n < 3 || m == 0) return bm;

    // Hausdorff dimension
    bm.d_H = estimate_hausdorff_dimension_sampled(graph, hausdorff_sources, seed);

    // Spectral dimension (v2)
    auto sd = compute_spectral_dimension_v2(graph, spectral_walkers, spectral_steps, seed);
    bm.d_s = sd.d_s;

    // Mean degree
    double sum_deg = 0;
    for (uint32_t i = 0; i < n; ++i) sum_deg += graph.degree(i);
    bm.mean_degree = sum_deg / n;

    // Curvature (sampled)
    std::vector<std::pair<uint32_t, uint32_t>> edge_list;
    std::set<std::pair<uint32_t, uint32_t>> seen;
    for (uint32_t i = 0; i < n; ++i) {
        if (graph.degree(i) == 0) continue;
        for (uint32_t nb : graph.neighbors(i)) {
            auto e = std::minmax(i, nb);
            if (seen.insert(e).second) edge_list.push_back(e);
        }
    }
    uint32_t sample_n = std::min(curvature_samples, static_cast<uint32_t>(edge_list.size()));
    if (sample_n > 0 && edge_list.size() > sample_n) {
        PCG32 rng(seed + 777);
        for (uint32_t i = 0; i < sample_n; ++i) {
            uint32_t j = i + (rng() % (static_cast<uint32_t>(edge_list.size()) - i));
            std::swap(edge_list[i], edge_list[j]);
        }
    } else {
        sample_n = static_cast<uint32_t>(edge_list.size());
    }
    double sum_k = 0.0, sum_k2 = 0.0;
    for (uint32_t i = 0; i < sample_n; ++i) {
        float k = compute_ollivier_ricci(graph, edge_list[i].first, edge_list[i].second, 0.5f);
        sum_k += k;
        sum_k2 += static_cast<double>(k) * k;
    }
    bm.mean_curvature = sum_k / std::max(1u, sample_n);
    double var_k = sum_k2 / std::max(1u, sample_n) - bm.mean_curvature * bm.mean_curvature;
    bm.std_curvature = std::sqrt(std::max(0.0, var_k));

    bm.valid = true;
    return bm;
}

FitnessResultV3 compute_fitness_v3(const DynamicGraph& graph, const FitnessParamsV3& params) {
    FitnessResultV3 r{};

    uint32_t n = graph.num_nodes();
    uint32_t m = graph.num_edges();
    r.n_final = static_cast<double>(n);

    // Degenerate graph penalty
    if (n < 3 || m == 0) {
        r.total = -100.0;
        r.f_connectivity = -100.0;
        r.n_components = n;
        r.is_connected = false;
        return r;
    }

    // Mean degree
    double sum_deg = 0;
    for (uint32_t i = 0; i < n; ++i) sum_deg += graph.degree(i);
    double mean_deg = sum_deg / n;
    r.mean_degree = mean_deg;

    // Hard density safeguard: abort if avg degree > 4*target_dim
    double hard_max_deg = 4.0 * std::max(params.target_dim, 3.0);
    if (mean_deg > hard_max_deg) {
        double excess = mean_deg / hard_max_deg - 1.0;
        r.f_density = -params.w_density * excess * excess * 25.0;
        r.total = -10.0 + r.f_density;
        return r;
    }

    // 7. Density penalty: penalise ⟨k⟩ > k_max_target
    double k_max_target = params.density_k_max_factor * params.target_dim;
    double k_excess = std::max(0.0, mean_deg - k_max_target);
    r.f_density = -params.w_density * k_excess * k_excess;

    // 6. Degree regularity
    double var_deg = 0.0;
    for (uint32_t i = 0; i < n; ++i) {
        double diff = graph.degree(i) - mean_deg;
        var_deg += diff * diff;
    }
    var_deg /= n;
    r.cv_degree = (mean_deg > 0) ? std::sqrt(var_deg) / mean_deg : 0.0;
    r.f_regularity = -params.w_regularity * r.cv_degree * r.cv_degree;

    // 5. Size stability
    double log_ratio = std::log(std::max(static_cast<double>(n) / params.n_initial, 0.01));
    r.f_stability = -params.w_stability * log_ratio * log_ratio;

    // 4. Connectivity
    auto components = graph.connected_components();
    r.n_components = static_cast<int>(components.size());
    r.is_connected = (components.size() == 1);
    if (!r.is_connected) {
        r.f_connectivity = -params.w_connectivity * static_cast<double>(components.size() - 1);
        r.f_hausdorff = -params.w_hausdorff * params.target_dim * params.target_dim;
        r.f_curvature = -params.w_curvature * 4.0;
        r.f_spectral = -params.w_spectral * params.target_dim * params.target_dim;
        r.f_degradation = -params.w_degradation * 4.0;
        r.total = r.f_hausdorff + r.f_curvature + r.f_spectral
                + r.f_connectivity + r.f_stability + r.f_regularity
                + r.f_density + r.f_degradation;
        return r;
    }

    // 1. Hausdorff dimension
    r.d_H = estimate_hausdorff_dimension_sampled(graph, params.hausdorff_sources, params.seed);
    double dH_err = r.d_H - params.target_dim;
    r.f_hausdorff = -params.w_hausdorff * dH_err * dH_err;

    // 2. Ollivier-Ricci curvature (sampled)
    {
        std::vector<std::pair<uint32_t, uint32_t>> edge_list;
        std::set<std::pair<uint32_t, uint32_t>> seen;
        for (uint32_t i = 0; i < n; ++i) {
            if (graph.degree(i) == 0) continue;
            for (uint32_t nb : graph.neighbors(i)) {
                auto e = std::minmax(i, nb);
                if (seen.insert(e).second) edge_list.push_back(e);
            }
        }
        uint32_t sample_n = std::min(params.curvature_samples,
                                     static_cast<uint32_t>(edge_list.size()));
        if (sample_n > 0 && edge_list.size() > sample_n) {
            PCG32 rng(params.seed + 777);
            for (uint32_t i = 0; i < sample_n; ++i) {
                uint32_t j = i + (rng() % (static_cast<uint32_t>(edge_list.size()) - i));
                std::swap(edge_list[i], edge_list[j]);
            }
        } else {
            sample_n = static_cast<uint32_t>(edge_list.size());
        }
        double sum_k = 0.0, sum_k2 = 0.0;
        for (uint32_t i = 0; i < sample_n; ++i) {
            float k = compute_ollivier_ricci(graph, edge_list[i].first,
                                              edge_list[i].second, params.ollivier_alpha);
            sum_k += k;
            sum_k2 += static_cast<double>(k) * k;
        }
        r.mean_curvature = sum_k / std::max(1u, sample_n);
        double var_k = sum_k2 / std::max(1u, sample_n) - r.mean_curvature * r.mean_curvature;
        r.std_curvature = std::sqrt(std::max(0.0, var_k));

        r.f_curvature = -params.w_curvature * (
            r.mean_curvature * r.mean_curvature
            + params.curvature_fluct_beta * var_k
        );
    }

    // 3. Spectral dimension (v2 with plateau-finding, or v1 fallback)
    if (params.use_spectral_v2) {
        auto sd = compute_spectral_dimension_v2(
            graph, params.spectral_walkers, params.spectral_steps, params.seed);
        r.d_s = sd.d_s;
        r.d_s_error = sd.d_s_error;
        r.d_s_has_plateau = sd.has_plateau;
    } else {
        auto sd = compute_spectral_dimension_detailed(
            graph, params.spectral_walkers, params.spectral_steps,
            0.05f, 0.5f, params.seed);
        r.d_s = sd.dimension;
        r.d_s_error = sd.fit_error;
        r.d_s_has_plateau = false;
    }
    double ds_dH = r.d_s - r.d_H;
    r.f_spectral = -params.w_spectral * ds_dH * ds_dH;

    // 8. Non-degradation penalty (if baseline is available)
    r.f_degradation = 0.0;
    if (params.baseline.valid) {
        double penalty = 0.0;

        // d_H: penalise if further from target than baseline
        double dH_evolved_err = std::abs(r.d_H - params.target_dim);
        double dH_baseline_err = std::abs(params.baseline.d_H - params.target_dim);
        if (dH_evolved_err > dH_baseline_err) {
            double degradation = dH_evolved_err - dH_baseline_err;
            penalty += degradation * degradation;
        }

        // κ: penalise if |κ| increases relative to baseline
        double kappa_evolved = std::abs(r.mean_curvature);
        double kappa_baseline = std::abs(params.baseline.mean_curvature);
        if (kappa_evolved > kappa_baseline + 0.01) {  // 0.01 tolerance
            double degradation = kappa_evolved - kappa_baseline;
            penalty += degradation * degradation;
        }

        r.f_degradation = -params.w_degradation * penalty;
    }

    // Total
    r.total = r.f_hausdorff + r.f_curvature + r.f_spectral
            + r.f_connectivity + r.f_stability + r.f_regularity
            + r.f_density + r.f_degradation;

    return r;
}

double compute_fitness_v3_total(const DynamicGraph& graph, const FitnessParamsV3& params) {
    return compute_fitness_v3(graph, params).total;
}

} // namespace discretum
