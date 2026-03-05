#include "geometry/spectral_dimension.hpp"
#include "utils/random.hpp"
#include "utils/logging.hpp"
#include "utils/timer.hpp"
#include <cmath>
#include <numeric>
#include <algorithm>
#include <unordered_set>

namespace discretum {

std::vector<uint32_t> run_random_walk(const DynamicGraph& graph,
                                     uint32_t start_node,
                                     uint32_t max_steps,
                                     PCG32& rng) {
    std::vector<uint32_t> return_times;
    uint32_t current_node = start_node;
    
    for (uint32_t step = 1; step <= max_steps; ++step) {
        // Get neighbors
        auto neighbors = graph.neighbors(current_node);
        if (neighbors.empty()) {
            // Isolated node - always at origin
            return_times.push_back(step);
            continue;
        }
        
        // Choose random neighbor
        uint32_t next_idx = rng.uniform(neighbors.size());
        current_node = neighbors[next_idx];
        
        // Record if at origin
        if (current_node == start_node) {
            return_times.push_back(step);
        }
    }
    
    return return_times;
}

/**
 * @brief Run random walk and record position at each step
 * 
 * Returns a vector of booleans: at_origin[t] = true if walker is at start_node at step t
 */
std::vector<bool> run_random_walk_trace(const DynamicGraph& graph,
                                        uint32_t start_node,
                                        uint32_t max_steps,
                                        PCG32& rng) {
    std::vector<bool> at_origin(max_steps + 1, false);
    at_origin[0] = true;
    uint32_t current_node = start_node;
    
    for (uint32_t step = 1; step <= max_steps; ++step) {
        auto neighbors = graph.neighbors(current_node);
        if (neighbors.empty()) {
            at_origin[step] = true;
            continue;
        }
        
        uint32_t next_idx = rng.uniform(neighbors.size());
        current_node = neighbors[next_idx];
        at_origin[step] = (current_node == start_node);
    }
    
    return at_origin;
}

std::pair<std::pair<float, float>, float> linear_regression(const std::vector<float>& x, 
                                                           const std::vector<float>& y) {
    if (x.size() != y.size() || x.size() < 2) {
        return {{0.0f, 0.0f}, 0.0f};
    }
    
    const size_t n = x.size();
    
    // Calculate means
    float mean_x = std::accumulate(x.begin(), x.end(), 0.0f) / n;
    float mean_y = std::accumulate(y.begin(), y.end(), 0.0f) / n;
    
    // Calculate covariance and variance
    float cov_xy = 0.0f;
    float var_x = 0.0f;
    float var_y = 0.0f;
    
    for (size_t i = 0; i < n; ++i) {
        float dx = x[i] - mean_x;
        float dy = y[i] - mean_y;
        cov_xy += dx * dy;
        var_x += dx * dx;
        var_y += dy * dy;
    }
    
    if (var_x < 1e-10f) {
        return {{mean_y, 0.0f}, 0.0f};
    }
    
    // Calculate slope and intercept
    float slope = cov_xy / var_x;
    float intercept = mean_y - slope * mean_x;
    
    // Calculate R-squared
    float r_squared = 0.0f;
    if (var_y > 1e-10f) {
        r_squared = (cov_xy * cov_xy) / (var_x * var_y);
    }
    
    return {{intercept, slope}, r_squared};
}

SpectralDimensionResult compute_spectral_dimension_detailed(const DynamicGraph& graph,
                                                           uint32_t num_walkers,
                                                           uint32_t max_steps,
                                                           float fit_start_fraction,
                                                           float fit_end_fraction,
                                                           uint64_t seed) {
    TIMED_SCOPE("compute_spectral_dimension");
    
    SpectralDimensionResult result{};
    
    if (graph.num_nodes() == 0 || num_walkers == 0 || max_steps == 0) {
        spdlog::warn("compute_spectral_dimension: Invalid parameters");
        return result;
    }
    
    // Initialize random number generator
    RandomGenerator::initialize(seed);
    
    // Count returns at each time step
    std::vector<uint64_t> return_counts(max_steps + 1, 0);
    std::vector<uint64_t> walker_counts(max_steps + 1, 0);
    
    // Get list of all nodes with nonzero degree
    std::vector<uint32_t> all_nodes;
    for (uint32_t i = 0; i < graph.get_nodes().size(); ++i) {
        if (graph.degree(i) > 0) {
            all_nodes.push_back(i);
        }
    }
    
    if (all_nodes.empty()) {
        spdlog::warn("compute_spectral_dimension: No connected nodes in graph");
        return result;
    }
    
    // Run random walks using trace-based approach
    // P(t) = average over walkers of (1 if walker is at origin at step t, 0 otherwise)
    for (uint32_t w = 0; w < num_walkers; ++w) {
        // Create independent RNG for each walker with unique seed
        PCG32 local_rng(seed + w * 6364136223846793005ULL + 1);
        
        // Choose random starting node
        uint32_t start_idx = local_rng.uniform(all_nodes.size());
        uint32_t start_node = all_nodes[start_idx];
        
        // Run walk and get full trace
        auto trace = run_random_walk_trace(graph, start_node, max_steps, local_rng);
        
        // Update counts
        for (uint32_t t = 1; t <= max_steps; ++t) {
            walker_counts[t]++;
            if (trace[t]) {
                return_counts[t]++;
            }
        }
    }
    
    // Calculate return probabilities — only include time steps with nonzero P(t).
    // On bipartite graphs (e.g. lattices), P(t) is zero at odd t, so we skip those.
    result.time_points.reserve(max_steps);
    result.return_probs.reserve(max_steps);
    
    for (uint32_t t = 2; t <= max_steps; ++t) {
        if (walker_counts[t] > 0 && return_counts[t] > 0) {
            float prob = static_cast<float>(return_counts[t]) / walker_counts[t];
            result.time_points.push_back(static_cast<float>(t));
            result.return_probs.push_back(prob);
        }
    }
    
    // Prepare log-log data for fitting
    for (size_t i = 0; i < result.time_points.size(); ++i) {
        result.log_time.push_back(std::log(result.time_points[i]));
        result.log_prob.push_back(std::log(result.return_probs[i]));
    }
    
    // Determine fitting range.
    // To mitigate finite-size effects, cap fit_end so that we only use
    // time steps t < sqrt(N).  For small graphs the random walk saturates
    // very quickly, which biases the slope toward 0.
    size_t n_pts = result.log_time.size();
    float t_max_fit = std::sqrt(static_cast<float>(graph.num_nodes()));
    size_t fit_start = static_cast<size_t>(n_pts * fit_start_fraction);
    size_t fit_end = static_cast<size_t>(n_pts * fit_end_fraction);
    // Restrict fit_end to the last time point <= t_max_fit
    for (size_t i = 0; i < result.time_points.size(); ++i) {
        if (result.time_points[i] > t_max_fit) {
            if (i > fit_start + 3) {
                fit_end = std::min(fit_end, i);
            }
            break;
        }
    }
    
    if (n_pts < 5 || fit_end <= fit_start + 3) {
        // Not enough points for reliable fit
        spdlog::warn("compute_spectral_dimension: Insufficient data points ({}) for fitting", n_pts);
        result.dimension = 0.0f;
        result.fit_error = 1.0f;
        return result;
    }
    
    // Extract fitting range
    std::vector<float> fit_x(result.log_time.begin() + fit_start, 
                            result.log_time.begin() + fit_end);
    std::vector<float> fit_y(result.log_prob.begin() + fit_start, 
                            result.log_prob.begin() + fit_end);
    
    // Perform linear regression on log-log plot
    auto [coeffs, r_squared] = linear_regression(fit_x, fit_y);
    auto [intercept, slope] = coeffs;
    
    // Extract spectral dimension from slope
    // P(t) ~ t^(-d_s/2) => log P(t) = -d_s/2 * log(t) + const
    // Therefore: d_s = -2 * slope
    result.dimension = -2.0f * slope;
    result.fit_error = 1.0f - r_squared;
    
    // Record scaling range
    if (!result.time_points.empty() && fit_start < result.time_points.size() && fit_end > 0) {
        result.scaling_range_min = result.time_points[fit_start];
        result.scaling_range_max = result.time_points[std::min(fit_end - 1, result.time_points.size() - 1)];
    }
    
    // Log results
    spdlog::info("Spectral dimension: d_s = {:.3f} ± {:.3f} (R² = {:.4f})", 
                 result.dimension, std::sqrt(result.fit_error), r_squared);
    spdlog::info("Scaling range: t ∈ [{:.0f}, {:.0f}]", 
                 result.scaling_range_min, result.scaling_range_max);
    
    return result;
}

SpectralDimensionResult compute_spectral_dimension(const DynamicGraph& graph, 
                                                  uint32_t num_walkers, 
                                                  uint32_t max_steps) {
    return compute_spectral_dimension_detailed(graph, num_walkers, max_steps, 
                                             0.05f, 0.5f, 42);
}

// ════════════════════════════════════════════════════════════════
// V2 SPECTRAL DIMENSION — Plateau-finding local derivative method
// ════════════════════════════════════════════════════════════════

SpectralDimensionResultV2 compute_spectral_dimension_v2(
    const DynamicGraph& graph,
    uint32_t num_walkers,
    uint32_t max_steps,
    uint64_t seed)
{
    TIMED_SCOPE("compute_spectral_dimension_v2");

    SpectralDimensionResultV2 result{};
    result.d_s = 0.0;
    result.d_s_error = 1.0;
    result.plateau_t_min = 0.0;
    result.plateau_t_max = 0.0;
    result.has_plateau = false;
    result.d_s_global_fit = 0.0;
    result.global_fit_r2 = 0.0;

    if (graph.num_nodes() == 0 || num_walkers == 0 || max_steps == 0) {
        spdlog::warn("compute_spectral_dimension_v2: invalid parameters");
        return result;
    }

    // Get list of all nodes with nonzero degree
    std::vector<uint32_t> all_nodes;
    for (uint32_t i = 0; i < graph.get_nodes().size(); ++i) {
        if (graph.degree(i) > 0) {
            all_nodes.push_back(i);
        }
    }
    if (all_nodes.empty()) {
        spdlog::warn("compute_spectral_dimension_v2: no connected nodes");
        return result;
    }

    // Step 1: Compute P(t) via random walks
    std::vector<uint64_t> return_counts(max_steps + 1, 0);
    uint64_t total_walkers = num_walkers;

    for (uint32_t w = 0; w < num_walkers; ++w) {
        PCG32 local_rng(seed + w * 6364136223846793005ULL + 1);
        uint32_t start_idx = local_rng.uniform(static_cast<uint32_t>(all_nodes.size()));
        uint32_t start_node = all_nodes[start_idx];

        auto trace = run_random_walk_trace(graph, start_node, max_steps, local_rng);
        for (uint32_t t = 1; t <= max_steps; ++t) {
            if (trace[t]) {
                return_counts[t]++;
            }
        }
    }

    // Step 2: Collect non-zero P(t) points (skip odd t for bipartite graphs)
    std::vector<double> times;
    std::vector<double> probs;
    times.reserve(max_steps);
    probs.reserve(max_steps);

    for (uint32_t t = 2; t <= max_steps; ++t) {
        if (return_counts[t] > 0) {
            double p = static_cast<double>(return_counts[t]) / total_walkers;
            times.push_back(static_cast<double>(t));
            probs.push_back(p);
        }
    }

    result.time_pts = times;
    result.P_t = probs;

    if (times.size() < 10) {
        spdlog::warn("compute_spectral_dimension_v2: only {} data points, need >= 10", times.size());
        return result;
    }

    // Step 3: Compute d_eff(t) = -2 * d(log P) / d(log t) using centred differences
    size_t np = times.size();
    std::vector<double> log_t(np), log_p(np);
    for (size_t i = 0; i < np; ++i) {
        log_t[i] = std::log(times[i]);
        log_p[i] = std::log(probs[i]);
    }

    std::vector<double> d_eff_raw(np, 0.0);
    // Centred differences for interior points, forward/backward for boundaries
    for (size_t i = 1; i + 1 < np; ++i) {
        double dlog_p = log_p[i + 1] - log_p[i - 1];
        double dlog_t = log_t[i + 1] - log_t[i - 1];
        if (std::abs(dlog_t) > 1e-15) {
            d_eff_raw[i] = -2.0 * dlog_p / dlog_t;
        }
    }
    // Forward difference for first point
    if (np >= 2) {
        double dlog_p = log_p[1] - log_p[0];
        double dlog_t = log_t[1] - log_t[0];
        if (std::abs(dlog_t) > 1e-15) d_eff_raw[0] = -2.0 * dlog_p / dlog_t;
    }
    // Backward difference for last point
    if (np >= 2) {
        double dlog_p = log_p[np - 1] - log_p[np - 2];
        double dlog_t = log_t[np - 1] - log_t[np - 2];
        if (std::abs(dlog_t) > 1e-15) d_eff_raw[np - 1] = -2.0 * dlog_p / dlog_t;
    }

    // Step 4: Smooth d_eff with moving average
    // Window size: ~10% of points, minimum 3
    size_t smooth_window = std::max(static_cast<size_t>(3),
                                     static_cast<size_t>(np * 0.1));
    if (smooth_window % 2 == 0) smooth_window++;  // make odd
    size_t half_w = smooth_window / 2;

    std::vector<double> d_eff_smooth(np, 0.0);
    for (size_t i = 0; i < np; ++i) {
        size_t lo = (i >= half_w) ? (i - half_w) : 0;
        size_t hi = std::min(np - 1, i + half_w);
        double sum = 0.0;
        for (size_t j = lo; j <= hi; ++j) sum += d_eff_raw[j];
        d_eff_smooth[i] = sum / static_cast<double>(hi - lo + 1);
    }
    result.d_eff_t = d_eff_smooth;

    // Step 5: Find plateau — widest window with variance below threshold
    // Skip only the first 2 points (discrete lattice effects at t=2,4)
    // and stop before finite-size saturation (P(t) → 1/N at t ~ N)
    size_t search_start = std::min(static_cast<size_t>(3), np / 2);
    // Cut off at t ≈ N/2 where finite-size effects dominate
    double t_saturation = static_cast<double>(graph.num_nodes()) * 0.5;
    size_t search_end = np;
    for (size_t i = 0; i < np; ++i) {
        if (times[i] > t_saturation) {
            search_end = i;
            break;
        }
    }
    if (search_end <= search_start + 5) {
        search_end = np;  // fallback: use everything
    }

    int best_start = -1;
    int best_width = 0;
    double best_mean = 0.0;
    double best_var = 1e30;
    int min_plateau_width = std::max(3, static_cast<int>((search_end - search_start) / 8));

    // Sliding window: try all windows of width >= min_plateau_width
    // Use incremental mean/variance computation for efficiency
    for (int width = static_cast<int>(search_end - search_start);
         width >= min_plateau_width; --width) {
        for (int start = static_cast<int>(search_start);
             start + width <= static_cast<int>(search_end); ++start) {
            // Compute mean and variance in this window
            double sum = 0.0, sum2 = 0.0;
            int count = 0;
            for (int t = start; t < start + width; ++t) {
                double v = d_eff_smooth[t];
                if (v > 0.0) {  // only count positive d_eff
                    sum += v;
                    sum2 += v * v;
                    count++;
                }
            }
            if (count < min_plateau_width) continue;

            double mean = sum / count;
            double var = sum2 / count - mean * mean;
            if (var < 0) var = 0;

            // Relative variance: var / mean^2
            double rel_var = (mean > 0.01) ? var / (mean * mean) : 1e30;

            // Accept this window if it's wider with acceptable variance,
            // or same width with better variance
            if (rel_var < 0.05) {  // CV < ~22%
                if (width > best_width ||
                    (width == best_width && var < best_var)) {
                    best_start = start;
                    best_width = width;
                    best_mean = mean;
                    best_var = var;
                }
                break;  // Found a valid window at this width, move to next
            }
        }
        if (best_width > 0) break;  // Found plateau, prefer widest
    }

    // If no plateau found with strict threshold, relax
    if (best_width == 0) {
        double relaxed_threshold = 0.15;  // CV < ~39%
        for (int width = static_cast<int>(search_end - search_start);
             width >= min_plateau_width; --width) {
            for (int start = static_cast<int>(search_start);
                 start + width <= static_cast<int>(search_end); ++start) {
                double sum = 0.0, sum2 = 0.0;
                int count = 0;
                for (int t = start; t < start + width; ++t) {
                    double v = d_eff_smooth[t];
                    if (v > 0.0) {
                        sum += v;
                        sum2 += v * v;
                        count++;
                    }
                }
                if (count < min_plateau_width) continue;
                double mean = sum / count;
                double var = sum2 / count - mean * mean;
                if (var < 0) var = 0;
                double rel_var = (mean > 0.01) ? var / (mean * mean) : 1e30;
                if (rel_var < relaxed_threshold) {
                    if (width > best_width ||
                        (width == best_width && var < best_var)) {
                        best_start = start;
                        best_width = width;
                        best_mean = mean;
                        best_var = var;
                    }
                    break;
                }
            }
            if (best_width > 0) break;
        }
    }

    if (best_width > 0 && best_start >= 0) {
        result.has_plateau = true;
        result.d_s = best_mean;
        result.d_s_error = std::sqrt(best_var) / std::sqrt(static_cast<double>(best_width));
        result.plateau_t_min = times[best_start];
        result.plateau_t_max = times[std::min(static_cast<size_t>(best_start + best_width - 1), np - 1)];
    }

    // Step 6: Global log-log fit as fallback
    // Use range: skip first 2 points, stop at t_saturation (N/2)
    {
        size_t fit_start = std::min(static_cast<size_t>(2), np / 2);
        size_t fit_end = search_end;  // reuse the saturation cutoff
        if (fit_end > fit_start + 3) {
            std::vector<float> fx_f, fy_f;
            fx_f.reserve(fit_end - fit_start);
            fy_f.reserve(fit_end - fit_start);
            for (size_t i = fit_start; i < fit_end; ++i) {
                fx_f.push_back(static_cast<float>(log_t[i]));
                fy_f.push_back(static_cast<float>(log_p[i]));
            }
            auto [coeffs, r2] = linear_regression(fx_f, fy_f);
            result.d_s_global_fit = -2.0 * coeffs.second;
            result.global_fit_r2 = r2;
        }
    }

    // Use plateau estimate if available, else fall back to global fit
    if (!result.has_plateau) {
        result.d_s = result.d_s_global_fit;
        result.d_s_error = 1.0 - result.global_fit_r2;
    }

    spdlog::info("Spectral dimension v2: d_s = {:.3f} ± {:.3f} (plateau={},"
                 " range=[{:.0f},{:.0f}], global_fit={:.3f}, R²={:.3f})",
                 result.d_s, result.d_s_error, result.has_plateau,
                 result.plateau_t_min, result.plateau_t_max,
                 result.d_s_global_fit, result.global_fit_r2);

    return result;
}

} // namespace discretum
