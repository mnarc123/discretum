#pragma once

#include "core/graph.hpp"
#include <vector>
#include <utility>

namespace discretum {

/**
 * @brief Result of spectral dimension calculation
 */
struct SpectralDimensionResult {
    float dimension;              ///< Estimated spectral dimension
    float fit_error;              ///< Error of the power-law fit
    float scaling_range_min;      ///< Minimum time in scaling regime
    float scaling_range_max;      ///< Maximum time in scaling regime
    std::vector<float> time_points;     ///< Time points sampled
    std::vector<float> return_probs;    ///< Return probabilities P(t)
    std::vector<float> log_time;        ///< log(t) for fitting
    std::vector<float> log_prob;        ///< log(P(t)) for fitting
};

/**
 * @brief Compute spectral dimension of a graph
 * 
 * The spectral dimension d_s is defined by the scaling of random walk return probability:
 * P(t) ~ t^(-d_s/2) for large t
 * 
 * where P(t) is the average probability of returning to the starting node after t steps.
 * 
 * References:
 * - Alexander & Orbach (1982) "Density of states on fractals: fractons"
 * - Burioni & Cassi (2005) "Random walks on graphs: ideas, techniques and results"
 * 
 * @param graph The graph
 * @param num_walkers Number of random walkers to simulate
 * @param max_steps Maximum number of steps per walker
 * @return Spectral dimension and fit details
 * 
 * Complexity: O(num_walkers * max_steps * avg_degree)
 */
SpectralDimensionResult compute_spectral_dimension(const DynamicGraph& graph, 
                                                  uint32_t num_walkers = 10000, 
                                                  uint32_t max_steps = 1000);

/**
 * @brief Compute spectral dimension with detailed control
 * 
 * @param graph The graph
 * @param num_walkers Number of random walkers
 * @param max_steps Maximum steps per walker
 * @param fit_start_fraction Fraction of max_steps to start fitting (default 0.1)
 * @param fit_end_fraction Fraction of max_steps to end fitting (default 0.9)
 * @param seed Random seed for reproducibility
 * @return Spectral dimension result
 */
SpectralDimensionResult compute_spectral_dimension_detailed(const DynamicGraph& graph,
                                                           uint32_t num_walkers,
                                                           uint32_t max_steps,
                                                           float fit_start_fraction = 0.1f,
                                                           float fit_end_fraction = 0.9f,
                                                           uint64_t seed = 42);

/**
 * @brief Perform linear regression for log-log fit
 * 
 * Fits y = a + b*x using least squares
 * 
 * @param x Independent variable (log time)
 * @param y Dependent variable (log probability)
 * @return Pair of (intercept, slope) and R-squared value
 */
std::pair<std::pair<float, float>, float> linear_regression(const std::vector<float>& x, 
                                                           const std::vector<float>& y);

/**
 * @brief Run a single random walk and record returns
 * 
 * @param graph The graph
 * @param start_node Starting node
 * @param max_steps Maximum steps
 * @param rng Random number generator
 * @return Vector of return times (steps when walker returned to start)
 */
std::vector<uint32_t> run_random_walk(const DynamicGraph& graph,
                                     uint32_t start_node,
                                     uint32_t max_steps,
                                     class PCG32& rng);

// ════════════════════════════════════════════════════════════════
// V2 SPECTRAL DIMENSION — Plateau-finding local derivative method
// ════════════════════════════════════════════════════════════════

/**
 * @brief Result of spectral dimension v2 calculation.
 *
 * Uses the local effective dimension d_eff(t) = -2 d(log P)/d(log t)
 * and finds the most stable plateau region.
 *
 * Reference: Ambjørn, Jurkiewicz, Loll, PRL 95, 171301 (2005)
 */
struct SpectralDimensionResultV2 {
    double d_s;                     ///< Best estimate from plateau
    double d_s_error;               ///< Std error within plateau
    double plateau_t_min;           ///< Start of plateau (time)
    double plateau_t_max;           ///< End of plateau (time)
    bool has_plateau;               ///< Whether a stable plateau was found
    double d_s_global_fit;          ///< Fallback: global log-log fit
    double global_fit_r2;           ///< R² of global fit
    std::vector<double> P_t;        ///< Return probability P(t)
    std::vector<double> d_eff_t;    ///< Local effective dimension d_eff(t)
    std::vector<double> time_pts;   ///< Time points (even only for bipartite)
};

/**
 * @brief Compute spectral dimension using plateau-finding method.
 *
 * Algorithm:
 *   1. Compute P(t) via random walks
 *   2. Compute d_eff(t) = -2 * d(log P)/d(log t) using centred differences
 *   3. Smooth d_eff(t) with a moving average
 *   4. Find the widest window where variance of d_eff is below threshold
 *   5. Return mean d_eff in that window as d_s
 *
 * @param graph       The graph
 * @param num_walkers Number of random walkers (recommend ≥ 10000)
 * @param max_steps   Max steps per walker (recommend ≥ 3*sqrt(N) for d~4)
 * @param seed        Random seed
 * @return SpectralDimensionResultV2
 */
SpectralDimensionResultV2 compute_spectral_dimension_v2(
    const DynamicGraph& graph,
    uint32_t num_walkers = 20000,
    uint32_t max_steps = 2000,
    uint64_t seed = 42);

} // namespace discretum
