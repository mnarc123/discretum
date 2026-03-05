#pragma once

#include "core/graph.hpp"
#include <string>
#include <cstdint>

namespace discretum {

/**
 * @brief Parameters controlling the fitness evaluation.
 *
 * The fitness function measures how well a graph approximates
 * a target geometry (e.g. flat d-dimensional spacetime).
 *
 * Components:
 *   - Ricci curvature: average Ollivier-Ricci should be near target
 *   - Spectral dimension: should match target dimension
 *   - Connectivity: graph should be connected
 *   - Degree regularity: degree variance should be low
 *   - Size stability: graph should maintain reasonable size
 */
struct FitnessParams {
    double weight_ricci = 1.0;
    double weight_dim = 1.0;
    double weight_connectivity = 5.0;
    double weight_degree_reg = 0.5;
    double weight_size = 0.3;
    
    double target_dimension = 3.0;      // Target spectral dimension
    double target_avg_curvature = 0.0;  // Flat spacetime → zero curvature
    double target_size = 1000.0;        // Target graph size (nodes)
    
    float ollivier_alpha = 0.5f;        // Laziness for Ollivier-Ricci
    uint32_t spectral_walkers = 5000;   // Walkers for spectral dim
    uint32_t spectral_steps = 100;      // Steps for spectral dim
    uint32_t max_curv_edges = 200;      // Max edges to sample for Ollivier-Ricci (0=all)
};

/**
 * @brief Compute fitness of a graph relative to target geometry.
 * @return Fitness value (higher is better, 0 is perfect match)
 *         Returns negative values; closer to 0 is better.
 */
double compute_fitness(const DynamicGraph& graph, const FitnessParams& params);

/**
 * @brief Detailed fitness breakdown for diagnostics.
 */
struct FitnessBreakdown {
    double total;
    double ricci_term;
    double dimension_term;
    double connectivity_term;
    double degree_reg_term;
    double size_term;
    double measured_avg_curvature;
    double measured_dimension;
    bool is_connected;
};

FitnessBreakdown compute_fitness_detailed(const DynamicGraph& graph, const FitnessParams& params);

// ════════════════════════════════════════════════════════════════
// V2 FITNESS FUNCTION — Hausdorff-based, physically motivated
// ════════════════════════════════════════════════════════════════

/**
 * @brief Configuration for the v2 fitness function.
 *
 * Design principles:
 *   1. d_H (Hausdorff) is the primary dimensionality measure (robust, no finite-size bias)
 *   2. d_s targets concordance with d_H, NOT a fixed value
 *   3. Curvature penalises both mean and fluctuations (flat = κ→0, σ(κ)→0)
 *   4. All penalty terms are squared for smoother gradients
 */
struct FitnessParamsV2 {
    // Target geometry
    double target_dim = 4.0;                ///< Target dimension D (Hausdorff)

    // Weights
    double w_hausdorff     = 2.0;           ///< Weight for |d_H - D|²
    double w_curvature     = 1.5;           ///< Weight for ⟨κ⟩² + β·σ(κ)²
    double w_spectral      = 1.0;           ///< Weight for |d_s - d_H|² (concordance)
    double w_connectivity  = 10.0;          ///< Penalty per extra component
    double w_stability     = 0.5;           ///< Weight for ln(N/N₀)²
    double w_regularity    = 0.3;           ///< Weight for CV(k)²

    // Curvature config
    double curvature_fluct_beta = 0.5;      ///< Relative weight of σ(κ)² vs ⟨κ⟩²
    float ollivier_alpha = 0.5f;            ///< Laziness for Ollivier-Ricci
    uint32_t curvature_samples = 300;       ///< Edges to sample for curvature

    // Spectral dimension config
    uint32_t spectral_walkers = 5000;
    uint32_t spectral_steps = 500;

    // Hausdorff dimension config
    uint32_t hausdorff_sources = 30;        ///< BFS sources for sampled d_H

    // Graph info
    double n_initial = 625.0;               ///< Initial node count (for stability)
    uint64_t seed = 42;
};

/**
 * @brief Detailed result from v2 fitness evaluation.
 */
struct FitnessResultV2 {
    double total;
    double f_hausdorff;
    double f_curvature;
    double f_spectral;
    double f_connectivity;
    double f_stability;
    double f_regularity;

    // Raw observables
    double d_H;
    double d_s;
    double mean_curvature;
    double std_curvature;
    double cv_degree;
    double mean_degree;
    double n_final;
    int n_components;
    bool is_connected;
};

/**
 * @brief Compute v2 fitness of a graph relative to target geometry.
 *        Uses Hausdorff dimension as primary measure, squared penalties.
 */
FitnessResultV2 compute_fitness_v2(const DynamicGraph& graph, const FitnessParamsV2& params);

/**
 * @brief Convenience: returns just the total fitness (for optimisers).
 */
double compute_fitness_v2_total(const DynamicGraph& graph, const FitnessParamsV2& params);

// ════════════════════════════════════════════════════════════════
// V3 FITNESS FUNCTION — Geometry-preserving with density + baseline
// ════════════════════════════════════════════════════════════════

/**
 * @brief Baseline metrics from a bare lattice (computed once, cached).
 *        Used by v3 non-degradation constraint.
 */
struct BaselineMetrics {
    double d_H = 0.0;
    double d_s = 0.0;
    double mean_curvature = 0.0;
    double std_curvature = 0.0;
    double mean_degree = 0.0;
    bool valid = false;           ///< Set to true once computed
};

/**
 * @brief Configuration for the v3 fitness function.
 *
 * Extends v2 with:
 *   - Density penalty: penalises ⟨k⟩ > k_max_target
 *   - Non-degradation: penalises metrics worse than bare lattice baseline
 *   - Spectral v2: plateau-finding estimator for d_s
 */
struct FitnessParamsV3 {
    // Target geometry
    double target_dim = 4.0;

    // Weights (same as v2 + new)
    double w_hausdorff     = 2.0;
    double w_curvature     = 1.5;
    double w_spectral      = 1.0;
    double w_connectivity  = 10.0;
    double w_stability     = 0.5;
    double w_regularity    = 0.3;
    double w_density       = 3.0;   ///< NEW: density penalty weight
    double w_degradation   = 2.0;   ///< NEW: non-degradation weight

    // Density constraint
    double density_k_max_factor = 2.2;  ///< k_max = factor * target_dim

    // Non-degradation baseline
    BaselineMetrics baseline;           ///< Bare lattice reference

    // Curvature config
    double curvature_fluct_beta = 0.5;
    float ollivier_alpha = 0.5f;
    uint32_t curvature_samples = 300;

    // Spectral dimension config (v2 plateau-finding)
    uint32_t spectral_walkers = 20000;
    uint32_t spectral_steps = 800;      ///< Recommend 3*sqrt(N)
    bool use_spectral_v2 = true;        ///< Use plateau-finding estimator

    // Hausdorff dimension config
    uint32_t hausdorff_sources = 30;

    // Graph info
    double n_initial = 625.0;
    uint64_t seed = 42;
};

/**
 * @brief Detailed result from v3 fitness evaluation.
 */
struct FitnessResultV3 {
    double total;
    double f_hausdorff;
    double f_curvature;
    double f_spectral;
    double f_connectivity;
    double f_stability;
    double f_regularity;
    double f_density;           ///< NEW: density penalty
    double f_degradation;       ///< NEW: non-degradation penalty

    // Raw observables
    double d_H;
    double d_s;
    double d_s_error;           ///< Spectral dim error from plateau
    bool d_s_has_plateau;       ///< Whether plateau was found
    double mean_curvature;
    double std_curvature;
    double cv_degree;
    double mean_degree;
    double n_final;
    int n_components;
    bool is_connected;
};

/**
 * @brief Compute v3 fitness with density + non-degradation constraints.
 */
FitnessResultV3 compute_fitness_v3(const DynamicGraph& graph, const FitnessParamsV3& params);

/**
 * @brief Convenience: returns just the total fitness.
 */
double compute_fitness_v3_total(const DynamicGraph& graph, const FitnessParamsV3& params);

/**
 * @brief Compute baseline metrics for a bare lattice.
 *        This should be called once before search and cached.
 */
BaselineMetrics compute_baseline_metrics(const DynamicGraph& graph,
                                         double target_dim,
                                         uint32_t hausdorff_sources = 30,
                                         uint32_t spectral_walkers = 20000,
                                         uint32_t spectral_steps = 800,
                                         uint32_t curvature_samples = 300,
                                         uint64_t seed = 42);

} // namespace discretum
