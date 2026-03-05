#pragma once

#include "core/graph.hpp"
#include "automaton/rule_parametric.hpp"
#include "automaton/evolution.hpp"
#include "search/fitness.hpp"
#include <vector>
#include <string>
#include <cstdint>

namespace discretum {

/**
 * @brief Configuration for ensemble runs.
 *
 * Runs N independent evolutions of the same rule with different seeds,
 * then computes statistics (mean, stderr, median, quartiles) on all
 * observables from the v2 fitness function.
 */
struct EnsembleConfig {
    uint32_t num_runs = 20;          ///< Number of independent evolutions
    uint32_t evo_steps = 300;        ///< Steps per evolution
    uint64_t master_seed = 42;       ///< Base seed; run i uses master_seed + i*1000003
    uint64_t seed_stride = 1000003;  ///< Stride between seeds

    // Graph creation
    std::string graph_type = "lattice_4d";
    uint32_t graph_size = 625;       ///< Target node count

    // Fitness evaluation
    int fitness_version = 2;         ///< 2 = v2, 3 = v3
    FitnessParamsV2 fitness_params;   ///< v2 params
    FitnessParamsV3 fitness_params_v3; ///< v3 params

    // Safety
    uint32_t max_nodes_factor = 20;  ///< Abort if N > factor * N_initial
};

/**
 * @brief Per-run result from a single ensemble member.
 */
struct EnsembleRunResult {
    uint64_t seed;
    FitnessResultV2 fitness;
    uint32_t steps_completed;
    bool aborted;  ///< True if evolution was aborted (runaway growth)
};

/**
 * @brief Summary statistics for an observable across the ensemble.
 */
struct ObservableStats {
    double mean;
    double std_err;    ///< Standard error of the mean = σ / √N
    double std_dev;    ///< Sample standard deviation
    double median;
    double q25, q75;   ///< Quartiles
    double min_val, max_val;
    int n_valid;       ///< Number of non-aborted runs contributing
};

/**
 * @brief Full ensemble result with statistics on all observables.
 */
struct EnsembleResult {
    std::vector<EnsembleRunResult> runs;

    // Summary statistics for each observable
    ObservableStats fitness_total;
    ObservableStats d_H;
    ObservableStats d_s;
    ObservableStats mean_curvature;
    ObservableStats std_curvature;
    ObservableStats cv_degree;
    ObservableStats mean_degree;
    ObservableStats n_final;

    // Component fitness terms
    ObservableStats f_hausdorff;
    ObservableStats f_curvature;
    ObservableStats f_spectral;
    ObservableStats f_stability;
    ObservableStats f_regularity;
    ObservableStats f_density;     ///< v3 only
    ObservableStats f_degradation; ///< v3 only

    int n_connected;   ///< Number of runs that produced connected graphs
    int n_total;       ///< Total number of runs
    int n_aborted;     ///< Number of aborted runs
};

/**
 * @brief Run an ensemble of independent evolutions for a given rule.
 *
 * Parallelised with OpenMP. Each run creates a fresh initial graph,
 * evolves it with a unique seed, then evaluates v2 fitness.
 */
EnsembleResult run_ensemble(const ParametricRule& rule, const EnsembleConfig& config);

/**
 * @brief Serialize ensemble result to JSON.
 */
std::string ensemble_result_to_json(const EnsembleResult& result, int indent = 2);

} // namespace discretum
