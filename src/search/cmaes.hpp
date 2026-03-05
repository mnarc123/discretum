#pragma once

#include "core/graph.hpp"
#include "automaton/rule_parametric.hpp"
#include "automaton/evolution.hpp"
#include "search/fitness.hpp"
#include <vector>
#include <functional>
#include <string>
#include <cstdint>

namespace discretum {

/**
 * @brief CMA-ES (Covariance Matrix Adaptation Evolution Strategy)
 *
 * Searches for parametric rule parameters that maximize fitness.
 * Uses the (μ/μ_w, λ)-CMA-ES variant with rank-μ and rank-1 updates.
 *
 * Reference: Hansen, N. (2016). "The CMA Evolution Strategy: A Tutorial"
 */
struct CMAESConfig {
    int dim = ParametricRule::TOTAL_PARAMS;  // Parameter dimension
    int lambda = 0;         // Population size (0 = auto: 4+3*ln(dim))
    int max_generations = 200;
    double sigma0 = 0.5;   // Initial step size
    double tol_fitness = 1e-8;
    double tol_sigma = 1e-12;
    
    // Evolution config for evaluating each candidate
    uint32_t evo_steps = 50;
    uint32_t graph_size = 100;  // Initial lattice size per dimension
    uint64_t seed = 42;
    
    // Graph type: "lattice_3d" or "lattice_4d"
    std::string graph_type = "lattice_3d";
    
    // Fitness version: 1 = original, 2 = v2, 3 = v3 (geometry-preserving)
    int fitness_version = 1;
    FitnessParams fitness_params;       // v1
    FitnessParamsV2 fitness_params_v2;  // v2
    FitnessParamsV3 fitness_params_v3;  // v3
    
    // Checkpoint config
    std::string checkpoint_dir = "checkpoints";
    int checkpoint_interval = 1;  // Save every N generations
};

struct CMAESResult {
    std::vector<double> best_params;
    double best_fitness;
    int generations_used;
    std::vector<double> fitness_history;  // Best fitness per generation
};

class CMAES {
public:
    CMAES() = default;
    explicit CMAES(const CMAESConfig& config);
    
    /// Run optimization, returns best parameters found
    CMAESResult optimize();
    
    /// Evaluate a single parameter vector
    double evaluate(const std::vector<double>& params);
    
    /// Save/load checkpoint for crash recovery
    void save_checkpoint(const std::string& path, const CMAESResult& result, int gen) const;
    bool load_checkpoint(const std::string& path, CMAESResult& result, int& gen);
    
private:
    CMAESConfig config_;
    
    // CMA-ES state
    int dim_;
    int lambda_;      // Population size
    int mu_;          // Parent count
    std::vector<double> weights_;  // Recombination weights
    double mu_eff_;   // Variance-effective selection mass
    
    // Adaptation parameters
    double cc_, cs_, c1_, cmu_, damps_;
    
    // State variables
    std::vector<double> mean_;               // Distribution mean
    double sigma_;                           // Step size
    std::vector<double> pc_, ps_;            // Evolution paths
    std::vector<std::vector<double>> C_;     // Covariance matrix
    std::vector<std::vector<double>> B_;     // Eigenvectors of C
    std::vector<double> D_;                  // Square root of eigenvalues of C
    int eigen_eval_;                         // Count since last eigendecomposition
    
    void init_constants();
    void eigendecomposition();
    std::vector<std::vector<double>> sample_population();
};

} // namespace discretum
