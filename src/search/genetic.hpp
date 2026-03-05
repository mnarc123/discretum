#pragma once

#include "core/graph.hpp"
#include "automaton/rule_parametric.hpp"
#include "automaton/evolution.hpp"
#include "search/fitness.hpp"
#include <vector>
#include <string>
#include <cstdint>
#include <random>

namespace discretum {

/**
 * @brief Genetic algorithm for rule parameter search.
 *
 * Uses tournament selection, BLX-α crossover, and Gaussian mutation.
 */
struct GAConfig {
    int dim = ParametricRule::TOTAL_PARAMS;
    int pop_size = 50;
    int max_generations = 200;
    int tournament_size = 3;
    double crossover_rate = 0.8;
    double mutation_rate = 0.1;
    double mutation_sigma = 0.3;
    double blx_alpha = 0.5;  // BLX-α crossover extension
    
    uint32_t evo_steps = 50;
    uint32_t graph_size = 100;
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
    int checkpoint_interval = 1;
};

struct GAResult {
    std::vector<double> best_params;
    double best_fitness;
    int generations_used;
    std::vector<double> fitness_history;
};

class GeneticAlgorithm {
public:
    GeneticAlgorithm() = default;
    explicit GeneticAlgorithm(const GAConfig& config);
    
    GAResult evolve();
    double evaluate(const std::vector<double>& params);
    
    void save_checkpoint(const std::string& path, const GAResult& result, int gen,
                         const std::vector<std::vector<double>>& pop,
                         const std::vector<double>& fitness) const;
    bool load_checkpoint(const std::string& path, GAResult& result, int& gen,
                         std::vector<std::vector<double>>& pop,
                         std::vector<double>& fitness);
    
private:
    GAConfig config_;
    
    std::vector<double> tournament_select(
        const std::vector<std::vector<double>>& pop,
        const std::vector<double>& fitness,
        std::mt19937_64& rng);
    
    std::pair<std::vector<double>, std::vector<double>> crossover(
        const std::vector<double>& p1,
        const std::vector<double>& p2,
        std::mt19937_64& rng);
    
    void mutate(std::vector<double>& individual, std::mt19937_64& rng);
};

} // namespace discretum
