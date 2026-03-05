#include "search/genetic.hpp"
#include "utils/logging.hpp"
#include <cmath>
#include <algorithm>
#include <numeric>
#include <random>
#include <chrono>
#include <fstream>
#include <filesystem>
#include <omp.h>
#include <nlohmann/json.hpp>

namespace discretum {

using json = nlohmann::json;
namespace fs = std::filesystem;

GeneticAlgorithm::GeneticAlgorithm(const GAConfig& config) : config_(config) {}

double GeneticAlgorithm::evaluate(const std::vector<double>& params) {
    std::vector<float> fparams(params.begin(), params.end());
    ParametricRule rule(fparams);
    
    // Create initial graph based on graph_type
    DynamicGraph graph;
    if (config_.graph_type == "lattice_4d") {
        int side = std::max(3, static_cast<int>(std::pow(config_.graph_size, 0.25)));
        graph = DynamicGraph::create_lattice_4d(side, side, side, side);
    } else {
        int side = std::max(3, static_cast<int>(std::cbrt(config_.graph_size)));
        graph = DynamicGraph::create_lattice_3d(side, side, side);
    }
    uint32_t N0 = graph.num_nodes();
    
    EvolutionConfig evo_cfg;
    evo_cfg.num_steps = config_.evo_steps;
    evo_cfg.seed = config_.seed;
    evo_cfg.max_nodes = N0 * 20;
    
    Evolution evo(std::move(graph), std::move(rule), evo_cfg);
    evo.run();
    
    if (config_.fitness_version == 3) {
        return compute_fitness_v3_total(evo.get_graph(), config_.fitness_params_v3);
    } else if (config_.fitness_version == 2) {
        return compute_fitness_v2_total(evo.get_graph(), config_.fitness_params_v2);
    }
    return compute_fitness(evo.get_graph(), config_.fitness_params);
}

void GeneticAlgorithm::save_checkpoint(const std::string& path, const GAResult& result, int gen,
                                        const std::vector<std::vector<double>>& pop,
                                        const std::vector<double>& fit) const {
    json j;
    j["generation"] = gen;
    j["best_fitness"] = result.best_fitness;
    j["best_params"] = result.best_params;
    j["fitness_history"] = result.fitness_history;
    j["population"] = pop;
    j["fitness"] = fit;
    
    std::string tmp_path = path + ".tmp";
    std::ofstream f(tmp_path);
    f << j.dump(2) << std::endl;
    f.close();
    fs::rename(tmp_path, path);
}

bool GeneticAlgorithm::load_checkpoint(const std::string& path, GAResult& result, int& gen,
                                        std::vector<std::vector<double>>& pop,
                                        std::vector<double>& fit) {
    if (!fs::exists(path)) return false;
    
    try {
        std::ifstream f(path);
        json j;
        f >> j;
        
        gen = j.at("generation").get<int>();
        result.best_fitness = j.at("best_fitness").get<double>();
        result.best_params = j.at("best_params").get<std::vector<double>>();
        result.fitness_history = j.at("fitness_history").get<std::vector<double>>();
        pop = j.at("population").get<std::vector<std::vector<double>>>();
        fit = j.at("fitness").get<std::vector<double>>();
        
        // Validate checkpoint integrity
        if (pop.empty() || fit.empty() || pop.size() != fit.size() ||
            static_cast<int>(pop.size()) != config_.pop_size) {
            spdlog::warn("GA checkpoint invalid (pop_size mismatch or empty), starting fresh");
            return false;
        }
        
        spdlog::info("Loaded GA checkpoint from gen {}: best_fitness={:.6f}", gen, result.best_fitness);
        return true;
    } catch (const std::exception& e) {
        spdlog::warn("Failed to load GA checkpoint {}: {}", path, e.what());
        return false;
    }
}

std::vector<double> GeneticAlgorithm::tournament_select(
    const std::vector<std::vector<double>>& pop,
    const std::vector<double>& fitness,
    std::mt19937_64& rng)
{
    std::uniform_int_distribution<int> dist(0, static_cast<int>(pop.size()) - 1);
    int best = dist(rng);
    for (int i = 1; i < config_.tournament_size; ++i) {
        int candidate = dist(rng);
        if (fitness[candidate] > fitness[best])
            best = candidate;
    }
    return pop[best];
}

std::pair<std::vector<double>, std::vector<double>> GeneticAlgorithm::crossover(
    const std::vector<double>& p1,
    const std::vector<double>& p2,
    std::mt19937_64& rng)
{
    int d = static_cast<int>(p1.size());
    std::vector<double> c1(d), c2(d);
    std::uniform_real_distribution<double> U(0.0, 1.0);
    
    for (int i = 0; i < d; ++i) {
        double lo = std::min(p1[i], p2[i]);
        double hi = std::max(p1[i], p2[i]);
        double range = hi - lo;
        double ext = config_.blx_alpha * range;
        
        std::uniform_real_distribution<double> blx(lo - ext, hi + ext);
        c1[i] = blx(rng);
        c2[i] = blx(rng);
    }
    return {c1, c2};
}

void GeneticAlgorithm::mutate(std::vector<double>& individual, std::mt19937_64& rng) {
    std::uniform_real_distribution<double> U(0.0, 1.0);
    std::normal_distribution<double> N(0.0, config_.mutation_sigma);
    
    for (auto& gene : individual) {
        if (U(rng) < config_.mutation_rate) {
            gene += N(rng);
        }
    }
}

GAResult GeneticAlgorithm::evolve() {
    GAResult result;
    result.best_fitness = -1e30;
    
    std::mt19937_64 rng(config_.seed);
    std::normal_distribution<double> N(0.0, 0.5);
    std::uniform_real_distribution<double> U(0.0, 1.0);
    
    int d = config_.dim;
    int pop_size = config_.pop_size;
    
    // Create checkpoint directory
    fs::create_directories(config_.checkpoint_dir);
    std::string ckpt_path = config_.checkpoint_dir + "/ga_checkpoint.json";
    
    int num_threads = omp_get_max_threads();
    spdlog::info("GA: pop={}, dim={}, threads={}, max_gen={}", pop_size, d, num_threads, config_.max_generations);
    
    // Try to resume from checkpoint
    int start_gen = 0;
    std::vector<std::vector<double>> pop;
    std::vector<double> fitness;
    
    if (load_checkpoint(ckpt_path, result, start_gen, pop, fitness)) {
        start_gen++;  // Resume from next gen
    } else {
        // Initialize random population
        pop.assign(pop_size, std::vector<double>(d));
        for (auto& ind : pop)
            for (auto& gene : ind)
                gene = N(rng);
        
        // Evaluate initial population in parallel
        fitness.resize(pop_size);
        #pragma omp parallel for schedule(dynamic)
        for (int i = 0; i < pop_size; ++i)
            fitness[i] = evaluate(pop[i]);
        
        // Track best
        for (int i = 0; i < pop_size; ++i) {
            if (fitness[i] > result.best_fitness) {
                result.best_fitness = fitness[i];
                result.best_params = pop[i];
            }
        }
        result.fitness_history.push_back(result.best_fitness);
        
        spdlog::info("GA init: best={:.6f}", result.best_fitness);
        save_checkpoint(ckpt_path, result, -1, pop, fitness);
    }
    
    auto search_start = std::chrono::steady_clock::now();
    
    for (int gen = start_gen; gen < config_.max_generations; ++gen) {
        auto gen_start = std::chrono::steady_clock::now();
        
        std::vector<std::vector<double>> new_pop;
        new_pop.reserve(pop_size);
        
        // Elitism: keep best individual
        int best_idx = static_cast<int>(
            std::max_element(fitness.begin(), fitness.end()) - fitness.begin());
        new_pop.push_back(pop[best_idx]);
        
        // Generate offspring
        while (static_cast<int>(new_pop.size()) < pop_size) {
            auto p1 = tournament_select(pop, fitness, rng);
            auto p2 = tournament_select(pop, fitness, rng);
            
            if (U(rng) < config_.crossover_rate) {
                auto [c1, c2] = crossover(p1, p2, rng);
                mutate(c1, rng);
                mutate(c2, rng);
                new_pop.push_back(std::move(c1));
                if (static_cast<int>(new_pop.size()) < pop_size)
                    new_pop.push_back(std::move(c2));
            } else {
                mutate(p1, rng);
                new_pop.push_back(std::move(p1));
            }
        }
        
        pop = std::move(new_pop);
        
        // Evaluate in parallel
        #pragma omp parallel for schedule(dynamic)
        for (int i = 0; i < pop_size; ++i)
            fitness[i] = evaluate(pop[i]);
        
        // Track best
        for (int i = 0; i < pop_size; ++i) {
            if (fitness[i] > result.best_fitness) {
                result.best_fitness = fitness[i];
                result.best_params = pop[i];
            }
        }
        result.fitness_history.push_back(result.best_fitness);
        
        auto gen_end = std::chrono::steady_clock::now();
        double gen_sec = std::chrono::duration<double>(gen_end - gen_start).count();
        
        // ETA
        int gens_done = gen - start_gen + 1;
        double elapsed = std::chrono::duration<double>(gen_end - search_start).count();
        double avg_gen_sec = elapsed / gens_done;
        int gens_remaining = config_.max_generations - gen - 1;
        double eta_sec = avg_gen_sec * gens_remaining;
        int eta_min = static_cast<int>(eta_sec) / 60;
        int eta_s = static_cast<int>(eta_sec) % 60;
        
        spdlog::info("GA gen {}/{}: best={:.6f}, {:.1f}s/gen, ETA {}m{:02d}s",
                     gen, config_.max_generations, result.best_fitness, gen_sec, eta_min, eta_s);
        
        // Save checkpoint
        if ((gen + 1) % config_.checkpoint_interval == 0) {
            save_checkpoint(ckpt_path, result, gen, pop, fitness);
        }
    }
    
    result.generations_used = static_cast<int>(result.fitness_history.size());
    return result;
}

} // namespace discretum
