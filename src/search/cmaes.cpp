#include "search/cmaes.hpp"
#include "utils/logging.hpp"
#include <cmath>
#include <algorithm>
#include <numeric>
#include <random>
#include <chrono>
#include <fstream>
#include <filesystem>
#include <omp.h>
#include <Eigen/Dense>
#include <nlohmann/json.hpp>

namespace discretum {

using json = nlohmann::json;
namespace fs = std::filesystem;

CMAES::CMAES(const CMAESConfig& config) : config_(config) {
    dim_ = config_.dim;
    init_constants();
}

void CMAES::init_constants() {
    // Population size
    lambda_ = config_.lambda > 0 ? config_.lambda : static_cast<int>(4 + 3 * std::log(dim_));
    mu_ = lambda_ / 2;
    
    // Recombination weights (log-linear)
    weights_.resize(mu_);
    double sum_w = 0.0;
    for (int i = 0; i < mu_; ++i) {
        weights_[i] = std::log(mu_ + 0.5) - std::log(i + 1.0);
        sum_w += weights_[i];
    }
    for (int i = 0; i < mu_; ++i) weights_[i] /= sum_w;
    
    double sum_w2 = 0.0;
    for (int i = 0; i < mu_; ++i) sum_w2 += weights_[i] * weights_[i];
    mu_eff_ = 1.0 / sum_w2;
    
    // Adaptation parameters (from Hansen's tutorial)
    cc_ = (4.0 + mu_eff_ / dim_) / (dim_ + 4.0 + 2.0 * mu_eff_ / dim_);
    cs_ = (mu_eff_ + 2.0) / (dim_ + mu_eff_ + 5.0);
    c1_ = 2.0 / ((dim_ + 1.3) * (dim_ + 1.3) + mu_eff_);
    cmu_ = std::min(1.0 - c1_, 2.0 * (mu_eff_ - 2.0 + 1.0 / mu_eff_)
                     / ((dim_ + 2.0) * (dim_ + 2.0) + mu_eff_));
    damps_ = 1.0 + 2.0 * std::max(0.0, std::sqrt((mu_eff_ - 1.0) / (dim_ + 1.0)) - 1.0) + cs_;
    
    // Initialize state
    mean_.assign(dim_, 0.0);
    sigma_ = config_.sigma0;
    pc_.assign(dim_, 0.0);
    ps_.assign(dim_, 0.0);
    
    // Identity covariance
    C_.assign(dim_, std::vector<double>(dim_, 0.0));
    B_.assign(dim_, std::vector<double>(dim_, 0.0));
    D_.assign(dim_, 1.0);
    for (int i = 0; i < dim_; ++i) {
        C_[i][i] = 1.0;
        B_[i][i] = 1.0;
    }
    eigen_eval_ = 0;
}

void CMAES::eigendecomposition() {
    // Use Eigen for symmetric eigendecomposition
    Eigen::MatrixXd Cm(dim_, dim_);
    for (int i = 0; i < dim_; ++i)
        for (int j = 0; j < dim_; ++j)
            Cm(i, j) = C_[i][j];
    
    // Enforce symmetry
    Cm = (Cm + Cm.transpose()) / 2.0;
    
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(Cm);
    Eigen::VectorXd evals = es.eigenvalues();
    Eigen::MatrixXd evecs = es.eigenvectors();
    
    for (int i = 0; i < dim_; ++i) {
        D_[i] = std::sqrt(std::max(evals(i), 1e-20));
        for (int j = 0; j < dim_; ++j)
            B_[j][i] = evecs(j, i);
    }
}

std::vector<std::vector<double>> CMAES::sample_population() {
    std::mt19937_64 gen(config_.seed + eigen_eval_);
    std::normal_distribution<double> N(0.0, 1.0);
    
    std::vector<std::vector<double>> pop(lambda_, std::vector<double>(dim_));
    
    for (int k = 0; k < lambda_; ++k) {
        // z ~ N(0, I)
        std::vector<double> z(dim_);
        for (int i = 0; i < dim_; ++i) z[i] = N(gen);
        
        // y = B * D * z
        std::vector<double> y(dim_, 0.0);
        for (int i = 0; i < dim_; ++i) {
            double tmp = D_[i] * z[i];
            for (int j = 0; j < dim_; ++j)
                y[j] += B_[j][i] * tmp;
        }
        
        // x = mean + sigma * y
        for (int i = 0; i < dim_; ++i)
            pop[k][i] = mean_[i] + sigma_ * y[i];
    }
    
    return pop;
}

double CMAES::evaluate(const std::vector<double>& params) {
    // Create rule from params
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
    
    // Evolve with safety cap to abort runaway growth
    EvolutionConfig evo_cfg;
    evo_cfg.num_steps = config_.evo_steps;
    evo_cfg.seed = config_.seed;
    evo_cfg.max_nodes = N0 * 20;
    
    Evolution evo(std::move(graph), std::move(rule), evo_cfg);
    evo.run();
    
    // Compute fitness
    if (config_.fitness_version == 3) {
        return compute_fitness_v3_total(evo.get_graph(), config_.fitness_params_v3);
    } else if (config_.fitness_version == 2) {
        return compute_fitness_v2_total(evo.get_graph(), config_.fitness_params_v2);
    }
    return compute_fitness(evo.get_graph(), config_.fitness_params);
}

void CMAES::save_checkpoint(const std::string& path, const CMAESResult& result, int gen) const {
    json j;
    j["generation"] = gen;
    j["best_fitness"] = result.best_fitness;
    j["best_params"] = result.best_params;
    j["fitness_history"] = result.fitness_history;
    j["sigma"] = sigma_;
    j["mean"] = mean_;
    j["pc"] = pc_;
    j["ps"] = ps_;
    j["eigen_eval"] = eigen_eval_;
    j["D"] = D_;
    
    // Save covariance and eigenvectors as flat arrays
    std::vector<double> C_flat(dim_ * dim_), B_flat(dim_ * dim_);
    for (int i = 0; i < dim_; ++i)
        for (int k = 0; k < dim_; ++k) {
            C_flat[i * dim_ + k] = C_[i][k];
            B_flat[i * dim_ + k] = B_[i][k];
        }
    j["C"] = C_flat;
    j["B"] = B_flat;
    
    // Write to temp file, then rename for atomicity
    std::string tmp_path = path + ".tmp";
    std::ofstream f(tmp_path);
    f << j.dump(2) << std::endl;
    f.close();
    fs::rename(tmp_path, path);
}

bool CMAES::load_checkpoint(const std::string& path, CMAESResult& result, int& gen) {
    if (!fs::exists(path)) return false;
    
    try {
        std::ifstream f(path);
        json j;
        f >> j;
        
        gen = j.at("generation").get<int>();
        result.best_fitness = j.at("best_fitness").get<double>();
        result.best_params = j.at("best_params").get<std::vector<double>>();
        result.fitness_history = j.at("fitness_history").get<std::vector<double>>();
        sigma_ = j.at("sigma").get<double>();
        mean_ = j.at("mean").get<std::vector<double>>();
        pc_ = j.at("pc").get<std::vector<double>>();
        ps_ = j.at("ps").get<std::vector<double>>();
        eigen_eval_ = j.at("eigen_eval").get<int>();
        D_ = j.at("D").get<std::vector<double>>();
        
        auto C_flat = j.at("C").get<std::vector<double>>();
        auto B_flat = j.at("B").get<std::vector<double>>();
        for (int i = 0; i < dim_; ++i)
            for (int k = 0; k < dim_; ++k) {
                C_[i][k] = C_flat[i * dim_ + k];
                B_[i][k] = B_flat[i * dim_ + k];
            }
        
        spdlog::info("Loaded checkpoint from gen {}: best_fitness={:.6f}", gen, result.best_fitness);
        return true;
    } catch (const std::exception& e) {
        spdlog::warn("Failed to load checkpoint {}: {}", path, e.what());
        return false;
    }
}

CMAESResult CMAES::optimize() {
    CMAESResult result;
    result.best_fitness = -1e30;
    result.best_params.assign(dim_, 0.0);
    
    // Create checkpoint directory
    fs::create_directories(config_.checkpoint_dir);
    std::string ckpt_path = config_.checkpoint_dir + "/cmaes_checkpoint.json";
    
    // Try to resume from checkpoint
    int start_gen = 0;
    if (load_checkpoint(ckpt_path, result, start_gen)) {
        start_gen++;  // Continue from next generation
    }
    
    double chi_n = std::sqrt(static_cast<double>(dim_))
                 * (1.0 - 1.0 / (4.0 * dim_) + 1.0 / (21.0 * dim_ * dim_));
    
    int num_threads = omp_get_max_threads();
    spdlog::info("CMA-ES: λ={}, dim={}, threads={}, max_gen={}", lambda_, dim_, num_threads, config_.max_generations);
    
    auto search_start = std::chrono::steady_clock::now();
    
    for (int gen = start_gen; gen < config_.max_generations; ++gen) {
        auto gen_start = std::chrono::steady_clock::now();
        
        // Sample population
        auto pop = sample_population();
        
        // Evaluate fitness in parallel using OpenMP
        std::vector<double> fitvals(lambda_);
        #pragma omp parallel for schedule(dynamic)
        for (int k = 0; k < lambda_; ++k) {
            fitvals[k] = evaluate(pop[k]);
        }
        
        auto gen_end = std::chrono::steady_clock::now();
        double gen_sec = std::chrono::duration<double>(gen_end - gen_start).count();
        
        // Sort by fitness (descending — we maximize)
        std::vector<int> idx(lambda_);
        std::iota(idx.begin(), idx.end(), 0);
        std::sort(idx.begin(), idx.end(), [&](int a, int b) {
            return fitvals[a] > fitvals[b];
        });
        
        // Track best
        if (fitvals[idx[0]] > result.best_fitness) {
            result.best_fitness = fitvals[idx[0]];
            result.best_params = pop[idx[0]];
        }
        result.fitness_history.push_back(fitvals[idx[0]]);
        
        // ETA calculation
        int gens_done = gen - start_gen + 1;
        double elapsed = std::chrono::duration<double>(gen_end - search_start).count();
        double avg_gen_sec = elapsed / gens_done;
        int gens_remaining = config_.max_generations - gen - 1;
        double eta_sec = avg_gen_sec * gens_remaining;
        int eta_min = static_cast<int>(eta_sec) / 60;
        int eta_s = static_cast<int>(eta_sec) % 60;
        
        spdlog::info("CMA-ES gen {}/{}: best={:.6f}, gen_best={:.6f}, σ={:.4e}, {:.1f}s/gen, ETA {}m{:02d}s",
                     gen, config_.max_generations, result.best_fitness,
                     fitvals[idx[0]], sigma_, gen_sec, eta_min, eta_s);
        
        // Compute new mean from best mu individuals
        std::vector<double> old_mean = mean_;
        std::fill(mean_.begin(), mean_.end(), 0.0);
        for (int i = 0; i < mu_; ++i) {
            for (int j = 0; j < dim_; ++j)
                mean_[j] += weights_[i] * pop[idx[i]][j];
        }
        
        // Update evolution paths
        std::vector<double> diff(dim_);
        for (int i = 0; i < dim_; ++i)
            diff[i] = (mean_[i] - old_mean[i]) / sigma_;
        
        // C^{-1/2} * diff = B * D^{-1} * B^T * diff
        std::vector<double> Bt_diff(dim_, 0.0);
        for (int i = 0; i < dim_; ++i)
            for (int j = 0; j < dim_; ++j)
                Bt_diff[i] += B_[j][i] * diff[j];
        
        std::vector<double> invsqrt_diff(dim_, 0.0);
        for (int i = 0; i < dim_; ++i)
            Bt_diff[i] /= std::max(D_[i], 1e-20);
        for (int i = 0; i < dim_; ++i)
            for (int j = 0; j < dim_; ++j)
                invsqrt_diff[i] += B_[i][j] * Bt_diff[j];
        
        double cs_factor = std::sqrt(cs_ * (2.0 - cs_) * mu_eff_);
        for (int i = 0; i < dim_; ++i)
            ps_[i] = (1.0 - cs_) * ps_[i] + cs_factor * invsqrt_diff[i];
        
        // Compute |ps|
        double ps_norm = 0.0;
        for (int i = 0; i < dim_; ++i) ps_norm += ps_[i] * ps_[i];
        ps_norm = std::sqrt(ps_norm);
        
        // Heaviside function for stalling indicator
        double gen_factor = 1.0 / (1.0 + 2.0 * dim_ / (gen + 1.0 + dim_));
        double threshold = (1.4 + 2.0 / (dim_ + 1.0)) * chi_n * std::sqrt(gen_factor);
        int hsig = (ps_norm < threshold) ? 1 : 0;
        
        // pc update
        double cc_factor = std::sqrt(cc_ * (2.0 - cc_) * mu_eff_);
        for (int i = 0; i < dim_; ++i)
            pc_[i] = (1.0 - cc_) * pc_[i] + hsig * cc_factor * diff[i];
        
        // Covariance matrix update
        double c1a = c1_ * (1.0 - (1.0 - hsig * hsig) * cc_ * (2.0 - cc_));
        for (int i = 0; i < dim_; ++i) {
            for (int j = 0; j <= i; ++j) {
                // Rank-1 update
                double rank1 = c1_ * pc_[i] * pc_[j];
                
                // Rank-mu update
                double rankmu = 0.0;
                for (int k = 0; k < mu_; ++k) {
                    double di = (pop[idx[k]][i] - old_mean[i]) / sigma_;
                    double dj = (pop[idx[k]][j] - old_mean[j]) / sigma_;
                    rankmu += weights_[k] * di * dj;
                }
                rankmu *= cmu_;
                
                C_[i][j] = (1.0 - c1a - cmu_) * C_[i][j] + rank1 + rankmu;
                C_[j][i] = C_[i][j];  // Symmetry
            }
        }
        
        // Step-size adaptation
        sigma_ *= std::exp((cs_ / damps_) * (ps_norm / chi_n - 1.0));
        
        // Eigendecomposition every few evaluations
        eigen_eval_ += lambda_;
        if (eigen_eval_ >= lambda_ * 10) {
            eigendecomposition();
            eigen_eval_ = 0;
        }
        
        // Save checkpoint after each generation (atomic write)
        if ((gen + 1) % config_.checkpoint_interval == 0) {
            save_checkpoint(ckpt_path, result, gen);
        }
        
        // Convergence checks
        if (sigma_ < config_.tol_sigma) {
            spdlog::info("CMA-ES converged: sigma < tol");
            save_checkpoint(ckpt_path, result, gen);
            break;
        }
        if (gen > 0 && std::abs(result.fitness_history.back() - result.fitness_history[gen - 1]) < config_.tol_fitness) {
            spdlog::info("CMA-ES converged: fitness stalled");
            save_checkpoint(ckpt_path, result, gen);
            break;
        }
    }
    
    result.generations_used = static_cast<int>(result.fitness_history.size());
    return result;
}

} // namespace discretum
