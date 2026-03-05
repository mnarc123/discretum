#pragma once

#include <string>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <toml.hpp>
#include "utils/logging.hpp"

namespace discretum {

/**
 * @brief Configuration management using TOML format
 * 
 * Provides type-safe configuration loading with validation
 * and default values.
 */
class Config {
public:
    /**
     * @brief Load configuration from TOML file
     * @param path Path to TOML file
     * @return Parsed configuration
     */
    static toml::table load(const std::string& path) {
        try {
            auto config = toml::parse_file(path);
            spdlog::info("Configuration loaded from: {}", path);
            return config;
        } catch (const toml::parse_error& err) {
            spdlog::error("Failed to parse config file '{}': {}", path, err.what());
            throw std::runtime_error("Configuration parse error");
        }
    }
    
    /**
     * @brief Save configuration to TOML file
     * @param config Configuration table
     * @param path Output path
     */
    static void save(const toml::table& config, const std::string& path) {
        std::ofstream file(path);
        if (!file) {
            throw std::runtime_error("Cannot open file for writing: " + path);
        }
        file << config;
        spdlog::info("Configuration saved to: {}", path);
    }
    
    /**
     * @brief Merge configurations (override defaults with user values)
     * @param base Base configuration
     * @param override Override configuration
     * @return Merged configuration
     */
    static toml::table merge(const toml::table& base, const toml::table& override) {
        toml::table result = base;
        
        for (auto&& [key, value] : override) {
            if (value.is_table() && result.contains(key) && result[key].is_table()) {
                // Recursively merge tables
                result.insert_or_assign(key, 
                    merge(*result[key].as_table(), *value.as_table()));
            } else {
                // Override value
                result.insert_or_assign(key, value);
            }
        }
        
        return result;
    }
    
    /**
     * @brief Get value with default fallback
     * @param config Configuration table
     * @param path Dot-separated path (e.g., "graph.initial_size")
     * @param default_value Default value if not found
     * @return Configuration value or default
     */
    template<typename T>
    static T get(const toml::table& config, const std::string& path, const T& default_value) {
        std::vector<std::string> keys;
        std::stringstream ss(path);
        std::string key;
        
        while (std::getline(ss, key, '.')) {
            keys.push_back(key);
        }
        
        const toml::node* current = &config;
        
        for (const auto& k : keys) {
            if (!current->is_table()) {
                return default_value;
            }
            
            auto table = current->as_table();
            if (!table->contains(k)) {
                return default_value;
            }
            
            current = &(*table)[k];
        }
        
        if (auto value = current->value<T>()) {
            return *value;
        }
        
        return default_value;
    }
    
    /**
     * @brief Validate configuration against schema
     * @param config Configuration to validate
     * @return True if valid
     */
    static bool validate(const toml::table& config) {
        // Check required sections
        const std::vector<std::string> required_sections = {
            "graph", "automaton", "evolution", "geometry", 
            "search", "fitness", "hardware", "output", "reproducibility"
        };
        
        for (const auto& section : required_sections) {
            if (!config.contains(section) || !config[section].is_table()) {
                spdlog::error("Missing required configuration section: {}", section);
                return false;
            }
        }
        
        // Validate graph configuration
        auto graph = config["graph"].as_table();
        if (!graph->contains("initial_type") || !graph->contains("initial_size")) {
            spdlog::error("Missing required graph parameters");
            return false;
        }
        
        // Validate initial_type
        std::string init_type = get(config, "graph.initial_type", std::string(""));
        const std::vector<std::string> valid_types = {
            "lattice_3d", "lattice_4d", "random_regular", "erdos_renyi"
        };
        if (std::find(valid_types.begin(), valid_types.end(), init_type) == valid_types.end()) {
            spdlog::error("Invalid graph initial_type: {}", init_type);
            return false;
        }
        
        // Validate numeric ranges
        uint32_t alphabet_size = get(config, "automaton.alphabet_size", 0u);
        if (alphabet_size < 2 || alphabet_size > 8) {
            spdlog::error("Invalid alphabet_size: {} (must be 2-8)", alphabet_size);
            return false;
        }
        
        uint32_t max_degree = get(config, "graph.max_degree", 0u);
        if (max_degree < 3 || max_degree > 12) {
            spdlog::error("Invalid max_degree: {} (must be 3-12)", max_degree);
            return false;
        }
        
        // Validate fitness weights
        double w_ricci = get(config, "fitness.weight_ricci", 0.0);
        double w_dim = get(config, "fitness.weight_dim", 0.0);
        double w_stab = get(config, "fitness.weight_stability", 0.0);
        double w_conn = get(config, "fitness.weight_connectivity", 0.0);
        
        if (w_ricci < 0 || w_dim < 0 || w_stab < 0 || w_conn < 0) {
            spdlog::error("Fitness weights must be non-negative");
            return false;
        }
        
        if (w_ricci + w_dim + w_stab + w_conn == 0) {
            spdlog::error("At least one fitness weight must be positive");
            return false;
        }
        
        spdlog::info("Configuration validation passed");
        return true;
    }
    
    /**
     * @brief Create default configuration
     * @return Default configuration table
     */
    static toml::table create_default() {
        return toml::table{
            {"graph", toml::table{
                {"initial_type", "lattice_4d"},
                {"initial_size", 10000},
                {"lattice_dims", toml::array{10, 10, 10, 10}},
                {"max_degree", 12}
            }},
            {"automaton", toml::table{
                {"alphabet_size", 3},
                {"laziness", 0.0},
                {"state_table_size", 64},
                {"num_weights", 3}
            }},
            {"evolution", toml::table{
                {"total_steps", 10000},
                {"transient_steps", 2000},
                {"sample_interval", 100},
                {"checkpoint_interval", 1000}
            }},
            {"geometry", toml::table{
                {"ollivier_alpha", 0.0},
                {"sinkhorn_iterations", 50},
                {"sinkhorn_epsilon", 0.01},
                {"random_walkers", 10000},
                {"walk_max_steps", 1000}
            }},
            {"search", toml::table{
                {"method", "ga_then_cmaes"},
                {"ga_population", 200},
                {"ga_generations", 3000},
                {"ga_tournament_k", 3},
                {"ga_crossover_alpha", 0.5},
                {"ga_mutation_sigma", 0.1},
                {"ga_elitism_fraction", 0.05},
                {"cmaes_sigma0", 0.1},
                {"cmaes_max_generations", 2000},
                {"num_ensemble", 5}
            }},
            {"fitness", toml::table{
                {"weight_ricci", 1.0},
                {"weight_dim", 0.5},
                {"weight_stability", 0.3},
                {"weight_connectivity", 10.0},
                {"target", "flat"}
            }},
            {"hardware", toml::table{
                {"use_gpu", true},
                {"gpu_block_size", 256},
                {"num_cpu_threads", 12}
            }},
            {"output", toml::table{
                {"results_dir", "data/results"},
                {"checkpoint_dir", "data/checkpoints"},
                {"log_dir", "data/logs"},
                {"log_level", "info"}
            }},
            {"reproducibility", toml::table{
                {"master_seed", 42}
            }}
        };
    }
};

/**
 * @brief Configuration holder for easy access
 */
class ConfigManager {
public:
    /**
     * @brief Load configuration from file or use defaults
     * @param path Configuration file path (empty for defaults)
     */
    static void load(const std::string& path = "") {
        if (path.empty()) {
            config_ = Config::create_default();
            spdlog::info("Using default configuration");
        } else {
            auto user_config = Config::load(path);
            config_ = Config::merge(Config::create_default(), user_config);
            spdlog::info("Loaded configuration from: {}", path);
        }
        
        if (!Config::validate(config_)) {
            throw std::runtime_error("Invalid configuration");
        }
    }
    
    /**
     * @brief Get configuration value
     * @param path Dot-separated path
     * @param default_value Default if not found
     * @return Configuration value
     */
    template<typename T>
    static T get(const std::string& path, const T& default_value) {
        return Config::get(config_, path, default_value);
    }
    
    /**
     * @brief Get entire configuration table
     * @return Configuration table
     */
    static const toml::table& get_config() {
        return config_;
    }
    
    /**
     * @brief Save current configuration
     * @param path Output path
     */
    static void save(const std::string& path) {
        Config::save(config_, path);
    }

private:
    static inline toml::table config_;
};

} // namespace discretum
