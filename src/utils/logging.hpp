#pragma once

#include <spdlog/spdlog.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/sinks/rotating_file_sink.h>
#include <spdlog/fmt/ostr.h>
#include <iostream>
#include <memory>
#include <string>

namespace discretum {

/**
 * @brief Centralized logging system using spdlog
 * 
 * Provides structured logging with multiple sinks (console, file)
 * and configurable log levels.
 * 
 * Features:
 * - Color-coded console output
 * - Rotating file logs
 * - JSON structured logging
 * - Thread-safe
 * - High performance
 */
class Logger {
public:
    /**
     * @brief Initialize the logging system
     * @param log_dir Directory for log files
     * @param console_level Minimum level for console output
     * @param file_level Minimum level for file output
     * @param max_file_size Maximum size of each log file in MB
     * @param max_files Maximum number of rotating log files
     */
    static void initialize(
        const std::string& log_dir = "data/logs",
        spdlog::level::level_enum console_level = spdlog::level::info,
        spdlog::level::level_enum file_level = spdlog::level::debug,
        size_t max_file_size = 100,  // MB
        size_t max_files = 3
    ) {
        try {
            // Create sinks
            auto console_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
            console_sink->set_level(console_level);
            console_sink->set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%^%l%$] [%t] %v");
            
            auto file_sink = std::make_shared<spdlog::sinks::rotating_file_sink_mt>(
                log_dir + "/discretum.log", 
                max_file_size * 1024 * 1024, 
                max_files
            );
            file_sink->set_level(file_level);
            file_sink->set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%l] [%t] [%@] %v");
            
            // Create logger with both sinks
            std::vector<spdlog::sink_ptr> sinks {console_sink, file_sink};
            auto logger = std::make_shared<spdlog::logger>("discretum", sinks.begin(), sinks.end());
            logger->set_level(spdlog::level::trace);  // Allow all levels, sinks filter
            
            // Register as default logger
            spdlog::set_default_logger(logger);
            
            // Set flush policy
            spdlog::flush_on(spdlog::level::warn);
            spdlog::flush_every(std::chrono::seconds(5));
            
            spdlog::info("Logger initialized successfully");
            spdlog::info("Console level: {}, File level: {}", 
                        spdlog::level::to_string_view(console_level),
                        spdlog::level::to_string_view(file_level));
        }
        catch (const spdlog::spdlog_ex& ex) {
            std::cerr << "Log initialization failed: " << ex.what() << std::endl;
        }
    }
    
    /**
     * @brief Create a specialized logger for a component
     * @param name Component name
     * @return Shared pointer to logger
     */
    static std::shared_ptr<spdlog::logger> create_component_logger(const std::string& name) {
        auto logger = spdlog::get(name);
        if (!logger) {
            logger = spdlog::default_logger()->clone(name);
            spdlog::register_logger(logger);
        }
        return logger;
    }
    
    /**
     * @brief Set global log level
     * @param level New log level
     */
    static void set_level(spdlog::level::level_enum level) {
        spdlog::set_level(level);
    }
    
    /**
     * @brief Flush all loggers
     */
    static void flush() {
        spdlog::apply_all([](std::shared_ptr<spdlog::logger> l) { l->flush(); });
    }
    
    /**
     * @brief Shutdown logging system
     */
    static void shutdown() {
        spdlog::shutdown();
    }
};

/**
 * @brief Scoped timer for performance profiling
 * 
 * RAII timer that logs elapsed time on destruction
 */
class ScopedTimer {
public:
    /**
     * @brief Construct timer with description
     * @param description Description of timed operation
     * @param log_level Level at which to log (default: debug)
     */
    explicit ScopedTimer(const std::string& description, 
                        spdlog::level::level_enum log_level = spdlog::level::debug)
        : description_(description), log_level_(log_level), 
          start_(std::chrono::high_resolution_clock::now()) {
        spdlog::log(log_level_, "[TIMER] {} started", description_);
    }
    
    ~ScopedTimer() {
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start_);
        
        double ms = duration.count() / 1000.0;
        spdlog::log(log_level_, "[TIMER] {} completed in {:.3f} ms", description_, ms);
    }
    
    // Delete copy/move to ensure single timing
    ScopedTimer(const ScopedTimer&) = delete;
    ScopedTimer& operator=(const ScopedTimer&) = delete;
    ScopedTimer(ScopedTimer&&) = delete;
    ScopedTimer& operator=(ScopedTimer&&) = delete;

private:
    std::string description_;
    spdlog::level::level_enum log_level_;
    std::chrono::high_resolution_clock::time_point start_;
};

/**
 * @brief Structured logging helpers
 */
namespace log {

/**
 * @brief Log graph statistics
 */
inline void graph_stats(const std::string& name, uint32_t nodes, uint32_t edges, 
                       float avg_degree, bool connected) {
    spdlog::info("Graph '{}': nodes={}, edges={}, avg_degree={:.2f}, connected={}", 
                 name, nodes, edges, avg_degree, connected);
}

/**
 * @brief Log curvature statistics
 */
inline void curvature_stats(const std::string& method, double mean, double std_dev, 
                           double min, double max, size_t samples) {
    spdlog::info("Curvature [{}]: mean={:.6f}, std={:.6f}, min={:.6f}, max={:.6f}, n={}", 
                 method, mean, std_dev, min, max, samples);
}

/**
 * @brief Log evolution step
 */
inline void evolution_step(uint32_t step, uint32_t nodes, uint32_t edges, 
                          double energy, double time_ms) {
    spdlog::debug("Evolution step {}: nodes={}, edges={}, energy={:.3f}, time={:.1f}ms", 
                  step, nodes, edges, energy, time_ms);
}

/**
 * @brief Log fitness evaluation
 */
inline void fitness_eval(uint32_t generation, uint32_t individual, double fitness,
                        double ricci_component, double dim_component, 
                        double stability_component, double conn_component) {
    spdlog::debug("Fitness [gen={}, ind={}]: total={:.6f} "
                  "(ricci={:.3f}, dim={:.3f}, stab={:.3f}, conn={:.3f})",
                  generation, individual, fitness, 
                  ricci_component, dim_component, stability_component, conn_component);
}

/**
 * @brief Log search progress
 */
inline void search_progress(const std::string& algorithm, uint32_t generation, 
                           double best_fitness, double avg_fitness, double diversity) {
    spdlog::info("Search [{}] gen={}: best={:.6f}, avg={:.6f}, diversity={:.3f}", 
                 algorithm, generation, best_fitness, avg_fitness, diversity);
}

/**
 * @brief Log validation result
 */
inline void validation_result(const std::string& test_name, bool passed, 
                             double expected, double actual, double tolerance) {
    if (passed) {
        spdlog::info("Validation '{}' PASSED: expected={:.6f}, actual={:.6f}, tol={:.6f}", 
                     test_name, expected, actual, tolerance);
    } else {
        spdlog::error("Validation '{}' FAILED: expected={:.6f}, actual={:.6f}, tol={:.6f}", 
                      test_name, expected, actual, tolerance);
    }
}

/**
 * @brief Log checkpoint save
 */
inline void checkpoint_saved(const std::string& path, size_t size_bytes, double time_ms) {
    spdlog::info("Checkpoint saved: path='{}', size={:.1f}MB, time={:.1f}ms", 
                 path, size_bytes / (1024.0 * 1024.0), time_ms);
}

/**
 * @brief Log memory usage
 */
inline void memory_usage(size_t graph_bytes, size_t state_bytes, size_t total_bytes) {
    spdlog::debug("Memory usage: graph={:.1f}MB, state={:.1f}MB, total={:.1f}MB",
                  graph_bytes / (1024.0 * 1024.0),
                  state_bytes / (1024.0 * 1024.0),
                  total_bytes / (1024.0 * 1024.0));
}

} // namespace log

} // namespace discretum

// Convenience macros for common logging patterns
#define LOG_TRACE(...) SPDLOG_TRACE(__VA_ARGS__)
#define LOG_DEBUG(...) SPDLOG_DEBUG(__VA_ARGS__)
#define LOG_INFO(...) SPDLOG_INFO(__VA_ARGS__)
#define LOG_WARN(...) SPDLOG_WARN(__VA_ARGS__)
#define LOG_ERROR(...) SPDLOG_ERROR(__VA_ARGS__)
#define LOG_CRITICAL(...) SPDLOG_CRITICAL(__VA_ARGS__)

// Scoped timer macro
#define TIMED_SCOPE(description) discretum::ScopedTimer _timer(description)
#define TIMED_SCOPE_LEVEL(description, level) discretum::ScopedTimer _timer(description, level)
