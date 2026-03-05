#pragma once

#include <chrono>
#include <string>
#include <unordered_map>
#include <vector>
#include <algorithm>
#include <numeric>
#include <cmath>
#include "utils/logging.hpp"

namespace discretum {

/**
 * @brief High-resolution timer for performance profiling
 * 
 * Provides nanosecond-precision timing with statistical analysis
 * of multiple measurements.
 */
class Timer {
public:
    using clock = std::chrono::high_resolution_clock;
    using time_point = clock::time_point;
    using duration = std::chrono::nanoseconds;
    
    /**
     * @brief Start timing
     */
    void start() {
        start_time_ = clock::now();
        is_running_ = true;
    }
    
    /**
     * @brief Stop timing and return elapsed time
     * @return Elapsed time in nanoseconds
     */
    duration stop() {
        if (!is_running_) {
            return duration::zero();
        }
        
        auto end_time = clock::now();
        is_running_ = false;
        return std::chrono::duration_cast<duration>(end_time - start_time_);
    }
    
    /**
     * @brief Get elapsed time without stopping
     * @return Elapsed time in nanoseconds
     */
    duration elapsed() const {
        if (!is_running_) {
            return duration::zero();
        }
        
        auto current_time = clock::now();
        return std::chrono::duration_cast<duration>(current_time - start_time_);
    }
    
    /**
     * @brief Check if timer is running
     * @return True if running
     */
    bool is_running() const { return is_running_; }
    
    /**
     * @brief Reset timer
     */
    void reset() {
        is_running_ = false;
        start_time_ = time_point{};
    }
    
    /**
     * @brief Convert duration to milliseconds
     * @param d Duration
     * @return Milliseconds as double
     */
    static double to_ms(duration d) {
        return d.count() / 1'000'000.0;
    }
    
    /**
     * @brief Convert duration to microseconds
     * @param d Duration
     * @return Microseconds as double
     */
    static double to_us(duration d) {
        return d.count() / 1'000.0;
    }
    
    /**
     * @brief Convert duration to seconds
     * @param d Duration
     * @return Seconds as double
     */
    static double to_s(duration d) {
        return d.count() / 1'000'000'000.0;
    }

private:
    time_point start_time_;
    bool is_running_ = false;
};

/**
 * @brief Statistical timer for collecting multiple measurements
 */
class StatisticalTimer {
public:
    /**
     * @brief Add a measurement
     * @param duration Measured duration
     */
    void add_measurement(Timer::duration duration) {
        measurements_.push_back(duration.count());
    }
    
    /**
     * @brief Add a measurement in nanoseconds
     * @param ns Nanoseconds
     */
    void add_measurement_ns(int64_t ns) {
        measurements_.push_back(ns);
    }
    
    /**
     * @brief Clear all measurements
     */
    void clear() {
        measurements_.clear();
    }
    
    /**
     * @brief Get number of measurements
     * @return Count
     */
    size_t count() const { return measurements_.size(); }
    
    /**
     * @brief Get mean duration
     * @return Mean in nanoseconds
     */
    double mean() const {
        if (measurements_.empty()) return 0.0;
        
        double sum = std::accumulate(measurements_.begin(), measurements_.end(), 0.0);
        return sum / measurements_.size();
    }
    
    /**
     * @brief Get standard deviation
     * @return Standard deviation in nanoseconds
     */
    double std_dev() const {
        if (measurements_.size() < 2) return 0.0;
        
        double m = mean();
        double sq_sum = 0.0;
        
        for (int64_t val : measurements_) {
            double diff = val - m;
            sq_sum += diff * diff;
        }
        
        return std::sqrt(sq_sum / (measurements_.size() - 1));
    }
    
    /**
     * @brief Get minimum duration
     * @return Min in nanoseconds
     */
    int64_t min() const {
        if (measurements_.empty()) return 0;
        return *std::min_element(measurements_.begin(), measurements_.end());
    }
    
    /**
     * @brief Get maximum duration
     * @return Max in nanoseconds
     */
    int64_t max() const {
        if (measurements_.empty()) return 0;
        return *std::max_element(measurements_.begin(), measurements_.end());
    }
    
    /**
     * @brief Get median duration
     * @return Median in nanoseconds
     */
    double median() const {
        if (measurements_.empty()) return 0.0;
        
        std::vector<int64_t> sorted = measurements_;
        std::sort(sorted.begin(), sorted.end());
        
        size_t n = sorted.size();
        if (n % 2 == 0) {
            return (sorted[n/2 - 1] + sorted[n/2]) / 2.0;
        } else {
            return sorted[n/2];
        }
    }
    
    /**
     * @brief Get percentile
     * @param p Percentile (0-100)
     * @return Value at percentile in nanoseconds
     */
    double percentile(double p) const {
        if (measurements_.empty()) return 0.0;
        if (p <= 0) return min();
        if (p >= 100) return max();
        
        std::vector<int64_t> sorted = measurements_;
        std::sort(sorted.begin(), sorted.end());
        
        double index = (p / 100.0) * (sorted.size() - 1);
        size_t lower = static_cast<size_t>(std::floor(index));
        size_t upper = static_cast<size_t>(std::ceil(index));
        
        if (lower == upper) {
            return sorted[lower];
        }
        
        double weight = index - lower;
        return sorted[lower] * (1 - weight) + sorted[upper] * weight;
    }
    
    /**
     * @brief Log statistics
     * @param name Timer name
     */
    void log_stats(const std::string& name) const {
        if (measurements_.empty()) {
            spdlog::warn("No measurements for timer '{}'", name);
            return;
        }
        
        spdlog::info("Timer '{}' statistics (n={}):", name, count());
        spdlog::info("  Mean: {:.3f} ms (std: {:.3f} ms)", 
                     Timer::to_ms(Timer::duration(static_cast<int64_t>(mean()))),
                     Timer::to_ms(Timer::duration(static_cast<int64_t>(std_dev()))));
        spdlog::info("  Min: {:.3f} ms, Max: {:.3f} ms", 
                     Timer::to_ms(Timer::duration(min())),
                     Timer::to_ms(Timer::duration(max())));
        spdlog::info("  Median: {:.3f} ms", 
                     Timer::to_ms(Timer::duration(static_cast<int64_t>(median()))));
        spdlog::info("  P95: {:.3f} ms, P99: {:.3f} ms", 
                     Timer::to_ms(Timer::duration(static_cast<int64_t>(percentile(95)))),
                     Timer::to_ms(Timer::duration(static_cast<int64_t>(percentile(99)))));
    }

private:
    std::vector<int64_t> measurements_;  // Nanoseconds
};

/**
 * @brief Global timer registry for named timers
 */
class TimerRegistry {
public:
    /**
     * @brief Get or create a statistical timer
     * @param name Timer name
     * @return Reference to timer
     */
    static StatisticalTimer& get(const std::string& name) {
        return timers_[name];
    }
    
    /**
     * @brief Time a function and add to named timer
     * @param name Timer name
     * @param func Function to time
     * @return Function result
     */
    template<typename Func>
    static auto time(const std::string& name, Func&& func) -> decltype(func()) {
        Timer timer;
        timer.start();
        
        auto result = func();
        
        auto duration = timer.stop();
        get(name).add_measurement(duration);
        
        return result;
    }
    
    /**
     * @brief Log all timer statistics
     */
    static void log_all() {
        spdlog::info("=== Timer Statistics ===");
        for (const auto& [name, timer] : timers_) {
            timer.log_stats(name);
        }
        spdlog::info("=======================");
    }
    
    /**
     * @brief Clear all timers
     */
    static void clear_all() {
        timers_.clear();
    }
    
    /**
     * @brief Clear specific timer
     * @param name Timer name
     */
    static void clear(const std::string& name) {
        timers_[name].clear();
    }

private:
    static inline std::unordered_map<std::string, StatisticalTimer> timers_;
};

/**
 * @brief RAII timer that automatically records to registry
 */
class AutoTimer {
public:
    /**
     * @brief Start timing for named timer
     * @param name Timer name
     */
    explicit AutoTimer(const std::string& name) : name_(name) {
        timer_.start();
    }
    
    ~AutoTimer() {
        auto duration = timer_.stop();
        TimerRegistry::get(name_).add_measurement(duration);
    }
    
    // Delete copy/move
    AutoTimer(const AutoTimer&) = delete;
    AutoTimer& operator=(const AutoTimer&) = delete;
    AutoTimer(AutoTimer&&) = delete;
    AutoTimer& operator=(AutoTimer&&) = delete;

private:
    std::string name_;
    Timer timer_;
};

// Convenience macros
#define AUTO_TIMER(name) discretum::AutoTimer _auto_timer_##__LINE__(name)
#define TIME_FUNCTION(name, func) discretum::TimerRegistry::time(name, func)

} // namespace discretum
