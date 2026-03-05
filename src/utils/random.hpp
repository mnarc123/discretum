#pragma once

#include <cstdint>
#include <vector>
#include <random>
#include <limits>

namespace discretum {

/**
 * @brief PCG-XSH-RR-64/32 (pcg32) random number generator
 * 
 * Implementation of the PCG family of random number generators.
 * 64-bit state, 32-bit output with XSH-RR output function.
 * 
 * References:
 * - O'Neill, M.E. (2014). "PCG: A Family of Simple Fast Space-Efficient 
 *   Statistically Good Algorithms for Random Number Generation"
 * - https://www.pcg-random.org/
 * 
 * Properties:
 * - Period: 2^64
 * - State size: 64 bits
 * - Output: 32 bits
 * - Passes all TestU01 BigCrush tests
 * 
 * Complexity: O(1) per generated number
 */
class PCG32 {
public:
    using result_type = uint32_t;
    
    /**
     * @brief Construct with default seed
     */
    PCG32() : PCG32(default_seed) {}
    
    /**
     * @brief Construct with specific seed
     * @param seed Initial seed value
     * @param stream Stream ID (allows multiple independent streams)
     */
    explicit PCG32(uint64_t seed, uint64_t stream = default_stream) {
        seed_impl(seed, stream);
    }
    
    /**
     * @brief Seed the generator
     * @param seed Seed value
     * @param stream Stream ID
     */
    void seed(uint64_t seed, uint64_t stream = default_stream) {
        seed_impl(seed, stream);
    }
    
    /**
     * @brief Generate next random number
     * @return Random 32-bit unsigned integer
     */
    uint32_t operator()() {
        return generate();
    }
    
    /**
     * @brief Generate random number in range [0, bound)
     * @param bound Upper bound (exclusive)
     * @return Random number in [0, bound)
     */
    uint32_t uniform(uint32_t bound) {
        // Lemire's nearly divisionless algorithm
        uint64_t product = static_cast<uint64_t>(generate()) * bound;
        uint32_t low = static_cast<uint32_t>(product);
        if (low < bound) {
            uint32_t threshold = -bound % bound;
            while (low < threshold) {
                product = static_cast<uint64_t>(generate()) * bound;
                low = static_cast<uint32_t>(product);
            }
        }
        return product >> 32;
    }
    
    /**
     * @brief Generate random float in [0, 1)
     * @return Random float
     */
    float uniform_float() {
        // Convert to float with 24 bits of precision
        return (generate() >> 8) * 0x1.0p-24f;
    }
    
    /**
     * @brief Generate random double in [0, 1)
     * @return Random double
     */
    double uniform_double() {
        // Use two calls to get 53 bits of precision
        uint64_t x = (static_cast<uint64_t>(generate()) << 20) | (generate() >> 12);
        return x * 0x1.0p-52;
    }
    
    /**
     * @brief Discard n values
     * @param n Number of values to discard
     */
    void discard(uint64_t n) {
        for (uint64_t i = 0; i < n; ++i) {
            generate();
        }
    }
    
    // STL UniformRandomBitGenerator requirements
    static constexpr result_type min() { return std::numeric_limits<result_type>::min(); }
    static constexpr result_type max() { return std::numeric_limits<result_type>::max(); }
    
    /**
     * @brief Get current state (for checkpointing)
     */
    uint64_t get_state() const { return state_; }
    
    /**
     * @brief Set state (for restoring from checkpoint)
     */
    void set_state(uint64_t state) { state_ = state; }

private:
    uint64_t state_;
    uint64_t increment_;
    
    static constexpr uint64_t default_seed = 0x853c49e6748fea9bULL;
    static constexpr uint64_t default_stream = 0xda3e39cb94b95bdbULL;
    static constexpr uint64_t multiplier = 0x5851f42d4c957f2dULL;
    
    void seed_impl(uint64_t seed, uint64_t stream) {
        state_ = 0;
        increment_ = (stream << 1) | 1;
        generate();
        state_ += seed;
        generate();
    }
    
    uint32_t generate() {
        uint64_t oldstate = state_;
        state_ = oldstate * multiplier + increment_;
        uint32_t xorshifted = ((oldstate >> 18) ^ oldstate) >> 27;
        uint32_t rot = oldstate >> 59;
        return (xorshifted >> rot) | (xorshifted << ((-rot) & 31));
    }
};

/**
 * @brief Thread-local random number generator with deterministic seeding
 * 
 * Provides thread-safe access to PCG32 generators with reproducible seeding
 * based on master seed and thread ID.
 */
class RandomGenerator {
public:
    /**
     * @brief Initialize with master seed
     * @param master_seed Global seed for reproducibility
     */
    static void initialize(uint64_t master_seed) {
        master_seed_ = master_seed;
    }
    
    /**
     * @brief Get thread-local generator
     * @param thread_id Thread identifier (defaults to 0 for single-threaded)
     * @return Reference to PCG32 generator
     */
    static PCG32& get(uint32_t thread_id = 0) {
        thread_local PCG32 generator(master_seed_ + thread_id);
        return generator;
    }
    
    /**
     * @brief Create independent generator with specific seed offset
     * @param offset Offset from master seed
     * @return New PCG32 generator
     */
    static PCG32 create_independent(uint64_t offset) {
        return PCG32(master_seed_ + offset);
    }
    
    /**
     * @brief Shuffle container elements
     * @param first Iterator to first element
     * @param last Iterator past last element
     * @param thread_id Thread identifier
     */
    template<typename RandomIt>
    static void shuffle(RandomIt first, RandomIt last, uint32_t thread_id = 0) {
        auto& gen = get(thread_id);
        using diff_t = typename std::iterator_traits<RandomIt>::difference_type;
        diff_t n = last - first;
        
        for (diff_t i = n - 1; i > 0; --i) {
            std::swap(first[i], first[gen.uniform(i + 1)]);
        }
    }
    
    /**
     * @brief Sample k elements without replacement
     * @param population Population to sample from
     * @param k Number of elements to sample
     * @param thread_id Thread identifier
     * @return Vector of sampled elements
     */
    template<typename T>
    static std::vector<T> sample(const std::vector<T>& population, size_t k, uint32_t thread_id = 0) {
        if (k > population.size()) {
            k = population.size();
        }
        
        auto& gen = get(thread_id);
        std::vector<T> result;
        result.reserve(k);
        
        // Use Floyd's algorithm for sampling without replacement
        std::unordered_set<size_t> selected;
        size_t n = population.size();
        
        for (size_t j = n - k; j < n; ++j) {
            size_t t = gen.uniform(j + 1);
            if (selected.find(t) == selected.end()) {
                selected.insert(t);
                result.push_back(population[t]);
            } else {
                selected.insert(j);
                result.push_back(population[j]);
            }
        }
        
        return result;
    }

private:
    static inline uint64_t master_seed_ = 42;  // Default seed
};

/**
 * @brief Utility functions for random number generation
 */
namespace random {

/**
 * @brief Generate random integer in range [min, max]
 * @param min Minimum value (inclusive)
 * @param max Maximum value (inclusive)
 * @param thread_id Thread identifier
 * @return Random integer in range
 */
inline int32_t uniform_int(int32_t min, int32_t max, uint32_t thread_id = 0) {
    auto& gen = RandomGenerator::get(thread_id);
    return min + gen.uniform(max - min + 1);
}

/**
 * @brief Generate random float in range [min, max)
 * @param min Minimum value (inclusive)
 * @param max Maximum value (exclusive)
 * @param thread_id Thread identifier
 * @return Random float in range
 */
inline float uniform_float(float min, float max, uint32_t thread_id = 0) {
    auto& gen = RandomGenerator::get(thread_id);
    return min + (max - min) * gen.uniform_float();
}

/**
 * @brief Generate random double in range [min, max)
 * @param min Minimum value (inclusive)
 * @param max Maximum value (exclusive)
 * @param thread_id Thread identifier
 * @return Random double in range
 */
inline double uniform_double(double min, double max, uint32_t thread_id = 0) {
    auto& gen = RandomGenerator::get(thread_id);
    return min + (max - min) * gen.uniform_double();
}

/**
 * @brief Generate random boolean with given probability
 * @param p Probability of true (default 0.5)
 * @param thread_id Thread identifier
 * @return Random boolean
 */
inline bool bernoulli(double p = 0.5, uint32_t thread_id = 0) {
    auto& gen = RandomGenerator::get(thread_id);
    return gen.uniform_double() < p;
}

/**
 * @brief Select random element from container
 * @param container Container to select from
 * @param thread_id Thread identifier
 * @return Random element
 */
template<typename Container>
inline auto choice(const Container& container, uint32_t thread_id = 0) 
    -> decltype(container[0]) {
    auto& gen = RandomGenerator::get(thread_id);
    return container[gen.uniform(container.size())];
}

} // namespace random

} // namespace discretum
