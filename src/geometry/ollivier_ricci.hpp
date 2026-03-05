#pragma once

#include "core/graph.hpp"
#include <vector>
#include <map>
#include <unordered_map>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <limits>

namespace discretum {

/**
 * @brief Compute Ollivier-Ricci curvature for an edge
 * 
 * The Ollivier-Ricci curvature is defined as:
 * κ(x,y) = 1 - W_1(μ_x, μ_y) / d(x,y)
 * 
 * where:
 * - W_1 is the Wasserstein-1 distance (Earth Mover's Distance)
 * - μ_x is the probability measure of lazy random walk from x
 * - d(x,y) is the graph distance between x and y
 * 
 * References:
 * - Ollivier, Y. (2009). "Ricci curvature of Markov chains on metric spaces"
 * - Lin, Y., Lu, L., & Yau, S. T. (2011). "Ricci curvature of graphs"
 * 
 * @param graph The graph
 * @param u First node of the edge
 * @param v Second node of the edge  
 * @param alpha Laziness parameter (probability of staying at current node)
 * @return Ollivier-Ricci curvature of edge (u,v)
 * 
 * Complexity: O(k^2 log k) where k is the maximum degree
 */
float compute_ollivier_ricci(const DynamicGraph& graph, uint32_t u, uint32_t v, float alpha = 0.0f);

/**
 * @brief Compute Ollivier-Ricci curvature for all edges
 * 
 * @param graph The graph
 * @param alpha Laziness parameter
 * @return Map from edge (as pair of nodes) to curvature
 */
std::map<std::pair<uint32_t, uint32_t>, float>
compute_all_ollivier_ricci(const DynamicGraph& graph, float alpha = 0.0f);

/**
 * @brief Compute average Ollivier-Ricci curvature
 * 
 * @param graph The graph
 * @param alpha Laziness parameter
 * @return Average curvature over all edges
 */
float compute_average_ollivier_ricci(const DynamicGraph& graph, float alpha = 0.0f);

/**
 * @brief Statistics for Ollivier-Ricci curvature distribution
 */
struct OllivierRicciStats {
    float mean;
    float std_dev;
    float min;
    float max;
    float median;
    size_t num_edges;
    size_t num_positive;  // Edges with positive curvature
    size_t num_negative;  // Edges with negative curvature
    size_t num_zero;      // Edges with zero curvature (within tolerance)
};

/**
 * @brief Compute statistics of Ollivier-Ricci curvature distribution
 * 
 * @param graph The graph
 * @param alpha Laziness parameter
 * @param zero_tolerance Tolerance for considering curvature as zero
 * @return Curvature statistics
 */
OllivierRicciStats compute_ollivier_ricci_stats(const DynamicGraph& graph, 
                                                float alpha = 0.0f,
                                                float zero_tolerance = 1e-6f);

/**
 * @brief Compute Wasserstein-1 distance between two probability distributions
 * 
 * Uses the Hungarian algorithm for optimal transport on small support sets.
 * For graphs with degree <= 12, this is very efficient.
 * 
 * @param dist_matrix Distance matrix between support points
 * @param mu_x Probability distribution on first support
 * @param mu_y Probability distribution on second support
 * @return Wasserstein-1 distance
 * 
 * Complexity: O(n^3) where n is the support size
 */
float wasserstein_1_distance(const std::vector<std::vector<float>>& dist_matrix,
                            const std::vector<float>& mu_x,
                            const std::vector<float>& mu_y);

/**
 * @brief Build lazy random walk probability measure
 * 
 * @param graph The graph
 * @param node Center node
 * @param alpha Laziness parameter
 * @return Map from node to probability
 */
std::unordered_map<uint32_t, float> build_probability_measure(const DynamicGraph& graph,
                                                             uint32_t node,
                                                             float alpha);

} // namespace discretum
