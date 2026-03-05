#pragma once

#include "core/graph.hpp"
#include <vector>
#include <cstdint>

namespace discretum {

/**
 * @brief Compute all-pairs shortest path distances via BFS.
 * @return dist[i][j] = shortest path length from node i to node j.
 *         UINT32_MAX if unreachable.
 */
std::vector<std::vector<uint32_t>> compute_all_pairs_shortest_paths(const DynamicGraph& graph);

/**
 * @brief Compute diameter (maximum shortest path distance) of the graph.
 */
uint32_t compute_diameter(const DynamicGraph& graph);

/**
 * @brief Compute distance distribution: count of node pairs at each distance.
 * @return dist_counts[d] = number of pairs (i,j) with i<j at distance d.
 */
std::vector<uint64_t> compute_distance_distribution(const DynamicGraph& graph);

/**
 * @brief Compute average shortest path length (over connected pairs).
 */
double compute_average_path_length(const DynamicGraph& graph);

/**
 * @brief Volume growth: N(r) = number of nodes within distance r of a source.
 * @return vol[r] = average over all sources of |{v : d(src,v) <= r}|.
 */
std::vector<double> compute_volume_growth(const DynamicGraph& graph);

/**
 * @brief Hausdorff dimension estimate from volume growth N(r) ~ r^{d_H}.
 *        Fits log N(r) vs log r in a scaling regime.
 *        Uses full APSP — O(N²) time and memory.
 */
double estimate_hausdorff_dimension(const DynamicGraph& graph);

/**
 * @brief Sampled Hausdorff dimension estimate using k random BFS sources.
 *        O(k·(V+E)) time, O(V) memory. Suitable for use in fitness evaluation.
 * @param num_sources Number of random BFS sources (default 30).
 * @param seed Random seed for source selection.
 */
double estimate_hausdorff_dimension_sampled(const DynamicGraph& graph,
                                            uint32_t num_sources = 30,
                                            uint64_t seed = 42);

} // namespace discretum
