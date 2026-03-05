#pragma once

#include "core/graph.hpp"
#include <Eigen/Dense>
#include <vector>

namespace discretum {

/**
 * @brief Estimate an effective metric tensor from a graph via MDS.
 *
 * Uses classical Multidimensional Scaling (MDS) on the distance matrix
 * to embed the graph in d-dimensional Euclidean space, then estimates
 * the metric tensor as G = J^T J where J is the Jacobian of the
 * embedding coordinates.
 *
 * @param graph The graph
 * @param target_dim Embedding dimension (default 3)
 * @return d×d metric tensor (symmetric positive semi-definite)
 */
Eigen::MatrixXd compute_metric_tensor(const DynamicGraph& graph, int target_dim = 3);

/**
 * @brief Embed graph in Euclidean space via classical MDS.
 * @return N×d matrix of node coordinates.
 */
Eigen::MatrixXd mds_embedding(const DynamicGraph& graph, int target_dim = 3);

/**
 * @brief Compute scalar curvature from metric tensor eigenvalues.
 *
 * For a graph embedded in d dimensions, estimates an effective scalar
 * curvature by comparing local volume elements to flat space.
 */
double estimate_scalar_curvature(const DynamicGraph& graph, int target_dim = 3);

} // namespace discretum
