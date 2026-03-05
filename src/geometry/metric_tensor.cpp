#include "geometry/metric_tensor.hpp"
#include "geometry/geodesic.hpp"
#include <cmath>
#include <algorithm>
#include <queue>
#include <unordered_map>

namespace discretum {

// Build sorted vector of active node IDs and ID-to-index mapping
static std::pair<std::vector<uint32_t>, std::unordered_map<uint32_t, uint32_t>>
build_id_mapping(const DynamicGraph& graph) {
    std::vector<uint32_t> ids(graph.get_active_nodes().begin(), graph.get_active_nodes().end());
    std::sort(ids.begin(), ids.end());
    std::unordered_map<uint32_t, uint32_t> id_to_idx;
    for (uint32_t i = 0; i < ids.size(); ++i)
        id_to_idx[ids[i]] = i;
    return {ids, id_to_idx};
}

Eigen::MatrixXd mds_embedding(const DynamicGraph& graph, int target_dim) {
    uint32_t n = graph.num_nodes();
    if (n < 2) return Eigen::MatrixXd::Zero(n, target_dim);
    
    auto [ids, id_to_idx] = build_id_mapping(graph);
    
    // Compute distance matrix using BFS from each active node
    Eigen::MatrixXd D2(n, n);
    D2.setZero();
    for (uint32_t ii = 0; ii < n; ++ii) {
        uint32_t src = ids[ii];
        // BFS from src
        std::vector<uint32_t> dist(graph.get_nodes().size(), UINT32_MAX);
        dist[src] = 0;
        std::queue<uint32_t> q;
        q.push(src);
        while (!q.empty()) {
            uint32_t u = q.front(); q.pop();
            for (uint32_t v : graph.neighbors(u)) {
                if (dist[v] == UINT32_MAX) {
                    dist[v] = dist[u] + 1;
                    q.push(v);
                }
            }
        }
        for (uint32_t jj = 0; jj < n; ++jj) {
            double dd = (dist[ids[jj]] == UINT32_MAX) ? 0.0 : static_cast<double>(dist[ids[jj]]);
            D2(ii, jj) = dd * dd;
        }
    }
    
    // Classical MDS: double-centering
    Eigen::MatrixXd H = Eigen::MatrixXd::Identity(n, n) 
                       - Eigen::MatrixXd::Ones(n, n) / n;
    Eigen::MatrixXd B = -0.5 * H * D2 * H;
    
    // Eigendecomposition of B
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(B);
    Eigen::VectorXd eigenvalues = es.eigenvalues();
    Eigen::MatrixXd eigenvectors = es.eigenvectors();
    
    // Select top target_dim eigenvalues (largest, at end of sorted array)
    int d = std::min(target_dim, static_cast<int>(n));
    Eigen::MatrixXd coords(n, d);
    
    for (int k = 0; k < d; ++k) {
        int idx = n - 1 - k;  // Eigenvalues sorted ascending
        double eval = std::max(eigenvalues(idx), 0.0);
        coords.col(k) = std::sqrt(eval) * eigenvectors.col(idx);
    }
    
    return coords;
}

Eigen::MatrixXd compute_metric_tensor(const DynamicGraph& graph, int target_dim) {
    uint32_t n = graph.num_nodes();
    if (n < 2) return Eigen::MatrixXd::Identity(target_dim, target_dim);
    
    auto [ids, id_to_idx] = build_id_mapping(graph);
    Eigen::MatrixXd coords = mds_embedding(graph, target_dim);
    int d = static_cast<int>(coords.cols());
    
    // Estimate metric tensor from edge-based finite differences.
    // For each edge (u,v), the displacement vector is dx = coords[v] - coords[u].
    Eigen::MatrixXd A = Eigen::MatrixXd::Zero(d, d);
    int num_edges = 0;
    
    for (uint32_t uid : ids) {
        auto it_u = id_to_idx.find(uid);
        if (it_u == id_to_idx.end()) continue;
        uint32_t ui = it_u->second;
        for (uint32_t vid : graph.neighbors(uid)) {
            if (vid > uid) {
                auto it_v = id_to_idx.find(vid);
                if (it_v == id_to_idx.end()) continue;
                uint32_t vi = it_v->second;
                Eigen::VectorXd dx = coords.row(vi).transpose() - coords.row(ui).transpose();
                A += dx * dx.transpose();
                num_edges++;
            }
        }
    }
    
    if (num_edges > 0) A /= num_edges;
    return A;
}

double estimate_scalar_curvature(const DynamicGraph& graph, int target_dim) {
    Eigen::MatrixXd G = compute_metric_tensor(graph, target_dim);
    int d = static_cast<int>(G.rows());
    
    // For flat space, G should be proportional to identity.
    // Deviation from isotropy indicates curvature.
    // Estimate: R ~ (λ_max/λ_min - 1) as a simple anisotropy measure
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(G);
    Eigen::VectorXd evals = es.eigenvalues();
    
    double lambda_min = evals(0);
    double lambda_max = evals(d - 1);
    
    if (lambda_min <= 1e-15) return 0.0;
    
    // Anisotropy ratio as curvature proxy
    return (lambda_max / lambda_min - 1.0);
}

} // namespace discretum
