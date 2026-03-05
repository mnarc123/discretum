#include "geometry/geodesic.hpp"
#include <queue>
#include <limits>
#include <cmath>
#include <algorithm>
#include <numeric>

namespace discretum {

static std::vector<uint32_t> bfs_distances(const DynamicGraph& graph, uint32_t source) {
    uint32_t n = graph.num_nodes();
    std::vector<uint32_t> dist(n, UINT32_MAX);
    dist[source] = 0;
    std::queue<uint32_t> q;
    q.push(source);
    while (!q.empty()) {
        uint32_t u = q.front(); q.pop();
        for (uint32_t v : graph.neighbors(u)) {
            if (dist[v] == UINT32_MAX) {
                dist[v] = dist[u] + 1;
                q.push(v);
            }
        }
    }
    return dist;
}

std::vector<std::vector<uint32_t>> compute_all_pairs_shortest_paths(const DynamicGraph& graph) {
    uint32_t n = graph.num_nodes();
    std::vector<std::vector<uint32_t>> all_dist(n);
    for (uint32_t i = 0; i < n; ++i)
        all_dist[i] = bfs_distances(graph, i);
    return all_dist;
}

uint32_t compute_diameter(const DynamicGraph& graph) {
    auto all_dist = compute_all_pairs_shortest_paths(graph);
    uint32_t diam = 0;
    for (auto& row : all_dist)
        for (uint32_t d : row)
            if (d != UINT32_MAX && d > diam) diam = d;
    return diam;
}

std::vector<uint64_t> compute_distance_distribution(const DynamicGraph& graph) {
    auto all_dist = compute_all_pairs_shortest_paths(graph);
    uint32_t n = graph.num_nodes();
    
    uint32_t max_d = 0;
    for (auto& row : all_dist)
        for (uint32_t d : row)
            if (d != UINT32_MAX && d > max_d) max_d = d;
    
    std::vector<uint64_t> counts(max_d + 1, 0);
    for (uint32_t i = 0; i < n; ++i)
        for (uint32_t j = i + 1; j < n; ++j)
            if (all_dist[i][j] != UINT32_MAX)
                counts[all_dist[i][j]]++;
    return counts;
}

double compute_average_path_length(const DynamicGraph& graph) {
    auto all_dist = compute_all_pairs_shortest_paths(graph);
    uint32_t n = graph.num_nodes();
    double sum = 0.0;
    uint64_t count = 0;
    for (uint32_t i = 0; i < n; ++i)
        for (uint32_t j = i + 1; j < n; ++j)
            if (all_dist[i][j] != UINT32_MAX) {
                sum += all_dist[i][j];
                count++;
            }
    return count > 0 ? sum / count : 0.0;
}

std::vector<double> compute_volume_growth(const DynamicGraph& graph) {
    auto all_dist = compute_all_pairs_shortest_paths(graph);
    uint32_t n = graph.num_nodes();
    
    uint32_t diam = 0;
    for (auto& row : all_dist)
        for (uint32_t d : row)
            if (d != UINT32_MAX && d > diam) diam = d;
    
    // vol[r] = average over sources of |{v : d(src,v) <= r}|
    std::vector<double> vol(diam + 1, 0.0);
    for (uint32_t src = 0; src < n; ++src) {
        for (uint32_t r = 0; r <= diam; ++r) {
            uint32_t count = 0;
            for (uint32_t v = 0; v < n; ++v)
                if (all_dist[src][v] <= r) count++;
            vol[r] += count;
        }
    }
    for (auto& v : vol) v /= n;
    return vol;
}

// Helper: linear regression slope on (x, y) arrays
static double linreg_slope(const std::vector<double>& x, const std::vector<double>& y,
                           int start, int end) {
    int n = end - start;
    if (n < 2) return 0.0;
    double sx = 0, sy = 0, sxx = 0, sxy = 0;
    for (int i = start; i < end; ++i) {
        sx += x[i]; sy += y[i];
        sxx += x[i] * x[i]; sxy += x[i] * y[i];
    }
    double denom = n * sxx - sx * sx;
    if (std::abs(denom) < 1e-20) return 0.0;
    return (n * sxy - sx * sy) / denom;
}

// Helper: estimate d_H from cumulative volume growth N(r).
//
// Uses two complementary methods and returns the better estimate:
//
// Method A — Shell volume regression:
//   S(r) = N(r) - N(r-1) ~ r^{d_H - 1}, so d_H = 1 + slope(log S vs log r).
//   The shell volume is more sensitive than cumulative volume and reaches the
//   correct scaling at smaller r.
//
// Method B — Local log-derivative of cumulative volume:
//   d_H(r) = d(ln N)/d(ln r), take maximum in the pre-saturation regime.
//
// Both methods exclude the saturation regime (N(r) > 50% of N_total).
// Returns the maximum of the two estimates.
static double fit_hausdorff_from_vol(const std::vector<double>& vol, double N_total) {
    if (vol.size() < 4) return 0.0;

    double sat_thresh = 0.5 * N_total;

    // Method A: shell volume S(r) = N(r) - N(r-1), fit log S vs log r
    std::vector<double> log_r_shell, log_shell;
    for (int r = 2; r < static_cast<int>(vol.size()); ++r) {
        if (vol[r] > sat_thresh) break;
        double shell = vol[r] - vol[r-1];
        if (shell > 0.5) {  // need at least ~1 node in shell
            log_r_shell.push_back(std::log(static_cast<double>(r)));
            log_shell.push_back(std::log(shell));
        }
    }

    double dH_shell = 0.0;
    if (log_r_shell.size() >= 2) {
        double slope = linreg_slope(log_r_shell, log_shell, 0,
                                    static_cast<int>(log_r_shell.size()));
        dH_shell = 1.0 + slope;  // S(r) ~ r^{d-1}
    }

    // Method B: local log-derivative, take max of plateau
    std::vector<double> local_dH;
    for (int r = 2; r + 1 < static_cast<int>(vol.size()); ++r) {
        if (vol[r] < 2.0 || vol[r+1] < 2.0) continue;
        if (vol[r] > sat_thresh) break;
        double dH_r = std::log(vol[r+1] / vol[r]) / std::log(static_cast<double>(r+1) / r);
        if (std::isfinite(dH_r) && dH_r > 0) {
            local_dH.push_back(dH_r);
        }
    }

    double dH_local = 0.0;
    if (!local_dH.empty()) {
        double max_val = *std::max_element(local_dH.begin(), local_dH.end());
        double threshold = max_val * 0.85;
        double sum = 0.0; int count = 0;
        for (double d : local_dH) {
            if (d >= threshold) { sum += d; count++; }
        }
        dH_local = (count > 0) ? sum / count : max_val;
    }

    // Return the maximum: shell method is typically better for small lattices,
    // local derivative is better for evolved graphs with irregular structure
    return std::max(dH_shell, dH_local);
}

double estimate_hausdorff_dimension(const DynamicGraph& graph) {
    auto vol = compute_volume_growth(graph);
    return fit_hausdorff_from_vol(vol, static_cast<double>(graph.num_nodes()));
}

double estimate_hausdorff_dimension_sampled(const DynamicGraph& graph,
                                            uint32_t num_sources,
                                            uint64_t seed) {
    uint32_t n = graph.num_nodes();
    if (n < 4) return 0.0;

    // Prefer high-eccentricity nodes as BFS sources (those deep in the interior).
    // Heuristic: nodes with maximum degree are most likely interior nodes
    // (boundary nodes have lower degree in lattices). Sort by degree descending
    // and pick top num_sources.
    struct NodeDeg { uint32_t id; uint16_t deg; };
    std::vector<NodeDeg> candidates;
    candidates.reserve(n);
    for (uint32_t i = 0; i < n; ++i)
        candidates.push_back({i, graph.degree(i)});

    // Sort by degree descending (interior nodes have max degree)
    std::sort(candidates.begin(), candidates.end(),
              [](const NodeDeg& a, const NodeDeg& b) { return a.deg > b.deg; });

    // Take the top nodes by degree, then shuffle among them
    uint32_t pool_size = std::min(static_cast<uint32_t>(candidates.size()),
                                   num_sources * 4);  // 4× oversampling pool
    uint32_t k = std::min(num_sources, pool_size);

    // Shuffle within the high-degree pool
    uint64_t state = seed;
    auto pcg = [&]() -> uint32_t {
        state = state * 6364136223846793005ULL + 1442695040888963407ULL;
        uint32_t x = static_cast<uint32_t>((state >> 22) ^ state) >> (state >> 61);
        return x;
    };
    for (uint32_t i = 0; i < k; ++i) {
        uint32_t j = i + (pcg() % (pool_size - i));
        std::swap(candidates[i], candidates[j]);
    }

    // BFS from each selected source, accumulate volume growth
    uint32_t max_diam = 0;
    std::vector<std::vector<uint32_t>> all_dist_counts(k);
    for (uint32_t si = 0; si < k; ++si) {
        auto dist = bfs_distances(graph, candidates[si].id);
        uint32_t local_max = 0;
        for (uint32_t d : dist)
            if (d != UINT32_MAX && d > local_max) local_max = d;
        max_diam = std::max(max_diam, local_max);
        all_dist_counts[si].resize(local_max + 1, 0);
        for (uint32_t d : dist)
            if (d != UINT32_MAX) all_dist_counts[si][d]++;
    }

    if (max_diam < 2) return 0.0;

    // Average volume growth: vol[r] = avg over sources of |{v : d(src,v) <= r}|
    std::vector<double> vol(max_diam + 1, 0.0);
    for (uint32_t si = 0; si < k; ++si) {
        double cumul = 0.0;
        for (uint32_t r = 0; r <= max_diam; ++r) {
            if (r < all_dist_counts[si].size())
                cumul += all_dist_counts[si][r];
            vol[r] += cumul;
        }
    }
    for (auto& v : vol) v /= k;

    // Use improved fitting with saturation cutoff and local derivative
    return fit_hausdorff_from_vol(vol, static_cast<double>(n));
}

} // namespace discretum
