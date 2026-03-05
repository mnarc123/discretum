#include <iostream>
#include <fstream>
#include <filesystem>
#include <cmath>
#include <vector>
#include <set>
#include <random>
#include <fmt/core.h>
#include <spdlog/spdlog.h>
#include <nlohmann/json.hpp>

#include "core/graph.hpp"
#include "geometry/ollivier_ricci.hpp"
#include "geometry/spectral_dimension.hpp"
#include "geometry/geodesic.hpp"
#include "search/fitness.hpp"

using namespace discretum;
using json = nlohmann::json;

int main(int argc, char* argv[]) {
    spdlog::set_level(spdlog::level::warn);

    std::string graph_type = "lattice_4d";
    int L_min = 4, L_max = 8;
    uint32_t spectral_walkers = 10000;
    uint32_t spectral_steps = 1000;
    uint32_t curv_samples = 500;
    uint32_t hausdorff_sources = 50;
    std::string output_dir = "data/results/calibration_4d";

    if (argc >= 2) graph_type = argv[1];
    if (argc >= 3) L_min = std::stoi(argv[2]);
    if (argc >= 4) L_max = std::stoi(argv[3]);
    if (argc >= 5) output_dir = argv[4];

    std::filesystem::create_directories(output_dir);

    fmt::print("\n══════════════════════════════════════════════════════════\n");
    fmt::print("   DISCRETUM — Bare Lattice Calibration ({})\n", graph_type);
    fmt::print("══════════════════════════════════════════════════════════\n\n");
    fmt::print("{:>4} {:>8} {:>8} {:>8} {:>8} {:>10} {:>8}\n",
               "L", "N", "<k>", "d_H", "d_s", "<κ>", "σ(κ)");
    fmt::print("{}\n", std::string(66, '-'));

    json all_results = json::array();

    for (int L = L_min; L <= L_max; ++L) {
        DynamicGraph g;
        int expected_degree_internal;

        if (graph_type == "lattice_4d") {
            g = DynamicGraph::create_lattice_4d(L, L, L, L);
            expected_degree_internal = 8;
        } else {
            g = DynamicGraph::create_lattice_3d(L, L, L);
            expected_degree_internal = 6;
        }

        uint32_t N = g.num_nodes();
        uint32_t M = g.num_edges();
        double avg_deg = 2.0 * M / N;

        // Hausdorff dimension (sampled)
        double d_H = estimate_hausdorff_dimension_sampled(g, hausdorff_sources, 42);

        // Spectral dimension
        auto sd = compute_spectral_dimension_detailed(
            g, spectral_walkers, spectral_steps, 0.05f, 0.5f, 42);
        double d_s = sd.dimension;

        // Ollivier-Ricci curvature (sampled)
        std::vector<std::pair<uint32_t, uint32_t>> edge_list;
        {
            std::set<std::pair<uint32_t, uint32_t>> seen;
            for (uint32_t i = 0; i < N; ++i) {
                for (uint32_t nb : g.neighbors(i)) {
                    auto e = std::minmax(i, nb);
                    if (seen.insert(e).second) edge_list.push_back(e);
                }
            }
        }
        uint32_t sample_n = std::min(curv_samples, static_cast<uint32_t>(edge_list.size()));
        // Partial shuffle
        std::mt19937 rng(42);
        for (uint32_t i = 0; i < sample_n; ++i) {
            std::uniform_int_distribution<uint32_t> dist(i, static_cast<uint32_t>(edge_list.size()) - 1);
            std::swap(edge_list[i], edge_list[dist(rng)]);
        }
        double sum_k = 0.0, sum_k2 = 0.0;
        for (uint32_t i = 0; i < sample_n; ++i) {
            float k = compute_ollivier_ricci(g, edge_list[i].first, edge_list[i].second, 0.5f);
            sum_k += k;
            sum_k2 += static_cast<double>(k) * k;
        }
        double mean_curv = sum_k / sample_n;
        double var_curv = sum_k2 / sample_n - mean_curv * mean_curv;
        double std_curv = std::sqrt(std::max(0.0, var_curv));

        // Degree CV
        double mean_d = 0.0;
        for (uint32_t i = 0; i < N; ++i) mean_d += g.degree(i);
        mean_d /= N;
        double var_d = 0.0;
        for (uint32_t i = 0; i < N; ++i) {
            double diff = g.degree(i) - mean_d;
            var_d += diff * diff;
        }
        var_d /= N;
        double cv_deg = (mean_d > 0) ? std::sqrt(var_d) / mean_d : 0.0;

        // Connectivity
        auto comp = g.connected_components();

        fmt::print("{:>4} {:>8} {:>8.2f} {:>8.3f} {:>8.3f} {:>10.5f} {:>8.4f}\n",
                   L, N, avg_deg, d_H, d_s, mean_curv, std_curv);

        // Save per-size JSON
        json jr;
        jr["graph_type"] = graph_type;
        jr["L"] = L;
        jr["N"] = N;
        jr["M"] = M;
        jr["avg_degree"] = avg_deg;
        jr["cv_degree"] = cv_deg;
        jr["d_H"] = d_H;
        jr["d_s"] = d_s;
        jr["d_s_error"] = sd.fit_error;
        jr["mean_curvature"] = mean_curv;
        jr["std_curvature"] = std_curv;
        jr["n_components"] = comp.size();
        jr["spectral_walkers"] = spectral_walkers;
        jr["spectral_steps"] = spectral_steps;
        jr["curvature_samples"] = sample_n;
        jr["hausdorff_sources"] = hausdorff_sources;

        all_results.push_back(jr);

        std::string fpath = fmt::format("{}/L{}.json", output_dir, L);
        std::ofstream fout(fpath);
        fout << jr.dump(2) << std::endl;
    }

    fmt::print("{}\n\n", std::string(66, '-'));

    // Save combined baseline
    std::string baseline_path = output_dir + "/baseline.json";
    std::ofstream fout(baseline_path);
    fout << all_results.dump(2) << std::endl;
    fmt::print("Results saved to {}/\n", output_dir);

    // Sanity check
    if (graph_type == "lattice_4d") {
        fmt::print("\n── Sanity Check ──\n");
        fmt::print("Expected: d_H → 4.0 as L → ∞\n");
        auto& last = all_results.back();
        double last_dH = last["d_H"].get<double>();
        if (last_dH < 3.0) {
            fmt::print("⚠ WARNING: d_H = {:.3f} at L={} is far from 4.0!\n",
                       last_dH, last["L"].get<int>());
            fmt::print("  Possible bug in Hausdorff dimension estimator or lattice.\n");
        } else {
            fmt::print("✓ d_H = {:.3f} at L={} — trending toward 4.0\n",
                       last_dH, last["L"].get<int>());
        }
    }

    return 0;
}
