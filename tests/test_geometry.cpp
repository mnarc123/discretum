#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include "geometry/geodesic.hpp"
#include "geometry/metric_tensor.hpp"

using namespace discretum;
using Catch::Matchers::WithinAbs;

TEST_CASE("Geodesic analysis", "[geometry][geodesic]") {
    SECTION("Path graph P_10 distances") {
        DynamicGraph graph(10);
        for (int i = 0; i < 9; ++i) graph.add_edge(i, i + 1);
        
        auto dist = compute_all_pairs_shortest_paths(graph);
        REQUIRE(dist[0][9] == 9);
        REQUIRE(dist[0][0] == 0);
        REQUIRE(dist[3][7] == 4);
    }
    
    SECTION("Path graph diameter") {
        DynamicGraph graph(10);
        for (int i = 0; i < 9; ++i) graph.add_edge(i, i + 1);
        
        REQUIRE(compute_diameter(graph) == 9);
    }
    
    SECTION("Cycle graph C_10 diameter") {
        DynamicGraph graph(10);
        for (int i = 0; i < 10; ++i) graph.add_edge(i, (i + 1) % 10);
        
        REQUIRE(compute_diameter(graph) == 5);
    }
    
    SECTION("Average path length of K_4") {
        DynamicGraph graph(4);
        for (int i = 0; i < 4; ++i)
            for (int j = i + 1; j < 4; ++j)
                graph.add_edge(i, j);
        
        // All pairs are distance 1
        REQUIRE_THAT(compute_average_path_length(graph), WithinAbs(1.0, 1e-10));
    }
    
    SECTION("Distance distribution of P_5") {
        DynamicGraph graph(5);
        for (int i = 0; i < 4; ++i) graph.add_edge(i, i + 1);
        
        auto dist_counts = compute_distance_distribution(graph);
        // d=1: 4 pairs, d=2: 3 pairs, d=3: 2 pairs, d=4: 1 pair
        REQUIRE(dist_counts.size() >= 5);
        REQUIRE(dist_counts[1] == 4);
        REQUIRE(dist_counts[2] == 3);
        REQUIRE(dist_counts[3] == 2);
        REQUIRE(dist_counts[4] == 1);
    }
    
    SECTION("Volume growth of lattice") {
        // 2D lattice 5x5
        auto graph = DynamicGraph::create_lattice_3d(5, 5, 1);
        auto vol = compute_volume_growth(graph);
        
        // vol[0] should be 1.0 (just the source)
        REQUIRE_THAT(vol[0], WithinAbs(1.0, 1e-10));
        // vol should be monotonically non-decreasing
        for (size_t r = 1; r < vol.size(); ++r) {
            REQUIRE(vol[r] >= vol[r - 1]);
        }
    }
    
    SECTION("Hausdorff dimension of 2D lattice") {
        auto graph = DynamicGraph::create_lattice_3d(8, 8, 1);
        double d_H = estimate_hausdorff_dimension(graph);
        // Should be close to 2.0 for a 2D lattice; boundary effects
        // reduce the estimate on small graphs
        REQUIRE(d_H > 1.0);
        REQUIRE(d_H < 3.0);
    }
}

TEST_CASE("Metric tensor", "[geometry][metric]") {
    SECTION("MDS embedding has correct dimensions") {
        DynamicGraph graph(10);
        for (int i = 0; i < 9; ++i) graph.add_edge(i, i + 1);
        
        auto coords = mds_embedding(graph, 3);
        REQUIRE(coords.rows() == 10);
        REQUIRE(coords.cols() == 3);
    }
    
    SECTION("Metric tensor is symmetric") {
        auto graph = DynamicGraph::create_lattice_3d(5, 5, 1);
        auto G = compute_metric_tensor(graph, 2);
        
        REQUIRE(G.rows() == 2);
        REQUIRE(G.cols() == 2);
        REQUIRE_THAT(G(0, 1), WithinAbs(G(1, 0), 1e-10));
    }
    
    SECTION("Metric tensor eigenvalues are non-negative") {
        auto graph = DynamicGraph::create_lattice_3d(5, 5, 1);
        auto G = compute_metric_tensor(graph, 2);
        
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(G);
        for (int i = 0; i < 2; ++i) {
            REQUIRE(es.eigenvalues()(i) >= -1e-10);
        }
    }
    
    SECTION("Scalar curvature of lattice is small") {
        auto graph = DynamicGraph::create_lattice_3d(6, 6, 1);
        double R = estimate_scalar_curvature(graph, 2);
        // 2D lattice is flat, curvature proxy should be small
        REQUIRE(R < 1.0);
    }
}
