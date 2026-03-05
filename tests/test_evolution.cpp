#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include "automaton/evolution.hpp"
#include "automaton/rule_parametric.hpp"

using namespace discretum;
using Catch::Matchers::WithinAbs;

TEST_CASE("ParametricRule", "[evolution][rule]") {
    SECTION("Default rule preserves state with zero params") {
        // All theta=0 → sigmoid(0)=0.5 for topo, state stays (0.5 bias)
        ParametricRule rule;
        DynamicGraph graph(10);
        for (int i = 0; i < 9; ++i) graph.add_edge(i, i + 1);
        for (uint32_t i = 0; i < 10; ++i) graph.set_state(i, i % 3);
        
        // State update only — with zero params and stability bias,
        // states should not change much
        std::vector<uint8_t> before(10);
        for (uint32_t i = 0; i < 10; ++i) before[i] = graph.get_state(i);
        
        rule.apply_state_update(graph);
        
        // At least some states should remain the same due to stability bias
        int same = 0;
        for (uint32_t i = 0; i < 10; ++i)
            if (graph.get_state(i) == before[i]) same++;
        REQUIRE(same >= 5);
    }
    
    SECTION("Param count") {
        REQUIRE(ParametricRule::num_params() == 14);  // 3*3 + 5
    }
    
    SECTION("Get/set params roundtrip") {
        std::vector<float> params(14);
        for (int i = 0; i < 14; ++i) params[i] = static_cast<float>(i) * 0.1f;
        
        ParametricRule rule(params);
        auto got = rule.get_params();
        
        REQUIRE(got.size() == 14);
        for (int i = 0; i < 14; ++i) {
            REQUIRE_THAT(got[i], WithinAbs(params[i], 1e-6f));
        }
    }
    
    SECTION("Topo update with strong add bias grows edges") {
        std::vector<float> params(14, 0.0f);
        params[9] = 5.0f;   // p_add → sigmoid(5) ≈ 0.993
        params[10] = -5.0f;  // p_remove → sigmoid(-5) ≈ 0.007
        params[11] = -5.0f;  // p_rewire → low
        params[12] = -5.0f;  // p_split → low
        params[13] = -5.0f;  // p_merge → low
        
        ParametricRule rule(params);
        DynamicGraph graph(20);
        for (int i = 0; i < 19; ++i) graph.add_edge(i, i + 1);
        
        uint32_t edges_before = graph.num_edges();
        PCG32 rng(42);
        rule.apply_topo_update(graph, rng);
        
        REQUIRE(graph.num_edges() > edges_before);
    }
    
    SECTION("Topo update with strong remove bias shrinks edges") {
        std::vector<float> params(14, 0.0f);
        params[9] = -5.0f;   // p_add → low
        params[10] = 5.0f;   // p_remove → high
        params[11] = -5.0f;
        params[12] = -5.0f;
        params[13] = -5.0f;
        
        ParametricRule rule(params);
        // Create a graph with plenty of edges to remove
        DynamicGraph graph(20);
        for (int i = 0; i < 20; ++i)
            for (int j = i + 1; j < 20; ++j)
                if ((i + j) % 3 == 0) graph.add_edge(i, j);
        
        uint32_t edges_before = graph.num_edges();
        PCG32 rng(42);
        rule.apply_topo_update(graph, rng);
        
        REQUIRE(graph.num_edges() < edges_before);
    }
}

TEST_CASE("Evolution engine", "[evolution]") {
    SECTION("Default evolution runs without crash") {
        DynamicGraph graph(20);
        for (int i = 0; i < 19; ++i) graph.add_edge(i, i + 1);
        
        ParametricRule rule;
        EvolutionConfig config;
        config.num_steps = 10;
        config.seed = 42;
        
        Evolution evo(std::move(graph), std::move(rule), config);
        auto result = evo.run();
        
        REQUIRE(result.steps_completed == 10);
        REQUIRE(result.node_count_history.size() == 11);  // initial + 10 steps
        REQUIRE(result.edge_count_history.size() == 11);
        REQUIRE(result.final_num_nodes > 0);
    }
    
    SECTION("Step-by-step matches batch run") {
        DynamicGraph graph1(10);
        for (int i = 0; i < 9; ++i) graph1.add_edge(i, i + 1);
        DynamicGraph graph2 = graph1;
        
        ParametricRule rule;
        
        // Run 5 steps via step()
        Evolution evo1(std::move(graph1), rule, {5, 0, 42});
        for (int i = 0; i < 5; ++i) evo1.step();
        
        // Run 5 steps via run()
        Evolution evo2(std::move(graph2), rule, {5, 0, 42});
        evo2.run();
        
        REQUIRE(evo1.get_graph().num_nodes() == evo2.get_graph().num_nodes());
        REQUIRE(evo1.get_graph().num_edges() == evo2.get_graph().num_edges());
    }
    
    SECTION("Graph with snapshots") {
        DynamicGraph graph(30);
        for (int i = 0; i < 29; ++i) graph.add_edge(i, i + 1);
        
        ParametricRule rule;
        EvolutionConfig config;
        config.num_steps = 20;
        config.snapshot_interval = 5;
        config.seed = 123;
        
        Evolution evo(std::move(graph), std::move(rule), config);
        auto result = evo.run();
        
        REQUIRE(result.steps_completed == 20);
        REQUIRE(result.final_num_nodes > 0);
    }
}
