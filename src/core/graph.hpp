#pragma once

#include <vector>
#include <span>
#include <cstdint>
#include <string>
#include <optional>
#include <limits>
#include <unordered_set>

namespace discretum {

/**
 * @brief Node structure for the dynamic graph
 * 
 * Mathematical representation: Each node i ∈ V_t has state s_i(t) ∈ Σ = {0, ..., q-1}
 * and degree k_i(t) = |N(i)| where N(i) is the neighborhood
 * 
 * Memory layout optimized for cache efficiency
 */
struct Node {
    uint32_t id;          ///< Unique node identifier
    uint8_t state;        ///< State s_i ∈ {0, ..., q-1}
    uint16_t degree;      ///< Current degree k_i
    uint32_t adj_offset;  ///< Offset into global adjacency array
};

/**
 * @brief Parameters for node splitting operation
 */
struct SplitParams {
    float redistribution_ratio = 0.5f;  ///< Fraction of edges going to first new node
    bool preserve_state = true;         ///< Whether both new nodes inherit parent state
};

/**
 * @brief Dynamic graph with support for topological operations
 * 
 * Implementation: Modified Compressed Sparse Row (CSR) format supporting
 * efficient insertion/deletion while maintaining memory contiguity
 * 
 * References:
 * - Leskovec & Sosič (2016) "SNAP: A General-Purpose Network Analysis Library"
 * - Shun & Blelloch (2013) "Ligra: A Lightweight Graph Processing Framework"
 * 
 * Complexity:
 * - Neighbor access: O(1)
 * - Edge addition: O(degree) amortized
 * - BFS shortest path: O(V + E)
 * - Connectivity check: O(V + E)
 */
class DynamicGraph {
public:
    // Constructors
    DynamicGraph() = default;
    explicit DynamicGraph(uint32_t num_nodes);
    
    // Factory methods for standard graph types
    static DynamicGraph create_lattice_3d(uint32_t nx, uint32_t ny, uint32_t nz);
    static DynamicGraph create_lattice_4d(uint32_t nx, uint32_t ny, uint32_t nz, uint32_t nw);
    static DynamicGraph create_random_regular(uint32_t num_nodes, uint32_t degree, uint64_t seed);
    static DynamicGraph create_erdos_renyi(uint32_t num_nodes, float edge_prob, uint64_t seed);
    
    // Node operations
    uint32_t add_node(uint8_t initial_state = 0);
    void remove_node(uint32_t id);
    void set_state(uint32_t id, uint8_t state);
    uint8_t get_state(uint32_t id) const;
    
    // Edge operations
    bool add_edge(uint32_t u, uint32_t v);
    bool remove_edge(uint32_t u, uint32_t v);
    bool has_edge(uint32_t u, uint32_t v) const;
    
    // Topological operations
    std::pair<uint32_t, uint32_t> split_node(uint32_t id, const SplitParams& params);
    uint32_t merge_nodes(uint32_t id1, uint32_t id2);
    bool rewire_edge(uint32_t from, uint32_t old_to, uint32_t new_to);
    
    // Query operations
    uint32_t num_nodes() const { return active_nodes.size(); }
    uint32_t num_edges() const;
    std::span<const uint32_t> neighbors(uint32_t id) const;
    uint32_t degree(uint32_t id) const;
    
    // Graph algorithms
    uint32_t shortest_path(uint32_t src, uint32_t dst) const;
    std::vector<uint32_t> shortest_path_with_path(uint32_t src, uint32_t dst) const;
    bool is_connected() const;
    std::vector<std::vector<uint32_t>> connected_components() const;
    
    // Validation
    bool check_invariants() const;
    
    // Serialization
    void save(const std::string& path) const;
    static DynamicGraph load(const std::string& path);
    
    // Direct access for performance-critical operations
    const std::vector<Node>& get_nodes() const { return nodes; }
    const std::vector<uint32_t>& get_adjacency() const { return adjacency; }
    const std::unordered_set<uint32_t>& get_active_nodes() const { return active_nodes; }

private:
    // Core data structures
    std::vector<Node> nodes;                    ///< Node array
    std::vector<uint32_t> adjacency;            ///< Flat adjacency list
    std::unordered_set<uint32_t> active_nodes;  ///< Set of active node IDs
    std::vector<uint32_t> free_ids;             ///< Pool of recyclable IDs
    
    // Internal helpers
    void compact_adjacency();
    uint32_t allocate_adjacency_space(uint32_t required_degree);
    void deallocate_adjacency_space(uint32_t offset, uint32_t size);
    bool is_valid_node(uint32_t id) const;
    
    // Constants
public:
    static constexpr uint32_t INVALID_ID = std::numeric_limits<uint32_t>::max();

private:
    static constexpr uint32_t MAX_DEGREE = 12;  // As specified in requirements
};

} // namespace discretum
