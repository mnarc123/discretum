#include "core/graph.hpp"
#include <algorithm>
#include <queue>
#include <random>
#include <fstream>
#include <stdexcept>
#include <numeric>
#include <unordered_map>
#include <set>

namespace discretum {

DynamicGraph::DynamicGraph(uint32_t num_nodes) {
    nodes.reserve(num_nodes);
    for (uint32_t i = 0; i < num_nodes; ++i) {
        nodes.push_back({i, 0, 0, INVALID_ID});
        active_nodes.insert(i);
    }
}

DynamicGraph DynamicGraph::create_lattice_3d(uint32_t nx, uint32_t ny, uint32_t nz) {
    DynamicGraph graph(nx * ny * nz);
    
    auto index_3d = [=](uint32_t x, uint32_t y, uint32_t z) {
        return x + nx * (y + ny * z);
    };
    
    // Create edges for 3D cubic lattice
    for (uint32_t z = 0; z < nz; ++z) {
        for (uint32_t y = 0; y < ny; ++y) {
            for (uint32_t x = 0; x < nx; ++x) {
                uint32_t current = index_3d(x, y, z);
                
                // Connect to neighbors in +x, +y, +z directions (undirected graph)
                if (x + 1 < nx) graph.add_edge(current, index_3d(x + 1, y, z));
                if (y + 1 < ny) graph.add_edge(current, index_3d(x, y + 1, z));
                if (z + 1 < nz) graph.add_edge(current, index_3d(x, y, z + 1));
            }
        }
    }
    
    return graph;
}

DynamicGraph DynamicGraph::create_lattice_4d(uint32_t nx, uint32_t ny, uint32_t nz, uint32_t nw) {
    DynamicGraph graph(nx * ny * nz * nw);
    
    auto index_4d = [=](uint32_t x, uint32_t y, uint32_t z, uint32_t w) {
        return x + nx * (y + ny * (z + nz * w));
    };
    
    // Create edges for 4D hypercubic lattice
    for (uint32_t w = 0; w < nw; ++w) {
        for (uint32_t z = 0; z < nz; ++z) {
            for (uint32_t y = 0; y < ny; ++y) {
                for (uint32_t x = 0; x < nx; ++x) {
                    uint32_t current = index_4d(x, y, z, w);
                    
                    // Connect to neighbors in +x, +y, +z, +w directions
                    if (x + 1 < nx) graph.add_edge(current, index_4d(x + 1, y, z, w));
                    if (y + 1 < ny) graph.add_edge(current, index_4d(x, y + 1, z, w));
                    if (z + 1 < nz) graph.add_edge(current, index_4d(x, y, z + 1, w));
                    if (w + 1 < nw) graph.add_edge(current, index_4d(x, y, z, w + 1));
                }
            }
        }
    }
    
    return graph;
}

DynamicGraph DynamicGraph::create_random_regular(uint32_t num_nodes, uint32_t degree, uint64_t seed) {
    if (degree >= num_nodes || degree % 2 != 0 || num_nodes * degree % 2 != 0) {
        throw std::invalid_argument("Invalid parameters for random regular graph");
    }
    
    DynamicGraph graph(num_nodes);
    std::mt19937_64 rng(seed);
    
    // Configuration model approach
    std::vector<uint32_t> stubs;
    for (uint32_t i = 0; i < num_nodes; ++i) {
        for (uint32_t j = 0; j < degree; ++j) {
            stubs.push_back(i);
        }
    }
    
    // Shuffle and pair stubs
    std::shuffle(stubs.begin(), stubs.end(), rng);
    
    for (size_t i = 0; i < stubs.size(); i += 2) {
        if (stubs[i] != stubs[i + 1]) {  // Avoid self-loops
            graph.add_edge(stubs[i], stubs[i + 1]);
        }
    }
    
    return graph;
}

DynamicGraph DynamicGraph::create_erdos_renyi(uint32_t num_nodes, float edge_prob, uint64_t seed) {
    DynamicGraph graph(num_nodes);
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    
    for (uint32_t i = 0; i < num_nodes; ++i) {
        for (uint32_t j = i + 1; j < num_nodes; ++j) {
            if (dist(rng) < edge_prob) {
                graph.add_edge(i, j);
            }
        }
    }
    
    return graph;
}

uint32_t DynamicGraph::add_node(uint8_t initial_state) {
    uint32_t new_id;
    
    if (!free_ids.empty()) {
        new_id = free_ids.back();
        free_ids.pop_back();
        nodes[new_id] = {new_id, initial_state, 0, INVALID_ID};
    } else {
        new_id = nodes.size();
        nodes.push_back({new_id, initial_state, 0, INVALID_ID});
    }
    
    active_nodes.insert(new_id);
    return new_id;
}

void DynamicGraph::remove_node(uint32_t id) {
    if (!is_valid_node(id)) return;
    
    // Remove all edges incident to this node
    std::vector<uint32_t> neighbors_copy(neighbors(id).begin(), neighbors(id).end());
    for (uint32_t neighbor : neighbors_copy) {
        remove_edge(id, neighbor);
    }
    
    // Mark node as inactive
    active_nodes.erase(id);
    free_ids.push_back(id);
    nodes[id] = {id, 0, 0, INVALID_ID};
}

void DynamicGraph::set_state(uint32_t id, uint8_t state) {
    if (is_valid_node(id)) {
        nodes[id].state = state;
    }
}

uint8_t DynamicGraph::get_state(uint32_t id) const {
    if (is_valid_node(id)) {
        return nodes[id].state;
    }
    return 0;
}

bool DynamicGraph::add_edge(uint32_t u, uint32_t v) {
    if (!is_valid_node(u) || !is_valid_node(v) || u == v) return false;
    if (has_edge(u, v)) return true;  // Edge already exists
    
    // Check degree constraints
    if (nodes[u].degree >= MAX_DEGREE || nodes[v].degree >= MAX_DEGREE) {
        return false;
    }
    
    // Helper: ensure a node has room for one more neighbor by relocating
    // its adjacency list to a fresh block at the end of the array.
    auto ensure_space = [&](uint32_t node) {
        uint32_t deg = nodes[node].degree;
        uint32_t old_offset = nodes[node].adj_offset;
        
        // Allocate a fresh block with room for (degree + 1) entries
        uint32_t new_offset = allocate_adjacency_space(deg + 1);
        
        // Copy existing neighbors to the new location
        if (deg > 0 && old_offset != INVALID_ID) {
            std::copy(adjacency.begin() + old_offset,
                     adjacency.begin() + old_offset + deg,
                     adjacency.begin() + new_offset);
            deallocate_adjacency_space(old_offset, deg);
        }
        
        nodes[node].adj_offset = new_offset;
    };
    
    // Relocate both adjacency lists to fresh space to avoid any overlap
    ensure_space(u);
    ensure_space(v);
    
    // Add the edge
    adjacency[nodes[u].adj_offset + nodes[u].degree] = v;
    adjacency[nodes[v].adj_offset + nodes[v].degree] = u;
    nodes[u].degree++;
    nodes[v].degree++;
    
    return true;
}

bool DynamicGraph::remove_edge(uint32_t u, uint32_t v) {
    if (!is_valid_node(u) || !is_valid_node(v)) return false;
    
    // Find and remove v from u's adjacency list
    auto u_neighbors = neighbors(u);
    auto it_u = std::find(u_neighbors.begin(), u_neighbors.end(), v);
    if (it_u == u_neighbors.end()) return false;
    
    uint32_t u_idx = std::distance(u_neighbors.begin(), it_u);
    
    // Find and remove u from v's adjacency list
    auto v_neighbors = neighbors(v);
    auto it_v = std::find(v_neighbors.begin(), v_neighbors.end(), u);
    if (it_v == v_neighbors.end()) return false;
    
    uint32_t v_idx = std::distance(v_neighbors.begin(), it_v);
    
    // Remove by swapping with last element
    if (nodes[u].degree > 1) {
        adjacency[nodes[u].adj_offset + u_idx] = adjacency[nodes[u].adj_offset + nodes[u].degree - 1];
    }
    nodes[u].degree--;
    
    if (nodes[v].degree > 1) {
        adjacency[nodes[v].adj_offset + v_idx] = adjacency[nodes[v].adj_offset + nodes[v].degree - 1];
    }
    nodes[v].degree--;
    
    return true;
}

bool DynamicGraph::has_edge(uint32_t u, uint32_t v) const {
    if (!is_valid_node(u) || !is_valid_node(v)) return false;
    
    // Check the node with smaller degree for efficiency
    if (nodes[u].degree > nodes[v].degree) std::swap(u, v);
    
    auto u_neighbors = neighbors(u);
    return std::find(u_neighbors.begin(), u_neighbors.end(), v) != u_neighbors.end();
}

std::pair<uint32_t, uint32_t> DynamicGraph::split_node(uint32_t id, const SplitParams& params) {
    if (!is_valid_node(id)) return {INVALID_ID, INVALID_ID};
    
    // Get current node info
    uint8_t state = nodes[id].state;
    std::vector<uint32_t> old_neighbors(neighbors(id).begin(), neighbors(id).end());
    
    // Remove the original node
    remove_node(id);
    
    // Create two new nodes
    uint32_t id1 = add_node(params.preserve_state ? state : 0);
    uint32_t id2 = add_node(params.preserve_state ? state : 0);
    
    // Redistribute edges
    std::mt19937 rng(id);  // Deterministic based on node ID
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    
    for (uint32_t neighbor : old_neighbors) {
        if (dist(rng) < params.redistribution_ratio) {
            add_edge(id1, neighbor);
        } else {
            add_edge(id2, neighbor);
        }
    }
    
    // Connect the two new nodes
    add_edge(id1, id2);
    
    return {id1, id2};
}

uint32_t DynamicGraph::merge_nodes(uint32_t id1, uint32_t id2) {
    if (!is_valid_node(id1) || !is_valid_node(id2) || !has_edge(id1, id2)) {
        return INVALID_ID;
    }
    
    // Collect all neighbors of both nodes
    std::unordered_set<uint32_t> all_neighbors;
    for (uint32_t n : neighbors(id1)) {
        if (n != id2) all_neighbors.insert(n);
    }
    for (uint32_t n : neighbors(id2)) {
        if (n != id1) all_neighbors.insert(n);
    }
    
    // Check if merged node would exceed degree limit
    if (all_neighbors.size() > MAX_DEGREE) {
        return INVALID_ID;
    }
    
    // Create new node with average state
    uint8_t new_state = (nodes[id1].state + nodes[id2].state) / 2;
    
    // Remove both nodes
    remove_node(id1);
    remove_node(id2);
    
    // Create merged node
    uint32_t new_id = add_node(new_state);
    
    // Connect to all neighbors
    for (uint32_t neighbor : all_neighbors) {
        add_edge(new_id, neighbor);
    }
    
    return new_id;
}

bool DynamicGraph::rewire_edge(uint32_t from, uint32_t old_to, uint32_t new_to) {
    if (!has_edge(from, old_to) || !is_valid_node(new_to) || from == new_to) {
        return false;
    }
    
    if (has_edge(from, new_to)) {
        // Edge already exists, just remove the old one
        remove_edge(from, old_to);
        return true;
    }
    
    if (nodes[from].degree >= MAX_DEGREE || nodes[new_to].degree >= MAX_DEGREE) {
        return false;
    }
    
    remove_edge(from, old_to);
    add_edge(from, new_to);
    
    return true;
}

uint32_t DynamicGraph::num_edges() const {
    uint32_t total = 0;
    for (const auto& node : nodes) {
        if (active_nodes.count(node.id) > 0) {
            total += node.degree;
        }
    }
    return total / 2;  // Each edge counted twice
}

std::span<const uint32_t> DynamicGraph::neighbors(uint32_t id) const {
    if (!is_valid_node(id) || nodes[id].adj_offset == INVALID_ID) {
        return {};
    }
    return std::span<const uint32_t>(&adjacency[nodes[id].adj_offset], nodes[id].degree);
}

uint32_t DynamicGraph::degree(uint32_t id) const {
    if (!is_valid_node(id)) return 0;
    return nodes[id].degree;
}

uint32_t DynamicGraph::shortest_path(uint32_t src, uint32_t dst) const {
    if (!is_valid_node(src) || !is_valid_node(dst)) return INVALID_ID;
    if (src == dst) return 0;
    
    std::queue<uint32_t> q;
    std::unordered_map<uint32_t, uint32_t> dist;
    
    q.push(src);
    dist[src] = 0;
    
    while (!q.empty()) {
        uint32_t current = q.front();
        q.pop();
        
        for (uint32_t neighbor : neighbors(current)) {
            if (dist.find(neighbor) == dist.end()) {
                dist[neighbor] = dist[current] + 1;
                if (neighbor == dst) {
                    return dist[neighbor];
                }
                q.push(neighbor);
            }
        }
    }
    
    return INVALID_ID;  // No path found
}

std::vector<uint32_t> DynamicGraph::shortest_path_with_path(uint32_t src, uint32_t dst) const {
    if (!is_valid_node(src) || !is_valid_node(dst)) return {};
    if (src == dst) return {src};
    
    std::queue<uint32_t> q;
    std::unordered_map<uint32_t, uint32_t> parent;
    
    q.push(src);
    parent[src] = src;
    
    while (!q.empty()) {
        uint32_t current = q.front();
        q.pop();
        
        for (uint32_t neighbor : neighbors(current)) {
            if (parent.find(neighbor) == parent.end()) {
                parent[neighbor] = current;
                if (neighbor == dst) {
                    // Reconstruct path
                    std::vector<uint32_t> path;
                    uint32_t node = dst;
                    while (node != src) {
                        path.push_back(node);
                        node = parent[node];
                    }
                    path.push_back(src);
                    std::reverse(path.begin(), path.end());
                    return path;
                }
                q.push(neighbor);
            }
        }
    }
    
    return {};  // No path found
}

bool DynamicGraph::is_connected() const {
    if (active_nodes.empty()) return true;
    
    // BFS from arbitrary node
    uint32_t start = *active_nodes.begin();
    std::unordered_set<uint32_t> visited;
    std::queue<uint32_t> q;
    
    q.push(start);
    visited.insert(start);
    
    while (!q.empty()) {
        uint32_t current = q.front();
        q.pop();
        
        for (uint32_t neighbor : neighbors(current)) {
            if (visited.find(neighbor) == visited.end()) {
                visited.insert(neighbor);
                q.push(neighbor);
            }
        }
    }
    
    return visited.size() == active_nodes.size();
}

std::vector<std::vector<uint32_t>> DynamicGraph::connected_components() const {
    std::vector<std::vector<uint32_t>> components;
    std::unordered_set<uint32_t> visited;
    
    for (uint32_t node_id : active_nodes) {
        if (visited.find(node_id) == visited.end()) {
            std::vector<uint32_t> component;
            std::queue<uint32_t> q;
            
            q.push(node_id);
            visited.insert(node_id);
            
            while (!q.empty()) {
                uint32_t current = q.front();
                q.pop();
                component.push_back(current);
                
                for (uint32_t neighbor : neighbors(current)) {
                    if (visited.find(neighbor) == visited.end()) {
                        visited.insert(neighbor);
                        q.push(neighbor);
                    }
                }
            }
            
            components.push_back(std::move(component));
        }
    }
    
    return components;
}

bool DynamicGraph::check_invariants() const {
    // Check 1: All active nodes have valid IDs
    for (uint32_t id : active_nodes) {
        if (id >= nodes.size()) return false;
    }
    
    // Check 2: All edges are symmetric
    for (uint32_t id : active_nodes) {
        for (uint32_t neighbor : neighbors(id)) {
            if (!has_edge(neighbor, id)) return false;
        }
    }
    
    // Check 3: No self-loops
    for (uint32_t id : active_nodes) {
        for (uint32_t neighbor : neighbors(id)) {
            if (neighbor == id) return false;
        }
    }
    
    // Check 4: Degrees match adjacency list sizes
    for (uint32_t id : active_nodes) {
        if (nodes[id].degree != neighbors(id).size()) return false;
    }
    
    // Check 5: No node exceeds MAX_DEGREE
    for (uint32_t id : active_nodes) {
        if (nodes[id].degree > MAX_DEGREE) return false;
    }
    
    return true;
}

void DynamicGraph::save(const std::string& path) const {
    std::ofstream file(path, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Cannot open file for writing: " + path);
    }
    
    // Write header
    uint32_t num_active = active_nodes.size();
    uint32_t num_edges_total = num_edges();
    file.write(reinterpret_cast<const char*>(&num_active), sizeof(num_active));
    file.write(reinterpret_cast<const char*>(&num_edges_total), sizeof(num_edges_total));
    
    // Write nodes
    for (uint32_t id : active_nodes) {
        file.write(reinterpret_cast<const char*>(&id), sizeof(id));
        file.write(reinterpret_cast<const char*>(&nodes[id].state), sizeof(nodes[id].state));
        file.write(reinterpret_cast<const char*>(&nodes[id].degree), sizeof(nodes[id].degree));
        
        // Write adjacency list
        for (uint32_t neighbor : neighbors(id)) {
            file.write(reinterpret_cast<const char*>(&neighbor), sizeof(neighbor));
        }
    }
}

DynamicGraph DynamicGraph::load(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Cannot open file for reading: " + path);
    }
    
    // Read header
    uint32_t num_active, num_edges_total;
    file.read(reinterpret_cast<char*>(&num_active), sizeof(num_active));
    file.read(reinterpret_cast<char*>(&num_edges_total), sizeof(num_edges_total));
    
    // Read all node data first
    struct NodeData {
        uint32_t id;
        uint8_t state;
        std::vector<uint32_t> neighbors;
    };
    
    std::vector<NodeData> node_data(num_active);
    uint32_t max_id = 0;
    
    for (uint32_t i = 0; i < num_active; ++i) {
        uint16_t degree;
        file.read(reinterpret_cast<char*>(&node_data[i].id), sizeof(uint32_t));
        file.read(reinterpret_cast<char*>(&node_data[i].state), sizeof(uint8_t));
        file.read(reinterpret_cast<char*>(&degree), sizeof(uint16_t));
        
        max_id = std::max(max_id, node_data[i].id);
        node_data[i].neighbors.resize(degree);
        for (uint16_t j = 0; j < degree; ++j) {
            file.read(reinterpret_cast<char*>(&node_data[i].neighbors[j]), sizeof(uint32_t));
        }
    }
    
    // Create graph with enough node slots, then activate the ones we need
    DynamicGraph graph(max_id + 1);
    
    // Set states for active nodes
    // graph(max_id+1) already created nodes 0..max_id as active
    // Set each node's state
    for (const auto& nd : node_data) {
        graph.set_state(nd.id, nd.state);
    }
    
    // Add edges (only once per pair)
    std::set<std::pair<uint32_t, uint32_t>> added_edges;
    for (const auto& nd : node_data) {
        for (uint32_t neighbor : nd.neighbors) {
            auto edge = std::minmax(nd.id, neighbor);
            if (added_edges.find(edge) == added_edges.end()) {
                graph.add_edge(nd.id, neighbor);
                added_edges.insert(edge);
            }
        }
    }
    
    return graph;
}

bool DynamicGraph::is_valid_node(uint32_t id) const {
    return id < nodes.size() && active_nodes.count(id) > 0;
}

uint32_t DynamicGraph::allocate_adjacency_space(uint32_t required_degree) {
    uint32_t offset = adjacency.size();
    adjacency.resize(adjacency.size() + required_degree, INVALID_ID);
    return offset;
}

void DynamicGraph::deallocate_adjacency_space(uint32_t offset, uint32_t size) {
    // Mark as free (could implement proper memory management later)
    for (uint32_t i = 0; i < size; ++i) {
        adjacency[offset + i] = INVALID_ID;
    }
}

void DynamicGraph::compact_adjacency() {
    // Reorganize the adjacency array to remove gaps left by deallocated slots.
    // Build a new contiguous array and update each node's adj_offset.
    std::vector<uint32_t> new_adj;
    new_adj.reserve(adjacency.size());
    
    for (uint32_t id : active_nodes) {
        auto& node = nodes[id];
        uint32_t new_offset = static_cast<uint32_t>(new_adj.size());
        // Copy only valid neighbor entries
        for (uint16_t j = 0; j < node.degree; ++j) {
            new_adj.push_back(adjacency[node.adj_offset + j]);
        }
        node.adj_offset = new_offset;
    }
    
    adjacency = std::move(new_adj);
}

} // namespace discretum
