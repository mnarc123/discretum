#include <hip/hip_runtime.h>
#include <cstdint>
#include <cstring>
#include <vector>
#include <stdexcept>
#include <algorithm>
#include <cmath>

namespace discretum {

// Maximum support size per edge (2 * max_degree + 2)
constexpr int GPU_MAX_SUPPORT = 32;
// Maximum BFS queue size per thread
constexpr int GPU_MAX_BFS_QUEUE = 256;

// -------------------------------------------------------------------
// Device helper: BFS shortest-path distances from a source node,
// restricted to a given support set.  Returns distances in dist_out[]
// indexed by support-set position.
// -------------------------------------------------------------------
__device__ void bfs_distances_device(
    const uint32_t* __restrict__ csr_offsets,
    const uint32_t* __restrict__ csr_neighbors,
    uint32_t /*num_nodes*/,
    uint32_t source,
    const uint32_t* __restrict__ support, int support_size,
    float* __restrict__ dist_out)
{
    // BFS from source, record distance to each support node
    // Use shared-memory-sized queue local to this thread
    // Initialize output distances to large value
    for (int i = 0; i < support_size; ++i)
        dist_out[i] = 1e6f;
    
    int found = 0;
    
    // Track visited via the queue itself — since we process level by level
    // we mark distance via a parallel distance array. Use dist_out for support nodes
    // and a separate mechanism for non-support nodes.
    // For simplicity with bounded BFS: track visited in the queue.
    
    // Since support is small, max BFS depth is also small.
    // Use a simple visited bitmap approach — but we can't allocate per-node arrays on GPU.
    // Instead, do level-by-level BFS with max depth = support_size (since max distance
    // between support nodes is bounded by graph diameter).
    
    // Actually for our small problem, let's just use a compact approach:
    // We know support_size <= GPU_MAX_SUPPORT, and BFS depth won't exceed ~10.
    // Store (node, dist) pairs in the queue.
    
    struct QEntry { uint32_t node; uint32_t dist; };
    QEntry bfs_queue[GPU_MAX_BFS_QUEUE];
    int bfs_front = 0, bfs_back = 0;
    
    bfs_queue[bfs_back++] = {source, 0};
    
    // Check if source is in support
    for (int i = 0; i < support_size; ++i) {
        if (support[i] == source) {
            dist_out[i] = 0.0f;
            found++;
            break;
        }
    }
    
    // Simple visited tracking: since we can't have per-node visited array,
    // use the fact that BFS on small local neighborhoods won't revisit.
    // We'll limit BFS depth and use a visited set.
    // For GPU: use a small hash set. With max depth ~6 and avg degree ~6,
    // we visit ~6^6 nodes worst case. Too many for a flat array.
    
    // Better approach: since support is small (≤32), we only need distances
    // between support nodes. Run BFS only until all support nodes are found.
    // Track visited nodes in a small buffer.
    uint32_t visited[GPU_MAX_BFS_QUEUE];
    int num_visited = 0;
    visited[num_visited++] = source;
    
    while (bfs_front < bfs_back && found < support_size) {
        QEntry cur = bfs_queue[bfs_front++];
        
        if (cur.dist > 20) break;  // Safety: don't BFS too deep
        
        uint32_t start = csr_offsets[cur.node];
        uint32_t end = csr_offsets[cur.node + 1];
        
        for (uint32_t e = start; e < end; ++e) {
            uint32_t nbr = csr_neighbors[e];
            
            // Check if already visited (linear scan of visited set)
            bool already_visited = false;
            for (int v = 0; v < num_visited; ++v) {
                if (visited[v] == nbr) {
                    already_visited = true;
                    break;
                }
            }
            if (already_visited) continue;
            
            if (num_visited < GPU_MAX_BFS_QUEUE)
                visited[num_visited++] = nbr;
            
            // Check if nbr is in support
            for (int i = 0; i < support_size; ++i) {
                if (support[i] == nbr) {
                    dist_out[i] = static_cast<float>(cur.dist + 1);
                    found++;
                    break;
                }
            }
            
            if (bfs_back < GPU_MAX_BFS_QUEUE) {
                bfs_queue[bfs_back++] = {nbr, cur.dist + 1};
            }
        }
    }
}

// -------------------------------------------------------------------
// Device helper: Solve small transportation problem for W₁ distance.
// Uses successive shortest paths (Bellman-Ford) min-cost flow.
// Exact solver for support sizes ≤ GPU_MAX_SUPPORT.
//
// Network: S → supply_i → demand_j → T
// Integer-scaled capacities for exact arithmetic.
// -------------------------------------------------------------------

// Max network nodes: S + n_supply + n_demand + T = 2*GPU_MAX_SUPPORT + 2
constexpr int GPU_MAX_NET_NODES = 2 * GPU_MAX_SUPPORT + 2;
// Max edges: S→supply(n) + supply→demand(n²) + demand→T(n), each with reverse
// = 2*(n + n² + n) = 2*n² + 4*n. For n=32: 2112. With reverses: 4224.
constexpr int GPU_MAX_EDGES_PER_NODE = 40;  // max adjacency per node

struct DevEdge {
    int to;
    int rev;       // index of reverse edge in adj list of 'to'
    int cap;       // remaining capacity (scaled integer)
    float cost;
};

__device__ float solve_transport_device(
    const float* __restrict__ cost_matrix,  // n x n row-major
    const float* __restrict__ supply,
    const float* __restrict__ demand,
    int n)
{
    if (n == 0) return 0.0f;
    
    // Scale probabilities to integers (sum ~ SCALE)
    constexpr int SCALE = 10000;
    int isupply[GPU_MAX_SUPPORT], idemand[GPU_MAX_SUPPORT];
    int total_supply = 0, total_demand = 0;
    for (int i = 0; i < n; ++i) {
        isupply[i] = static_cast<int>(supply[i] * SCALE + 0.5f);
        total_supply += isupply[i];
        idemand[i] = static_cast<int>(demand[i] * SCALE + 0.5f);
        total_demand += idemand[i];
    }
    int total_flow_target = min(total_supply, total_demand);
    if (total_flow_target == 0) return 0.0f;
    
    // Network: S=0, supply nodes 1..n, demand nodes n+1..2n, T=2n+1
    const int S = 0;
    const int T = 2 * n + 1;
    const int V = T + 1;
    
    // Adjacency lists stored in flat arrays per node
    DevEdge adj[GPU_MAX_NET_NODES][GPU_MAX_EDGES_PER_NODE];
    int adj_size[GPU_MAX_NET_NODES];
    for (int i = 0; i < V; ++i) adj_size[i] = 0;
    
    // Helper to add forward+reverse edge pair
    auto add_edge = [&](int from, int to, int cap, float c) {
        int fi = adj_size[from];
        int ti = adj_size[to];
        if (fi >= GPU_MAX_EDGES_PER_NODE || ti >= GPU_MAX_EDGES_PER_NODE) return;
        adj[from][fi] = {to, ti, cap, c};
        adj[to][ti] = {from, fi, 0, -c};
        adj_size[from]++;
        adj_size[to]++;
    };
    
    // S → supply nodes
    for (int i = 0; i < n; ++i) {
        if (isupply[i] > 0)
            add_edge(S, i + 1, isupply[i], 0.0f);
    }
    // supply → demand
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            add_edge(i + 1, n + 1 + j, total_flow_target, cost_matrix[i * n + j]);
        }
    }
    // demand → T
    for (int j = 0; j < n; ++j) {
        if (idemand[j] > 0)
            add_edge(n + 1 + j, T, idemand[j], 0.0f);
    }
    
    // Successive shortest paths with SPFA (Bellman-Ford with queue)
    float total_cost = 0.0f;
    int flow_remaining = total_flow_target;
    
    // SPFA arrays
    float dist[GPU_MAX_NET_NODES];
    bool in_queue[GPU_MAX_NET_NODES];
    int prev_node[GPU_MAX_NET_NODES];
    int prev_edge[GPU_MAX_NET_NODES];
    int queue[GPU_MAX_NET_NODES * 4];  // circular queue
    
    while (flow_remaining > 0) {
        // Initialize SPFA
        for (int i = 0; i < V; ++i) {
            dist[i] = 1e18f;
            in_queue[i] = false;
            prev_node[i] = -1;
            prev_edge[i] = -1;
        }
        dist[S] = 0.0f;
        int q_front = 0, q_back = 0;
        int q_cap = GPU_MAX_NET_NODES * 4;
        queue[q_back % q_cap] = S;
        q_back++;
        in_queue[S] = true;
        
        while (q_front < q_back) {
            int u = queue[q_front % q_cap];
            q_front++;
            in_queue[u] = false;
            
            for (int idx = 0; idx < adj_size[u]; ++idx) {
                DevEdge& e = adj[u][idx];
                if (e.cap > 0 && dist[u] + e.cost < dist[e.to] - 1e-6f) {
                    dist[e.to] = dist[u] + e.cost;
                    prev_node[e.to] = u;
                    prev_edge[e.to] = idx;
                    if (!in_queue[e.to]) {
                        queue[q_back % q_cap] = e.to;
                        q_back++;
                        in_queue[e.to] = true;
                    }
                }
            }
        }
        
        if (dist[T] >= 1e17f) break;  // No augmenting path
        
        // Find bottleneck
        int path_flow = flow_remaining;
        for (int v = T; v != S; v = prev_node[v]) {
            DevEdge& e = adj[prev_node[v]][prev_edge[v]];
            path_flow = min(path_flow, e.cap);
        }
        
        // Augment
        for (int v = T; v != S; v = prev_node[v]) {
            DevEdge& e = adj[prev_node[v]][prev_edge[v]];
            e.cap -= path_flow;
            adj[v][e.rev].cap += path_flow;
        }
        
        total_cost += path_flow * dist[T];
        flow_remaining -= path_flow;
    }
    
    return total_cost / SCALE;
}

// -------------------------------------------------------------------
// Main kernel: one thread per edge computes Ollivier-Ricci curvature
// -------------------------------------------------------------------
__global__ void ollivier_ricci_kernel(
    const uint32_t* __restrict__ csr_offsets,    // [num_nodes + 1]
    const uint32_t* __restrict__ csr_neighbors,  // [num_edges_2x]
    uint32_t num_nodes,
    const uint32_t* __restrict__ edge_u,         // [num_edges]
    const uint32_t* __restrict__ edge_v,         // [num_edges]
    uint32_t num_edges,
    float alpha,
    float* __restrict__ curvatures)              // [num_edges] output
{
    uint32_t eid = blockIdx.x * blockDim.x + threadIdx.x;
    if (eid >= num_edges) return;
    
    uint32_t u = edge_u[eid];
    uint32_t v = edge_v[eid];
    
    // Build probability measures for u and v
    uint32_t deg_u = csr_offsets[u + 1] - csr_offsets[u];
    uint32_t deg_v = csr_offsets[v + 1] - csr_offsets[v];
    
    if (deg_u == 0 || deg_v == 0) {
        curvatures[eid] = 0.0f;
        return;
    }
    
    // Build support set (union of neighbors of u and v, plus u and v themselves)
    uint32_t support[GPU_MAX_SUPPORT];
    int support_size = 0;
    
    auto add_to_support = [&](uint32_t node) {
        for (int i = 0; i < support_size; ++i)
            if (support[i] == node) return;
        if (support_size < GPU_MAX_SUPPORT)
            support[support_size++] = node;
    };
    
    // Add u, v, and their neighbors
    add_to_support(u);
    add_to_support(v);
    for (uint32_t e = csr_offsets[u]; e < csr_offsets[u + 1]; ++e)
        add_to_support(csr_neighbors[e]);
    for (uint32_t e = csr_offsets[v]; e < csr_offsets[v + 1]; ++e)
        add_to_support(csr_neighbors[e]);
    
    // Build probability vectors indexed by support position
    float mu_u[GPU_MAX_SUPPORT] = {};
    float mu_v[GPU_MAX_SUPPORT] = {};
    
    // Find positions of u and its neighbors in support
    float neighbor_prob_u = (1.0f - alpha) / deg_u;
    for (uint32_t e = csr_offsets[u]; e < csr_offsets[u + 1]; ++e) {
        uint32_t nbr = csr_neighbors[e];
        for (int i = 0; i < support_size; ++i) {
            if (support[i] == nbr) { mu_u[i] += neighbor_prob_u; break; }
        }
    }
    // Add self-loop mass for u
    if (alpha > 0.0f) {
        for (int i = 0; i < support_size; ++i) {
            if (support[i] == u) { mu_u[i] += alpha; break; }
        }
    }
    
    float neighbor_prob_v = (1.0f - alpha) / deg_v;
    for (uint32_t e = csr_offsets[v]; e < csr_offsets[v + 1]; ++e) {
        uint32_t nbr = csr_neighbors[e];
        for (int i = 0; i < support_size; ++i) {
            if (support[i] == nbr) { mu_v[i] += neighbor_prob_v; break; }
        }
    }
    if (alpha > 0.0f) {
        for (int i = 0; i < support_size; ++i) {
            if (support[i] == v) { mu_v[i] += alpha; break; }
        }
    }
    
    // Compute pairwise distances between support nodes via BFS
    float dist_matrix[GPU_MAX_SUPPORT * GPU_MAX_SUPPORT];
    for (int i = 0; i < support_size; ++i) {
        float row_dists[GPU_MAX_SUPPORT];
        bfs_distances_device(csr_offsets, csr_neighbors, num_nodes,
                            support[i], support, support_size, row_dists);
        for (int j = 0; j < support_size; ++j)
            dist_matrix[i * support_size + j] = row_dists[j];
    }
    
    // Solve optimal transport
    float w1 = solve_transport_device(dist_matrix, mu_u, mu_v, support_size);
    
    // Curvature = 1 - W₁/d(u,v), where d(u,v) = 1 for adjacent nodes
    curvatures[eid] = 1.0f - w1;
}

// -------------------------------------------------------------------
// Host-side launch function
// -------------------------------------------------------------------

struct GPUGraph {
    uint32_t* d_csr_offsets;
    uint32_t* d_csr_neighbors;
    uint32_t num_nodes;
    uint32_t num_csr_entries;
};

#define HIP_CHECK(call) do { \
    hipError_t err = (call); \
    if (err != hipSuccess) { \
        throw std::runtime_error(std::string("HIP error: ") + hipGetErrorString(err)); \
    } \
} while(0)

GPUGraph upload_graph_to_gpu(
    const uint32_t* csr_offsets, uint32_t num_nodes,
    const uint32_t* csr_neighbors, uint32_t num_csr_entries)
{
    GPUGraph g;
    g.num_nodes = num_nodes;
    g.num_csr_entries = num_csr_entries;
    
    HIP_CHECK(hipMalloc(&g.d_csr_offsets, (num_nodes + 1) * sizeof(uint32_t)));
    HIP_CHECK(hipMalloc(&g.d_csr_neighbors, num_csr_entries * sizeof(uint32_t)));
    HIP_CHECK(hipMemcpy(g.d_csr_offsets, csr_offsets,
                        (num_nodes + 1) * sizeof(uint32_t), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(g.d_csr_neighbors, csr_neighbors,
                        num_csr_entries * sizeof(uint32_t), hipMemcpyHostToDevice));
    return g;
}

void free_gpu_graph(GPUGraph& g) {
    (void)hipFree(g.d_csr_offsets);
    (void)hipFree(g.d_csr_neighbors);
    g.d_csr_offsets = nullptr;
    g.d_csr_neighbors = nullptr;
}

void gpu_compute_ollivier_ricci(
    const uint32_t* csr_offsets,
    const uint32_t* csr_neighbors,
    uint32_t num_nodes,
    uint32_t num_csr_entries,
    const uint32_t* h_edge_u,
    const uint32_t* h_edge_v,
    uint32_t num_edges,
    float alpha,
    float* h_curvatures)
{
    // Upload graph
    GPUGraph g = upload_graph_to_gpu(csr_offsets, num_nodes, csr_neighbors, num_csr_entries);
    
    // Upload edge lists
    uint32_t *d_edge_u, *d_edge_v;
    float *d_curvatures;
    
    HIP_CHECK(hipMalloc(&d_edge_u, num_edges * sizeof(uint32_t)));
    HIP_CHECK(hipMalloc(&d_edge_v, num_edges * sizeof(uint32_t)));
    HIP_CHECK(hipMalloc(&d_curvatures, num_edges * sizeof(float)));
    
    HIP_CHECK(hipMemcpy(d_edge_u, h_edge_u, num_edges * sizeof(uint32_t), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_edge_v, h_edge_v, num_edges * sizeof(uint32_t), hipMemcpyHostToDevice));
    
    // Launch kernel
    int block_size = 64;  // Conservative for high register usage per thread
    int grid_size = (num_edges + block_size - 1) / block_size;
    
    hipLaunchKernelGGL(ollivier_ricci_kernel, dim3(grid_size), dim3(block_size), 0, 0,
                       g.d_csr_offsets, g.d_csr_neighbors, num_nodes,
                       d_edge_u, d_edge_v, num_edges, alpha, d_curvatures);
    
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());
    
    // Download results
    HIP_CHECK(hipMemcpy(h_curvatures, d_curvatures, num_edges * sizeof(float), hipMemcpyDeviceToHost));
    
    // Cleanup
    (void)hipFree(d_edge_u);
    (void)hipFree(d_edge_v);
    (void)hipFree(d_curvatures);
    free_gpu_graph(g);
}

} // namespace discretum
