#include <hip/hip_runtime.h>
#include <cstdint>
#include <vector>
#include <stdexcept>

namespace discretum {

#define HIP_CHECK_RW(call) do { \
    hipError_t err = (call); \
    if (err != hipSuccess) { \
        throw std::runtime_error(std::string("HIP error: ") + hipGetErrorString(err)); \
    } \
} while(0)

// -------------------------------------------------------------------
// Device: PCG32 random number generator (matches CPU implementation)
// -------------------------------------------------------------------
struct PCG32_Device {
    uint64_t state;
    uint64_t inc;
    
    __device__ PCG32_Device(uint64_t seed, uint64_t seq) {
        state = 0;
        inc = (seq << 1u) | 1u;
        next();
        state += seed;
        next();
    }
    
    __device__ uint32_t next() {
        uint64_t old_state = state;
        state = old_state * 6364136223846793005ULL + inc;
        uint32_t xorshifted = static_cast<uint32_t>(((old_state >> 18u) ^ old_state) >> 27u);
        uint32_t rot = static_cast<uint32_t>(old_state >> 59u);
        return (xorshifted >> rot) | (xorshifted << ((-rot) & 31));
    }
    
    __device__ uint32_t uniform(uint32_t bound) {
        uint32_t threshold = (-bound) % bound;
        uint32_t r;
        do { r = next(); } while (r < threshold);
        return r % bound;
    }
};

// -------------------------------------------------------------------
// Kernel: each thread runs one random walk, records at_origin per step
// -------------------------------------------------------------------
__global__ void random_walk_trace_kernel(
    const uint32_t* __restrict__ csr_offsets,
    const uint32_t* __restrict__ csr_neighbors,
    const uint32_t* __restrict__ start_nodes,  // [num_walkers]
    uint32_t num_walkers,
    uint32_t max_steps,
    uint64_t base_seed,
    uint32_t* __restrict__ return_counts)       // [max_steps+1] atomicAdd
{
    uint32_t wid = blockIdx.x * blockDim.x + threadIdx.x;
    if (wid >= num_walkers) return;
    
    PCG32_Device rng(base_seed + wid * 6364136223846793005ULL + 1, wid);
    
    uint32_t start = start_nodes[wid];
    uint32_t current = start;
    
    for (uint32_t step = 1; step <= max_steps; ++step) {
        uint32_t deg = csr_offsets[current + 1] - csr_offsets[current];
        if (deg == 0) {
            // Isolated node: always at origin
            atomicAdd(&return_counts[step], 1u);
            continue;
        }
        
        uint32_t nbr_idx = rng.uniform(deg);
        current = csr_neighbors[csr_offsets[current] + nbr_idx];
        
        if (current == start) {
            atomicAdd(&return_counts[step], 1u);
        }
    }
}

// -------------------------------------------------------------------
// Host launch function
// -------------------------------------------------------------------
void gpu_random_walk_trace(
    const uint32_t* csr_offsets,
    const uint32_t* csr_neighbors,
    uint32_t num_nodes,
    uint32_t num_csr_entries,
    const uint32_t* h_start_nodes,
    uint32_t num_walkers,
    uint32_t max_steps,
    uint64_t seed,
    uint32_t* h_return_counts)  // [max_steps+1], zeroed by caller
{
    // Upload graph
    uint32_t *d_csr_offsets, *d_csr_neighbors, *d_start_nodes, *d_return_counts;
    
    HIP_CHECK_RW(hipMalloc(&d_csr_offsets, (num_nodes + 1) * sizeof(uint32_t)));
    HIP_CHECK_RW(hipMalloc(&d_csr_neighbors, num_csr_entries * sizeof(uint32_t)));
    HIP_CHECK_RW(hipMalloc(&d_start_nodes, num_walkers * sizeof(uint32_t)));
    HIP_CHECK_RW(hipMalloc(&d_return_counts, (max_steps + 1) * sizeof(uint32_t)));
    
    HIP_CHECK_RW(hipMemcpy(d_csr_offsets, csr_offsets,
                           (num_nodes + 1) * sizeof(uint32_t), hipMemcpyHostToDevice));
    HIP_CHECK_RW(hipMemcpy(d_csr_neighbors, csr_neighbors,
                           num_csr_entries * sizeof(uint32_t), hipMemcpyHostToDevice));
    HIP_CHECK_RW(hipMemcpy(d_start_nodes, h_start_nodes,
                           num_walkers * sizeof(uint32_t), hipMemcpyHostToDevice));
    HIP_CHECK_RW(hipMemset(d_return_counts, 0, (max_steps + 1) * sizeof(uint32_t)));
    
    // Launch
    int block_size = 256;
    int grid_size = (num_walkers + block_size - 1) / block_size;
    
    hipLaunchKernelGGL(random_walk_trace_kernel, dim3(grid_size), dim3(block_size), 0, 0,
                       d_csr_offsets, d_csr_neighbors, d_start_nodes,
                       num_walkers, max_steps, seed, d_return_counts);
    
    HIP_CHECK_RW(hipGetLastError());
    HIP_CHECK_RW(hipDeviceSynchronize());
    
    // Download
    HIP_CHECK_RW(hipMemcpy(h_return_counts, d_return_counts,
                           (max_steps + 1) * sizeof(uint32_t), hipMemcpyDeviceToHost));
    
    // Cleanup
    hipFree(d_csr_offsets);
    hipFree(d_csr_neighbors);
    hipFree(d_start_nodes);
    hipFree(d_return_counts);
}

} // namespace discretum
