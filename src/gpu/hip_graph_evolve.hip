#include <hip/hip_runtime.h>
#include <cstdint>
#include <cstring>
#include <vector>

namespace discretum {

static constexpr int GPU_NUM_STATES = 3;

/**
 * GPU kernel: state update for parametric cellular automaton.
 * Each thread updates one node's state based on neighbor majority vote
 * and the transition weight matrix theta_state.
 *
 * CSR format: row_ptr[i]..row_ptr[i+1]-1 are the column indices of node i's neighbors.
 * states[i] is the current state of node i.
 * theta_state[s * NUM_STATES + m] is the weight for transitioning to state s
 * when the neighbor majority is m.
 */
__global__ void evolve_states_kernel(
    const uint32_t* __restrict__ row_ptr,
    const uint32_t* __restrict__ col_idx,
    const uint8_t*  __restrict__ states,
    uint8_t*        __restrict__ new_states,
    const float*    __restrict__ theta_state,
    uint32_t num_nodes)
{
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_nodes) return;
    
    uint32_t start = row_ptr[tid];
    uint32_t end   = row_ptr[tid + 1];
    uint32_t degree = end - start;
    
    if (degree == 0) {
        new_states[tid] = states[tid];
        return;
    }
    
    // Count neighbor states
    int counts[GPU_NUM_STATES] = {};
    for (uint32_t e = start; e < end; ++e) {
        uint8_t s = states[col_idx[e]];
        if (s < GPU_NUM_STATES) counts[s]++;
    }
    
    // Find majority state
    int majority = 0;
    for (int s = 1; s < GPU_NUM_STATES; ++s) {
        if (counts[s] > counts[majority]) majority = s;
    }
    
    // Score each possible new state
    uint8_t cur = states[tid];
    float best_score = -1e30f;
    int best_state = cur;
    for (int s = 0; s < GPU_NUM_STATES; ++s) {
        float score = theta_state[s * GPU_NUM_STATES + majority];
        if (s == cur) score += 0.5f;  // Stability bias
        if (score > best_score) {
            best_score = score;
            best_state = s;
        }
    }
    
    new_states[tid] = static_cast<uint8_t>(best_state);
}

/**
 * Host-side function: launch GPU state evolution kernel.
 *
 * @param row_ptr  CSR row pointers (size num_nodes+1)
 * @param col_idx  CSR column indices (size num_edges*2)
 * @param states   Current node states (size num_nodes)
 * @param theta_state  Transition weight matrix (size NUM_STATES*NUM_STATES)
 * @param num_nodes  Number of nodes
 * @param num_edges_csr  Total entries in col_idx
 * @param new_states_out  Output: new states (size num_nodes)
 */
void launch_evolve_states(
    const uint32_t* row_ptr,
    const uint32_t* col_idx,
    const uint8_t*  states,
    const float*    theta_state,
    uint32_t num_nodes,
    uint32_t num_edges_csr,
    uint8_t* new_states_out)
{
    if (num_nodes == 0) return;
    
    uint32_t* d_row_ptr = nullptr;
    uint32_t* d_col_idx = nullptr;
    uint8_t*  d_states = nullptr;
    uint8_t*  d_new_states = nullptr;
    float*    d_theta = nullptr;
    
    size_t rp_bytes = (num_nodes + 1) * sizeof(uint32_t);
    size_t ci_bytes = num_edges_csr * sizeof(uint32_t);
    size_t st_bytes = num_nodes * sizeof(uint8_t);
    size_t th_bytes = GPU_NUM_STATES * GPU_NUM_STATES * sizeof(float);
    
    (void)hipMalloc(&d_row_ptr, rp_bytes);
    (void)hipMalloc(&d_col_idx, ci_bytes);
    (void)hipMalloc(&d_states, st_bytes);
    (void)hipMalloc(&d_new_states, st_bytes);
    (void)hipMalloc(&d_theta, th_bytes);
    
    (void)hipMemcpy(d_row_ptr, row_ptr, rp_bytes, hipMemcpyHostToDevice);
    (void)hipMemcpy(d_col_idx, col_idx, ci_bytes, hipMemcpyHostToDevice);
    (void)hipMemcpy(d_states, states, st_bytes, hipMemcpyHostToDevice);
    (void)hipMemcpy(d_theta, theta_state, th_bytes, hipMemcpyHostToDevice);
    
    int block_size = 256;
    int grid_size = (num_nodes + block_size - 1) / block_size;
    
    evolve_states_kernel<<<grid_size, block_size>>>(
        d_row_ptr, d_col_idx, d_states, d_new_states, d_theta, num_nodes);
    
    (void)hipDeviceSynchronize();
    
    (void)hipMemcpy(new_states_out, d_new_states, st_bytes, hipMemcpyDeviceToHost);
    
    (void)hipFree(d_row_ptr);
    (void)hipFree(d_col_idx);
    (void)hipFree(d_states);
    (void)hipFree(d_new_states);
    (void)hipFree(d_theta);
}

} // namespace discretum
