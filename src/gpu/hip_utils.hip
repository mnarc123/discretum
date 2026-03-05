#include <hip/hip_runtime.h>
#include <cstdio>

namespace discretum {

__device__ float atomic_add_float(float* addr, float val) {
    return atomicAdd(addr, val);
}

void check_hip_error(hipError_t err, const char* msg) {
    if (err != hipSuccess) {
        fprintf(stderr, "HIP error at %s: %s\n", msg, hipGetErrorString(err));
    }
}

bool gpu_available() {
    int device_count = 0;
    hipError_t err = hipGetDeviceCount(&device_count);
    return (err == hipSuccess && device_count > 0);
}

void gpu_print_info() {
    int device_count = 0;
    hipError_t err = hipGetDeviceCount(&device_count);
    if (err != hipSuccess || device_count == 0) {
        printf("No HIP-capable GPU detected.\n");
        return;
    }
    for (int i = 0; i < device_count; ++i) {
        hipDeviceProp_t prop;
        (void)hipGetDeviceProperties(&prop, i);
        printf("GPU %d: %s (%.0f MHz, %zu MB VRAM, %d CUs)\n",
               i, prop.name, prop.clockRate / 1000.0,
               prop.totalGlobalMem / (1024 * 1024),
               prop.multiProcessorCount);
    }
}

} // namespace discretum
