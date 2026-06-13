#include <stdio.h>

// This is a CUDA kernel - it runs on the GPU
// The __global__ keyword means: "callable from CPU, runs on GPU"
__global__ void addArrays(float* a, float* b, float* result, int n) {
    // Each thread computes its own unique index
    // blockIdx.x  = which block this thread is in
    // blockDim.x  = how many threads per block
    // threadIdx.x = which thread within the block
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    // Guard: don't go out of bounds
    if (index < n) {
        result[index] = a[index] + b[index];
    }
}

int main() {
    int n = 1000000; // 1 million elements
    size_t bytes = n * sizeof(float);

    // Allocate memory on the CPU (called "host")
    float* h_a      = (float*)malloc(bytes);
    float* h_b      = (float*)malloc(bytes);
    float* h_result = (float*)malloc(bytes);

    // Fill arrays with test values
    for (int i = 0; i < n; i++) {
        h_a[i] = (float)i;
        h_b[i] = (float)(i * 2);
    }

    // Allocate memory on the GPU (called "device")
    float* d_a;
    float* d_b;
    float* d_result;
    cudaMalloc(&d_a,      bytes);
    cudaMalloc(&d_b,      bytes);
    cudaMalloc(&d_result, bytes);

    // Copy data from CPU to GPU
    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);

    // Launch the kernel
    // 256 threads per block, enough blocks to cover all n elements
    int threadsPerBlock = 256;
    int blocksPerGrid   = (n + threadsPerBlock - 1) / threadsPerBlock;

    addArrays<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_result, n);

    // Copy result back from GPU to CPU
    cudaMemcpy(h_result, d_result, bytes, cudaMemcpyDeviceToHost);

    // Verify a few results
    printf("Verifying results...\n");
    printf("result[0]      = %.0f (expected %.0f)\n", h_result[0],      0.0f + 0.0f);
    printf("result[1]      = %.0f (expected %.0f)\n", h_result[1],      1.0f + 2.0f);
    printf("result[999999] = %.0f (expected %.0f)\n", h_result[999999], 999999.0f + 1999998.0f);

    // Check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
        return 1;
    }

    printf("\nSuccess! 1 million additions completed in parallel on your RTX 3050.\n");

    // Free memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_result);
    free(h_a);
    free(h_b);
    free(h_result);

    return 0;
}