#include <stdio.h>

__global__ void addArrays(float* a, float* b, float* result, int n) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < n) {
        result[index] = a[index] + b[index];
    }
}

int main() {
    // Test with different sizes to see how GPU scales
    int sizes[] = {1000, 100000, 1000000, 10000000};
    int numSizes = 4;

    printf("%-15s %-20s %-20s\n", "Elements", "Time (ms)", "Throughput (M/s)");
    printf("%-15s %-20s %-20s\n", "---------", "---------", "----------------");

    for (int s = 0; s < numSizes; s++) {
        int n = sizes[s];
        size_t bytes = n * sizeof(float);

        // Allocate host memory
        float* h_a      = (float*)malloc(bytes);
        float* h_b      = (float*)malloc(bytes);
        float* h_result = (float*)malloc(bytes);

        for (int i = 0; i < n; i++) {
            h_a[i] = (float)i;
            h_b[i] = (float)i;
        }

        // Allocate device memory
        float *d_a, *d_b, *d_result;
        cudaMalloc(&d_a,      bytes);
        cudaMalloc(&d_b,      bytes);
        cudaMalloc(&d_result, bytes);

        cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);

        // Create CUDA events for timing
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        int threadsPerBlock = 256;
        int blocksPerGrid   = (n + threadsPerBlock - 1) / threadsPerBlock;

        // Record start event on GPU timeline
        cudaEventRecord(start);

        // Launch kernel
        addArrays<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_result, n);

        // Record stop event and wait for GPU to finish
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        // Calculate elapsed time in milliseconds
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);

        float throughput = (float)n / milliseconds / 1000.0f;

        printf("%-15d %-20.4f %-20.1f\n", n, milliseconds, throughput);

        // Cleanup
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        cudaFree(d_a);
        cudaFree(d_b);
        cudaFree(d_result);
        free(h_a);
        free(h_b);
        free(h_result);
    }

    printf("\nYour RTX 3050 can process hundreds of millions of elements per second.\n");
    printf("At 1M creatures, each update step will take a fraction of a millisecond.\n");

    return 0;
}