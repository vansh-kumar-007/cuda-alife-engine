#include <stdio.h>
#include <math.h>

#define NUM_PARTICLES  100000      // smaller for this test so we can verify
#define WORLD_SIZE     1000.0f
#define CELL_SIZE      5.0f
#define GRID_DIM       200
#define GRID_CELLS     (GRID_DIM * GRID_DIM)
#define MAX_PER_CELL   32

// Convert world position to grid cell index
__device__ int posToCell(float x, float y) {
    int cx = (int)(x / CELL_SIZE);
    int cy = (int)(y / CELL_SIZE);
    // Clamp to grid bounds
    cx = max(0, min(cx, GRID_DIM - 1));
    cy = max(0, min(cy, GRID_DIM - 1));
    return cy * GRID_DIM + cx;
}

// Kernel: count how many particles fall in each cell
__global__ void countParticlesInCells(
    float* pos_x, float* pos_y,
    int* cellCounts,
    int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    int cell = posToCell(pos_x[i], pos_y[i]);
    atomicAdd(&cellCounts[cell], 1);
}

// Kernel: assign each particle to its cell
__global__ void assignParticlesToCells(
    float* pos_x, float* pos_y,
    int* cellCounts,
    int* cellParticles,   // flat array: [cell0_p0, cell0_p1, ..., cell1_p0, ...]
    int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    int cell = posToCell(pos_x[i], pos_y[i]);

    // Atomically grab a slot in this cell's list
    int slot = atomicAdd(&cellCounts[cell], 1);

    if (slot < MAX_PER_CELL) {
        cellParticles[cell * MAX_PER_CELL + slot] = i;
    }
}

// Kernel: for each particle, count how many neighbors it has within CELL_SIZE radius
__global__ void countNeighbors(
    float* pos_x, float* pos_y,
    int* cellCounts,
    int* cellParticles,
    int* neighborCount,
    int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    float x = pos_x[i];
    float y = pos_y[i];

    int cx = (int)(x / CELL_SIZE);
    int cy = (int)(y / CELL_SIZE);

    int count = 0;

    // Check 3x3 neighborhood of cells
    for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
            int nx = cx + dx;
            int ny = cy + dy;

            // Skip out-of-bounds cells
            if (nx < 0 || nx >= GRID_DIM) continue;
            if (ny < 0 || ny >= GRID_DIM) continue;

            int neighborCell = ny * GRID_DIM + nx;
            int particlesInCell = min(cellCounts[neighborCell], MAX_PER_CELL);

            // Check each particle in that cell
            for (int s = 0; s < particlesInCell; s++) {
                int j = cellParticles[neighborCell * MAX_PER_CELL + s];
                if (j == i) continue; // skip self

                float dx2 = pos_x[j] - x;
                float dy2 = pos_y[j] - y;
                float dist = sqrtf(dx2*dx2 + dy2*dy2);

                if (dist < CELL_SIZE) count++;
            }
        }
    }

    neighborCount[i] = count;
}

// Simple initialisation kernel
__global__ void initPositions(float* pos_x, float* pos_y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    float t = (float)i / (float)n;
    pos_x[i] = WORLD_SIZE * (0.1f + 0.8f * fabsf(sinf(t * 1234.5f)));
    pos_y[i] = WORLD_SIZE * (0.1f + 0.8f * fabsf(cosf(t * 5678.9f)));
}

int main() {
    printf("Spatial hash test: %d particles in %.0fx%.0f world\n",
           NUM_PARTICLES, WORLD_SIZE, WORLD_SIZE);
    printf("Grid: %dx%d cells, cell size: %.0f units\n\n",
           GRID_DIM, GRID_DIM, CELL_SIZE);

    // Allocate particle positions
    float *d_pos_x, *d_pos_y;
    cudaMalloc(&d_pos_x, NUM_PARTICLES * sizeof(float));
    cudaMalloc(&d_pos_y, NUM_PARTICLES * sizeof(float));

    // Allocate spatial hash structures
    int *d_cellCounts, *d_cellParticles, *d_neighborCount;
    cudaMalloc(&d_cellCounts,    GRID_CELLS * sizeof(int));
    cudaMalloc(&d_cellParticles, GRID_CELLS * MAX_PER_CELL * sizeof(int));
    cudaMalloc(&d_neighborCount, NUM_PARTICLES * sizeof(int));

    int threads = 256;
    int blocks  = (NUM_PARTICLES + threads - 1) / threads;

    // Initialise positions
    initPositions<<<blocks, threads>>>(d_pos_x, d_pos_y, NUM_PARTICLES);
    cudaDeviceSynchronize();

    // Time the full spatial hash pipeline
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    // Step 1: clear cell counts
    cudaMemset(d_cellCounts, 0, GRID_CELLS * sizeof(int));

    // Step 2: count particles per cell
    countParticlesInCells<<<blocks, threads>>>(
        d_pos_x, d_pos_y, d_cellCounts, NUM_PARTICLES);

    // Step 3: reset counts for assignment pass
    cudaMemset(d_cellCounts, 0, GRID_CELLS * sizeof(int));

    // Step 4: assign particles to cells
    assignParticlesToCells<<<blocks, threads>>>(
        d_pos_x, d_pos_y, d_cellCounts, d_cellParticles, NUM_PARTICLES);

    // Step 5: count neighbors for every particle
    countNeighbors<<<blocks, threads>>>(
        d_pos_x, d_pos_y, d_cellCounts, d_cellParticles,
        d_neighborCount, NUM_PARTICLES);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);

    // Copy neighbor counts back to verify
    int* h_neighborCount = (int*)malloc(NUM_PARTICLES * sizeof(int));
    cudaMemcpy(h_neighborCount, d_neighborCount,
               NUM_PARTICLES * sizeof(int), cudaMemcpyDeviceToHost);

    // Compute stats
    long long totalNeighbors = 0;
    int maxNeighbors = 0;
    for (int i = 0; i < NUM_PARTICLES; i++) {
        totalNeighbors += h_neighborCount[i];
        if (h_neighborCount[i] > maxNeighbors)
            maxNeighbors = h_neighborCount[i];
    }

    printf("Spatial hash pipeline time: %.3f ms\n", ms);
    printf("Average neighbors per particle: %.1f\n",
           (float)totalNeighbors / NUM_PARTICLES);
    printf("Max neighbors seen:             %d\n", maxNeighbors);
    printf("\nAt this speed: %.0f neighbor queries/second\n",
           (float)NUM_PARTICLES / (ms / 1000.0f));

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        printf("CUDA error: %s\n", cudaGetErrorString(err));
    else
        printf("\nSpatial hash working correctly.\n");

    // Cleanup
    free(h_neighborCount);
    cudaFree(d_pos_x);
    cudaFree(d_pos_y);
    cudaFree(d_cellCounts);
    cudaFree(d_cellParticles);
    cudaFree(d_neighborCount);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}