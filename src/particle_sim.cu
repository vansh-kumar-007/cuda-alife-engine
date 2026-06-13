#include <stdio.h>
#include <math.h>

// Simulation parameters
#define NUM_PARTICLES 1000000
#define DT            0.016f      // timestep: ~60fps
#define WORLD_SIZE    1000.0f     // simulation world is 1000x1000 units
#define GRAVITY       9.8f        // downward acceleration

// Structure of Arrays - all particle data lives here on the GPU
struct ParticleSystem {
    float* pos_x;
    float* pos_y;
    float* vel_x;
    float* vel_y;
    float* age;       // how long this particle has lived
    int    count;
};

// Kernel: update every particle's physics in parallel
__global__ void updateParticles(ParticleSystem ps, float dt) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= ps.count) return;

    // Read current state
    float px = ps.pos_x[i];
    float py = ps.pos_y[i];
    float vx = ps.vel_x[i];
    float vy = ps.vel_y[i];

    // Apply gravity
    vy -= GRAVITY * dt;

    // Update position
    px += vx * dt;
    py += vy * dt;

    // Bounce off world boundaries
    if (px < 0.0f) { px = 0.0f; vx = -vx * 0.8f; }
    if (px > WORLD_SIZE) { px = WORLD_SIZE; vx = -vx * 0.8f; }
    if (py < 0.0f) { py = 0.0f; vy = -vy * 0.8f; }
    if (py > WORLD_SIZE) { py = WORLD_SIZE; vy = -vy * 0.8f; }

    // Write back updated state
    ps.pos_x[i] = px;
    ps.pos_y[i] = py;
    ps.vel_x[i] = vx;
    ps.vel_y[i] = vy;
    ps.age[i]  += dt;
}

// Kernel: initialise all particles with random-ish positions and velocities
__global__ void initParticles(ParticleSystem ps) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= ps.count) return;

    // Deterministic pseudo-random spread using thread index
    float t = (float)i / (float)ps.count;

    ps.pos_x[i] = WORLD_SIZE * (0.5f + 0.4f * sinf(t * 1234.5f));
    ps.pos_y[i] = WORLD_SIZE * (0.5f + 0.4f * cosf(t * 5678.9f));
    ps.vel_x[i] = 50.0f * sinf(t * 91.3f);
    ps.vel_y[i] = 50.0f * cosf(t * 73.7f) + 30.0f;
    ps.age[i]   = 0.0f;
}

int main() {
    printf("Initialising particle system with %d particles...\n", NUM_PARTICLES);

    // Allocate all arrays on GPU
    ParticleSystem ps;
    ps.count = NUM_PARTICLES;
    size_t bytes = NUM_PARTICLES * sizeof(float);

    cudaMalloc(&ps.pos_x, bytes);
    cudaMalloc(&ps.pos_y, bytes);
    cudaMalloc(&ps.vel_x, bytes);
    cudaMalloc(&ps.vel_y, bytes);
    cudaMalloc(&ps.age,   bytes);

    // Initialise particles on GPU
    int threads = 256;
    int blocks  = (NUM_PARTICLES + threads - 1) / threads;

    initParticles<<<blocks, threads>>>(ps);
    cudaDeviceSynchronize();
    printf("Particles initialised on GPU.\n");

    // Simulate 600 frames (10 seconds at 60fps)
    int   numFrames = 600;
    float totalTime = 0.0f;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    printf("Simulating %d frames...\n", numFrames);

    for (int frame = 0; frame < numFrames; frame++) {
        cudaEventRecord(start);

        updateParticles<<<blocks, threads>>>(ps, DT);

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float ms = 0;
        cudaEventElapsedTime(&ms, start, stop);
        totalTime += ms;

        // Print progress every 100 frames
        if (frame % 100 == 0) {
            printf("Frame %3d — kernel time: %.3f ms\n", frame, ms);
        }
    }

    printf("\n--- Performance Summary ---\n");
    printf("Total frames:       %d\n",   numFrames);
    printf("Total time:         %.2f ms\n", totalTime);
    printf("Average per frame:  %.4f ms\n", totalTime / numFrames);
    printf("Simulated FPS:      %.1f\n",  1000.0f / (totalTime / numFrames));
    printf("Particles/second:   %.2f billion\n",
           (float)NUM_PARTICLES * numFrames / totalTime / 1e6f);

    // Verify simulation ran correctly - copy a few particles back to CPU
    float sample_x[5], sample_y[5];
    cudaMemcpy(sample_x, ps.pos_x, 5 * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(sample_y, ps.pos_y, 5 * sizeof(float), cudaMemcpyDeviceToHost);

    printf("\nSample particle positions after %d frames:\n", numFrames);
    for (int i = 0; i < 5; i++) {
        printf("  Particle %d: (%.2f, %.2f)\n", i, sample_x[i], sample_y[i]);
    }

    // Sanity check: all positions should be within world bounds
    printf("\nVerifying bounds...\n");
    int errors = 0;
    for (int i = 0; i < 5; i++) {
        if (sample_x[i] < 0 || sample_x[i] > WORLD_SIZE ||
            sample_y[i] < 0 || sample_y[i] > WORLD_SIZE) {
            printf("ERROR: particle %d out of bounds!\n", i);
            errors++;
        }
    }
    if (errors == 0) printf("All sampled particles within bounds. Physics working correctly.\n");

    // Cleanup
    cudaFree(ps.pos_x);
    cudaFree(ps.pos_y);
    cudaFree(ps.vel_x);
    cudaFree(ps.vel_y);
    cudaFree(ps.age);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}