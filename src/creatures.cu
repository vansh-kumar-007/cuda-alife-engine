#include "creatures.h"
#include <stdio.h>
#include <cuda_runtime.h>

// Allocate all arrays on GPU
void allocCreatures(CreatureArrays& c, int count) {
    c.count = count;
    size_t fBytes = count * sizeof(float);
    size_t iBytes = count * sizeof(int);

    cudaMalloc(&c.pos_x,   fBytes);
    cudaMalloc(&c.pos_y,   fBytes);
    cudaMalloc(&c.vel_x,   fBytes);
    cudaMalloc(&c.vel_y,   fBytes);
    cudaMalloc(&c.energy,  fBytes);
    cudaMalloc(&c.age,     fBytes);
    cudaMalloc(&c.size,    fBytes);
    cudaMalloc(&c.color_r, fBytes);
    cudaMalloc(&c.color_g, fBytes);
    cudaMalloc(&c.color_b, fBytes);
    cudaMalloc(&c.species, iBytes);
    cudaMalloc(&c.state,   iBytes);

    printf("Creature arrays allocated: %d slots, %.1f MB VRAM\n",
           count,
           (float)(fBytes * 10 + iBytes * 2) / (1024.0f * 1024.0f));
}

// Free all GPU arrays
void freeCreatures(CreatureArrays& c) {
    cudaFree(c.pos_x);
    cudaFree(c.pos_y);
    cudaFree(c.vel_x);
    cudaFree(c.vel_y);
    cudaFree(c.energy);
    cudaFree(c.age);
    cudaFree(c.size);
    cudaFree(c.color_r);
    cudaFree(c.color_g);
    cudaFree(c.color_b);
    cudaFree(c.species);
    cudaFree(c.state);
}

// Sample 5 creatures from GPU and print their state
void printCreatureStats(CreatureArrays& c) {
    int   n        = min(c.count, 5);
    float h_px[5], h_py[5], h_energy[5], h_age[5];
    int   h_species[5], h_state[5];

    cudaMemcpy(h_px,      c.pos_x,   n*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_py,      c.pos_y,   n*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_energy,  c.energy,  n*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_age,     c.age,     n*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_species, c.species, n*sizeof(int),   cudaMemcpyDeviceToHost);
    cudaMemcpy(h_state,   c.state,   n*sizeof(int),   cudaMemcpyDeviceToHost);

    const char* speciesNames[] = {"Plant", "Herbivore", "Predator"};

    printf("\n--- Creature Sample ---\n");
    for (int i = 0; i < n; i++) {
        printf("  [%d] %-10s  pos=(%.1f,%.1f)  energy=%.1f  age=%.1f  %s\n",
               i,
               speciesNames[h_species[i]],
               h_px[i], h_py[i],
               h_energy[i],
               h_age[i],
               h_state[i] == STATE_ALIVE ? "ALIVE" : "DEAD");
    }
    printf("-----------------------\n\n");
}