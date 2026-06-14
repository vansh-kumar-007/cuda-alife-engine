#include "creatures.h"
#include <stdio.h>

// Kernel: initialise creatures across three species
__global__ void initCreatures(CreatureArrays c) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= c.count) return;

    // Divide population into thirds
    int third = c.count / 3;
    int sp;
    if      (i < third)         sp = SPECIES_PLANT;
    else if (i < third * 2)     sp = SPECIES_HERBIVORE;
    else                        sp = SPECIES_PREDATOR;

    c.species[i] = sp;
    c.state[i]   = STATE_ALIVE;
    c.age[i]     = 0.0f;

    // Scatter across world using deterministic pattern
    float t = (float)i / (float)c.count;
    c.pos_x[i] = WORLD_SIZE_F * (0.1f + 0.8f * fabsf(sinf(t * 2137.0f)));
    c.pos_y[i] = WORLD_SIZE_F * (0.1f + 0.8f * fabsf(cosf(t * 3571.0f)));
    c.vel_x[i] = 0.0f;
    c.vel_y[i] = 0.0f;

    // Species-specific starting values
    if (sp == SPECIES_PLANT) {
        c.energy[i]  = ENERGY_START_PLANT;
        c.size[i]    = 1.5f;
        c.color_r[i] = 0.1f; c.color_g[i] = 0.9f; c.color_b[i] = 0.1f;
    }
    else if (sp == SPECIES_HERBIVORE) {
        c.energy[i]  = ENERGY_START_HERBIVORE;
        c.size[i]    = 2.5f;
        c.color_r[i] = 0.2f; c.color_g[i] = 0.5f; c.color_b[i] = 1.0f;
    }
    else {
        c.energy[i]  = ENERGY_START_PREDATOR;
        c.size[i]    = 4.0f;
        c.color_r[i] = 1.0f; c.color_g[i] = 0.2f; c.color_b[i] = 0.1f;
    }
}

// Kernel: one timestep of biological update
__global__ void updateCreatures(CreatureArrays c, float dt) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= c.count) return;
    if (c.state[i] == STATE_DEAD) return;

    // Age every creature
    c.age[i] += dt;

    // Drain energy by metabolism
    float cost = 0.0f;
    if      (c.species[i] == SPECIES_PLANT)      cost = ENERGY_COST_PLANT;
    else if (c.species[i] == SPECIES_HERBIVORE)  cost = ENERGY_COST_HERBIVORE;
    else                                          cost = ENERGY_COST_PREDATOR;

    // Movement costs extra energy
    float speed = sqrtf(c.vel_x[i]*c.vel_x[i] + c.vel_y[i]*c.vel_y[i]);
    cost += speed * ENERGY_MOVE_COST;

    c.energy[i] -= cost * dt;

    // Death check
    if (c.energy[i] <= ENERGY_DEATH) {
        c.state[i]  = STATE_DEAD;
        c.energy[i] = 0.0f;
        return;
    }

    // Clamp energy to max
    if (c.energy[i] > ENERGY_MAX) c.energy[i] = ENERGY_MAX;

    // Update position
    c.pos_x[i] += c.vel_x[i] * dt;
    c.pos_y[i] += c.vel_y[i] * dt;

    // Wrap world boundaries
    if (c.pos_x[i] < 0)            c.pos_x[i] += WORLD_SIZE_F;
    if (c.pos_x[i] > WORLD_SIZE_F) c.pos_x[i] -= WORLD_SIZE_F;
    if (c.pos_y[i] < 0)            c.pos_y[i] += WORLD_SIZE_F;
    if (c.pos_y[i] > WORLD_SIZE_F) c.pos_y[i] -= WORLD_SIZE_F;
}

// Kernel: set velocities based on species behavior
__global__ void behaviorUpdate(CreatureArrays c, float time) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= c.count) return;
    if (c.state[i] == STATE_DEAD) return;

    float t = (float)i / (float)c.count;

    if (c.species[i] == SPECIES_PLANT) {
        // Plants don't move
        c.vel_x[i] = 0.0f;
        c.vel_y[i] = 0.0f;
    }
    else if (c.species[i] == SPECIES_HERBIVORE) {
        // Random wander — direction changes slowly over time
        float angle = t * 6.283f + time * (0.3f + t * 0.4f);
        float speed = 80.0f + t * 40.0f;
        c.vel_x[i] = cosf(angle) * speed;
        c.vel_y[i] = sinf(angle) * speed;
    }
    else {
        // Predators move faster
        float angle = t * 6.283f + time * (0.6f + t * 0.5f);
        float speed = 150.0f + t * 50.0f;
        c.vel_x[i] = cosf(angle) * speed;
        c.vel_y[i] = sinf(angle) * speed;
    }
}

// Count alive creatures per species on CPU (small diagnostic)
void countAlive(CreatureArrays& c) {
    int n = c.count;
    int* h_state   = new int[n];
    int* h_species = new int[n];
    cudaMemcpy(h_state,   c.state,   n*sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_species, c.species, n*sizeof(int), cudaMemcpyDeviceToHost);

    int alive[NUM_SPECIES] = {0,0,0};
    for (int i = 0; i < n; i++)
        if (h_state[i] == STATE_ALIVE)
            alive[h_species[i]]++;

    printf("Alive — Plants: %6d  Herbivores: %6d  Predators: %6d  Total: %d\n",
           alive[0], alive[1], alive[2],
           alive[0]+alive[1]+alive[2]);

    delete[] h_state;
    delete[] h_species;
}

int main() {
    printf("=== Creature System Test ===\n\n");

    // Allocate 300K creatures (100K per species)
    CreatureArrays creatures;
    allocCreatures(creatures, 300000);

    // Initialise on GPU
    int threads = 256;
    int blocks  = (creatures.count + threads - 1) / threads;
    initCreatures<<<blocks, threads>>>(creatures);
    cudaDeviceSynchronize();

    printf("Creatures initialised.\n");
    printCreatureStats(creatures);

    // Simulate 600 frames and track population
    printf("Simulating 3000 frames (48 seconds)...\n\n");

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float totalMs = 0;

    for (int frame = 0; frame < 3000; frame++) {
        cudaEventRecord(start);
        float simTime = frame * 0.016f;
        behaviorUpdate<<<blocks, threads>>>(creatures, simTime);
        updateCreatures<<<blocks, threads>>>(creatures, 0.016f);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float ms; cudaEventElapsedTime(&ms, start, stop);
        totalMs += ms;

        // Print population every 100 frames
        if (frame % 300 == 299) {
            printf("Frame %3d | kernel: %.3fms | ", frame+1, ms);
            countAlive(creatures);
        }
    }

    printf("\n--- Final State ---\n");
    printCreatureStats(creatures);
    printf("Average kernel time: %.4f ms\n", totalMs / 600.0f);
    printf("Simulated FPS:       %.1f\n",    1000.0f / (totalMs / 600.0f));

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    freeCreatures(creatures);
    return 0;
}