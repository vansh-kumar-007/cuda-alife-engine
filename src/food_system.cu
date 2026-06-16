#include "creatures.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>

// ─────────────────────────────────────────────────────────────────────────────
// Kernel: plants gain energy from sunlight each frame
// ─────────────────────────────────────────────────────────────────────────────
__global__ void sunlightUpdate(CreatureArrays c, float dt) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= c.count) return;
    if (c.state[i] != STATE_ALIVE) return;
    if (c.species[i] != SPECIES_PLANT) return;

    // Passive energy gain from sunlight
    c.energy[i] += ENERGY_SUNLIGHT * dt;

    // Clamp to max
    if (c.energy[i] > ENERGY_MAX) c.energy[i] = ENERGY_MAX;
}

// ─────────────────────────────────────────────────────────────────────────────
// Kernel: plants reproduce into dead slots
//
// Strategy: each plant checks ONE candidate slot (based on its own index).
// If that slot is dead, atomically claim it and spawn a child.
// This avoids multiple plants fighting over the same slot.
// ─────────────────────────────────────────────────────────────────────────────
__global__ void plantReproduction(CreatureArrays c, int* birthCount) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= c.count) return;
    if (c.state[i] != STATE_ALIVE) return;
    if (c.species[i] != SPECIES_PLANT) return;
    if (c.energy[i] < ENERGY_REPRODUCE) return;

    // Try 4 candidate slots before giving up
    for (int attempt = 0; attempt < 4; attempt++) {
        int candidate = (i * 1013 + attempt * 97 + 7) % c.count;
        int old = atomicCAS(&c.state[candidate], STATE_DEAD, STATE_ALIVE);

        if (old == STATE_DEAD) {
            float childEnergy    = c.energy[i] * ENERGY_INHERIT;
            c.energy[i]         -= childEnergy;
            float angle          = (float)candidate * 2.399f;
            float offset         = 15.0f;
            c.pos_x[candidate]   = fmodf(c.pos_x[i] + cosf(angle)*offset + WORLD_SIZE_F, WORLD_SIZE_F);
            c.pos_y[candidate]   = fmodf(c.pos_y[i] + sinf(angle)*offset + WORLD_SIZE_F, WORLD_SIZE_F);
            c.vel_x[candidate]   = 0.0f;
            c.vel_y[candidate]   = 0.0f;
            c.energy[candidate]  = childEnergy;
            c.age[candidate]     = 0.0f;
            c.size[candidate]    = 1.5f;
            c.species[candidate] = SPECIES_PLANT;
            c.color_r[candidate] = 0.1f;
            c.color_g[candidate] = 0.9f;
            c.color_b[candidate] = 0.1f;
            atomicAdd(birthCount, 1);
            break; // one birth per parent per frame
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Metabolism kernel (plants only version for this test)
// ─────────────────────────────────────────────────────────────────────────────
__global__ void metabolismUpdate(CreatureArrays c, float dt) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= c.count) return;
    if (c.state[i] != STATE_ALIVE) return;

    float cost = 0.0f;
    if      (c.species[i] == SPECIES_PLANT)     cost = ENERGY_COST_PLANT;
    else if (c.species[i] == SPECIES_HERBIVORE) cost = ENERGY_COST_HERBIVORE;
    else                                         cost = ENERGY_COST_PREDATOR;

    c.energy[i] -= cost * dt;
    c.age[i]    += dt;

    if (c.energy[i] <= ENERGY_DEATH) {
        c.state[i]  = STATE_DEAD;
        c.energy[i] = 0.0f;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Test program
// ─────────────────────────────────────────────────────────────────────────────
void countSpecies(CreatureArrays& c, int* plants, int* herbs, int* preds) {
    int n = c.count;
    int* h_state   = new int[n];
    int* h_species = new int[n];
    cudaMemcpy(h_state,   c.state,   n*sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_species, c.species, n*sizeof(int), cudaMemcpyDeviceToHost);

    *plants = *herbs = *preds = 0;
    for (int i = 0; i < n; i++) {
        if (h_state[i] != STATE_ALIVE) continue;
        if      (h_species[i] == SPECIES_PLANT)     (*plants)++;
        else if (h_species[i] == SPECIES_HERBIVORE) (*herbs)++;
        else                                         (*preds)++;
    }
    delete[] h_state;
    delete[] h_species;
}

__global__ void initPlantsOnly(CreatureArrays c, int plantCount) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= c.count) return;

    if (i < plantCount) {
        // Alive plant
        float t = (float)i / (float)plantCount;
        c.pos_x[i]   = WORLD_SIZE_F * (0.1f + 0.8f * fabsf(sinf(t * 2137.0f)));
        c.pos_y[i]   = WORLD_SIZE_F * (0.1f + 0.8f * fabsf(cosf(t * 3571.0f)));
        c.vel_x[i]   = 0.0f;
        c.vel_y[i]   = 0.0f;
        c.energy[i]  = 40.0f + t * 30.0f; // varied starting energy
        c.age[i]     = 0.0f;
        c.size[i]    = 1.5f;
        c.species[i] = SPECIES_PLANT;
        c.state[i]   = STATE_ALIVE;
        c.color_r[i] = 0.1f;
        c.color_g[i] = 0.9f;
        c.color_b[i] = 0.1f;
    } else {
        // Dead slot — available for reproduction
        c.state[i]   = STATE_DEAD;
        c.species[i] = SPECIES_PLANT;
        c.energy[i]  = 0.0f;
        c.age[i]     = 0.0f;
        c.pos_x[i]   = 0.0f;
        c.pos_y[i]   = 0.0f;
        c.vel_x[i]   = 0.0f;
        c.vel_y[i]   = 0.0f;
        c.color_r[i] = 0.0f;
        c.color_g[i] = 0.0f;
        c.color_b[i] = 0.0f;
        c.size[i]    = 0.0f;
    }
}

int main() {
    printf("=== Food System Test: Plant Sunlight + Reproduction ===\n\n");

    // 500K total slots, start with 10K plants
    // Remaining 490K slots are dead — available for reproduction
    int totalSlots  = 500000;
    int startPlants = 10000;

    CreatureArrays c;
    allocCreatures(c, totalSlots);

    int threads = 256;
    int blocks  = (totalSlots + threads - 1) / threads;

    initPlantsOnly<<<blocks, threads>>>(c, startPlants);
    cudaDeviceSynchronize();

    // Allocate birth counter on GPU
    int* d_birthCount;
    cudaMalloc(&d_birthCount, sizeof(int));

    printf("Starting with %d plants in %d slots.\n", startPlants, totalSlots);
    printf("Watching population grow via sunlight + reproduction...\n\n");
    printf("%-8s %-10s %-12s\n", "Time(s)", "Plants", "Births/step");
    printf("%-8s %-10s %-12s\n", "-------", "------", "-----------");

    float simTime = 0.0f;
    float dt      = 0.016f;

    for (int frame = 0; frame < 6000; frame++) {
        

        // 1. Sunlight gives energy to plants
        sunlightUpdate<<<blocks, threads>>>(c, dt);

        // 2. Plants with enough energy reproduce
        plantReproduction<<<blocks, threads>>>(c, d_birthCount);

        // 3. Metabolism drains energy, kills starved creatures
        metabolismUpdate<<<blocks, threads>>>(c, dt);

        simTime += dt;

        // Report every 500 frames
        if (frame % 500 == 499) {
            int births;
            cudaMemcpy(&births, d_birthCount, sizeof(int), cudaMemcpyDeviceToHost);

            int plants, herbs, preds;
            countSpecies(c, &plants, &herbs, &preds);

            printf("%-8.1f %-10d %-12d\n", simTime, plants, births);

            // Stop early if slots are full
            if (plants >= totalSlots * 95 / 100) {
                printf("\nSlots 95%% full at t=%.1fs — population stabilised.\n", simTime);
                break;
            }
        }
    }

    printf("\nFinal count:\n");
    int p, h, pr;
    countSpecies(c, &p, &h, &pr);
    printf("Plants: %d / %d slots used\n", p, totalSlots);

    cudaFree(d_birthCount);
    freeCreatures(c);
    return 0;
}