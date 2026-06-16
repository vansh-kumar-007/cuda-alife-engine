#include "creatures.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>

// ─── Spatial hash constants ───────────────────────────────────────────────────
#define CELL_SIZE    20.0f
#define GRID_DIM     50          // 1000 / 20 = 50
#define GRID_CELLS   (GRID_DIM * GRID_DIM)
#define MAX_PER_CELL 64

__device__ int posToCell(float x, float y) {
    int cx = (int)(x / CELL_SIZE);
    int cy = (int)(y / CELL_SIZE);
    cx = max(0, min(cx, GRID_DIM - 1));
    cy = max(0, min(cy, GRID_DIM - 1));
    return cy * GRID_DIM + cx;
}

// ─── Spatial hash build kernels ───────────────────────────────────────────────

__global__ void buildHashCounts(CreatureArrays c,
                                 int* cellCounts,
                                 int targetSpecies)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= c.count) return;
    if (c.state[i]   != STATE_ALIVE)    return;
    if (c.species[i] != targetSpecies)  return;

    int cell = posToCell(c.pos_x[i], c.pos_y[i]);
    atomicAdd(&cellCounts[cell], 1);
}

__global__ void buildHashAssign(CreatureArrays c,
                                 int* cellCounts,
                                 int* cellParticles,
                                 int targetSpecies)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= c.count) return;
    if (c.state[i]   != STATE_ALIVE)   return;
    if (c.species[i] != targetSpecies) return;

    int cell = posToCell(c.pos_x[i], c.pos_y[i]);
    int slot = atomicAdd(&cellCounts[cell], 1);
    if (slot < MAX_PER_CELL)
        cellParticles[cell * MAX_PER_CELL + slot] = i;
}

// ─── Sunlight kernel ──────────────────────────────────────────────────────────

__global__ void sunlightUpdate(CreatureArrays c, float dt) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= c.count) return;
    if (c.state[i]   != STATE_ALIVE)  return;
    if (c.species[i] != SPECIES_PLANT) return;

    c.energy[i] += ENERGY_SUNLIGHT * dt;
    if (c.energy[i] > ENERGY_MAX) c.energy[i] = ENERGY_MAX;
}

// ─── Plant reproduction ───────────────────────────────────────────────────────

__global__ void plantReproduction(CreatureArrays c, int* birthCount) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= c.count) return;
    if (c.state[i]   != STATE_ALIVE)  return;
    if (c.species[i] != SPECIES_PLANT) return;
    if (c.energy[i]  < ENERGY_REPRODUCE) return;

    for (int attempt = 0; attempt < 4; attempt++) {
        int candidate = (i * 1013 + attempt * 97 + 7) % c.count;
        int old = atomicCAS(&c.state[candidate], STATE_DEAD, STATE_ALIVE);

        if (old == STATE_DEAD) {
            float childEnergy        = c.energy[i] * ENERGY_INHERIT;
            c.energy[i]             -= childEnergy;
            float angle              = (float)candidate * 2.399f;
            c.pos_x[candidate]       = fmodf(c.pos_x[i] + cosf(angle)*15.0f + WORLD_SIZE_F, WORLD_SIZE_F);
            c.pos_y[candidate]       = fmodf(c.pos_y[i] + sinf(angle)*15.0f + WORLD_SIZE_F, WORLD_SIZE_F);
            c.vel_x[candidate]       = 0.0f;
            c.vel_y[candidate]       = 0.0f;
            c.energy[candidate]      = childEnergy;
            c.age[candidate]         = 0.0f;
            c.size[candidate]        = 1.5f;
            c.species[candidate]     = SPECIES_PLANT;
            c.state[candidate]       = STATE_ALIVE;
            c.color_r[candidate]     = 0.1f;
            c.color_g[candidate]     = 0.9f;
            c.color_b[candidate]     = 0.1f;
            atomicAdd(birthCount, 1);
            break;
        }
    }
}

// ─── Herbivore eating kernel ──────────────────────────────────────────────────

__global__ void herbivoresEat(
    CreatureArrays c,
    int* cellCounts,
    int* cellParticles,
    float eatRange,
    int*  eatCount)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= c.count) return;
    if (c.state[i]   != STATE_ALIVE)       return;
    if (c.species[i] != SPECIES_HERBIVORE) return;
    if (c.energy[i]  > 80.0f)             return; // not hungry

    float hx = c.pos_x[i];
    float hy = c.pos_y[i];

    int cx = (int)(hx / CELL_SIZE);
    int cy = (int)(hy / CELL_SIZE);

    for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
            int nx = cx + dx;
            int ny = cy + dy;
            if (nx < 0 || nx >= GRID_DIM) continue;
            if (ny < 0 || ny >= GRID_DIM) continue;

            int cell  = ny * GRID_DIM + nx;
            int count = min(cellCounts[cell], MAX_PER_CELL);

            for (int s = 0; s < count; s++) {
                int j = cellParticles[cell * MAX_PER_CELL + s];
                if (c.state[j]   != STATE_ALIVE)   continue;
                if (c.species[j] != SPECIES_PLANT) continue;

                float dist = hypotf(c.pos_x[j] - hx, c.pos_y[j] - hy);
                if (dist > eatRange) continue;

                int old = atomicCAS(&c.state[j], STATE_ALIVE, STATE_DEAD);
                if (old == STATE_ALIVE) {
                    c.energy[i] += ENERGY_GAIN_HERBIVORE;
                    if (c.energy[i] > 55.0f) return;   // was 80.0f - only eat when quite hungry
                    atomicAdd(eatCount, 1);
                    return;
                }
            }
        }
    }
}

// ─── Movement kernel ──────────────────────────────────────────────────────────

__global__ void movementUpdate(CreatureArrays c, float dt, float simTime) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= c.count) return;
    if (c.state[i] != STATE_ALIVE) return;
    if (c.species[i] == SPECIES_PLANT) return;

    float t = (float)i / (float)c.count;

    if (c.species[i] == SPECIES_HERBIVORE) {
        float angle = t * 6.283f + simTime * (0.3f + t * 0.4f);
        float speed = 15.0f + t * 10.0f;   // was 40+20
        c.vel_x[i]  = cosf(angle) * speed;
        c.vel_y[i]  = sinf(angle) * speed;
    }

    c.pos_x[i] += c.vel_x[i] * dt;
    c.pos_y[i] += c.vel_y[i] * dt;
    if (c.pos_x[i] < 0)            c.pos_x[i] += WORLD_SIZE_F;
    if (c.pos_x[i] > WORLD_SIZE_F) c.pos_x[i] -= WORLD_SIZE_F;
    if (c.pos_y[i] < 0)            c.pos_y[i] += WORLD_SIZE_F;
    if (c.pos_y[i] > WORLD_SIZE_F) c.pos_y[i] -= WORLD_SIZE_F;
}

// ─── Metabolism + death ───────────────────────────────────────────────────────

__global__ void metabolismUpdate(CreatureArrays c, float dt) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= c.count) return;
    if (c.state[i] != STATE_ALIVE) return;

    float cost = 0.0f;
    if      (c.species[i] == SPECIES_PLANT)     cost = ENERGY_COST_PLANT;
    else if (c.species[i] == SPECIES_HERBIVORE) cost = ENERGY_COST_HERBIVORE;
    else                                         cost = ENERGY_COST_PREDATOR;

    float speed  = hypotf(c.vel_x[i], c.vel_y[i]);
    cost        += speed * ENERGY_MOVE_COST;

    c.energy[i] -= cost * dt;
    c.age[i]    += dt;

    if (c.energy[i] <= ENERGY_DEATH) {
        c.state[i]  = STATE_DEAD;
        c.energy[i] = 0.0f;
    }
}

// ─── Herbivore reproduction ───────────────────────────────────────────────────

__global__ void herbivoreReproduction(CreatureArrays c, int* birthCount) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= c.count) return;
    if (c.state[i]   != STATE_ALIVE)       return;
    if (c.species[i] != SPECIES_HERBIVORE) return;
    if (c.energy[i] < 90.0f) return;   // was ENERGY_REPRODUCE (75)

    for (int attempt = 0; attempt < 4; attempt++) {
        int candidate = (i * 1013 + attempt * 97 + 13) % c.count;
        int old = atomicCAS(&c.state[candidate], STATE_DEAD, STATE_ALIVE);

        if (old == STATE_DEAD) {
            float childEnergy        = c.energy[i] * ENERGY_INHERIT;
            c.energy[i]             -= childEnergy;
            c.pos_x[candidate]       = c.pos_x[i];
            c.pos_y[candidate]       = c.pos_y[i];
            c.vel_x[candidate]       = 0.0f;
            c.vel_y[candidate]       = 0.0f;
            c.energy[candidate]      = childEnergy;
            c.age[candidate]         = 0.0f;
            c.size[candidate]        = 2.5f;
            c.species[candidate]     = SPECIES_HERBIVORE;
            c.state[candidate]       = STATE_ALIVE;
            c.color_r[candidate]     = 0.2f;
            c.color_g[candidate]     = 0.5f;
            c.color_b[candidate]     = 1.0f;
            atomicAdd(birthCount, 1);
            break;
        }
    }
}

// ─── Helpers ─────────────────────────────────────────────────────────────────

void countSpecies(CreatureArrays& c, int* plants, int* herbs) {
    int n = c.count;
    int* h_state   = new int[n];
    int* h_species = new int[n];
    cudaMemcpy(h_state,   c.state,   n*sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_species, c.species, n*sizeof(int), cudaMemcpyDeviceToHost);
    *plants = *herbs = 0;
    for (int i = 0; i < n; i++) {
        if (h_state[i] != STATE_ALIVE) continue;
        if      (h_species[i] == SPECIES_PLANT)     (*plants)++;
        else if (h_species[i] == SPECIES_HERBIVORE) (*herbs)++;
    }
    delete[] h_state;
    delete[] h_species;
}

// ─── Init ─────────────────────────────────────────────────────────────────────

__global__ void initEcosystem(CreatureArrays c,
                               int plantCount,
                               int herbCount)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= c.count) return;

    // Default: dead slot
    c.state[i]   = STATE_DEAD;
    c.energy[i]  = 0.0f;
    c.age[i]     = 0.0f;
    c.vel_x[i]   = 0.0f;
    c.vel_y[i]   = 0.0f;
    c.pos_x[i]   = 0.0f;
    c.pos_y[i]   = 0.0f;
    c.size[i]    = 0.0f;
    c.color_r[i] = 0.0f;
    c.color_g[i] = 0.0f;
    c.color_b[i] = 0.0f;
    c.species[i] = SPECIES_PLANT;

    float t = (float)i / (float)c.count;

    if (i < plantCount) {
        c.state[i]   = STATE_ALIVE;
        c.species[i] = SPECIES_PLANT;
        c.energy[i]  = 50.0f + t * 30.0f;
        c.size[i]    = 1.5f;
        c.pos_x[i] = WORLD_SIZE_F*(0.05f+0.9f*fabsf(sinf(t*2137.0f)));
        c.pos_y[i] = WORLD_SIZE_F*(0.05f+0.9f*fabsf(cosf(t*3571.0f)));
        c.color_r[i] = 0.1f;
        c.color_g[i] = 0.9f;
        c.color_b[i] = 0.1f;
    }
    else if (i < plantCount + herbCount) {
        c.state[i]   = STATE_ALIVE;
        c.species[i] = SPECIES_HERBIVORE;
        c.energy[i]  = ENERGY_START_HERBIVORE;
        c.size[i]    = 2.5f;
        c.pos_x[i]   = WORLD_SIZE_F*(0.05f+0.9f*fabsf(sinf(t*4231.0f)));
        c.pos_y[i]   = WORLD_SIZE_F*(0.05f+0.9f*fabsf(cosf(t*6173.0f)));
        c.color_r[i] = 0.2f;
        c.color_g[i] = 0.5f;
        c.color_b[i] = 1.0f;
    }
}

// ─── Main ─────────────────────────────────────────────────────────────────────

int main() {
    printf("=== Ecosystem Test: Plants + Herbivores ===\n\n");

    int totalSlots = 500000;
    int initPlants = 200000;
    int initHerbs  = 500;


    CreatureArrays c;
    allocCreatures(c, totalSlots);

    int threads = 256;
    int blocks  = (totalSlots + threads - 1) / threads;

    initEcosystem<<<blocks, threads>>>(c, initPlants, initHerbs);
    cudaDeviceSynchronize();

    // Spatial hash arrays
    int *d_cellCounts, *d_cellParticles;
    cudaMalloc(&d_cellCounts,
               GRID_CELLS * sizeof(int));
    cudaMalloc(&d_cellParticles,
               GRID_CELLS * MAX_PER_CELL * sizeof(int));

    // Counters
    int *d_births, *d_eats;
    cudaMalloc(&d_births, sizeof(int));
    cudaMalloc(&d_eats,   sizeof(int));

    printf("Initial state:\n");
    int p, h;
    countSpecies(c, &p, &h);
    printf("  Plants: %d   Herbivores: %d\n\n", p, h);

    printf("%-8s %-10s %-12s %-10s\n",
           "Time(s)", "Plants", "Herbivores", "Eats/500f");
    printf("%-8s %-10s %-12s %-10s\n",
           "-------", "------", "----------", "---------");

    float simTime  = 0.0f;
    float dt       = 0.016f;
    float eatRange = 5.0f;     // was 15.0f - must be closer to eat
    int   totalEats = 0;

    for (int frame = 0; frame < 6000; frame++) {
        cudaMemset(d_births, 0, sizeof(int));
        cudaMemset(d_eats,   0, sizeof(int));

        // 1. Sunlight feeds plants
        sunlightUpdate<<<blocks, threads>>>(c, dt);

        // 2. Build spatial hash of plants only
        cudaMemset(d_cellCounts, 0, GRID_CELLS * sizeof(int));
        buildHashCounts<<<blocks, threads>>>(c, d_cellCounts, SPECIES_PLANT);
        cudaMemset(d_cellCounts, 0, GRID_CELLS * sizeof(int));
        buildHashAssign<<<blocks, threads>>>(c, d_cellCounts,
                                             d_cellParticles, SPECIES_PLANT);

        // 3. Herbivores eat nearby plants
        herbivoresEat<<<blocks, threads>>>(c, d_cellCounts,
                                           d_cellParticles, eatRange, d_eats);

        // 4. Movement
        movementUpdate<<<blocks, threads>>>(c, dt, simTime);

        // 5. Metabolism drains energy, kills starved creatures
        metabolismUpdate<<<blocks, threads>>>(c, dt);

        // 6. Reproduction
        plantReproduction<<<blocks, threads>>>(c, d_births);
        herbivoreReproduction<<<blocks, threads>>>(c, d_births);

        simTime += dt;

        int eats;
        cudaMemcpy(&eats, d_eats, sizeof(int), cudaMemcpyDeviceToHost);
        totalEats += eats;

        if (frame % 500 == 499) {
            countSpecies(c, &p, &h);
            printf("%-8.1f %-10d %-12d %-10d\n",
                   simTime, p, h, totalEats);
            totalEats = 0;

            // Stop if herbivores extinct
            if (h == 0) {
                printf("\nHerbivores extinct at t=%.1fs\n", simTime);
                break;
            }
        }
    }

    printf("\nFinal state:\n");
    countSpecies(c, &p, &h);
    printf("  Plants: %d   Herbivores: %d\n", p, h);

    cudaFree(d_cellCounts);
    cudaFree(d_cellParticles);
    cudaFree(d_births);
    cudaFree(d_eats);
    freeCreatures(c);
    return 0;
}