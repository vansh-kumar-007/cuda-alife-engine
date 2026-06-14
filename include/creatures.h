#pragma once
#include <cuda_runtime.h>

// Maximum number of creatures the engine supports
// Sized for 6GB VRAM — each creature uses ~80 bytes
#define MAX_CREATURES  1000000
#define WORLD_SIZE_F   1000.0f

// Species definitions
#define SPECIES_PLANT      0
#define SPECIES_HERBIVORE  1
#define SPECIES_PREDATOR   2
#define NUM_SPECIES        3

// Creature states
#define STATE_DEAD  0
#define STATE_ALIVE 1

// ─────────────────────────────────────────────────────────────────────────────
// CreatureArrays: Structure of Arrays layout for GPU efficiency
//
// Every field is a separate flat array of length MAX_CREATURES.
// Thread i owns element i across all arrays.
//
// Memory layout on GPU:
//   pos_x:    [x0, x1, x2, ... x999999]   <- all x positions contiguous
//   pos_y:    [y0, y1, y2, ... y999999]   <- all y positions contiguous
//   ... etc
//
// When warp of 32 threads reads pos_x[i..i+31], it's one memory transaction.
// This is why SoA beats AoS by 2-4x on GPU.
// ─────────────────────────────────────────────────────────────────────────────
struct CreatureArrays {
    // Spatial
    float* pos_x;       // world x position
    float* pos_y;       // world y position
    float* vel_x;       // velocity x
    float* vel_y;       // velocity y

    // Biology
    float* energy;      // current energy (0 = dead)
    float* age;         // age in simulation seconds
    float* size;        // physical size (affects energy cost + sensing)

    // Classification
    int*   species;     // SPECIES_PLANT / HERBIVORE / PREDATOR
    int*   state;       // STATE_ALIVE / STATE_DEAD

    // Rendering hint (packed color for OpenGL)
    float* color_r;
    float* color_g;
    float* color_b;

    int count;          // total slots (alive + dead)
};

// ─────────────────────────────────────────────────────────────────────────────
// Energy constants — tuned so populations self-regulate
// ─────────────────────────────────────────────────────────────────────────────
#define ENERGY_START_PLANT      80.0f
#define ENERGY_START_HERBIVORE  60.0f
#define ENERGY_START_PREDATOR   50.0f

#define ENERGY_MAX              100.0f

// Cost per second just to exist (metabolism)
#define ENERGY_COST_PLANT       0.5f
#define ENERGY_COST_HERBIVORE   1.2f
#define ENERGY_COST_PREDATOR    1.8f

// Cost per unit of velocity (movement is expensive)
#define ENERGY_MOVE_COST        0.002f

// Gain when eating
#define ENERGY_GAIN_HERBIVORE   25.0f   // herbivore eats plant
#define ENERGY_GAIN_PREDATOR    35.0f   // predator eats herbivore

// Reproduce when energy exceeds this threshold
#define ENERGY_REPRODUCE        85.0f

// Die when energy drops to zero
#define ENERGY_DEATH            0.0f

// ─────────────────────────────────────────────────────────────────────────────
// GPU allocation / deallocation helpers (defined in creatures.cu)
// ─────────────────────────────────────────────────────────────────────────────
void allocCreatures(CreatureArrays& c, int count);
void freeCreatures(CreatureArrays& c);
void printCreatureStats(CreatureArrays& c); // copies small sample to CPU