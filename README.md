# CUDA ALife Engine

A large-scale Artificial Life simulation engine built from scratch in CUDA C++,
running on GPU with real-time visualization. Simulates millions of evolving
organisms with neural-network brains, genetics, energy systems, and emergent
food chains.

**Target scale:** 1,000,000+ organisms  
**Hardware:** NVIDIA RTX 3050 (Ampere, 6GB VRAM)  
**Tech stack:** CUDA 13.3 · C++17 · CMake · OpenGL  

---

## Current Status

| Phase | Description | Status |
|-------|-------------|--------|
| 0 | Environment setup — CUDA 13.3, MSVC, CMake, Git | ✅ Complete |
| 1 | CUDA fundamentals — kernels, memory, performance measurement | ✅ Complete |
| 2 | GPU particle simulation — 1M particles at 3776 FPS, spatial hashing | ✅ Complete |
| 3 | Real-time visualization — OpenGL + CUDA interop | 🔄 In progress |
| 4 | Creature architecture — energy, age, species | ⏳ Planned |
| 5 | Food chains and energy systems | ⏳ Planned |
| 6 | DNA and genetics | ⏳ Planned |
| 7 | Neural network brains — neuroevolution | ⏳ Planned |
| 8 | Species emergence and phylogenetics | ⏳ Planned |
| 9 | Performance optimization — Nsight profiling | ⏳ Planned |
| 10 | Large-scale experiments and portfolio | ⏳ Planned |

---

## Performance Benchmarks (RTX 3050 6GB Laptop)

| System | Count | Time per frame | FPS equivalent |
|--------|-------|---------------|----------------|
| Particle physics | 1,000,000 | 0.26 ms | 3,776 |
| Spatial neighbor hash | 100,000 | 1.32 ms | 757 |

---

## Architecture

cuda-alife-engine/

├── src/

│   ├── hello_cuda.cu        # Phase 1: first kernel

│   ├── performance_test.cu  # Phase 1: CUDA events timing

│   ├── particle_sim.cu      # Phase 2: 1M particle physics

│   └── spatial_hash.cu      # Phase 2: GPU spatial hashing

├── include/                 # Headers (Phase 3+)

├── shaders/                 # GLSL shaders (Phase 3+)

├── docs/                    # Architecture documentation

├── CMakeLists.txt           # CMake build system

└── rebuild.bat              # Clean rebuild script


---

## Building

Requires: CUDA Toolkit 13.x · Visual Studio Build Tools 2026 · CMake 4.x

```cmd
# Open Developer Command Prompt for VS 2026
git clone https://github.com/vansh-kumar-007/cuda-alife-engine.git
cd cuda-alife-engine
rebuild.bat
build\particle_sim.exe
build\spatial_hash.exe
```

---

## About

Built as a deep-dive into GPU programming, parallel algorithms, and artificial
life research. Every phase teaches a new layer of CUDA — from basic kernels
through shared memory, occupancy optimization, and CUDA-OpenGL interop.

**Author:** Vansh Kumar · [GitHub](https://github.com/vansh-kumar-007)