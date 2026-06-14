# CUDA ALife Engine

A large-scale Artificial Life simulation engine built from scratch in CUDA C++,
simulating millions of evolving organisms with neural-network brains, genetics,
energy systems, and emergent food chains — all running on GPU in real time.

**Target scale:** 1,000,000+ organisms  
**Hardware:** NVIDIA RTX 3050 Laptop (Ampere, 6GB VRAM)  
**Tech stack:** CUDA 13.3 · C++17 · OpenGL 3.3 · GLFW · CMake  

---

## Current Status

| Phase | Description | Status |
|-------|-------------|--------|
| 0 | Environment setup — CUDA 13.3, MSVC, CMake, Git | ✅ Complete |
| 1 | CUDA fundamentals — kernels, memory, events, benchmarking | ✅ Complete |
| 2 | GPU particle simulation — 1M particles at 3776 FPS, spatial hashing | ✅ Complete |
| 3 | Real-time visualization — CUDA-OpenGL interop, live physics | ✅ Complete |
| 4 | Creature architecture — energy, age, species tags | 🔄 In progress |
| 5 | Food chains and energy systems | ⏳ Planned |
| 6 | DNA and genetics | ⏳ Planned |
| 7 | Neural network brains — neuroevolution | ⏳ Planned |
| 8 | Species emergence and phylogenetics | ⏳ Planned |
| 9 | Performance optimization — Nsight profiling | ⏳ Planned |
| 10 | Large-scale experiments and portfolio demo | ⏳ Planned |

---

## Performance Benchmarks (RTX 3050 6GB Laptop GPU)

| System | Count | Metric | Result |
|--------|-------|--------|--------|
| CUDA particle physics | 1,000,000 | Kernel time | 0.26 ms |
| CUDA particle physics | 1,000,000 | Simulated FPS | 3,776 |
| GPU spatial hash | 100,000 | Pipeline time | 1.32 ms |
| GPU spatial hash | 100,000 | Queries/second | 75M |
| CUDA-OpenGL interop | 1,000,000 | Render FPS | 490 |
| CUDA-OpenGL interop | 1,000,000 | Physics kernel | 0.22 ms |

---

## Architecture
cuda-alife-engine/

├── src/

│   ├── hello_cuda.cu          # Phase 1: first CUDA kernel

│   ├── performance_test.cu    # Phase 1: GPU timing with CUDA events

│   ├── particle_sim.cu        # Phase 2: 1M particle physics simulation

│   ├── spatial_hash.cu        # Phase 2: GPU spatial hashing for neighbor queries

│   ├── window_test.cpp        # Phase 3: OpenGL window + NVIDIA GPU selection

│   ├── particle_render.cpp    # Phase 3: 1M static particles rendered via OpenGL

│   └── cuda_gl_interop.cu     # Phase 3: CUDA+OpenGL interop, live physics

├── external/

│   ├── glfw/                  # GLFW windowing library

│   └── glad/                  # OpenGL function loader

├── include/                   # Engine headers (Phase 4+)

├── shaders/                   # GLSL shaders (Phase 4+)

├── docs/                      # Architecture documentation

├── CMakeLists.txt

└── rebuild.bat
---

## How It Works

### CUDA-OpenGL Interop Pipeline
Every frame:

1. CUDA maps the OpenGL VBO → gets raw GPU pointer
2. CUDA physics kernel updates 1M positions (0.22ms)
3. CUDA unmaps buffer → ownership returns to OpenGL
4. OpenGL renders 1M points from same buffer (no copy)
5. Total: ~2ms per frame → 490 FPS

### Key CUDA Concepts Used
- Thread hierarchy: threads → blocks → grids
- Structure of Arrays (SoA) for coalesced memory access
- CUDA Events for microsecond-precision GPU timing
- cudaGraphicsGLRegisterBuffer for zero-copy GPU sharing
- Atomic operations for spatial hash construction

---

## Building

**Requirements:** CUDA Toolkit 13.x · Visual Studio Build Tools 2026 · CMake 4.x · Windows 11

```cmd
# Open Developer Command Prompt for VS 2026
git clone https://github.com/vansh-kumar-007/cuda-alife-engine.git
cd cuda-alife-engine
rebuild.bat

# Run the live simulation
build\cuda_gl_interop.exe
```

---

## Roadmap

This project is being built phase by phase through summer 2026.  
Each phase introduces new GPU programming concepts and biological systems.  
The final engine will simulate 1M+ organisms with evolving neural-network brains.

---

**Author:** Vansh Kumar  
[GitHub](https://github.com/vansh-kumar-007) · Delhi Technological University