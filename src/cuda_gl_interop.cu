#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <stdio.h>
#include <math.h>

extern "C" {
    __declspec(dllexport) unsigned long NvOptimusEnablement = 1;
    __declspec(dllexport) unsigned long AmdPowerXpressRequestHighPerformance = 1;
}

#define NUM_PARTICLES 1000000
#define WORLD_SIZE    1000.0f
#define DT            0.016f

// ─── Shaders ────────────────────────────────────────────────────────────────

const char* vertSrc = R"(
#version 330 core
layout (location = 0) in vec2 position;
layout (location = 1) in vec2 velocity;

uniform vec2 worldSize;

out vec3 vColor;

void main() {
    vec2 clip = (position / worldSize) * 2.0 - 1.0;
    gl_Position  = vec4(clip, 0.0, 1.0);
    gl_PointSize = 2.0;

    // Color by speed: slow=blue, fast=orange-white
    float speed = length(velocity) / 150.0;
    speed = clamp(speed, 0.0, 1.0);
    vColor = mix(vec3(0.1, 0.4, 1.0), vec3(1.0, 0.7, 0.2), speed);
}
)";

const char* fragSrc = R"(
#version 330 core
in  vec3 vColor;
out vec4 fragColor;

void main() {
    vec2  coord = gl_PointCoord - vec2(0.5);
    float dist  = length(coord);
    if (dist > 0.5) discard;

    float alpha = 1.0 - (dist * 2.0);
    fragColor = vec4(vColor, alpha * 0.9);
}
)";

// ─── CUDA Kernels ────────────────────────────────────────────────────────────

// Each particle: [px, py, vx, vy] interleaved in one buffer
// This matches OpenGL's vertex attribute layout

__global__ void initParticles(float* buf, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    float t     = (float)i / (float)n;
    float angle = t * 6.28318f * 25.0f;
    float radius = t * WORLD_SIZE * 0.45f;
    float cx = WORLD_SIZE * 0.5f;
    float cy = WORLD_SIZE * 0.5f;

    // Position: spiral galaxy
    buf[i * 4 + 0] = cx + radius * cosf(angle);  // px
    buf[i * 4 + 1] = cy + radius * sinf(angle);  // py

    // Velocity: tangential (orbiting)
    float speed = 20.0f + t * 80.0f;
    buf[i * 4 + 2] = -sinf(angle) * speed;       // vx
    buf[i * 4 + 3] =  cosf(angle) * speed;       // vy
}

__global__ void updateParticles(float* buf, int n, float dt, float time) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    float px = buf[i * 4 + 0];
    float py = buf[i * 4 + 1];
    float vx = buf[i * 4 + 2];
    float vy = buf[i * 4 + 3];

    // Gravity toward center - creates orbital motion
    float cx  = WORLD_SIZE * 0.5f;
    float cy  = WORLD_SIZE * 0.5f;
    float dx  = cx - px;
    float dy  = cy - py;
    float dist = sqrtf(dx*dx + dy*dy) + 1.0f; // +1 avoids divide by zero

    float force = 8000.0f / (dist * dist);
    vx += (dx / dist) * force * dt;
    vy += (dy / dist) * force * dt;

    // Slight drag to keep things stable
    vx *= 0.999f;
    vy *= 0.999f;

    // Update position
    px += vx * dt;
    py += vy * dt;

    // Wrap at boundaries
    if (px < 0)          px += WORLD_SIZE;
    if (px > WORLD_SIZE) px -= WORLD_SIZE;
    if (py < 0)          py += WORLD_SIZE;
    if (py > WORLD_SIZE) py -= WORLD_SIZE;

    buf[i * 4 + 0] = px;
    buf[i * 4 + 1] = py;
    buf[i * 4 + 2] = vx;
    buf[i * 4 + 3] = vy;
}

// ─── OpenGL helpers ──────────────────────────────────────────────────────────

GLuint compileShader(GLenum type, const char* src) {
    GLuint s = glCreateShader(type);
    glShaderSource(s, 1, &src, NULL);
    glCompileShader(s);
    GLint ok; glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
    if (!ok) { char log[512]; glGetShaderInfoLog(s, 512, NULL, log); printf("Shader error: %s\n", log); }
    return s;
}

GLuint createProgram(const char* v, const char* f) {
    GLuint p = glCreateProgram();
    GLuint vs = compileShader(GL_VERTEX_SHADER, v);
    GLuint fs = compileShader(GL_FRAGMENT_SHADER, f);
    glAttachShader(p, vs); glAttachShader(p, fs);
    glLinkProgram(p);
    GLint ok; glGetProgramiv(p, GL_LINK_STATUS, &ok);
    if (!ok) { char log[512]; glGetProgramInfoLog(p, 512, NULL, log); printf("Link error: %s\n", log); }
    glDeleteShader(vs); glDeleteShader(fs);
    return p;
}

void keyCallback(GLFWwindow* w, int key, int, int action, int) {
    if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
        glfwSetWindowShouldClose(w, true);
}

// ─── Main ────────────────────────────────────────────────────────────────────

int main() {
    // Init GLFW + OpenGL
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    GLFWwindow* window = glfwCreateWindow(1280, 720,
        "CUDA ALife Engine - Live Physics", NULL, NULL);
    glfwMakeContextCurrent(window);
    glfwSetKeyCallback(window, keyCallback);
    glfwSwapInterval(0);

    gladLoadGL();
    printf("Renderer: %s\n\n", glGetString(GL_RENDERER));

    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE);
    glEnable(GL_PROGRAM_POINT_SIZE);

    // Build shaders
    GLuint program    = createProgram(vertSrc, fragSrc);
    GLint  worldSzLoc = glGetUniformLocation(program, "worldSize");

    // Create VBO — will hold [px,py,vx,vy] for every particle
    // stride = 4 floats per particle
    size_t bufBytes = NUM_PARTICLES * 4 * sizeof(float);

    GLuint VAO, VBO;
    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);
    glBindVertexArray(VAO);
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, bufBytes, NULL, GL_DYNAMIC_DRAW);

    // Attribute 0: position (px, py) — offset 0
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE,
                          4 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);

    // Attribute 1: velocity (vx, vy) — offset 2 floats
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE,
                          4 * sizeof(float), (void*)(2 * sizeof(float)));
    glEnableVertexAttribArray(1);

    // Register VBO with CUDA — this is the interop magic
    cudaGraphicsResource* cudaVBO;
    cudaGraphicsGLRegisterBuffer(&cudaVBO, VBO,
                                 cudaGraphicsMapFlagsNone);

    // Map buffer, init particles with CUDA, unmap
    float* d_buf;
    size_t mappedBytes;
    cudaGraphicsMapResources(1, &cudaVBO, 0);
    cudaGraphicsResourceGetMappedPointer((void**)&d_buf, &mappedBytes, cudaVBO);

    int threads = 256;
    int blocks  = (NUM_PARTICLES + threads - 1) / threads;
    initParticles<<<blocks, threads>>>(d_buf, NUM_PARTICLES);
    cudaDeviceSynchronize();

    cudaGraphicsUnmapResources(1, &cudaVBO, 0);

    printf("Particles initialised via CUDA.\n");
    printf("Running live simulation... Press ESC to close.\n\n");

    // Timing
    int    frameCount = 0;
    double lastTime   = glfwGetTime();
    float  simTime    = 0.0f;

    cudaEvent_t cudaStart, cudaStop;
    cudaEventCreate(&cudaStart);
    cudaEventCreate(&cudaStop);
    float cudaMs = 0;

    while (!glfwWindowShouldClose(window)) {
        // ── CUDA physics update ──────────────────────────────────────────
        cudaGraphicsMapResources(1, &cudaVBO, 0);
        cudaGraphicsResourceGetMappedPointer((void**)&d_buf, &mappedBytes, cudaVBO);

        cudaEventRecord(cudaStart);
        updateParticles<<<blocks, threads>>>(d_buf, NUM_PARTICLES, DT, simTime);
        cudaEventRecord(cudaStop);
        cudaEventSynchronize(cudaStop);
        cudaEventElapsedTime(&cudaMs, cudaStart, cudaStop);

        cudaGraphicsUnmapResources(1, &cudaVBO, 0);

        simTime += DT;

        // ── OpenGL render ────────────────────────────────────────────────
        glClearColor(0.02f, 0.02f, 0.08f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);

        glUseProgram(program);
        glUniform2f(worldSzLoc, WORLD_SIZE, WORLD_SIZE);
        glBindVertexArray(VAO);
        glDrawArrays(GL_POINTS, 0, NUM_PARTICLES);

        glfwSwapBuffers(window);
        glfwPollEvents();

        // ── FPS + timing ─────────────────────────────────────────────────
        frameCount++;
        double now = glfwGetTime();
        if (now - lastTime >= 1.0) {
            printf("FPS: %4d  |  CUDA physics: %.3f ms  |  Particles: 1M\n",
                   frameCount, cudaMs);
            frameCount = 0;
            lastTime   = now;
        }
    }

    // Cleanup
    cudaGraphicsUnregisterResource(cudaVBO);
    cudaEventDestroy(cudaStart);
    cudaEventDestroy(cudaStop);
    glDeleteVertexArrays(1, &VAO);
    glDeleteBuffers(1, &VBO);
    glDeleteProgram(program);
    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}