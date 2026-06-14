#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include "creatures.h"
#include <stdio.h>
#include <math.h>

extern "C" {
    __declspec(dllexport) unsigned long NvOptimusEnablement = 1;
    __declspec(dllexport) unsigned long AmdPowerXpressRequestHighPerformance = 1;
}

// ─── Shaders ─────────────────────────────────────────────────────────────────

const char* vertSrc = R"(
#version 330 core
layout (location = 0) in vec2 position;
layout (location = 1) in vec3 color;

uniform vec2 worldSize;

out vec3 vColor;

void main() {
    vec2 clip = (position / worldSize) * 2.0 - 1.0;
    gl_Position  = vec4(clip, 0.0, 1.0);
    gl_PointSize = 2.5;
    vColor = color;
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
    fragColor = vec4(vColor, alpha);
}
)";

// ─── Render buffer layout ─────────────────────────────────────────────────────
// Each creature in the VBO: [px, py, r, g, b] = 5 floats
// position attribute: offset 0,  stride 5
// color    attribute: offset 2,  stride 5

__global__ void initCreaturesViz(
    float* renderBuf,       // VBO mapped by CUDA
    CreatureArrays c)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= c.count) return;

    int third = c.count / 3;
    int sp;
    if      (i < third)       sp = SPECIES_PLANT;
    else if (i < third * 2)   sp = SPECIES_HERBIVORE;
    else                      sp = SPECIES_PREDATOR;

    c.species[i] = sp;
    c.state[i]   = STATE_ALIVE;
    c.age[i]     = 0.0f;

    float t = (float)i / (float)c.count;
    c.pos_x[i]  = WORLD_SIZE_F * (0.1f + 0.8f * fabsf(sinf(t * 2137.0f)));
    c.pos_y[i]  = WORLD_SIZE_F * (0.1f + 0.8f * fabsf(cosf(t * 3571.0f)));
    c.vel_x[i]  = 0.0f;
    c.vel_y[i]  = 0.0f;

    if (sp == SPECIES_PLANT) {
        c.energy[i] = ENERGY_START_PLANT;
        c.size[i]   = 1.5f;
        c.color_r[i] = 0.1f; c.color_g[i] = 0.9f; c.color_b[i] = 0.1f;
    } else if (sp == SPECIES_HERBIVORE) {
        c.energy[i] = ENERGY_START_HERBIVORE;
        c.size[i]   = 2.5f;
        c.color_r[i] = 0.2f; c.color_g[i] = 0.5f; c.color_b[i] = 1.0f;
    } else {
        c.energy[i] = ENERGY_START_PREDATOR;
        c.size[i]   = 4.0f;
        c.color_r[i] = 1.0f; c.color_g[i] = 0.2f; c.color_b[i] = 0.1f;
    }

    // Write initial render buffer
    renderBuf[i * 5 + 0] = c.pos_x[i];
    renderBuf[i * 5 + 1] = c.pos_y[i];
    renderBuf[i * 5 + 2] = c.color_r[i];
    renderBuf[i * 5 + 3] = c.color_g[i];
    renderBuf[i * 5 + 4] = c.color_b[i];
}

__global__ void behaviorUpdateViz(CreatureArrays c, float time) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= c.count) return;
    if (c.state[i] == STATE_DEAD) return;

    float t = (float)i / (float)c.count;

    if (c.species[i] == SPECIES_PLANT) {
        c.vel_x[i] = 0.0f;
        c.vel_y[i] = 0.0f;
    } else if (c.species[i] == SPECIES_HERBIVORE) {
        float angle = t * 6.283f + time * (0.3f + t * 0.4f);
        float speed = 80.0f + t * 40.0f;
        c.vel_x[i] = cosf(angle) * speed;
        c.vel_y[i] = sinf(angle) * speed;
    } else {
        float angle = t * 6.283f + time * (0.6f + t * 0.5f);
        float speed = 150.0f + t * 50.0f;
        c.vel_x[i] = cosf(angle) * speed;
        c.vel_y[i] = sinf(angle) * speed;
    }
}

__global__ void updateCreaturesViz(
    float* renderBuf,
    CreatureArrays c,
    float dt)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= c.count) return;

    if (c.state[i] == STATE_DEAD) {
        // Hide dead creatures off screen
        renderBuf[i * 5 + 0] = -9999.0f;
        renderBuf[i * 5 + 1] = -9999.0f;
        return;
    }

    c.age[i] += dt;

    float cost = 0.0f;
    if      (c.species[i] == SPECIES_PLANT)     cost = ENERGY_COST_PLANT;
    else if (c.species[i] == SPECIES_HERBIVORE) cost = ENERGY_COST_HERBIVORE;
    else                                         cost = ENERGY_COST_PREDATOR;

    float speed = sqrtf(c.vel_x[i]*c.vel_x[i] + c.vel_y[i]*c.vel_y[i]);
    cost += speed * ENERGY_MOVE_COST;
    c.energy[i] -= cost * dt;

    if (c.energy[i] <= ENERGY_DEATH) {
        c.state[i]  = STATE_DEAD;
        c.energy[i] = 0.0f;
        renderBuf[i * 5 + 0] = -9999.0f;
        renderBuf[i * 5 + 1] = -9999.0f;
        return;
    }

    c.pos_x[i] += c.vel_x[i] * dt;
    c.pos_y[i] += c.vel_y[i] * dt;
    if (c.pos_x[i] < 0)            c.pos_x[i] += WORLD_SIZE_F;
    if (c.pos_x[i] > WORLD_SIZE_F) c.pos_x[i] -= WORLD_SIZE_F;
    if (c.pos_y[i] < 0)            c.pos_y[i] += WORLD_SIZE_F;
    if (c.pos_y[i] > WORLD_SIZE_F) c.pos_y[i] -= WORLD_SIZE_F;

    // Write updated position and energy-tinted color to render buffer
    float energyRatio = c.energy[i] / ENERGY_MAX;
    renderBuf[i * 5 + 0] = c.pos_x[i];
    renderBuf[i * 5 + 1] = c.pos_y[i];
    renderBuf[i * 5 + 2] = c.color_r[i] * energyRatio;
    renderBuf[i * 5 + 3] = c.color_g[i] * energyRatio;
    renderBuf[i * 5 + 4] = c.color_b[i] * energyRatio;
}

// ─── OpenGL helpers ───────────────────────────────────────────────────────────

GLuint compileShader(GLenum type, const char* src) {
    GLuint s = glCreateShader(type);
    glShaderSource(s, 1, &src, NULL);
    glCompileShader(s);
    GLint ok; glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
    if (!ok) { char log[512]; glGetShaderInfoLog(s,512,NULL,log); printf("Shader: %s\n",log); }
    return s;
}

GLuint createProgram(const char* v, const char* f) {
    GLuint p  = glCreateProgram();
    GLuint vs = compileShader(GL_VERTEX_SHADER, v);
    GLuint fs = compileShader(GL_FRAGMENT_SHADER, f);
    glAttachShader(p, vs); glAttachShader(p, fs);
    glLinkProgram(p);
    glDeleteShader(vs); glDeleteShader(fs);
    return p;
}

void keyCallback(GLFWwindow* w, int key, int, int action, int) {
    if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
        glfwSetWindowShouldClose(w, true);
}

// ─── Main ─────────────────────────────────────────────────────────────────────

int main() {
    // Init window
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    GLFWwindow* window = glfwCreateWindow(1280, 720,
        "CUDA ALife - Species Visualization", NULL, NULL);
    glfwMakeContextCurrent(window);
    glfwSetKeyCallback(window, keyCallback);
    glfwSwapInterval(0);
    gladLoadGL();

    printf("Renderer: %s\n", glGetString(GL_RENDERER));

    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glEnable(GL_PROGRAM_POINT_SIZE);

    GLuint program   = createProgram(vertSrc, fragSrc);
    GLint  worldLoc  = glGetUniformLocation(program, "worldSize");

    // Allocate creature state arrays
    CreatureArrays creatures;
    allocCreatures(creatures, 300000);

    // Create VBO: 5 floats per creature [px, py, r, g, b]
    size_t vboBytes = creatures.count * 5 * sizeof(float);
    GLuint VAO, VBO;
    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);
    glBindVertexArray(VAO);
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, vboBytes, NULL, GL_DYNAMIC_DRAW);

    // position: location 0, 2 floats, stride 5
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE,
        5*sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);

    // color: location 1, 3 floats, offset 2
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE,
        5*sizeof(float), (void*)(2*sizeof(float)));
    glEnableVertexAttribArray(1);

    // Register VBO with CUDA
    cudaGraphicsResource* cudaVBO;
    cudaGraphicsGLRegisterBuffer(&cudaVBO, VBO, cudaGraphicsMapFlagsNone);

    // Init creatures via CUDA
    int threads = 256;
    int blocks  = (creatures.count + threads - 1) / threads;

    float* d_buf; size_t mappedBytes;
    cudaGraphicsMapResources(1, &cudaVBO, 0);
    cudaGraphicsResourceGetMappedPointer((void**)&d_buf, &mappedBytes, cudaVBO);
    initCreaturesViz<<<blocks, threads>>>(d_buf, creatures);
    cudaDeviceSynchronize();
    cudaGraphicsUnmapResources(1, &cudaVBO, 0);

    printf("300K creatures initialised.\n");
    printf("Green=Plants  Blue=Herbivores  Red=Predators\n");
    printf("Watch populations collapse. ESC to close.\n\n");

    int    frameCount = 0;
    double lastTime   = glfwGetTime();
    float  simTime    = 0.0f;

    while (!glfwWindowShouldClose(window)) {
        // CUDA update
        cudaGraphicsMapResources(1, &cudaVBO, 0);
        cudaGraphicsResourceGetMappedPointer((void**)&d_buf, &mappedBytes, cudaVBO);

        behaviorUpdateViz<<<blocks, threads>>>(creatures, simTime);
        updateCreaturesViz<<<blocks, threads>>>(d_buf, creatures, 0.016f);

        cudaGraphicsUnmapResources(1, &cudaVBO, 0);
        simTime += 0.016f;

        // Render
        glClearColor(0.02f, 0.02f, 0.06f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);
        glUseProgram(program);
        glUniform2f(worldLoc, WORLD_SIZE_F, WORLD_SIZE_F);
        glBindVertexArray(VAO);
        glDrawArrays(GL_POINTS, 0, creatures.count);

        glfwSwapBuffers(window);
        glfwPollEvents();

        frameCount++;
        double now = glfwGetTime();
        if (now - lastTime >= 2.0) {
            // Count alive per species
            int* h_state   = new int[creatures.count];
            int* h_species = new int[creatures.count];
            cudaMemcpy(h_state,   creatures.state,
                       creatures.count*sizeof(int), cudaMemcpyDeviceToHost);
            cudaMemcpy(h_species, creatures.species,
                       creatures.count*sizeof(int), cudaMemcpyDeviceToHost);

            int alive[3] = {0,0,0};
            for (int i = 0; i < creatures.count; i++)
                if (h_state[i] == STATE_ALIVE) alive[h_species[i]]++;

            printf("FPS:%4d | t=%.0fs | Plants:%6d  Herbs:%6d  Preds:%6d\n",
                   frameCount/2, simTime,
                   alive[0], alive[1], alive[2]);

            delete[] h_state;
            delete[] h_species;
            frameCount = 0;
            lastTime   = now;
        }
    }

    cudaGraphicsUnregisterResource(cudaVBO);
    freeCreatures(creatures);
    glDeleteVertexArrays(1, &VAO);
    glDeleteBuffers(1, &VBO);
    glDeleteProgram(program);
    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}