#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>

extern "C" { __declspec(dllexport) unsigned long NvOptimusEnablement = 1; }
extern "C" { __declspec(dllexport) unsigned long AmdPowerXpressRequestHighPerformance = 1; }

#define NUM_PARTICLES 1000000
#define WORLD_SIZE    1000.0f

// Vertex shader: converts world position to screen position
// gl_Position is the built-in output — where the point appears on screen
// gl_PointSize controls how large each point is drawn
const char* vertexShaderSrc = R"(
#version 330 core
layout (location = 0) in vec2 position;

uniform vec2 worldSize;

void main() {
    // Convert from world space (0..1000) to clip space (-1..1)
    vec2 clip = (position / worldSize) * 2.0 - 1.0;
    gl_Position = vec4(clip, 0.0, 1.0);
    gl_PointSize = 1.5;
}
)";

// Fragment shader: colors each particle
// We color by position — creates a beautiful gradient across the world
const char* fragmentShaderSrc = R"(
#version 330 core
out vec4 fragColor;

void main() {
    // Soft circular point (distance from center of point sprite)
    vec2 coord = gl_PointCoord - vec2(0.5);
    float dist = length(coord);
    if (dist > 0.5) discard; // clip to circle shape

    // Glow effect: brighter in center, fades at edge
    float alpha = 1.0 - (dist * 2.0);
    fragColor = vec4(0.2, 0.8, 1.0, alpha * 0.8); // cyan-blue glow
}
)";

// Compile a shader and check for errors
GLuint compileShader(GLenum type, const char* src) {
    GLuint shader = glCreateShader(type);
    glShaderSource(shader, 1, &src, NULL);
    glCompileShader(shader);

    GLint success;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
    if (!success) {
        char log[512];
        glGetShaderInfoLog(shader, 512, NULL, log);
        printf("Shader compile error: %s\n", log);
    }
    return shader;
}

// Link vertex + fragment shaders into a program
GLuint createShaderProgram(const char* vertSrc, const char* fragSrc) {
    GLuint vert = compileShader(GL_VERTEX_SHADER,   vertSrc);
    GLuint frag = compileShader(GL_FRAGMENT_SHADER, fragSrc);

    GLuint program = glCreateProgram();
    glAttachShader(program, vert);
    glAttachShader(program, frag);
    glLinkProgram(program);

    GLint success;
    glGetProgramiv(program, GL_LINK_STATUS, &success);
    if (!success) {
        char log[512];
        glGetProgramInfoLog(program, 512, NULL, log);
        printf("Shader link error: %s\n", log);
    }

    glDeleteShader(vert);
    glDeleteShader(frag);
    return program;
}

void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods) {
    if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
        glfwSetWindowShouldClose(window, true);
}

void framebufferSizeCallback(GLFWwindow* window, int width, int height) {
    glViewport(0, 0, width, height);
}

int main() {
    // Init GLFW
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    GLFWwindow* window = glfwCreateWindow(1280, 720,
        "CUDA ALife Engine - 1M Particles", NULL, NULL);
    glfwMakeContextCurrent(window);
    glfwSetKeyCallback(window, keyCallback);
    glfwSetFramebufferSizeCallback(window, framebufferSizeCallback);
    glfwSwapInterval(0); // disable vsync - we want max FPS

    gladLoadGL();

    printf("Renderer: %s\n", glGetString(GL_RENDERER));

    // Enable blending for glow effect
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE); // additive blending = glow

    // Enable variable point sizes from vertex shader
    glEnable(GL_PROGRAM_POINT_SIZE);

    // Build shader program
    GLuint shaderProgram = createShaderProgram(vertexShaderSrc, fragmentShaderSrc);
    GLint worldSizeLoc = glGetUniformLocation(shaderProgram, "worldSize");

    // Generate 1 million particle positions on CPU
    printf("Generating %d particles...\n", NUM_PARTICLES);
    float* positions = (float*)malloc(NUM_PARTICLES * 2 * sizeof(float)); // x,y pairs

    for (int i = 0; i < NUM_PARTICLES; i++) {
        float t = (float)i / NUM_PARTICLES;
        // Spiral galaxy pattern - looks impressive on first render
        float angle  = t * 6.28318f * 20.0f;
        float radius = t * WORLD_SIZE * 0.45f;
        float cx = WORLD_SIZE * 0.5f;
        float cy = WORLD_SIZE * 0.5f;

        positions[i * 2 + 0] = cx + radius * cosf(angle) + ((i % 7) - 3) * 2.0f;
        positions[i * 2 + 1] = cy + radius * sinf(angle) + ((i % 5) - 2) * 2.0f;
    }

    // Upload to GPU via Vertex Buffer Object
    GLuint VAO, VBO;
    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);

    glBindVertexArray(VAO);
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER,
                 NUM_PARTICLES * 2 * sizeof(float),
                 positions,
                 GL_DYNAMIC_DRAW);

    // Tell OpenGL: attribute 0 = 2 floats (x, y) per vertex
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);

    free(positions); // CPU copy no longer needed - data is on GPU now
    printf("Particles uploaded to GPU.\n");
    printf("Rendering... Press ESC to close.\n\n");

    // Render loop
    int   frameCount = 0;
    double lastTime  = glfwGetTime();

    while (!glfwWindowShouldClose(window)) {
        // Dark background
        glClearColor(0.02f, 0.02f, 0.08f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);

        // Draw all particles
        glUseProgram(shaderProgram);
        glUniform2f(worldSizeLoc, WORLD_SIZE, WORLD_SIZE);
        glBindVertexArray(VAO);
        glDrawArrays(GL_POINTS, 0, NUM_PARTICLES);

        glfwSwapBuffers(window);
        glfwPollEvents();

        frameCount++;
        double now = glfwGetTime();
        if (now - lastTime >= 1.0) {
            printf("FPS: %d  |  Particles: %dM\n",
                   frameCount, NUM_PARTICLES / 1000000);
            frameCount = 0;
            lastTime   = now;
        }
    }

    // Cleanup
    glDeleteVertexArrays(1, &VAO);
    glDeleteBuffers(1, &VBO);
    glDeleteProgram(shaderProgram);
    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}