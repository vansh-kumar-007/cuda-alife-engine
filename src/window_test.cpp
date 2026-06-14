#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <stdio.h>

// Force Windows to use the NVIDIA GPU instead of integrated graphics
extern "C" { __declspec(dllexport) unsigned long NvOptimusEnablement = 1; }
extern "C" { __declspec(dllexport) unsigned long AmdPowerXpressRequestHighPerformance = 1; }

// Called whenever the window is resized
void framebufferSizeCallback(GLFWwindow* window, int width, int height) {
    glViewport(0, 0, width, height);
}

// Called whenever a key is pressed
void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods) {
    if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
        glfwSetWindowShouldClose(window, true);
}

int main() {
    // Initialise GLFW
    if (!glfwInit()) {
        printf("ERROR: Failed to initialise GLFW\n");
        return -1;
    }

    // Tell GLFW we want OpenGL 3.3 Core Profile
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    // Create the window
    GLFWwindow* window = glfwCreateWindow(
        1280, 720,
        "CUDA ALife Engine - Phase 3",
        NULL, NULL
    );

    if (!window) {
        printf("ERROR: Failed to create GLFW window\n");
        glfwTerminate();
        return -1;
    }

    glfwMakeContextCurrent(window);
    glfwSetFramebufferSizeCallback(window, framebufferSizeCallback);
    glfwSetKeyCallback(window, keyCallback);

    // Enable vsync
    glfwSwapInterval(1);

    // Load OpenGL function pointers via GLAD
    if (!gladLoadGL()) {
        printf("ERROR: Failed to load OpenGL functions via GLAD\n");
        return -1;
    }

    // Print what we got
    printf("OpenGL Version:  %s\n", glGetString(GL_VERSION));
    printf("GPU Renderer:    %s\n", glGetString(GL_RENDERER));
    printf("GLSL Version:    %s\n", glGetString(GL_SHADING_LANGUAGE_VERSION));
    printf("\nWindow open! Press ESC to close.\n");

    // Main loop
    int frameCount = 0;
    double lastTime = glfwGetTime();

    while (!glfwWindowShouldClose(window)) {
        // Clear screen to a dark blue-green
        glClearColor(0.05f, 0.05f, 0.15f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);

        // Swap front and back buffers
        glfwSwapBuffers(window);

        // Poll keyboard/mouse events
        glfwPollEvents();

        // Print FPS every second
        frameCount++;
        double now = glfwGetTime();
        if (now - lastTime >= 1.0) {
            printf("FPS: %d\n", frameCount);
            frameCount = 0;
            lastTime = now;
        }
    }

    printf("Window closed cleanly.\n");
    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}