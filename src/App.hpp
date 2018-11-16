#ifndef APP_HPP
#define APP_HPP

#include "ModelLoader.hpp"
#include "GLContext.hpp"
#include "GLTexture.hpp"

#include "CudaRenderer.hpp"

#define LAST_SCENEFILE_NAME "last.scene"

enum ActiveRenderer
{
	OpenGL,
	CUDA
};

class App { 
public:
    // Singleton
    App(App const&) = delete;
    void operator=(App& app) = delete;
    static App& getInstance() {static App app; return app;}

    void showWindow();

    void MainLoop();
    void drawDebugInfo();

    void mouseCallback(int button, int action, int modifiers);
    void scrollCallback(double xOffset, double yOffset);
    void keyboardCallback(int key, int scancode, int action, int modifiers);
    void resizeCallbackEvent(int width, int height);
    void initProjection(int width, int height, float near, float far);
    void handleControl(float dTime);
    void addLight();
    void createSceneFile(const std::string& path);
    void loadSceneFile(const std::string& path);
    void loadModel(const std::string& path);
    void writeTextureToFile(const GLTexture& texture, const std::string& path);

#ifdef ENABLE_CUDA
    void rayTraceToFile(const std::string& sceneFile, const std::string& outFile);
    void pathTraceToFile(const std::string& sceneFile, const std::string& outFile, const int iterations);
#endif
private:
    App();
    ~App();

    ActiveRenderer activeRenderer;

    bool mousePressed;
    glm::dvec2 mousePrevPos;

    GLContext glcontext;
    CudaRenderer cudaRenderer;

    CudaModel cudaModel;
    GLModel glModel;
    GLTexture glcanvas;
    
    std::string modelPath;
    Camera camera;
    ModelLoader loader;
};

#endif
