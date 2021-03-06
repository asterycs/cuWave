#include "App.hpp"

#include <iostream>
#include <string>
#include <fstream>
#include <vector>
#include <exception>
#include <sstream>

#include "nfd.h"

#define ILUT_USE_OPENGL
#include <IL/il.h>
#include <IL/ilu.h>
#include <IL/ilut.h>

App::App() :
	activeRenderer(OpenGL),
    mousePressed(false),
    mousePrevPos(glcontext.getCursorPos()),
    glcontext(),
#ifdef ENABLE_CUDA
    cudaRenderer(),
#endif
    cudaModel(),
    glModel(),
    glcanvas(glm::ivec2(WWIDTH, WHEIGHT)),
    camera(),
    loader()
{
  ilInit();
  iluInit();
}

App::~App()
{

}

void App::resizeCallbackEvent(int width, int height)
{
  int newWidth = width;
  int newHeight = height;

  const glm::ivec2 newSize = glm::ivec2(newWidth, newHeight);

  glcontext.resize(newSize);
  glcanvas.resize(newSize);
#ifdef ENABLE_CUDA
  cudaRenderer.resize(newSize);
#endif
}

void App::MainLoop()
{
  while (glcontext.isAlive())
  {

	glcontext.clear();
	float dTime = glcontext.getDTime();
	handleControl(dTime);

	switch (activeRenderer)
	{
		case CUDA:
		    cudaRenderer.pathTraceToCanvas(glcanvas, camera, cudaModel);
		    glcontext.draw(glcanvas);
		    break;

		case OpenGL:
		    glcontext.draw(glModel, camera);
		    break;

		default:
			break;
	}


    glcontext.drawUI();
    glcontext.swapBuffers();

  }

  if (cudaModel.getNTriangles() != 0) // Check if model is loaded
    createSceneFile(LAST_SCENEFILE_NAME);
}

void App::showWindow()
{
  glcontext.showWindow();
}

void App::handleControl(float dTime)
{
  // For mouse
  glm::dvec2 mousePos = glcontext.getCursorPos();

  if (!glcontext.UiWantsMouseInput())
  {
    if (mousePressed)
    {
      glm::dvec2 dir = mousePos - mousePrevPos;

      if (glm::length(dir) > 0.0)
        camera.rotate(dir, dTime);
    }
  }

  mousePrevPos = mousePos;

  if (glcontext.isKeyPressed(GLFW_KEY_W, 0))
    camera.translate(glm::vec3(0.f, 1.f, 0.f), dTime);

  if (glcontext.isKeyPressed(GLFW_KEY_S, 0))
    camera.translate(glm::vec3(0.f, -1.f, 0.f), dTime);

  if (glcontext.isKeyPressed(GLFW_KEY_A, 0))
    camera.translate(glm::vec3(1.f, 0.f, 0.f), dTime);

  if (glcontext.isKeyPressed(GLFW_KEY_D, 0))
    camera.translate(glm::vec3(-1.f, 0.f, 0.f), dTime);
  
  if (glcontext.isKeyPressed(GLFW_KEY_R, 0))
    camera.translate(glm::vec3(0.f, 0.f, 1.f), dTime);

  if (glcontext.isKeyPressed(GLFW_KEY_F, 0))
    camera.translate(glm::vec3(0.f, 0.f, -1.f), dTime);

  if (glcontext.isKeyPressed(GLFW_KEY_RIGHT, 0))
    camera.rotate(glm::vec2(2.f, 0.f), dTime);

  if (glcontext.isKeyPressed(GLFW_KEY_LEFT, 0))
    camera.rotate(glm::vec2(-2.f, 0.f), dTime);
    
  if (glcontext.isKeyPressed(GLFW_KEY_UP, 0))
    camera.rotate(glm::vec2(0.f, -2.f), dTime);
    
  if (glcontext.isKeyPressed(GLFW_KEY_DOWN, 0))
    camera.rotate(glm::vec2(0.f, 2.f), dTime);
}

void App::mouseCallback(int button, int action, int /*modifiers*/)
{
  if (button == GLFW_MOUSE_BUTTON_LEFT)
  {
    if (action == GLFW_PRESS)
      mousePressed = true;
    else if (action == GLFW_RELEASE)
      mousePressed = false;
  }
}

void App::scrollCallback(double /*xOffset*/, double yOffset)
{
  if (yOffset < 0)
    camera.increaseFOV();
  else if (yOffset > 0)
    camera.decreaseFOV();
}

void App::keyboardCallback(int key, int /*scancode*/, int action, int modifiers)
{
  if (key == GLFW_KEY_ENTER && action == GLFW_PRESS)
  {
	  activeRenderer = static_cast<ActiveRenderer>((activeRenderer + 1) % 2);
  }
  else if (key == GLFW_KEY_SPACE && action == GLFW_PRESS)
  {
	if (modifiers & GLFW_MOD_CONTROL)
	{
		cudaModel.clearLights();
		cudaRenderer.reset();
	}else
		addLight();
  }
  else if (key == GLFW_KEY_O && action == GLFW_PRESS && (modifiers & GLFW_MOD_CONTROL))
  {
    std::cout << "Choose scene file" << std::endl;
    nfdchar_t *outPath = NULL;
    nfdresult_t result = NFD_OpenDialog( NULL, NULL, &outPath);

    if (result == NFD_OKAY)
    {
      loadSceneFile(outPath);
      free(outPath);
    }
  }else if (key == GLFW_KEY_S && action == GLFW_PRESS && (modifiers & GLFW_MOD_CONTROL))
  {
    std::cout << "Choose model file" << std::endl;
    nfdchar_t *outPath = NULL;
    nfdresult_t result = NFD_SaveDialog( NULL, NULL, &outPath);

    if (result == NFD_OKAY)
    {
      createSceneFile(outPath);
      free(outPath);
    }
  }else if (key == GLFW_KEY_O && action == GLFW_PRESS)
  {
    nfdchar_t *outPath = NULL;
    nfdresult_t result = NFD_OpenDialog( NULL, NULL, &outPath);

    if (result == NFD_OKAY)
    {
      std::cout << "Opening model: " << outPath << std::endl;
      loadModel(outPath);
      free(outPath);
    }
  }
  else if (key == GLFW_KEY_L && action == GLFW_PRESS)
  {
    nfdchar_t *outPath = NULL;
    nfdresult_t result = NFD_OpenDialog( NULL, NULL, &outPath);

    if (result == NFD_OKAY)
    {
      std::cout << "Loading scene file: " << outPath << std::endl;
      loadSceneFile(outPath);
      free(outPath);
    }
  }

}

void App::addLight()
{
  cudaRenderer.reset();
  const glm::mat4 v = camera.getView();
  const glm::mat4 tform = glm::inverse(v);

  cudaModel.addLight(tform);
}

void App::createSceneFile(const std::string& filename)
{
  std::ofstream sceneFile;
  sceneFile.open(filename, std::ofstream::out | std::ofstream::trunc);

  /* Order:
   *  Model filename
   *  light
   *  camera
   */

  if (!sceneFile.is_open())
  {
    std::cerr << "Couldn't write scenefile" << std::endl;
    return;
  }

  sceneFile << modelPath << std::endl;
  sceneFile << camera << std::endl;
  sceneFile << cudaModel.getNAddedLights() << std::endl;

  const thrust::host_vector<Triangle> triangles = cudaModel.getTriangles();
  const thrust::host_vector<uint32_t> lightTriangles = cudaModel.getLightIds();
  const thrust::host_vector<uint32_t> triangleMaterialIds = cudaModel.getTriangleMaterialIds();

  for (std::size_t i = 0; i < cudaModel.getNAddedLights(); ++i)
  {
	  sceneFile << triangles[triangles.size() - 1 - i] << std::endl;
	  sceneFile << triangleMaterialIds[triangleMaterialIds.size() - 1 - i] << std::endl;
  }

  sceneFile.close();

  std::cout << "Wrote scene file " << filename << std::endl;
}

void App::loadModel(const std::string& path)
{
  AbstractModel abstractModel;

  if (!loader.loadOBJ(path, abstractModel))
  {
    std::cerr << "Couldn't load model" << std::endl;
  }

  modelPath = path;

  cudaModel = CudaModel(abstractModel);
  cudaRenderer.reset();
  glModel = GLModel(abstractModel);
}

void App::loadSceneFile(const std::string& filename)
{
  if (filename.find(".scene") == std::string::npos) {
    std::cout << "Invalid .scene file selected" << std::endl;
    return;
  }

  std::ifstream sceneFile;
  sceneFile.open(filename);

  /* Order:
   *  Model filename
   *  light
   *  camera
   */

  if (!sceneFile.is_open())
  {
    std::cerr << "Couldn't open scenefile" << std::endl;
    return;
  }

  std::string modelName;
  std::getline(sceneFile, modelName);
  loadModel(modelName);

  sceneFile >> camera;

  uint32_t additionalLights;
  sceneFile >> additionalLights;

  std::vector<Triangle> additionalLightTriangles;
  std::vector<uint32_t> additionalLightMaterialIds;

  for (std::size_t i = 0; i < additionalLights; ++i)
  {
	  Triangle lightTriangle;
	  uint32_t materialId;

	  sceneFile >> lightTriangle;
	  sceneFile >> materialId;

	  additionalLightTriangles.push_back(lightTriangle);
	  additionalLightMaterialIds.push_back(materialId);
  }

  cudaModel.addLights(additionalLightTriangles, additionalLightMaterialIds);

  sceneFile.close();

  std::cout << "Loaded scene file " << filename << std::endl;
}

void App::pathTraceToFile(const std::string& sceneFile, const std::string& outfile, const int iterations)
{
  loadSceneFile(sceneFile);

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start);

  for (int i = 0; i < iterations; ++i)
  {
	std::cout << "Iteration: " << i+1 << "/" << iterations << "  \r" << std::flush;
    cudaRenderer.pathTraceToCanvas(glcanvas, camera, cudaModel);
  }

  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  float millis = 0;
  cudaEventElapsedTime(&millis, start, stop);

  writeTextureToFile(glcanvas, outfile);
  std::cout << "Rendering time [ms]: " << millis << std::endl;

  return;
}

void App::writeTextureToFile(const GLTexture& texture, const std::string& fileName)
{
  ILuint imgID;

  IL_CHECK(ilGenImages(1, &imgID));
  IL_CHECK(ilBindImage(imgID));

  const glm::ivec2 size = texture.getSize();

  // Direct copy of the texture does not seem to work. Must pass it via a PBO.
  // Some others seem to experience the same: https://devtalk.nvidia.com/default/topic/913544/glgetteximage-gives-blank-result-after-cuda-surface-write/
  GLuint  pbo;
  GL_CHECK(glGenBuffers(1, &pbo));
  GL_CHECK(glBindBuffer(GL_PIXEL_PACK_BUFFER, pbo));
  GL_CHECK(glBufferData(GL_PIXEL_PACK_BUFFER, size.x*size.y*3*sizeof(float), nullptr, GL_STREAM_COPY));

  GL_CHECK(glBindTexture(GL_TEXTURE_2D, texture.getTextureID()));

  GL_CHECK(glGetTexImage(GL_TEXTURE_2D, 0, GL_RGB, GL_FLOAT, NULL));
  void *mem = glMapBuffer(GL_PIXEL_PACK_BUFFER, GL_READ_ONLY);

  // When asking OpenGL kindly for GL_UNSIGNED_BYTE data with glGetTexImage the result is just zeros.
  // Workaround by asking GL_FLOAT and manually converting.
  std::vector<unsigned char> tmp(size.x*size.y*3, 255);

  for (int i = 0; i < size.x*size.y*3; ++i)
    tmp[i] *= static_cast<unsigned char>(*(static_cast<float*>(mem) + i));

  IL_CHECK(ilTexImage(size.x, size.y, 1, 3, IL_RGB, IL_UNSIGNED_BYTE, tmp.data()));

  IL_CHECK(ilEnable(IL_FILE_OVERWRITE));
  IL_CHECK(ilSaveImage(fileName.c_str()));

  GL_CHECK(glDeleteBuffers(1, &pbo));
  IL_CHECK(ilDeleteImages(1, &imgID));
  GL_CHECK(glBindTexture(GL_TEXTURE_2D, 0));
  GL_CHECK(glBindBuffer(GL_PIXEL_PACK_BUFFER, 0));
}

