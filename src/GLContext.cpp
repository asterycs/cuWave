#include "GLContext.hpp"
#include "App.hpp"
#include "Utils.hpp"

#include <imgui.h>

glm::fvec3 cudaf32glmf3(const float3 in)
{
	return glm::fvec3(in.x, in.y, in.z);
}

GLContext::GLContext() :
  canvasShader(),
  modelShader(),
  window(nullptr),
  size(WWIDTH, WHEIGHT),
  ui()
{
  if (!glfwInit())
  {
    std::cerr << "GLFW init failed" << std::endl;
  }

  glfwSetTime(0);

  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 2);
  glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

  glfwWindowHint(GLFW_VISIBLE, GL_FALSE);
  glfwWindowHint(GLFW_SAMPLES, 0);
  glfwWindowHint(GLFW_RED_BITS, 8);
  glfwWindowHint(GLFW_GREEN_BITS, 8);
  glfwWindowHint(GLFW_BLUE_BITS, 8);
  glfwWindowHint(GLFW_ALPHA_BITS, 8);
  glfwWindowHint(GLFW_STENCIL_BITS, 8);
  glfwWindowHint(GLFW_DEPTH_BITS, 24);
  glfwWindowHint(GLFW_RESIZABLE, GL_TRUE);
  glfwWindowHint(GLFW_SAMPLES, 8);

  window = glfwCreateWindow(size.x, size.y, "cuWave", nullptr, nullptr);
  if (window == nullptr)
    throw std::runtime_error("Failed to create GLFW window");

  GL_CHECK(glfwMakeContextCurrent(window));

  glewExperimental = GL_TRUE;
  GLenum err = glewInit();
  if(err!=GLEW_OK) {
    throw std::runtime_error("glewInit failed");
  }

  // Defuse bogus error
  glGetError();

  ui.init(window);

  GL_CHECK(glClearColor(0.2f, 0.25f, 0.3f, 1.0f));
  GL_CHECK(glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT));

  int width, height;
  GL_CHECK(glfwGetFramebufferSize(window, &width, &height));
  GL_CHECK(glViewport(0, 0, width, height));
  glfwSwapInterval(1);
  glfwSwapBuffers(window);

  glfwSetMouseButtonCallback(window,
      [](GLFWwindow *, int button, int action, int modifiers) {
          App::getInstance().mouseCallback(button, action, modifiers);
      }
  );
  
  glfwSetScrollCallback(window,
      [](GLFWwindow *, double xOffset, double yOffset) {
          App::getInstance().scrollCallback(xOffset, yOffset);
      }
  );

  glfwSetKeyCallback(window,
      [](GLFWwindow *, int key, int scancode, int action, int mods) {
          App::getInstance().keyboardCallback(key, scancode, action, mods);
      }
  );

  glfwSetWindowSizeCallback(window,
      [](GLFWwindow *, int width, int height) {
          App::getInstance().resizeCallbackEvent(width, height);
      }
  );

  glfwSetErrorCallback([](int error, const char* description) {
    std::cout << "Error: " << error << " " << description << std::endl;
  });

  canvasShader.loadShader("shaders/canvas/vshader.glsl", "shaders/canvas/fshader.glsl");
  modelShader.loadShader("shaders/model/vshader.glsl", "shaders/model/fshader.glsl");
  
  std::cout << "OpenGL context initialized" << std::endl;
}

GLContext::~GLContext()
{
  glfwDestroyWindow(window);
  glfwTerminate();
}

bool GLContext::shadersLoaded() const
{
  if (canvasShader.isLoaded() && modelShader.isLoaded())
    return true;
  else
    return false;
}

void GLContext::clear()
{
  if (!shadersLoaded())
    return;
    
  GL_CHECK(glClearColor(0.2f, 0.25f, 0.3f, 1.0f));
  GL_CHECK(glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT));
  GL_CHECK(glEnable(GL_DEPTH_TEST));

  glfwPollEvents();
}

void GLContext::swapBuffers()
{
  glfwSwapBuffers(window);
}

bool GLContext::isAlive()
{
  return !glfwWindowShouldClose(window);
}

float GLContext::getDTime()
{
  return ui.getDTime();
}

float GLContext::getTime() const
{
  return (float) glfwGetTime();
}

void GLContext::drawUI()
{
  ui.draw();
}

bool GLContext::UiWantsMouseInput()
{
  ImGuiIO& io = ImGui::GetIO();

  if (io.WantCaptureMouse || io.WantMoveMouse)
    return true;
  else
    return false;
}

void GLContext::showWindow()
{
  glfwShowWindow(window);
}

glm::ivec2 GLContext::getCursorPos()
{
  double x, y;
  glfwGetCursorPos(window, &x, &y);

  return glm::ivec2(x, y);
}

bool GLContext::isKeyPressed(const int glfwKey, const int modifiers) const
{
  const int ctrl  = glfwGetKey(window, GLFW_KEY_LEFT_CONTROL) || glfwGetKey(window, GLFW_KEY_RIGHT_CONTROL) << 1;
  const int shift = glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) || glfwGetKey(window, GLFW_KEY_RIGHT_SHIFT) << 0;
  const int super = glfwGetKey(window, GLFW_KEY_LEFT_SUPER) || glfwGetKey(window, GLFW_KEY_RIGHT_SUPER) << 3;
  const int alt   = glfwGetKey(window, GLFW_KEY_LEFT_ALT) || glfwGetKey(window, GLFW_KEY_RIGHT_ALT) << 2;

  const int pressedMods = shift | ctrl | alt | super;

  if (modifiers == pressedMods)
    return glfwGetKey(window, glfwKey);
  else
    return false;
}

void GLContext::resize(const glm::ivec2& newSize)
{
  GL_CHECK(glViewport(0,0,newSize.x, newSize.y));
  this->size = newSize;

  ui.resize(newSize);
}

void GLContext::draw(const GLTexture& canvas)
{
  if (!shadersLoaded() || canvas.getTextureID() == 0)
     return;

   GL_CHECK(glActiveTexture(GL_TEXTURE0));
   canvasShader.bind();
   GL_CHECK(glBindTexture(GL_TEXTURE_2D, canvas.getTextureID()));
   
   //canvasShader.getAttribLocation("texture");
   canvasShader.updateUniform1i("texture", 0);
  
   GLuint dummyVao;
   GL_CHECK(glGenVertexArrays(1, &dummyVao));
   GL_CHECK(glBindVertexArray(dummyVao));

   GL_CHECK(glDrawArrays(GL_TRIANGLES, 0, 3));

   GL_CHECK(glBindVertexArray(0));
   GL_CHECK(glDeleteVertexArrays(1, &dummyVao));
   canvasShader.unbind();
}

void GLContext::draw(const GLModel& model, const Camera& camera)
{
	if (!shadersLoaded())
		return;

	GL_CHECK(glViewport(0,0,size.x, size.y));
	GL_CHECK(glCullFace(GL_BACK));
	GL_CHECK(glBindFramebuffer(GL_FRAMEBUFFER, 0));
	GL_CHECK(glDepthFunc(GL_LEQUAL));

	const auto& vaoID = model.getVaoID();
	const auto& materialIds = model.getMaterialIds();
	const auto& materials = model.getMaterials();

	modelShader.bind();
	modelShader.updateUniformMat4f("posToCamera", camera.getMVP(size));
	//modelShader.updateUniformMat3f("normalToCamera", glm::mat3(glm::transpose(glm::inverse(camera.getMVP(size)))));

	GL_CHECK(glBindVertexArray(vaoID));

	for (std::size_t i = 0; i < materialIds.size(); ++i)
	{
		const auto material = materials[i];

		modelShader.updateUniform3fv("material.colorAmbient", cudaf32glmf3(material.colorAmbient));
		modelShader.updateUniform3fv("material.colorDiffuse", cudaf32glmf3(material.colorDiffuse));
		//modelShader.updateUniform3fv("material.colorSpecular", material.colorSpecular);

		GL_CHECK(glDrawElements(GL_TRIANGLES, materialIds[i].size(), GL_UNSIGNED_INT, materialIds[i].data()));
	}

	GL_CHECK(glBindVertexArray(0));
	modelShader.unbind();
}

glm::ivec2 GLContext::getSize() const
{
  return size;
}
