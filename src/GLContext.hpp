#ifndef GLCONTEXT_HPP
#define GLCONTEXT_HPP

#include <GL/glew.h>
#include <GL/gl.h>
#include <GLFW/glfw3.h>

#include <glm/glm.hpp>

#include <ft2build.h>
#include FT_FREETYPE_H

#include "Camera.hpp"
#include "GLTexture.hpp"
#include "GLShader.hpp"
#include "UI.hpp"
#include "GLModel.hpp"



class GLContext
{
public:
  GLContext();
  ~GLContext();

  GLContext& operator=(GLContext& other) = default;
  GLContext(GLContext& other) = default;

  void draw(const GLTexture& canvas);
  void draw(const GLModel& model, const Camera& camera);

  void drawUI();
  bool UiWantsMouseInput();
  void resize(const glm::ivec2& newSize);
  bool shadersLoaded() const;
  
  void showWindow();

  glm::ivec2 getSize() const;

  void clear();
  void swapBuffers();
  bool isAlive();
  
  float getDTime();
  glm::ivec2 getCursorPos();
  bool isKeyPressed(const int glfwKey, const int modifiers) const;
  
  float getTime() const;

private:
  void updateUniformMat4f(const glm::mat4& mat, const std::string& identifier);
  void updateUniform3fv(const glm::vec3& vec, const std::string& identifier);

  GLuint loadShader(const char *vertex_path, const char *fragment_path);
  GLShader canvasShader;
  GLShader modelShader;
  GLFWwindow* window;

  glm::ivec2 size;

  UI ui;
};

#endif // GLCONTEXT_HPP
