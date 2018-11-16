#ifndef GLMODEL_HPP
#define GLMODEL_HPP

#include "Utils.hpp"
#include "Triangle.hpp"

#include <GL/glew.h>
#include <GL/gl.h>

class GLModel
{
public:
  GLModel();
  GLModel(const AbstractModel& abstractModel);
  ~GLModel();

  GLuint getVaoID() const;
  GLuint getIndexID() const;

  const std::vector<uint32_t>& getMeshSizes() const;
  const std::vector<Material>& getMaterials() const;

private:
  std::vector<uint32_t> meshSizes_;
  std::vector<Material> materials_;

  GLuint vaoID;
  GLuint vboID;
  GLuint indexID;
};

#endif
