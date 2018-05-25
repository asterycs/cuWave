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
  GLModel(std::vector<Triangle> triangles, std::vector<Material> materials, std::vector<std::vector<uint32_t>> materialIds, const std::string& fileName);
  ~GLModel();

  GLuint getVaoID() const;
  GLuint getIndexID() const;

  const std::vector<uint32_t>& getMeshSizes() const;
  const std::vector<Material>& getMaterials() const;
  const std::string getFileName() const;

private:
  std::vector<uint32_t> meshSizes;
  std::vector<Material> materials;
  std::string fileName;

  GLuint vaoID;
  GLuint vboID;
  GLuint indexID;
};

#endif
