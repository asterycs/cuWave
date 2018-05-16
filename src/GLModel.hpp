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

  const std::vector<GLuint>& getIndexBuffers() const;
  const std::vector<Material>& getMaterials() const;
  const std::string getFileName() const;

private:
  std::vector<GLuint> indexBuffers;
  std::vector<Material> materials;
  std::string fileName;

  GLuint vaoID;
  GLuint vboID;
};

#endif
