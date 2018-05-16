#ifndef MODELLOADER_HPP
#define MODELLOADER_HPP

#include "CudaModel.hpp"
#include "GLModel.hpp"

class ModelLoader
{
public:
  ModelLoader();
  ~ModelLoader();
  
  bool loadOBJ(const std::string& path, std::vector<Triangle>& triangles, std::vector<uint32_t>& triangleMaterialIds, std::vector<uint32_t>& lightTriangles, std::vector<Material>& materials, std::vector<std::vector<uint32_t>>& materialIds) const;
  CudaModel loadCudaModel(const std::string& path) const;
  GLModel loadGLModel(const std::string& path) const;

};

#endif // SCENELOADER_HPP
