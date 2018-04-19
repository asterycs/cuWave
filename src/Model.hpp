#ifndef MODEL_HPP
#define MODEL_HPP

#include <vector>
#include <string>
#include <memory>

#include "assimp/scene.h"

#include "Utils.hpp"
#include "Triangle.hpp"
#include "BVHBuilder.hpp"

struct CudaDeleter
{
  void operator() (void *ptr) const
  {
    cudaFree(ptr);
  }
};

class Model
{
public:
  Model();
  ~Model();
  Model(Model&) = delete;
  Model(Model&&) = default;
  Model& operator=(Model&) = delete;
  Model& operator=(Model&&) = default;

  Model(const aiScene *scene, const std::string& fileName);
  const Triangle* getDeviceTriangles() const;
  const Material* getDeviceMaterials() const;
  const unsigned int* getDeviceTriangleMaterialIds() const;
  unsigned int getNTriangles() const;
  
  const AABB& getBbox() const;
  const Node* getDeviceBVH() const;
  const std::string& getFileName() const;
private:
  void initialize(const aiScene *scene, std::vector<Triangle>& triangles, std::vector<Material>& materials, std::vector<unsigned int>& triMaterialIds);

  std::unique_ptr<Triangle, CudaDeleter> devTriangles;
  std::unique_ptr<Material, CudaDeleter> devMaterials;
  std::unique_ptr<unsigned int, CudaDeleter> devTriangleMaterialIds;
  unsigned int nTriangles;

  std::string fileName;

  AABB boundingBox;
  std::unique_ptr<Node, CudaDeleter> devBVH;
};

#endif
