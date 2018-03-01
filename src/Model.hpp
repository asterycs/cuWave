#ifndef MODEL_HPP
#define MODEL_HPP

#include <vector>
#include <string>

#include "assimp/scene.h"

#include "Utils.hpp"
#include "Triangle.hpp"
#include "BVHBuilder.hpp"

class Model
{
public:
  Model();
  ~Model();
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

  Triangle* devTriangles;
  Material* devMaterials;
  unsigned int* devTriangleMaterialIds;
  unsigned int nTriangles;

  std::string fileName;

  AABB boundingBox;
  Node* devBVH;
};

#endif
