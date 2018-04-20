#ifndef MODEL_HPP
#define MODEL_HPP

#include <vector>
#include <string>
#include <memory>

#include <thrust/device_vector.h>

#include "assimp/scene.h"

#include "Utils.hpp"
#include "Triangle.hpp"
#include "BVHBuilder.hpp"
#include "Light.hpp"

class Model
{
public:
  Model();
  Model(const aiScene *scene, const std::string& fileName);
  ~Model();
  Model(Model&) = delete;
  Model(Model&&) = default;
  Model& operator=(Model&) = delete;
  Model& operator=(Model&&) = default;

  const Triangle* getDeviceTriangles() const;
  const Material* getDeviceMaterials() const;
  const unsigned int* getDeviceTriangleMaterialIds() const;
  const Light* getDeviceLights() const;
  unsigned int getNTriangles() const;
  
  const AABB& getBbox() const;
  const Node* getDeviceBVH() const;
  const std::string& getFileName() const;
private:
  void initialize(const aiScene *scene, std::vector<Triangle>& triangles, std::vector<Material>& materials, std::vector<unsigned int>& triMaterialIds);

  thrust::device_vector<Light> lights;
  thrust::device_vector<Triangle> triangles;
  thrust::device_vector<Material> materials;
  thrust::device_vector<unsigned int> triangleMaterialIds;
  unsigned int nTriangles;

  std::string fileName;

  AABB boundingBox;
  thrust::device_vector<Node> bvh;
};

#endif
