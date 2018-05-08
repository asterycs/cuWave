#ifndef MODEL_HPP
#define MODEL_HPP

#include <vector>
#include <string>
#include <memory>

#include <thrust/device_vector.h>

#include "Utils.hpp"
#include "Triangle.hpp"
#include "BVHBuilder.hpp"

class Model
{
public:
  Model();
  Model(std::vector<Triangle> triangles, std::vector<Material> materials, std::vector<unsigned int> triMatIds, std::vector<unsigned int> lightTriangles, const std::string& fileName);
  ~Model();
  Model(Model&) = delete;
  Model(Model&&) = default;
  Model& operator=(Model&) = delete;
  Model& operator=(Model&&) = default;

  void addLight(const glm::mat4 tform);
  void rebuild();

  const Triangle* getDeviceTriangles() const;
  const Material* getDeviceMaterials() const;
  const uint32_t* getDeviceTriangleMaterialIds() const;
  const uint32_t* getDeviceLightIds() const;
  uint32_t  getNLights() const;
  uint32_t getNTriangles() const;
  
  const AABB& getBbox() const;
  const Node* getDeviceBVH() const;
  const std::string& getFileName() const;
private:
  thrust::device_vector<Triangle> triangles;
  thrust::device_vector<Material> materials;
  thrust::device_vector<uint32_t> triangleMaterialIds;
  thrust::device_vector<uint32_t> lightTriangles;

  uint32_t nTriangles;
  uint32_t nMaterials;

  std::string fileName;

  AABB boundingBox;
  thrust::device_vector<Node> bvh;
};

#endif
