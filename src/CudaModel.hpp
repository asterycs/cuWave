#ifndef CUDAMODEL_HPP
#define CUDAMODEL_HPP

#include <vector>
#include <string>
#include <memory>

#include <thrust/device_vector.h>

#include "glm/glm.hpp"

#include "Utils.hpp"
#include "Triangle.hpp"
#include "BVHBuilder.hpp"

class CudaModel
{
public:
  CudaModel();
  CudaModel(const AbstractModel& abstractModel);
  ~CudaModel();
  CudaModel(CudaModel&) = delete;
  CudaModel(CudaModel&&) = default;
  CudaModel& operator=(CudaModel&) = delete;
  CudaModel& operator=(CudaModel&&) = default;

  void clearLights();
  void addLight(const glm::mat4 tform);
  void addLights(const std::vector<Triangle>& lightTriangles, const std::vector<uint32_t> materialIds);
  void rebuild();

  uint32_t getNAddedLights() const;

  thrust::host_vector<Triangle> getTriangles() const;
  thrust::host_vector<uint32_t> getLightIds() const;
  thrust::host_vector<uint32_t> getTriangleMaterialIds() const;

  const Triangle* getDeviceTriangles() const;
  const Material* getDeviceMaterials() const;
  const uint32_t* getDeviceTriangleMaterialIds() const;
  const uint32_t* getDeviceLightIds() const;
  uint32_t  getNLights() const;
  uint32_t getNTriangles() const;
  
  const AABB& getBbox() const;
  const Node* getDeviceBVH() const;
private:
  uint32_t addedLights_;

  thrust::device_vector<Triangle> triangles_;
  thrust::device_vector<Material> materials_;
  thrust::device_vector<uint32_t> triangleMaterialIds_;
  thrust::device_vector<uint32_t> lightTriangles_;

  std::string fileName_;

  AABB boundingBox_;
  thrust::device_vector<Node> bvh_;
};

#endif
