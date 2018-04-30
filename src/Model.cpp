#include "Model.hpp"

#include <numeric>
#include <memory>
#include <stack>
#include <cmath>


Model::Model() : nTriangles(0)
{
  
}

Model::~Model()
{

}


Model::Model(std::vector<Triangle> triangles, std::vector<Material> materials, std::vector<uint32_t> triMatIds, std::vector<uint32_t> lightTriangles, const std::string& fileName) : fileName(fileName)
{
  BVHBuilder bvhbuilder;
  bvhbuilder.build(triangles, triMatIds, lightTriangles);
  
  std::vector<Node> bvh = bvhbuilder.getBVH();
  std::vector<Triangle> newTriangles = bvhbuilder.getTriangles();
  std::vector<uint32_t> newTriMatIds = bvhbuilder.getTriangleMaterialIds();
  std::vector<uint32_t> newLightTriangles = bvhbuilder.getLightTriangleIds();

  this->triangles = newTriangles;
  this->materials = materials;
  this->triangleMaterialIds = newTriMatIds;
  this->lightTriangles = newLightTriangles;
  this->bvh = bvh;

  nTriangles = triangles.size();
}

const Triangle* Model::getDeviceTriangles() const
{
  return thrust::raw_pointer_cast(&triangles[0]);
}

const Material* Model::getDeviceMaterials() const
{
  return thrust::raw_pointer_cast(&materials[0]);
}

const uint32_t* Model::getDeviceTriangleMaterialIds() const
{
  return thrust::raw_pointer_cast(&triangleMaterialIds[0]);
}

const std::string& Model::getFileName() const
{
  return fileName;
}

const AABB& Model::getBbox() const
{
  return this->boundingBox;
}

const Node* Model::getDeviceBVH() const
{
  return thrust::raw_pointer_cast(&bvh[0]);
}

unsigned int Model::getNTriangles() const
{
  return nTriangles;
}

const uint32_t* Model::getDeviceLights() const
{
  return thrust::raw_pointer_cast(&lightTriangles[0]);
}
