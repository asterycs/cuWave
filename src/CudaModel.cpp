#include "CudaModel.hpp"

#include <numeric>
#include <memory>
#include <stack>
#include <cmath>


CudaModel::CudaModel() : addedLights_(0), triangles_(), materials_(), triangleMaterialIds_(), lightTriangles_(), fileName_(),
boundingBox_(), bvh_()
{
  
}

CudaModel::~CudaModel()
{

}

void CudaModel::clearLights()
{
	for (std::size_t i = 0; i < addedLights_; ++i)
	{
		triangleMaterialIds_.pop_back();
		triangleMaterialIds_.pop_back();

		triangles_.pop_back();
		triangles_.pop_back();

	  lightTriangles_.pop_back();
    lightTriangles_.pop_back();
	}

	addedLights_ = 0;
	rebuild();
}

void CudaModel::addLight(const glm::mat4 tform)
{
	const glm::vec2 lightSize(0.02f, 0.02f);

	const std::vector<glm::vec4> contour {
		glm::vec4(-lightSize.x, lightSize.y, 0.0, 1.0),
		glm::vec4(lightSize.x, -lightSize.y, 0.0, 1.0),
		glm::vec4(-lightSize.x, -lightSize.y, 0.0, 1.0),

		glm::vec4(lightSize.x, lightSize.y, 0.0, 1.0),
		glm::vec4(lightSize.x, -lightSize.y, 0.0, 1.0),
		glm::vec4(-lightSize.x, lightSize.y, 0.0, 1.0),
	};

	std::vector<Vertex> vertices;

	for (auto c : contour)
	{
		const glm::mat3 N = glm::transpose(glm::inverse(glm::mat3(tform))); // Normal to world

		Vertex v;
		v.p = glm42float3(tform * c);
		v.n = glm32float3(N * glm::vec3(0.0f, 0.0f, -1.f));
		vertices.push_back(v);
	}

	for (unsigned int i = 0; i < 2; ++i)
		triangles_.push_back(Triangle(vertices[i*3], vertices[i*3+1], vertices[i*3+2]));

	Material lightMaterial;
	lightMaterial.colorEmission = make_float3(600.f, 600.f, 600.f);
	lightMaterial.colorDiffuse = make_float3(1.f, 1.f, 1.f);

	materials_.push_back(lightMaterial);

	const uint32_t lightMaterialId = materials_.size()-1;
	triangleMaterialIds_.push_back(lightMaterialId);
	triangleMaterialIds_.push_back(lightMaterialId);

	lightTriangles_.push_back(triangles_.size()-1);
	lightTriangles_.push_back(triangles_.size()-1);

	addedLights_ += 1;

	rebuild();
}

void CudaModel::addLights(const std::vector<Triangle>& lightTriangles, const std::vector<uint32_t> materialIds)
{
	for (std::size_t i = 0; i < lightTriangles.size(); ++i)
	{
		triangles_.push_back(lightTriangles[i]);
		triangleMaterialIds_.push_back(materialIds[i]);
	}

	addedLights_ += lightTriangles.size();

	rebuild();
}


CudaModel::CudaModel(const AbstractModel& abstractModel) : addedLights_(0)
{
  std::cout << "Building BVH..." << std::endl;
  BVHBuilder bvhbuilder;
  bvhbuilder.build(abstractModel.triangles, abstractModel.triangleMaterialIds, abstractModel.lightTriangleIds);
  
  std::vector<Node> bvh = bvhbuilder.getBVH();
  std::vector<Triangle> newTriangles = bvhbuilder.getTriangles();
  std::vector<uint32_t> newTriMatIds = bvhbuilder.getTriangleMaterialIds();
  std::vector<uint32_t> newLightTriangles = bvhbuilder.getLightTriangleIds();

  std::cout << "Done!" << std::endl;

  this->triangles_ = newTriangles;
  this->materials_ = abstractModel.materials;
  this->triangleMaterialIds_ = newTriMatIds;
  this->lightTriangles_ = newLightTriangles;
  this->bvh_ = bvh;
}

void CudaModel::rebuild()
{
  std::cout << "Building BVH..." << std::endl;
  std::vector<Triangle> newTris(triangles_.begin(), triangles_.end());
  std::vector<uint32_t > newTriMatIds(triangleMaterialIds_.begin(), triangleMaterialIds_.end());
  std::vector<uint32_t > newLightTriangles(lightTriangles_.begin(), lightTriangles_.end());

  BVHBuilder bvhbuilder;
  bvhbuilder.build(newTris, newTriMatIds, newLightTriangles);

  this->triangles_ = bvhbuilder.getTriangles();
  this->triangleMaterialIds_ = bvhbuilder.getTriangleMaterialIds();
  this->lightTriangles_ = bvhbuilder.getLightTriangleIds();
  this->bvh_ = bvhbuilder.getBVH();
  std::cout << "Done!" << std::endl;
}

const Triangle* CudaModel::getDeviceTriangles() const
{
  return thrust::raw_pointer_cast(&triangles_[0]);
}

const Material* CudaModel::getDeviceMaterials() const
{
  return thrust::raw_pointer_cast(&materials_[0]);
}

const uint32_t* CudaModel::getDeviceTriangleMaterialIds() const
{
  return thrust::raw_pointer_cast(&triangleMaterialIds_[0]);
}

const AABB& CudaModel::getBbox() const
{
  return this->boundingBox_;
}

const Node* CudaModel::getDeviceBVH() const
{
  return thrust::raw_pointer_cast(&bvh_[0]);
}

uint32_t CudaModel::getNTriangles() const
{
  return triangles_.size();
}

uint32_t CudaModel::getNLights() const
{
  return lightTriangles_.size();
}

const uint32_t* CudaModel::getDeviceLightIds() const
{
  return thrust::raw_pointer_cast(&lightTriangles_[0]);
}

uint32_t CudaModel::getNAddedLights() const
{
	return addedLights_;
}

thrust::host_vector<Triangle> CudaModel::getTriangles() const
{
	return triangles_;
}

thrust::host_vector<uint32_t> CudaModel::getLightIds() const
{
	return triangleMaterialIds_;
}

thrust::host_vector<uint32_t> CudaModel::getTriangleMaterialIds() const
{
	return triangleMaterialIds_;
}
