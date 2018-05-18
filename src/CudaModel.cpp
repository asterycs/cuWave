#include "CudaModel.hpp"

#include <numeric>
#include <memory>
#include <stack>
#include <cmath>


CudaModel::CudaModel() : addedLights(0)
{
  
}

CudaModel::~CudaModel()
{

}

void CudaModel::clearLights()
{
	for (std::size_t i = 0; i < addedLights; ++i)
	{
		triangleMaterialIds.pop_back();
		triangleMaterialIds.pop_back();

		triangles.pop_back();
		triangles.pop_back();
	}

	addedLights = 0;

	rebuild();
}

void CudaModel::addLight(const glm::mat4 tform)
{
	const glm::vec2 lightSize(0.15f, 0.15f);

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
		triangles.push_back(Triangle(vertices[i*3], vertices[i*3+1], vertices[i*3+2]));

	triangleMaterialIds.push_back(materials.size()-2);
	triangleMaterialIds.push_back(materials.size()-2);

	lightTriangles.push_back(triangles.size()-1);
	lightTriangles.push_back(triangles.size()-1);

	addedLights += 2;

	rebuild();
}

void CudaModel::addLights(const std::vector<Triangle>& lightTriangles, const std::vector<uint32_t> materialIds)
{
	for (std::size_t i = 0; i < lightTriangles.size(); ++i)
	{
		triangles.push_back(lightTriangles[i]);
		triangleMaterialIds.push_back(materialIds[i]);
	}

	addedLights += lightTriangles.size();

	rebuild();
}


CudaModel::CudaModel(std::vector<Triangle> triangles, std::vector<Material> materials, std::vector<uint32_t> triMatIds, std::vector<uint32_t> lightTriangles, const std::string& fileName) : addedLights(0), fileName(fileName)
{
  std::cout << "Building BVH..." << std::endl;
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

  std::cout << "Done!" << std::endl;

}

void CudaModel::rebuild()
{
  std::cout << "Building BVH..." << std::endl;
  std::vector<Triangle> newTris(triangles.begin(), triangles.end());
  std::vector<uint32_t > newTriMatIds(triangleMaterialIds.begin(), triangleMaterialIds.end());
  std::vector<uint32_t > newLightTriangles(lightTriangles.begin(), lightTriangles.end());

  BVHBuilder bvhbuilder;
  bvhbuilder.build(newTris, newTriMatIds, newLightTriangles);

  this->triangles = bvhbuilder.getTriangles();
  this->triangleMaterialIds = bvhbuilder.getTriangleMaterialIds();
  this->lightTriangles = bvhbuilder.getLightTriangleIds();
  this->bvh = bvhbuilder.getBVH();
  std::cout << "Done!" << std::endl;
}

const Triangle* CudaModel::getDeviceTriangles() const
{
  return thrust::raw_pointer_cast(&triangles[0]);
}

const Material* CudaModel::getDeviceMaterials() const
{
  return thrust::raw_pointer_cast(&materials[0]);
}

const uint32_t* CudaModel::getDeviceTriangleMaterialIds() const
{
  return thrust::raw_pointer_cast(&triangleMaterialIds[0]);
}

const std::string& CudaModel::getFileName() const
{
  return fileName;
}

const AABB& CudaModel::getBbox() const
{
  return this->boundingBox;
}

const Node* CudaModel::getDeviceBVH() const
{
  return thrust::raw_pointer_cast(&bvh[0]);
}

uint32_t CudaModel::getNTriangles() const
{
  return triangles.size();
}

uint32_t CudaModel::getNLights() const
{
  return lightTriangles.size();
}

const uint32_t* CudaModel::getDeviceLightIds() const
{
  return thrust::raw_pointer_cast(&lightTriangles[0]);
}

uint32_t CudaModel::getNAddedLights() const
{
	return addedLights;
}

thrust::host_vector<Triangle> CudaModel::getTriangles() const
{
	return triangles;
}

thrust::host_vector<uint32_t> CudaModel::getLightIds() const
{
	return triangleMaterialIds;
}

thrust::host_vector<uint32_t> CudaModel::getTriangleMaterialIds() const
{
	return triangleMaterialIds;
}
