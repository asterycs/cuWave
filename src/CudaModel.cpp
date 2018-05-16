#include "CudaModel.hpp"

#include <numeric>
#include <memory>
#include <stack>
#include <cmath>


CudaModel::CudaModel() : nTriangles(0)
{
  
}

CudaModel::~CudaModel()
{

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

	triangleMaterialIds.push_back(nMaterials-2);
	triangleMaterialIds.push_back(nMaterials-2);

	lightTriangles.push_back(triangles.size()-1);
	lightTriangles.push_back(triangles.size()-1);

	rebuild();
}


CudaModel::CudaModel(std::vector<Triangle> triangles, std::vector<Material> materials, std::vector<uint32_t> triMatIds, std::vector<uint32_t> lightTriangles, const std::string& fileName) : fileName(fileName)
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
  this->nMaterials = materials.size();
  std::cout << "Done!" << std::endl;

  nMaterials = materials.size();
  nTriangles = triangles.size();
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

  nTriangles = triangles.size();
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
  return nTriangles;
}

uint32_t CudaModel::getNLights() const
{
  return lightTriangles.size();
}

const uint32_t* CudaModel::getDeviceLightIds() const
{
  return thrust::raw_pointer_cast(&lightTriangles[0]);
}
