#include "Model.hpp"

#include <numeric>
#include <memory>
#include <stack>
#include <cmath>


glm::fvec3 ai2glm3f(aiColor3D v)
{
  return glm::fvec3(v[0], v[1], v[2]);
}

Model::Model() : nTriangles(0)
{
  
}

Model::~Model()
{

}


Model::Model(const aiScene *scene, const std::string& fileName) : fileName(fileName)
{
  std::vector<Triangle> tris;
  std::vector<Material> mats;
  std::vector<unsigned int> triMatIds;

  initialize(scene, tris, mats, triMatIds);
  
  BVHBuilder bvhbuilder;
  bvhbuilder.build(tris, triMatIds);
  
  std::vector<Node> bvh = bvhbuilder.getBVH();
  tris = bvhbuilder.getTriangles();
  triMatIds = bvhbuilder.getTriangleMaterialIds();

  this->triangles = triangles;
  this->materials = materials;
  this->triangleMaterialIds = triMatIds;
  this->bvh = bvh;

  nTriangles = triangles.size();
}

void Model::initialize(const aiScene *scene, std::vector<Triangle>& triangles, std::vector<Material>& materials, std::vector<unsigned int>& triMaterialIds)
{
  std::cout << "Creating model with " << scene->mNumMeshes << " meshes" << std::endl;
  
  unsigned int triangleOffset = 0;
  
  float3 maxTri = make_float3(-999.f,-999.f,-999.f);
  float3 minTri = make_float3(999.f,999.f,999.f);

  for (std::size_t mi = 0; mi < scene->mNumMeshes; mi++)
  {
    aiMesh *mesh = scene->mMeshes[mi];

    std::vector<Vertex> vertices;
    std::vector<unsigned int> indices;

    bool createLight(false);

    if (mesh->mMaterialIndex > 0)
    {
      Material material = Material();

      aiMaterial& mat = *scene->mMaterials[mesh->mMaterialIndex];

      aiColor3D aiAmbient    (0.f,0.f,0.f);
      aiColor3D aiDiffuse    (0.f,0.f,0.f);
      aiColor3D aiSpecular   (0.f,0.f,0.f);
      aiColor3D aiEmission   (0.f,0.f,0.f);
      aiColor3D aiTransparent(0.f,0.f,0.f);

      mat.Get(AI_MATKEY_COLOR_AMBIENT,     aiAmbient);
      mat.Get(AI_MATKEY_COLOR_DIFFUSE,     aiDiffuse);
      mat.Get(AI_MATKEY_COLOR_SPECULAR,    aiSpecular);
      mat.Get(AI_MATKEY_COLOR_EMISSIVE,    aiEmission);
      mat.Get(AI_MATKEY_COLOR_TRANSPARENT, aiTransparent);

      mat.Get(AI_MATKEY_REFRACTI,          material.refrIdx);
      mat.Get(AI_MATKEY_SHININESS,         material.shininess);

      material.colorAmbient     = glm32cuda3(ai2glm3f(aiAmbient));
      material.colorDiffuse     = glm32cuda3(ai2glm3f(aiDiffuse));
      material.colorEmission    = glm32cuda3(ai2glm3f(aiEmission));
      material.colorSpecular    = glm32cuda3(ai2glm3f(aiSpecular));
      material.colorTransparent = glm32cuda3(glm::sqrt(glm::fvec3(1.f) - ai2glm3f(aiTransparent)));

      if (material.colorEmission != make_float3(0.f, 0.f, 0.f))
    	createLight = true;

      int sm;
      mat.Get(AI_MATKEY_SHADING_MODEL, sm);

      switch (sm)
      {
      case aiShadingMode_Gouraud:
        material.shadingMode = material.GORAUD;
        break;
      case aiShadingMode_Fresnel:
        material.shadingMode = material.FRESNEL;
        break;
      default:
        material.shadingMode = material.PHONG;
      }


      std::vector<unsigned int> vertexIds(mesh->mNumFaces * 3);
      std::iota(vertexIds.begin(), vertexIds.end(), triangleOffset * 3);

      materials.push_back(material);
      triangleOffset += mesh->mNumFaces;
    }else
      continue;

    std::vector<unsigned int> lightIds;

    for (std::size_t vi = 0; vi < mesh->mNumVertices; vi++)
    {
      Vertex newVertex;
      auto& oldVertex = mesh->mVertices[vi];

      newVertex.p = make_float3(oldVertex.x, oldVertex.y, oldVertex.z);

      if (mesh->HasNormals())
      {
        auto& oldNormal = mesh->mNormals[vi];
        newVertex.n = make_float3(oldNormal.x, oldNormal.y, oldNormal.z);
      }

      if (mesh->mTextureCoords[0])
      {
        auto& tc = mesh->mTextureCoords[0][vi];
        newVertex.t = make_float2(tc.x, tc.y);
      }

      vertices.push_back(newVertex);
    }

    for (std::size_t i = 0; i < mesh->mNumFaces; ++i)
    {
      aiFace& face = mesh->mFaces[i];

      if (face.mNumIndices == 3)
      {
        Triangle triangle = Triangle(vertices[face.mIndices[0]], vertices[face.mIndices[1]], vertices[face.mIndices[2]]);

        if (createLight)
          lightIds.push_back(triangles.size());

        triangles.push_back(triangle);

        maxTri = fmaxf(maxTri, triangle.max());
        minTri = fminf(minTri, triangle.min());
      }

      triMaterialIds.push_back(materials.size() - 1);
    }

    if (createLight)
      lights.push_back(Light(lightIds));
  }
}

const Triangle* Model::getDeviceTriangles() const
{
  return thrust::raw_pointer_cast(&triangles[0]);
}

const Material* Model::getDeviceMaterials() const
{
  return thrust::raw_pointer_cast(&materials[0]);
}

const unsigned int* Model::getDeviceTriangleMaterialIds() const
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

const Light* Model::getDeviceLights() const
{
  return thrust::raw_pointer_cast(&lights[0]);
}
