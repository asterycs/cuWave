#include "ModelLoader.hpp"

#include <ostream>

#define TINYOBJLOADER_IMPLEMENTATION
#include "tiny_obj_loader.h"

ModelLoader::ModelLoader()
{

}

ModelLoader::~ModelLoader()
{
}

std::ostream& operator<<(std::ostream&os, const float3& v)
{
	os << v.x << " " << v.y << " " << v.z;
	return os;
}

Model ModelLoader::loadOBJ(const std::string& path)
{
  tinyobj::attrib_t attrib;
  std::vector<tinyobj::shape_t> shapes;
  std::vector<tinyobj::material_t> tinymaterials;

  std::vector<Triangle> triangles;
  std::vector<Material> materials;
  std::vector<unsigned int> triangleMaterialIds;
  std::vector<unsigned int> lightTriangles;

  size_t fileMarker;
  fileMarker = path.find_last_of("/\\");

  std::string err;
  bool ret = tinyobj::LoadObj(&attrib, &shapes, &tinymaterials, &err, path.c_str(), path.substr(0,fileMarker+1).c_str(), true);

  if (!err.empty())
    std::cout << err;

  if (!ret)
  {
    std::cerr << "Couldn't load model" << std::endl;

    return Model();
  }

  const float m = std::numeric_limits<float>::max();
  float3 maxv = make_float3(-m,-m,-m);
  float3 minv = make_float3(m,m,m);

  for (auto& tm : tinymaterials)
  {
    Material material;

    material.colorAmbient = make_float3(tm.ambient[0], tm.ambient[1], tm.ambient[2]);
    material.colorDiffuse = make_float3(tm.diffuse[0], tm.diffuse[1], tm.diffuse[2]);
    material.colorSpecular = make_float3(tm.specular[0], tm.specular[1], tm.specular[2]);
    material.colorEmission = make_float3(tm.emission[0], tm.emission[1], tm.emission[2]);
    material.colorTransparent = make_float3(1-sqrtf(tm.transmittance[0]), 1-sqrtf(tm.transmittance[1]), 1-sqrtf(tm.transmittance[2]));
    material.refractionIndex = tm.ior;

    switch (tm.illum)
    {
		default:
			std::cerr << "Unknown shading mode for material: " << tm.illum << std::endl;
		case 2:
		case 5:
		case 7:
			material.mode = static_cast<Material::ShadingMode>(tm.illum);
    }
/*
    std::cout << "ambient: " << material.colorAmbient << std::endl;
    std::cout << "diffuse: " << material.colorDiffuse << std::endl;
    std::cout << "specular: " << material.colorSpecular << std::endl;
    std::cout << "emission: " << material.colorEmission << std::endl;
    std::cout << "transparent: " << material.colorTransparent << std::endl;
    std::cout << "refractionIndex: " << material.refractionIndex << std::endl;
    std::cout << "mode: " << material.mode << std::endl << std::endl;
*/
    materials.push_back(material);
  }

	Material lightMaterial;
	lightMaterial.colorEmission = make_float3(300.f, 300.f, 300.f);
	lightMaterial.colorDiffuse = make_float3(1.f, 1.f, 1.f);
	materials.push_back(lightMaterial);

	Material defaultMaterial;
	defaultMaterial.colorAmbient = make_float3(0.f, 1.f, 0.f);
	defaultMaterial.colorDiffuse = make_float3(0.f, 0.f, 0.f);
	defaultMaterial.colorSpecular = make_float3(0.f, 0.f, 0.f);
	defaultMaterial.colorEmission = make_float3(0.f, 0.f, 0.f);
	defaultMaterial.colorTransparent = make_float3(0.f, 0.f, 0.f);
	materials.push_back(defaultMaterial);

  for (size_t s = 0; s < shapes.size(); s++) {
    size_t index_offset = 0;

    for (size_t f = 0; f < shapes[s].mesh.num_face_vertices.size(); f++) {
      size_t fv = shapes[s].mesh.num_face_vertices[f];

      Triangle triangle;
      bool compute_normal(false);

      for (size_t v = 0; v < fv; v++) {
        tinyobj::index_t idx = shapes[s].mesh.indices[index_offset + v];

        tinyobj::real_t vx = attrib.vertices[3*idx.vertex_index+0];
        tinyobj::real_t vy = attrib.vertices[3*idx.vertex_index+1];
        tinyobj::real_t vz = attrib.vertices[3*idx.vertex_index+2];

        const float3 newv = make_float3(vx, vy, vz);
        maxv = fmax(newv, maxv);
        minv = fmin(newv, minv);

        triangle.vertices[v].p = newv;

        if (idx.normal_index >= 0)
        {
          tinyobj::real_t nx = attrib.normals[3*idx.normal_index+0];
          tinyobj::real_t ny = attrib.normals[3*idx.normal_index+1];
          tinyobj::real_t nz = attrib.normals[3*idx.normal_index+2];

          triangle.vertices[v].n = make_float3(nx, ny, nz);
        }

        if (idx.texcoord_index >= 0)
        {
          tinyobj::real_t tx = attrib.texcoords[2*idx.texcoord_index+0];
          tinyobj::real_t ty = attrib.texcoords[2*idx.texcoord_index+1];

          triangle.vertices[v].t = make_float2(tx, ty);
        }

      }
      index_offset += fv;

      if (compute_normal)
      {
        float3 n = normalize(cross(triangle.vertices[1].p - triangle.vertices[0].p, triangle.vertices[2].p - triangle.vertices[0].p));
        triangle.vertices[0].n = n;
        triangle.vertices[1].n = n;
        triangle.vertices[2].n = n;
      }

      triangles.push_back(triangle);

      int32_t materialId = shapes[s].mesh.material_ids[f];

      if (materialId < 0 || materialId >= static_cast<int32_t>(materials.size()))
        materialId = materials.size() - 1;

      const Material& material = materials[materialId];
      triangleMaterialIds.push_back(materialId);

      if (material.colorEmission.x != 0.f || material.colorEmission.y != 0.f || material.colorEmission.z != 0.f)
        lightTriangles.push_back(triangles.size() - 1);
    }
  }

  const float3 bbDiagonal = maxv - minv;
  const float diagonalMaxComponent = fmax_compf(bbDiagonal);

  for (auto& t : triangles)
  {
	  for (auto& v : t.vertices)
	  {
		  v.p += minv;
		  v.p /= diagonalMaxComponent;
	  }
  }

  std::cout << "Creating model with " << triangles.size() << " triangles, " << materials.size() << " materials and " << lightTriangles.size() << " lights" << std::endl;

  Model model(triangles, materials, triangleMaterialIds, lightTriangles, path);

  return model;
}

