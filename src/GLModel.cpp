#include "GLModel.hpp"

#ifdef ENABLE_CUDA
  #include <cuda_runtime.h>
#endif

#include <vector>

GLModel::GLModel()
{

}

GLModel::~GLModel()
{

}

GLModel::GLModel(const std::vector<Triangle> triangles, const std::vector<Material> materials, const std::vector<std::vector<uint32_t>> materialIds, const std::string& fileName) : fileName(fileName)
{
	if (triangles.size() == 0 || materialIds.size() == 0)
	{
		std::cerr << "Model is empty!" << std::endl;
		return;
	}

	this->materials = materials;

	GL_CHECK(glGenVertexArrays(1, &vaoID));
	GL_CHECK(glBindVertexArray(vaoID));

	GL_CHECK(glGenBuffers(1, &vboID));
	GL_CHECK(glBindBuffer(GL_ARRAY_BUFFER, vboID));
	GL_CHECK(glBufferData(GL_ARRAY_BUFFER, triangles.size() * 3 * sizeof(Vertex), triangles.data(), GL_STATIC_DRAW));

	GL_CHECK(glEnableVertexAttribArray(0));
	GL_CHECK(glVertexAttribPointer(
		 0,
		 3,
		 GL_FLOAT,
		 GL_FALSE,
		 sizeof(Vertex),
		 (GLvoid*)0
	));

	GL_CHECK(glEnableVertexAttribArray(1));
	GL_CHECK(glVertexAttribPointer(
		1,
		3,
		GL_FLOAT,
		GL_FALSE,
		sizeof(Vertex),
		(GLvoid*)offsetof(Vertex, n)
	));

	std::vector<uint32_t> flattenedIds;
	meshSizes.clear();

	for (const auto& i : materialIds)
	{
	    flattenedIds.insert(flattenedIds.end(), i.begin(), i.end());
	    meshSizes.push_back(i.size());
	}

    GL_CHECK(glGenBuffers(1, &indexID));
    GL_CHECK(glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, indexID));
    GL_CHECK(glBufferData(GL_ELEMENT_ARRAY_BUFFER, flattenedIds.size() * sizeof(uint32_t), flattenedIds.data(), GL_STATIC_DRAW));

	GL_CHECK(glBindVertexArray(0));
}

GLuint GLModel::getIndexID() const
{
	return indexID;
}

const std::vector<uint32_t>& GLModel::getMeshSizes() const
{
  return meshSizes;
}

const std::vector<Material>& GLModel::getMaterials() const
{
  return materials;
}

const std::string GLModel::getFileName() const
{
  return fileName;
}

GLuint GLModel::getVaoID() const
{
	return vaoID;
}
