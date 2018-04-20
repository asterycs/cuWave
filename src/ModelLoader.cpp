#include "ModelLoader.hpp"

ModelLoader::ModelLoader()
{

}

ModelLoader::~ModelLoader()
{
}

const aiScene* ModelLoader::loadOBJ(const std::string& path)
{
  const aiScene* model = importer.ReadFile( path,
        aiProcess_CalcTangentSpace       |
        aiProcess_JoinIdenticalVertices  |
        aiProcess_Triangulate            |
        aiProcess_GenNormals);

  if (!model)
  {
    std::cerr << "Error loading file: " << importer.GetErrorString() << std::endl;
    return nullptr;
  }

  return model;
}

