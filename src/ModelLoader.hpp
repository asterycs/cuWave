#ifndef MODELLOADER_HPP
#define MODELLOADER_HPP

#include "CudaModel.hpp"
#include "GLModel.hpp"

class ModelLoader
{
public:
  ModelLoader();
  ~ModelLoader();
  
  bool loadOBJ(const std::string& path, AbstractModel& model) const;
};

#endif // SCENELOADER_HPP
