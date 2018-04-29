#ifndef MODELLOADER_HPP
#define MODELLOADER_HPP

#include "Model.hpp"

class ModelLoader
{
public:
  ModelLoader();
  ~ModelLoader();
  
  Model loadOBJ(const std::string& path);

};

#endif // SCENELOADER_HPP
