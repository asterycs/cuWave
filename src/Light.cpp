#include "Light.hpp"

CUDA_HOST Light::Light(std::vector<unsigned int> triIds) : enabled(true)
{
  ids = new unsigned int[triIds.size()];
  std::copy(triIds.begin(), triIds.end(), ids);
}

CUDA_HOST_DEVICE Light::~Light()
{
  delete[] ids;
}

CUDA_HOST_DEVICE bool Light::isEnabled() const
{
  return enabled;
}

CUDA_HOST_DEVICE void Light::enable()
{
  enabled = true;
}

CUDA_HOST_DEVICE void Light::disable()
{
  enabled = false;
}
