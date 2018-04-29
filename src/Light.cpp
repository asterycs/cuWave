#include "Light.hpp"

CUDA_HOST Light::Light(const uint32_t start, const uint32_t end) : start(start), end(end), enabled(true)
{

}

CUDA_HOST_DEVICE Light::~Light()
{

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
