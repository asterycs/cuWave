#include "Light.hpp"

#include <glm/gtc/type_ptr.hpp>
#include <glm/gtc/matrix_access.hpp>

CUDA_HOST_DEVICE Light::Light(std::vector<unsigned int> triIds) : ids(triIds), enabled(false)
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

/*
CUDA_HOST std::ostream& operator<<(std::ostream& os, const Light& light)
{
  os << light.enabled << " ";
  os << light.emission.x << " " << light.emission.y << " " << light.emission.z << " ";
  os << light.size.x << " " << light.size.y << " ";

  for (int c = 0; c < 4; ++c)
  {
    const glm::vec4 col = light.modelMat[c];

    os << col.x << " " << col.y << " " << col.z << " " << col.w;

    if (c < 3)
      os << " ";
  }

  return os;
}

CUDA_HOST std::istream& operator>>(std::istream& is, Light& light)
{
  is >> light.enabled;
  is >> light.emission.x >> light.emission.y >> light.emission.z;
  is >> light.size.x >> light.size.y;

  for (int c = 0; c < 4; ++c)
  {
    glm::vec4 col;

    is >> col.x >> col.y >> col.z >> col.w;

    light.modelMat[c] = col;
  }

  return is;
}
*/
