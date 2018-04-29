#ifndef LIGHT_HPP
#define LIGHT_HPP

#ifdef __CUDACC__
  #define CUDA_HOST_DEVICE __host__ __device__
  #define CUDA_DEVICE               __device__
  #define CUDA_HOST        __host__
#else
  #define CUDA_HOST_DEVICE
  #define CUDA_DEVICE
  #define CUDA_HOST
#endif

#include <curand.h>
#include <curand_kernel.h>

#include "Utils.hpp"

class Light
{
public:
  CUDA_HOST_DEVICE Light();
  CUDA_HOST Light(const uint32_t start, const uint32_t end);
  CUDA_HOST_DEVICE ~Light();
  
  CUDA_HOST_DEVICE bool isEnabled() const;
  CUDA_HOST_DEVICE void enable();
  CUDA_HOST_DEVICE void disable();

  template<typename curandState>
  CUDA_DEVICE void sample(float& pdf, glm::vec3& point, curandState& randomState1, curandState& randomState2) const;
private:
  glm::fvec2 size;
  uint32_t start;
  uint32_t end;
  bool enabled;
};

template<typename curandState>
CUDA_DEVICE void Light::sample(float& pdf, glm::vec3& point, curandState& randomState1, curandState& randomState2) const
{
  const float x = curand_uniform(&randomState1);
  const float y = curand_uniform(&randomState2);

  const glm::fvec2 span = glm::fvec2(x * 2.f, y * 2.f);
  glm::fvec2 rf(span.x - 1.f, span.y - 1.f);

  pdf = 1.0f / (size.x * size.y);

  const glm::fvec2 rndClip(rf.x * size.x, rf.y * size.y);
  const glm::fvec4 p4 = glm::vec4(rndClip, 0, 1);
  point = glm::fvec3(p4);
}

#endif // LIGHT_HPP
