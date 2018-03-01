#ifndef CUDARENDERER_HPP
#define CUDARENDERER_HPP

#include <thrust/device_vector.h>

#include "GLTexture.hpp"
#include "Camera.hpp"
#include "Light.hpp"
#include "Model.hpp"

#define QUASIRANDOM

#ifdef QUASIRANDOM
#define CURAND_TYPE curandStateScrambledSobol64
#else
#define CURAND_TYPE curandState_t
#endif

class CudaRenderer
{
public:
  CudaRenderer();
  ~CudaRenderer();

  void pathTraceToCanvas(GLTexture& canvas, const Camera& camera, Model& model, std::vector<Light>& lights);
  void resize(const glm::ivec2 size);
  void reset();

private:
  thrust::device_vector<CURAND_TYPE> curandStateDevVecX;
  thrust::device_vector<CURAND_TYPE> curandStateDevVecY;

  Camera lastCamera;
  glm::ivec2 lastSize;
  unsigned int currentPath;
};

#endif // CUDARENDERER_HPP
