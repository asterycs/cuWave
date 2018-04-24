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

struct Queues
{
  uint32_t* extensionQueue;
  uint32_t extensionQueueSize;

  uint32_t* shadowQueue;
  uint32_t shadowQueueSize;

  uint32_t* endQueue;
  uint32_t endQueueSize;

  Queues()
    :
      extensionQueue(nullptr),
      extensionQueueSize(0),
      shadowQueue(nullptr),
      shadowQueueSize(0),
      endQueue(nullptr),
      endQueueSize(0) {};
};

struct Paths
{
  glm::fvec2* pixels;
  Ray* rays;
  RaycastResult* results;
};

class CudaRenderer
{
public:
  CudaRenderer();
  ~CudaRenderer();

  void pathTraceToCanvas(GLTexture& canvas, const Camera& camera, Model& model);
  void resize(const glm::ivec2 size);
  void reset();

private:
  thrust::device_vector<CURAND_TYPE> curandStateDevVecX;
  thrust::device_vector<CURAND_TYPE> curandStateDevVecY;

  Camera lastCamera;
  glm::ivec2 lastSize;
  uint32_t currentPath;

  Queues queues;
  Paths paths;
};

#endif // CUDARENDERER_HPP
