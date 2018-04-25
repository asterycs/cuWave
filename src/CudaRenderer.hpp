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
  uint32_t* extensionQueueSize;

  uint32_t* diffuseQueue;
  uint32_t* diffuseQueueSize;

  uint32_t* shadowQueue;
  uint32_t* shadowQueueSize;

  uint32_t* endQueue;
  uint32_t* endQueueSize;

  Queues()
  :
    extensionQueue(nullptr),
    extensionQueueSize(nullptr),
    diffuseQueue(nullptr),
    diffuseQueueSize(nullptr),
    shadowQueue(nullptr),
    shadowQueueSize(nullptr),
    endQueue(nullptr),
    endQueueSize(nullptr) {};

  ~Queues()
  {
    release();
  }

  __host__ void resize(const glm::ivec2 size)
  {
    release();

    CUDA_CHECK(cudaMalloc((void**) &extensionQueue, size.x*size.y*sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc((void**) &extensionQueueSize, sizeof(uint32_t)));

    CUDA_CHECK(cudaMalloc((void**) &diffuseQueue, size.x*size.y*sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc((void**) &diffuseQueueSize, sizeof(uint32_t)));

    CUDA_CHECK(cudaMalloc((void**) &shadowQueue, size.x*size.y*sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc((void**) &shadowQueueSize, sizeof(uint32_t)));

    CUDA_CHECK(cudaMalloc((void**) &endQueue, size.x*size.y*sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc((void**) &endQueueSize, sizeof(uint32_t)));
  }

  __host__ void release()
  {
    CUDA_CHECK(cudaFree((void**) &extensionQueue));
    CUDA_CHECK(cudaFree((void**) &diffuseQueue));
    CUDA_CHECK(cudaFree((void**) &shadowQueue));
    CUDA_CHECK(cudaFree((void**) &endQueue));
  }
};

struct Paths
{
  glm::fvec2* pixels;
  Ray* rays;
  RaycastResult* results;

  Paths()
  :
    pixels(nullptr),
    rays(nullptr),
    results(nullptr) {};

  ~Paths()
  {
    release();
  }

  __host__ void resize(const glm::ivec2 size)
  {
    release();

    CUDA_CHECK(cudaMalloc((void**) &rays, size.x*size.y*sizeof(Ray)));
    CUDA_CHECK(cudaMalloc((void**) &pixels, size.x*size.y*sizeof(float2)));
    CUDA_CHECK(cudaMalloc((void**) &results, size.x*size.y*sizeof(RaycastResult)));
  }

  __host__ void release()
  {
    CUDA_CHECK(cudaFree((void**) &rays));
    CUDA_CHECK(cudaFree((void**) &pixels));
    CUDA_CHECK(cudaFree((void**) &results));
  }
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
