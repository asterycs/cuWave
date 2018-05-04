#ifndef CUDARENDERER_HPP
#define CUDARENDERER_HPP

#include <thrust/device_vector.h>

#include "GLTexture.hpp"
#include "Camera.hpp"
#include "Light.hpp"
#include "Model.hpp"

//#define QUASIRANDOM

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

  uint32_t* specularQueue;
  uint32_t* specularQueueSize;

  uint32_t* shadowQueue;
  uint32_t* shadowQueueSize;

  Queues()
  :
    extensionQueue(nullptr),
    extensionQueueSize(nullptr),
    diffuseQueue(nullptr),
    diffuseQueueSize(nullptr),
    specularQueue(nullptr),
    specularQueueSize(nullptr),
    shadowQueue(nullptr),
    shadowQueueSize(nullptr) {};

  Queues(const Queues& other) = default;

  ~Queues()
  {

  }

  __host__ void resize(const glm::ivec2 size)
  {
    release();

    CUDA_CHECK(cudaMalloc((void**) &extensionQueue, size.x*size.y*sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc((void**) &extensionQueueSize, sizeof(uint32_t)));

    CUDA_CHECK(cudaMalloc((void**) &diffuseQueue, size.x*size.y*sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc((void**) &diffuseQueueSize, sizeof(uint32_t)));

    CUDA_CHECK(cudaMalloc((void**) &specularQueue, size.x*size.y*sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc((void**) &specularQueueSize, sizeof(uint32_t)));

    CUDA_CHECK(cudaMalloc((void**) &shadowQueue, size.x*size.y*sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc((void**) &shadowQueueSize, sizeof(uint32_t)));

    cudaError_t err = cudaGetLastError();

    if (cudaSuccess != err)
        throw std::runtime_error("Couldn't allocate memory");
  }

  __host__ void release()
  {
    CUDA_CHECK(cudaFree(extensionQueue));
    CUDA_CHECK(cudaFree(extensionQueueSize));
    CUDA_CHECK(cudaFree(diffuseQueue));
    CUDA_CHECK(cudaFree(diffuseQueueSize));
    CUDA_CHECK(cudaFree(specularQueue));
    CUDA_CHECK(cudaFree(specularQueueSize));
    CUDA_CHECK(cudaFree(shadowQueue));
    CUDA_CHECK(cudaFree(shadowQueueSize));
  }

  __host__ void reset()
  {
    CUDA_CHECK(cudaMemset(extensionQueueSize, 0, sizeof(uint32_t)));
    CUDA_CHECK(cudaMemset(diffuseQueueSize, 0, sizeof(uint32_t)));
    CUDA_CHECK(cudaMemset(specularQueueSize, 0, sizeof(uint32_t)));
    CUDA_CHECK(cudaMemset(shadowQueueSize, 0, sizeof(uint32_t)));
  }
};

struct Paths
{
  Ray* rays;
  uint2* pixels;
  RaycastResult* results;
  float3* colors;
  float3* throughputs;
  float* p;

  CURAND_TYPE* random0;
  CURAND_TYPE* random1;

  Paths(const Paths& other) = default;

  Paths()
  :
    rays(nullptr),
    pixels(nullptr),
    results(nullptr),
    colors(nullptr),
    throughputs(nullptr),
    p(nullptr),

    random0(nullptr),
    random1(nullptr) {};

  ~Paths()
  {

  }

  __host__ void resize(const glm::ivec2 size)
  {
    release();

    CUDA_CHECK(cudaMalloc((void**) &rays, size.x*size.y*sizeof(Ray)));
    CUDA_CHECK(cudaMalloc((void**) &pixels, size.x*size.y*sizeof(uint2)));
    CUDA_CHECK(cudaMalloc((void**) &results, size.x*size.y*sizeof(RaycastResult)));
    CUDA_CHECK(cudaMalloc((void**) &colors, size.x*size.y*sizeof(float3)));
    CUDA_CHECK(cudaMalloc((void**) &throughputs, size.x*size.y*sizeof(float3)));
    CUDA_CHECK(cudaMalloc((void**) &p, size.x*size.y*sizeof(float)));

    CUDA_CHECK(cudaMalloc((void**) &random0, size.x*size.y*sizeof(CURAND_TYPE)));
    CUDA_CHECK(cudaMalloc((void**) &random1, size.x*size.y*sizeof(CURAND_TYPE)));
  }

  __host__ void release()
  {
    CUDA_CHECK(cudaFree(rays));
    CUDA_CHECK(cudaFree(pixels));
    CUDA_CHECK(cudaFree(results));
    CUDA_CHECK(cudaFree(colors));
    CUDA_CHECK(cudaFree(throughputs));
    CUDA_CHECK(cudaFree(p));

    CUDA_CHECK(cudaFree(random0));
    CUDA_CHECK(cudaFree(random1));
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
  Camera lastCamera;
  glm::ivec2 lastSize;
  uint32_t currentPath;

  Queues queues;
  Paths paths;
};

#endif // CUDARENDERER_HPP
