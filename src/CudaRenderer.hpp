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

    CUDA_CHECK(cudaMallocManaged((void**) &extensionQueue, size.x*size.y*sizeof(uint32_t)));
    CUDA_CHECK(cudaMallocManaged((void**) &extensionQueueSize, sizeof(uint32_t)));

    CUDA_CHECK(cudaMallocManaged((void**) &diffuseQueue, size.x*size.y*sizeof(uint32_t)));
    CUDA_CHECK(cudaMallocManaged((void**) &diffuseQueueSize, sizeof(uint32_t)));

    CUDA_CHECK(cudaMallocManaged((void**) &specularQueue, size.x*size.y*sizeof(uint32_t)));
    CUDA_CHECK(cudaMallocManaged((void**) &specularQueueSize, sizeof(uint32_t)));

    CUDA_CHECK(cudaMallocManaged((void**) &shadowQueue, size.x*size.y*sizeof(uint32_t)));
    CUDA_CHECK(cudaMallocManaged((void**) &shadowQueueSize, sizeof(uint32_t)));

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
  float2* pixels;
  Ray* rays;
  RaycastResult* results;
  float3* colors;
  float3* filters;
  CURAND_TYPE* random0;
  CURAND_TYPE* random1;

  Paths(const Paths& other) = default;

  Paths()
  :
    pixels(nullptr),
    rays(nullptr),
    results(nullptr),
    colors(nullptr),
    filters(nullptr),
    random0(nullptr),
    random1(nullptr) {};

  ~Paths()
  {

  }

  __host__ void resize(const glm::ivec2 size)
  {
    release();

    CUDA_CHECK(cudaMallocManaged((void**) &rays, size.x*size.y*sizeof(Ray)));
    CUDA_CHECK(cudaMallocManaged((void**) &pixels, size.x*size.y*sizeof(float2)));
    CUDA_CHECK(cudaMallocManaged((void**) &results, size.x*size.y*sizeof(RaycastResult)));
    CUDA_CHECK(cudaMallocManaged((void**) &colors, size.x*size.y*sizeof(float3)));
    CUDA_CHECK(cudaMallocManaged((void**) &filters, size.x*size.y*sizeof(float3)));
    CUDA_CHECK(cudaMallocManaged((void**) &random0, size.x*size.y*sizeof(CURAND_TYPE)));
    CUDA_CHECK(cudaMallocManaged((void**) &random1, size.x*size.y*sizeof(CURAND_TYPE)));
  }

  __host__ void release()
  {
    CUDA_CHECK(cudaFree(rays));
    CUDA_CHECK(cudaFree(pixels));
    CUDA_CHECK(cudaFree(results));
    CUDA_CHECK(cudaFree(colors));
    CUDA_CHECK(cudaFree(filters));
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
