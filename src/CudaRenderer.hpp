#ifndef CUDARENDERER_HPP
#define CUDARENDERER_HPP

#include <curand.h>
#include <curand_kernel.h>

#include <thrust/device_vector.h>

#include "GLTexture.hpp"
#include "Camera.hpp"
#include "Model.hpp"

#define PREGEN_RANDS 20

__host__ __device__ int getRandomSizeMult(const glm::ivec2 size);

struct Queues
{
  uint32_t* diffuseQueue;
  uint32_t* diffuseQueueSize;

  uint32_t* specularQueue;
  uint32_t* specularQueueSize;

  uint32_t* transparentQueue;
  uint32_t* transparentQueueSize;

  uint32_t* newPathQueue;
  uint32_t* newPathQueueSize;

  Queues()
  :
    diffuseQueue(nullptr),
    diffuseQueueSize(nullptr),
    specularQueue(nullptr),
    specularQueueSize(nullptr),
    transparentQueue(nullptr),
    transparentQueueSize(nullptr),
    newPathQueue(nullptr),
    newPathQueueSize(nullptr) {};

  Queues(const Queues& other) = default;

  ~Queues()
  {

  }

  __host__ void resize(const glm::ivec2 size)
  {
    release();

    CUDA_CHECK(cudaMalloc((void**) &diffuseQueue, size.x*size.y*sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc((void**) &diffuseQueueSize, sizeof(uint32_t)));

    CUDA_CHECK(cudaMalloc((void**) &specularQueue, size.x*size.y*sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc((void**) &specularQueueSize, sizeof(uint32_t)));

    CUDA_CHECK(cudaMalloc((void**) &transparentQueue, size.x*size.y*sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc((void**) &transparentQueueSize, sizeof(uint32_t)));

    CUDA_CHECK(cudaMalloc((void**) &newPathQueue, size.x*size.y*sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc((void**) &newPathQueueSize, sizeof(uint32_t)));

    cudaError_t err = cudaGetLastError();

    if (cudaSuccess != err)
        throw std::runtime_error("Couldn't allocate memory");
  }

  __host__ void release()
  {
    CUDA_CHECK(cudaFree(diffuseQueue));
    CUDA_CHECK(cudaFree(diffuseQueueSize));
    CUDA_CHECK(cudaFree(specularQueue));
    CUDA_CHECK(cudaFree(specularQueueSize));
    CUDA_CHECK(cudaFree(transparentQueue));
    CUDA_CHECK(cudaFree(transparentQueueSize));
    CUDA_CHECK(cudaFree(newPathQueue));
    CUDA_CHECK(cudaFree(newPathQueueSize));
  }

  __host__ void reset()
  {
    CUDA_CHECK(cudaMemset(diffuseQueueSize,     0, sizeof(uint32_t)));
    CUDA_CHECK(cudaMemset(specularQueueSize,    0, sizeof(uint32_t)));
    CUDA_CHECK(cudaMemset(transparentQueueSize, 0, sizeof(uint32_t)));
    CUDA_CHECK(cudaMemset(newPathQueueSize,     0, sizeof(uint32_t)));
  }
};

struct Paths
{
  Ray* ray;
  RaycastResult* result;
  float3* color;
  float3* throughput;
  float* p;
  uint32_t* pathNr;
  uint32_t* rayNr;

  float* floats;

  Paths(const Paths& other) = default;

  Paths()
  :
    ray(nullptr),
    result(nullptr),
    color(nullptr),
    throughput(nullptr),
    p(nullptr),
    pathNr(nullptr),
    rayNr(nullptr),

    floats(nullptr) {};

  ~Paths()
  {

  }

  __host__ void resize(const glm::ivec2 size)
  {
    release();

    CUDA_CHECK(cudaMalloc((void**) &ray, size.x*size.y*sizeof(Ray)));
    CUDA_CHECK(cudaMalloc((void**) &result, size.x*size.y*sizeof(RaycastResult)));
    CUDA_CHECK(cudaMalloc((void**) &color, size.x*size.y*sizeof(float3)));
    CUDA_CHECK(cudaMalloc((void**) &throughput, size.x*size.y*sizeof(float3)));
    CUDA_CHECK(cudaMalloc((void**) &p, size.x*size.y*sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**) &pathNr, size.x*size.y*sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc((void**) &rayNr, size.x*size.y*sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc((void**) &floats, 32*sizeof(float)));
  }

  __host__ void release()
  {
    CUDA_CHECK(cudaFree(ray));
    CUDA_CHECK(cudaFree(result));
    CUDA_CHECK(cudaFree(color));
    CUDA_CHECK(cudaFree(throughput));
    CUDA_CHECK(cudaFree(p));
    CUDA_CHECK(cudaFree(pathNr));
    CUDA_CHECK(cudaFree(rayNr));
    CUDA_CHECK(cudaFree(floats));
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

  Queues queues;
  Paths paths;

  curandGenerator_t randGen;

  uint32_t callcntr;
};

#endif // CUDARENDERER_HPP
