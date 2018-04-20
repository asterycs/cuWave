#include "CudaRenderer.hpp"

#include <GL/glew.h>
#include <GL/gl.h>

#include <cuda.h>
#include <cuda_gl_interop.h>
#include <curand.h>
#include <curand_kernel.h>

#include "Utils.hpp"
#include "Triangle.hpp"


#define BLOCKWIDTH 8
#define INTERSECT_EPSILON 0.0000001f
#define OFFSET_EPSILON 0.00001f
#define BIGT 99999.f
#define SHADOWSAMPLING 8
#define SECONDARY_RAYS 3
#define AIR_INDEX 1.f

#define REFLECTIVE_BIT 0x80000000
#define REFRACTIVE_BIT 0x40000000
#define INSIDE_BIT 0x20000000

#define LEFT_HIT_BIT 0x80000000
#define RIGHT_HIT_BIT 0x40000000

#define PATH_TRACE_BOUNCES 6


template <typename curandState>
__global__ void initRand(const int seed, curandState* const curandStateDevPtr, const glm::ivec2 size)
{
  const int x = threadIdx.x + blockIdx.x * blockDim.x;
  const int y = threadIdx.y + blockIdx.y * blockDim.y;

  if (x >= size.x || y >= size.y)
    return;

  curandState localState;
  curand_init(seed, x + y*size.x, 0, &localState);
  curandStateDevPtr[x + y * size.x] = localState;
}

__global__ void initRand(curandDirectionVectors64_t* sobolDirectionVectors, unsigned long long* sobolScrambleConstants, curandStateScrambledSobol64* state, const glm::ivec2 size)
{
  const int x = threadIdx.x + blockIdx.x * blockDim.x;
  const int y = threadIdx.y + blockIdx.y * blockDim.y;
  
  if (x >= size.x || y >= size.y)
    return;
    
  const unsigned int scrIdx = x + size.x * y;
  const unsigned int dirIdx = (x + size.x * y) % 10000;

  curandDirectionVectors64_t* dir = &sobolDirectionVectors[dirIdx];
  unsigned long long scr = sobolScrambleConstants[scrIdx];
  curandStateScrambledSobol64 localState;
    
  curand_init(*dir, scr, 0, &localState);

  state[x + size.x * y] = localState;
}

__device__ void writeToCanvas(const unsigned int x, const unsigned int y, const cudaSurfaceObject_t& surfaceObj, const glm::ivec2 canvasSize, const glm::vec3 data)
{
  const float4 out = make_float4(data.x, data.y, data.z, 1.f);
  surf2Dwrite(out, surfaceObj, (canvasSize.x - 1 - x) * sizeof(out), y);
  return;
}

__device__ glm::fvec3 readFromCanvas(const unsigned int x, const unsigned int y, const cudaSurfaceObject_t& surfaceObj, const glm::ivec2 canvasSize)
{
  float4 in;
  surf2Dread(&in, surfaceObj, (canvasSize.x - 1 - x) * sizeof(in), y);

  const glm::fvec3 ret(in.x, in.y, in.z);

  return ret;
}

template <typename curandStateType>
__global__ void
cudaDebugPathTrace(
    const glm::ivec2 pixelPos,
    glm::fvec3* devPosPtr,
    const glm::ivec2 size,
    const Triangle* triangles,
    const Camera camera,
    const Material* materials,
    const unsigned int* triangleMaterialIds,
    const Light light,
    curandStateType* curandStateDevXPtr,
    curandStateType* curandStateDevYPtr,
    const Node* bvh)
{
  const glm::fvec2 nic = camera.normalizedImageCoordinateFromPixelCoordinate(pixelPos.x, pixelPos.y, size);
  const float ar = (float) size.x / size.y;
  const Ray ray = camera.generateRay(nic, ar);
/*
  (void) pathTrace<true>(
      bvh,
      ray,
      triangles,
      camera,
      materials,
      triangleMaterialIds,
      light,
      curandStateDevXPtr[pixelPos.x + size.x * pixelPos.y],
      curandStateDevYPtr[pixelPos.x + size.x * pixelPos.y],
      devPosPtr);
*/
  return;
}

template <typename curandStateType>
__global__ void
pathTraceKernel(
    const unsigned int path,
    const cudaSurfaceObject_t canvas,
    const glm::ivec2 canvasSize,
    const Triangle* triangles,
    const Camera camera,
    const Material* materials,
    const unsigned int* triangleMaterialIds,
    const Light* lights,
    curandStateType* curandStateDevXPtr,
    curandStateType* curandStateDevYPtr,
    const Node* bvh)
{
  const int x = threadIdx.x + blockIdx.x * blockDim.x;
  const int y = threadIdx.y + blockIdx.y * blockDim.y;

  if (x >= canvasSize.x || y >= canvasSize.y)
    return;

  glm::vec2 nic = camera.normalizedImageCoordinateFromPixelCoordinate(x, y, canvasSize);

  Ray ray = camera.generateRay(nic, (float) canvasSize.x/canvasSize.y);

  curandStateType state1 = curandStateDevXPtr[x + y * canvasSize.x];
  curandStateType state2 = curandStateDevYPtr[x + y * canvasSize.x];
/*
  glm::fvec3 color = pathTrace<false>(\
      bvh,
      ray, \
      triangles, \
      camera, \
      materials, \
      triangleMaterialIds, \
      lights, \
      state1, \
      state2);
*/
  glm::fvec3 color = glm::fvec3(0.4,0.4,0.4);
  curandStateDevXPtr[x + y * canvasSize.x] = state1;
  curandStateDevYPtr[x + y * canvasSize.x] = state2;

  if (path == 1)
  {
    writeToCanvas(x, y, canvas, canvasSize, color);
  }
  else
  {
    const glm::fvec3 oldCol = readFromCanvas(x, y, canvas, canvasSize);
    const glm::fvec3 blend = oldCol * glm::fvec3((float) (path - 1) / path) + glm::fvec3((float) 1 / path) * color;
    writeToCanvas(x, y, canvas, canvasSize, blend);
  }

  return;
}

template <typename curandStateType>
__global__ void
cudaTestRnd(\
    const cudaSurfaceObject_t canvas, \
    const glm::ivec2 canvasSize, \
    curandStateType* curandStateDevXPtr, \
    curandStateType* curandStateDevYPtr)
{
  const int x = threadIdx.x + blockIdx.x * blockDim.x;
  const int y = threadIdx.y + blockIdx.y * blockDim.y;

  if (x >= canvasSize.x || y >= canvasSize.y)
    return;

  curandStateType localState1 = curandStateDevXPtr[x + y * canvasSize.x];
  curandStateType localState2 = curandStateDevYPtr[x + y * canvasSize.x];

  float r = curand_uniform(&localState1);
  float g = curand_uniform(&localState2);

  curandStateDevXPtr[x + y * canvasSize.x] = localState1;
  curandStateDevYPtr[x + y * canvasSize.x] = localState2;

  writeToCanvas(x, y, canvas, canvasSize, glm::fvec3(r, g, 0.f));

  return;
}

void CudaRenderer::reset()
{
  currentPath = 1;
}

void CudaRenderer::resize(const glm::ivec2 size)
{
  curandStateDevVecX.resize(size.x * size.y);
  curandStateDevVecY.resize(size.x * size.y);
  auto* curandStateDevXRaw = thrust::raw_pointer_cast(&curandStateDevVecX[0]);
  auto* curandStateDevYRaw = thrust::raw_pointer_cast(&curandStateDevVecY[0]);

  dim3 block(BLOCKWIDTH, BLOCKWIDTH);
  dim3 grid( (size.x + block.x - 1) / block.x, (size.y + block.y - 1) / block.y);

  cudaStream_t streams[2];
  CUDA_CHECK(cudaStreamCreate(&streams[0]));
  CUDA_CHECK(cudaStreamCreate(&streams[1]));

#ifdef QUASIRANDOM
  curandDirectionVectors64_t* hostDirectionVectors64;
  unsigned long long int* hostScrambleConstants64;
  
  curandDirectionVectors64_t* devDirectionVectors64;
  unsigned long long int* devScrambleConstants64;
  
  curandGetDirectionVectors64(&hostDirectionVectors64, CURAND_SCRAMBLED_DIRECTION_VECTORS_64_JOEKUO6);
  curandGetScrambleConstants64(&hostScrambleConstants64);
  
  CUDA_CHECK(cudaMalloc((void **)&(devDirectionVectors64),             20000 * sizeof(curandDirectionVectors64_t)));
  CUDA_CHECK(cudaMemcpy(devDirectionVectors64, hostDirectionVectors64, 20000 * sizeof(curandDirectionVectors64_t), cudaMemcpyHostToDevice));
  
  CUDA_CHECK(cudaMalloc((void **)&(devScrambleConstants64),              size.x * size.y * sizeof(unsigned long long int)));
  CUDA_CHECK(cudaMemcpy(devScrambleConstants64, hostScrambleConstants64, size.x * size.y * sizeof(unsigned long long int), cudaMemcpyHostToDevice));
  
  initRand<<<grid, block, 0, streams[0]>>>(devDirectionVectors64, devScrambleConstants64, curandStateDevXRaw, size);
  initRand<<<grid, block, 0, streams[1]>>>(devDirectionVectors64 + 10000, devScrambleConstants64, curandStateDevYRaw, size);
  
  CUDA_CHECK(cudaFree(devDirectionVectors64));
  CUDA_CHECK(cudaFree(devScrambleConstants64));

#else
  initRand<<<grid, block, 0, streams[0]>>>(0, curandStateDevXRaw, size);
  initRand<<<grid, block, 0, streams[1]>>>(5, curandStateDevYRaw, size);
#endif

  CUDA_CHECK(cudaStreamDestroy(streams[0]));
  CUDA_CHECK(cudaStreamDestroy(streams[1]));

  CUDA_CHECK(cudaDeviceSynchronize());
}


CudaRenderer::CudaRenderer() : curandStateDevVecX(), curandStateDevVecY(), lastCamera(), lastSize(), currentPath(1)
{
  unsigned int cudaDeviceCount = 0;
  int cudaDevices[8];
  unsigned int cudaDevicesCount = 8;

  cudaGLGetDevices(&cudaDeviceCount, cudaDevices, cudaDevicesCount, cudaGLDeviceListCurrentFrame);

  if (cudaDeviceCount < 1)
  {
     std::cout << "No CUDA devices found" << std::endl;
     throw std::runtime_error("No CUDA devices available");
  }

  CUDA_CHECK(cudaSetDevice(cudaDevices[0]));

  resize(glm::ivec2(WWIDTH, WHEIGHT));
}

CudaRenderer::~CudaRenderer()
{

}

void CudaRenderer::pathTraceToCanvas(GLTexture& canvas, const Camera& camera, Model& model)
{
  if (model.getNTriangles() == 0)
    return;

  const glm::ivec2 canvasSize = canvas.getSize();
  const bool diffCamera = std::memcmp(&camera, &lastCamera, sizeof(Camera));
  const bool diffSize = (canvasSize != lastSize);

  if (diffCamera != 0 || diffSize != 0)
  {
    lastCamera = camera;
    lastSize = canvasSize;
    currentPath = 1;
  }

  auto* curandStateDevXRaw = thrust::raw_pointer_cast(&curandStateDevVecX[0]);
  auto* curandStateDevYRaw = thrust::raw_pointer_cast(&curandStateDevVecY[0]);

  auto surfaceObj = canvas.getCudaMappedSurfaceObject();
  const Triangle* devTriangles = model.getDeviceTriangles();

  const dim3 block(BLOCKWIDTH, BLOCKWIDTH);
  const dim3 grid( (canvasSize.x+ block.x - 1) / block.x, (canvasSize.y + block.y - 1) / block.y);

  pathTraceKernel<<<grid, block>>>(
      currentPath,
      surfaceObj,
      canvasSize,
      devTriangles,
      camera,
      model.getDeviceMaterials(),
      model.getDeviceTriangleMaterialIds(),
      model.getDeviceLights(),
      curandStateDevXRaw,
      curandStateDevYRaw,
      model.getDeviceBVH());

  ++currentPath;

  CUDA_CHECK(cudaDeviceSynchronize());
  canvas.cudaUnmap();
}

