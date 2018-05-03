#include "CudaRenderer.hpp"

#include <GL/glew.h>
#include <GL/gl.h>

#include <cuda.h>
#include <cuda_gl_interop.h>
#include <curand.h>
#include <curand_kernel.h>
#include <math_constants.h>

#include "Utils.hpp"
#include "Triangle.hpp"

#define BLOCKWIDTH 8
#define INTERSECT_EPSILON 0.0000001f
#define OFFSET_EPSILON 0.00001f
#define BIGT 99999.f
#define AIR_INDEX 1.f

#define REFLECTIVE_BIT 0x80000000
#define REFRACTIVE_BIT 0x40000000
#define INSIDE_BIT 0x20000000

#define LEFT_HIT_BIT 0x80000000
#define RIGHT_HIT_BIT 0x40000000

__device__ bool bboxIntersect(const AABB box, const float3 origin, const float3 inverseDirection, float& t)
{
  float3 tmin = make_float3(-BIGT, -BIGT, -BIGT),
         tmax = make_float3(BIGT, BIGT, BIGT);

  const float3 tdmin = (box.min - origin) * inverseDirection;
  const float3 tdmax = (box.max - origin) * inverseDirection;

  tmin = fminf(tdmin, tdmax);
  tmax = fmaxf(tdmin, tdmax);

  const float tmind = fmin_compf(tmin);
  const float tmaxd = fmin_compf(tmax);

  t = fminf(tmind, tmaxd);

  return tmaxd >= tmind && !(tmaxd < 0.f && tmind < 0.f);
}

__device__ bool rayTriangleIntersection(const Ray ray, const Triangle& triangle, float& t, float2& uv)
{
  /* Möller-Trumbore algorithm
   * https://en.wikipedia.org/wiki/M%C3%B6ller%E2%80%93Trumbore_intersection_algorithm
   */

  // TODO: Experiment with __ldg

  const float3 vertex0 = triangle.vertices[0].p;

  const float3 edge1 = triangle.vertices[1].p - vertex0;
  const float3 edge2 = triangle.vertices[2].p - vertex0;

  const float3 h = cross(ray.direction, edge2);
  const float a = dot(edge1, h);

  if (a > -INTERSECT_EPSILON && a < INTERSECT_EPSILON)
    return false;

  const float f = __fdividef(1.f, a);
  const float3 s = ray.origin - vertex0;
  const float u = f * dot(s, h);

  if (u < 0.f || u > 1.0f)
    return false;

  const float3 q = cross(s, edge1);
  const float v = f * dot(ray.direction, q);

  if (v < 0.0 || u + v > 1.0)
    return false;

  t = f * dot(edge2, q);

  if (t > INTERSECT_EPSILON)
  {
    uv = make_float2(u, v);
    return true;
  }
  else
    return false;
}


enum HitType
{
    ANY,
    CLOSEST
};

template <const HitType hitType>
__device__
RaycastResult rayCast(const Ray ray, const Node* bvh, const Triangle* triangles, const float maxT)
{
  float tMin = maxT;
  int minTriIdx = -1;
  float2 minUV;
  RaycastResult result;
  const float3 inverseDirection = make_float3(1.f, 1.f, 1.f) / ray.direction;

  int ptr = 0;
  unsigned int stack[16] { 0 };
  int i = -1;
  float t = 0;
  float2 uv;
  bool getNextNode = true;

  while (ptr >= 0)
  {
    unsigned int currentNodeIdx = stack[ptr];
    Node currentNode = bvh[currentNodeIdx];


    if (currentNode.rightIndex == -1)
    {
      getNextNode = false;

      if (i >= currentNode.startTri && i < currentNode.startTri + currentNode.nTri)
      {
        if (rayTriangleIntersection(ray, triangles[i], t, uv))
        {

          if(t < tMin)
          {
            tMin = t;
            minTriIdx = i;
            minUV = uv;

            if (hitType == HitType::ANY)
              break;
          }
        }

        ++i;

        if (i >= currentNode.startTri + currentNode.nTri)
          getNextNode = true;

      }else
      {
        i = currentNode.startTri;
      }

    }else
    {
      const AABB leftBox = bvh[stack[ptr] + 1].bbox;
      const AABB rightBox = bvh[currentNode.rightIndex].bbox;

      float leftt, rightt;

      unsigned int hitMask = bboxIntersect(leftBox, ray.origin, inverseDirection, leftt) ? LEFT_HIT_BIT : 0x00;
      hitMask = bboxIntersect(rightBox, ray.origin, inverseDirection, rightt) ? hitMask | RIGHT_HIT_BIT : hitMask;

      // TODO: Push closer one last, don't intersect if closest hit is closer than box
      if ((hitMask & LEFT_HIT_BIT) != 0x00 && leftt < tMin)
      {
        stack[ptr] = currentNodeIdx + 1;
        ++ptr;
      }

      if ((hitMask & RIGHT_HIT_BIT) != 0x00 && rightt < tMin)
      {
        stack[ptr] = currentNode.rightIndex;
        ++ptr;
      }
    }

    if (getNextNode)
    {
      --ptr;
      i = -1;
    }

  }

  if (minTriIdx == -1)
    return result;

  result.point = ray.origin + ray.direction * tMin;
  result.t = tMin;
  result.triangleIdx = minTriIdx;
  result.uv = minUV;


  return result;
}

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
    
  const uint32_t scrIdx = x + size.x * y;
  const uint32_t dirIdx = (x + size.x * y) % 10000;

  curandDirectionVectors64_t* dir = &sobolDirectionVectors[dirIdx];
  unsigned long long scr = sobolScrambleConstants[scrIdx];
  curandStateScrambledSobol64 localState;
    
  curand_init(*dir, scr, 0, &localState);

  state[x + size.x * y] = localState;
}

__device__ void writeToCanvas(const uint32_t x, const uint32_t y, const cudaSurfaceObject_t& surfaceObj, const glm::ivec2 canvasSize, const float3 data)
{
  const float4 out = make_float4(data.x, data.y, data.z, 1.f);
  surf2Dwrite(out, surfaceObj, (canvasSize.x - 1 - x) * sizeof(out), y);
  return;
}

__device__ float3 readFromCanvas(const uint32_t x, const uint32_t y, const cudaSurfaceObject_t& surfaceObj, const glm::ivec2 canvasSize)
{
  float4 in;
  surf2Dread(&in, surfaceObj, (canvasSize.x - 1 - x) * sizeof(in), y);
  const float3 ret = make_float3(in.x, in.y, in.z);

  return ret;
}

__global__ void
logicKernel(
    const glm::ivec2 canvasSize,
    Queues queues,
    Paths paths,
    const Material* materials,
    const uint32_t* triangleMaterialIds)
{
  const int x = threadIdx.x + blockIdx.x * blockDim.x;
  const int y = threadIdx.y + blockIdx.y * blockDim.y;
  const int idx = x + y * canvasSize.x;

  const float3 float3_zero = make_float3(0.f, 0.f, 0.f);

  if (x >= canvasSize.x || y >= canvasSize.y)
    return;

  paths.colors[idx] = float3_zero;

  const RaycastResult result = paths.results[idx];

  if (!result)
  {
    return;
  }

  const Material material = materials[triangleMaterialIds[result.triangleIdx]];

  if (material.colorDiffuse != float3_zero)
  {
    const uint32_t new_idx = atomicAdd(queues.diffuseQueueSize, 1);
    queues.diffuseQueue[new_idx] = idx;
  }

  if (material.colorSpecular != float3_zero)
  {
    const uint32_t new_idx = atomicAdd(queues.specularQueueSize, 1);
    queues.specularQueue[new_idx] = idx;
  }

  return;
}

__global__ void
writeToCanvas(
    const glm::ivec2 canvasSize,
    cudaSurfaceObject_t canvas,
    Paths paths,
    const int currentPath
    )
{
  const uint32_t x = threadIdx.x + blockIdx.x * blockDim.x;
  const uint32_t y = threadIdx.y + blockIdx.y * blockDim.y;
  const int idx = x + y * canvasSize.x;

  if (x >= canvasSize.x || y >= canvasSize.y)
    return;

  const float3 newColor = paths.colors[idx];
  float3 oldColor = readFromCanvas(x, y, canvas, canvasSize);
  float3 blend = static_cast<float>(currentPath-1)/currentPath * oldColor + 1.f/currentPath * newColor;

  writeToCanvas(x, y, canvas, canvasSize, blend);
}

__global__ void
diffuseKernel(
    const glm::ivec2 canvasSize,
    const Queues queues,
    Paths paths,
    const Triangle* triangles,
    const uint32_t* lightTriangleIds,
    const uint32_t  lightTriangles,
    const uint32_t* triangleMaterialIds,
    const Material* materials,
    const Node* bvh)
{
  const uint32_t x = threadIdx.x + blockIdx.x * blockDim.x;
  const uint32_t y = threadIdx.y + blockIdx.y * blockDim.y;
  const uint32_t idx = x + y * canvasSize.x;

  if (idx >= *queues.diffuseQueueSize)
    return;

  const float3 float3_zero = make_float3(0.f, 0.f, 0.f);
  const uint32_t pathIdx = queues.diffuseQueue[idx];

  const RaycastResult result = paths.results[pathIdx];
  const Material& material = materials[triangleMaterialIds[result.triangleIdx]];

  const Triangle triangle = triangles[result.triangleIdx];
  float3 hitNormal = triangle.normal();

  CURAND_TYPE randomState1 = paths.random0[pathIdx];
  CURAND_TYPE randomState2 = paths.random1[pathIdx];

  const float3 shadowRayOrigin = result.point + hitNormal * OFFSET_EPSILON;

  float3 brightness = make_float3(0.f, 0.f, 0.f);

  for (uint32_t i = 0; i < lightTriangles; ++i)
  {
    float pdf;
    float3 shadowPoint;
    triangles[lightTriangleIds[i]].sample(pdf, shadowPoint, randomState1, randomState2);

    const float3 shadowRayDirection = shadowPoint - shadowRayOrigin;
    const Ray shadowRay(shadowRayOrigin, normalize(shadowRayDirection));
    const float shadowRayLength = length(shadowRayDirection);

    const Triangle lightTriangle = triangles[lightTriangleIds[i]];
    const Material lightTriangleMaterial = materials[triangleMaterialIds[lightTriangleIds[i]]];
    const float3 lightEmission = lightTriangleMaterial.colorEmission;

    RaycastResult shadowResult = rayCast<HitType::ANY>(shadowRay, bvh, triangles, shadowRayLength);

    if ((shadowResult && shadowResult.t >= shadowRayLength + OFFSET_EPSILON) || !shadowResult)
    {
      const float cosOmega = __saturatef(dot(normalize(shadowRayDirection), hitNormal));
      const float cosL = __saturatef(dot(-normalize(shadowRayDirection), lightTriangle.normal()));

      brightness += 1.f / (shadowRayLength * shadowRayLength * pdf) * lightEmission * cosL * cosOmega;
    }
  }

  paths.random0[pathIdx] = randomState1;
  paths.random1[pathIdx] = randomState2;

  const float3 filteredAmbient = paths.filters[pathIdx] * material.colorAmbient;
  const float3 filteredDiffuse = paths.filters[pathIdx] * material.colorDiffuse;
  const float3 fiteredEmission = paths.filters[pathIdx] * material.colorEmission;

  paths.colors[pathIdx] += fiteredEmission + filteredAmbient + brightness * filteredDiffuse / CUDART_PI_F;
}

inline __device__ float3 reflectionDirection(const float3 normal, const float3 incomingDirection) {

  const float cosT = dot(incomingDirection, normal);

  return incomingDirection - 2 * cosT * normal;
}

__global__ void
specularKernel(
    const glm::ivec2 canvasSize,
    const Queues queues,
    Paths paths,
    const Triangle* triangles,
    const uint32_t* lightTriangleIds,
    const uint32_t  lightTriangles,
    const uint32_t* triangleMaterialIds,
    const Material* materials,
    const Node* bvh)
{
  const uint32_t x = threadIdx.x + blockIdx.x * blockDim.x;
  const uint32_t y = threadIdx.y + blockIdx.y * blockDim.y;
  const uint32_t idx = x + y * canvasSize.x;

  if (idx >= *queues.specularQueueSize)
    return;

  const float3 float3_zero = make_float3(0.f, 0.f, 0.f);
  const uint32_t pathIdx = queues.specularQueue[idx];

  const Ray hitRay = paths.rays[pathIdx];
  const RaycastResult result = paths.results[pathIdx];
  const Material material = materials[triangleMaterialIds[result.triangleIdx]];

  const Triangle triangle = triangles[result.triangleIdx];
  float3 hitNormal = triangle.normal(result.uv);

  const float3 reflectionRayOrigin = result.point + hitNormal * OFFSET_EPSILON;
  const float3 reflectionRayDir = reflectionDirection(hitNormal, hitRay.direction);

  const Ray reflectionRay(reflectionRayOrigin, reflectionRayDir);
  RaycastResult reflectionResult = rayCast<HitType::CLOSEST>(reflectionRay, bvh, triangles, BIGT);

  float3 reflectionHitColor = make_float3(0.f, 0.f, 0.f);

  if (reflectionResult)
    reflectionHitColor = materials[triangleMaterialIds[reflectionResult.triangleIdx]].colorAmbient;

  const float3 filteredSpecular = material.colorSpecular * reflectionHitColor;

  paths.colors[pathIdx] += filteredSpecular;
}

__global__ void newPaths(
    Paths paths,
    Queues queues,
    Camera camera,
    const glm::fvec2 canvasSize)
{
  const uint32_t x = threadIdx.x + blockIdx.x * blockDim.x;
  const uint32_t y = threadIdx.y + blockIdx.y * blockDim.y;

  const glm::fvec2 nic = camera.normalizedImageCoordinateFromPixelCoordinate(x, y, canvasSize);
  const Ray ray = camera.generateRay(nic, static_cast<float>(canvasSize.x)/canvasSize.y);

  const int idx = x + y*canvasSize.x;
  paths.rays[idx] = ray;
  paths.pixels[idx] = make_float2(x, y);
  queues.extensionQueue[idx] = idx;
  paths.colors[idx] = make_float3(0.f, 0.f, 0.f);
  paths.filters[idx] = make_float3(1.f, 1.f, 1.f);
}

__global__ void castExtensionRays(Paths paths, Queues queues, const glm::fvec2 canvasSize, const Triangle* triangles, const Node* bvh, const Material* materials, const unsigned int* traingelMaterialIds)
{
  const int x = threadIdx.x + blockIdx.x * blockDim.x;
  const int y = threadIdx.y + blockIdx.y * blockDim.y;
  const int idx = x + y * canvasSize.x;

  Ray ray = paths.rays[idx];
  RaycastResult result = rayCast<HitType::CLOSEST>(ray, bvh, triangles, BIGT);
  paths.results[idx] = result;
}

void CudaRenderer::reset()
{
  currentPath = 1;
}

void CudaRenderer::resize(const glm::ivec2 size)
{
  queues.resize(size);
  paths.resize(size);

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
  
  initRand<<<grid, block, 0, streams[0]>>>(devDirectionVectors64, devScrambleConstants64, paths.random0, size);
  initRand<<<grid, block, 0, streams[1]>>>(devDirectionVectors64 + 10000, devScrambleConstants64, paths.random1, size);
  
  CUDA_CHECK(cudaFree(devDirectionVectors64));
  CUDA_CHECK(cudaFree(devScrambleConstants64));

#else
  initRand<<<grid, block, 0, streams[0]>>>(0, paths.random0, size);
  initRand<<<grid, block, 0, streams[1]>>>(5, paths.random1, size);
#endif

  CUDA_CHECK(cudaStreamDestroy(streams[0]));
  CUDA_CHECK(cudaStreamDestroy(streams[1]));

  CUDA_CHECK(cudaDeviceSynchronize());
}


CudaRenderer::CudaRenderer() : lastCamera(), lastSize(), currentPath(1)
{
  uint32_t cudaDeviceCount = 0;
  int cudaDevices[8];
  uint32_t cudaDevicesCount = 8;

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
  queues.release();
  paths.release();
}

void CudaRenderer::pathTraceToCanvas(GLTexture& canvas, const Camera& camera, Model& model)
{
  if (model.getNTriangles() == 0)
    return;

  const glm::ivec2 canvasSize = canvas.getSize();
  const bool diffCamera = std::memcmp(&camera, &lastCamera, sizeof(Camera));
  const bool diffSize = (canvasSize != lastSize);

  const dim3 block(BLOCKWIDTH, BLOCKWIDTH);
  const dim3 grid( (canvasSize.x + block.x - 1) / block.x, (canvasSize.y + block.y - 1) / block.y);

  if (diffCamera != 0 || diffSize != 0)
  {
    lastCamera = camera;
    lastSize = canvasSize;
    currentPath = 1;

    queues.reset();

    newPaths<<<grid, block>>>(paths, queues, camera, canvasSize);
    castExtensionRays<<<grid, block>>>(paths, queues, canvasSize, model.getDeviceTriangles(), model.getDeviceBVH(), model.getDeviceMaterials(), model.getDeviceTriangleMaterialIds());
  }

  auto surfaceObj = canvas.getCudaMappedSurfaceObject();

  logicKernel<<<grid, block>>>(
      canvasSize,
      queues,
      paths,
      model.getDeviceMaterials(),
      model.getDeviceTriangleMaterialIds());

  CUDA_CHECK(cudaDeviceSynchronize());

  diffuseKernel<<<grid, block>>>(
      canvasSize,
      queues,
      paths,
      model.getDeviceTriangles(),
      model.getDeviceLightIds(),
      model.getNLights(),
      model.getDeviceTriangleMaterialIds(),
      model.getDeviceMaterials(),
      model.getDeviceBVH()
      );

  CUDA_CHECK(cudaDeviceSynchronize());
  *queues.diffuseQueueSize = 0;

  specularKernel<<<grid, block>>>(
      canvasSize,
      queues,
      paths,
      model.getDeviceTriangles(),
      model.getDeviceLightIds(),
      model.getNLights(),
      model.getDeviceTriangleMaterialIds(),
      model.getDeviceMaterials(),
      model.getDeviceBVH()
      );

  CUDA_CHECK(cudaDeviceSynchronize());
  *queues.specularQueueSize = 0;

  writeToCanvas<<<grid, block>>>(
      canvasSize,
      surfaceObj,
      paths,
      currentPath
      );

  ++currentPath;

  CUDA_CHECK(cudaDeviceSynchronize());
  canvas.cudaUnmap();
}

