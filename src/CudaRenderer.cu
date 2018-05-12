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

__device__ uint32_t mix(uint32_t a, uint32_t b, uint32_t c)
{
  a -= b; a -= c; a ^= (c>>13);
  b -= c; b -= a; b ^= (a<<8);
  c -= a; c -= b; c ^= (b>>13);
  a -= b; a -= c; a ^= (c>>12);
  b -= c; b -= a; b ^= (a<<16);
  c -= a; c -= b; c ^= (b>>5);
  a -= b; a -= c; a ^= (c>>3);
  b -= c; b -= a; b ^= (a<<10);
  c -= a; c -= b; c ^= (b>>15);

  return c;
}

__device__ float getNextRandom(const glm::ivec2 canvasSize, const uint32_t idx, uint32_t consumed, const float* floats, const uint32_t* scrambleConstants)
{
  const uint32_t totalFloats = canvasSize.x*canvasSize.y*PREGEN_RANDS;
  const uint32_t scrambleConst = scrambleConstants[idx];

  const float f = floats[((PREGEN_RANDS*idx)+consumed) % totalFloats];

  //const uint32_t i = mix(scrambleConst, idx, f * 4294967296);
  //const float rf = 2.3283064365386963e-10f * i;

  return f;
}

__device__ bool bboxIntersect(const AABB box, const float3 origin,
    const float3 inverseDirection, float& t)
{
  float3 tmin = make_float3(-BIGT, -BIGT, -BIGT), tmax = make_float3(BIGT, BIGT,
      BIGT);

  const float3 tdmin = (box.min - origin) * inverseDirection;
  const float3 tdmax = (box.max - origin) * inverseDirection;

  tmin = fminf(tdmin, tdmax);
  tmax = fmaxf(tdmin, tdmax);

  const float tmind = fmin_compf(tmin);
  const float tmaxd = fmin_compf(tmax);

  t = fminf(tmind, tmaxd);

  return tmaxd >= tmind && !(tmaxd < 0.f && tmind < 0.f);
}

__device__ bool rayTriangleIntersection(const Ray ray, const Triangle& triangle,
    float& t, float2& uv)
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
  ANY, CLOSEST
};

template<const HitType hitType>
__device__
RaycastResult rayCast(const Ray ray, const Node* bvh, const Triangle* triangles,
    const float maxT)
{
  float tMin = maxT;
  int32_t minTriIdx = -1;
  float2 minUV;
  RaycastResult result;
  const float3 inverseDirection = make_float3(1.f, 1.f, 1.f) / ray.direction;

  int32_t ptr = 0;
  unsigned int stack[16] { 0 };
  int32_t i = -1;
  float t = 0;
  float2 uv;
  bool getNextNode = true;

  while (ptr >= 0)
  {
    uint32_t currentNodeIdx = stack[ptr];
    Node currentNode = bvh[currentNodeIdx];

    if (currentNode.rightIndex == -1)
    {
      getNextNode = false;

      if (i >= currentNode.startTri
          && i < currentNode.startTri + currentNode.nTri)
      {
        if (rayTriangleIntersection(ray, triangles[i], t, uv))
        {

          if (t < tMin)
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

      }
      else
      {
        i = currentNode.startTri;
      }

    }
    else
    {
      const AABB leftBox = bvh[stack[ptr] + 1].bbox;
      const AABB rightBox = bvh[currentNode.rightIndex].bbox;

      float leftt, rightt;

      uint32_t hitMask =
          bboxIntersect(leftBox, ray.origin, inverseDirection, leftt) ?
              LEFT_HIT_BIT : 0x00;
      hitMask =
          bboxIntersect(rightBox, ray.origin, inverseDirection, rightt) ?
              hitMask | RIGHT_HIT_BIT : hitMask;

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

__device__ void writeToCanvas(const uint32_t x, const uint32_t y,
    const cudaSurfaceObject_t& surfaceObj, const glm::ivec2 canvasSize,
    const float3 data)
{
  const float4 out = make_float4(data.x, data.y, data.z, 1.f);
  surf2Dwrite(out, surfaceObj, (canvasSize.x - 1 - x) * sizeof(out), y);
  return;
}

__device__ float3 readFromCanvas(const uint32_t x, const uint32_t y,
    const cudaSurfaceObject_t& surfaceObj, const glm::ivec2 canvasSize)
{
  float4 in;
  surf2Dread(&in, surfaceObj, (canvasSize.x - 1 - x) * sizeof(in), y);
  const float3 ret = make_float3(in.x, in.y, in.z);

  return ret;
}

__global__ void logicKernel(const glm::ivec2 canvasSize, Queues queues,
    Paths paths, const Material* materials, const uint32_t* triangleMaterialIds)
{
  const uint32_t x = threadIdx.x + blockIdx.x * blockDim.x;
  const uint32_t y = threadIdx.y + blockIdx.y * blockDim.y;
  const uint32_t idx = x + y * canvasSize.x;

  const float3 float3_zero = make_float3(0.f, 0.f, 0.f);

  if (x >= canvasSize.x  || y >= canvasSize.y)
    return;

  const RaycastResult result = paths.result[idx];
  const uint32_t rayNr = paths.rayNr[idx];

  if (!result || rayNr >= 5)
  {
    const uint32_t new_idx = atomicAdd(queues.newPathQueueSize, 1);
    queues.newPathQueue[new_idx] = idx;
    return;
  }else
  {
    const uint32_t new_idx = atomicAdd(queues.extensionQueueSize, 1);
    queues.extensionQueue[new_idx] = idx;
  }

  const Material material = materials[triangleMaterialIds[result.triangleIdx]];

  if (material.colorDiffuse != float3_zero)
  {
    const uint32_t new_idx = atomicAdd(queues.diffuseQueueSize, 1);
    queues.diffuseQueue[new_idx] = idx;
  }

  /*if (material.colorSpecular != float3_zero)
   {
   const uint32_t new_idx = atomicAdd(queues.specularQueueSize, 1);
   queues.specularQueue[new_idx] = idx;
   }*/

  return;
}

__global__ void writeToCanvas(const glm::ivec2 canvasSize,
    cudaSurfaceObject_t canvas, Paths paths)
{
  const uint32_t x = threadIdx.x + blockIdx.x * blockDim.x;
  const uint32_t y = threadIdx.y + blockIdx.y * blockDim.y;
  const int idx = x + y * canvasSize.x;

  if (x >= canvasSize.x || y >= canvasSize.y)
    return;

  const uint32_t currentPath = paths.pathNr[idx];
  const float3 newColor = paths.color[idx];
  const uint2 pixel = paths.pixel[idx];
  float3 oldColor = readFromCanvas(pixel.x, pixel.y, canvas, canvasSize);
  float3 blend = static_cast<float>(currentPath - 1) / currentPath * oldColor
      + 1.f / currentPath * newColor;

  writeToCanvas(pixel.x, pixel.y, canvas, canvasSize, blend);
}

typedef struct
{
  float3 col[3];
} float33;

__device__
inline void setZero(float33& m)
{
  m.col[0] = make_float3(0.0f, 0.0f, 0.0f);
  m.col[1] = make_float3(0.0f, 0.0f, 0.0f);
  m.col[2] = make_float3(0.0f, 0.0f, 0.0f);
}

__device__
inline float3 operator*(const float33 m, const float3 v)
{
  float3 res;
  res.x = m.col[0].x * v.x + m.col[1].x * v.y + m.col[2].x * v.z;
  res.y = m.col[0].y * v.x + m.col[1].y * v.y + m.col[2].y * v.z;
  res.z = m.col[0].z * v.x + m.col[1].z * v.y + m.col[2].z * v.z;

  return res;
}

__device__ float33 getBasis(const float3 n)
{

  float33 R;

  float3 Q = n;
  const float3 absq = abs(Q);
  float absqmin = fmin(absq);

  if (absq.x == absqmin)
    Q.x = 1;
  else if (absq.y == absqmin)
    Q.y = 1;
  else
    Q.z = 1;

  float3 T = normalize(cross(Q, n));
  float3 B = normalize(cross(n, T));

  R.col[0] = T;
  R.col[1] = B;
  R.col[2] = n;

  return R;
}

__global__ void diffuseKernel(const glm::ivec2 canvasSize, const Queues queues,
    Paths paths, const Triangle* triangles, const uint32_t* lightTriangleIds,
    const uint32_t lightTriangles, const uint32_t* triangleMaterialIds,
    const Material* materials, const Node* bvh)
{
  const uint32_t x = threadIdx.x + blockIdx.x * blockDim.x;
  const uint32_t y = threadIdx.y + blockIdx.y * blockDim.y;
  const uint32_t idx = x + y * canvasSize.x;

  if (idx >= *queues.diffuseQueueSize)
    return;

  const float3 float3_zero = make_float3(0.f, 0.f, 0.f);
  const uint32_t pathIdx = queues.diffuseQueue[idx];

  const RaycastResult result = paths.result[pathIdx];
  const Material& material = materials[triangleMaterialIds[result.triangleIdx]];

  const Triangle triangle = triangles[result.triangleIdx];
  float3 hitNormal = triangle.normal();
  uint32_t consumed = paths.randomNumbersConsumed[pathIdx];

  const float3 shadowRayOrigin = result.point + hitNormal * OFFSET_EPSILON;

  float3 brightness = make_float3(0.f, 0.f, 0.f);

  for (uint32_t i = 0; i < lightTriangles; ++i)
  {
    float pdf;
    float3 shadowPoint;

	const float r0 = getNextRandom(canvasSize, pathIdx, consumed, paths.randomFloatsX, paths.scrambleConstants);
	const float r1 = getNextRandom(canvasSize, pathIdx, consumed, paths.randomFloatsY, paths.scrambleConstants);
	++consumed;

    triangles[lightTriangleIds[i]].sample(pdf, shadowPoint, r0, r1);

    const float3 shadowRayDirection = shadowPoint - shadowRayOrigin;
    const Ray shadowRay(shadowRayOrigin, normalize(shadowRayDirection));
    const float shadowRayLength = length(shadowRayDirection);

    const Triangle lightTriangle = triangles[lightTriangleIds[i]];
    const Material lightTriangleMaterial = materials[triangleMaterialIds[lightTriangleIds[i]]];
    const float3 lightEmission = lightTriangleMaterial.colorEmission;

    RaycastResult shadowResult = rayCast<HitType::ANY>(shadowRay, bvh, triangles, shadowRayLength);

    if ((shadowResult && shadowResult.t >= shadowRayLength + OFFSET_EPSILON) || !shadowResult)
    {
      const float cosOmega = __saturatef(
          dot(normalize(shadowRayDirection), hitNormal));
      const float cosL = __saturatef(
          dot(-normalize(shadowRayDirection), lightTriangle.normal()));

      brightness += 1.f / (shadowRayLength * shadowRayLength * pdf) * lightEmission * cosL * cosOmega;
    }
  }

  const float3 filteredAmbient = paths.throughput[pathIdx]
      * material.colorAmbient;
  const float3 filteredDiffuse = paths.throughput[pathIdx]
      * material.colorDiffuse;
  const float3 fiteredEmission = paths.throughput[pathIdx]
      * material.colorEmission;

  paths.color[pathIdx] += fiteredEmission + filteredAmbient + brightness / lightTriangles * filteredDiffuse / CUDART_PI_F;

  paths.randomNumbersConsumed[pathIdx] = consumed;
}

__global__ void newPathsKernel(const glm::ivec2 canvasSize, const Queues queues,
    Paths paths, const Camera camera)
{
  const uint32_t x = threadIdx.x + blockIdx.x * blockDim.x;
  const uint32_t y = threadIdx.y + blockIdx.y * blockDim.y;
  const int idx = x + y * canvasSize.x;

  if (idx >= *queues.newPathQueueSize)
    return;

  const uint32_t pathIdx = queues.newPathQueue[idx];
  const uint2 pixel = paths.pixel[pathIdx];

  const glm::fvec2 nic = camera.normalizedImageCoordinateFromPixelCoordinate(pixel.x, pixel.y, canvasSize);
  const Ray ray = camera.generateRay(nic,
      static_cast<float>(canvasSize.x) / canvasSize.y);

  paths.ray[pathIdx] = ray;

  const uint32_t newExtensionIdx = atomicAdd(queues.extensionQueueSize, 1);
  queues.extensionQueue[newExtensionIdx] = pathIdx;
  paths.color[pathIdx] = make_float3(0.f, 0.f, 0.f);
  paths.throughput[pathIdx] = make_float3(1.f, 1.f, 1.f);
  paths.p[pathIdx] = 1.f;
  paths.rayNr[pathIdx] = 1;
  paths.pathNr[pathIdx] += 1;
}

inline __device__ float3 reflectionDirection(const float3 normal,
    const float3 incomingDirection)
{

  const float cosT = dot(incomingDirection, normal);

  return incomingDirection - 2 * cosT * normal;
}

__global__ void createExtensionKernel(const glm::ivec2 canvasSize,
    const Queues queues, Paths paths, const Triangle* triangles,
    const uint32_t* triangleMaterialIds, const Material* materials)
{
  const uint32_t x = threadIdx.x + blockIdx.x * blockDim.x;
  const uint32_t y = threadIdx.y + blockIdx.y * blockDim.y;
  const uint32_t idx = x + y * canvasSize.x;

  if (idx >= *queues.extensionQueueSize)
    return;

  const uint32_t pathIdx = queues.extensionQueue[idx];

  const RaycastResult result = paths.result[pathIdx];
  const Triangle triangle = triangles[result.triangleIdx];
  const Material& material = materials[triangleMaterialIds[result.triangleIdx]];
  float3 hitNormal = triangle.normal();
  uint32_t consumed = paths.randomNumbersConsumed[pathIdx];

  float33 B = getBasis(hitNormal);
  float3 extensionDir;

  do
  {
	const float r0 = getNextRandom(canvasSize, pathIdx, consumed, paths.randomFloatsX, paths.scrambleConstants);
	const float r1 = getNextRandom(canvasSize, pathIdx, consumed, paths.randomFloatsY, paths.scrambleConstants);
	++consumed;

    extensionDir = make_float3(r0 * 2.0f - 1.0f, r1 * 2.0f - 1.0f, 0.f);
  } while ((extensionDir.x * extensionDir.x + extensionDir.y * extensionDir.y) >= 1);

  extensionDir.z = sqrt(1 - extensionDir.x * extensionDir.x - extensionDir.y * extensionDir.y);
  extensionDir = B * extensionDir;
  extensionDir = normalize(extensionDir); // Unnecessary
  const float3 extensionOrig = result.point + OFFSET_EPSILON * hitNormal;
  const Ray extensionRay(extensionOrig, extensionDir);

  float cosO = dot(extensionDir, hitNormal);
  float p = cosO * dot(extensionDir, hitNormal) * (1.f / CUDART_PI_F);
  float3 throughput = material.colorDiffuse / CUDART_PI_F* dot(extensionDir, hitNormal);

  paths.ray[pathIdx] = extensionRay;
  paths.throughput[pathIdx] = paths.throughput[pathIdx] * throughput;
  paths.p[pathIdx] *= p;
  paths.rayNr[pathIdx] += 1;
  paths.randomNumbersConsumed[pathIdx] = consumed;
}

/*__global__ void
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

 if (idx >= *queues.specularQueueSize - 1)
	 return;

 const float3 float3_zero = make_float3(0.f, 0.f, 0.f);
 const uint32_t pathIdx = queues.specularQueue[idx];

 const Ray hitRay = paths.ray[pathIdx];
 const RaycastResult hitResult = paths.result[pathIdx];

 const Triangle triangle = triangles[hitResult.triangleIdx];
 float3 hitNormal = triangle.normal();

 const float3 reflectionRayOrigin = hitResult.point + hitNormal * OFFSET_EPSILON;
 const float3 reflectionRayDir = reflectionDirection(hitNormal, hitRay.direction);

 const Ray reflectionRay(reflectionRayOrigin, reflectionRayDir);

 const uint32_t newRayIdx = atomicAdd(paths.secondaryPathCount, 1);
 paths.secondaryRays[newRayIdx] = reflectionRay;
 paths.secondaryPixels[newRayIdx] = currentPixel;
 paths.secondaryFilters[newRayIdx] = paths.primaryFilters[primaryIdx];
}*/

__global__ void resetAllPaths(Paths paths, Camera camera,
    const glm::fvec2 canvasSize)
{
  const uint32_t x = threadIdx.x + blockIdx.x * blockDim.x;
  const uint32_t y = threadIdx.y + blockIdx.y * blockDim.y;
  const int idx = x + y * canvasSize.x;

  if (x >= canvasSize.x || y >= canvasSize.y)
    return;

  const glm::fvec2 nic = camera.normalizedImageCoordinateFromPixelCoordinate(x,
      y, canvasSize);
  const Ray ray = camera.generateRay(nic,
      static_cast<float>(canvasSize.x) / canvasSize.y);

  paths.ray[idx] = ray;
  paths.color[idx] = make_float3(0.f, 0.f, 0.f);
  paths.throughput[idx] = make_float3(1.f, 1.f, 1.f);
  paths.pixel[idx] = make_uint2(x, y);
  paths.p[idx] = 1.f;
  paths.rayNr[idx] = 1;
  paths.pathNr[idx] = 1;
}

__global__ void castRays(Paths paths, const glm::ivec2 canvasSize,
    const Triangle* triangles, const Node* bvh, const Material* materials,
    const unsigned int* traingelMaterialIds)
{
  const int x = threadIdx.x + blockIdx.x * blockDim.x;
  const int y = threadIdx.y + blockIdx.y * blockDim.y;
  const int idx = x + y * canvasSize.x;

  if (x >= canvasSize.x || y >= canvasSize.y)
    return;

  const Ray ray = paths.ray[idx];
  RaycastResult result = rayCast<HitType::CLOSEST>(ray, bvh, triangles, BIGT);
  paths.result[idx] = result;
}

__global__ void generateRandom(
		curandStateScrambledSobol32* state,
		const glm::ivec2 size,
		const uint32_t nRands,
		float* output
    )
{
  const int x = threadIdx.x + blockIdx.x * blockDim.x;
  const int y = threadIdx.y + blockIdx.y * blockDim.y;

  if (x >= size.x || y >= size.y)
	return;

  curandStateScrambledSobol32 localState = state[x + size.x * y];

  for (uint32_t i = 0; i < nRands; ++i)
	  output[nRands*(x + y*size.x) + i] = curand_uniform(&localState);

  state[x + size.x * y] = localState;
}

__global__ void initRand(curandDirectionVectors32_t* sobolDirectionVectors, uint32_t* sobolScrambleConstants, curandStateScrambledSobol32* state, const glm::ivec2 size)
{
  const int x = threadIdx.x + blockIdx.x * blockDim.x;
  const int y = threadIdx.y + blockIdx.y * blockDim.y;

  if (x >= size.x || y >= size.y)
    return;

  const uint32_t scrIdx = x + size.x * y;
  const uint32_t dirIdx = (x + size.x * y) % 10000;

  curandDirectionVectors32_t* dir = &sobolDirectionVectors[dirIdx];
  uint32_t scr = sobolScrambleConstants[scrIdx];
  curandStateScrambledSobol32 localState;

  curand_init(*dir, scr, 0, &localState);

  state[x + size.x * y] = localState;
}

void CudaRenderer::reset()
{
  queues.reset();
  callcntr = 0;

  dim3 block(BLOCKWIDTH, BLOCKWIDTH);
  dim3 grid((lastSize.x + block.x - 1) / block.x, (lastSize.y + block.y - 1) / block.y);

  resetAllPaths<<<grid, block>>>(paths, lastCamera, lastSize);
  CUDA_CHECK(cudaDeviceSynchronize());
}

void CudaRenderer::resize(const glm::ivec2 size)
{
  queues.resize(size);
  paths.resize(size);

  lastSize = size;

  dim3 block(BLOCKWIDTH, BLOCKWIDTH);
  dim3 grid((size.x + block.x - 1) / block.x, (size.y + block.y - 1) / block.y);

  curandStateDevVecX.resize(size.x * size.y);
  curandStateDevVecY.resize(size.x * size.y);

  auto* curandStateDevXRaw = thrust::raw_pointer_cast(&curandStateDevVecX[0]);
  auto* curandStateDevYRaw = thrust::raw_pointer_cast(&curandStateDevVecY[0]);

  curandDirectionVectors32_t* hostDirectionVectors32;
  uint32_t* hostScrambleConstants32;

  curandDirectionVectors32_t* devDirectionVectors32;
  uint32_t* devScrambleConstants32;

  CURAND_CHECK(curandGetDirectionVectors32(&hostDirectionVectors32, CURAND_SCRAMBLED_DIRECTION_VECTORS_32_JOEKUO6));
  CURAND_CHECK(curandGetScrambleConstants32(&hostScrambleConstants32));

  CUDA_CHECK(cudaMalloc((void **)&(devDirectionVectors32),             20000 * sizeof(curandDirectionVectors32_t)));
  CUDA_CHECK(cudaMemcpy(devDirectionVectors32, hostDirectionVectors32, 20000 * sizeof(curandDirectionVectors32_t), cudaMemcpyHostToDevice));

  CUDA_CHECK(cudaMalloc((void **)&(devScrambleConstants32),              size.x * size.y * sizeof(uint32_t)));
  CUDA_CHECK(cudaMemcpy(devScrambleConstants32, hostScrambleConstants32, size.x * size.y * sizeof(uint32_t), cudaMemcpyHostToDevice));

  initRand<<<grid, block>>>(devDirectionVectors32, devScrambleConstants32, curandStateDevXRaw, size);
  initRand<<<grid, block>>>(devDirectionVectors32 + 10000, devScrambleConstants32, curandStateDevYRaw, size);

  CUDA_CHECK(cudaFree(devDirectionVectors32));
  CUDA_CHECK(cudaFree(devScrambleConstants32));

  generateRandom<<<grid, block>>>(curandStateDevXRaw, size, PREGEN_RANDS, paths.randomFloatsX);
  generateRandom<<<grid, block>>>(curandStateDevYRaw, size, PREGEN_RANDS, paths.randomFloatsY);

  CUDA_CHECK(cudaMemset(paths.randomNumbersConsumed, 0, size.x*size.y*sizeof(uint32_t)));

  CUDA_CHECK(cudaDeviceSynchronize());

  reset();
}

CudaRenderer::CudaRenderer()
    : lastCamera(), lastSize(), callcntr(0)
{
  uint32_t cudaDeviceCount = 0;
  int cudaDevices[8];
  uint32_t cudaDevicesCount = 8;

  cudaGLGetDevices(&cudaDeviceCount, cudaDevices, cudaDevicesCount,
      cudaGLDeviceListCurrentFrame);

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
  CUDA_CHECK(cudaDeviceReset());
}

void CudaRenderer::pathTraceToCanvas(GLTexture& canvas, const Camera& camera,
    Model& model)
{
  if (model.getNTriangles() == 0)
    return;

  const glm::ivec2 canvasSize = canvas.getSize();
  const bool diffCamera = std::memcmp(&camera, &lastCamera, sizeof(Camera));
  const bool diffSize = (canvasSize != lastSize);
  const auto surfaceObj = canvas.getCudaMappedSurfaceObject();

  const dim3 block(BLOCKWIDTH, BLOCKWIDTH);
  const dim3 grid((canvasSize.x + block.x - 1) / block.x, (canvasSize.y + block.y - 1) / block.y);

  if (diffCamera != 0 || diffSize != 0)
  {
    lastCamera = camera;

    reset();
  }

  /*testRnd<<<grid, block>>>(surfaceObj,
		    canvasSize,
		    paths.randomNumbersConsumed,
		    paths.randomFloats,
		    paths.scrambleConstants);*/


  castRays<<<grid, block>>>(paths, canvasSize, model.getDeviceTriangles(),
      model.getDeviceBVH(), model.getDeviceMaterials(),
      model.getDeviceTriangleMaterialIds());

  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaMemset(queues.extensionQueueSize, 0, sizeof(uint32_t)));

  logicKernel<<<grid, block>>>(canvasSize, queues, paths,
      model.getDeviceMaterials(), model.getDeviceTriangleMaterialIds());

  CUDA_CHECK(cudaDeviceSynchronize());

  diffuseKernel<<<grid, block>>>(canvasSize, queues, paths,
      model.getDeviceTriangles(), model.getDeviceLightIds(), model.getNLights(),
      model.getDeviceTriangleMaterialIds(), model.getDeviceMaterials(),
      model.getDeviceBVH());

  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaMemset(queues.diffuseQueueSize, 0, sizeof(uint32_t)));

  /*specularKernel<<<grid, block>>>(
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
  *queues.specularQueueSize = 0;*/

  writeToCanvas<<<grid, block>>>(canvasSize, surfaceObj, paths);

  CUDA_CHECK(cudaDeviceSynchronize());

  createExtensionKernel<<<grid, block>>>(canvasSize, queues, paths,
      model.getDeviceTriangles(), model.getDeviceTriangleMaterialIds(),
      model.getDeviceMaterials());

  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaMemset(queues.extensionQueueSize, 0, sizeof(uint32_t)));

  newPathsKernel<<<grid, block>>>(canvasSize, queues, paths, camera);

  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaMemset(queues.newPathQueueSize, 0, sizeof(uint32_t)));

  canvas.cudaUnmap();
}

