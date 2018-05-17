#include "CudaRenderer.hpp"

#include <GL/glew.h>
#include <GL/gl.h>

#include <cuda.h>
#include <cuda_gl_interop.h>
#include <curand.h>
#include <curand_kernel.h>
#include <math_constants.h>
#include <cooperative_groups.h>

#include "Utils.hpp"
#include "Triangle.hpp"

#define BLOCKWIDTH 8
#define INTERSECT_EPSILON 0.0000001f
#define OFFSET_EPSILON 0.00001f
#define BIGT 99999.f
#define AIR_INDEX 1.f
#define MIN_PATHS 3
#define RUSSIAN_PROB 0.8f

enum RandDim
{
	SHADING = 0,
	RUSSIAN,
	LIGHT,
	DIFF0,
	DIFF1,
	DIFF2,
	DIFF3,
	EXT0,
	EXT1
};

#define LEFT_HIT_BIT 0x80000000
#define RIGHT_HIT_BIT 0x40000000

__device__ inline float fresnelReflectioncoefficient(const float sin2t, const float cosi, const float idx1, const float idx2)
{
  const float cost = sqrt(1 - sin2t);

  const float Rs = (idx1 * cosi - idx2 * cost) / (idx1 * cosi + idx2 * cost);
  const float Rp = (idx1 * cost - idx2 * cosi) / (idx1 * cost + idx2 * cosi);

  return (Rs * Rs + Rp * Rp) * 0.5f;
}

__device__ float3 createDirection(const float r0, const float r1)
{
	const float sinTheta = sqrtf(r0);
	const float cosTheta = sqrtf(1-sinTheta*sinTheta);

	const float psi = r1*2*CUDART_PI_F;

	return make_float3(sinTheta*cosf(psi), sinTheta*sinf(psi), cosTheta);
}

// Veach
__device__ inline float powerHeuristic(const float f, const float g)
{
    return (f*f)/(f*f + g*g);
}

// https://devblogs.nvidia.com/cuda-pro-tip-optimized-filtering-warp-aggregated-atomics/
__device__ int atomicAggInc(uint32_t *ctr) {
	auto g = cooperative_groups::coalesced_threads();
	uint32_t warp_res;

	if(g.thread_rank() == 0)
		warp_res = atomicAdd(ctr, g.size());
	return g.shfl(warp_res, 0) + g.thread_rank();
}

__device__ inline float scramble(const uint32_t scrambleConstant, const float f)
{
	const uint32_t i = static_cast<uint32_t>(f * 0x100000000) ^ scrambleConstant;

	const float r = i * 2.3283064365386963e-10f;

	return r;
}

__device__ float bboxIntersect(const AABB box, const float3 origin,	const float3 inverseDirection)
{
	float3 tmin = make_float3(-BIGT, -BIGT, -BIGT), tmax = make_float3(BIGT, BIGT, BIGT);

	const float3 tdmin = (box.min - origin) * inverseDirection;
	const float3 tdmax = (box.max - origin) * inverseDirection;

	tmin = fminf(tdmin, tdmax);
	tmax = fmaxf(tdmin, tdmax);

	const float tmind = fmin_compf(tmin);
	const float tmaxd = fmin_compf(tmax);

	return (tmaxd >= tmind && !(tmaxd < 0.f && tmind < 0.f)) ? fminf(tmind, tmaxd) : -1.f;
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
__device__ RaycastResult rayCast(const Ray ray, const Node* bvh,
		const Triangle* triangles, const float maxT)
{
	float tMin = maxT;
	int32_t minTriIdx = -1;
	float2 minUV;
	RaycastResult result;
	const float3 inverseDirection = make_float3(1.f, 1.f, 1.f) / ray.direction;

	int32_t ptr = 0;
	uint32_t stack[16] { 0 };
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

			if (i >= currentNode.startTri && i < currentNode.startTri + currentNode.nTri)
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

			float leftt = bboxIntersect(leftBox, ray.origin, inverseDirection);
			float rightt = bboxIntersect(rightBox, ray.origin, inverseDirection);

			uint32_t hitMask = leftt != -1.f ? LEFT_HIT_BIT : 0x00;
			hitMask = rightt != -1.f ? hitMask | RIGHT_HIT_BIT : hitMask;

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

	result.t = tMin;
	result.triangleIdx = minTriIdx;
	result.uv = minUV;

	return result;
}

__device__ void writeToCanvas(const uint32_t x, const uint32_t y,
		const cudaSurfaceObject_t& surfaceObj, const glm::ivec2 canvasSize,
		const float3 data)
{
	const float4 out = make_float4(__saturatef(data.x), __saturatef(data.y), __saturatef(data.z), 1.f);
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
		Paths paths, const Material* materials,
		const uint32_t* triangleMaterialIds,
		const Triangle* triangles
		)
{
	const uint32_t x = threadIdx.x + blockIdx.x * blockDim.x;
	const uint32_t y = threadIdx.y + blockIdx.y * blockDim.y;
	const uint32_t pathIdx = x + y * canvasSize.x;

	const float3 float3_zero = make_float3(0.f, 0.f, 0.f);

	if (x >= canvasSize.x || y >= canvasSize.y)
		return;

	const uint32_t scrambleConstant = paths.scrambleConstants[pathIdx];
	const float rf = scramble(scrambleConstant, paths.floats[RandDim::SHADING]);

	const RaycastResult result = paths.result[pathIdx];
	const uint32_t rayNr = paths.rayNr[pathIdx];

	if (!result)
	{
		const uint32_t new_idx = atomicAggInc(queues.newPathQueueSize);
		queues.newPathQueue[new_idx] = pathIdx;

		if (rayNr == 1)
			paths.color[pathIdx] = make_float3(0.3f, 0.3f, 0.3f);

		return;
	}else if (rayNr >= MIN_PATHS)
	{
		const float rr = scramble(scrambleConstant, paths.floats[RandDim::RUSSIAN]);

		if (rr > RUSSIAN_PROB) // terminated by Russian roulette
		{
			const uint32_t new_idx = atomicAggInc(queues.newPathQueueSize);
			queues.newPathQueue[new_idx] = pathIdx;
			return;
		}

		paths.p[pathIdx] *= RUSSIAN_PROB;
	}

	const Ray ray = paths.ray[pathIdx];
	const float3 normal = triangles[result.triangleIdx].normal(result.uv);
	const Material material = materials[triangleMaterialIds[result.triangleIdx]];

	uint32_t new_idx;

	switch (material.mode)
	{
		case (Material::TRANSPARENCY_REFLECTION_FRESNEL):
			{
				const float idx1 = AIR_INDEX;
				const float idx2 = material.refractionIndex;

				float refractionRatio;

				const bool outside = dot(ray.direction, normal) < 0.f;
				float3 flippedNormal = normal;

				if (outside)
					refractionRatio = idx2 / idx1;
				else
				{
					refractionRatio = idx1 / idx2;
					flippedNormal = -normal;
				}

				const float cosi = dot(ray.direction, -flippedNormal);

				if (sinf(acosf(cosi)) <= refractionRatio) // Refraction allowed
				{
					const float sin2t = abs((idx1 / idx2) * (idx1 / idx2) * (1 - cosi * cosi));
					const float twoR = outside ? 2.f*fresnelReflectioncoefficient(sin2t, cosi, idx1, idx2): 1.f;
					const float total = length(material.colorDiffuse) + twoR * length(material.colorSpecular) + (2.f-twoR) * length(material.colorTransparent);
					const float diffuseTreshold = length(material.colorDiffuse) / total;
					const float specularTreshold = diffuseTreshold + twoR * length(material.colorSpecular) / total;
					//const float transparentTreshold = diffuseTreshold + specularTreshold + ((1-R) / 0.5f) * length(material.colorTransparent) / total;

					if (rf < diffuseTreshold)
					{
						new_idx = atomicAggInc(queues.diffuseQueueSize);
						queues.diffuseQueue[new_idx] = pathIdx;
					}else if (rf > diffuseTreshold && rf < specularTreshold)
					{
						 new_idx = atomicAggInc(queues.specularQueueSize);
						 queues.specularQueue[new_idx] = pathIdx;
						 paths.throughput[pathIdx] *= material.colorSpecular * twoR;
					}else
					{
						 new_idx = atomicAggInc(queues.transparentQueueSize);
						 queues.transparentQueue[new_idx] = pathIdx;
						 paths.throughput[pathIdx] *= material.colorTransparent * (2.f-twoR);
					}

					break;
				} // Total reflection, handle as reflection using falltrough. Not entirely correct.
			}

		case (Material::REFLECTION_FRESNEL):
			{
				const float total = length(material.colorDiffuse) + length(material.colorSpecular);
				const float diffuseTreshold = length(material.colorDiffuse) / total;
				const float specularTreshold = diffuseTreshold + length(material.colorSpecular) / total;

				if (rf < diffuseTreshold)
				{
					new_idx = atomicAggInc(queues.diffuseQueueSize);
					queues.diffuseQueue[new_idx] = pathIdx;
				}else
				{
					 new_idx = atomicAggInc(queues.specularQueueSize);
					 queues.specularQueue[new_idx] = pathIdx;
				}
			}
			break;

		default:
			new_idx = atomicAggInc(queues.diffuseQueueSize);
			queues.diffuseQueue[new_idx] = pathIdx;
			break;
	}

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
	float3 oldColor = readFromCanvas(x, y, canvas, canvasSize);
	float3 blend = static_cast<float>(currentPath - 1) / currentPath * oldColor
			+ 1.f / currentPath * newColor;

	writeToCanvas(x, y, canvas, canvasSize, blend);
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
		Paths paths, const Triangle* triangles,
		const uint32_t* lightTriangleIds, const uint32_t lightTriangles,
		const uint32_t* triangleMaterialIds, const Material* materials,
		const Node* bvh)
{
	const uint32_t x = threadIdx.x + blockIdx.x * blockDim.x;
	const uint32_t y = threadIdx.y + blockIdx.y * blockDim.y;
	const uint32_t idx = x + y * canvasSize.x;

	if (idx >= *queues.diffuseQueueSize)
		return;

	const uint32_t pathIdx = queues.diffuseQueue[idx];
	const uint32_t scrambleConstant = paths.scrambleConstants[pathIdx];

	const RaycastResult result = paths.result[pathIdx];
	const Material material = materials[triangleMaterialIds[result.triangleIdx]];
	const Ray ray = paths.ray[pathIdx];
	const float3 hitNormal = triangles[result.triangleIdx].normal();

	const float3 shadowRayOrigin = ray.origin + ray.direction*result.t + hitNormal * OFFSET_EPSILON;

	float3 directLightning = make_float3(0.f, 0.f, 0.f);

	if (lightTriangles > 0) // TODO: Make check reduntant
	{
		// Choose light by uniform sampling
		const float p = paths.p[pathIdx];
		const float lightPdf = 1.f / lightTriangles;
		const float lightF = paths.floats[RandDim::LIGHT];
		const uint32_t lightIdx = lightF / lightPdf;

		float r0 = paths.floats[RandDim::DIFF0];
		float r1 = paths.floats[RandDim::DIFF1];

		r0 = scramble(scrambleConstant, r0);
		r1 = scramble(scrambleConstant, r1);

		const float4 pointPdf = triangles[lightTriangleIds[lightIdx]].sample(r0, r1);
		const float3 shadowPoint = make_float3(pointPdf.x, pointPdf.y, pointPdf.z);

		const float3 shadowRayDirection = shadowPoint - shadowRayOrigin;
		const Ray shadowRay(shadowRayOrigin, normalize(shadowRayDirection));
		const float shadowRayLength = length(shadowRayDirection);

		const Triangle lightTriangle = triangles[lightTriangleIds[lightIdx]];
		const Material lightTriangleMaterial = materials[triangleMaterialIds[lightTriangleIds[lightIdx]]];
		const float3 lightEmission = lightTriangleMaterial.colorEmission;

		const RaycastResult shadowResult = rayCast<HitType::ANY>(shadowRay, bvh, triangles, shadowRayLength);

		if ((shadowResult && shadowResult.t >= shadowRayLength + OFFSET_EPSILON) || !shadowResult)
		{
			const float misWeight = powerHeuristic(p*lightPdf*pointPdf.w, 2*CUDART_PI_F);
			const float cosOmega = __saturatef(dot(normalize(shadowRayDirection), hitNormal));
			const float cosL = __saturatef(dot(-normalize(shadowRayDirection), lightTriangle.normal()));

			directLightning += misWeight * 1.f / (shadowRayLength * shadowRayLength * lightPdf * pointPdf.w * p) * lightEmission * cosL * cosOmega;
		}


		const float33 B = getBasis(hitNormal);

		float r2 = scramble(scrambleConstant, paths.floats[RandDim::DIFF2]);
		float r3 = scramble(scrambleConstant, paths.floats[RandDim::DIFF3]);

		float3 brdfDir = createDirection(r2, r3);
		brdfDir = B * brdfDir;
		const Ray brdfRay(shadowRayOrigin, brdfDir);

		const RaycastResult brdfResult = rayCast<HitType::CLOSEST>(brdfRay, bvh, triangles, BIGT);

		if (brdfResult.triangleIdx == result.triangleIdx)
		{
			const float misWeight = powerHeuristic(2*CUDART_PI_F, lightPdf*pointPdf.w*p);

			const float cosOmega = __saturatef(dot(normalize(brdfDir), hitNormal));
			const float cosL = __saturatef(dot(-normalize(brdfDir), lightTriangle.normal()));

			directLightning += misWeight * 1.f / (length(brdfDir*brdfResult.t) * length(brdfDir*brdfResult.t) * 2* CUDART_PI_F * p) * lightEmission * cosL * cosOmega;
		}
	}

	const float3 currentTroughput = paths.throughput[pathIdx];
	const float3 filteredAmbient = currentTroughput * material.colorAmbient;
	const float3 filteredDiffuse = currentTroughput * material.colorDiffuse;
	const float3 fiteredEmission = paths.rayNr[pathIdx] == 1 ? currentTroughput * material.colorEmission : make_float3(0.f, 0.f, 0.f);

	paths.color[pathIdx] += fiteredEmission + filteredAmbient + directLightning * filteredDiffuse / CUDART_PI_F;
}

__global__ void newPathsKernel(const glm::ivec2 canvasSize, const Queues queues,
		const Paths paths, const Camera camera)
{
	const uint32_t x = threadIdx.x + blockIdx.x * blockDim.x;
	const uint32_t y = threadIdx.y + blockIdx.y * blockDim.y;
	const int idx = x + y * canvasSize.x;

	if (idx >= *queues.newPathQueueSize)
		return;

	const uint32_t pathIdx = queues.newPathQueue[idx];

	const uint32_t xCoordinate = pathIdx % canvasSize.x;
	const uint32_t yCoordinate = pathIdx / canvasSize.x;

	const glm::fvec2 nic = camera.normalizedImageCoordinateFromPixelCoordinate(
			xCoordinate, yCoordinate, canvasSize);
	const Ray ray = camera.generateRay(nic, static_cast<float>(canvasSize.x) / canvasSize.y);

	paths.ray[pathIdx] = ray;

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

inline __device__ float3 refractionDirection(const float cosInAng, const float sin2t, const float3 normal, const float3 incoming, const float index1, const float index2)
{
    return index1 / index2 * incoming + (index1 / index2 * cosInAng - sqrt(1 - sin2t)) * normal;
}

__global__ void diffuseExtensionKernel(const glm::ivec2 canvasSize,
		const Queues queues, const Paths paths, const Triangle* triangles,
		const uint32_t* triangleMaterialIds, const Material* materials, const uint32_t lightTriangles)
{
	const uint32_t x = threadIdx.x + blockIdx.x * blockDim.x;
	const uint32_t y = threadIdx.y + blockIdx.y * blockDim.y;
	const uint32_t idx = x + y * canvasSize.x;

	if (idx >= *queues.diffuseQueueSize)
		return;

	const uint32_t pathIdx = queues.diffuseQueue[idx];
	const uint32_t scrambleConstant = paths.scrambleConstants[idx];

	const Ray ray = paths.ray[pathIdx];
	const RaycastResult result = paths.result[pathIdx];
	const Triangle triangle = triangles[result.triangleIdx];
	const Material& material = materials[triangleMaterialIds[result.triangleIdx]];
	float3 hitNormal = triangle.normal();

	float33 B = getBasis(hitNormal);

	float r0 = scramble(scrambleConstant, paths.floats[RandDim::EXT0]);
	float r1 = scramble(scrambleConstant, paths.floats[RandDim::EXT1]);

	float3 extensionDir = createDirection(r0, r1);

	extensionDir = B * extensionDir;
	extensionDir = normalize(extensionDir); // Unnecessary
	const float3 extensionOrig = ray.origin + ray.direction*result.t + OFFSET_EPSILON * hitNormal;
	const Ray extensionRay(extensionOrig, extensionDir);

	const float cosO = dot(extensionDir, hitNormal);
	const float p = cosO * dot(extensionDir, hitNormal) * (1.f / CUDART_PI_F);
	float3 throughput = material.colorDiffuse / CUDART_PI_F * dot(extensionDir, hitNormal);

	paths.ray[pathIdx] = extensionRay;
	paths.throughput[pathIdx] = paths.throughput[pathIdx] * throughput;
	paths.p[pathIdx] *= p;
	paths.rayNr[pathIdx] += 1;
}

__global__ void
 specularKernel(
	 const glm::ivec2 canvasSize,
	 const Queues queues,
	 const Paths paths,
	 const Triangle* triangles
	 )
 {
	 const uint32_t x = threadIdx.x + blockIdx.x * blockDim.x;
	 const uint32_t y = threadIdx.y + blockIdx.y * blockDim.y;
	 const uint32_t idx = x + y * canvasSize.x;

	 if (idx >= *queues.specularQueueSize)
		 return;

	 const float3 float3_zero = make_float3(0.f, 0.f, 0.f);
	 const uint32_t pathIdx = queues.specularQueue[idx];

	 const Ray hitRay = paths.ray[pathIdx];
	 const RaycastResult hitResult = paths.result[pathIdx];

	 const Triangle triangle = triangles[hitResult.triangleIdx];
	 const float3 hitNormal = triangle.normal(hitResult.uv);

	 const float3 reflectionRayOrigin = hitRay.origin + hitRay.direction*hitResult.t + hitNormal * OFFSET_EPSILON;
	 const float3 reflectionRayDir = reflectionDirection(hitNormal, hitRay.direction);

	 const Ray reflectionRay(reflectionRayOrigin, reflectionRayDir);

	 paths.ray[pathIdx] = reflectionRay;
	 //paths.throughput[pathIdx] = paths.throughput[pathIdx] * material.colorSpecular; // Handled in logicKernel
	 paths.rayNr[pathIdx] += 1;
 }

__global__ void
 transparentKernel(
	 const glm::ivec2 canvasSize,
	 const Queues queues,
	 const Paths paths,
	 const Triangle* triangles,
	 const uint32_t* triangleMaterialIds,
	 const Material* materials
	 )
 {
	const uint32_t x = threadIdx.x + blockIdx.x * blockDim.x;
	const uint32_t y = threadIdx.y + blockIdx.y * blockDim.y;
	const uint32_t idx = x + y * canvasSize.x;

	if (idx >= *queues.transparentQueueSize)
		return;

	const float3 float3_zero = make_float3(0.f, 0.f, 0.f);
	const uint32_t pathIdx = queues.transparentQueue[idx];

	const Ray hitRay = paths.ray[pathIdx];
	const RaycastResult hitResult = paths.result[pathIdx];

	const Triangle triangle = triangles[hitResult.triangleIdx];
	float3 hitNormal = triangle.normal(hitResult.uv);
	const Material material = materials[triangleMaterialIds[hitResult.triangleIdx]];

	const float idx1 = AIR_INDEX;
	const float idx2 = material.refractionIndex;

	const bool outside = dot(hitRay.direction, hitNormal) < 0.f;

	if (!outside)
		hitNormal = -hitNormal;

	const float cosi = __saturatef(dot(hitRay.direction, hitNormal));
	const float sin2t = abs((idx1 / idx2) * (idx1 / idx2) * (1 - cosi * cosi));

	const float3 transOrig = hitRay.origin + hitRay.direction*hitResult.t - hitNormal * OFFSET_EPSILON;
	const float3 transDir = refractionDirection(cosi, sin2t, hitNormal, hitRay.direction, idx1, idx2);

	paths.ray[pathIdx] = Ray(transOrig, transDir);
	//paths.throughput[pathIdx] = paths.throughput[pathIdx] * material.colorTransparent; // Handled in logicKernel
	paths.rayNr[pathIdx] += 1;
 }

__global__ void resetAllPaths(Paths paths, Camera camera,
		const glm::fvec2 canvasSize)
{
	const uint32_t x = threadIdx.x + blockIdx.x * blockDim.x;
	const uint32_t y = threadIdx.y + blockIdx.y * blockDim.y;
	const int idx = x + y * canvasSize.x;

	if (x >= canvasSize.x || y >= canvasSize.y)
		return;

	const glm::fvec2 nic = camera.normalizedImageCoordinateFromPixelCoordinate(
			x, y, canvasSize);
	const Ray ray = camera.generateRay(nic,
			static_cast<float>(canvasSize.x) / canvasSize.y);

	paths.ray[idx] = ray;
	paths.color[idx] = make_float3(0.f, 0.f, 0.f);
	paths.throughput[idx] = make_float3(1.f, 1.f, 1.f);
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

__global__ void generateRandom(curandStateSobol32* state, const glm::ivec2 size,
		const uint32_t nRands, float* output)
{
	const int x = threadIdx.x + blockIdx.x * blockDim.x;
	const int y = threadIdx.y + blockIdx.y * blockDim.y;

	if (x >= size.x || y >= size.y)
		return;

	curandStateSobol32 localState = state[x + size.x * y];

	for (uint32_t i = 0; i < nRands; ++i)
		output[nRands * (x + y * size.x) + i] = curand_uniform(&localState);

	state[x + size.x * y] = localState;
}

__global__ void initRand(const int seq, curandState_t* const curandStateDevPtr,
		const glm::ivec2 size)
{
	const int x = threadIdx.x + blockIdx.x * blockDim.x;
	const int y = threadIdx.y + blockIdx.y * blockDim.y;

	if (x >= size.x || y >= size.y)
		return;

	curandState_t localState;
	curand_init(x + y * size.x, seq, 0, &localState);
	curandStateDevPtr[x + y * size.x] = localState;
}

void CudaRenderer::reset()
{
	queues.reset();
	callcntr = 0;

	dim3 block(BLOCKWIDTH, BLOCKWIDTH);
	dim3 grid((lastSize.x + block.x - 1) / block.x,
			(lastSize.y + block.y - 1) / block.y);

	resetAllPaths<<<grid, block>>>(paths, lastCamera, lastSize);
	CUDA_CHECK(cudaDeviceSynchronize());
}

void CudaRenderer::resize(const glm::ivec2 size)
{
	queues.resize(size);
	paths.resize(size);

	lastSize = size;

	dim3 block(BLOCKWIDTH, BLOCKWIDTH);
	dim3 grid((size.x + block.x - 1) / block.x,
			(size.y + block.y - 1) / block.y);

	uint32_t* hostScrambleConstants;

	CURAND_CHECK(curandGetScrambleConstants32(&hostScrambleConstants));
	CUDA_CHECK(cudaMemcpy(paths.scrambleConstants, hostScrambleConstants, size.x * size.y * sizeof(uint32_t), cudaMemcpyHostToDevice));

	reset();
}

CudaRenderer::CudaRenderer() :
		lastCamera(), lastSize(), callcntr(0)
{
	uint32_t cudaDeviceCount = 0;
	int cudaDevices[8];
	uint32_t cudaDevicesCount = 8;

	cudaGLGetDevices(&cudaDeviceCount, cudaDevices, cudaDevicesCount,
			cudaGLDeviceListCurrentFrame);

	if (cudaDeviceCount < 1)
		throw std::runtime_error("No CUDA devices available");


	CURAND_CHECK(curandCreateGenerator(&randGen, CURAND_RNG_QUASI_SOBOL32));
	CURAND_CHECK(curandSetQuasiRandomGeneratorDimensions(randGen, 32));

	CUDA_CHECK(cudaSetDevice(cudaDevices[0]));

	resize(glm::ivec2(WWIDTH, WHEIGHT));
}

CudaRenderer::~CudaRenderer()
{
	queues.release();
	paths.release();
}

void CudaRenderer::pathTraceToCanvas(GLTexture& canvas, const Camera& camera,
		CudaModel& model)
{
	if (model.getNTriangles() == 0)
		return;

	const glm::ivec2 canvasSize = canvas.getSize();
	const bool diffCamera = std::memcmp(&camera, &lastCamera, sizeof(Camera));
	const bool diffSize = (canvasSize != lastSize);
	const auto surfaceObj = canvas.getCudaMappedSurfaceObject();

	const dim3 block(BLOCKWIDTH, BLOCKWIDTH);
	const dim3 grid((canvasSize.x + block.x - 1) / block.x,
			(canvasSize.y + block.y - 1) / block.y);

	if (diffCamera != 0 || diffSize != 0)
	{
		lastCamera = camera;
		reset();
	}

    CURAND_CHECK(curandGenerateUniform(randGen, paths.floats, 32));

	castRays<<<grid, block>>>(paths, canvasSize, model.getDeviceTriangles(),
			model.getDeviceBVH(), model.getDeviceMaterials(),
			model.getDeviceTriangleMaterialIds());

	CUDA_CHECK(cudaDeviceSynchronize());

	logicKernel<<<grid, block>>>(canvasSize, queues, paths,
			model.getDeviceMaterials(), model.getDeviceTriangleMaterialIds(), model.getDeviceTriangles());

	CUDA_CHECK(cudaDeviceSynchronize());

	diffuseKernel<<<grid, block>>>(canvasSize, queues, paths,
			model.getDeviceTriangles(), model.getDeviceLightIds(),
			model.getNLights(), model.getDeviceTriangleMaterialIds(),
			model.getDeviceMaterials(), model.getDeviceBVH());

	CUDA_CHECK(cudaDeviceSynchronize());

	specularKernel<<<grid, block>>>(
		canvasSize,
		queues,
		paths,
		model.getDeviceTriangles()
	);

	 CUDA_CHECK(cudaDeviceSynchronize());
	 CUDA_CHECK(cudaMemset(queues.specularQueueSize, 0, sizeof(uint32_t)));

	transparentKernel<<<grid, block>>>(
		canvasSize,
		queues,
		paths,
		model.getDeviceTriangles(),
		model.getDeviceTriangleMaterialIds(),
		model.getDeviceMaterials()
	 );

	 CUDA_CHECK(cudaDeviceSynchronize());
	 CUDA_CHECK(cudaMemset(queues.transparentQueueSize, 0, sizeof(uint32_t)));

	writeToCanvas<<<grid, block>>>(canvasSize, surfaceObj, paths);

	diffuseExtensionKernel<<<grid, block>>>(canvasSize, queues, paths,
			model.getDeviceTriangles(), model.getDeviceTriangleMaterialIds(),
			model.getDeviceMaterials(), model.getNLights());

	CUDA_CHECK(cudaDeviceSynchronize());
	CUDA_CHECK(cudaMemset(queues.diffuseQueueSize, 0, sizeof(uint32_t)));

	newPathsKernel<<<grid, block>>>(canvasSize, queues, paths, camera);

	CUDA_CHECK(cudaDeviceSynchronize());
	CUDA_CHECK(cudaMemset(queues.newPathQueueSize, 0, sizeof(uint32_t)));

	canvas.cudaUnmap();
}

