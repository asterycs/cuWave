#ifndef UTILS_HPP
#define UTILS_HPP

#ifdef __CUDACC__
  #define CUDA_HOST_DEVICE __host__ __device__
  #define CUDA_HOST __host__
#else
  #define CUDA_HOST_DEVICE
  #define CUDA_HOST
#endif

#include "glm/glm.hpp"

#include <vector_types.h>
#include <vector_functions.h>
#include "vector_math.hpp"

#include <string>
#include <iostream>
#include <memory>
#include <vector>
#include <algorithm>
#include <numeric>

#include <curand.h>

#define WWIDTH 640
#define WHEIGHT 480

class Triangle;
class Model;

#ifndef NDEBUG
  #define GL_CHECK(call) do { \
          call; \
          CheckOpenGLError(#call, __FILE__, __LINE__); \
      } while (0)

  #define IL_CHECK(call) do { \
        call; \
        CheckILError(#call, __FILE__, __LINE__); \
    } while (0)

#else
  #define GL_CHECK(call) call
  #define IL_CHECK(call) call
#endif


#ifndef NDEBUG
  #define CUDA_CHECK(call) do { \
          call;\
          CheckCudaError(#call, __FILE__, __LINE__); \
      } while (0)

  #define CURAND_CHECK(call) do { \
	    curandStatus_t status = (call); \
		CheckCurandError(status, __FILE__, __LINE__); \
	  } while(0)
#else
  #define CUDA_CHECK(call) call
  #define CURAND_CHECK(call) call
#endif

CUDA_HOST void CheckCurandError(const curandStatus_t, const char* fname, int line);
CUDA_HOST void CheckCudaError(const char* stmt, const char* fname, int line);
CUDA_HOST void CheckILError(const char* stmt, const char* fname, int line);
CUDA_HOST void CheckOpenGLError(const char* call, const char* fname, int line);

CUDA_HOST std::string readFile(const std::string& filePath);

CUDA_HOST bool fileExists(const std::string& fileName);

CUDA_HOST_DEVICE float3 glm42float3(const glm::fvec4 g);
CUDA_HOST_DEVICE float3 glm32float3(const glm::fvec3 g);

struct Material
{
  float3 colorAmbient;
  float3 colorDiffuse;
  float3 colorEmission;
  float3 colorSpecular;
  float3 colorTransparent;

  float shininess;
  float refractionIndex;
  enum ShadingMode
  {
	  GORAUD = 1,
	  HIGHLIGHT = 2,
	  REFLECTION = 3,
	  TRANSPARENCY_GLASS_REFLECTION = 4,
	  REFLECTION_FRESNEL = 5,
	  TRANSPARENCY_REFLECTION = 6,
	  TRANSPARENCY_REFLECTION_FRESNEL = 7
  } mode;

  // TextureIndex?

  Material() : shininess(1.f), refractionIndex(1.f), mode(HIGHLIGHT)
  {
    colorAmbient 	= make_float3(0.f, 0.f, 0.f);
    colorDiffuse 	= make_float3(0.f, 0.f, 0.f);
    colorSpecular	= make_float3(0.f, 0.f, 0.f);
    colorEmission	= make_float3(0.f, 0.f, 0.f);
    colorTransparent= make_float3(0.f, 0.f, 0.f);
  };

};


struct AABB
{
  float3 max;
  float3 min;

  CUDA_HOST_DEVICE AABB(const float3& a, const float3& b) : max(fmaxf(a, b)), min(fminf(a, b)) { }
  CUDA_HOST_DEVICE AABB() : max(make_float3(0.f, 0.f, 0.f)), min(make_float3(0.f, 0.f, 0.f)) { }

  CUDA_HOST_DEVICE void add(const Triangle& v);
  CUDA_HOST_DEVICE void add(const float3 v);
  CUDA_HOST_DEVICE float area() const;
  CUDA_HOST_DEVICE unsigned int maxAxis() const;
};

struct Node
{
  AABB bbox;
  int startTri;
  int nTri; // exclusive
  int rightIndex;
};

std::ostream& operator<<(std::ostream &os, const float3 v);
std::ostream& operator<<(std::ostream &os, const Node& pn);

struct Ray
{
  float3 origin;
  float3 direction;

  CUDA_HOST_DEVICE  Ray(const float3 o, const float3 d) : origin(o), direction(d) {};
  CUDA_HOST_DEVICE  Ray() = default;
};

struct RaycastResult {
  int triangleIdx;
  float t;
  float2 uv;


  CUDA_HOST_DEVICE  RaycastResult(const unsigned int i,
    const float t,
    const float2& uv)
    :
    triangleIdx(i),
    t(t),
    uv(uv)
  {}

  CUDA_HOST_DEVICE  RaycastResult()
    :
    triangleIdx(-1),
    t(999999.f),
    uv()
  {}

  CUDA_HOST_DEVICE  operator bool() const { return (triangleIdx != -1); }
};

#endif
