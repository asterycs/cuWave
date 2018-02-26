#ifndef UTILS_HPP
#define UTILS_HPP

#ifdef __CUDACC__
  #define CUDA_FUNCTION __host__ __device__
#else
  #define CUDA_FUNCTION
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

#define WWIDTH 600
#define WHEIGHT 600

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

#ifdef ENABLE_CUDA
  #ifndef NDEBUG
    #define CUDA_CHECK(call) do { \
            call;\
            CheckCudaError(#call, __FILE__, __LINE__); \
        } while (0)
  #else
    #define CUDA_CHECK(call) call
  #endif
#else
  #define CUDA_CHECK(call)
#endif

void CheckCudaError(const char* stmt, const char* fname, int line);
void CheckILError(const char* stmt, const char* fname, int line);
void CheckOpenGLError(const char* call, const char* fname, int line);

std::string readFile(const std::string& filePath);

bool fileExists(const std::string& fileName);

CUDA_FUNCTION float3 glm32cuda3(const glm::fvec3 v)
{
  return make_float3(v.x, v.y, v.z);
}

struct Material
{
  float3 colorAmbient;
  float3 colorDiffuse;
  float3 colorEmission;
  float3 colorSpecular;
  float3 colorTransparent;

  float shininess;
  float refrIdx;
  enum
  {
    GORAUD,
    PHONG,
    FRESNEL
  } shadingMode;

  // TextureIndex?

  Material() : shininess(1.f), refrIdx(1.f), shadingMode(GORAUD)
  {
    colorAmbient = make_float3(0.f, 0.f, 0.f);
    colorDiffuse = make_float3(0.f, 0.f, 0.f);
    colorSpecular= make_float3(0.f, 0.f, 0.f);
  };

};


struct AABB
{
  float3 max;
  float3 min;

  CUDA_FUNCTION AABB(const float3& a, const float3& b) : max(fmaxf(a, b)), min(fminf(a, b)) { }
  CUDA_FUNCTION AABB() : max(make_float3(0.f, 0.f, 0.f)), min(make_float3(0.f, 0.f, 0.f)) { }

  CUDA_FUNCTION void add(const Triangle& v);
  CUDA_FUNCTION void add(const float3 v);
  CUDA_FUNCTION float area() const;
  CUDA_FUNCTION unsigned int maxAxis() const;
};

struct Node
{
  AABB bbox;
  int startTri;
  int nTri; // exclusive
  int rightIndex;
};


struct Ray
{
  float3 origin;
  float3 direction;

  CUDA_FUNCTION Ray(const float3& o, const float3& d) : origin(o), direction(d) {};
  CUDA_FUNCTION Ray() = default;
};

struct RaycastResult {
  int triangleIdx;
  float t;
  float2 uv;
  float3 point;


  CUDA_FUNCTION RaycastResult(const unsigned int i,
    const float t,
    const float2& uv,
    const float3& point)
    :
    triangleIdx(i),
    t(t),
    uv(uv),
    point(point)
  {}

  CUDA_FUNCTION RaycastResult()
    :
    triangleIdx(-1),
    t(999999.f),
    uv(),
    point()
  {}

  CUDA_FUNCTION operator bool() const { return (triangleIdx != -1); }
};

#endif
