#ifndef TRIANGLE_HPP
#define TRIANGLE_HPP

#ifdef __CUDACC__
#define CUDA_FUNCTION __host__ __device__
#define CUDA_DEVICE_FUNCTION __device__
#define CUDA_HOST_FUNCTION __host__
#else
#define CUDA_FUNCTION
#define CUDA_DEVICE_FUNCTION
#define CUDA_HOST_FUNCTION
#endif

#include "Utils.hpp"

#include <vector_types.h>
#include "vector_math.hpp"

struct Vertex
{
  float3 p;
  float3 n;
  float2 t;

  CUDA_FUNCTION Vertex() : p(make_float3(0.0f)), n(make_float3(0.0f)), t(make_float2(0.0f)) {};
  CUDA_FUNCTION Vertex(const float3& pp, const float3& nn, const float2& tt) : p(pp), n(nn), t(tt) {};
};

CUDA_HOST_FUNCTION std::ostream& operator<<(std::ostream& os, const Vertex& v);
CUDA_HOST_FUNCTION std::istream& operator>>(std::istream& is, Vertex& v);

struct Triangle {
  Vertex vertices[3];

  CUDA_FUNCTION Triangle() = default;
  CUDA_FUNCTION Triangle(const Vertex v0, const Vertex v1, const Vertex v2);
  // Return float4(point, pdf) to registers instead of via reference to memory
  CUDA_DEVICE_FUNCTION float4 sample(float x, float y) const;
  CUDA_FUNCTION Triangle& operator=(const Triangle& that) = default;
  CUDA_FUNCTION float3 min() const;
  CUDA_FUNCTION float3 max() const;
  CUDA_FUNCTION AABB bbox() const;
  CUDA_FUNCTION float3 normal() const;
  CUDA_FUNCTION float3 normal(const float2 uv) const;
  CUDA_FUNCTION float3 center() const;
  CUDA_FUNCTION float area() const;
};

CUDA_HOST_FUNCTION std::ostream& operator<<(std::ostream &os, const Triangle& t);
CUDA_HOST_FUNCTION std::istream& operator>>(std::istream &is, Triangle& t);

#endif
