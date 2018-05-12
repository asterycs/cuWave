#ifndef TRIANGLE_HPP
#define TRIANGLE_HPP

#ifdef __CUDACC__
#define CUDA_FUNCTION __host__ __device__
#define CUDA_DEVICE_FUNCTION __device__
#else
#define CUDA_FUNCTION
#define CUDA_DEVICE_FUNCTION
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

struct Triangle {
  Vertex vertices[3];

  CUDA_FUNCTION Triangle() = default;
  CUDA_FUNCTION Triangle(const Vertex v0, const Vertex v1, const Vertex v2) {
		vertices[0] = v0;
		vertices[1] = v1;
		vertices[2] = v2;
	}

  CUDA_DEVICE_FUNCTION void sample(float& pdf, float3& point, float x, float y) const
  {
    pdf = 1.0f / area();

    const float3 v0 = vertices[1].p - vertices[0].p;
    const float3 v1 = vertices[2].p - vertices[0].p;

    /*if (x + y > 1.f)
    {
    	if (x > y)
    		x -= 0.5f;
    	else
    		y -= 0.5f;
    }*/

    point = vertices[0].p + x*v0 + y*v1;
  }

  CUDA_FUNCTION Triangle& operator=(const Triangle& that) = default;

  CUDA_FUNCTION inline float3 min() const
  {
		return fminf(fminf(vertices[0].p, vertices[1].p), vertices[2].p);
	}

  CUDA_FUNCTION inline float3 max() const
  {
		return fmaxf(fmaxf(vertices[0].p, vertices[1].p), vertices[2].p);
	}

  CUDA_FUNCTION inline AABB bbox() const
  {
    return AABB(max(), min());
  }

  CUDA_FUNCTION float3 normal() const
  {
		return normalize(cross(vertices[1].p - vertices[0].p, vertices[2].p - vertices[0].p));
	}

  CUDA_FUNCTION float3 normal(const float2& uv) const
  {
    return normalize((1 - uv.x - uv.y) * vertices[0].n + uv.x * vertices[1].n + uv.y * vertices[2].n);
  }

  CUDA_FUNCTION float3 center() const
  {
    return (vertices[0].p + vertices[1].p + vertices[2].p) / 3.f;
  }

  CUDA_FUNCTION float area() const
  {
    const float3 e1 = vertices[1].p - vertices[0].p;
    const float3 e2 = vertices[2].p - vertices[0].p;

    return 0.5f * length(cross(e1, e2));
  }
};

#endif
