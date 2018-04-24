#ifndef VECTOR_MATH_HPP
#define VECTOR_MATH_HPP

#include "cuda_runtime.h"

#ifndef __CUDACC__
#include <math.h>

inline float fminf(float a, float b)
{
  return a < b ? a : b;
}

inline float fmaxf(float a, float b)
{
  return a > b ? a : b;
}

inline int max(int a, int b)
{
  return a > b ? a : b;
}

inline int min(int a, int b)
{
  return a < b ? a : b;
}

inline float rsqrtf(float x)
{
    return 1.0f / sqrtf(x);
}
#endif




inline __device__ __host__ float lerp(const float a, const float b, const float t)
{
  return a + t * (b - a);
}

inline __device__ __host__ float clamp(const float f, const float a, const float b)
{
  return fmaxf(a, fminf(f, b));
}

inline __host__ __device__ float2 make_float2(const float s)
{
  return make_float2(s, s);
}

inline __host__ __device__ float2 operator-(float2 &a)
{
  return make_float2(-a.x, -a.y);
}

inline __host__ __device__ float2 operator+(const float2 a, const float2 b)
{
  return make_float2(a.x + b.x, a.y + b.y);
}

inline __host__ __device__ void operator+=(float2 &a, float2 b)
{
  a.x += b.x; a.y += b.y;
}

inline __host__ __device__ float2 operator-(const float2 a, const float2 b)
{
  return make_float2(a.x - b.x, a.y - b.y);
}
inline __host__ __device__ void operator-=(float2 &a, const float2 b)
{
  a.x -= b.x; a.y -= b.y;
}

inline __host__ __device__ float2 operator*(const float2 a, const float2 b)
{
  return make_float2(a.x * b.x, a.y * b.y);
}
inline __host__ __device__ float2 operator*(const float2 a, const float s)
{
  return make_float2(a.x * s, a.y * s);
}
inline __host__ __device__ float2 operator*(const float s, const float2 a)
{
  return make_float2(a.x * s, a.y * s);
}
inline __host__ __device__ void operator*=(float2 &a, const float s)
{
  a.x *= s; a.y *= s;
}

inline __host__ __device__ float2 operator/(const float2 a, const float2 b)
{
  return make_float2(a.x / b.x, a.y / b.y);
}

inline __host__ __device__ float2 operator/(const float2 a, const float s)
{
  float inv = 1.0f / s;
  return a * inv;
}

inline __host__ __device__ float2 operator/(const float s, const float2 a)
{
  float inv = 1.0f / s;
  return a * inv;
}

inline __host__ __device__ void operator/=(float2 &a, const float s)
{
  float inv = 1.0f / s;
  a *= inv;
}

inline __device__ __host__ float2 lerp(float2 a, float2 b, float t)
{
  return a + t*(b-a);
}

inline __device__ __host__ float2 clamp(const float2 v, const float a, const float b)
{
  return make_float2(clamp(v.x, a, b), clamp(v.y, a, b));
}

inline __device__ __host__ float2 clamp(const float2 v, const float2 a, const float2 b)
{
  return make_float2(clamp(v.x, a.x, b.x), clamp(v.y, a.y, b.y));
}

inline __host__ __device__ float dot(const float2 a, const float2 b)
{
  return a.x * b.x + a.y * b.y;
}

inline __host__ __device__ float length(const float2 v)
{
  return sqrtf(dot(v, v));
}

inline __host__ __device__ float2 normalize(const float2 v)
{
  float invLen = rsqrtf(dot(v, v));
  return v * invLen;
}

inline __host__ __device__ float2 floor(const float2 v)
{
  return make_float2(floor(v.x), floor(v.y));
}

inline __host__ __device__ float2 reflect(const float2 i, const float2 n)
{
  return i - 2.0f * n * dot(n,i);
}

inline __host__ __device__ float2 fabs(const float2 v)
{
  return make_float2(fabs(v.x), fabs(v.y));
}

inline __host__ __device__ float3 make_float3(const float s)
{
  return make_float3(s, s, s);
}

inline __host__ __device__ float3 make_float3(const float2 a)
{
  return make_float3(a.x, a.y, 0.0f);
}

inline __host__ __device__ float3 make_float3(const float2 a, const float s)
{
  return make_float3(a.x, a.y, s);
}

inline __host__ __device__ float3 make_float3(const float4 a)
{
  return make_float3(a.x, a.y, a.z);
}

inline __host__ __device__ float3 operator-(const float3 &a)
{
  return make_float3(-a.x, -a.y, -a.z);
}

inline __host__ __device__ float3 abs(const float3 &a)
{
  return make_float3(fabsf(a.x), fabsf(a.y), fabsf(a.z));
}

inline __host__ __device__ float3 fmax(const float3 &a, const float3 &b)
{
  return make_float3(max(a.x, b.x), max(a.y, b.y), max(a.z, b.z));
}

inline __host__ __device__ float fmin_compf(const float3 &v)
{
  return fminf(v.x, fminf(v.y, v.z));
}

inline __host__ __device__ float3 fmin(const float3 &a, const float3 &b)
{
  return make_float3(min(a.x, b.x), min(a.y, b.y), min(a.z, b.z));
}

static __inline__ __host__ __device__ float3 fminf(const float3 a, const float3 b)
{
  return make_float3(fminf(a.x,b.x), fminf(a.y,b.y), fminf(a.z,b.z));
}

static __inline__ __host__ __device__ float3 fmaxf(const float3 a, const float3 b)
{
  return make_float3(fmaxf(a.x,b.x), fmaxf(a.y,b.y), fmaxf(a.z,b.z));
}

inline __host__ __device__ float3 operator+(float3 a, float3 b)
{
  return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

inline __host__ __device__ float3 operator+(const float3 a, const float b)
{
  return make_float3(a.x + b, a.y + b, a.z + b);
}

inline __host__ __device__ void operator+=(float3 &a, const float3 b)
{
  a.x += b.x; a.y += b.y; a.z += b.z;
}

inline __host__ __device__ float3 operator-(const float3 a, const float3 b)
{
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

inline __host__ __device__ float3 operator-(const float3 a, const float b)
{
  return make_float3(a.x - b, a.y - b, a.z - b);
}

inline __host__ __device__ void operator-=(float3 &a, const float3 b)
{
  a.x -= b.x; a.y -= b.y; a.z -= b.z;
}

inline __host__ __device__ float3 operator*(const float3 a, const float3 b)
{
  return make_float3(a.x * b.x, a.y * b.y, a.z * b.z);
}

inline __host__ __device__ float3 operator*(const float3 a, const float s)
{
  return make_float3(a.x * s, a.y * s, a.z * s);
}

inline __host__ __device__ float3 operator*(const float s, const float3 a)
{
  return make_float3(a.x * s, a.y * s, a.z * s);
}

inline __host__ __device__ void operator*=(float3 &a, const float s)
{
  a.x *= s; a.y *= s; a.z *= s;
}

inline __host__ __device__ float3 operator/(const float3 a, const float3 b)
{
  return make_float3(a.x / b.x, a.y / b.y, a.z / b.z);
}

inline __host__ __device__ float3 operator/(const float3 a, const float s)
{
  float inv = 1.0f / s;
  return a * inv;
}

inline __host__ __device__ float3 operator/(const float s, const float3 a)
{
  float inv = 1.0f / s;
  return a * inv;
}

inline __host__ __device__ void operator/=(float3 &a, const float s)
{
  float inv = 1.0f / s;
  a *= inv;
}

inline __device__ __host__ float3 lerp(const float3 a, const float3 b, const float t)
{
  return a + t*(b-a);
}

inline __device__ __host__ bool operator!=(const float3 a, const float3 b)
{
  return a.x == b.x && a.y == b.y && a.z == b.z;
}

inline __device__ __host__ float3 clamp(const float3 v, const float a, const float b)
{
  return make_float3(clamp(v.x, a, b), clamp(v.y, a, b), clamp(v.z, a, b));
}

inline __device__ __host__ float3 clamp(const float3 v, const float3 a, const float3 b)
{
  return make_float3(clamp(v.x, a.x, b.x), clamp(v.y, a.y, b.y), clamp(v.z, a.z, b.z));
}

inline __host__ __device__ float dot(const float3 a, const float3 b)
{
  return a.x * b.x + a.y * b.y + a.z * b.z;
}

inline __host__ __device__ float3 cross(const float3 a, const float3 b)
{
  return make_float3(a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x);
}

inline __host__ __device__ float length(const float3 v)
{
  return sqrtf(dot(v, v));
}

inline __host__ __device__ float3 normalize(const float3 v)
{
  float invLen = rsqrtf(dot(v, v));
  return v * invLen;
}

inline __host__ __device__ float3 floor(const float3 v)
{
  return make_float3(floor(v.x), floor(v.y), floor(v.z));
}

inline __host__ __device__ float3 reflect(const float3 i, const float3 n)
{
  return i - 2.0f * n * dot(n,i);
}

inline __host__ __device__ float3 fabs(float3 v)
{
  return make_float3(fabs(v.x), fabs(v.y), fabs(v.z));
}

inline __host__ __device__ float4 make_float4(float s)
{
  return make_float4(s, s, s, s);
}

inline __host__ __device__ float4 make_float4(float3 a)
{
  return make_float4(a.x, a.y, a.z, 0.0f);
}

inline __host__ __device__ float4 make_float4(float3 a, float w)
{
  return make_float4(a.x, a.y, a.z, w);
}

inline __host__ __device__ float4 operator-(const float4 &a)
{
  return make_float4(-a.x, -a.y, -a.z, -a.w);
}

static __inline__ __host__ __device__ float4 fminf(const float4 a, const float4 b)
{
  return make_float4(fminf(a.x,b.x), fminf(a.y,b.y), fminf(a.z,b.z), fminf(a.w,b.w));
}

static __inline__ __host__ __device__ float4 fmaxf(const float4 a, const float4 b)
{
  return make_float4(fmaxf(a.x,b.x), fmaxf(a.y,b.y), fmaxf(a.z,b.z), fmaxf(a.w,b.w));
}

inline __host__ __device__ float4 operator+(const float4 a, const float4 b)
{
  return make_float4(a.x + b.x, a.y + b.y, a.z + b.z,  a.w + b.w);
}

inline __host__ __device__ void operator+=(float4 &a, const float4 b)
{
    a.x += b.x; a.y += b.y; a.z += b.z; a.w += b.w;
}

inline __host__ __device__ float4 operator-(const float4 a, const float4 b)
{
  return make_float4(a.x - b.x, a.y - b.y, a.z - b.z,  a.w - b.w);
}
inline __host__ __device__ void operator-=(float4 &a, const float4 b)
{
  a.x -= b.x; a.y -= b.y; a.z -= b.z; a.w -= b.w;
}

inline __host__ __device__ float4 operator*(const float4 a, const float s)
{
  return make_float4(a.x * s, a.y * s, a.z * s, a.w * s);
}

inline __host__ __device__ float4 operator*(const float s, const float4 a)
{
  return make_float4(a.x * s, a.y * s, a.z * s, a.w * s);
}

inline __host__ __device__ void operator*=(float4 &a, const float s)
{
  a.x *= s; a.y *= s; a.z *= s; a.w *= s;
}

inline __host__ __device__ float4 operator/(const float4 a, const float4 b)
{
  return make_float4(a.x / b.x, a.y / b.y, a.z / b.z, a.w / b.w);
}

inline __host__ __device__ float4 operator/(const float4 a, const float s)
{
  float inv = 1.0f / s;
  return a * inv;
}
inline __host__ __device__ float4 operator/(const float s, const float4 a)
{
  float inv = 1.0f / s;
  return a * inv;
}

inline __host__ __device__ void operator/=(float4 &a, const float s)
{
  float inv = 1.0f / s;
  a *= inv;
}

inline __device__ __host__ float4 lerp(const float4 a, const float4 b, const float t)
{
  return a + t * (b - a);
}

inline __device__ __host__ float4 clamp(const float4 v, const float a, const float b)
{
  return make_float4(clamp(v.x, a, b), clamp(v.y, a, b), clamp(v.z, a, b), clamp(v.w, a, b));
}

inline __device__ __host__ float4 clamp(const float4 v, const float4 a, const float4 b)
{
  return make_float4(clamp(v.x, a.x, b.x), clamp(v.y, a.y, b.y), clamp(v.z, a.z, b.z), clamp(v.w, a.w, b.w));
}

inline __host__ __device__ float dot(const float4 a, const float4 b)
{
  return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
}

inline __host__ __device__ float length(const float4 r)
{
  return sqrtf(dot(r, r));
}

inline __host__ __device__ float4 normalize(const float4 v)
{
  float invLen = rsqrtf(dot(v, v));
  return v * invLen;
}

inline __host__ __device__ float4 floor(const float4 v)
{
  return make_float4(floor(v.x), floor(v.y), floor(v.z), floor(v.w));
}


inline __host__ __device__ float4 fabs(const float4 v)
{
  return make_float4(fabs(v.x), fabs(v.y), fabs(v.z), fabs(v.w));
}

#endif /* VECTOR_MATH_HPP_ */
