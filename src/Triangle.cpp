#include "Triangle.hpp"


CUDA_HOST_FUNCTION std::ostream& operator<<(std::ostream& os, const Vertex& v)
{
	  os << v.p.x << " " << v.p.y << " " << v.p.z << " " << v.n.x << " " << v.n.y << " " << v.n.z << " " << v.t.x << " " << v.t.y;
	  return os;
}

CUDA_HOST_FUNCTION std::istream& operator>>(std::istream& is, Vertex& v)
{
	  is >> v.p.x >> v.p.y >> v.p.z >> v.n.x >> v.n.y >> v.n.z >> v.t.x >> v.t.y;
	  return is;
}

CUDA_FUNCTION Triangle::Triangle(const Vertex v0, const Vertex v1, const Vertex v2) {
	vertices[0] = v0;
	vertices[1] = v1;
	vertices[2] = v2;
}

CUDA_FUNCTION bool Triangle::centroidIsInside(const AABB bbox) const
{
	const float3 center = this->center();

	if (center.x > bbox.min.x && center.x < bbox.max.x && center.y > bbox.min.y && center.y < bbox.max.y && center.z > bbox.min.z && center.z < bbox.max.z)
		return true;

    return false;
}

CUDA_FUNCTION bool Triangle::touches(const AABB bbox) const
{
    for (int i = 0; i < 3; ++i)
    {
        const float3 p = vertices[i].p;

        if (p.x >= bbox.min.x && p.x <= bbox.max.x && p.y >= bbox.min.y && p.y <= bbox.max.y && p.z >= bbox.min.z && p.z <= bbox.max.z)
            return true;
    }

    return false;
}


CUDA_DEVICE_FUNCTION float4 Triangle::sample(float x, float y) const
{
	const float3 v0 = vertices[1].p - vertices[0].p;
	const float3 v1 = vertices[2].p - vertices[0].p;

	if (x + y > 1.f)
	{
		if (x > y)
			x -= 0.5f;
		else
			y -= 0.5f;
	}

	const float3 point = vertices[0].p + x*v0 + y*v1;

	return make_float4(point.x, point.y, point.z, 1.0f/area());
}

CUDA_FUNCTION float3 Triangle::min() const
{
	return fminf(fminf(vertices[0].p, vertices[1].p), vertices[2].p);
}

CUDA_FUNCTION float3 Triangle::max() const
{
	return fmaxf(fmaxf(vertices[0].p, vertices[1].p), vertices[2].p);
}

CUDA_FUNCTION AABB Triangle::bbox() const
{
	return AABB(max(), min());
}

CUDA_FUNCTION float3 Triangle::normal() const
{
	return normalize(cross(vertices[1].p - vertices[0].p, vertices[2].p - vertices[0].p));
}

CUDA_FUNCTION float3 Triangle::normal(const float2 uv) const
{
	return normalize((1 - uv.x - uv.y) * vertices[0].n + uv.x * vertices[1].n + uv.y * vertices[2].n);
}

CUDA_FUNCTION float3 Triangle::center() const
{
	return (vertices[0].p + vertices[1].p + vertices[2].p) / 3.f;
}

CUDA_FUNCTION float Triangle::area() const
{
	const float3 e1 = vertices[1].p - vertices[0].p;
	const float3 e2 = vertices[2].p - vertices[0].p;

	return 0.5f * length(cross(e1, e2));
}

CUDA_HOST_FUNCTION std::ostream& operator<<(std::ostream &os, const Triangle& t)
{
	  os << t.vertices[0] << " " << t.vertices[1] << " " << t.vertices[2];

	  return os;
}

CUDA_HOST_FUNCTION std::istream& operator>>(std::istream &is, Triangle& t)
{
	  is >> t.vertices[0] >> t.vertices[1] >> t.vertices[2];

	  return is;
}
