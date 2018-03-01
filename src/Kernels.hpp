#ifndef KERNELS_HPP
#define KERNELS_HPP

inline __device__ float fresnelReflectioncoefficient(const float sin2t, const float cosi, const float idx1, const float idx2)
{
  const float cost = sqrt(1 - sin2t);

  float Rs = (idx1 * cosi - idx2 * cost) / (idx1 * cosi + idx2 * cost);
  Rs = Rs * Rs;

  float Rp = (idx2 * cosi - idx1 * cost) / (idx2 * cosi + idx1 * cost);
  Rp = Rp * Rp;

  return (Rs + Rp) * 0.5f;
}

__device__ glm::fmat3 getBasis(const glm::fvec3 n) {

  glm::fmat3 R;

  glm::fvec3 Q = n;
  const glm::fvec3 absq = glm::abs(Q);
  float absqmin = glm::compMin(absq);

  for (int i = 0; i < 3; ++i) {
    if (absq[i] == absqmin) {
      Q[i] = 1;
      break;
    }
  }

  glm::fvec3 T = glm::normalize(glm::cross(Q, n));
  glm::fvec3 B = glm::normalize(glm::cross(n, T));

  R[0] = T;
  R[1] = B;
  R[2] = n;

  return R;
}

__device__ bool bboxIntersect(const AABB box, const glm::fvec3 origin, const glm::fvec3 inverseDirection, float& t)
{
  glm::fvec3 tmin(-BIGT), tmax(BIGT);

  const glm::fvec3 tdmin = (box.min - origin) * inverseDirection;
  const glm::fvec3 tdmax = (box.max - origin) * inverseDirection;

  tmin = glm::min(tdmin, tdmax);
  tmax = glm::max(tdmin, tdmax);

  const float tmind = glm::compMax(tmin);
  const float tmaxd = glm::compMin(tmax);

  t = fminf(tmind, tmaxd);

  return tmaxd >= tmind && !(tmaxd < 0.f && tmind < 0.f);
}

__device__ void debug_fvec3(const glm::fvec3 v)
{
  printf("%f %f %f\n", v.x, v.y, v.z);
}

inline __device__ glm::fvec3 reflectionDirection(const glm::vec3 normal, const glm::vec3 incoming) {

  const float cosT = glm::dot(incoming, normal);

  return incoming - 2 * cosT * normal;
}

inline __device__ glm::fvec3 refractionDirection(const float cosInAng, const float sin2t, const glm::vec3 normal, const glm::vec3 incoming, const float index1, const float index2)
{
    return index1 / index2 * incoming + (index1 / index2 * cosInAng - sqrt(1 - sin2t)) * normal;
}

__device__ bool rayTriangleIntersection(const Ray ray, const Triangle& triangle, float& t, glm::fvec2& uv)
{
  /* Möller-Trumbore algorithm
   * https://en.wikipedia.org/wiki/M%C3%B6ller%E2%80%93Trumbore_intersection_algorithm
   */

  // TODO: Experiment with __ldg

  const glm::vec3 vertex0 = triangle.vertices[0].p;

  const glm::fvec3 edge1 = triangle.vertices[1].p - vertex0;
  const glm::fvec3 edge2 = triangle.vertices[2].p - vertex0;

  const glm::fvec3 h = glm::cross(ray.direction, edge2);
  const float a = glm::dot(edge1, h);

  if (a > -INTERSECT_EPSILON && a < INTERSECT_EPSILON)
    return false;

  const float f = __fdividef(1.f, a);
  const glm::fvec3 s = ray.origin - vertex0;
  const float u = f * glm::dot(s, h);

  if (u < 0.f || u > 1.0f)
    return false;

  const glm::fvec3 q = glm::cross(s, edge1);
  const float v = f * glm::dot(ray.direction, q);

  if (v < 0.0 || u + v > 1.0)
    return false;

  t = f * glm::dot(edge2, q);

  if (t > INTERSECT_EPSILON)
  {
    uv = glm::fvec2(u, v);
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

template <bool debug, const HitType hitType>
__device__
RaycastResult rayCast(const Ray ray, const Node* bvh, const Triangle* triangles, const float maxT)
{
  float tMin = maxT;
  int minTriIdx = -1;
  glm::fvec2 minUV;
  RaycastResult result;
  const glm::fvec3 inverseDirection = glm::fvec3(1.f) / ray.direction;

  int ptr = 0;
  unsigned int stack[8] { 0 };
  int i = -1;
  float t = 0;
  glm::fvec2 uv;
  bool getNextNode = true;

  while (ptr >= 0)
  {
    unsigned int currentNodeIdx = stack[ptr];
    Node currentNode = bvh[currentNodeIdx];


    if (currentNode.rightIndex == -1)
    {
      getNextNode = false;

      if (debug)
      {
        const AABB b = currentNode.bbox;
        printf("\nHit bbox %d:\n", currentNodeIdx);
        printf("min: %f %f %f\n", b.min[0], b.min[1], b.min[2]);
        printf("max: %f %f %f\n", b.max[0], b.max[1], b.max[2]);
        printf("StartIdx: %d, endIdx: %d, nTris: %d\n\n", currentNode.startTri, currentNode.startTri + currentNode.nTri, currentNode.nTri);
      }

      if (i >= currentNode.startTri && i < currentNode.startTri + currentNode.nTri)
      {
        if (rayTriangleIntersection(ray, triangles[i], t, uv))
        {
          if (debug)
            printf("Hit triangle %d\n", i);

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

  if (debug)
    printf("///////////////////\n\n");

  return result;
}


template<unsigned int samples, typename curandState>
__device__ glm::fvec3 areaLightShading(const glm::fvec3 interpolatedNormal, const Light& light, const Node* bvh, const RaycastResult& result, const Triangle* triangles, curandState& curandState1, curandState& curandState2)
{
  glm::fvec3 brightness(0.f);

  //if (!light.isEnabled()) // Surprisingly slow
  //  return brightness;

  const glm::fvec3 shadowRayOrigin = result.point + interpolatedNormal * OFFSET_EPSILON;

  glm::fvec3 lightSamplePoint;
  float pdf;

  const glm::fvec3 emission = light.getEmission();

  // TODO: Unroll using templates
  for (unsigned int i = 0; i < samples; ++i)
  {
    light.sample(pdf, lightSamplePoint, curandState1, curandState2);

    const glm::fvec3 shadowRayDir = lightSamplePoint - shadowRayOrigin;

    const float maxT = glm::length(shadowRayDir); // Distance to the light
    const glm::fvec3 shadowRayDirNormalized = shadowRayDir / maxT;

    const Ray shadowRay(shadowRayOrigin, shadowRayDirNormalized);

    const RaycastResult shadowResult = rayCast<false, HitType::ANY>(shadowRay, bvh, triangles, maxT);

    if ((shadowResult && shadowResult.t >= maxT + OFFSET_EPSILON) || !shadowResult)
    {
      const float cosOmega = __saturatef(glm::dot(shadowRayDirNormalized, interpolatedNormal));
      const float cosL = __saturatef(glm::dot(-shadowRayDirNormalized, light.getNormal()));

      brightness += __fdividef(1.f, (maxT * maxT * pdf)) * emission * cosL * cosOmega;
    }
  }

  brightness /= samples;

  return brightness;
}

__device__ inline constexpr unsigned int cpow(const unsigned int base, const unsigned int exponent)
{
    return (exponent == 0) ? 1 : (base * cpow(base, exponent - 1));
}

struct RaycastTask
{
  Ray outRay;
  unsigned short levelsLeft;
  glm::fvec3 filter;
};

template <bool debug, typename curandStateType>
__device__ glm::fvec3 rayTrace(\
    const Node* bvh, \
    const Ray& ray, \
    const Triangle* triangles, \
    const Camera camera, \
    const Material* materials, \
    const unsigned int* triangleMaterialIds, \
    const Light light, \
    curandStateType& curandState1, \
    curandStateType& curandState2, \
    glm::fvec3* hitPoints = nullptr)
{
  constexpr unsigned int stackSize = cpow(2, SECONDARY_RAYS);
  RaycastTask stack[stackSize];
  glm::fvec3 color(0.f);
  int stackPtr = 0;
  int posPtr = 0; // Probably optimized away when not used

  // Primary ray
  stack[stackPtr].outRay = ray;
  stack[stackPtr].levelsLeft = SECONDARY_RAYS;
  stack[stackPtr].filter = glm::fvec3(1.f);
  ++stackPtr;

  while (stackPtr > 0)
  {
    --stackPtr;

    const RaycastTask currentTask = stack[stackPtr];
    const RaycastResult result = rayCast<false, HitType::CLOSEST>(currentTask.outRay, bvh, triangles, BIGT);

    if (!result)
      continue;

    if (debug)
    {
      hitPoints[posPtr++] = currentTask.outRay.origin;
      hitPoints[posPtr++] = result.point;
    }

    const Triangle triangle = triangles[result.triangleIdx];
    const Material material = materials[triangleMaterialIds[result.triangleIdx]];
    glm::fvec3 interpolatedNormal = triangle.normal(result.uv);

    unsigned int mask = INSIDE_BIT;

    if (glm::dot(interpolatedNormal, currentTask.outRay.direction) > 0.f)
      interpolatedNormal = -interpolatedNormal;  // We are inside an object. Flip the normal.
    else
      mask = 0x00000000; // We are outside. Unset bit.

    color += currentTask.filter * material.colorAmbient * 0.25f;

    const glm::fvec3 brightness = areaLightShading<SHADOWSAMPLING>(interpolatedNormal, light, bvh, result, triangles, curandState1, curandState2);
    color += currentTask.filter * material.colorDiffuse / glm::pi<float>() * brightness;

    if (material.shadingMode == material.GORAUD)
    {
      continue;
    }

    // Phong's specular highlight
    if ((mask & INSIDE_BIT) == 0x00 && material.shadingMode == material.PHONG)
    {
      const glm::fvec3 rm = reflectionDirection(interpolatedNormal, glm::normalize(light.getPosition() - result.point));
      color += material.colorSpecular * powf(__saturatef(glm::dot(rm, currentTask.outRay.direction)), material.shininess);
    }

    if (material.shadingMode == material.FRESNEL)
    {

      if (currentTask.levelsLeft == 0)
        continue;

      RaycastTask newTask; // Used twice for pushing

      mask = (material.colorSpecular.x != 0.f ||
          material.colorSpecular.y != 0.f ||
          material.colorSpecular.z != 0.f) ? REFLECTIVE_BIT | mask : mask;

      mask = (material.colorTransparent.x != 0.f ||
          material.colorTransparent.y != 0.f ||
          material.colorTransparent.z != 0.f) ? REFRACTIVE_BIT | mask : mask;

      float R = 1.f;

      if ((mask & REFRACTIVE_BIT) != 0x00) // Refractive
      {
        float idx1 = AIR_INDEX;
        float idx2 = material.refrIdx;

        float rat;

        if ((mask & INSIDE_BIT) != 0x00) // inside
          rat = __fdividef(idx1, idx2);
        else
          rat = __fdividef(idx2, idx1);

        // Something's not right here...

        // Transmittance and reflection according to fresnel
        const float cosi = fabsf(glm::dot(currentTask.outRay.direction, interpolatedNormal));

        if (sinf(acosf(cosi)) <= rat) // Check for total internal reflection
        {
          const float sin2t = fabs((idx1 / idx2) * (idx1 / idx2) * (1 - cosi * cosi));

          R = fresnelReflectioncoefficient(sin2t, cosi, idx1, idx2);

          const glm::fvec3 transOrig = result.point - interpolatedNormal * OFFSET_EPSILON;
          const glm::fvec3 transDir = refractionDirection(cosi, sin2t, interpolatedNormal, currentTask.outRay.direction, idx1, idx2);

          newTask.outRay = Ray(transOrig, transDir);
          newTask.levelsLeft = currentTask.levelsLeft - 1;
          newTask.filter = currentTask.filter * material.colorTransparent * (1 - R);
          stack[stackPtr] = newTask;
          ++stackPtr;
        }
      }

      if ((mask & REFLECTIVE_BIT) != 0x00) // Reflective
      {
        const glm::fvec3 reflOrig = result.point + interpolatedNormal * OFFSET_EPSILON;
        const glm::fvec3 reflDir = reflectionDirection(interpolatedNormal, currentTask.outRay.direction);

        newTask.outRay = Ray(reflOrig, reflDir);
        newTask.levelsLeft = currentTask.levelsLeft - 1;
        newTask.filter = currentTask.filter * material.colorSpecular * R;
        stack[stackPtr] = newTask;
        ++stackPtr;
      }
    }

  }

  return color;
}

template <const bool debug, typename curandStateType>
__device__ glm::fvec3 pathTrace(\
    const Node* bvh, \
    const Ray& ray, \
    const Triangle* triangles, \
    const Camera camera, \
    const Material* materials, \
    const unsigned int* triangleMaterialIds, \
    const Light light, \
    curandStateType& curandState1, \
    curandStateType& curandState2, \
    glm::fvec3* hitPoints = nullptr)
{
  unsigned int posPtr = 0;

  Ray currentRay = ray;
  glm::fvec3 color(0.f, 0.f, 0.f);
  glm::fvec3 throughput(1.f, 1.f, 1.f);

  float p = 1.0f;
  bool roulette = false;

  unsigned int bounces = PATH_TRACE_BOUNCES;
  bool terminate = false;
  unsigned int currentBounce = 0;

  do
  {
    const RaycastResult result = rayCast<debug, HitType::CLOSEST>(currentRay, bvh, triangles, BIGT);

    if (!result)
      return color;

    if (debug)
    {
      hitPoints[posPtr++] = currentRay.origin;
      hitPoints[posPtr++] = result.point;
    }

    const Triangle triangle = triangles[result.triangleIdx];
    const Material material = materials[triangleMaterialIds[result.triangleIdx]];
    glm::fvec3 interpolatedNormal = triangle.normal(result.uv);

    unsigned int mask = INSIDE_BIT;

    if (glm::dot(interpolatedNormal, currentRay.direction) > 0.f)
      interpolatedNormal = -interpolatedNormal;  // We are inside an object. Flip the normal.
    else
      mask = 0x00; // We are outside. Unset bit.

    color += throughput * material.colorAmbient * 0.25f;
    const glm::fvec3 brightness = areaLightShading<1>(interpolatedNormal, light, bvh, result, triangles, curandState1, curandState2);
    color += throughput * material.colorDiffuse / (glm::pi<float>() * p) * brightness;

    // Phong's specular highlight
    if ((mask & INSIDE_BIT) == 0x00 && material.shadingMode == material.PHONG)
    {
      const glm::fvec3 rm = reflectionDirection(interpolatedNormal, glm::normalize(light.getPosition() - result.point));
      color += material.colorSpecular * powf(__saturatef(glm::dot(rm, currentRay.direction)), material.shininess);
    }


    glm::fvec3 newDir, newOrig;
    glm::fmat3 B;

    if (material.shadingMode == material.FRESNEL)
    {
      mask = (material.colorSpecular.x != 0.f ||
          material.colorSpecular.y != 0.f ||
          material.colorSpecular.z != 0.f) ? REFLECTIVE_BIT | mask : mask;

      mask = (material.colorTransparent.x != 0.f ||
          material.colorTransparent.y != 0.f ||
          material.colorTransparent.z != 0.f) ? REFRACTIVE_BIT | mask : mask;

      float rP = 1.f; // Probability for reflection to occur. Depends on the strength of the specular and transparent colors.

      float R = 1.f; // Fresnel reflection coefficient
      float cosi, sin2t, idx1, idx2;

      if ((mask & REFRACTIVE_BIT) != 0x00)
      {
        float rLen = glm::length(material.colorSpecular);
        float tLen = glm::length(material.colorTransparent);

        rP = rLen / (rLen + tLen);

        idx1 = AIR_INDEX;
        idx2 = material.refrIdx;

        float rat;

        if ((mask & INSIDE_BIT) != 0x00) // inside
          rat = __fdividef(idx1, idx2);
        else
          rat = __fdividef(idx2, idx1);

        cosi = fabsf(glm::dot(currentRay.direction, interpolatedNormal));

        if (sinf(acosf(cosi)) <= rat) // Check for total internal reflection
        {
          sin2t = fabs((idx1 / idx2) * (idx1 / idx2) * (1 - cosi * cosi));
          R = fresnelReflectioncoefficient(sin2t, cosi, idx1, idx2);
        }
      }

      rP *= R;

      rP = rP / (rP + (1.f - rP) * (1.f - R));

      bool refl = curand_uniform(&curandState1) < rP;

      if (refl)
      {
        newDir = reflectionDirection(interpolatedNormal, currentRay.direction);
        newOrig = result.point + interpolatedNormal * OFFSET_EPSILON;
        throughput *= material.colorSpecular;
      }
      else
      {
        newDir = refractionDirection(cosi, sin2t, interpolatedNormal, currentRay.direction, idx1, idx2);
        newOrig = result.point - interpolatedNormal * OFFSET_EPSILON;
        throughput *= material.colorTransparent;
      }

    }
    else // Diffuse
    {
      B = getBasis(interpolatedNormal);

      do {
        newDir = glm::fvec3(curand_uniform(&curandState1) * 2.0f - 1.0f, curand_uniform(&curandState1) * 2.0f - 1.0f, 0.f);
      } while ((newDir.x * newDir.x + newDir.y * newDir.y) >= 1);

      newDir.z = glm::sqrt(1 - newDir.x * newDir.x - newDir.y * newDir.y);
      newDir = B * newDir;
      newDir = glm::normalize(newDir);

      newOrig = result.point + OFFSET_EPSILON * interpolatedNormal;

      p *= glm::dot(newDir, interpolatedNormal) * (1.f / glm::pi<float>());
      throughput *= material.colorDiffuse / glm::pi<float>() * glm::dot(newDir, interpolatedNormal);
    }

    currentRay = Ray(newOrig, newDir);

    if (currentBounce < bounces)
    {
      ++currentBounce;
    }
    else if (roulette)
    {
      ++currentBounce;
      p *= 0.8f; // Continuation probability
      terminate = curand_uniform(&curandState1) < 0.2f;
    }
    else
      terminate = true;

  } while (!terminate);


  return color;
}




#endif /* KERNELS_HPP_ */
