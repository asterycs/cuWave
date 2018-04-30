#include "BVHBuilder.hpp"

#include <stack>
#include <parallel/algorithm>

BVHBuilder::BVHBuilder()
{
  
}

BVHBuilder::~BVHBuilder()
{
  
}

AABB BVHBuilder::computeBB(const Node node)
{
  // Construct BB
  float3 minXYZ = make_float3(std::numeric_limits<float>::max());
  float3 maxXYZ = make_float3(-std::numeric_limits<float>::max());

  if (node.nTri > 0)
  {
    for (int ti = node.startTri; ti < node.startTri + node.nTri; ++ti)
    {
      float3 pmin = trisWithIds[ti].first.min();
      float3 pmax = trisWithIds[ti].first.max();

      minXYZ = fminf(pmin, minXYZ);
      maxXYZ = fmaxf(pmax, maxXYZ);
    }
  }
  else
  {
    minXYZ = make_float3(0.f);
    maxXYZ = make_float3(0.f);
  }

  AABB ret(minXYZ, maxXYZ);

  return ret;
}

void BVHBuilder::sortTrisOnAxis(const Node& node, const unsigned int axis)
{
  const auto start = trisWithIds.begin() + node.startTri;
  const auto end = start + node.nTri;

  if (axis == 0)
  {
  __gnu_parallel::sort(start, end, [](const std::pair<Triangle, uint32_t>& l, const std::pair<Triangle, uint32_t>& r)
      {
        return l.first.center().x < r.first.center().x;
      });
  }else if (axis == 1)
  {
    __gnu_parallel::sort(start, end, [](const std::pair<Triangle, uint32_t>& l, const std::pair<Triangle, uint32_t>& r)
        {
          return l.first.center().y < r.first.center().y;
        });
  }else
  {
    __gnu_parallel::sort(start, end, [](const std::pair<Triangle, uint32_t>& l, const std::pair<Triangle, uint32_t>& r)
        {
          return l.first.center().z < r.first.center().z;
        });
  }

}

bool BVHBuilder::splitNode(const Node& node, Node& leftChild, Node& rightChild)
{
  if (node.nTri <= static_cast<int>(MAX_TRIS_PER_LEAF))
    return false;

  const float sa = node.bbox.area();
  const float parentCost = node.nTri * sa;

  float minCost = std::numeric_limits<float>::max();
  int minStep = -1;
  const unsigned int a = node.bbox.maxAxis();

  sortTrisOnAxis(node, a);

  const int fStart = node.startTri;
  const int fEnd = node.startTri + node.nTri - 1;

  AABB fBox = trisWithIds[fStart].first.bbox();
  std::vector<AABB> fBoxes(node.nTri - 1);

  for (int i = fStart; i < fEnd; ++i)
  {
    fBox.add(trisWithIds[i].first);
    fBoxes[i - node.startTri] = fBox;
  }

  AABB rBox = trisWithIds[fEnd].first.bbox();
  std::vector<AABB> rBoxes(node.nTri - 1);

  for (int i = fEnd - 1; i > fStart - 1; --i)
  {
    rBox.add(trisWithIds[i].first);
    rBoxes[i - node.startTri] = rBox;
  }

#pragma omp parallel for
  for (int s = 1; s < node.nTri - 1; ++s)
  {
    const float currentCost = fBoxes[s - 1].area() * s + rBoxes[s - 1].area() * (node.nTri - s);

#pragma omp critical
    if (currentCost < minCost)
    {
      minCost = currentCost;
      minStep = s;
    }
  }

  if (minCost < parentCost)
  {
          leftChild.startTri = node.startTri;
          leftChild.nTri = minStep;
          leftChild.bbox = fBoxes[minStep - 1];

          rightChild.startTri = node.startTri + minStep;
          rightChild.nTri = node.nTri - minStep;
          rightChild.bbox = rBoxes[minStep - 1];

          return true;
  }else
  {
    return false;
  }
}

void BVHBuilder::reorderTrianglesAndMaterialIds()
{
  std::vector<uint32_t> triIdxMap;
  triIdxMap.resize(trisWithIds.size());

  for (std::size_t i = 0; i < trisWithIds.size(); ++i)
    triIdxMap[trisWithIds[i].second] = i;

  std::vector<uint32_t> orderedTriangleMaterialIds(triangleMaterialIds.size());

  for (std::size_t ti = 0; ti < trisWithIds.size(); ++ti)
    orderedTriangleMaterialIds[ti] = triangleMaterialIds[trisWithIds[ti].second];

  triangleMaterialIds = orderedTriangleMaterialIds;

  std::vector<uint32_t> orderedLightTriangles(lightTriangles.size());

  for (std::size_t ti = 0; ti < lightTriangles.size(); ++ti)
    orderedLightTriangles[ti] = std::find_if(trisWithIds.begin(), trisWithIds.end(), [&](const auto& x){ return lightTriangles[ti] == x.second; }) - trisWithIds.begin();

  lightTriangles = orderedLightTriangles;

  return;
}

void BVHBuilder::build(const std::vector<Triangle>& triangles, const std::vector<uint32_t>& triangleMaterialIds, const std::vector<uint32_t>& lightTriangles)
{
  this->triangleMaterialIds = triangleMaterialIds;
  this->lightTriangles = lightTriangles;
  
  unsigned int idx = 0;

  for (auto t : triangles)
  {
    trisWithIds.push_back(std::make_pair(t, idx++));
  }
  
  Node root;
  root.startTri = 0;
  root.nTri = triangles.size();
  root.bbox = computeBB(root);
  root.rightIndex = -1;
  
  
  // This is a simple top down approach that places the nodes in an array.
  // This makes the transfer to GPU simple.
  std::stack<Node> stack;
  std::stack<int> parentIndices;

  std::vector<Node> finishedNodes;
  std::vector<int> touchCount;

  const unsigned int nodecountAppr = 0;
  finishedNodes.reserve(nodecountAppr);
  touchCount.reserve(nodecountAppr);

  int leafCount = 0;
  int nodeCount = 0;

  stack.push(root);
  parentIndices.push(-1);

  while (!stack.empty()) {

    Node node = stack.top();
    stack.pop();
    int parentIndex = parentIndices.top();
    parentIndices.pop();

    Node left, right;
    const bool wasSplit = splitNode(node, left, right);

    if (wasSplit)
    {
      stack.push(right);
      stack.push(left);

      parentIndices.push(nodeCount);
      parentIndices.push(nodeCount);
    }
    else
    {
      ++leafCount;
      node.rightIndex = -1;
    }

    finishedNodes.push_back(node);

    touchCount.push_back(0);
    touchCount[nodeCount] = 0;
    ++nodeCount;

    if (parentIndex != -1)
    {
      ++touchCount[parentIndex];

      if (touchCount[parentIndex] == 2)
      {
        finishedNodes[parentIndex].rightIndex = nodeCount - 1;
      }
    }

  }

  this->bvh = finishedNodes;
  
  reorderTrianglesAndMaterialIds();
}

std::vector<Node> BVHBuilder::getBVH() const
{
  return this->bvh;
}

std::vector<Triangle> BVHBuilder::getTriangles() const
{
  std::vector<Triangle> triangles(trisWithIds.size());
  
  for (std::size_t i = 0; i < trisWithIds.size(); ++i)
    triangles[i] = trisWithIds[i].first;
    
  return triangles;
}

std::vector<uint32_t> BVHBuilder::getTriangleMaterialIds() const
{
  return triangleMaterialIds;
}

std::vector<uint32_t> BVHBuilder::getLightTriangleIds() const
{
  return lightTriangles;
}
