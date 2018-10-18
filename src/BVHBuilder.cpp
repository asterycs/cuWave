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
      float3 pmin = triangles_[ti].triangle.min();
      float3 pmax = triangles_[ti].triangle.max();

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
  const auto start = triangles_.begin() + node.startTri;
  const auto end = start + node.nTri;

  if (axis == 0)
  {
      __gnu_parallel::stable_sort(start, end, [](const auto& l, const auto& r)
          {
            return l.triangle.center().x < r.triangle.center().x;
          });
  }else if (axis == 1)
  {
      __gnu_parallel::stable_sort(start, end, [](const auto& l, const auto& r)
        {
          return l.triangle.center().y < r.triangle.center().y;
        });
  }else
  {
      __gnu_parallel::stable_sort(start, end, [](const auto& l, const auto& r)
        {
          return l.triangle.center().z < r.triangle.center().z;
        });
  }
}

SplitCandidate BVHBuilder::proposeSpatialSplit(const Node& node)
{
	AABB bestLeftBox, bestRightBox;

	float minCost = std::numeric_limits<float>::max();
	const unsigned int a = node.bbox.maxAxis();

	const float axisLength = getElement(node.bbox.max - node.bbox.min, a);
	const float stepLength = axisLength / node.nTri;

	for (int i = 1; i < node.nTri-1; ++i)
	{
		std::size_t leftIds(0);
		std::size_t rightIds(0);

		float3 leftMax = make_float3(node.bbox.max.x, node.bbox.max.y, node.bbox.max.z);
		float3 rightMin = make_float3(node.bbox.min.x, node.bbox.min.y, node.bbox.min.z);

		updateElement(leftMax, a, getElement(node.bbox.min, a) + stepLength*i);
		updateElement(rightMin, a, getElement(node.bbox.min, a) + stepLength*i);

		AABB left(node.bbox.min, leftMax);
		AABB right(rightMin, node.bbox.max);

		for (int ti = node.startTri; ti < node.startTri + node.nTri; ++ti)
		{
			if (triangles_[ti].triangle.touches(left))
				++leftIds;

			if (triangles_[ti].triangle.touches(right))
				++rightIds;
		}

		const float currentCost = left.area() * leftIds + right.area() * rightIds;

		if (currentCost < minCost)
		{
			minCost = currentCost;
			bestLeftBox = left;
			bestRightBox = right;
		}

	}

	std::vector<int> leftIds;
	std::vector<int> rightIds;


	for (int ti = node.startTri; ti < node.startTri + node.nTri; ++ti)
	{
		if (triangles_[ti].triangle.touches(bestLeftBox))
			leftIds.push_back(ti);

		if (triangles_[ti].triangle.touches(bestRightBox))
			rightIds.push_back(ti);
	}

	SplitCandidate splitCandidate;
	splitCandidate.type = SPATIAL;
	splitCandidate.cost = minCost;
	splitCandidate.splitAxis = a;

	splitCandidate.leftChild.startTri = node.startTri;
	splitCandidate.leftChild.nTri = leftIds.size();
	splitCandidate.leftChild.bbox = bestLeftBox;

	splitCandidate.rightChild.startTri = node.startTri + leftIds.size();
	splitCandidate.rightChild.nTri = rightIds.size();
	splitCandidate.rightChild.bbox = bestRightBox;

	return splitCandidate;
}

SplitCandidate BVHBuilder::proposeSAHSplit(const Node& node)
{
	float minCost = std::numeric_limits<float>::max();
	int minStep = -1;
	const unsigned int a = node.bbox.maxAxis();

	sortTrisOnAxis(node, a);

	const int fStart = node.startTri;
	const int fEnd = node.startTri + node.nTri - 1;

	AABB fBox = triangles_[fStart].triangle.bbox();
	std::vector<AABB> fBoxes(node.nTri - 1);

	for (int i = fStart; i < fEnd; ++i)
	{
		fBox.add(triangles_[i].triangle);
		fBoxes[i - node.startTri] = fBox;
	}

	AABB rBox = triangles_[fEnd].triangle.bbox();
	std::vector<AABB> rBoxes(node.nTri - 1);

	for (int i = fEnd - 1; i > fStart - 1; --i)
	{
		rBox.add(triangles_[i].triangle);
		rBoxes[i - node.startTri] = rBox;
	}

	for (int s = 1; s < node.nTri - 1; ++s)
	{
		const float currentCost = fBoxes[s - 1].area() * s + rBoxes[s - 1].area() * (node.nTri - s);

		if (currentCost < minCost)
		{
		  minCost = currentCost;
		  minStep = s;
		}
	}

	SplitCandidate splitCandidate;
	splitCandidate.type = SAH;
	splitCandidate.cost = minCost;
	splitCandidate.splitAxis = a;

	splitCandidate.leftChild.startTri = node.startTri;
	splitCandidate.leftChild.nTri = minStep;
	splitCandidate.leftChild.bbox = fBoxes[minStep - 1];

	splitCandidate.rightChild.startTri = node.startTri + minStep;
	splitCandidate.rightChild.nTri = node.nTri - minStep;
	splitCandidate.rightChild.bbox = rBoxes[minStep - 1];

	return splitCandidate;
}

bool BVHBuilder::splitNode(const Node& node, Node& leftChild, Node& rightChild)
{
  if (node.nTri <= static_cast<int>(MAX_TRIS_PER_LEAF))
    return false;

  const SplitCandidate sahCandidate = proposeSAHSplit(node);
  const SplitCandidate spatialCandidate = proposeSpatialSplit(node);


  const float sa = node.bbox.area();
  const float parentCost = node.nTri * sa;

  if (sahCandidate.cost < spatialCandidate.cost && sahCandidate.cost < parentCost-1e-5f)
  {
      performSplit(sahCandidate, node, leftChild, rightChild);
      return true;
  }else if (spatialCandidate.cost < sahCandidate.cost && spatialCandidate.cost < parentCost-1e-5f)
  {
      performSplit(spatialCandidate, node, leftChild, rightChild);

      return true;
  }else
      return false;
}

void BVHBuilder::performSplit(const SplitCandidate& split, const Node& node, Node& leftChild, Node& rightChild)
{
    const unsigned int a = split.splitAxis;
    sortTrisOnAxis(node, a);

    if (split.type == SAH)
    {
      leftChild = split.leftChild;
      rightChild = split.rightChild;
    }else if (split.type == SPATIAL)
    {
        std::vector<TriangleHolder> leftTris;
        std::vector<TriangleHolder> rightTris;

        for (int ti = node.startTri; ti < node.startTri + node.nTri; ++ti)
        {
            if (triangles_[ti].triangle.touches(split.leftChild.bbox))
            	leftTris.push_back(triangles_[ti]);

            if (triangles_[ti].triangle.touches(split.rightChild.bbox))
            	rightTris.push_back(triangles_[ti]);
        }

        leftChild = split.leftChild;
        rightChild = split.rightChild;

        triangles_.erase(triangles_.begin() + node.startTri, triangles_.begin() + node.startTri + node.nTri);
        triangles_.insert(triangles_.begin() + node.startTri, leftTris.begin(), leftTris.end());
        triangles_.insert(triangles_.begin() + node.startTri + leftTris.size(), rightTris.begin(), rightTris.end());
    }else
        return;
}

void BVHBuilder::build(const std::vector<Triangle>& triangles, const std::vector<uint32_t>& triangleMaterialIds, const std::vector<uint32_t>& lightTriangles)
{
  this->lightTriangleIds_ = lightTriangles;
  
  unsigned int idx = 0;

  for (auto t : triangles)
  {
	  this->triangles_.push_back(TriangleHolder(t, triangleMaterialIds[idx], idx));
    ++idx;
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

  //int leafCount = 0;
  //int nodeCount = 0;

  stack.push(root);
  parentIndices.push(-1);

  	  	  finishedNodes.push_back(root);
  /*while (!stack.empty()) {

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

  }*/

  this->bvh_ = finishedNodes;
}

std::vector<Node> BVHBuilder::getBVH() const
{
  return this->bvh_;
}

std::vector<Triangle> BVHBuilder::getTriangles() const
{
  std::vector<Triangle> triangles(this->triangles_.size());
  
  for (std::size_t i = 0; i < triangles.size(); ++i)
    triangles[i] = this->triangles_[i].triangle;
    
  return triangles;
}

std::vector<uint32_t> BVHBuilder::getTriangleMaterialIds() const
{
  std::vector<uint32_t> triangleMaterialIds(triangles_.size());

  for (std::size_t i = 0; i < triangles_.size(); ++i)
	  triangleMaterialIds[i] = triangles_[i].materialIdx;

  return triangleMaterialIds;
}

std::vector<uint32_t> BVHBuilder::getLightTriangleIds() const
{
  std::vector<uint32_t> newLightTriangleIds(lightTriangleIds_.size());

  for (std::size_t ti = 0; ti < lightTriangleIds_.size(); ++ti)
	  newLightTriangleIds[ti] = std::find_if(triangles_.begin(), triangles_.end(), [&](const auto& x){ return lightTriangleIds_[ti] == x.triangleIdx; }) - triangles_.begin();

  return newLightTriangleIds;
}
