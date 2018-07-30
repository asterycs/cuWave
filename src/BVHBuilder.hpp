#ifndef BVHBUILDER_HPP
#define BVHBUILDER_HPP

#include <vector>
#include <string>

#include "Utils.hpp"
#include "Triangle.hpp"

#define MAX_TRIS_PER_LEAF 2

enum SplitType
{
    SAH,
    SPATIAL
};

struct SplitCandidate
{
    enum SplitType type;

    float cost;
    unsigned int splitAxis;

    Node leftChild;
    Node rightChild;

    SplitCandidate() : type(SAH), cost(0.f), splitAxis(0), leftChild(), rightChild() {};
};

class BVHBuilder
{
public:

  struct TriangleHolder {
	  Triangle triangle;
	  uint32_t materialIdx;
	  uint32_t triangleIdx;

	  TriangleHolder(const Triangle& t, const uint32_t mIdx, const uint32_t tIdx) : triangle(t), materialIdx(mIdx), triangleIdx(tIdx) {};
  };

  BVHBuilder();
  ~BVHBuilder();
  
  std::vector<Node> getBVH() const;
  std::vector<Triangle> getTriangles() const;
  std::vector<uint32_t> getTriangleMaterialIds() const;
  std::vector<uint32_t> getLightTriangleIds() const;
  
  void build(const std::vector<Triangle>& triangles, const std::vector<uint32_t>& triangleMaterialIds, const std::vector<uint32_t>& lightTriangles);
  AABB computeBB(const Node node);
  void sortTrisOnAxis(const Node& node, const unsigned int axis);
  bool splitNode(const Node& node, Node& leftChild, Node& rightChild);
  
private:
  SplitCandidate proposeSAHSplit(const Node& node);
  SplitCandidate proposeSpatialSplit(const Node& node);

  void performSplit(const SplitCandidate& split, const Node& node, Node& leftChild, Node& rightChild);

  std::vector<Node> bvh_;
  std::vector<TriangleHolder> triangles_;
  
  std::vector<uint32_t> lightTriangleIds_;
  std::vector<Material> bvhBoxMaterials_;
};

#endif // BVHBUILDER_HPP
