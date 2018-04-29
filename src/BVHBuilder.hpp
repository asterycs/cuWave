#ifndef BVHBUILDER_HPP
#define BVHBUILDER_HPP

#include <vector>
#include <string>

#include "Utils.hpp"
#include "Triangle.hpp"

#define MAX_TRIS_PER_LEAF 8

class BVHBuilder
{
public:
  BVHBuilder();
  ~BVHBuilder();
  
  std::vector<Node> getBVH() const;
  std::vector<Triangle> getTriangles() const;
  std::vector<unsigned int> getTriangleMaterialIds() const;
  std::vector<unsigned int> getLightTriangleIds() const;
  
  void build(const std::vector<Triangle>& triangles, const std::vector<unsigned int>& triangleMaterialIds, const std::vector<unsigned int>& lightTriangles);
  AABB computeBB(const Node node);
  void sortTrisOnAxis(const Node& node, const unsigned int axis);
  bool splitNode(const Node& node, Node& leftChild, Node& rightChild);
  void reorderTrianglesAndMaterialIds();
  
private:
  std::vector<Node> bvh;
  std::vector<std::pair<Triangle, unsigned int>> trisWithIds;
  
  std::vector<unsigned int> lightTriangles;
  std::vector<unsigned int> triangleMaterialIds;
  std::vector<Material> bvhBoxMaterials;
};

#endif // BVHBUILDER_HPP
