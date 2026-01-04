#pragma once

#include <random>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include "buffer/buffer_pool_manager.h"
#include "common/macros.h"
#include "execution/expressions/vector_expression.h"
#include "storage/index/index.h"
#include "storage/index/vector_index.h"

namespace bustub {

struct NSW {
  using Vector = std::vector<double>;
  // reference to HNSW's vertices vector
  const std::vector<Vector> &vertices_;
  // distance function
  VectorExpressionType dist_fn_;
  //每一个顶点在该层的最大边数
  size_t m_max_{};
  // edges of each vertex in this layer, key is the vertex id of HNSW
  std::unordered_map<size_t, std::unordered_set<size_t>> edges_{};
  //这一层的顶点    
  std::vector<size_t> in_vertices_{};


  //搜索层：从指定的入口开始搜索，返回距离base_vector最近的limit个顶点id
  auto SearchLayer(const std::vector<double> &base_vector, size_t limit, const std::vector<size_t> &entry_points)
      -> std::vector<size_t>;
  //在该层插入一个顶点，仅用于实现仅NSW索引
  auto Insert(const std::vector<double> &vec, size_t vertex_id, size_t ef_construction, size_t m);
  //添加一个顶点到该层
  auto AddVertex(size_t vertex_id);
  // connect two vertices
  void Connect(size_t vertex_a, size_t vertex_b);
  //该层的默认入口点是插入的第一个元素
  auto DefaultEntryPoint() -> size_t { return in_vertices_[0]; }
  // 裁剪边
  auto PurgeEdges(size_t m) -> void;
};

// select m nearest elements from the base vector in vertex_ids
auto SelectNeighbors(const std::vector<double> &vec, const std::vector<size_t> &vertex_ids,
                     const std::vector<std::vector<double>> &vertices, size_t m, VectorExpressionType dist_fn)
    -> std::vector<size_t>;

class HNSWIndex : public VectorIndex {
 public:
  HNSWIndex(std::unique_ptr<IndexMetadata> &&metadata, BufferPoolManager *buffer_pool_manager,
            VectorExpressionType distance_fn, const std::vector<std::pair<std::string, int>> &options);

  ~HNSWIndex() override = default;
  //创建索引    
  void BuildIndex(std::vector<std::pair<std::vector<double>, RID>> initial_data) override;
  //扫描向量键，返回距离base_vector最近的limit个RID
  auto ScanVectorKey(const std::vector<double> &base_vector, size_t limit) -> std::vector<RID> override;
  //插入一个向量键-RID对到索引中
  void InsertVectorEntry(const std::vector<double> &key, RID rid) override;
  //添加一个顶点到图中，返回该顶点id
  auto AddVertex(const std::vector<double> &vec, RID rid) -> size_t;

  using Vector = std::vector<double>;
  std::unique_ptr<std::vector<Vector>> vertices_;
  std::vector<RID> rids_;
  std::vector<NSW> layers_;   //图的各个层

  //每次插入顶点时要创建的边数
  size_t m_;
  // 插入时要搜索的邻居数
  size_t ef_construction_;
  // 查找时要搜索的邻居数
  size_t ef_search_;
  // 除了层0之外的所有层的最大边数
  size_t m_max_;
  // 层0的最大边数
  size_t m_max_0_;
  // 随机数生成器
  std::mt19937 generator_;
  // 级别归一化因子：用于计算新顶点该插入的层数
  double m_l_;
  //距离函数
  VectorExpressionType dist_fn_;
};

}  // namespace bustub
