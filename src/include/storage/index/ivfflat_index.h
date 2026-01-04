
#pragma once

#include <memory>
#include <vector>
#include "buffer/buffer_pool_manager.h"
#include "execution/expressions/vector_expression.h"
#include "storage/index/index.h"
#include "storage/index/vector_index.h"

namespace bustub {

class IVFFlatIndex : public VectorIndex {
 public:
  IVFFlatIndex(std::unique_ptr<IndexMetadata> &&metadata, BufferPoolManager *buffer_pool_manager,
               VectorExpressionType distance_fn, const std::vector<std::pair<std::string, int>> &options);

  ~IVFFlatIndex() override = default;

  void BuildIndex(std::vector<std::pair<std::vector<double>, RID>> initial_data) override;
  auto ScanVectorKey(const std::vector<double> &base_vector, size_t limit) -> std::vector<RID> override;
  void InsertVectorEntry(const std::vector<double> &key, RID rid) override;

  BufferPoolManager *bpm_;
  // number of buckets or lists to create when building the index
  //构建索引时要创建的存储桶或列表的数量
  size_t lists_{0}; 
  // number of buckets or lists to probe when lookup
  //查找时要探测的存储桶或列表的数量
  size_t probe_lists_{0};

  using Vector = std::vector<double>;

  // vector of each centroid
  //每个质心的向量  
  std::vector<Vector> centroids_;

  // vectors and RIDs in each of the centroid list
  //每个质心列表中的向量和RID
  std::vector<std::vector<std::pair<Vector, RID>>> centroids_buckets_;
};

}  // namespace bustub
