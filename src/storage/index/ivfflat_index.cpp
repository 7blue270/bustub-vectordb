#include "storage/index/ivfflat_index.h"
#include <algorithm>
#include <optional>
#include <random>
#include "common/exception.h"
#include "execution/expressions/vector_expression.h"
#include "storage/index/index.h"
#include "storage/index/vector_index.h"

namespace bustub {
using Vector = std::vector<double>;

IVFFlatIndex::IVFFlatIndex(std::unique_ptr<IndexMetadata> &&metadata, BufferPoolManager *buffer_pool_manager,
                           VectorExpressionType distance_fn, const std::vector<std::pair<std::string, int>> &options)
    : VectorIndex(std::move(metadata), distance_fn) {
  std::optional<size_t> lists;
  std::optional<size_t> probe_lists;
  for (const auto &[k, v] : options) {
    if (k == "lists") {
      lists = v;
    } else if (k == "probe_lists") {
      probe_lists = v;
    }
  }
  if (!lists.has_value() || !probe_lists.has_value()) {
    throw Exception("missing options: lists / probe_lists for ivfflat index");
  }
  lists_ = *lists;
  probe_lists_ = *probe_lists;
}

void VectorAdd(Vector &a, const Vector &b) {
  for (size_t i = 0; i < a.size(); i++) {
    a[i] += b[i];
  }
}

void VectorScalarDiv(Vector &a, double x) {
  for (auto &y : a) {
    y /= x;
  }
}

// Find the nearest centroid to the base vector in all centroids
//在所有质心中找到与基向量最近的质心    
auto FindCentroid(const Vector &vec, const std::vector<Vector> &centroids, VectorExpressionType dist_fn) -> size_t {
  size_t nearest_centroid = 0;
  double nearest_distance = ComputeDistance(vec, centroids[0], dist_fn);
  for (size_t i = 1; i < centroids.size(); i++) {
    double current_distance = ComputeDistance(vec, centroids[i], dist_fn);
    if (current_distance < nearest_distance) {
      nearest_centroid = i;
      nearest_distance = current_distance;
    }
  }
  return nearest_centroid;
}

// Compute new centroids based on the original centroids.
auto FindCentroids(const std::vector<std::pair<Vector, RID>> &data, const std::vector<Vector> &centroids,
                   VectorExpressionType dist_fn, std::vector<std::vector<std::pair<Vector, RID>>> &buckets) -> std::vector<Vector> {
  std::vector<Vector> new_centroids(centroids.size(), Vector(centroids[0].size(), 0));
  buckets = std::vector<std::vector<std::pair<Vector, RID>>>(centroids.size());
  for (const auto &[vec, rid] : data) {
    size_t centroid_id = FindCentroid(vec, centroids, dist_fn);
    VectorAdd(new_centroids[centroid_id], vec);
    buckets[centroid_id].push_back({vec, rid});
  }
  for (size_t i = 0; i < new_centroids.size(); i++) {
    if (!buckets[i].empty()) {
      VectorScalarDiv(new_centroids[i], (double)buckets[i].size());
    }
  }
  return new_centroids;
}

void IVFFlatIndex::BuildIndex(std::vector<std::pair<Vector, RID>> initial_data) {
  if (initial_data.empty()) {
    return;
  }
  //分配初始的质心
  for (size_t i = 0; i < lists_; i++) {
    centroids_.push_back(initial_data[i].first);
  }
  //kmeans迭代50次
  for (size_t i = 0; i < 50; i++) {
    centroids_ = FindCentroids(initial_data, centroids_, distance_fn_, centroids_buckets_);
    // printCentroidsBucketsInfo();
  }
}

void IVFFlatIndex::InsertVectorEntry(const std::vector<double> &key, RID rid) {
  size_t id = FindCentroid(key, centroids_, distance_fn_);
  centroids_buckets_[id].emplace_back(key, rid);
}

// 1. 定义一个专门只比较距离的比较器（忽略 RID）
struct DistanceComparator {
    bool operator()(const std::pair<double, RID>& a, const std::pair<double, RID>& b) const {
        return a.first < b.first; // 大顶堆逻辑：距离大的在堆顶
    }
};

auto IVFFlatIndex::ScanVectorKey(const std::vector<double> &base_vector, size_t limit) -> std::vector<RID> {
  // 需要探测 probe_lists 个质心桶的数量，从每个桶中检索 limit 个最近邻（局部结果），并对局部结果进行 top-n
  // 排序，以从局部结果中获得 limit 个最近邻（全局结果）。
  std::priority_queue<std::pair<double, size_t>, std::vector<std::pair<double, size_t>>, std::greater<>> pq;
  std::vector<size_t> bucket_probe_lists;
  for (size_t i = 0; i < centroids_.size(); i++) {
    if (!centroids_buckets_[i].empty()) {
      pq.push({ComputeDistance(base_vector, centroids_[i], distance_fn_), i});
    }
  }
  while (bucket_probe_lists.size() < probe_lists_ && !pq.empty()) {
    bucket_probe_lists.push_back(pq.top().second);
    pq.pop();
  }
  std::vector<RID> result;
  //std::priority_queue<std::pair<double, RID>, std::vector<std::pair<double, RID>>, std::less<>> global_pq;
  // 2. 在这里使用我们自定义的比较器
  std::priority_queue<std::pair<double, RID>, std::vector<std::pair<double, RID>>, DistanceComparator> global_pq;
  for (const auto &bucket_id : bucket_probe_lists) {
    std::priority_queue<std::pair<double, RID>, std::vector<std::pair<double, RID>>, DistanceComparator> local_pq;
    for (const auto &[vec, rid] : centroids_buckets_[bucket_id]) {
      local_pq.push({ComputeDistance(base_vector, vec, distance_fn_), rid});
      if (local_pq.size() > limit) {
        local_pq.pop();
      }
    }
    while (!local_pq.empty()) {
      global_pq.push(local_pq.top());
      if (global_pq.size() > limit) {
        global_pq.pop();
      }
      local_pq.pop();
    }
  }
  while (!global_pq.empty() && result.size() < limit) {
    result.push_back(global_pq.top().second);
    global_pq.pop();
  }
  std::reverse(result.begin(), result.end());
  return result;
}

}  // namespace bustub
