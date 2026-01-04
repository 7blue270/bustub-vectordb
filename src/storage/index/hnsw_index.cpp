#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <iterator>
#include <memory>
#include <queue>
#include <random>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "common/macros.h"
#include "execution/expressions/vector_expression.h"
#include "fmt/format.h"
#include "fmt/std.h"
#include "storage/index/hnsw_index.h"
#include "storage/index/index.h"
#include "storage/index/vector_index.h"

//nsw:停止探索的条件为搜索队列中的最小距离节点距离大于当前结果集中最大距离节点距离
//nsw：且探索的时候需要多个入口

//hnsw：如果不是最低层，则搜索最近邻，如果是最底层，则搜索最近k个邻居
namespace bustub {
HNSWIndex::HNSWIndex(std::unique_ptr<IndexMetadata> &&metadata, BufferPoolManager *buffer_pool_manager,
                     VectorExpressionType distance_fn, const std::vector<std::pair<std::string, int>> &options)
    : VectorIndex(std::move(metadata), distance_fn),
      vertices_(std::make_unique<std::vector<Vector>>()),
      layers_{{*vertices_, distance_fn}} {
  std::optional<size_t> m;
  std::optional<size_t> ef_construction;
  std::optional<size_t> ef_search;
  for (const auto &[k, v] : options) {
    if (k == "m") {
      m = v;
    } else if (k == "ef_construction") {
      ef_construction = v;
    } else if (k == "ef_search") {
      ef_search = v;
    }
  }
  if (!m.has_value() || !ef_construction.has_value() || !ef_search.has_value()) {
    throw Exception("missing options: m / ef_construction / ef_search for hnsw index");
  }
  ef_construction_ = *ef_construction;
  m_ = *m;
  ef_search_ = *ef_search;
  m_max_ = m_;
  m_max_0_ = m_ * m_;
  layers_[0].m_max_ = m_max_0_;
  m_l_ = 1.0 / std::log(m_);
  std::random_device rand_dev;
  generator_ = std::mt19937(rand_dev());
  //距离函数
  dist_fn_ = distance_fn;   
}

//在该层中从基向量vertex_ids中选择离vec最近的m个邻居：抽象化的工具函数
auto SelectNeighbors(const std::vector<double> &vec, const std::vector<size_t> &vertex_ids,
                     const std::vector<std::vector<double>> &vertices, size_t m, VectorExpressionType dist_fn)
    -> std::vector<size_t> {
  //创建一个大根堆，容量为m
  std::priority_queue<std::pair<double, size_t>, std::vector<std::pair<double, size_t>>, std::less<>> W;
  for (const auto &id : vertex_ids) {
    auto dist = ComputeDistance(vec, vertices[id], dist_fn);
    W.emplace(dist, id);
    if (W.size() > m) {
      W.pop();
    }
  }  
  //将堆中的元素弹出，得到结果集合
  std::vector<size_t> result;
  result.reserve(W.size());
  while (!W.empty()) {
    result.push_back(W.top().second);
    W.pop();
  }
  std::reverse(result.begin(), result.end());
  return result;
}

//搜索层：从指定的入口开始搜索，返回距离base_vector最近的limit个顶点id
auto NSW::SearchLayer(const std::vector<double> &base_vector, size_t limit, const std::vector<size_t> &entry_points)
    -> std::vector<size_t> {
  //c,探索候选优先队列，按距离从小到大排序，因为每次都要取距离最小的点进行探索，所以用小根堆
  //w,结果集，要保存limit个最近邻，按距离从大到小排序，所以用大根堆
  std::priority_queue<std::pair<double, size_t>, std::vector<std::pair<double, size_t>>, std::greater<>> C;
  std::priority_queue<std::pair<double, size_t>, std::vector<std::pair<double, size_t>>, std::less<>> W;
  std::unordered_set<size_t> visited;

  //吧初始入口点加入探索队列和结果集
  for(const auto &entry : entry_points) {
    auto dist = ComputeDistance(base_vector, vertices_[entry], dist_fn_);
    C.emplace(dist, entry);
    W.emplace(dist, entry);
    visited.insert(entry);
  }
  //根据nsw的伪代码逻辑进行搜索
  while(!C.empty()) {
    auto [dist, vertex_id] = C.top();
    C.pop();
    if (dist > W.top().first) {
      break;
    }
    for (const auto &neighbor : edges_[vertex_id]) {
      //如果邻居没有被访问过，则计算距离并加入探索队列和结果集
      if (visited.find(neighbor) == visited.end()) {
        visited.insert(neighbor);
        auto neighbor_dist = ComputeDistance(base_vector, vertices_[neighbor], dist_fn_);
        C.emplace(neighbor_dist, neighbor);
        W.emplace(neighbor_dist, neighbor);
        if (W.size() > limit) {
          W.pop();
        }
      }
    }
  }
  //通过w堆得到结果集合result
  std::vector<size_t> result;
  result.reserve(W.size());
  while (!W.empty()) {
    result.push_back(W.top().second);
    W.pop();
  }
  std::reverse(result.begin(), result.end());
  return result;
}

//在该层插入一个顶点
auto NSW::AddVertex(size_t vertex_id) { in_vertices_.push_back(vertex_id); }

//在该层插入一个顶点，仅用于实现仅NSW索引
auto NSW::Insert(const std::vector<double> &vec, size_t vertex_id, size_t ef_construction, size_t m) {
  //先将顶点加入该层
  AddVertex(vertex_id);
  //在in_vertices_中搜索最近的m个邻居建立连接边
  std::vector<size_t> entry_points = SearchLayer(vec, ef_construction, {DefaultEntryPoint()});
  for(const auto &neighbor : entry_points) {
    Connect(vertex_id, neighbor);
  }   
  //修剪边，使每个顶点的边数不超过m 
  PurgeEdges(m);
}

void NSW::Connect(size_t vertex_a, size_t vertex_b) {
  edges_[vertex_a].insert(vertex_b);
  edges_[vertex_b].insert(vertex_a);
}

// 裁剪边:将裁剪边的操作抽象化为一个函数
void NSW::PurgeEdges(size_t m) {
  std::vector<size_t> neighbors;
  for (auto id : in_vertices_) {
    if (edges_[id].size() > m_max_) {
      neighbors = SelectNeighbors(vertices_[id], in_vertices_, vertices_, m, dist_fn_);
      // 删除与id相关所有边
      edges_[id].clear();
      for (const auto &edge : edges_) {
        edges_[edge.first].erase(id);
      }
      // 重新建立边
      for (const auto &neighbor : neighbors) {
        Connect(id, neighbor);
      }
    }
  }
}

auto HNSWIndex::AddVertex(const std::vector<double> &vec, RID rid) -> size_t {
  auto id = vertices_->size();
  vertices_->emplace_back(vec);
  rids_.emplace_back(rid);
  return id;
}

//创建hnsw索引：通过随机打乱初始数据，然后依次插入每个向量-RID对
void HNSWIndex::BuildIndex(std::vector<std::pair<std::vector<double>, RID>> initial_data) {
  std::shuffle(initial_data.begin(), initial_data.end(), generator_);  //随机打乱
  for (const auto &[vec, rid] : initial_data) {
    InsertVectorEntry(vec, rid);
  }
}

//扫描向量键，从最上面的层开始搜索，逐层向下，最终在层0中找到limit个最近邻
auto HNSWIndex::ScanVectorKey(const std::vector<double> &base_vector, size_t limit) -> std::vector<RID> {
  int32_t level_max = (int32_t)layers_.size() - 1;
  auto entry_points = layers_[level_max].SearchLayer(base_vector, ef_search_, {layers_[level_max].DefaultEntryPoint()});
  for (int32_t level = level_max - 1; level > 0; level--) {
    entry_points = layers_[level].SearchLayer(base_vector, ef_search_, entry_points);
  }
  auto vertex_ids = layers_[0].SearchLayer(base_vector, limit, entry_points);
  std::vector<RID> result;
  result.reserve(vertex_ids.size());
  for (const auto &id : vertex_ids) {
    result.push_back(rids_[id]);
  }
  return result;
}

//插入向量键：随机生成该顶点的层数，然后依次在每一层插入该顶点
void HNSWIndex::InsertVectorEntry(const std::vector<double> &key, RID rid) {
  auto id = AddVertex(key, rid);
  //随机生成层数
  std::uniform_real_distribution<double> distribution(0.0, 1.0);
  double r = -log(distribution(generator_)) * m_l_;
  int32_t level = static_cast<int32_t>(r);   //向下取整，得到随机后的层数
  
  auto target_level = level;
  int32_t level_max = (int32_t)layers_.size() - 1;
  //处理插入第一个顶点的特殊情况
  if(layers_[0].in_vertices_.empty()) {
    for(int32_t i = 0; i <= target_level; i++) {
      if(i == 0){
        layers_[0].AddVertex(id);
      } else {
        layers_.push_back({*vertices_, dist_fn_});
        layers_[i].AddVertex(id);
      }
    }
  return;
  }
  //如果随机生成的层数小于当前最大层数，则先从最大层开始搜索进入到下一层的入口，到达目标层后插入顶点并建立连接边，直到层0
  if (level < level_max) {
      std::vector<size_t> entry_points ;
      entry_points = layers_[level_max].SearchLayer(key, 1, {layers_[level_max].DefaultEntryPoint()});
      //逐层向下搜索直到目标层的上一层   
      for (int32_t curr_level = level_max - 1; curr_level >= level + 1; curr_level--) {
        entry_points = layers_[curr_level].SearchLayer(key, 1, entry_points);
      }
      //在目标层及以下层插入顶点并建立连接边
      for (int32_t curr_level = level; curr_level >= 0; curr_level--) {
        //插入顶点
        entry_points = layers_[curr_level].SearchLayer(key, ef_construction_, entry_points);
        layers_[curr_level].AddVertex(id);
        //建立连接边
        for (size_t i = 0, len = std::min(ef_construction_, entry_points.size()); i < len; i++) {
          layers_[level].Connect(id, entry_points[i]);
        }
        //修剪边
        layers_[curr_level].PurgeEdges(m_);
      }
  }else if(target_level >= level_max) {
    //如果随机生成的层数大于等于当前最大层数，则需要增加新的层或者仅在当前层插入顶点
    if(target_level == 0){
      layers_[0].Insert(key, id, ef_construction_, m_);
      return; 
    }
    //从当前最大层开始一致创建到目标层，每一个新创建的层都包含该顶点  
    for (int32_t level = level_max + 1; level <= target_level; level++) {
      layers_.push_back({*vertices_, dist_fn_});
      layers_[level].AddVertex(id);
      layers_[level].m_max_ = m_max_;
    }
    //然后再从当前最大层开始逐层向下插入顶点并建立连接边，直到层0
    std::vector<size_t> entry_points;
    entry_points = layers_[level_max].SearchLayer(key, ef_construction_, {layers_[level_max].DefaultEntryPoint()});
    if (level_max == 0) {
      layers_[level_max].AddVertex(id);
      for (size_t i = 0, len = std::min(ef_construction_, entry_points.size()); i < len; i++) {
        layers_[level_max].Connect(id, entry_points[i]);
      }
      // 裁剪边
      layers_[level_max].PurgeEdges(m_);
    } else {
      for (auto level = level_max - 1; level >= 0; level--) {
        entry_points = layers_[level].SearchLayer(key, ef_construction_, entry_points);
        layers_[level].AddVertex(id);
        for (size_t i = 0, len = std::min(ef_construction_, entry_points.size()); i < len; i++) {
          layers_[level].Connect(id, entry_points[i]);
        }
        // 裁剪边
        layers_[level].PurgeEdges(m_);
      }
    }
  }
}

}  // namespace bustub
