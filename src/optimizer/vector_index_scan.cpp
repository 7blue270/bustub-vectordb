#include <cstdint>
#include <memory>
#include <optional>
#include "binder/bound_order_by.h"
#include "catalog/catalog.h"
#include "catalog/column.h"
#include "concurrency/transaction.h"
#include "execution/expressions/array_expression.h"
#include "execution/expressions/column_value_expression.h"
#include "execution/expressions/comparison_expression.h"
#include "execution/expressions/constant_value_expression.h"
#include "execution/expressions/vector_expression.h"
#include "execution/plans/abstract_plan.h"
#include "execution/plans/index_scan_plan.h"
#include "execution/plans/limit_plan.h"
#include "execution/plans/topn_plan.h" 
#include "execution/plans/projection_plan.h"
#include "execution/plans/seq_scan_plan.h"
#include "execution/plans/sort_plan.h"
#include "execution/plans/vector_index_scan_plan.h"
#include "fmt/core.h"
#include "optimizer/optimizer.h"
#include "type/type.h"
#include "type/type_id.h"

namespace bustub {

  //用来判断是否有合适的向量索引可以使用
auto MatchVectorIndex(const Catalog &catalog, table_oid_t table_oid, uint32_t col_idx, VectorExpressionType dist_fn,
                      const std::string &vector_index_match_method) -> const IndexInfo * {
  //先获取表的所有索引
  auto tablename = catalog.GetTable(table_oid)->name_;
  auto indexes = catalog.GetTableIndexes(tablename);
  const IndexInfo *matched_index = nullptr; 

  //逐个检查索引是否符合要求
  for(const auto *index_info :indexes){
    if(index_info->key_schema_.GetColumn(0).GetOffset() != col_idx) { 
      continue;
    }
    auto index_type = index_info->index_type_;
    if(vector_index_match_method == "ivfflat" && index_type != IndexType::VectorIVFFlatIndex) continue;
    if(vector_index_match_method == "hnsw" && index_type != IndexType::VectorHNSWIndex) continue;   
    if(vector_index_match_method == "none") return nullptr;

    //检查距离函数是否匹配
    const auto *vector_index = dynamic_cast<const VectorIndex *>(index_info->index_.get());
    if(vector_index == nullptr) continue; 
    if(vector_index->distance_fn_ == dist_fn) {
      return index_info;
    }
    if(matched_index == nullptr) {
      matched_index = index_info; 
    } 
  } 

  //如果策略是空或者unset，则返回找到的第一个索引
  if(vector_index_match_method.empty() && !indexes.empty()){
    return matched_index;
  }
  return nullptr;
}

auto Optimizer::OptimizeAsVectorIndexScan(const AbstractPlanNodeRef &plan) -> AbstractPlanNodeRef {
  std::vector<AbstractPlanNodeRef> children;
  for (const auto &child : plan->GetChildren()) {
    children.emplace_back(OptimizeAsVectorIndexScan(child));
  }
  auto optimized_plan = plan->CloneWithChildren(std::move(children));

  if(optimized_plan->GetType() == PlanType::TopN){
    const auto &topn_plan = dynamic_cast<const TopNPlanNode &>(*optimized_plan);

    //检查order by条件是否只有一个，并且是向量距离计算（只有当order by只有一个条件时，才考虑使用向量索引）
    if(topn_plan.GetOrderBy().size() != 1) {
      return optimized_plan;    
    }
    auto order_bys = topn_plan.GetOrderBy();
    auto dist_expr = std::dynamic_pointer_cast<VectorExpression>(order_bys[0].second);
    if(!dist_expr) {
      return optimized_plan;    
    }

    //识别子节点结构，找出待检索的列和查询向量
    AbstractExpressionRef maybe_left = dist_expr->GetChildAt(0);
    AbstractExpressionRef maybe_right = dist_expr->GetChildAt(1);
    std::shared_ptr<const ArrayExpression> base_vector = nullptr;
    std::shared_ptr<const ColumnValueExpression> col_val_expr = nullptr; 
    if(auto left_array_expr = std::dynamic_pointer_cast<ArrayExpression>(maybe_left)) {
      base_vector = left_array_expr;
      col_val_expr = std::dynamic_pointer_cast<ColumnValueExpression>(maybe_right);
    } else if(auto right_array_expr = std::dynamic_pointer_cast<ArrayExpression>(maybe_right)) {
      base_vector = right_array_expr;
      col_val_expr = std::dynamic_pointer_cast<ColumnValueExpression>(maybe_left);
    } else {
      return optimized_plan;    
    }
    //确保只有当为Column <-> Constant 这种标准模式才能进入后续的索引优化逻辑
    if (!col_val_expr || !base_vector) return optimized_plan;

    //定位到topn的子节点，并分析是否是projection或seqscan 
    AbstractPlanNodeRef current = topn_plan.GetChildAt(0);
    const ProjectionPlanNode *proj_plan = nullptr;
    uint32_t target_col_idx = col_val_expr->GetColIdx();
    if(current->GetType() == PlanType::Projection) {
      proj_plan = dynamic_cast<const ProjectionPlanNode *>(current.get());
      if(!proj_plan){
        return optimized_plan;    
      }
      const auto &proj_exprs = proj_plan->GetExpressions();
      if(target_col_idx >= proj_exprs.size()) {
        return optimized_plan;    
      }   
      const auto *inner_col = dynamic_cast<const ColumnValueExpression *>(proj_exprs[target_col_idx].get());
      if(!inner_col) {
        return optimized_plan;
      }
      target_col_idx = inner_col->GetColIdx();
      current = proj_plan->GetChildAt(0);
    }

    //现在current应该是SeqScan节点
    if(current->GetType() != PlanType::SeqScan) {
      return optimized_plan;  
    }
    const auto *seq_scan = dynamic_cast<const SeqScanPlanNode *>(current.get()); 
    if(!seq_scan) {
      return optimized_plan;  
    }

    //进行索引匹配
    const IndexInfo *matched = MatchVectorIndex(
        catalog_, seq_scan->GetTableOid(), target_col_idx, dist_expr->expr_type_,
        vector_index_match_method_);
    if(!matched) {  
      return optimized_plan;  
    }
    auto vector_index_scan = std::make_shared<VectorIndexScanPlanNode>(
        topn_plan.output_schema_, seq_scan->GetTableOid(),seq_scan->table_name_, matched->index_oid_,
        matched->name_, base_vector, topn_plan.GetN()); 
    //投影补充：如果原来有投影，则保留投影
    if(proj_plan) {
      return std::make_shared<ProjectionPlanNode>(
          topn_plan.output_schema_ , proj_plan->GetExpressions(), vector_index_scan);
    }
    return vector_index_scan;
  }
  return optimized_plan;
}

}  // namespace bustub
