#include "execution/plans/abstract_plan.h"
#include "optimizer/optimizer.h"

// Note for 2023 Fall: You can add all optimizer rule implementations and apply the rules as you want in this file.
// Note that for some test cases, we force using starter rules, so that the configuration here won't take effects.
// Starter rule can be forcibly enabled by `set force_optimizer_starter_rule=yes`.

namespace bustub {

auto Optimizer::OptimizeCustom(const AbstractPlanNodeRef &plan) -> AbstractPlanNodeRef {
  //需要修改这里使得优化器的顺序符合要求
  auto p = plan;
  p = OptimizeMergeProjection(p);
  p = OptimizeMergeFilterNLJ(p);
  p = OptimizeNLJAsHashJoin(p);
  //p = OptimizeAsVectorIndexScan(p);
  p = OptimizeOrderByAsIndexScan(p);
  p = OptimizeSortLimitAsTopN(p);
  p = OptimizeAsVectorIndexScan(p); //将向量索引优化放在SortLimitAsTopN之后
  p = OptimizeMergeFilterScan(p);
  p = OptimizeSeqScanAsIndexScan(p);
  return p;
}

}  // namespace bustub
