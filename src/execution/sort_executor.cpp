#include "execution/executors/sort_executor.h"
#include <algorithm>
#include <queue>

namespace bustub {

SortExecutor::SortExecutor(ExecutorContext *exec_ctx, const SortPlanNode *plan,
                           std::unique_ptr<AbstractExecutor> &&child_executor)
    : AbstractExecutor(exec_ctx) ,plan_(plan),child_executor_(std::move(child_executor)){}

//从子执行器中拉去所有数据，在内存中排序，然后输出排序结果；使用std::sort进行排序
void SortExecutor::Init() { 
    child_executor_->Init();
    current_index_ = 0;
    sort_tuples_.clear();
    Tuple tup;
    RID rid;
    while(child_executor_->Next(&tup, &rid)){
        sort_tuples_.emplace_back(tup);
    }
    std::sort(sort_tuples_.begin(), sort_tuples_.end(),
        [this](const Tuple &a, const Tuple &b) {
            for (const auto &[order_type, expr] : plan_->GetOrderBy()) {   //表达式中，order_type表示计算得到的距离该怎样排序，expr表示该用什么样的表达式来计算距离
                auto val_a = expr->Evaluate(&a, child_executor_->GetOutputSchema());
                auto val_b = expr->Evaluate(&b, child_executor_->GetOutputSchema());
                if (val_a.CompareLessThan(val_b) == CmpBool::CmpTrue) {
                    return order_type == OrderByType::ASC || order_type == OrderByType::DEFAULT;
                } else if (val_a.CompareGreaterThan(val_b) == CmpBool::CmpTrue) {
                    return order_type == OrderByType::DESC;
                }
            }
            return false; // they are equal
        });
}

auto SortExecutor::Next(Tuple *tuple, RID *rid) -> bool {
    auto n = sort_tuples_.size();
    if(current_index_ >= n) {
        return false;
    }
    *tuple = sort_tuples_[current_index_++];
    return true; 
}

}  // namespace bustub
