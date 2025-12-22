//===----------------------------------------------------------------------===//
//
//                         BusTub
//
// limit_executor.cpp
//
// Identification: src/execution/limit_executor.cpp
//
// Copyright (c) 2015-2021, Carnegie Mellon University Database Group
//
//===----------------------------------------------------------------------===//

#include "execution/executors/limit_executor.h"

namespace bustub {

LimitExecutor::LimitExecutor(ExecutorContext *exec_ctx, const LimitPlanNode *plan,
                             std::unique_ptr<AbstractExecutor> &&child_executor)
    : AbstractExecutor(exec_ctx),plan_(plan),child_executor_(std::move(child_executor)) {}

void LimitExecutor::Init() { 
    child_executor_->Init();
    std::size_t count = 0;    //记录已经输出的元组数量
    auto limit = plan_->GetLimit();  //获取plan中的给定的限制的元组数量
    tuples_.clear();
    Tuple tuple;
    RID rid;
    //从子执行器中获取元组，直到达到限制数量或者没有更多元组
    while (count < limit && child_executor_->Next(&tuple, &rid)) {
        tuples_.emplace_back(tuple);
        count++;
    }
    if(!tuples_.empty()) {
        iter_ = tuples_.begin();
    }
    //throw NotImplementedException("LimitExecutor is not implemented"); 
}

auto LimitExecutor::Next(Tuple *tuple, RID *rid) -> bool { 
    if(!tuples_.empty() && iter_ != tuples_.end()) {
        *tuple = *iter_;
        ++iter_;
        return true;
    }
    return false; 
}

}  // namespace bustub
