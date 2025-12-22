//===----------------------------------------------------------------------===//
//
//                         BusTub
//
// seq_scan_executor.cpp
//
// Identification: src/execution/seq_scan_executor.cpp
//
// Copyright (c) 2015-2021, Carnegie Mellon University Database Group
//
//===----------------------------------------------------------------------===//

#include "execution/executors/seq_scan_executor.h"

namespace bustub {

SeqScanExecutor::SeqScanExecutor(ExecutorContext *exec_ctx, const SeqScanPlanNode *plan) :
 AbstractExecutor(exec_ctx),plan_(plan) {}

void SeqScanExecutor::Init() { 
    //初始化要扫描的表的table_heap和tablepage最开始的迭代器iter
    table_heap_ = GetExecutorContext() ->GetCatalog() -> GetTable(plan_ -> GetTableOid()) -> table_.get();
    auto iter = table_heap_ ->MakeIterator();
    rids_.clear();
    while(!iter.IsEnd()){
        rids_.push_back(iter.GetRID());
        ++iter;
    }
    rid_iter_ = rids_.begin();
    //throw NotImplementedException("SeqScanExecutor is not implemented"); 
}

auto SeqScanExecutor::Next(Tuple *tuple, RID *rid) -> bool {
    //顺序扫描，利用表堆的迭代器，从头到尾扫描整个表堆
    TupleMeta meta{};
    while(rid_iter_ != rids_.end()){
        //获取当前rid_iter对应的元数据和元组
        meta = table_heap_->GetTuple(*rid_iter_).first;
        //当当前的tuple为脏的时候，不记录它
        if(!meta.is_deleted_){
            *tuple = table_heap_ -> GetTuple(*rid_iter_).second;
            *rid = *rid_iter_;
        }
        ++rid_iter_;
        //检查是否满足条件，即未删除且通过过滤
        bool is_vaild = !meta.is_deleted_;
        if(plan_->filter_predicate_ != nullptr){
            is_vaild = is_vaild && plan_->filter_predicate_->
            Evaluate(tuple, GetExecutorContext()->GetCatalog()->GetTable(plan_->GetTableOid())->schema_).GetAs<bool>();
        }
        if(is_vaild){
            return true;
        }
    }
    return false; 
}

}  // namespace bustub
