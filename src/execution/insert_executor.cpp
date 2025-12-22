//===----------------------------------------------------------------------===//
//
//                         BusTub
//
// insert_executor.cpp
//
// Identification: src/execution/insert_executor.cpp
//
// Copyright (c) 2015-2021, Carnegie Mellon University Database Group
//
//===----------------------------------------------------------------------===//

#include <memory>

#include "execution/executors/insert_executor.h"

namespace bustub {

InsertExecutor::InsertExecutor(ExecutorContext *exec_ctx, const InsertPlanNode *plan,
                               std::unique_ptr<AbstractExecutor> &&child_executor)
    : AbstractExecutor(exec_ctx) ,plan_(plan),child_executor_(std::move(child_executor)){}

void InsertExecutor::Init() { 
    //子执行器为valuesexecutor
    this -> child_executor_ -> Init();
}

auto InsertExecutor::Next(Tuple *tuple, RID *rid) -> bool {
    if(emitted_){
        return false;
    }   
    //先设置插入标志位
    emitted_ = true;
    //count 表示插入操作期间插入的行数;获取待插入的表信息和索引列表
    int count = 0;
    auto table_info = this -> exec_ctx_ ->GetCatalog() ->GetTable(this ->plan_->GetTableOid());
    auto schema = table_info ->schema_;
    auto indexes = this -> exec_ctx_->GetCatalog()->GetTableIndexes(table_info->name_);
    //从下一个算子中逐个获得元组并插入到表中，更新所有索引（索引相关的暂时不太清楚）
    //调用value的next，将plan中的value树都插入其中，如果每一行该插入的都插入了，就会返回false；每调用一次,cursor_的值就会增加1
    //待插入的数据在plan_中
    while(this->child_executor_->Next(tuple, rid)){      
        count++;
        table_info -> table_ -> InsertTuple(TupleMeta{0,false}, *tuple);
        //======================暂时没有管索引更新的问题==========================  
    }
    std::vector<Value> result = {{TypeId::INTEGER,count}};  //构造包含元素count的向量result
    *tuple = Tuple(result,&GetOutputSchema());
    return true; 
}

}  // namespace bustub
