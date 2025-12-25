#include "execution/executors/topn_executor.h"
#include <cstddef>
#include <queue>
#include "common/rid.h"
#include "storage/table/tuple.h"

namespace bustub {

TopNExecutor::TopNExecutor(ExecutorContext *exec_ctx, const TopNPlanNode *plan,
                           std::unique_ptr<AbstractExecutor> &&child_executor)
    : AbstractExecutor(exec_ctx),plan_(plan),child_executor_(std::move(child_executor)) {}

//【堆的结构】：每一个元素包含一个tuple和其向量和目标向量之间的距离值；维护一个【最大堆】，堆顶元素为距离值最大的元素
//思想：每次从子执行器中获取一个元组，计算其向量与目标向量之间的距离值，然后将该元组和距离值作为一个整体插入到一个大小为N的最大堆中。
//如果当前新的向量比堆顶的最远的向量更近，则将堆顶元素弹出，插入新的向量。
void TopNExecutor::Init() {
    child_executor_->Init();
    auto n = plan_->GetN();
    current_index_ = 0;
    Tuple tup;
    RID rid;
    //重置堆
    top_entries_ = std::priority_queue<SearchResult>();
    //从子执行器中拉取数据并维护一个大小为n的最大堆
    while(child_executor_->Next(&tup, &rid)){
        //计算距离,并从堆小于n和堆等于n两种情况进行处理
        for(const auto &[order_type, expr] : plan_->GetOrderBy()){
            auto val = expr->Evaluate(&tup, child_executor_->GetOutputSchema());
            double distance = val.GetAs<double>();
            if(top_entries_.size() < n){
                top_entries_.push(SearchResult{tup, distance});
            } else if(distance < top_entries_.top().distance_){
                top_entries_.pop();
                top_entries_.push(SearchResult{tup, distance});
            }
        }
    }
    while(!top_entries_.empty()){
        output_tuples_.emplace_back(top_entries_.top().tuple_);
        top_entries_.pop();
    }
    if(plan_->GetOrderBy()[0].first == OrderByType::ASC || plan_->GetOrderBy()[0].first == OrderByType::DEFAULT){
        //reverse为STL库中的反转容器/序列的元素的工具； reserve为vector的容器函数，为预分配容器的容量的工具
        std::reverse(output_tuples_.begin(),output_tuples_.end());
    }
}

auto TopNExecutor::Next(Tuple *tuple, RID *rid) -> bool { 
    if(current_index_ < output_tuples_.size()){
        *tuple = output_tuples_[current_index_];
        current_index_++;
        return true;
    }
    return false; 
}

//【思想】：这里有些类似于外部归并排序的思想。最开始是想着init()得到一个n大小的堆，在next()中一次性直接弹出n个元素返回，但这样违反了迭代器模型，且数据无法持久化。
//所以这里需要将排序和转换的逻辑放在Init中，next中就只需要把最终结果vector一个一个弹出来

auto TopNExecutor::GetNumInHeap() -> size_t {   //返回当前堆top_entries_的大小
    return output_tuples_.size();
};

}  // namespace bustub
