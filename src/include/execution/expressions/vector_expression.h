#pragma once

#include <cmath>
#include <string>
#include <utility>
#include <vector>

#include "catalog/schema.h"
#include "execution/expressions/abstract_expression.h"
#include "fmt/format.h"
#include "storage/table/tuple.h"
#include "type/value_factory.h"

namespace bustub {

/** ComparisonType represents the type of comparison that we want to perform. */
enum class VectorExpressionType { L2Dist, InnerProduct, CosineSimilarity };

inline auto ComputeDistance(const std::vector<double> &left, const std::vector<double> &right,
                            VectorExpressionType dist_fn) {
  auto sz = left.size();
  BUSTUB_ASSERT(sz == right.size(), "vector length mismatched!");
  switch (dist_fn) {
    case VectorExpressionType::L2Dist: {
      // 欧式距离
      double sum = 0.0;
      for (size_t i = 0; i < sz; i++) {
        double diff = left[i] - right[i];
        sum += diff * diff;
      }
      return std::sqrt(sum);
    }
    case VectorExpressionType::InnerProduct: {
      // 内积
      double sum = 0.0;
      for (size_t i = 0; i < sz; i++) {
        sum += left[i] * right[i];  
      }
      return sum;
    }
    case VectorExpressionType::CosineSimilarity: {
      // 余弦距离
      double dot_product = 0.0;
      for (size_t i = 0; i < sz; ++i) {
          dot_product += left[i] * right[i];
      }

      double norm_a = 0.0;
      double norm_b = 0.0;
      for (size_t i = 0; i < sz; ++i) {
          norm_a += left[i] * left[i];
          norm_b += right[i] * right[i];
      }
      norm_a = std::sqrt(norm_a);
      norm_b = std::sqrt(norm_b);

      double similarity = dot_product / (norm_a * norm_b);
      return similarity;
    }
    default:
      BUSTUB_ASSERT(false, "Unsupported vector expr type.");
  }
}

class VectorExpression : public AbstractExpression {
 public:
  VectorExpression(VectorExpressionType expr_type, AbstractExpressionRef left, AbstractExpressionRef right)
      : AbstractExpression({std::move(left), std::move(right)}, Column{"<val>", TypeId::DECIMAL}),
        expr_type_{expr_type} {}

  auto Evaluate(const Tuple *tuple, const Schema &schema) const -> Value override {
    Value lhs = GetChildAt(0)->Evaluate(tuple, schema);
    Value rhs = GetChildAt(1)->Evaluate(tuple, schema);
    return ValueFactory::GetDecimalValue(PerformComputation(lhs, rhs));
  }

  auto EvaluateJoin(const Tuple *left_tuple, const Schema &left_schema, const Tuple *right_tuple,
                    const Schema &right_schema) const -> Value override {
    Value lhs = GetChildAt(0)->EvaluateJoin(left_tuple, left_schema, right_tuple, right_schema);
    Value rhs = GetChildAt(1)->EvaluateJoin(left_tuple, left_schema, right_tuple, right_schema);
    return ValueFactory::GetDecimalValue(PerformComputation(lhs, rhs));
  }

  /** @return the string representation of the expression node and its children */
  auto ToString() const -> std::string override {
    return fmt::format("{}({}, {})", expr_type_, *GetChildAt(0), *GetChildAt(1));
  }

  BUSTUB_EXPR_CLONE_WITH_CHILDREN(VectorExpression);

  VectorExpressionType expr_type_;

 private:
  auto PerformComputation(const Value &lhs, const Value &rhs) const -> double {
    auto left_vec = lhs.GetVector();
    auto right_vec = rhs.GetVector();
    return ComputeDistance(left_vec, right_vec, expr_type_);
  }
};

}  // namespace bustub

template <>
struct fmt::formatter<bustub::VectorExpressionType> : formatter<string_view> {
  template <typename FormatContext>
  auto format(bustub::VectorExpressionType c, FormatContext &ctx) const {
    string_view name;
    switch (c) {
      case bustub::VectorExpressionType::L2Dist:
        name = "l2_dist";
        break;
      case bustub::VectorExpressionType::CosineSimilarity:
        name = "cosine_similarity";
        break;
      case bustub::VectorExpressionType::InnerProduct:
        name = "inner_product";
        break;
      default:
        name = "Unknown";
        break;
    }
    return formatter<string_view>::format(name, ctx);
  }
};
