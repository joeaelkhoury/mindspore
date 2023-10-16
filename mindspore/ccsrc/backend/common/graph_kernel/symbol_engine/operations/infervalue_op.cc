/**
 * Copyright 2023 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include "backend/common/graph_kernel/symbol_engine/operations/infervalue_op.h"
#include <algorithm>
#include <utility>
#include "abstract/abstract_value.h"
#include "abstract/dshape.h"
#include "utils/check_convert_utils.h"
#include "backend/common/graph_kernel/symbol_engine/operations/infershape_op.h"

namespace mindspore::graphkernel::symbol {
namespace ops::infervalue {
SymbolPtr RealValue::GenVarByShape(const IListSymbol &shape) {
  if (shape.HasData()) {
    // scalar value
    if (shape.size() == 0) {
      return GenVInt();
    }
    if (shape.AllHaveData() && shape.size() == 1) {
      return GenVIntList(LongToSize(shape.item(0)));
    }
  }
  return GenVList();
}

SymbolPtr RealValue::GenListVariables(const ListSymbol &list) {
  auto ilist = list.as<IListSymbol>();
  if (ilist != nullptr) {
    return GenVarByShape(*ilist);
  }
  if (!list.HasData()) {
    return ListSymbol::Make(shared_from_this());
  }
  SymbolPtrList result(list.size());
  (void)std::transform(list.symbols().begin(), list.symbols().end(), result.begin(), [this](const SymbolPtr &shape) {
    auto inner_list = shape->as<ListSymbol>();
    MS_EXCEPTION_IF_NULL(inner_list);
    return GenListVariables(*inner_list);
  });
  return ListSymbol::Make(std::move(result));
}

SymbolPtr RealValue::Eval() {
  auto v = input_as<InputSymbol>(0)->abstract()->BuildValue();
  if (is_building() && v->isa<ValueAny>()) {
    OperationEmitter e;
    auto list = e.RealShape(input(0))->as_sptr<ListSymbol>();
    MS_EXCEPTION_IF_NULL(list);
    return GenListVariables(*list);
  }
  if (v->isa<ValueSequence>()) {
    return FromShape(GetValue<std::vector<int64_t>>(v), true);
  }
  if (v->isa<tensor::Tensor>()) {
    auto tensor_value = CheckAndConvertUtils::CheckTensorIntValue(v->ToString(), v, "RealValue");
    return FromShape(tensor_value, true);
  }
  if (v->isa<IntegerImm>()) {
    return GenInt(GetValue<int64_t>(v));
  }
  MS_LOG(EXCEPTION) << "Value should be one of {ValueSequence, Tensor, Integer}, but got " << v->ToString();
}
}  // namespace ops::infervalue
}  // namespace mindspore::graphkernel::symbol
