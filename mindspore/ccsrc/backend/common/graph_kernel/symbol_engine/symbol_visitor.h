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
#ifndef MINDSPORE_CCSRC_BACKEND_COMMON_GRAPH_KERNEL_SYMBOL_ENGINE_SYMBOL_VISITOR_H_
#define MINDSPORE_CCSRC_BACKEND_COMMON_GRAPH_KERNEL_SYMBOL_ENGINE_SYMBOL_VISITOR_H_
#include "backend/common/graph_kernel/symbol_engine/symbol.h"
#include "backend/common/graph_kernel/symbol_engine/operations/operation.h"
#include "backend/common/graph_kernel/symbol_engine/operations/common_op.h"
#include "backend/common/graph_kernel/symbol_engine/operations/infershape_op.h"
#include "backend/common/graph_kernel/symbol_engine/operations/infervalue_op.h"

namespace mindspore::graphkernel::symbol {
class SymbolVisitor {
 public:
  SymbolVisitor() = default;
  ~SymbolVisitor() = default;

  void Visit(Symbol *symbol);

  virtual void Visit(DynamicSymbol *symbol) {}
  virtual void Visit(InputSymbol *symbol) {}
  virtual void Visit(ScalarSymbol *symbol) {}
  virtual void Visit(IntSymbol *symbol) {}
  virtual void Visit(BoolSymbol *symbol) {}
  virtual void Visit(FloatSymbol *symbol) {}
  virtual void Visit(ListSymbol *symbol) {}
  virtual void Visit(IListSymbol *symbol) {}

  virtual void Visit(ops::Operation *op) { VisitInputs(op); }

  inline void VisitInputs(ops::Operation *op) {
    for (auto &s : op->inputs()) {
      Visit(s.get());
    }
  }
};
}  // namespace mindspore::graphkernel::symbol
#endif  // MINDSPORE_CCSRC_BACKEND_COMMON_GRAPH_KERNEL_SYMBOL_ENGINE_SYMBOL_VISITOR_H_
