/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_JIT_GRAPH_BPROP_FUNC_CONTEXT_H_
#define MINDSPORE_JIT_GRAPH_BPROP_FUNC_CONTEXT_H_

#include <memory>
#include <vector>
#include "ir/value.h"

namespace mindspore {
namespace jit {
namespace grad {
using InputList = std::vector<ValuePtr>;

/// \brief FunctionContext is a class, which include the inputs and output of FunctionNode.
class FunctionContext : public std::enable_shared_from_this<FunctionContext> {
 public:
  /// \brief The constructor of FunctionContext.
  ///
  /// \return The instance of FunctionContext.
  FunctionContext() : FunctionContext({}, nullptr) {}

  /// \brief The constructor of FunctionContext.
  ///
  /// \param[in] inputs The inputs of FunctionContext.
  ///
  /// \return The instance of FunctionContext.
  explicit FunctionContext(const InputList &inputs) : FunctionContext(inputs, nullptr) {}

  /// \brief The constructor of FunctionContext.
  ///
  /// \param[in] inputs The inputs of FunctionContext.
  /// \param[in] output The output of FunctionContext.
  ///
  /// \return The instance of FunctionContext.
  explicit FunctionContext(const InputList &inputs, const ValuePtr &output) : inputs_(inputs), output_(output) {}

  /// \brief Destructor.
  virtual ~FunctionContext() = default;

  /// \brief Get the inputs of the function node.
  ///
  /// \return The inputs of the function node.
  const InputList &GetInputs() const { return inputs_; }

  /// \brief Set the inputs of the function node.
  ///
  /// \param[in] inputs The inputs.
  void SetInputs(const InputList &inputs) { inputs_ = inputs; }

  /// \brief Add a input at the end of the input list.
  ///
  /// \param[in] input The input.
  void AddInput(const ValuePtr &input) { inputs_.push_back(input); }

  /// \brief Get the output of the function node.
  ///
  /// \return The output of the function node.
  const ValuePtr &GetOutput() const { return output_; }

  /// \brief Set the output of the function node.
  ///
  /// \param[in] output The output.
  void SetOutput(const ValuePtr &output) { output_ = output; }

  /// \brief Get the grad value of the function node.
  ///
  /// \return The grad value of the function node.
  const ValuePtr &GetGrad() const { return dout_; }

  /// \brief Set the output of the function node.
  ///
  /// \param[in] grad The dout_.
  void SetGrad(const ValuePtr &grad) { dout_ = grad; }

  /// \brief A helper templated function for casting "this" pointer to shared_ptr<derived>
  ///     Similar to shared_from_this, except this one will give you the derived class as shared_ptr
  /// \return A shared_ptr casted to the derived class
  template <typename Derived>
  std::shared_ptr<Derived> shared_from_base() {
    return std::static_pointer_cast<Derived>(shared_from_this());
  }

 private:
  /// \brief The input list of the function node.
  InputList inputs_;
  /// \brief The output of the function node.
  ValuePtr output_;
  /// \brief The delta out of the function node.
  ValuePtr dout_;
};

using FuncCtxPtr = std::shared_ptr<FunctionContext>;
}  // namespace grad
}  // namespace jit
}  // namespace mindspore
#endif  // MINDSPORE_JIT_GRAPH_BPROP_FUNC_CONTEXT_H_
