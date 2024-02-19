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

#ifndef MINDSPORE_CCSRC_GRAPH_IR_CUSTOM_OP_PROTO_CUST_NN_TRAINING_OPS_H_
#define MINDSPORE_CCSRC_GRAPH_IR_CUSTOM_OP_PROTO_CUST_NN_TRAINING_OPS_H_

#include "graph/operator_reg.h"
#include "graph/operator.h"
#include "transform/graph_ir/custom_op_proto/op_proto_macro.h"

/* clang-format off */

namespace ge {
REG_CUST_OP(FusedSparseProximalAdagrad)
  .INPUT(var, TensorType({DT_FLOAT}))
  .INPUT(accum, TensorType({DT_FLOAT}))
  .INPUT(lr, TensorType({DT_FLOAT}))
  .INPUT(l1, TensorType({DT_FLOAT}))
  .INPUT(l2, TensorType({DT_FLOAT}))
  .INPUT(grad, TensorType({DT_FLOAT}))
  .INPUT(indices, TensorType({DT_INT32}))
  .OUTPUT(var, TensorType({DT_FLOAT}))
  .OUTPUT(accum, TensorType({DT_FLOAT}))
  .REQUIRED_ATTR(use_locking, Bool)
  .CUST_OP_END_FACTORY_REG(FusedSparseProximalAdagrad)
}  // namespace ge
#endif  // MINDSPORE_CCSRC_GRAPH_IR_CUSTOM_OP_PROTO_CUST_NN_TRAINING_OPS_H_
