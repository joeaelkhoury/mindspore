/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#include "transform/graph_ir/op_declare/linalg_ops_declare.h"

#include "mindspore/core/ops/arithmetic_ops.h"
#include "mindspore/core/ops/math_ops.h"
namespace mindspore::transform {
// Ger
INPUT_MAP(Ger) = {{1, INPUT_DESC(x1)}, {2, INPUT_DESC(x2)}};
ATTR_MAP(Ger) = EMPTY_ATTR_MAP;
OUTPUT_MAP(Ger) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(Ger, prim::kGer, ADPT_DESC(Ger))

// Svd
INPUT_MAP(Svd) = {{1, INPUT_DESC(x)}};
ATTR_MAP(Svd) = {{"compute_uv", ATTR_DESC(compute_uv, AnyTraits<bool>())},
                 {"full_matrices", ATTR_DESC(full_matrices, AnyTraits<bool>())}};
OUTPUT_MAP(Svd) = {{0, OUTPUT_DESC(sigma)}, {1, OUTPUT_DESC(u)}, {2, OUTPUT_DESC(v)}};
REG_ADPT_DESC(Svd, prim::kPrimSvd->name(), ADPT_DESC(Svd))

// LogMatrixDeterminant
INPUT_MAP(LogMatrixDeterminant) = {{1, INPUT_DESC(x)}};
ATTR_MAP(LogMatrixDeterminant) = EMPTY_ATTR_MAP;
OUTPUT_MAP(LogMatrixDeterminant) = {{0, OUTPUT_DESC(sign)}, {1, OUTPUT_DESC(y)}};
REG_ADPT_DESC(LogMatrixDeterminant, prim::kLogMatrixDeterminant, ADPT_DESC(LogMatrixDeterminant))

// MatrixInverse
INPUT_MAP(MatrixInverse) = {{1, INPUT_DESC(x)}};
ATTR_MAP(MatrixInverse) = {{"adjoint", ATTR_DESC(adjoint, AnyTraits<bool>())}};
OUTPUT_MAP(MatrixInverse) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(MatrixInverse, prim::kMatrixInverse, ADPT_DESC(MatrixInverse))

// MatrixDeterminant
INPUT_MAP(MatrixDeterminant) = {{1, INPUT_DESC(x)}};
ATTR_MAP(MatrixDeterminant) = EMPTY_ATTR_MAP;
OUTPUT_MAP(MatrixDeterminant) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(MatrixDeterminant, prim::kMatrixDeterminant, ADPT_DESC(MatrixDeterminant))

// MatrixSolve
INPUT_MAP(MatrixSolve) = {{1, INPUT_DESC(matrix)}, {2, INPUT_DESC(rhs)}};
ATTR_MAP(MatrixSolve) = {{"adjoint", ATTR_DESC(adjoint, AnyTraits<bool>())}};
OUTPUT_MAP(MatrixSolve) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(MatrixSolve, prim::kMatrixSolve, ADPT_DESC(MatrixSolve))
}  // namespace mindspore::transform
