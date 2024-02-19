/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_TRANSFORM_GRAPH_IR_OP_DECLARE_MATH_OPS_DECLARE_H_
#define MINDSPORE_CCSRC_TRANSFORM_GRAPH_IR_OP_DECLARE_MATH_OPS_DECLARE_H_

#include "op_proto/inc/math_ops.h"
#include "op_proto/inc/ragged_math_ops.h"
#include "op_proto/inc/spectral_ops.h"
#include "transform/graph_ir/custom_op_proto/cust_math_ops.h"
#include "mindspore/ccsrc/include/common/utils/utils.h"
#include "transform/graph_ir/op_declare/op_declare_macro.h"
#include "utils/hash_map.h"

DECLARE_OP_ADAPTER(ActsULQ)
DECLARE_OP_USE_OUTPUT(ActsULQ)

DECLARE_OP_ADAPTER(ActsULQInputGrad)
DECLARE_OP_USE_OUTPUT(ActsULQInputGrad)

DECLARE_OP_ADAPTER(ActULQClampMaxGrad)
DECLARE_OP_USE_OUTPUT(ActULQClampMaxGrad)

DECLARE_OP_ADAPTER(ActULQClampMinGrad)
DECLARE_OP_USE_OUTPUT(ActULQClampMinGrad)

DECLARE_OP_ADAPTER(IFMR)
DECLARE_OP_USE_OUTPUT(IFMR)

DECLARE_OP_ADAPTER(NLLLoss)
DECLARE_OP_USE_OUTPUT(NLLLoss)

DECLARE_OP_ADAPTER(NLLLossGrad)
DECLARE_OP_USE_OUTPUT(NLLLossGrad)

DECLARE_OP_ADAPTER(Erf)
DECLARE_OP_USE_OUTPUT(Erf)

DECLARE_OP_ADAPTER(Erfc)
DECLARE_OP_USE_OUTPUT(Erfc)

DECLARE_OP_ADAPTER(WtsARQ)
DECLARE_OP_USE_OUTPUT(WtsARQ)

DECLARE_OP_ADAPTER(IsFinite)
DECLARE_OP_USE_OUTPUT(IsFinite)

DECLARE_OP_ADAPTER(IsNan)
DECLARE_OP_USE_OUTPUT(IsNan)

DECLARE_OP_ADAPTER(IsInf)
DECLARE_OP_USE_OUTPUT(IsInf)

DECLARE_OP_ADAPTER(LpNorm)
DECLARE_OP_USE_OUTPUT(LpNorm)

DECLARE_OP_ADAPTER(Trunc)
DECLARE_OP_USE_OUTPUT(Trunc)

DECLARE_OP_ADAPTER(HistogramFixedWidth)
DECLARE_OP_USE_OUTPUT(HistogramFixedWidth)

DECLARE_OP_ADAPTER(Pdist)
DECLARE_OP_USE_OUTPUT(Pdist)

DECLARE_OP_ADAPTER(SoftMarginLossGrad)
DECLARE_OP_USE_OUTPUT(SoftMarginLossGrad)

DECLARE_OP_ADAPTER(Cdist)
DECLARE_OP_USE_OUTPUT(Cdist)

DECLARE_OP_ADAPTER(CdistGrad)
DECLARE_OP_USE_OUTPUT(CdistGrad)

DECLARE_OP_ADAPTER(Conj)
DECLARE_OP_USE_OUTPUT(Conj)

DECLARE_OP_ADAPTER(NextAfter)
DECLARE_OP_USE_OUTPUT(NextAfter)

DECLARE_OP_ADAPTER(InitData)
DECLARE_OP_USE_OUTPUT(InitData)

DECLARE_OP_ADAPTER(GetNext)
DECLARE_OP_USE_DYN_OUTPUT(GetNext)

DECLARE_OP_ADAPTER(STFT)
DECLARE_OP_USE_OUTPUT(STFT)

DECLARE_OP_ADAPTER(Histogram)
DECLARE_OP_USE_OUTPUT(Histogram)

DECLARE_OP_ADAPTER(Complex)
DECLARE_OP_USE_OUTPUT(Complex)

DECLARE_OP_ADAPTER(Betainc)
DECLARE_OP_USE_OUTPUT(Betainc)

DECLARE_CUST_OP_ADAPTER(CholeskySolve)
DECLARE_CUST_OP_USE_OUTPUT(CholeskySolve)

DECLARE_OP_ADAPTER(ComplexAbs)
DECLARE_OP_USE_OUTPUT(ComplexAbs)

DECLARE_OP_ADAPTER(Bucketize)
DECLARE_OP_USE_OUTPUT(Bucketize)

DECLARE_CUST_OP_ADAPTER(Cauchy)
DECLARE_CUST_OP_USE_OUTPUT(Cauchy)

DECLARE_OP_ADAPTER(Bincount)
DECLARE_OP_USE_OUTPUT(Bincount)

DECLARE_CUST_OP_ADAPTER(CholeskyInverse)
DECLARE_CUST_OP_USE_OUTPUT(CholeskyInverse)

DECLARE_CUST_OP_ADAPTER(Eig)
DECLARE_CUST_OP_USE_OUTPUT(Eig)

DECLARE_CUST_OP_ADAPTER(Eps)
DECLARE_CUST_OP_USE_OUTPUT(Eps)

DECLARE_CUST_OP_ADAPTER(Hypot)
DECLARE_CUST_OP_USE_OUTPUT(Hypot)

DECLARE_CUST_OP_ADAPTER(MatrixLogarithm)
DECLARE_CUST_OP_USE_OUTPUT(MatrixLogarithm)

DECLARE_CUST_OP_ADAPTER(Lcm)
DECLARE_CUST_OP_USE_OUTPUT(Lcm)

DECLARE_CUST_OP_ADAPTER(MatrixExp)
DECLARE_CUST_OP_USE_OUTPUT(MatrixExp)

DECLARE_CUST_OP_ADAPTER(Heaviside)
DECLARE_CUST_OP_USE_OUTPUT(Heaviside)

DECLARE_CUST_OP_ADAPTER(Gcd)
DECLARE_CUST_OP_USE_OUTPUT(Gcd)

DECLARE_CUST_OP_ADAPTER(Orgqr)
DECLARE_CUST_OP_USE_OUTPUT(Orgqr)

DECLARE_OP_ADAPTER(RaggedRange)
DECLARE_OP_USE_OUTPUT(RaggedRange)

DECLARE_OP_ADAPTER(Imag)
DECLARE_OP_USE_OUTPUT(Imag)

DECLARE_CUST_OP_ADAPTER(Lgamma)
DECLARE_CUST_OP_USE_OUTPUT(Lgamma)

DECLARE_CUST_OP_ADAPTER(Diagonal)
DECLARE_CUST_OP_USE_OUTPUT(Diagonal)

DECLARE_CUST_OP_ADAPTER(FFT)
DECLARE_CUST_OP_USE_OUTPUT(FFT)

DECLARE_CUST_OP_ADAPTER(IFFT)
DECLARE_CUST_OP_USE_OUTPUT(IFFT)

DECLARE_CUST_OP_ADAPTER(FFTShift)
DECLARE_CUST_OP_USE_OUTPUT(FFTShift)

DECLARE_CUST_OP_ADAPTER(IFFTShift)
DECLARE_CUST_OP_USE_OUTPUT(IFFTShift)

DECLARE_CUST_OP_ADAPTER(Correlate)
DECLARE_CUST_OP_USE_OUTPUT(Correlate)

DECLARE_CUST_OP_ADAPTER(DCT)
DECLARE_CUST_OP_USE_OUTPUT(DCT)

DECLARE_CUST_OP_ADAPTER(Polar)
DECLARE_CUST_OP_USE_OUTPUT(Polar)

DECLARE_CUST_OP_ADAPTER(Real)
DECLARE_CUST_OP_USE_OUTPUT(Real)

DECLARE_CUST_OP_ADAPTER(TriuIndices)
DECLARE_CUST_OP_USE_OUTPUT(TriuIndices)

DECLARE_OP_ADAPTER(Digamma)
DECLARE_OP_USE_OUTPUT(Digamma)

DECLARE_CUST_OP_ADAPTER(TrilIndices)
DECLARE_CUST_OP_USE_OUTPUT(TrilIndices)

DECLARE_OP_ADAPTER(Angle)
DECLARE_OP_USE_OUTPUT(Angle)

DECLARE_CUST_OP_ADAPTER(Polygamma)
DECLARE_CUST_OP_USE_OUTPUT(Polygamma)

DECLARE_OP_ADAPTER(Igammac)
DECLARE_OP_USE_OUTPUT(Igammac)

DECLARE_CUST_OP_ADAPTER(FFTWithSize)
DECLARE_CUST_OP_USE_OUTPUT(FFTWithSize)

DECLARE_OP_ADAPTER(IgammaGradA)
DECLARE_OP_USE_OUTPUT(IgammaGradA)
#endif  // MINDSPORE_CCSRC_TRANSFORM_GRAPH_IR_OP_DECLARE_MATH_OPS_DECLARE_H_
