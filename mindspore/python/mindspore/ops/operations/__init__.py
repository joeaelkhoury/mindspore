# Copyright 2021-2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

"""
Primitive operator classes.

A collection of operators to build neural networks or to compute functions.
"""

from ._embedding_cache_ops import (CacheSwapTable, UpdateCache, MapCacheIdx, SubAndFilter,
                                   MapUniform, DynamicAssign, PadAndShift)
from ._inner_ops import (MatmulDDS, DSDMatmul, Cummin, ExtractImagePatches)
from ._quant_ops import *
from ._thor_ops import (CusBatchMatMul, CusCholeskyTrsm, CusFusedAbsMax1, CusImg2Col, CusMatMulCubeDenseLeft,
                        CusMatMulCubeFraczRightMul, CusMatMulCube, CusMatrixCombine, CusTranspose02314,
                        CusMatMulCubeDenseRight, CusMatMulCubeFraczLeftCast, NewIm2Col,
                        LoadIm2Col, UpdateThorGradient, CholeskyTrsm,
                        DetTriangle, ProdForceSeA)
from ._ms_kernel import (ms_kernel, kernel)
from .array_ops import (ArgMaxWithValue, ArgMinWithValue, Argmax, Argmin, BatchToSpace, BatchToSpaceND,
                        BatchToSpaceNDV2, BroadcastTo, Cast, Coalesce, Concat, Cummax, DType, DepthToSpace, Diag,
                        DiagPart, DynamicShape, EditDistance, EmbeddingLookup, ExpandDims, ExtractVolumePatches,
                        Eye, Fill, Gather, GatherD, GatherNd, GatherV2, Identity, Im2Col, InvertPermutation,
                        LowerBound, Lstsq, MaskedFill, MaskedSelect, Meshgrid, Mvlgamma, Ones, OnesLike,
                        Pack, Padding, ParallelConcat, PopulationCount, Range, Rank, Reshape, ResizeNearestNeighbor,
                        ReverseSequence, ReverseV2, Rint, ScalarToArray, ScalarToTensor, ScatterAdd,
                        ScatterDiv, ScatterMax, ScatterMin, ScatterMul, ScatterNd, ScatterNdAdd, ScatterNdDiv,
                        ScatterNdMax, ScatterNdMin, ScatterNdSub, ScatterNdUpdate, ScatterNonAliasingAdd, ScatterSub,
                        ScatterUpdate, SearchSorted, Select, Shape, Size, Slice, Sort, SpaceToBatch, SpaceToBatchND,
                        SpaceToDepth, SparseGatherV2, Split, SplitV, Squeeze, Stack, StridedSlice, TensorScatterAdd,
                        TensorScatterDiv, TensorScatterMax, TensorScatterMin, TensorScatterMul, TensorScatterSub,
                        TensorScatterUpdate, TensorShape, Tile, TopK, TransShape, Transpose, TupleToArray, Unique,
                        UniqueWithPad, Unpack, UnsortedSegmentMax, UnsortedSegmentMin, UnsortedSegmentProd,
                        UnsortedSegmentSum, Unstack, UpperBound, Zeros, ZerosLike, AffineGrid, Bincount, CheckNumerics,
                        HammingWindow, IdentityN, IndexFill, LeftShift, ListDiff, LogSpace, MatrixBandPart,
                        MatrixDiagPartV3, MatrixDiagV3, MatrixSetDiagV3, NonZero, Expand, Col2Im, ConjugateTranspose,
                        FillDiagonal, Fills, ResizeNearestNeighborV2, RightShift, ScatterAddWithAxis,
                        ScatterNdMul, SegmentMean, SegmentProd, SegmentSum, SegmentMax, SegmentMin, Tril, Triu,
                        UniqueConsecutive, UnravelIndex, FillV2, CountNonZero)
from .comm_ops import (AllGather, AllReduce, NeighborExchange, NeighborExchangeV2, AlltoAll, _AllSwap, ReduceScatter,
                       Broadcast,
                       _MirrorOperator, _MirrorMiniStepOperator, _MiniStepAllGather, ReduceOp, _VirtualDataset,
                       _VirtualOutput, _VirtualDiv, _GetTensorSlice, _VirtualAdd, _VirtualAssignAdd, _VirtualAccuGrad,
                       _HostAllGather, _HostReduceScatter, _MirrorMicroStepOperator, _MicroStepAllGather,
                       _VirtualPipelineEnd)
from .control_ops import GeSwitch, Merge
from .custom_ops import (Custom)
from .debug_ops import (ImageSummary, InsertGradientOf, HookBackward, ScalarSummary,
                        TensorSummary, HistogramSummary, Print, Assert)
from .image_ops import (CropAndResize, NonMaxSuppressionV3, HSVToRGB, AdjustHue, AdjustSaturation,
                        NonMaxSuppressionWithOverlaps, ResizeArea, ResizeBilinearV2, ExtractGlimpse,
                        CombinedNonMaxSuppression, RGBToHSV, ScaleAndTranslate, ResizeLinear1D, ResizeBicubic)
from .inner_ops import (ScalarCast, Randperm, NoRepeatNGram, LambApplyOptimizerAssign, LambApplyWeightAssign,
                        FusedWeightScaleApplyMomentum, FusedCastAdamWeightDecay, FusedAdaFactor,
                        FusedAdaFactorWithGlobalNorm)
from .linalg_ops import (Svd, Geqrf)
from .math_ops import (Abs, ACos, Asin, Asinh, AddN, AccumulateNV2, AssignAdd, AssignSub, Atan2, BatchMatMul,
                       BitwiseAnd, BitwiseOr, Ger, BitwiseXor, Inv, Invert, ApproximateEqual,
                       InplaceAdd, InplaceSub, InplaceUpdate, InplaceUpdateV2,
                       ReduceMax, ReduceMin, ReduceMean, ReduceSum, ReduceAll, ReduceProd, CumProd, Cdist, ReduceAny,
                       Cos, Cross, Div, DivNoNan, Equal, EqualCount, Exp, Expm1, Erf, Erfc, Floor, FloorDiv, FloorMod,
                       Ceil, Acosh, Greater, GreaterEqual, Lerp, Less, LessEqual, Log, Log1p, LogicalAnd, Mod,
                       LogicalNot, LogicalOr, LogicalXor, LpNorm, MatMul, Maximum, MulNoNan,
                       MatrixDeterminant, LogMatrixDeterminant, Minimum, Mul, Neg, NMSWithMask, NotEqual,
                       NPUAllocFloatStatus, NPUClearFloatStatus, LinSpace, Einsum, Renorm,
                       NPUGetFloatStatus, Pow, RealDiv, IsNan, IsInf, IsFinite, FloatStatus,
                       Reciprocal, CumSum, HistogramFixedWidth, SquaredDifference, Xdivy, Xlogy,
                       Sin, Sqrt, Rsqrt, BesselI0e, BesselI1e, TruncateDiv, TruncateMod, Addcdiv,
                       Addcmul, Square, Sub, TensorAdd, Add, Sign, Round, SquareSumAll, Atan, Atanh, Cosh, Sinh, Eps,
                       Tan, MatrixInverse, IndexAdd, Erfinv, Conj, Real, Imag, Complex, Trunc, IsClose, LuSolve,
                       CholeskyInverse, BesselJ0, BesselJ1, BesselK0, BesselK0e, BesselK1, BesselK1e, BesselY0,
                       BesselY1, Bucketize, Cauchy, Cholesky, CholeskySolve, Betainc,
                       FFTWithSize, Heaviside, Histogram, Hypot, Lcm, LuUnpack, MatrixExp,
                       MatrixLogarithm, MatrixPower, MatrixSolve, MatrixTriangularSolve, ReduceStd, STFT,
                       NextAfter, Orgqr, Qr, RaggedRange, Digamma, Eig, EuclideanNorm, CompareAndBitpack, ComplexAbs,
                       CumulativeLogsumexp, Gcd, Polygamma, Trace, TridiagonalMatMul, TrilIndices, TriuIndices, Zeta,
                       Roll, Lgamma, Logit, MatrixSolveLs, Polar, Fmax, Fmin, Quantile)
from .nn_ops import (LSTM, SGD, Adam, AdamWeightDecay, FusedSparseAdam, FusedSparseLazyAdam, AdamNoUpdateParam,
                     ApplyMomentum, BatchNorm, BiasAdd, Conv2D, Conv3D, Conv2DTranspose, Conv3DTranspose,
                     DepthwiseConv2dNative,
                     DropoutDoMask, Dropout, Dropout2D, Dropout3D, DropoutGenMask, Flatten,
                     InstanceNorm, BNTrainingReduce, BNTrainingUpdate,
                     GeLU, Gelu, FastGeLU, FastGelu, Elu, CeLU,
                     GetNext, L2Normalize, LayerNorm, L2Loss, CTCLoss, CTCLossV2, CTCLossV2Grad, CTCGreedyDecoder,
                     LogSoftmax, MaxPool3D, AvgPool3D,
                     MaxPool, DataFormatDimMap,
                     AvgPool, Conv2DBackpropInput, ComputeAccidentalHits,
                     MaxPoolWithArgmax, MaxPoolWithArgmaxV2, OneHot, Pad, MirrorPad, Mish, PReLU, ReLU, ReLU6, ReLUV2,
                     HSwish, HSigmoid,
                     ResizeBilinear, Sigmoid, SeLU, HShrink, ApplyKerasMomentum,
                     SigmoidCrossEntropyWithLogits, NLLLoss, BCEWithLogitsLoss,
                     SmoothL1Loss, SoftMarginLoss, Softmax, Softsign, Softplus, LRN, RNNTLoss, DynamicRNN, DynamicGRUV2,
                     SoftmaxCrossEntropyWithLogits, ROIAlign,
                     SparseSoftmaxCrossEntropyWithLogits, Tanh,
                     BinaryCrossEntropy, KLDivLoss, SparseApplyAdagrad, LARSUpdate, ApplyFtrl, SparseApplyFtrl,
                     ApplyProximalAdagrad, SparseApplyProximalAdagrad, SparseApplyAdagradV2, SparseApplyFtrlV2,
                     FusedSparseFtrl, FusedSparseProximalAdagrad, SparseApplyRMSProp, SparseApplyAdadelta,
                     ApplyAdaMax, ApplyAdadelta, ApplyAdagrad, ApplyAdagradV2, MultiMarginLoss, ApplyAdagradDA,
                     ApplyAddSign, ApplyPowerSign, ApplyGradientDescent, ApplyProximalGradientDescent,
                     ApplyRMSProp, ApplyCenteredRMSProp, BasicLSTMCell, InTopK, AdaptiveAvgPool2D, SoftShrink,
                     ApplyAdamWithAmsgrad, AdaptiveAvgPool3D, AdaptiveMaxPool2D, AdaptiveMaxPool3D,
                     GridSampler3D, MaxPool3DWithArgmax, MaxUnpool2D, NuclearNorm, NthElement, MultilabelMarginLoss,
                     Dilation2D, DataFormatVecPermute, DeformableOffsets, FractionalAvgPool,
                     FractionalMaxPool, FractionalMaxPool3DWithFixedKsize, FractionalMaxPoolWithFixedKsize,
                     GridSampler2D, TripletMarginLoss, UpsampleNearest3D, UpsampleTrilinear3D, PadV3)
from .other_ops import (Assign, IOU, BoundingBoxDecode, BoundingBoxEncode,
                        ConfusionMatrix, UpdateState, Load, StopGradient,
                        CheckValid, Partial, Depend, identity, Push, Pull, PyExecute, PyFunc, _DynamicLossScale,
                        SampleDistortedBoundingBoxV2)
from .random_ops import (RandomChoiceWithMask, StandardNormal, Gamma, RandomGamma, Poisson, UniformInt, UniformReal,
                         RandomCategorical, StandardLaplace, Multinomial, UniformCandidateSampler,
                         LogUniformCandidateSampler, TruncatedNormal, LogNormalReverse, NonDeterministicInts,
                         ParameterizedTruncatedNormal, RandomPoisson)
from .rl_ops import (BufferAppend, BufferGetItem, BufferSample)
from .sparse_ops import (SparseToDense, SparseTensorDenseMatmul, SparseTensorDenseAdd, SparseSlice)
from .spectral_ops import (BartlettWindow, BlackmanWindow)

__all__ = [
    'HSVToRGB',
    'CeLU',
    'Ger',
    'Unique',
    'ReverseSequence',
    'Sort',
    'EditDistance',
    'CropAndResize',
    'Add',
    'TensorAdd',
    'Argmax',
    'Argmin',
    'MaxPool3D',
    'AvgPool3D',
    'ArgMaxWithValue',
    'ArgMinWithValue',
    'AddN',
    'AccumulateNV2',
    'TensorShape',
    'Sub',
    'Coalesce',
    'CumSum',
    'MatMul',
    'BatchMatMul',
    'Mul',
    'MaskedFill',
    'MaskedSelect',
    'Meshgrid',
    'MultiMarginLoss',
    'Pow',
    'Exp',
    'Expm1',
    'Rsqrt',
    'Sqrt',
    'Square',
    'DynamicGRUV2',
    'SquaredDifference',
    'Xdivy',
    'Xlogy',
    'Conv2D',
    'Conv3D',
    'Conv2DTranspose',
    'Conv3DTranspose',
    'FillV2',
    'Flatten',
    'MaxPoolWithArgmax',
    'MaxPoolWithArgmaxV2',
    'BNTrainingReduce',
    'BNTrainingUpdate',
    'BatchNorm',
    'MaxPool',
    'TopK',
    'LinSpace',
    'Adam',
    'AdamWeightDecay',
    'FusedCastAdamWeightDecay',
    'FusedSparseAdam',
    'FusedSparseLazyAdam',
    'AdamNoUpdateParam',
    'FusedAdaFactor',
    'FusedAdaFactorWithGlobalNorm',
    'Softplus',
    'Softmax',
    'Softsign',
    'LogSoftmax',
    'SoftmaxCrossEntropyWithLogits',
    'BCEWithLogitsLoss',
    'ROIAlign',
    'SparseSoftmaxCrossEntropyWithLogits',
    'NLLLoss',
    'SGD',
    'ApplyMomentum',
    "ApplyKerasMomentum",
    'FusedWeightScaleApplyMomentum',
    'ExpandDims',
    'Einsum',
    'Renorm',
    'Cast',
    'Reshape',
    'Squeeze',
    'Transpose',
    'OneHot',
    'GatherV2',
    'Gather',
    'SparseGatherV2',
    'EmbeddingLookup',
    'Padding',
    'GatherD',
    'Identity',
    'UniqueWithPad',
    'Concat',
    'Pack',
    'NonMaxSuppressionV3',
    'Stack',
    'Unpack',
    'Unstack',
    'UpperBound',
    'Tile',
    'BiasAdd',
    'GeLU',
    'Gelu',
    'FastGeLU',
    'FastGelu',
    'Minimum',
    'Maximum',
    'StridedSlice',
    'ReduceSum',
    'ReduceMean',
    'LayerNorm',
    'Rank',
    'Lerp',
    'Less',
    'LessEqual',
    'LowerBound',
    'RealDiv',
    'Div',
    'DivNoNan',
    'Inv',
    'Invert',
    'Fill',
    'Ones',
    'Zeros',
    'OnesLike',
    'ZerosLike',
    'Select',
    'Split',
    'SplitV',
    'Mish',
    'SeLU',
    'MulNoNan',
    'ReLU',
    'ReLU6',
    'Elu',
    'Erf',
    "Erfinv",
    'Erfc',
    'Sigmoid',
    'HSwish',
    'HSigmoid',
    'Tanh',
    'NoRepeatNGram',
    'Randperm',
    'RandomChoiceWithMask',
    'StandardNormal',
    'Multinomial',
    'TruncatedNormal',
    'Gamma',
    'RandomGamma',
    'Mvlgamma',
    'Poisson',
    'UniformInt',
    'UniformReal',
    'StandardLaplace',
    'RandomCategorical',
    'ResizeBilinear',
    'ScalarSummary',
    'ImageSummary',
    'TensorSummary',
    'HistogramSummary',
    "Print",
    "Assert",
    'InsertGradientOf',
    'HookBackward',
    'InvertPermutation',
    'Shape',
    'DynamicShape',
    'DropoutDoMask',
    'DropoutGenMask',
    'Dropout',
    'Dropout2D',
    'Dropout3D',
    'Neg',
    'InplaceAdd',
    'InplaceSub',
    'Slice',
    'DType',
    'NPUAllocFloatStatus',
    'NPUGetFloatStatus',
    'NPUClearFloatStatus',
    'IsNan',
    'IsFinite',
    'IsInf',
    'Addcdiv',
    'Addcmul',
    'FloatStatus',
    'Reciprocal',
    'SmoothL1Loss',
    'SoftMarginLoss',
    'L2Loss',
    'CTCLoss',
    'CTCGreedyDecoder',
    'RNNTLoss',
    'DynamicRNN',
    'ReduceAll',
    'ReduceAny',
    'ScalarToArray',
    'ScalarToTensor',
    'TupleToArray',
    'GeSwitch',
    'Merge',
    'CheckValid',
    'BartlettWindow',
    'BlackmanWindow',
    'BoundingBoxEncode',
    'BoundingBoxDecode',
    'L2Normalize',
    'ScatterAdd',
    'ScatterSub',
    'ScatterMul',
    'ScatterDiv',
    'ScatterNd',
    'ScatterMax',
    'ScatterMin',
    'ScatterNdAdd',
    'ScatterNdSub',
    'ScatterNdDiv',
    'ScatterNdMin',
    'ScatterNdMax',
    'ScatterNonAliasingAdd',
    'ReverseV2',
    'Rint',
    'ResizeNearestNeighbor',
    'HistogramFixedWidth',
    'Pad',
    'MirrorPad',
    'GatherNd',
    'ScatterUpdate',
    'ScatterNdUpdate',
    'Floor',
    'NMSWithMask',
    'IOU',
    'Partial',
    'Depend',
    'UpdateState',
    'StopGradient',
    'identity',
    'AvgPool',
    # Back Primitive
    'Equal',
    'EqualCount',
    'NotEqual',
    'Greater',
    'GreaterEqual',
    'LogicalNot',
    'LogicalAnd',
    'LogicalOr',
    'LogicalXor',
    'Size',
    'DepthwiseConv2dNative',
    'UnsortedSegmentSum',
    'UnsortedSegmentMin',
    'UnsortedSegmentMax',
    'UnsortedSegmentProd',
    "AllGather",
    "AllReduce",
    "_AllSwap",
    "ReduceScatter",
    "Broadcast",
    "ReduceOp",
    'ScalarCast',
    'GetNext',
    'ReduceMax',
    'ReduceMin',
    'ReduceProd',
    'CumProd',
    'Cdist',
    'Log',
    'Log1p',
    'SigmoidCrossEntropyWithLogits',
    'FloorDiv',
    'FloorMod',
    'TruncateDiv',
    'TruncateMod',
    'Ceil',
    'Acosh',
    'Asinh',
    "PReLU",
    "Cos",
    "Cosh",
    "ACos",
    "Diag",
    "DiagPart",
    'Eye',
    'Assign',
    'AssignAdd',
    'AssignSub',
    "Sin",
    "Sinh",
    "Asin",
    "LSTM",
    "Lstsq",
    "Abs",
    "BinaryCrossEntropy",
    "KLDivLoss",
    "SparseApplyAdagrad",
    "SparseApplyAdagradV2",
    "SpaceToDepth",
    "DepthToSpace",
    "Conv2DBackpropInput",
    "ComputeAccidentalHits",
    "Sign",
    "LpNorm",
    "LARSUpdate",
    "Round",
    "Eps",
    "ApplyFtrl",
    "SpaceToBatch",
    "SparseApplyFtrl",
    "SparseApplyFtrlV2",
    "FusedSparseFtrl",
    "ApplyProximalAdagrad",
    "SparseApplyProximalAdagrad",
    "FusedSparseProximalAdagrad",
    "SparseApplyRMSProp",
    "ApplyAdaMax",
    "ApplyAdadelta",
    "ApplyAdagrad",
    "ApplyAdagradV2",
    "ApplyAdagradDA",
    "ApplyAdamWithAmsgrad",
    "ApplyAddSign",
    "ApplyPowerSign",
    "ApplyGradientDescent",
    "ApplyProximalGradientDescent",
    "BatchToSpace",
    "Atan2",
    "ApplyRMSProp",
    "ApplyCenteredRMSProp",
    "SpaceToBatchND",
    "BatchToSpaceND",
    "BatchToSpaceNDV2",
    "SquareSumAll",
    "BitwiseAnd",
    "BitwiseOr",
    "BitwiseXor",
    "BesselI0e",
    "BesselI1e",
    "Atan",
    "Atanh",
    "Tan",
    "BasicLSTMCell",
    "BroadcastTo",
    "DataFormatDimMap",
    "ApproximateEqual",
    "InplaceUpdate",
    "InplaceUpdateV2",
    "InTopK",
    "UniformCandidateSampler",
    "LogUniformCandidateSampler",
    "LRN",
    "Mod",
    "ConfusionMatrix",
    "PopulationCount",
    "ParallelConcat",
    "Push",
    "Pull",
    "ReLUV2",
    "SparseToDense",
    "SparseTensorDenseAdd",
    "SparseTensorDenseMatmul",
    "SparseApplyAdadelta",
    "MatrixInverse",
    "MatrixDeterminant",
    "LogMatrixDeterminant",
    "Range",
    "SearchSorted",
    "IndexAdd",
    "AdaptiveAvgPool2D",
    'TensorScatterUpdate',
    "TensorScatterMax",
    "TensorScatterMin",
    "TensorScatterAdd",
    "TensorScatterSub",
    "TensorScatterMul",
    "TensorScatterDiv",
    "SoftShrink",
    "HShrink",
    "PyExecute",
    "PyFunc",
    "BufferAppend",
    "BufferGetItem",
    "BufferSample",
    "Erfinv",
    "Conj",
    "Real",
    "Imag",
    "Trunc",
    "Complex",
    "ExtractVolumePatches",
    "NeighborExchangeV2",
    "NeighborExchange",
    "AlltoAll",
    "Custom",
    "LuSolve",
    "CholeskyInverse",
    "Cummax",
    "AdjustHue",
    "AdjustSaturation",
    "AffineGrid",
    "BesselJ0",
    "BesselJ1",
    "BesselK0",
    "BesselK0e",
    "BesselK1",
    "BesselK1e",
    "BesselY0",
    "BesselY1",
    "Bincount",
    "Bucketize",
    "Cauchy",
    "CheckNumerics",
    "AdaptiveAvgPool3D",
    "AdaptiveMaxPool2D",
    "AdaptiveMaxPool3D",
    "Cholesky",
    "CholeskySolve",
    "CTCLossV2",
    "Betainc",
    "FFTWithSize",
    "GridSampler3D",
    "HammingWindow",
    "Heaviside",
    "Histogram",
    "Hypot",
    "IdentityN",
    "IndexFill",
    "IsClose",
    "Lcm",
    "LeftShift",
    "ListDiff",
    "LogNormalReverse",
    "LogSpace",
    "LuUnpack",
    "Im2Col",
    "MapCacheIdx",
    "MatrixBandPart",
    "MatrixDiagPartV3",
    "MatrixDiagV3",
    "MatrixExp",
    "MatrixLogarithm",
    "MatrixPower",
    "MatrixSetDiagV3",
    "MatrixSolve",
    "MatrixTriangularSolve",
    "MaxPool3DWithArgmax",
    "MapUniform",
    "MaxUnpool2D",
    "ReduceStd",
    "STFT",
    "SubAndFilter",
    "Svd",
    "NuclearNorm",
    "NthElement",
    "MultilabelMarginLoss",
    "NonMaxSuppressionWithOverlaps",
    "NextAfter",
    "Orgqr",
    "NonDeterministicInts",
    "NonZero",
    "ResizeArea",
    "ResizeBilinearV2",
    "Qr",
    "ParameterizedTruncatedNormal",
    "RandomGamma",
    "RandomPoisson",
    "RaggedRange",
    "Digamma",
    "Dilation2D",
    "Eig",
    "Expand",
    "ExtractGlimpse",
    "EuclideanNorm",
    "CombinedNonMaxSuppression",
    "Col2Im",
    "CompareAndBitpack",
    "ComplexAbs",
    "ConjugateTranspose",
    "Cross",
    "Cummin",
    "CumulativeLogsumexp",
    "DataFormatVecPermute",
    "DeformableOffsets",
    "ExtractImagePatches",
    "FillDiagonal",
    "Fills",
    "Gcd",
    "Geqrf",
    "GridSampler2D",
    "FractionalAvgPool",
    "FractionalMaxPool",
    "FractionalMaxPool3DWithFixedKsize",
    "FractionalMaxPoolWithFixedKsize",
    "RGBToHSV",
    "ResizeNearestNeighborV2",
    "RightShift",
    "SampleDistortedBoundingBoxV2",
    "ScaleAndTranslate",
    "ScatterAddWithAxis",
    "ScatterNdMul",
    "SegmentMean",
    "SegmentProd",
    "SegmentSum",
    "SegmentMax",
    "SegmentMin",
    "TripletMarginLoss",
    "UpsampleNearest3D",
    "UpsampleTrilinear3D",
    "Trace",
    "TridiagonalMatMul",
    "Tril",
    "TrilIndices",
    "Triu",
    "TriuIndices",
    "UniqueConsecutive",
    "UnravelIndex",
    "Zeta",
    "PadV3",
    "Roll",
    "Lgamma",
    "SparseSlice",
    "ResizeLinear1D",
    "ResizeBicubic",
    "Logit",
    "MatrixSolveLs",
    "Polar",
    "Fmax",
    "Fmin",
    "Polygamma",
    "CountNonZero",
    "Quantile"
]

__custom__ = [
    "ms_kernel",
    "kernel",
]

__all__.extend(__custom__)

__all__.sort()
