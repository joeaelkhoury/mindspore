# Copyright 2024 Huawei Technologies Co., Ltd
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
import numpy as np
from mindspore.ops.auto_generate import QuantBatchMatmul
from mindspore import Tensor
from mindspore import context
import mindspore.nn as nn
from mindspore.common import dtype as mstype
import pytest

def np_quant(np_data, scale=256.0, offset=-128.0, sqrt_mode=False):
    if sqrt_mode:
        np_quant_data = np.round(np_data * scale * scale + offset).astype(np.int8)
    else:
        np_quant_data = np.round(np_data * scale + offset).astype(np.int8)
    return np_quant_data

def np_dequant(np_quant_data, dep_scale, sqrt_mode=False):
    np_quant_data = np_quant_data.astype(np.float32)
    if sqrt_mode:
        np_dequant_data = np_quant_data * dep_scale * dep_scale
    else:
        np_dequant_data = np_quant_data * dep_scale
    return np_dequant_data.astype(np.float16)

def np_requant(np_data, scale, offset, sqrt_mode=False):
    np_data = np_data.astype(np.int32)
    if sqrt_mode:
        np_requant_data = np.round(np_data * scale * scale + offset).astype(np.int8)
    else:
        np_requant_data = np.round(np_data * scale + offset).astype(np.int8)
    return np_requant_data

def trans_float32_scale_deq_to_uint64(scale_deq):
    float32_scale_deq = np.array(scale_deq, np.float32)
    uint32_scale_deq = np.frombuffer(float32_scale_deq, np.uint32)
    uint64_result = np.zeros(float32_scale_deq.shape, np.uint64)
    uint64_result |= np.uint64(uint32_scale_deq)
    return uint64_result


class NpQuantBatchMatmulNet():
    def __init__(self, transpose_x1=False, transpose_x2=False, dtype=mstype.float16):
        super().__init__()
        self.transpose_x1 = transpose_x1
        self.transpose_x2 = transpose_x2
        self.dtype = dtype

    def run(self, x1, x2, scale, offset, bias):
        input_data = x1
        weight = x2
        if self.transpose_x1:
            axes = list(range(len(x1.shape)))
            axes[-1], axes[-2] = axes[-2], axes[-1]
            input_data = np.transpose(x1.astype(np.int32), axes=tuple(axes))
        if self.transpose_x2:
            axes = list(range(len(x2.shape)))
            axes[-1], axes[-2] = axes[-2], axes[-1]
            weight = np.transpose(weight.astype(np.int32), axes=tuple(axes))

        out = np.matmul(input_data.astype(np.int32), weight.astype(np.int32)) + bias
        if self.dtype == mstype.int8:
            out = np.clip(out * scale + offset, -128, 127).astype(np.int8)
        else:
            out = np_dequant(out, scale)    # must be uint64
        return out

def np_qbmm_valid_net(x1_int8, x2_int8, scale, offset, bias):
    qbmm_net = NpQuantBatchMatmulNet(transpose_x1=False, transpose_x2=False, dtype=mstype.float16)
    out = qbmm_net.run(x1_int8, x2_int8, scale, offset, bias)
    return out


class QuantBatchMatmulNet(nn.Cell):
    """
    QuantBatchMatmulNet.
    """
    def __init__(self, transpose_x1=False, transpose_x2=False, dtype=mstype.float16):
        super().__init__()
        self.qbmm = QuantBatchMatmul(transpose_x1, transpose_x2, dtype)

    def construct(self, x1, x2, scale, offset, bias):
        out = self.qbmm(x1, x2, scale, offset, bias)
        return out

class QuantBatchMatmulValidNet():
    def __init__(self, transpose_x1=False, transpose_x2=False, dtype=mstype.float16, scale_fp16_flag=False):
        super().__init__()
        self.ms_qbmm_net = QuantBatchMatmulNet(transpose_x1, transpose_x2, dtype)
        self.np_qbmm_net = NpQuantBatchMatmulNet(transpose_x1, transpose_x2, dtype)
        self.transpose_x1 = transpose_x1
        self.transpose_x2 = transpose_x2
        self.dtype = dtype
        self.scale_fp16_flag = scale_fp16_flag

    def gen_input(self, batch, row, deep, col, channel_flag=True):
        x1_row, x1_col = (deep, row) if self.transpose_x1 else (row, deep)
        x2_row, x2_col = (col, deep) if self.transpose_x2 else (deep, col)

        if batch is None:
            x1 = np.random.rand(x1_row * x1_col).reshape((x1_row, x1_col))
            x1 = np_quant(x1, scale=256.0, offset=-128.0, sqrt_mode=False)
            x2 = np.random.rand(x2_row * x2_col).reshape((x2_row, x2_col))
            x2 = np_quant(x2, scale=256.0, offset=-128.0, sqrt_mode=False)
        else:
            x1 = np.random.rand(batch * row * deep).reshape((batch, x1_row, x1_col))
            x1 = np_quant(x1, scale=256.0, offset=-128.0, sqrt_mode=False)
            x2 = np.random.rand(batch * deep * col).reshape((batch, x2_row, x2_col))
            x2 = np_quant(x2, scale=256.0, offset=-128.0, sqrt_mode=False)

        channel_col = col if channel_flag else 1
        scale = (np.ones([channel_col]) * (((1 - 0) / 256) * (5 / 256))).astype(np.float32)
        offset = np.random.rand(channel_col).astype(np.float32)
        bias = (np.random.rand(col) * 10).astype(np.int32)
        return x1, x2, scale, offset, bias

    def run(self, x1, x2, scale, offset, bias):
        np_out = self.np_qbmm_net.run(x1, x2, scale, offset, bias)

        if self.dtype != mstype.float16:
            ms_x1 = Tensor(x1)
            ms_x2 = Tensor(x2, const_arg=True)
            ms_scale = Tensor(scale, const_arg=True)
            ms_offset = Tensor(offset, const_arg=True)
            ms_bias = Tensor(bias, const_arg=True)
        else:
            ms_x1 = Tensor(x1)
            ms_x2 = Tensor(x2, const_arg=True)
            ms_scale = Tensor(trans_float32_scale_deq_to_uint64(scale), const_arg=True)
            ms_offset = Tensor(offset, const_arg=True)
            ms_bias = Tensor(bias, const_arg=True)
        ms_out = self.ms_qbmm_net(ms_x1, ms_x2, ms_scale, ms_offset, ms_bias)

        print(ms_out.asnumpy())
        print(np_out)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.env_onecard
def test_quant_batch_matmul_v2():
    """
    Feature: Test quant_batch_matmul
    Description: quant_batch_matmul
    Expectation: Success
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    np.random.seed(42)
    valid_net = QuantBatchMatmulValidNet(transpose_x1=False,
                                         transpose_x2=False,
                                         dtype=mstype.float16,
                                         scale_fp16_flag=False)
    x1, x2, scale, offset, bias = valid_net.gen_input(batch=None, row=256, deep=128, col=64, channel_flag=True)
    valid_net.run(x1, x2, scale, offset, bias)
