# Copyright 2023 Huawei Technologies Co., Ltd
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

import numpy as np

import mindspore as ms
from mindspore import context, Tensor, Parameter
from mindspore.nn import Cell
from mindspore.ops.operations._inner_ops import AntiQuant
from parallel.utils.utils import ParallelValidator, compile_net

def setup_function():
    context.set_auto_parallel_context(dataset_strategy="full_batch")


class AntiQuantNet(Cell):
    def __init__(self, scale, offset, sqrt_mode, strategy):
        super().__init__()
        self.anti_quant = AntiQuant(scale, offset, sqrt_mode).shard(strategy)

    def construct(self, data):
        out = self.anti_quant(data)
        return out


def test_anti_quant_1D():
    """
    Feature: test quant ops
    Description:
    Expectation: compile success
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=4, global_rank=0)

    strategy = ((4,),)

    net = AntiQuantNet(2.0, 1.0, False, strategy)

    data = Parameter(Tensor(np.ones([4096]), dtype=ms.int8), "data")
    net.set_inputs(data)

    phase = compile_net(net, data)
    validator = ParallelValidator(net, phase)

    assert validator.check_parameter_shape("data", [1024])


def test_anti_quant_2D():
    """
    Feature: test quant ops
    Description:
    Expectation: compile success
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=4, global_rank=0)

    strategy = ((4, 1),)

    net = AntiQuantNet(2.0, 1.0, False, strategy)

    data = Parameter(Tensor(np.ones([4, 1024]), dtype=ms.int8), "data")
    net.set_inputs(data)

    phase = compile_net(net, data)
    validator = ParallelValidator(net, phase)

    assert validator.check_parameter_shape("data", [1, 1024])

def test_anti_quant_3D():
    """
    Feature: test quant ops
    Description:
    Expectation: compile success
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=4, global_rank=0)

    strategy = ((4, 1, 1),)

    net = AntiQuantNet(2.0, 1.0, False, strategy)

    data = Parameter(Tensor(np.ones([4, 1024, 1024]), dtype=ms.int8), "data")
    net.set_inputs(data)

    phase = compile_net(net, data)
    validator = ParallelValidator(net, phase)

    assert validator.check_parameter_shape("data", [1, 1024, 1024])

def test_anti_quant_4D():
    """
    Feature: test quant ops
    Description:
    Expectation: compile success
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=4, global_rank=0)

    strategy = ((2, 1, 1, 2),)

    net = AntiQuantNet(2.0, 1.0, False, strategy)

    data = Parameter(Tensor(np.ones([4, 128, 128, 128]), dtype=ms.int8), "data")
    net.set_inputs(data)

    phase = compile_net(net, data)
    validator = ParallelValidator(net, phase)

    assert validator.check_parameter_shape("data", [2, 128, 128, 64])
