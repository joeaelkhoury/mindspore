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
# ============================================================================
import pytest
import numpy as np
import mindspore as ms
from mindspore import ops
import test_utils


@test_utils.run_with_cell
def sigmoid_forward_func(x):
    return ops.auto_generate.sigmoid(x)


@test_utils.run_with_cell
def sigmoid_backward_func(x):
    return ops.grad(sigmoid_forward_func, (0,))(x)

def sigmoid_dyn_shape_func(x):
    return ops.auto_generate.sigmoid(x)


@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.parametrize('mode', [ms.context.GRAPH_MODE, ms.context.PYNATIVE_MODE])
def test_sigmoid_forward(mode):
    """
    Feature: Ops.
    Description: Test op Sigmoid forward.
    Expectation: Correct result.
    """
    ms.context.set_context(mode=mode)
    expect_out = ms.Tensor([[0.5, 0.7310586], [0.8807971, 0.95257413]], ms.float32)
    x = ms.Tensor([[0, 1], [2, 3]], ms.float32)
    out = sigmoid_forward_func(x)
    assert np.allclose(out.numpy(), expect_out.numpy())


@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.parametrize('mode', [ms.context.GRAPH_MODE, ms.context.PYNATIVE_MODE])
def test_sigmoid_backward(mode):
    """
    Feature: Ops.
    Description: Test op Sigmoid backward.
    Expectation: Correct result.
    """
    ms.context.set_context(mode=mode)
    expect_out = ms.Tensor([[0.25, 0.19661193], [0.10499357, 0.04517666]], ms.float32)
    x = ms.Tensor([[0, 1], [2, 3]], ms.float32)
    grads = sigmoid_backward_func(x)
    assert np.allclose(grads.numpy(), expect_out.numpy())

@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_ascend_training
def test_sigmoid_vmap():
    """
    Feature: test vmap function.
    Description: test sigmoid op vmap.
    Expectation: expect correct result.
    """
    x = ms.Tensor([[0, 1], [2, 3]], ms.float32)
    expect_out = ms.Tensor([[0.5, 0.7310586], [0.8807971, 0.95257413]], ms.float32)
    nest_vmap = ops.vmap(ops.vmap(sigmoid_forward_func, in_axes=0), in_axes=0)
    out = nest_vmap(x)
    assert np.allclose(out.numpy(), expect_out.numpy())

@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.context.PYNATIVE_MODE])
def test_sigmoid_dynamic(mode):
    """
    Feature: test dynamic tensor and dynamic scalar of sigmoid.
    Description: test dynamic tensor and dynamic scalar of sigmoid.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=mode)
    x_dyn1 = ms.Tensor(shape=None, dtype=ms.float32)
    expect_out1 = ms.Tensor([0.5], ms.float32)
    x1 = ms.Tensor([0], ms.float32)
    test_cell = test_utils.to_cell_obj(sigmoid_dyn_shape_func)
    test_cell.set_inputs(x_dyn1)
    out1 = test_cell(x1)
    assert np.allclose(out1.numpy(), expect_out1.numpy())
    x_dyn2 = ms.Tensor(shape=[None, None], dtype=ms.float32)
    expect_out2 = ms.Tensor([[0.5, 0.7310586], [0.8807971, 0.95257413]], ms.float32)
    x2 = ms.Tensor([[0, 1], [2, 3]], ms.float32)
    test_cell.set_inputs(x_dyn2)
    out2 = test_cell(x2)
    assert np.allclose(out2.numpy(), expect_out2.numpy())
