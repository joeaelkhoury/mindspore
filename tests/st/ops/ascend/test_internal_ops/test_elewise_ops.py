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

from mindspore import context, nn, Tensor, ops


class AddNet(nn.Cell):
    def construct(self, x, y):
        return x + y


class SubNet(nn.Cell):
    def construct(self, x, y):
        return x - y


class EqualNet(nn.Cell):
    def construct(self, x, y):
        return x == y


class LessNet(nn.Cell):
    def construct(self, x, y):
        return x < y


class MulNet(nn.Cell):
    def construct(self, x, y):
        return x * y


class RealDivNet(nn.Cell):
    def __init__(self):
        super().__init__()
        self.realDiv = ops.RealDiv()

    def construct(self, x, y):
        return self.realDiv(x, y)


def add_net(x_shape, y_shape, dtype):
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")

    x = np.random.randn(*x_shape).astype(dtype)
    y = np.random.randn(*y_shape).astype(dtype)

    net = AddNet()
    output = net(Tensor(x), Tensor(y))
    expected = x + y

    np.testing.assert_array_almost_equal(output.asnumpy(), expected)


def sub_net(x_shape, y_shape, dtype):
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")

    x = np.random.randn(*x_shape).astype(dtype)
    y = np.random.randn(*y_shape).astype(dtype)

    net = SubNet()
    output = net(Tensor(x), Tensor(y))
    expected = x - y

    np.testing.assert_array_almost_equal(output.asnumpy(), expected)


def equal_net(x_shape, y_shape, dtype):
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")

    x = np.random.randn(*x_shape).astype(dtype)
    y = np.random.randn(*y_shape).astype(dtype)

    net = EqualNet()
    output = net(Tensor(x), Tensor(y))
    expected = x == y

    np.testing.assert_array_almost_equal(output.asnumpy(), expected)


def less_net(x_shape, y_shape, dtype):
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")

    x = np.random.randn(*x_shape).astype(dtype)
    y = np.random.randn(*y_shape).astype(dtype)

    net = LessNet()
    output = net(Tensor(x), Tensor(y))
    expected = x < y

    np.testing.assert_array_almost_equal(output.asnumpy(), expected)


def mul_net(x_shape, y_shape, dtype):
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")

    x = np.random.randn(*x_shape).astype(dtype)
    y = np.random.randn(*y_shape).astype(dtype)

    net = MulNet()
    output = net(Tensor(x), Tensor(y))
    expected = x * y

    np.testing.assert_array_almost_equal(output.asnumpy(), expected)


def realdiv_net(x_shape, y_shape, dtype):
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")

    x = np.random.randn(*x_shape).astype(dtype)
    y = np.random.randn(*y_shape).astype(dtype)

    net = RealDivNet()
    output = net(Tensor(x), Tensor(y))
    expected = x / y

    np.testing.assert_array_almost_equal(output.asnumpy(), expected)


def test_add_broadcast1(dtype=np.float16):
    """
    Feature: test add operator in graph mode
    Description: test add.
    Expectation: the result is correct
    """
    x_shape = (1024, 1664)
    y_shape = (1664,)
    add_net(x_shape, y_shape, dtype)


def test_add_nobroadcast(dtype=np.float16):
    """
    Feature: test add operator in graph mode
    Description: test add.
    Expectation: the result is correct
    """
    x_shape = (1024, 1664)
    y_shape = (1024, 1664)
    add_net(x_shape, y_shape, dtype)


def test_add_broadcast2(dtype=np.float16):
    """
    Feature: test add operator in graph mode
    Description: test add.
    Expectation: the result is correct
    """
    x_shape = (32, 1, 1664)
    y_shape = (1, 1024, 1664)
    add_net(x_shape, y_shape, dtype)


def test_add_bf16(dtype=np.float32):
    """
    Feature: test add operator in graph mode
    Description: test add.
    Expectation: the result is correct
    """
    x_shape = (1024, 1664)
    y_shape = (1, 1024, 1664)
    add_net(x_shape, y_shape, dtype)


def test_sub():
    """
    Feature: test sub operator in graph mode.
    Description: test sub.
    Expectation: the result is correct
    """
    x_shape = (1,)
    y_shape = (1,)
    dtype = np.int64
    sub_net(x_shape, y_shape, dtype)


def test_equal():
    """
    Feature: test equal operator in graph mode.
    Description: test equal.
    Expectation: the result is correct
    """
    x_shape = (1024, 1664)
    y_shape = (1664,)
    dtype = np.float16
    equal_net(x_shape, y_shape, dtype)


def test_less():
    """
    Feature: test less operator in graph mode.
    Description: test less.
    Expectation: the result is correct
    """
    x_shape = (1024, 1664)
    y_shape = (1664,)
    dtype = np.float16
    less_net(x_shape, y_shape, dtype)


def test_mul():
    """
    Feature: test mul operator in graph mode.
    Description: test mul.
    Expectation: the result is correct
    """
    x_shape = (1024, 1664)
    y_shape = (1664,)
    dtype = np.float16
    mul_net(x_shape, y_shape, dtype)


def test_realdiv():
    """
    Feature: test mrealDivul operator in graph mode.
    Description: test realDiv.
    Expectation: the result is correct
    """
    x_shape = (1024, 1664)
    y_shape = (1664,)
    dtype = np.float16
    realdiv_net(x_shape, y_shape, dtype)
