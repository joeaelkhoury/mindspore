# Copyright 2021 Huawei Technologies Co., Ltd
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
import mindspore.context as context
import mindspore.nn as nn
import mindspore.common.dtype as mstype
from mindspore import Tensor

recompute_prefix = 'recompute_'


class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def construct(self, input_x):
        output = self.pool(input_x)
        return output


def test_set_recompute_true():
    context.set_context(mode=context.GRAPH_MODE)
    net = Net()
    net.pool.recompute()
    assert net.pool.get_scope() == recompute_prefix


def test_set_recompute_true_with_mp_comm_recompute():
    context.set_context(mode=context.GRAPH_MODE)
    net = Net()
    net.pool.recompute(mp_comm_recompute=True)
    assert net.pool.get_scope() == recompute_prefix


def test_set_recompute_true_with_mp_comm_recompute_false():
    context.set_context(mode=context.GRAPH_MODE)
    net = Net()
    net.pool.recompute(mp_comm_recompute=False)
    assert net.pool.get_scope() == recompute_prefix


def test_set_recompute_true_twice():
    context.set_context(mode=context.GRAPH_MODE)
    net = Net()
    net.pool.recompute()
    with pytest.raises(RuntimeError):
        net.pool.recompute()


def test_set_recompute_in_pynative_mode():
    """
    Feature: Recomputation.
    Description: Call recompute api of Cell in PyNative mode.
    Expectation: Raise TypeError when call the cell.
    """
    context.set_context(mode=context.PYNATIVE_MODE)
    net = Net()
    try:
        net.pool.recompute()
        x = Tensor([[0.5, 0.6, 0.4], [1.2, 1.3, 1.1]], dtype=mstype.float32)
        net(x)
    except TypeError as e:
        assert "Recompute is not supported in PyNative mode currently" in str(e)
