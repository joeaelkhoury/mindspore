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
import numpy as np
import pytest

import mindspore as ms
from mindspore import Tensor
from mindspore import nn


class Net(nn.Cell):
    def construct(self, x, num_samples, replacement=True, seed=None):
        return x.multinomial(num_samples, replacement, seed)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_arm_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_tensor_multinomial(mode):
    """
    Feature: tensor.multinomial
    Description: Verify the result of multinomial
    Expectation: success
    """
    ms.set_context(mode=mode)
    x = Tensor(np.array([[0.9, 0.2], [0.9, 0.2]]).astype(np.float32))
    net = Net()
    output = net(x, 6, True, 2)
    print(output)
    assert output.asnumpy().shape == (2, 6)
