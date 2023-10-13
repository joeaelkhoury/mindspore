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
import os
import sys
import pytest


def run_testcase(file_name, case_name=""):
    log_file = file_name + "_" + case_name + '.log'
    if case_name == "":
        ret = os.system(f'{sys.executable} {file_name}.py &> {log_file}')
    else:
        ret = os.system(f"{sys.executable} -c 'import {file_name};{file_name}.{case_name}()' &> {log_file}")
    os.system(f'grep -E "CRITICAL|ERROR|Error" {log_file} -C 3')
    os.system(f'rm {log_file} -rf')
    assert ret == 0


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
def test_ge_call():
    """
    Description: Test GE Call.
    Description: Support call node.
    Expectation: The result match with expect.
    """
    run_testcase("run_ge_call", "test_ge_call_in_control_flow")
