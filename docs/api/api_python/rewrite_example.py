# Copyright 2022 Huawei Technologies Co., Ltd
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
# ==============================================================================
"""
This example mainly illustrates the usage of rewrite.
"""
from typing import OrderedDict
import numpy as np

from mindspore.common import dtype as mstype
from mindspore import Tensor, export
from mindspore.rewrite import SymbolTree, ScopedValue, Node, NodeType, Replacement, PatternEngine, PatternNode
import mindspore.nn as nn
import mindspore.ops as ops


class SubNet(nn.Cell):
    """子网络定义"""

    def __init__(self):
        super().__init__()
        self.dense = nn.Dense(in_channels=32, out_channels=32, weight_init="ones")
        self.mean = ops.ReduceMean(keep_dims=False)
        self.conv1 = nn.Conv2d(1, 1, 1, stride=1)

    def construct(self, x):
        x = self.conv1(x)
        x = self.dense(x)
        return x


class Net(nn.Cell):
    """网络定义"""

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 1, 1, pad_mode='valid')
        self.conv2 = nn.Conv2d(1, 1, 1, pad_mode='valid')
        self.relu = nn.ReLU()
        self.max_pool2d = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        self.simnet = SubNet()

    def construct(self, x):
        """网络的前向计算过程"""
        x = self.conv1(x)
        x = self.relu(x)
        x = self.max_pool2d(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.max_pool2d(x)
        x = self.simnet(x)
        x = self.flatten(x)
        return x


def create_stree(network):
    """创建SymbolTree"""
    stree = SymbolTree.create(network)
    return stree


def insert_node(stree):
    """在网络中插入节点"""
    for node in stree.nodes():
        if node.get_name() == "conv2":  # 在名称为'conv2'的节点前面插入新的节点
            position = stree.before(node)
            new_conv = nn.Conv2d(1, 1, 1)
            new_conv_node = Node.create_call_cell(new_conv, targets=['output'], name='new_conv',
                                                  args=node.get_args())
            stree.insert(position, new_conv_node)
            break
    # 使用新节点更新已有节点的输入
    if new_conv_node is not None:
        for node in stree.nodes():
            if node.get_name() == "relu_1":
                node.set_arg_by_node(0, new_conv_node)
                break


def insert_node_to_subtree(stree):
    """在子网络中插入节点"""

    def _insert_conv(stree: SymbolTree):
        for node in stree.nodes():
            if node.get_instance_type() == nn.Conv2d:
                position = stree.after(node)
                new_conv = nn.Conv2d(1, 1, 1)
                new_conv_node = Node.create_call_cell(new_conv, targets=['x'], name='new_conv',
                                                      args=[ScopedValue.create_naming_value('x')])
                stree.insert(position, new_conv_node)
                break

    # 在名称为'simnet'的子网络中插入新节点
    for node in stree.nodes():
        if node.get_node_type() == NodeType.Tree and node.get_name() == "simnet":
            _insert_conv(node.get_sub_tree())
            break


def erase_node(stree):
    """删除类型为nn.Flatten的节点"""
    for node in stree.nodes():
        if node.get_instance_type() == nn.Flatten:
            stree.erase(node)
            break


def replace_node(stree):
    """替换网络中的节点"""
    new_conv = nn.Conv2d(1, 1, 1)
    new_conv_node = Node.create_call_cell(new_conv, targets=[ScopedValue.create_naming_value("x")],
                                          args=[ScopedValue.create_naming_value('x')], name="replace_conv")
    for node in stree.nodes():
        if node.get_name() == "conv1":
            new_conv_node = stree.replace(node, [new_conv_node])


def pattern_replace(stree):
    """通过模式匹配的方式替换节点"""

    class ConvReplacement(Replacement):
        """创建新节点类的实现"""

        def build(self, pattern: PatternNode, is_chain_pattern: bool, matched: OrderedDict) -> [Node]:
            bn_node: Node = matched.get(pattern.name())
            conv = nn.Conv2d(1, 1, 1)
            conv_node = Node.create_call_cell(conv, ['x'], bn_node.get_args(), bn_node.get_kwargs(),
                                              name="pattern_conv")
            return [conv_node]

    class BnReplace(PatternEngine):
        """替换网络中nn.MaxPool2d类型的节点"""

        def __init__(self):
            super().__init__([nn.MaxPool2d], ConvReplacement())

    bn_replace = BnReplace()
    bn_replace.apply(stree)


def get_net(stree):
    """获取修改后的网络"""
    return stree.get_network()


def get_code(stree):
    """获取修改后的网络代码"""
    return stree.get_code()


def print_symbol_tree_info(stree):
    """打印SymbolTree里的信息"""
    stree.print_node_tabulate()
    for node in stree.nodes():
        if node.get_node_type() == NodeType.Tree:
            print(f"subtree:{node.get_name()}")
            subtree = node.get_sub_tree()
            subtree.print_node_tabulate()
            break


def test_rewrite():
    """ReWrite测试函数"""
    net = Net()
    stree = create_stree(net)

    print("[origin code]")
    print_symbol_tree_info(stree)

    print("[insert code: new_conv]")
    insert_node(stree)
    print_symbol_tree_info(stree)

    print("[insert code to subtree code: new_conv]")
    insert_node_to_subtree(stree)
    print_symbol_tree_info(stree)

    print("[erase code: flatten]")
    erase_node(stree)
    print_symbol_tree_info(stree)

    print("[replace code: conv1 -> replace_conv]")
    replace_node(stree)
    print_symbol_tree_info(stree)

    print("[pattern replace code: max_pool2d -> pattern_conv]")
    pattern_replace(stree)
    print_symbol_tree_info(stree)

    inputs = Tensor(np.ones([1, 1, 32, 32]), mstype.float32) # pylint: disable=not-callable
    new_net = get_net(stree)
    source_code = get_code(stree)

    print("[rewritten code]")
    print(source_code)

    print("[output]")
    output = new_net(inputs)
    print(output)

    export(new_net, inputs, file_name="new_net", file_format="MINDIR")


if __name__ == "__main__":
    test_rewrite()
