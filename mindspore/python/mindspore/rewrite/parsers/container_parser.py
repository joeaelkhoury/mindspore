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
# ============================================================================
"""Parse Container in construct function to node of SymbolTree."""
import ast

from mindspore.rewrite.parser import Parser
from mindspore.rewrite.symbol_tree import SymbolTree
from mindspore.rewrite.parser_register import ParserRegister

from mindspore.rewrite.parser_register import reg_parser


class ListParser(Parser):
    """Parse list in construct function to node of SymbolTree."""
    def target(self):
        """Parse target type."""
        return list

    def process(self, stree: SymbolTree, node: list):
        """
        Parse list.

        Args:
            stree ([SymbolTree]): Symbol Tree under parsing.
            node ([list]): An list of node.

        Returns:
            A list of value.

        Raises:
            TypeError:Str parser only supports parsing list type nodes.
        """
        if not isinstance(node, ast.Str):
            raise TypeError("Str parser only supports parsing list type nodes.")
        result = []
        for n in node:
            parser = ParserRegister.instance().get_parser(type(n))
            value = parser.process(stree, n)
            result.append(value)
        return result


class TupleParser(Parser):
    """Parse tuple in construct function to node of SymbolTree."""
    def target(self):
        """Parse target type."""
        return tuple

    def process(self, stree: SymbolTree, node: tuple):
        """
        Parse tuple.

        Args:
            stree ([SymbolTree]): Symbol Tree under parsing.
            node ([tuple]): An tuple of node.

        Returns:
            A tuple of value.

        Raises:
            TypeError:Tuple parser only supports parsing Tuple type nodes.
        """
        result = []
        for n in node:
            parser = ParserRegister.instance().get_parser(type(n))
            value = parser.process(stree, n)
            result.append(value)
        return tuple(result)

g_list_parser = reg_parser(ListParser())
g_tuple_parser = reg_parser(TupleParser())
