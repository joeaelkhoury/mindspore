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
"""Context for PackFunc"""
import functools
import types
import textwrap
import inspect
import os
import mindspore as ms
from mindspore import nn
from mindspore import log as logger
from mindspore.common.tensor import Tensor
from mindspore.ops.primitive import _RunOpHook, Primitive
from mindspore._c_expression import PackExpander, PackNode
from mindspore.common._stub_tensor import StubTensor
from mindspore.common._register_for_tensor import tensor_operator_registry
from mindspore.common.api import _handle_func_args, _pynative_executor


class _PackTensor(StubTensor):
    """pack tensor for expander trace"""

    def __setitem__(self, index, value):
        out = tensor_operator_registry.get('__setitem__')(self, index, value)
        self.stub = out.stub
        if self.parent_tensor_ is not None and self.index_of_parent_ is not None:
            self.parent_tensor_.__setitem__(self.index_of_parent_, self)
        return self

    def __pack__(self):
        """for parse check."""

    def stub_sync(self):
        """subclass hook for Tensor"""
        if self.tensor is None:
            val = self.stub.get_value()
            if val is None:
                raise RuntimeError("During the trace operation, the data flow of the Tensor could be tracked, "
                                   "which consequently prevented the creation of a proper trace subgraph.")
            self.tensor = Tensor(val, internal=True)
            if hasattr(self, "member_cache"):
                for k, v in self.member_cache.items():
                    setattr(self.tensor, k, v)
        return self.tensor


def _convert_tensor(node):
    if isinstance(node, PackNode):
        return _PackTensor(node)
    if isinstance(node, tuple):
        return tuple(_convert_tensor(e) for e in node)
    return node

class PackFunc:
    pass

def _trace_cell_call(self, *args, **kwargs):
    """ Run Packed Cell in Pack."""
    if self.__class__.construct is nn.Cell.construct:
        raise AttributeError("For 'Cell', the method 'construct' is not defined.")

    if kwargs:
        bound_arguments = inspect.signature(self.construct).bind(*args, **kwargs)
        bound_arguments.apply_defaults()
        args = bound_arguments.args
        kwargs = bound_arguments.kwargs
    args = self._mixed_precision_cast(args)
    need_subgraph = hasattr(self, "bprop") or hasattr(self, "_pipeline_stage") or self.get_flags()
    if not PackFunc.current.is_pynative_mode and need_subgraph:
        expander = PackExpander.get_instance()
        args = expander.begin_subgraph(self, *args)
        args = [_convert_tensor(a) for a in args]
        output = self._run_construct(args, kwargs)
        ret = expander.end_subgraph(self, output)
        output = _convert_tensor(ret)
    else:
        with _SetMixedPrecision(self):
            output = self._run_construct(args, kwargs)
    return output

class _PrimitiveHook:
    """PrimitiveHook for trace run"""
    def __init__(self):
        self.origin_calls = list()
        for op in ms.ops.operations.__all__:
            prim = getattr(ms.ops.auto_generate, op, None)
            if hasattr(prim, '__call__')  and 'super().__call__' not in inspect.getsource(prim.__call__):
                self.origin_calls.append((prim, prim.__call__))

    def __enter__(self):
        for prim, _ in self.origin_calls:
            prim.__call__ = Primitive.__call__
        return self

    def __exit__(self, *err):
        for prim, origin_call in self.origin_calls:
            prim.__call__ = origin_call

class _PackHook:
    """Hook for trace run"""

    def __init__(self):
        self.origin_call = nn.Cell.__call__

    def __enter__(self):
        nn.Cell.__call__ = _trace_cell_call
        return self

    def __exit__(self, *err):
        nn.Cell.__call__ = self.origin_call

class PackFunc(Primitive):
    """pack function with lazy expander"""

    expander = PackExpander.get_instance()
    current = None

    def __init__(self, fun, unique_key, cell_obj, is_pynative_mode=False):
        super(PackFunc, self).__init__(self.__class__.__name__)
        self.func = fun
        self.cell_obj = cell_obj
        self.kwargs = {}
        self.add_prim_attr("unique_key", unique_key)
        self.add_prim_attr("is_pynative_mode", is_pynative_mode)
        sig = inspect.signature(fun)
        arg_num = len(sig.parameters) - 1
        self.add_prim_attr("arg_num", arg_num)

    def __call__(self, *args, **kwargs):
        if PackFunc.is_tracing():
            if self.cell_obj:
                args = (self.cell_obj, *args)
            return self.func(*args, **kwargs)
        self.kwargs = kwargs
        output = super().__call__(*args)
        if self.is_pynative_mode and self.grad_attach_num > 0:
            output_num = len(output) - self.grad_attach_num
            if output_num == 1:
                return output[0]
            return output[:output_num]
        return output

    def __expand__(self, args):
        old = PackFunc.current
        PackFunc.current = self
        if self.cell_obj:
            args = (self.cell_obj, *args)
            with _SetMixedPrecision(self.cell_obj):
                ret = self._run_op(args)
        else:
            ret = self._run_op(args)
        PackFunc.current = old
        return ret

    @staticmethod
    def is_tracing():
        return PackFunc.current is not None

    @staticmethod
    def _trace_run_op(obj, args):
        ret = PackFunc.expander.emit(obj, *args)
        return _convert_tensor(ret)

    def _run_op(self, args):
        with _PrimitiveHook(), _RunOpHook(PackFunc._trace_run_op), _PackHook():
            fun_args = [_convert_tensor(a) for a in args]
            ret = self.func(*fun_args, **self.kwargs)
        return ret


class _PackSourceBuilder:
    """Generation Pack Python code by method"""

    def __init__(self, original_fn):
        """Initialize the _PackSourceBuilder"""
        self.original_fn = original_fn
        self.is_method = isinstance(self.original_fn, types.MethodType)
        self.pack_fn_name = f"_{original_fn.__name__}_pack"
        self._generate_pack_op()

    def get_code_source(self):
        """Return Pack Python code"""
        if self.is_method:
            sig = inspect.signature(self.original_fn.pack_fn)
            arg_num = len(sig.parameters) - 1
            arg_str = ", ".join(["a{}".format(i) for i in range(arg_num)])
            new_src = textwrap.dedent(f"""
                def {self.pack_fn_name}_wrap(self, {arg_str}):
                    return self.{self.pack_fn_name}({arg_str})
                """)
        else:
            new_src = textwrap.dedent(f"""
                def {self.pack_fn_name}_wrap(*args, **kwargs):
                    return {self.pack_fn_name}(*args, **kwargs)
                """)
        return new_src

    def _generate_pack_op(self):
        """Generate the pack operation and attach it to the original function or method"""
        obj = self.original_fn.__self__ if self.is_method else None
        fn = self.original_fn.pack_fn
        key = f"{id(obj)}_{id(fn)}"
        if self.is_method:
            setattr(obj, self.pack_fn_name, PackFunc(fn, key, obj))
        else:
            fn.__globals__[self.pack_fn_name] = PackFunc(fn, key, obj)


class _SetMixedPrecision:
    """"Set MixedPrecison by the Cell"""

    def __init__(self, cell_obj):
        self.mixed_precision_change = False
        self.cell_obj = cell_obj

    def __enter__(self):
        self.mixed_precision_change = PackFunc.expander.set_mixed_precision(
            self.cell_obj)

    def __exit__(self, *err):
        if self.mixed_precision_change:
            PackFunc.expander.recover_mixed_precision()


def trace(fn):
    """
    Create a traceable function from a python function. The python function will be traced with fake tensor to
    capture the corresponding runtime graph at compilation phase.Tracing is ideal for some scenarios to reduce
    python overhead or graph compilation time.

    Note:
        - Dynamic control flow and other tensor data dependent scenarios are not supported.
        - Dynamic shape operator is not fully supported.

    Args:
        fn (Function) - The Python function to be compiled into a graph. The return value of the function must be
        a Tensor or a Tuple containing Tensors. When a network's construct is traced, only its forward method is traced.

    Returns:
        Function, if `fn` is not None, returns a callable function that will execute the compiled function; If `fn` is
        None, returns a decorator and when this decorator invokes with a single `fn` argument, the callable function is
        equal to the case when `fn` is not None.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> from mindspore import Tensor, nn
        >>> from mindspore.ops._tracefunc import trace
        ...
        >>> class Net(nn.Cell):
        ...     @trace
        ...     def construct(self, x, y):
        ...         return x + y
        >>> x = Tensor([1, 1, 3, 3])
        >>> y = Tensor([1, 1, 3, 3])
        >>> net = Net()
        >>> output = net(x, y)
    """

    @functools.wraps(fn)
    def _trace_wrap(*args, **kwargs):
        pynative_grad_flag = _pynative_executor.grad_flag()
        grad_flag_expr = "1" if pynative_grad_flag else "0"
        if _trace_wrap.is_method is None:
            if args and not isinstance(args[0], Tensor) and hasattr(args[0], fn.__name__):
                _trace_wrap.is_method = False
            else:
                _trace_wrap.is_method = True
        if _trace_wrap.is_method:
            # Similar processing has been done in the __call__ of Cell,
            # so only when obj is None, there is need to do `_handle_func_args`.
            args, kwargs = _handle_func_args(fn, *args, **kwargs)
            pack_func_name = "pack" + grad_flag_expr
            pack_func = getattr(fn, pack_func_name, None)
            if pack_func is None:
                pack_func = PackFunc(fn, f"{id(fn)}_{grad_flag_expr}", None, True)
                setattr(fn, pack_func_name, pack_func)
            return pack_func(*args, **kwargs)
        obj, args = args[0], args[1:]
        pack_func_name = "".join((fn.__name__, "pack", grad_flag_expr))
        pack_func = getattr(obj, pack_func_name, None)
        if pack_func is None:
            pack_func = PackFunc(fn, f"{id(obj)}_{id(fn)}_{grad_flag_expr}", obj, True)
            setattr(obj, pack_func_name, pack_func)
        return pack_func(*args, **kwargs)
    logger.warning("The trace feature is not supported in r2.3")
    if "MS_DEV_ENABLE_TRACE" in os.environ and os.environ["MS_DEV_ENABLE_TRACE"] == "on":
        trace_wrap.pack_fn = fn
        _trace_wrap.is_method = None
        return _trace_wrap
    return fn
