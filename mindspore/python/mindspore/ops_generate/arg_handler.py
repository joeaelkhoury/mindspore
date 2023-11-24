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
"""Operator argument handle function."""

from mindspore.ops_generate.gen_ops_inner_prim import DtypeToEnum

ops_dtype_to_enum = DtypeToEnum()


def to_kernel_size(kernel_size):
    """
    convert kernel_size: int/tuple[int*4] -> tuple[int*2].
    """
    if isinstance(kernel_size, int):
        return (kernel_size, kernel_size)
    if isinstance(kernel_size, (tuple, list)):
        if len(kernel_size) == 4:
            return (kernel_size[2], kernel_size[3])
        return kernel_size
    raise ValueError(f"For arg 'kernel_size', the value is invalid: {kernel_size}.")


def to_strides(stride):
    """
    convert strides: int/tuple[int*4] -> tuple[int*2].
    """
    if isinstance(stride, int):
        return (stride, stride)
    if isinstance(stride, (tuple, list)):
        if len(stride) == 4:
            return (stride[2], stride[3])
        return stride
    raise ValueError(f"For arg 'stride', the value is invalid: {stride}.")


def to_rates(rates):
    """
    convert rates: int/tuple[int*4] -> tuple[int*2].
    """
    if isinstance(rates, int):
        return (rates, rates)
    if isinstance(rates, (tuple, list)):
        if len(rates) == 4:
            return (rates[2], rates[3])
        return rates
    raise ValueError(f"For arg 'rate', the value is invalid: {rates}.")


def to_dilations(dilation):
    """
    convert dilations: int/tuple[int*4] -> tuple[int*2].
    """
    if isinstance(dilation, int):
        return (dilation, dilation)
    if isinstance(dilation, (tuple, list)):
        if len(dilation) == 4:
            return (dilation[2], dilation[3])
        return dilation
    raise ValueError(f"For arg 'dilation', the value is invalid: {dilation}.")


def to_paddings(pad):
    """
    convert paddings: int -> tuple[int*4].
    """
    if isinstance(pad, int):
        return (pad,) * 4
    if isinstance(pad, (tuple, list)):
        return pad
    raise ValueError(f"For arg 'pad', the value is invalid: {pad}.")


def to_3d_kernel_size(kernel_size):
    """
    convert 3d kernel_size: int/tuple[int*6] -> tuple[int*3].
    """
    if isinstance(kernel_size, int):
        return (kernel_size, kernel_size, kernel_size)
    if isinstance(kernel_size, (tuple, list)):
        if len(kernel_size) == 5:
            return (kernel_size[2], kernel_size[3], kernel_size[4])
        return kernel_size
    raise ValueError(f"For arg 'kernel_size', the value is invalid: {kernel_size}.")


def to_3d_strides(stride):
    """
    convert 3d stride: int/tuple[int*6] -> tuple[int*3].
    """
    if isinstance(stride, int):
        return (stride, stride, stride)
    if isinstance(stride, (tuple, list)):
        if len(stride) == 5:
            return (stride[2], stride[3], stride[4])
        return stride
    raise ValueError(f"For arg 'stride', the value is invalid: {stride}.")


def to_3d_dilations(dilation):
    """
    convert 3d dilation: int/tuple[int*6] -> tuple[int*3].
    """
    if isinstance(dilation, int):
        return (dilation, dilation, dilation)
    if isinstance(dilation, (tuple, list)):
        if len(dilation) == 5:
            return (dilation[2], dilation[3], dilation[4])
        return dilation
    raise ValueError(f"For arg 'dilation', the value is invalid: {dilation}.")


def to_3d_paddings(pad):
    """
    convert 3d paddings: int -> tuple[int*6].
    """
    if isinstance(pad, int):
        return (pad,) * 6
    if isinstance(pad, (tuple, list)):
        return pad
    raise ValueError(f"For arg 'pad', the value is invalid: {pad}.")


def dtype_to_enum(dtype):
    """
    convert mindspore.dtype to enum.
    """
    return ops_dtype_to_enum(dtype)
