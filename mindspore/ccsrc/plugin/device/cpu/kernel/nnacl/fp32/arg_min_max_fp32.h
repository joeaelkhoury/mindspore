/**
 * Copyright 2020 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef MINDSPORE_NNACL_FP32_ARG_MIN_MAX_H_
#define MINDSPORE_NNACL_FP32_ARG_MIN_MAX_H_

#include "nnacl/nnacl_common.h"
#include "nnacl/arg_min_max_parameter.h"

#ifdef __cplusplus
extern "C" {
#endif
void ArgMinMaxFp32(const float *input, void *output, float *output_value, const int32_t *in_shape,
                   const ArgMinMaxParameter *param);
void ArgMinMaxInt32(const int32_t *input, void *output, int32_t *output_value, const int32_t *in_shape,
                    const ArgMinMaxParameter *para);
#ifdef __cplusplus
}
#endif

#endif  // MINDSPORE_NNACL_FP32_ARG_MIN_MAX_H_
