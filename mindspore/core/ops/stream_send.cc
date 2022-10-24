/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#include "ops/stream_send.h"

#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "mindapi/src/helper.h"
#include "include/common/utils/utils.h"

namespace mindspore {
namespace ops {
MIND_API_OPERATOR_IMPL(StreamSend, BaseOperator);
REGISTER_PRIMITIVE_C(kNameStreamSend, StreamSend);
}  // namespace ops
}  // namespace mindspore
