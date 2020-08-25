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

#include "nnacl/fp32/matmul.h"

void RowMajor2Row8Major(float *src_ptr, float *dst_ptr, int row, int col) {
  for (int r = 0; r < row; r++) {
    float *src = src_ptr + r * col;
    for (int c = 0; c < col; c++) {
      int cd8 = c / 8;
      int cm8 = c % 8;
      dst_ptr[cd8 * 8 * row + r * 8 + cm8] = src[c];
    }
  }
  return;
}

void RowMajor2Row12Major(float *src_ptr, float *dst_ptr, int row, int col) {
  for (int r = 0; r < row; r++) {
    float *src = src_ptr + r * col;
    for (int c = 0; c < col; c++) {
      int cd8 = c / C12NUM;
      int cm8 = c % C12NUM;
      dst_ptr[cd8 * C12NUM * row + r * C12NUM + cm8] = src[c];
    }
  }
  return;
}

void RowMajor2Col12Major(float *src_ptr, float *dst_ptr, size_t row, size_t col) {
  size_t row12 = row / C12NUM * C12NUM;
  size_t col4 = col / C4NUM * C4NUM;
  float *src_r = src_ptr;
  float *dst_r = dst_ptr;

  size_t ri = 0;
  for (; ri < row12; ri += C12NUM) {
    size_t ci = 0;
    for (; ci < col4; ci += C4NUM) {
      float *src_c = src_r + ci;
      float *dst_c = dst_r + ci * C12NUM;

      /* 12x4 row-major to col-major */
#ifdef ENABLE_ARM64
      size_t stride = col * sizeof(float);
      asm volatile(
        "mov x10, %[src_c]\n"
        "mov x11, %[dst_c]\n"

        "ld1 {v0.4s}, [x10], %[stride]\n"
        "ld1 {v1.4s}, [x10], %[stride]\n"
        "ld1 {v2.4s}, [x10], %[stride]\n"
        "ld1 {v3.4s}, [x10], %[stride]\n"

        "ld1 {v4.4s}, [x10], %[stride]\n"
        "ld1 {v5.4s}, [x10], %[stride]\n"
        "ld1 {v6.4s}, [x10], %[stride]\n"
        "ld1 {v7.4s}, [x10], %[stride]\n"

        "zip1 v12.4s, v0.4s, v1.4s\n"
        "zip2 v13.4s, v0.4s, v1.4s\n"
        "zip1 v14.4s, v2.4s, v3.4s\n"
        "zip2 v15.4s, v2.4s, v3.4s\n"

        "ld1 {v8.4s}, [x10], %[stride]\n"
        "ld1 {v9.4s}, [x10], %[stride]\n"
        "ld1 {v10.4s}, [x10], %[stride]\n"
        "ld1 {v11.4s}, [x10], %[stride]\n"

        "zip1 v16.4s, v4.4s, v5.4s\n"
        "zip2 v17.4s, v4.4s, v5.4s\n"
        "zip1 v18.4s, v6.4s, v7.4s\n"
        "zip2 v19.4s, v6.4s, v7.4s\n"

        "trn1 v20.2d, v12.2d, v14.2d\n"
        "trn2 v23.2d, v12.2d, v14.2d\n"
        "trn1 v26.2d, v13.2d, v15.2d\n"
        "trn2 v29.2d, v13.2d, v15.2d\n"

        "trn1 v21.2d, v16.2d, v18.2d\n"
        "trn2 v24.2d, v16.2d, v18.2d\n"
        "trn1 v27.2d, v17.2d, v19.2d\n"
        "trn2 v30.2d, v17.2d, v19.2d\n"

        "zip1 v12.4s, v8.4s, v9.4s\n"
        "zip2 v13.4s, v8.4s, v9.4s\n"
        "zip1 v14.4s, v10.4s, v11.4s\n"
        "zip2 v15.4s, v10.4s, v11.4s\n"

        "trn1 v22.2d, v12.2d, v14.2d\n"
        "trn2 v25.2d, v12.2d, v14.2d\n"
        "trn1 v28.2d, v13.2d, v15.2d\n"
        "trn2 v31.2d, v13.2d, v15.2d\n"

        "st1 {v20.4s, v21.4s, v22.4s, v23.4s}, [x11], #64\n"
        "st1 {v24.4s, v25.4s, v26.4s, v27.4s}, [x11], #64\n"
        "st1 {v28.4s, v29.4s, v30.4s, v31.4s}, [x11], #64\n"

        :
        : [ dst_c ] "r"(dst_c), [ src_c ] "r"(src_c), [ stride ] "r"(stride)
        : "x10", "x11", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14",
          "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29",
          "v30", "v31");
#else
      for (int tr = 0; tr < C12NUM; tr++) {
        for (int tc = 0; tc < C4NUM; tc++) {
          dst_c[tc * C12NUM + tr] = src_c[tr * col + tc];
        }
      }
#endif
    }
    for (; ci < col; ci++) {
      float *src_c = src_r + ci;
      float *dst_c = dst_r + ci * C12NUM;
      for (size_t i = 0; i < C12NUM; i++) {
        dst_c[i] = src_c[i * col];
      }
    }
    src_r += C12NUM * col;
    dst_r += C12NUM * col;
  }
  for (; ri < row; ri++) {
    for (size_t i = 0; i < col; i++) {
      dst_r[i * C12NUM] = src_r[i];
    }
    src_r += col;
    dst_r += 1;
  }
  return;
}

void RowMajor2Col8Major(float *src_ptr, float *dst_ptr, size_t row, size_t col) {
  size_t row8 = row / C8NUM * C8NUM;
  size_t col4 = col / C4NUM * C4NUM;
  float *src_r = src_ptr;
  float *dst_r = dst_ptr;

  size_t ri = 0;
  for (; ri < row8; ri += C8NUM) {
    size_t ci = 0;
    for (; ci < col4; ci += C4NUM) {
      float *src_c = src_r + ci;
      float *dst_c = dst_r + ci * C8NUM;

      /* 8x4 row-major to col-major */
#ifdef ENABLE_ARM64
      size_t stride = col * 4;
      asm volatile(
        "mov x10, %[src_c]\n"
        "mov x11, %[dst_c]\n"

        "ld1 {v0.4s}, [x10], %[stride]\n"
        "ld1 {v1.4s}, [x10], %[stride]\n"
        "ld1 {v2.4s}, [x10], %[stride]\n"
        "ld1 {v3.4s}, [x10], %[stride]\n"

        "zip1 v4.4s, v0.4s, v1.4s\n"
        "zip2 v5.4s, v0.4s, v1.4s\n"
        "zip1 v6.4s, v2.4s, v3.4s\n"
        "zip2 v7.4s, v2.4s, v3.4s\n"

        "ld1 {v8.4s},  [x10], %[stride]\n"
        "ld1 {v9.4s},  [x10], %[stride]\n"
        "ld1 {v10.4s}, [x10],  %[stride]\n"
        "ld1 {v11.4s}, [x10],  %[stride]\n"

        "trn1 v0.2d, v4.2d, v6.2d\n"
        "trn2 v1.2d, v4.2d, v6.2d\n"
        "trn1 v2.2d, v5.2d, v7.2d\n"
        "trn2 v3.2d, v5.2d, v7.2d\n"

        "zip1 v12.4s, v8.4s, v9.4s\n"
        "zip2 v13.4s, v8.4s, v9.4s\n"
        "zip1 v14.4s, v10.4s, v11.4s\n"
        "zip2 v15.4s, v10.4s, v11.4s\n"

        "trn1 v8.2d, v12.2d, v14.2d\n"
        "trn2 v9.2d, v12.2d, v14.2d\n"
        "trn1 v10.2d, v13.2d, v15.2d\n"
        "trn2 v11.2d, v13.2d, v15.2d\n"

        "st1 {v0.4s}, [x11],  #16\n"
        "st1 {v8.4s}, [x11],  #16\n"
        "st1 {v1.4s}, [x11],  #16\n"
        "st1 {v9.4s}, [x11],  #16\n"
        "st1 {v2.4s},  [x11],#16\n"
        "st1 {v10.4s}, [x11], #16\n"
        "st1 {v3.4s},  [x11],#16\n"
        "st1 {v11.4s}, [x11], #16\n"

        :
        : [ dst_c ] "r"(dst_c), [ src_c ] "r"(src_c), [ stride ] "r"(stride)
        : "x10", "x11", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14",
          "v15");
#else
      for (int tr = 0; tr < 8; tr++) {
        for (int tc = 0; tc < 4; tc++) {
          dst_c[tc * 8 + tr] = src_c[tr * col + tc];
        }
      }
#endif
    }
    for (; ci < col; ci++) {
      float *src_c = src_r + ci;
      float *dst_c = dst_r + ci * C8NUM;
      for (size_t i = 0; i < C8NUM; i++) {
        dst_c[i] = src_c[i * col];
      }
    }
    src_r += C8NUM * col;
    dst_r += C8NUM * col;
  }
  for (; ri < row; ri++) {
    for (size_t i = 0; i < col; i++) {
      dst_r[i * C8NUM] = src_r[i];
    }
    src_r += col;
    dst_r += 1;
  }
  return;
}

void MatrixUnPackUnit(const void *src, void *dst, size_t row, size_t col, size_t src_stride, size_t dst_stride,
                      size_t data_lenth) {
  size_t copy_size = col * data_lenth;
  size_t src_size = src_stride * data_lenth;
  size_t dst_size = dst_stride * data_lenth;
  char *src_ptr = (char *)src;
  char *dst_ptr = (char *)dst;
  for (int r = 0; r < row; r++) {
    memcpy(dst_ptr, src_ptr, copy_size);
    src_ptr += src_size;
    dst_ptr += dst_size;
  }
}

void Row8x8Major2RowMajor(float *src_ptr, float *dst_ptr, size_t row, size_t col, size_t stride) {
  size_t row_up8 = UP_ROUND(row, C8NUM);
  size_t row_8div = row / C8NUM * C8NUM;
  size_t row_8res = row - row_8div;
  size_t col_8div = col / C8NUM * C8NUM;
  size_t col_8res = col - col_8div;
  float *src_c = src_ptr;
  float *dst_c = dst_ptr;

  for (size_t ci = 0; ci < col_8div; ci += C8NUM) {
#ifdef ENABLE_ARM64
    size_t offset = stride * 4 - 16;
    asm volatile(
      "mov x0, #0 \n"
      "mov x1, %[row_8div] \n"
      "mov x10, %[src_c] \n"
      "mov x11, %[dst_c] \n"

      "1: \n"
      "cmp x0, x1 \n"
      "beq 2f \n"

      "ld1 {v0.4s}, [x10], #16\n"
      "ld1 {v1.4s}, [x10], #16\n"
      "ld1 {v2.4s}, [x10], #16\n"
      "ld1 {v3.4s}, [x10], #16\n"
      "ld1 {v4.4s}, [x10], #16\n"
      "ld1 {v5.4s}, [x10], #16\n"
      "ld1 {v6.4s}, [x10], #16\n"
      "ld1 {v7.4s}, [x10], #16\n"
      "ld1 {v8.4s}, [x10], #16\n"
      "ld1 {v9.4s}, [x10], #16\n"
      "ld1 {v10.4s}, [x10], #16\n"
      "ld1 {v11.4s}, [x10], #16\n"
      "ld1 {v12.4s}, [x10], #16\n"
      "ld1 {v13.4s}, [x10], #16\n"
      "ld1 {v14.4s}, [x10], #16\n"
      "ld1 {v15.4s}, [x10], #16\n"

      "add x0, x0, #8\n"

      "st1 {v0.4s}, [x11], #16\n"
      "st1 {v1.4s}, [x11], %[offset]\n"
      "st1 {v2.4s}, [x11], #16\n"
      "st1 {v3.4s}, [x11], %[offset]\n"
      "st1 {v4.4s}, [x11], #16\n"
      "st1 {v5.4s}, [x11], %[offset]\n"
      "st1 {v6.4s}, [x11], #16\n"
      "st1 {v7.4s}, [x11], %[offset]\n"
      "st1 {v8.4s}, [x11], #16\n"
      "st1 {v9.4s}, [x11], %[offset]\n"
      "st1 {v10.4s}, [x11], #16\n"
      "st1 {v11.4s}, [x11], %[offset]\n"
      "st1 {v12.4s}, [x11], #16\n"
      "st1 {v13.4s}, [x11], %[offset]\n"
      "st1 {v14.4s}, [x11], #16\n"
      "st1 {v15.4s}, [x11], %[offset]\n"
      "b 1b\n"

      "2:\n"

      :
      : [ dst_c ] "r"(dst_c), [ src_c ] "r"(src_c), [ offset ] "r"(offset), [ row_8div ] "r"(row_8div)
      : "x0", "x1", "x10", "x11", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12",
        "v13", "v14", "v15");
#else
    for (size_t ri = 0; ri < row_8div; ri += C8NUM) {
      float *src_r = src_c + ri * C8NUM;
      float *dst_r = dst_c + ri * stride;
      MatrixUnPackUnit(src_r, dst_r, C8NUM, C8NUM, C8NUM, stride, sizeof(float));
    }
#endif

    if (row != row_8div) {
      float *src_r = src_c + row_8div * C8NUM;
      float *dst_r = dst_c + row_8div * stride;
      MatrixUnPackUnit(src_r, dst_r, row_8res, C8NUM, C8NUM, stride, sizeof(float));
    }
    src_c += row_up8 * C8NUM;
    dst_c += C8NUM;
  }

  if (col != col_8div) {
    MatrixUnPackUnit(src_c, dst_c, row, col_8res, C8NUM, stride, sizeof(float));
  }
  return;
}

void MatMul12x8(const float *a, const float *b, float *dst, const float *bias, ActType act_type, int deep, int row,
                int col, int stride, int out_type) {
  if (out_type == OutType_Nhwc) {
    /*  col8-major * row8-major => col-major  */
    for (int r = 0; r < row; r++) {
      for (int c = 0; c < col; c++) {
        int r12div = r / 12, r12mod = r % 12;
        int c8div = c / 8, c8mod = c % 8;
        size_t ci = r * stride + c;
        float value = 0;
        for (int d = 0; d < deep; d++) {
          size_t ai = r12div * deep * 12 + d * 12 + r12mod;
          size_t bi = c8div * deep * 8 + d * 8 + c8mod;
          value = value + a[ai] * b[bi];
        }
        if (bias != NULL) value += bias[c];
        if (act_type == ActType_Relu6) value = MSMIN(6.0f, value);
        if (act_type != ActType_No) value = MSMAX(0.0f, value);
        dst[ci] = value;
      }
    }
  } else if (out_type == OutType_C8) {
    /*  col8-major * row8-major => col12x8-major  */
    int col_8 = UP_ROUND(col, C8NUM);
    int row_12 = UP_ROUND(row, C12NUM);
    for (int r = 0; r < row_12; r++) {
      for (int c = 0; c < col_8; c++) {
        int r12div = r / C12NUM, r12mod = r % C12NUM;
        int c8div = c / C8NUM, c8mod = c % C8NUM;
        size_t ci = (c8div * C8NUM * row_12 + r * C8NUM + c8mod);
        float value = 0;
        for (int d = 0; d < deep; d++) {
          size_t ai = r12div * deep * C12NUM + d * C12NUM + r12mod;
          size_t bi = c8div * deep * C8NUM + d * C8NUM + c8mod;
          value = value + a[ai] * b[bi];
        }
        if (bias != NULL) value += bias[c];
        if (act_type == ActType_Relu6) value = MSMIN(6.0f, value);
        if (act_type != ActType_No) value = MSMAX(0.0f, value);
        dst[ci] = value;
      }
    }
  } else {
    for (int i = 0; i < row; ++i) {
      int src_r_offset = i;
      int dst_r_offset = i * col * stride;
      for (int j = 0; j < col; ++j) {
        int c8div = j / 8, c8mod = j % 8;
        size_t ci = dst_r_offset + c8div * 8 * stride + c8mod;
        float value = 0;
        for (int d = 0; d < deep; ++d) {
          size_t ai = src_r_offset + d * C12NUM;
          size_t bi = c8div * deep * 8 + d * 8 + c8mod;
          value = value + a[ai] * b[bi];
        }
        if (bias != NULL) value += bias[j];
        if (act_type == ActType_Relu6) value = MSMIN(6.0f, value);
        if (act_type != ActType_No) value = MSMAX(0.0f, value);
        dst[ci] = value;
      }
    }
  }
  return;
}

void MatMulOpt(const float *a, const float *b, float *c, const float *bias, ActType act_type, int deep, int row,
               int col, size_t stride, int out_type) {
#ifdef ENABLE_ARM64
  if (out_type == 2 && row <= 8) {
    MatmulFloatNeon64OptRemain(a, b, c, deep, row, col, stride);
  } else {
    MatmulFloatNeon64Opt(a, b, c, bias, (int)act_type, deep, row, col, stride, (int)(out_type == OutType_Nhwc),
                         (int)(out_type == OutType_TileC8));
  }
#else
  MatMul12x8(a, b, c, bias, act_type, deep, row, col, stride, out_type);
#endif
}
