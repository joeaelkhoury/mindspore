/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_AUDIO_KERNELS_AUDIO_UTILS_H_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_AUDIO_KERNELS_AUDIO_UTILS_H_

#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "minddata/dataset/core/tensor.h"
#include "minddata/dataset/kernels/tensor_op.h"
#include "minddata/dataset/util/status.h"

constexpr double PI = 3.141592653589793;

namespace mindspore {
namespace dataset {

/// \brief Turn a tensor from the power/amplitude scale to the decibel scale.
/// \param input/output: Tensor of shape <..., freq, time>.
/// \param multiplier: power - 10, amplitude - 20.
/// \param amin: lower bound.
/// \param db_multiplier: multiplier for decibels.
/// \param top_db: the lower bound for decibels cut-off.
/// \return Status code.
template <typename T>
Status AmplitudeToDB(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output, T multiplier, T amin,
                     T db_multiplier, T top_db) {
  TensorShape input_shape = input->shape();
  TensorShape to_shape = input_shape.Rank() == 2
                           ? TensorShape({1, 1, input_shape[-2], input_shape[-1]})
                           : TensorShape({input->Size() / (input_shape[-3] * input_shape[-2] * input_shape[-1]),
                                          input_shape[-3], input_shape[-2], input_shape[-1]});
  RETURN_IF_NOT_OK(input->Reshape(to_shape));

  std::vector<T> max_val;
  int step = to_shape[-3] * input_shape[-2] * input_shape[-1];
  int cnt = 0;
  T temp_max = std::numeric_limits<T>::lowest();
  for (auto itr = input->begin<T>(); itr != input->end<T>(); itr++) {
    // do clamp
    *itr = *itr < amin ? log10(amin) * multiplier : log10(*itr) * multiplier;
    *itr -= multiplier * db_multiplier;
    // calculate max by axis
    cnt++;
    if ((*itr) > temp_max) temp_max = *itr;
    if (cnt % step == 0) {
      max_val.push_back(temp_max);
      temp_max = std::numeric_limits<T>::lowest();
    }
  }

  if (!std::isnan(top_db)) {
    int ind = 0;
    for (auto itr = input->begin<T>(); itr != input->end<T>(); itr++, ind++) {
      float lower_bound = max_val[ind / step] - top_db;
      *itr = std::max((*itr), static_cast<T>(lower_bound));
    }
  }
  RETURN_IF_NOT_OK(input->Reshape(input_shape));
  *output = input;
  return Status::OK();
}

/// \brief Calculate the angles of the complex numbers.
/// \param input/output: Tensor of shape <..., time>.
template <typename T>
Status Angle(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) {
  TensorShape shape = input->shape();
  std::vector output_shape = shape.AsVector();
  output_shape.pop_back();
  std::shared_ptr<Tensor> output_tensor;
  std::vector<T> out;
  T o;
  T x;
  T y;
  for (auto itr = input->begin<T>(); itr != input->end<T>(); itr++) {
    x = static_cast<T>(*itr);
    itr++;
    y = static_cast<T>(*itr);
    o = std::atan2(y, x);
    out.emplace_back(o);
  }
  // Generate multidimensional results corresponding to input
  Tensor::CreateFromVector(out, TensorShape{output_shape}, &output_tensor);
  *output = output_tensor;
  return Status::OK();
}

/// \brief Perform a biquad filter of input tensor.
/// \param input/output: Tensor of shape <..., time>.
/// \param a0: denominator coefficient of current output y[n], typically 1.
/// \param a1: denominator coefficient of current output y[n-1].
/// \param a2: denominator coefficient of current output y[n-2].
/// \param b0: numerator coefficient of current input, x[n].
/// \param b1: numerator coefficient of input one time step ago x[n-1].
/// \param b2: numerator coefficient of input two time steps ago x[n-2].
/// \return Status code.
template <typename T>
Status Biquad(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output, T b0, T b1, T b2, T a0, T a1,
              T a2) {
  std::vector<T> a_coeffs;
  std::vector<T> b_coeffs;
  a_coeffs.push_back(a0);
  a_coeffs.push_back(a1);
  a_coeffs.push_back(a2);
  b_coeffs.push_back(b0);
  b_coeffs.push_back(b1);
  b_coeffs.push_back(b2);
  return LFilter(input, output, a_coeffs, b_coeffs, true);
}

/// \brief Apply contrast effect.
/// \param input/output: Tensor of shape <..., time>.
/// \param enhancement_amount: controls the amount of the enhancement.
/// \return Status code.
template <typename T>
Status Contrast(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output, T enhancement_amount) {
  const float enhancement_zoom = 750.0;
  T enhancement_amount_value = enhancement_amount / enhancement_zoom;
  TensorShape output_shape{input->shape()};
  std::shared_ptr<Tensor> out;
  RETURN_IF_NOT_OK(Tensor::CreateEmpty(output_shape, input->type(), &out));
  auto itr_out = out->begin<T>();
  for (auto itr_in = input->begin<T>(); itr_in != input->end<T>(); itr_in++) {
    T temp1, temp2 = 0;
    temp1 = static_cast<T>(*itr_in) * (PI / 2);
    temp2 = enhancement_amount_value * std::sin(temp1 * 4);
    *itr_out = std::sin(temp1 + temp2);
    itr_out++;
  }
  *output = out;
  return Status::OK();
}

/// \brief Apply a DC shift to the audio.
/// \param input/output: Tensor of shape <...,time>.
/// \param shift: the amount to shift the audio.
/// \param limiter_gain: used only on peaks to prevent clipping.
/// \return Status code.
template <typename T>
Status DCShift(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output, float shift, float limiter_gain) {
  float limiter_threshold = 0.0;
  if (shift != limiter_gain && shift != 0) {
    limiter_threshold = 1.0 - (std::abs(shift) - limiter_gain);
    for (auto itr = input->begin<T>(); itr != input->end<T>(); itr++) {
      if (*itr > limiter_threshold && shift > 0) {
        T peak = (*itr - limiter_threshold) * limiter_gain / (1 - limiter_threshold);
        T sample = (peak + limiter_threshold + shift);
        *itr = sample > limiter_threshold ? limiter_threshold : sample;
      } else if (*itr < -limiter_threshold && shift < 0) {
        T peak = (*itr + limiter_threshold) * limiter_gain / (1 - limiter_threshold);
        T sample = (peak + limiter_threshold + shift);
        *itr = sample < -limiter_threshold ? -limiter_threshold : sample;
      } else {
        T sample = (*itr + shift);
        *itr = (sample > 1 || sample < -1) ? (sample > 1 ? 1 : -1) : sample;
      }
    }
  } else {
    for (auto itr = input->begin<T>(); itr != input->end<T>(); itr++) {
      T sample = (*itr + shift);
      *itr = sample > 1 || sample < -1 ? (sample > 1 ? 1 : -1) : sample;
    }
  }
  *output = input;
  return Status::OK();
}

/// \brief Perform an IIR filter by evaluating difference equation.
/// \param input/output: Tensor of shape <..., time>
/// \param a_coeffs: denominator coefficients of difference equation of dimension of (n_order + 1).
/// \param b_coeffs: numerator coefficients of difference equation of dimension of (n_order + 1).
/// \param clamp: If True, clamp the output signal to be in the range [-1, 1] (Default: True).
/// \return Status code
template <typename T>
Status LFilter(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output, std::vector<T> a_coeffs,
               std::vector<T> b_coeffs, bool clamp) {
  //  pack batch
  TensorShape input_shape = input->shape();
  TensorShape toShape({input->Size() / input_shape[-1], input_shape[-1]});
  input->Reshape(toShape);
  auto shape_0 = input->shape()[0];
  auto shape_1 = input->shape()[1];
  std::vector<T> signal;
  std::shared_ptr<Tensor> out;
  std::vector<T> out_vect(shape_0 * shape_1);
  size_t x_idx = 0;
  size_t channel_idx = 1;
  size_t m_num_order = b_coeffs.size() - 1;
  size_t m_den_order = a_coeffs.size() - 1;
  // init A_coeffs and B_coeffs by div(a0)
  for (size_t i = 1; i < a_coeffs.size(); i++) {
    a_coeffs[i] /= a_coeffs[0];
  }
  for (size_t i = 0; i < b_coeffs.size(); i++) {
    b_coeffs[i] /= a_coeffs[0];
  }
  // Sliding window
  T *m_px = new T[m_num_order + 1];
  T *m_py = new T[m_den_order + 1];

  // Tensor -> vector
  for (auto itr = input->begin<T>(); itr != input->end<T>();) {
    while (x_idx < shape_1 * channel_idx) {
      signal.push_back(*itr);
      itr++;
      x_idx++;
    }
    // Sliding window
    for (size_t j = 0; j < m_den_order; j++) {
      m_px[j] = static_cast<T>(0);
    }
    for (size_t j = 0; j <= m_den_order; j++) {
      m_py[j] = static_cast<T>(0);
    }
    // Each channel is processed with the sliding window
    for (size_t i = x_idx - shape_1; i < x_idx; i++) {
      m_px[m_num_order] = signal[i];
      for (size_t j = 0; j < m_num_order + 1; j++) {
        m_py[m_num_order] += b_coeffs[j] * m_px[m_num_order - j];
      }
      for (size_t j = 1; j < m_den_order + 1; j++) {
        m_py[m_num_order] -= a_coeffs[j] * m_py[m_num_order - j];
      }
      if (clamp) {
        if (m_py[m_num_order] > static_cast<T>(1.))
          out_vect[i] = static_cast<T>(1.);
        else if (m_py[m_num_order] < static_cast<T>(-1.))
          out_vect[i] = static_cast<T>(-1.);
        else
          out_vect[i] = m_py[m_num_order];
      } else {
        out_vect[i] = m_py[m_num_order];
      }
      if (i + 1 == x_idx) continue;
      for (size_t j = 0; j < m_num_order; j++) {
        m_px[j] = m_px[j + 1];
      }
      for (size_t j = 0; j < m_num_order; j++) {
        m_py[j] = m_py[j + 1];
      }
      m_py[m_num_order] = static_cast<T>(0);
    }
    if (x_idx % shape_1 == 0) {
      ++channel_idx;
    }
  }
  // unpack batch
  Tensor::CreateFromVector(out_vect, input_shape, &out);
  *output = out;
  delete[] m_px;
  delete[] m_py;
  return Status::OK();
}

/// \brief Stretch STFT in time at a given rate, without changing the pitch.
/// \param input: Tensor of shape <..., freq, time>.
/// \param rate: Stretch factor.
/// \param phase_advance: Expected phase advance in each bin.
/// \param output: Tensor after stretch in time domain.
/// \return Status code.
Status TimeStretch(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output, float rate, float hop_length,
                   float n_freq);

/// \brief Apply a mask along axis.
/// \param input: Tensor of shape <..., freq, time>.
/// \param output: Tensor of shape <..., freq, time>.
/// \param mask_param: Number of columns to be masked will be uniformly sampled from [0, mask_param].
/// \param mask_value: Value to assign to the masked columns.
/// \param axis: Axis to apply masking on (1 -> frequency, 2 -> time).
/// \param rnd: Number generator.
/// \return Status code.
Status RandomMaskAlongAxis(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output, int32_t mask_param,
                           float mask_value, int axis, std::mt19937 rnd);

/// \brief Apply a mask along axis. All examples will have the same mask interval.
/// \param input: Tensor of shape <..., freq, time>.
/// \param output: Tensor of shape <..., freq, time>.
/// \param mask_width: The width of the mask.
/// \param mask_start: Starting position of the mask.
///     Mask will be applied from indices [mask_start, mask_start + mask_width).
/// \param mask_value: Value to assign to the masked columns.
/// \param axis: Axis to apply masking on (1 -> frequency, 2 -> time).
/// \return Status code.
Status MaskAlongAxis(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output, int32_t mask_width,
                     int32_t mask_start, float mask_value, int32_t axis);

/// \brief Compute the norm of complex tensor input.
/// \param power Power of the norm description (optional).
/// \param input Tensor shape of <..., complex=2>.
/// \param output Tensor shape of <..., >.
/// \return Status code.
Status ComplexNorm(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output, float power);

/// \brief Decode mu-law encoded signal.
/// \param input Tensor of shape <..., time>.
/// \param output Tensor of shape <..., time>.
/// \param quantization_channels Number of channels.
/// \return Status code.
Status MuLawDecoding(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output, int quantization_channels);

/// \brief Add a fade in and/or fade out to an input.
/// \param[in] input: The input tensor.
/// \param[out] output: Added fade in and/or fade out audio with the same shape.
/// \param[in] fade_in_len: Length of fade-in (time frames).
/// \param[in] fade_out_len: Length of fade-out (time frames).
/// \param[in] fade_shape: Shape of fade.
Status Fade(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output, int32_t fade_in_len,
            int32_t fade_out_len, FadeShape fade_shape);

/// \brief Add a volume to an waveform.
/// \param input/output: Tensor of shape <..., time>.
/// \param gain: Gain value, varies according to the value of gain_type.
/// \param gain_type: Type of gain, should be one of [GainType::kAmplitude, GainType::kDb, GainType::kPower].
/// \return Status code.
template <typename T>
Status Vol(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output, T gain, GainType gain_type) {
  const T lower_bound = -1;
  const T upper_bound = 1;

  // DB is a unit which converts a numeric value into decibel scale and for conversion, we have to use log10
  // A(in dB) = 20log10(A in amplitude)
  // When referring to measurements of power quantities, a ratio can be expressed as a level in decibels by evaluating
  // ten times the base-10 logarithm of the ratio of the measured quantity to reference value
  // A(in dB) = 10log10(A in power)
  const int power_factor_div = 20;
  const int power_factor_mul = 10;
  const int base = 10;

  if (gain_type == GainType::kDb) {
    if (gain != 0) {
      gain = std::pow(base, (gain / power_factor_div));
    }
  } else if (gain_type == GainType::kPower) {
    gain = power_factor_mul * std::log10(gain);
    gain = std::pow(base, (gain / power_factor_div));
  }

  for (auto itr = input->begin<T>(); itr != input->end<T>(); itr++) {
    if (gain != 0 || gain_type == GainType::kAmplitude) {
      *itr = (*itr) * gain;
    }
    *itr = std::min(std::max((*itr), lower_bound), upper_bound);
  }

  *output = input;

  return Status::OK();
}

/// \brief Separate a complex-valued spectrogram with shape (…, 2) into its magnitude and phase.
/// \param input: Complex tensor.
/// \param output: The magnitude and phase of the complex tensor.
/// \param power: Power of the norm.
Status Magphase(const TensorRow &input, TensorRow *output, float power);

/// \brief Compute Normalized Cross-Correlation Function (NCCF).
/// \param input: Tensor of shape <channel,waveform_length>.
/// \param output: Tensor of shape <channel, num_of_frames, lags>.
/// \param sample_rate: The sample rate of the waveform (Hz).
/// \param frame_time: Duration of a frame.
/// \param freq_low: Lowest frequency that can be detected (Hz).
/// \return Status code.
template <typename T>
Status ComputeNccf(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output, int32_t sample_rate,
                   float frame_time, int32_t freq_low) {
  auto channel = input->shape()[0];
  auto waveform_length = input->shape()[1];
  size_t idx = 0;
  size_t channel_idx = 1;
  int32_t lags = static_cast<int32_t>(ceil(static_cast<float>(sample_rate) / freq_low));
  int32_t frame_size = static_cast<int32_t>(ceil(sample_rate * frame_time));
  int32_t num_of_frames = static_cast<int32_t>(ceil(static_cast<float>(waveform_length) / frame_size));
  int32_t p = lags + num_of_frames * frame_size - waveform_length;
  TensorShape output_shape({channel, num_of_frames, lags});
  DataType intput_type = input->type();
  RETURN_IF_NOT_OK(Tensor::CreateEmpty(output_shape, intput_type, output));
  // pad p 0 in -1 dimension
  std::vector<T> signal;
  // Tensor -> vector
  for (auto itr = input->begin<T>(); itr != input->end<T>();) {
    while (idx < waveform_length * channel_idx) {
      signal.push_back(*itr);
      ++itr;
      ++idx;
    }
    // Each channel is processed with the sliding window
    // waveform：[channel, time] -->  waveform：[channel, time+p]
    for (size_t i = 0; i < p; ++i) {
      signal.push_back(static_cast<T>(0.0));
    }
    if (idx % waveform_length == 0) {
      ++channel_idx;
    }
  }
  // compute ncc
  for (dsize_t lag = 1; lag <= lags; ++lag) {
    // compute one ncc
    // one ncc out
    std::vector<T> out;
    channel_idx = 1;
    idx = 0;
    size_t win_idx = 0;
    size_t waveform_length_p = waveform_length + p;
    // Traversal signal
    for (auto itr = signal.begin(); itr != signal.end();) {
      // Each channel is processed with the sliding window
      size_t s1 = idx;
      size_t s2 = idx + lag;
      size_t frame_count = 0;
      T s1_norm = static_cast<T>(0);
      T s2_norm = static_cast<T>(0);
      T ncc_umerator = static_cast<T>(0);
      T ncc = static_cast<T>(0);
      while (idx < waveform_length_p * channel_idx) {
        // Sliding window
        if (frame_count == num_of_frames) {
          ++itr;
          ++idx;
          continue;
        }
        if (win_idx < frame_size) {
          ncc_umerator += signal[s1] * signal[s2];
          s1_norm += signal[s1] * signal[s1];
          s2_norm += signal[s2] * signal[s2];
          ++win_idx;
          ++s1;
          ++s2;
        }
        if (win_idx == frame_size) {
          if (s1_norm != static_cast<T>(0.0) && s2_norm != static_cast<T>(0.0)) {
            ncc = ncc_umerator / s1_norm / s2_norm;
          } else {
            ncc = static_cast<T>(0.0);
          }
          out.push_back(ncc);
          ncc_umerator = static_cast<T>(0.0);
          s1_norm = static_cast<T>(0.0);
          s2_norm = static_cast<T>(0.0);
          ++frame_count;
          win_idx = 0;
        }
        ++itr;
        ++idx;
      }
      if (idx % waveform_length_p == 0) {
        ++channel_idx;
      }
    }  // compute one ncc
    // cat tensor
    auto itr_out = out.begin();
    for (dsize_t row_idx = 0; row_idx < channel; ++row_idx) {
      for (dsize_t frame_idx = 0; frame_idx < num_of_frames; ++frame_idx) {
        RETURN_IF_NOT_OK((*output)->SetItemAt({row_idx, frame_idx, lag - 1}, *itr_out));
        ++itr_out;
      }
    }
  }  // compute ncc
  return Status::OK();
}

/// \brief For each frame, take the highest value of NCCF.
/// \param input: Tensor of shape <channel, num_of_frames, lags>.
/// \param output: Tensor of shape <channel, num_of_frames>.
/// \param sample_rate: The sample rate of the waveform (Hz).
/// \param freq_high: Highest frequency that can be detected (Hz).
/// \return Status code.
template <typename T>
Status FindMaxPerFrame(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output, int32_t sample_rate,
                       int32_t freq_high) {
  std::vector<T> signal;
  std::vector<int> out;
  auto channel = input->shape()[0];
  auto num_of_frames = input->shape()[1];
  auto lags = input->shape()[2];
  int32_t lag_min = static_cast<int32_t>(ceil(static_cast<float>(sample_rate) / freq_high));
  TensorShape out_shape({channel, num_of_frames});
  // pack batch
  for (auto itr = input->begin<T>(); itr != input->end<T>(); ++itr) {
    signal.push_back(*itr);
  }
  // find the best nccf
  T best_max_value = static_cast<T>(0.0);
  T half_max_value = static_cast<T>(0.0);
  int32_t best_max_indices = 0;
  int32_t half_max_indices = 0;
  auto thresh = static_cast<T>(0.99);
  auto lags_half = lags / 2;
  for (dsize_t channel_idx = 0; channel_idx < channel; ++channel_idx) {
    for (dsize_t frame_idx = 0; frame_idx < num_of_frames; ++frame_idx) {
      auto index_01 = channel_idx * num_of_frames * lags + frame_idx * lags + lag_min;
      best_max_value = signal[index_01];
      half_max_value = signal[index_01];
      best_max_indices = lag_min;
      half_max_indices = lag_min;
      for (dsize_t lag_idx = 0; lag_idx < lags; ++lag_idx) {
        if (lag_idx > lag_min) {
          auto index_02 = channel_idx * num_of_frames * lags + frame_idx * lags + lag_idx;
          if (signal[index_02] > best_max_value) {
            best_max_value = signal[index_02];
            best_max_indices = lag_idx;
            if (lag_idx < lags_half) {
              half_max_value = signal[index_02];
              half_max_indices = lag_idx;
            }
          }
        }
      }
      // Add back minimal lag
      // Add 1 empirical calibration offset
      if (half_max_value > best_max_value * thresh) {
        out.push_back(half_max_indices + 1);
      } else {
        out.push_back(best_max_indices + 1);
      }
    }
  }
  // unpack batch
  RETURN_IF_NOT_OK(Tensor::CreateFromVector(out, out_shape, output));
  return Status::OK();
}

/// \brief Apply median smoothing to the 1D tensor over the given window.
/// \param input: Tensor of shape<channel, num_of_frames>.
/// \param output: Tensor of shape <channel, num_of_window>.
/// \param win_length: The window length for median smoothing (in number of frames).
/// \return Status code.
Status MedianSmoothing(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output, int32_t win_length);

/// \brief Detect pitch frequency.
/// \param input: Tensor of shape <channel,waveform_length>.
/// \param output: Tensor of shape <channel, num_of_frames, lags>.
/// \param sample_rate: The sample rate of the waveform (Hz).
/// \param frame_time: Duration of a frame.
/// \param win_length: The window length for median smoothing (in number of frames).
/// \param freq_low: Lowest frequency that can be detected (Hz).
/// \param freq_high: Highest frequency that can be detected (Hz).
/// \return Status code.
Status DetectPitchFrequency(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output, int32_t sample_rate,
                            float frame_time, int32_t win_length, int32_t freq_low, int32_t freq_high);

}  // namespace dataset
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_AUDIO_KERNELS_AUDIO_UTILS_H_
