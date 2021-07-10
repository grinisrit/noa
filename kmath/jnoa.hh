/*
 * BSD 2-Clause License
 *
 * Copyright (c) 2021, Roland Grinis, GrinisRIT ltd. (roland.grinis@grinisrit.com)
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 *    list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#pragma once

#include <noa/utils/common.hh>

#include <jni.h>
#include <torch/torch.h>

namespace jnoa {

    using namespace noa::utils;

    using TensorHandle = void *;

    template<typename Handle>
    inline Tensor &cast_tensor(const Handle &tensor_handle) {
        return *static_cast<Tensor *>((TensorHandle) tensor_handle);
    }

    template<typename Dtype>
    inline c10::TensorOptions dtype() {
        return torch::dtype(c10::CppTypeToScalarType<Dtype>{});
    }

    template<typename Result, typename Runner, typename... Args>
    std::optional<Result> safe_run(JNIEnv *env, const Runner &runner, Args &&... args) {
        const auto noa_exception = env->FindClass("space/kscience/kmath/noa/NoaException");
        try {
            return std::make_optional(runner(std::forward<Args>(args)...));
        } catch (const std::exception &e) {
            env->ThrowNew(noa_exception, e.what());
            return std::nullopt;
        }
    }

    template<typename Runner, typename... Args>
    void safe_run(JNIEnv *env, const Runner &runner, Args &&... args) {
        const auto noa_exception = env->FindClass("space/kscience/kmath/noa/NoaException");
        try {
            runner(std::forward<Args>(args)...);
        } catch (const std::exception &e) {
            env->ThrowNew(noa_exception, e.what());
        }
    }

    inline int device_to_int(const Tensor &tensor) {
        return (tensor.device().type() == torch::kCPU) ? 0 : 1 + tensor.device().index();
    }

    inline torch::Device int_to_device(int device_int) {
        return (device_int == 0) ? torch::kCPU : torch::Device(torch::kCUDA, device_int - 1);
    }

    inline std::vector<int64_t> to_vec_int(const int *arr, const int arr_size) {
        auto vec = std::vector<int64_t>(arr_size);
        vec.assign(arr, arr + arr_size);
        return vec;
    }

    inline std::vector<int64_t> to_shape(JNIEnv *env, const jintArray &shape) {
        return jnoa::to_vec_int(env->GetIntArrayElements(shape, nullptr),
                                env->GetArrayLength(shape));
    }

    inline std::vector<at::indexing::TensorIndex> to_index(const int *arr, const int arr_size)
    {
        std::vector<at::indexing::TensorIndex> index;
        for (int i = 0; i < arr_size; i++)
        {
            index.emplace_back(arr[i]);
        }
        return index;
    }

    inline const auto test_exception = [](const int seed) {
        torch::rand({2, 3}) + torch::rand({3, 2}); //this should throw
        return seed;
    };

    template<typename Handle>
    inline void dispose_tensor(const Handle &tensor_handle) {
        delete static_cast<Tensor *>((TensorHandle) tensor_handle);
    }

    template<typename Dtype>
    inline const auto from_blob = [](Dtype *data, const std::vector<int64_t> &shape, const torch::Device &device) {
        return torch::from_blob(data, shape, dtype<Dtype>()).to(
                dtype<Dtype>()
                        .layout(torch::kStrided)
                        .device(device),
                false, true);
    };

    inline std::string tensor_to_string(const Tensor &tensor)
    {
        std::stringstream bufrep;
        bufrep << tensor;
        return bufrep.str();
    }

    template <typename DType>
    inline const auto getter = [](const torch::Tensor &tensor, const int *index)
    {
        return tensor.index(to_index(index, tensor.dim())).item<DType>();
    };

    template <typename DType>
    inline const auto setter = [](const torch::Tensor &tensor, const int *index, const DType &value)
    {
        tensor.index(to_index(index, tensor.dim())) = value;
    };

    template <typename Dtype>
    inline const auto randn = [](const std::vector<int64_t> &shape, const torch::Device &device)
    {
        return torch::randn(shape, dtype<Dtype>().layout(torch::kStrided).device(device));
    };

    template <typename Dtype>
    inline const auto rand = [](const std::vector<int64_t> &shape, const torch::Device &device)
    {
        return torch::rand(shape, dtype<Dtype>().layout(torch::kStrided).device(device));
    };

    template <typename Dtype>
    inline const auto randint = [](long low, long high, const std::vector<int64_t> &shape, const torch::Device &device)
    {
        return torch::randint(low, high, shape, dtype<Dtype>().layout(torch::kStrided).device(device));
    };

    template <typename Dtype>
    inline const auto full = [](const Dtype &value, const std::vector<int64_t> &shape, const torch::Device &device)
    {
        return torch::full(shape, value, dtype<Dtype>().layout(torch::kStrided).device(device));
    };


} // namespace jnoa
