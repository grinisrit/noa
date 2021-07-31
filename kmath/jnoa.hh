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

#include <noa/utils/numerics.hh>
#include <noa/utils/common.hh>

#include <jni.h>
#include <torch/torch.h>

namespace jnoa {

    using namespace noa::utils;

    using TensorPair = std::tuple<Tensor, Tensor>;
    using TensorTriple = std::tuple<Tensor, Tensor, Tensor>;
    using VoidHandle = void *;
    using AdamOptim = torch::optim::Adam;
    using AdamOptimOpts = torch::optim::AdamOptions;

    struct JitModule {
        torch::jit::Module jit_module;
        NamedTensors parameters;
        NamedTensors buffers;

        explicit JitModule(const std::string &path) {
            const auto jit_module_ = load_module(path);
            if (!jit_module_.has_value())
                throw std::invalid_argument("Failed to load JIT module from:\n" + path);
            else
                jit_module = jit_module_.value();
            parameters = named_parameters(jit_module, false);
            buffers = named_buffers(jit_module, false);
        }

        Tensor &get_parameter(const std::string &name) {
            if (parameters.find(name) == parameters.end())
                throw std::invalid_argument("No parameter with name: " + name + "\n");
            return parameters.at(name);
        }

        Tensor &get_buffer(const std::string &name) {
            if (buffers.find(name) == buffers.end())
                throw std::invalid_argument("No buffer with name: " + name + "\n");
            return buffers.at(name);
        }
    };

    template<typename Target, typename Handle>
    inline Target &cast(const Handle &handle) {
        return *static_cast<Target *>((VoidHandle) handle);
    }

    template<typename Target, typename Handle>
    inline void dispose(const Handle &handle) {
        delete static_cast<Target *>((VoidHandle) handle);
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

    template<typename Optim, typename OptimOptions, typename... Options>
    Optim get_optim(JitModule &module, Options &&... opts) {
        return Optim{parameters(module.jit_module, false), OptimOptions(std::forward<Options>(opts)...)};
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

    inline std::vector<at::indexing::TensorIndex> to_index(const int *arr, const int arr_size) {
        std::vector<at::indexing::TensorIndex> index;
        for (int i = 0; i < arr_size; i++) {
            index.emplace_back(arr[i]);
        }
        return index;
    }

    inline const auto test_exception = [](const int seed) {
        torch::rand({2, 3}) + torch::rand({3, 2}); //this should throw
        return seed;
    };

    template<typename Dtype>
    inline const auto from_blob = [](Dtype *data, const std::vector<int64_t> &shape, const torch::Device &device) {
        return torch::from_blob(data, shape, dtype<Dtype>()).to(
                dtype<Dtype>()
                        .layout(torch::kStrided)
                        .device(device),
                false, true);
    };

    template<typename Dtype>
    inline const auto assign_blob = [](Tensor &tensor, Dtype *data) {
        tensor = torch::from_blob(data, tensor.sizes(), dtype<Dtype>()).to(
                dtype<Dtype>()
                .layout(torch::kStrided)
                .device(tensor.device()),
                false, true);
    };

    template<typename Dtype>
    inline const auto set_blob = [](Tensor &tensor, int i, Dtype *data) {
        tensor[i] = torch::from_blob(data, tensor[i].sizes(), dtype<Dtype>()).to(
                dtype<Dtype>()
                .layout(torch::kStrided)
                .device(tensor.device()),
                false, true);
    };

    template<typename Dtype>
    inline const auto get_blob = [](const Tensor &tensor, Dtype *data) {
        const auto cpu_tensor = tensor.to(torch::device(torch::kCPU), false, false);
        const Dtype *ptr = cpu_tensor.data_ptr<Dtype>();
        const int n = tensor.numel();
        for(int i = 0; i < n; i++)
            data[i] = ptr[i];
    };


    inline std::string tensor_to_string(const Tensor &tensor) {
        std::stringstream bufrep;
        bufrep << tensor;
        return bufrep.str();
    }

    template<typename DType>
    inline const auto getter = [](const torch::Tensor &tensor, const int *index) {
        return tensor.index(to_index(index, tensor.dim())).item<DType>();
    };

    template<typename DType>
    inline const auto setter = [](torch::Tensor &tensor, const int *index, const DType &value) {
        tensor.index(to_index(index, tensor.dim())) = value;
    };

    template<typename Dtype>
    inline const auto rand_normal = [](const std::vector<int64_t> &shape, const torch::Device &device) {
        return torch::randn(shape, dtype<Dtype>().layout(torch::kStrided).device(device));
    };

    template<typename Dtype>
    inline const auto rand_uniform = [](const std::vector<int64_t> &shape, const torch::Device &device) {
        return torch::rand(shape, dtype<Dtype>().layout(torch::kStrided).device(device));
    };

    template<typename Dtype>
    inline const auto rand_discrete = [](long low, long high, const std::vector<int64_t> &shape,
                                         const torch::Device &device) {
        return torch::randint(low, high, shape, dtype<Dtype>().layout(torch::kStrided).device(device));
    };

    template<typename Dtype>
    inline const auto full = [](const Dtype &value, const std::vector<int64_t> &shape, const torch::Device &device) {
        return torch::full(shape, value, dtype<Dtype>().layout(torch::kStrided).device(device));
    };

    inline const auto hess = [](const Tensor &value, const Tensor &variable) {
        const auto hess_ = numerics::hessian(ADGraph{value, {variable}});
        if (hess_.has_value())
            return hess_.value()[0];
        else
            throw std::invalid_argument("Failed to compute Hessian");
    };

    std::string to_string(JNIEnv *env, const jstring &jstr) {
        if (!jstr)
            return "";

        const auto string_class = env->GetObjectClass(jstr);
        const auto get_bytes = env->GetMethodID(string_class, "getBytes", "(Ljava/lang/String;)[B");
        const auto jbytes = (jbyteArray) env->CallObjectMethod(jstr, get_bytes, env->NewStringUTF("UTF-8"));

        const auto length = (size_t) env->GetArrayLength(jbytes);
        const auto pbytes = env->GetByteArrayElements(jbytes, nullptr);

        std::string ret = std::string((char *) pbytes, length);
        env->ReleaseByteArrayElements(jbytes, pbytes, JNI_ABORT);

        env->DeleteLocalRef(jbytes);
        env->DeleteLocalRef(string_class);
        return ret;
    }

    inline const auto load_jit_module =
            [](const std::string &path, torch::ScalarType dtype, torch::Device device) {
                auto module = JitModule(path);
                module.jit_module.to(dtype);
                module.jit_module.to(device);
                return module;
            };

    inline const auto unsafe_load_tensor =
            [](const std::string &path, torch::ScalarType dtype, torch::Device device) {
                auto tensor = load_tensor(path);
                if (tensor.has_value())
                    return tensor.value().to(dtype).to(device);
                else
                    throw std::invalid_argument("Failed to load tensor from " + path);
            };

} // namespace jnoa
