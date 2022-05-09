/*****************************************************************************
 *   Copyright (c) 2022, Roland Grinis, GrinisRIT ltd.                       *
 *   (roland.grinis@grinisrit.com)                                           *
 *   All rights reserved.                                                    *
 *   See the file COPYING for full copying permissions.                      *
 *                                                                           *
 *   This program is free software: you can redistribute it and/or modify    *
 *   it under the terms of the GNU General Public License as published by    *
 *   the Free Software Foundation, either version 3 of the License, or       *
 *   (at your option) any later version.                                     *
 *                                                                           *
 *   This program is distributed in the hope that it will be useful,         *
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of          *
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the            *
 *   GNU General Public License for more details.                            *
 *                                                                           *
 *   You should have received a copy of the GNU General Public License       *
 *   along with this program.  If not, see <http://www.gnu.org/licenses/>.   *
 *****************************************************************************/
/**
 * Implemented by: Roland Grinis
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
    using RmsOptim = torch::optim::RMSprop;
    using RmsOptimOpts = torch::optim::RMSpropOptions;
    using AdamWOptim = torch::optim::AdamW;
    using AdamWOptimOpts = torch::optim::AdamWOptions;
    using AdagradOptim = torch::optim::Adagrad;
    using AdagradOptimOpts = torch::optim::AdagradOptions;
    using SgdOptim = torch::optim::SGD;
    using SgdOptimOpts = torch::optim::SGDOptions;

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

    template<typename Runner, typename... Args>
    void safe_run(JNIEnv *env, const Runner &runner, Args &&... args) {
        const auto noa_exception = env->FindClass("space/kscience/kmath/noa/NoaException");
        try {
            runner(std::forward<Args>(args)...);
        } catch (const std::exception &e) {
            env->ThrowNew(noa_exception, e.what());
        }
    }

    template<typename Optim, typename OptimOptions, typename... Options>
    Optim get_optim(JitModule &module, Options &&... opts) {
        return Optim{parameters(module.jit_module, false), OptimOptions(std::forward<Options>(opts)...)};
    }

    template<typename Optim, typename OptimOptions>
    Optim get_rms_optim(JitModule &module, double learningRate, double alpha, 
        double eps, double weightDecay, double momentum, bool centered) {
        return Optim{parameters(module.jit_module, false), 
        OptimOptions(learningRate).alpha(alpha).eps(eps).weight_decay(weightDecay).momentum(momentum).centered(centered)};
    }

    template<typename Optim, typename OptimOptions>
    Optim get_adamw_optim(JitModule &module, double learningRate, double beta1,
        double beta2, double eps, double weightDecay, bool amsgrad) {
        return Optim{parameters(module.jit_module, false), 
        OptimOptions(learningRate).betas(std::tuple<double,double>(beta1, beta2)).eps(eps).weight_decay(weightDecay).amsgrad(amsgrad)};
    }

    template<typename Optim, typename OptimOptions>
    Optim get_adagrad_optim(JitModule &module, double learningRate, double weightDecay,
        double lrDecay, double initialAccumulatorValue, double eps) {
        return Optim{parameters(module.jit_module, false), 
        OptimOptions(learningRate).weight_decay(weightDecay).lr_decay(lrDecay).initial_accumulator_value(initialAccumulatorValue).eps(eps)};
    }

    template<typename Optim, typename OptimOptions>
    Optim get_sgd_optim(JitModule &module, double learningRate, double momentum,
        double dampening, double weightDecay, bool nesterov) {
        return Optim{parameters(module.jit_module, false), 
        OptimOptions(learningRate).momentum(momentum).dampening(dampening).weight_decay(weightDecay).nesterov(nesterov)};
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

    template<typename Dtype>
    inline const auto set_slice_blob = [](Tensor &tensor, int d, int s, int e, Dtype *data) {
        tensor.slice(d,s,e) = torch::from_blob(data, tensor.slice(d,s,e).sizes(), dtype<Dtype>()).to(
                dtype<Dtype>()
                .layout(torch::kStrided)
                .device(tensor.device()),
                false, true);
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
