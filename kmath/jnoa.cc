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

#include "jnoa.hh"
#include "space_kscience_kmath_noa_JNoa.h"

JNIEXPORT jint JNICALL Java_space_kscience_kmath_noa_JNoa_testException
        (JNIEnv *env, jclass, jint seed) {
    const auto res = jnoa::safe_run<int>(env, jnoa::test_exception, seed);
    return res.has_value() ? res.value() : 0;
}

JNIEXPORT jboolean JNICALL Java_space_kscience_kmath_noa_JNoa_cudaIsAvailable
        (JNIEnv *, jclass) {
    return torch::cuda::is_available();
}

JNIEXPORT jint JNICALL Java_space_kscience_kmath_noa_JNoa_getNumThreads
        (JNIEnv *, jclass) {
    return torch::get_num_threads();
}

JNIEXPORT void JNICALL Java_space_kscience_kmath_noa_JNoa_setNumThreads
        (JNIEnv *, jclass, jint num_threads) {
    torch::set_num_threads(num_threads);
}

JNIEXPORT void JNICALL Java_space_kscience_kmath_noa_JNoa_setSeed
        (JNIEnv *, jclass, jint seed) {
    torch::manual_seed(seed);
}

JNIEXPORT void JNICALL Java_space_kscience_kmath_noa_JNoa_disposeTensor
        (JNIEnv *, jclass, jlong tensor_handle) {
    if (tensor_handle != 0L)
        jnoa::dispose_tensor(tensor_handle);
}

JNIEXPORT jlong JNICALL Java_space_kscience_kmath_noa_JNoa_emptyTensor
        (JNIEnv *, jclass) {
    return (long) new jnoa::Tensor;
}

JNIEXPORT jlong JNICALL Java_space_kscience_kmath_noa_JNoa_fromBlobDouble
        (JNIEnv *env, jclass, jdoubleArray data, jintArray shape, jint device) {
    const auto res =
            jnoa::safe_run<jnoa::Tensor>(env,
                                         jnoa::from_blob<double>,
                                         env->GetDoubleArrayElements(data, nullptr),
                                         jnoa::to_shape(env, shape),
                                         jnoa::int_to_device(device));
    return res.has_value() ? (long) new jnoa::Tensor(res.value()) : 0L;
}

JNIEXPORT jlong JNICALL Java_space_kscience_kmath_noa_JNoa_fromBlobFloat
        (JNIEnv *env, jclass, jfloatArray data, jintArray shape, jint device) {
    const auto res =
            jnoa::safe_run<jnoa::Tensor>(env,
                                         jnoa::from_blob<float>,
                                         env->GetFloatArrayElements(data, nullptr),
                                         jnoa::to_shape(env, shape),
                                         jnoa::int_to_device(device));
    return res.has_value() ? (long) new jnoa::Tensor(res.value()) : 0L;
}

JNIEXPORT jlong JNICALL Java_space_kscience_kmath_noa_JNoa_fromBlobLong
        (JNIEnv *env, jclass, jlongArray data, jintArray shape, jint device) {
    const auto res =
            jnoa::safe_run<jnoa::Tensor>(env,
                                         jnoa::from_blob<long>,
                                         env->GetLongArrayElements(data, nullptr),
                                         jnoa::to_shape(env, shape),
                                         jnoa::int_to_device(device));
    return res.has_value() ? (long) new jnoa::Tensor(res.value()) : 0L;
}

JNIEXPORT jlong JNICALL Java_space_kscience_kmath_noa_JNoa_fromBlobInt
        (JNIEnv *env, jclass, jintArray data, jintArray shape, jint device) {
    const auto res =
            jnoa::safe_run<jnoa::Tensor>(env,
                                         jnoa::from_blob<int>,
                                         env->GetIntArrayElements(data, nullptr),
                                         jnoa::to_shape(env, shape),
                                         jnoa::int_to_device(device));
    return res.has_value() ? (long) new jnoa::Tensor(res.value()) : 0L;
}

JNIEXPORT jlong JNICALL Java_space_kscience_kmath_noa_JNoa_copyTensor
        (JNIEnv *, jclass, jlong tensor_handle) {
    return (long) new jnoa::Tensor(jnoa::cast_tensor(tensor_handle).clone());
}

JNIEXPORT jlong JNICALL Java_space_kscience_kmath_noa_JNoa_copyToDevice
        (JNIEnv *env, jclass, jlong tensor_handle, jint device) {
    const auto res =
            jnoa::safe_run<jnoa::Tensor>(env,
                                         [](const auto &tensor, const auto &device) {
                                             return tensor.to(device, false, true);
                                         },
                                         jnoa::cast_tensor(tensor_handle),
                                         jnoa::int_to_device(device));
    return res.has_value() ? (long) new jnoa::Tensor(res.value()) : 0L;
}

JNIEXPORT jlong JNICALL Java_space_kscience_kmath_noa_JNoa_copyToDouble
        (JNIEnv *, jclass, jlong tensor_handle) {
    return (long) new jnoa::Tensor(
            jnoa::cast_tensor(tensor_handle)
                    .to(jnoa::dtype<double>(), false, false));
}

JNIEXPORT jlong JNICALL Java_space_kscience_kmath_noa_JNoa_copyToFloat
        (JNIEnv *, jclass, jlong tensor_handle) {
    return (long) new jnoa::Tensor(
            jnoa::cast_tensor(tensor_handle)
                    .to(jnoa::dtype<float>(), false, false));
}

JNIEXPORT jlong JNICALL Java_space_kscience_kmath_noa_JNoa_copyToLong
        (JNIEnv *, jclass, jlong tensor_handle) {
    return (long) new jnoa::Tensor(
            jnoa::cast_tensor(tensor_handle)
                    .to(jnoa::dtype<long>(), false, false));
}

JNIEXPORT jlong JNICALL Java_space_kscience_kmath_noa_JNoa_copyToInt
        (JNIEnv *, jclass, jlong tensor_handle) {
    return (long) new jnoa::Tensor(
            jnoa::cast_tensor(tensor_handle)
                    .to(jnoa::dtype<int>(), false, false));
}

JNIEXPORT jlong JNICALL Java_space_kscience_kmath_noa_JNoa_viewTensor
        (JNIEnv *env, jclass, jlong tensor_handle, jintArray shape) {
    const auto res =
            jnoa::safe_run<jnoa::Tensor>(env,
                                         [](const auto &tensor, const auto &shape) {
                                             return tensor.view(shape);
                                         },
                                         jnoa::cast_tensor(tensor_handle),
                                         jnoa::to_shape(env, shape));
    return res.has_value() ? (long) new jnoa::Tensor(res.value()) : 0L;
}

JNIEXPORT jlong JNICALL Java_space_kscience_kmath_noa_JNoa_viewAsTensor
        (JNIEnv *env, jclass, jlong tensor_handle, jlong as_tensor_handle) {
    const auto res =
            jnoa::safe_run<jnoa::Tensor>(env,
                                         [](const auto &tensor, const auto &tensor_ref) {
                                             return tensor.view_as(tensor_ref);
                                         },
                                         jnoa::cast_tensor(tensor_handle),
                                         jnoa::cast_tensor(as_tensor_handle));
    return res.has_value() ? (long) new jnoa::Tensor(res.value()) : 0L;
}

JNIEXPORT jstring JNICALL Java_space_kscience_kmath_noa_JNoa_tensorToString
        (JNIEnv *env, jclass, jlong tensor_handle) {
    return env->NewStringUTF(jnoa::tensor_to_string(jnoa::cast_tensor(tensor_handle)).c_str());
}

JNIEXPORT jint JNICALL Java_space_kscience_kmath_noa_JNoa_getDim
        (JNIEnv *, jclass, jlong tensor_handle) {
    return jnoa::cast_tensor(tensor_handle).dim();
}

JNIEXPORT jint JNICALL Java_space_kscience_kmath_noa_JNoa_getNumel
        (JNIEnv *, jclass, jlong tensor_handle) {
    return jnoa::cast_tensor(tensor_handle).numel();
}

JNIEXPORT jint JNICALL Java_space_kscience_kmath_noa_JNoa_getShapeAt
        (JNIEnv *, jclass, jlong tensor_handle, jint d) {
    return jnoa::cast_tensor(tensor_handle).size(d);
}

JNIEXPORT jint JNICALL Java_space_kscience_kmath_noa_JNoa_getStrideAt
        (JNIEnv *, jclass, jlong tensor_handle, jint d) {
    return jnoa::cast_tensor(tensor_handle).stride(d);
}

JNIEXPORT jint JNICALL Java_space_kscience_kmath_noa_JNoa_getDevice
        (JNIEnv *, jclass, jlong tensor_handle) {
    return jnoa::device_to_int(jnoa::cast_tensor(tensor_handle));
}

JNIEXPORT jdouble JNICALL Java_space_kscience_kmath_noa_JNoa_getItemDouble
        (JNIEnv *env, jclass, jlong tensor_handle) {
    const auto res =
            jnoa::safe_run<double>(env,
                                   [](const jnoa::Tensor &tensor) { return tensor.item<double>(); },
                                   jnoa::cast_tensor(tensor_handle));
    return res.has_value() ? res.value() : 0.;
}

JNIEXPORT jfloat JNICALL Java_space_kscience_kmath_noa_JNoa_getItemFloat
        (JNIEnv *env, jclass, jlong tensor_handle) {
    const auto res =
            jnoa::safe_run<float>(env,
                                  [](const jnoa::Tensor &tensor) { return tensor.item<float>(); },
                                  jnoa::cast_tensor(tensor_handle));
    return res.has_value() ? res.value() : 0.f;
}

JNIEXPORT jlong JNICALL Java_space_kscience_kmath_noa_JNoa_getItemLong
        (JNIEnv *env, jclass, jlong tensor_handle) {
    const auto res =
            jnoa::safe_run<long>(env,
                                 [](const jnoa::Tensor &tensor) { return tensor.item<long>(); },
                                 jnoa::cast_tensor(tensor_handle));
    return res.has_value() ? res.value() : 0L;
}

JNIEXPORT jint JNICALL Java_space_kscience_kmath_noa_JNoa_getItemInt
        (JNIEnv *env, jclass, jlong tensor_handle) {
    const auto res =
            jnoa::safe_run<int>(env,
                                [](const jnoa::Tensor &tensor) { return tensor.item<int>(); },
                                jnoa::cast_tensor(tensor_handle));
    return res.has_value() ? res.value() : 0;
}

JNIEXPORT jlong JNICALL Java_space_kscience_kmath_noa_JNoa_getIndex
        (JNIEnv *env, jclass, jlong tensor_handle, jint index) {
    const auto res =
            jnoa::safe_run<jnoa::Tensor>(env,
                                         [](const auto &tensor, const int index) {
                                             return tensor[index];
                                         },
                                         jnoa::cast_tensor(tensor_handle), index);
    return res.has_value() ? (long) new jnoa::Tensor(res.value()) : 0L;
}

JNIEXPORT jdouble JNICALL Java_space_kscience_kmath_noa_JNoa_getDouble
        (JNIEnv *env, jclass, jlong tensor_handle, jintArray index) {
    const auto res =
            jnoa::safe_run<double>(env,
                                   jnoa::getter<double>,
                                   jnoa::cast_tensor(tensor_handle),
                                   env->GetIntArrayElements(index, nullptr));
    return res.has_value() ? res.value() : 0.;
}

JNIEXPORT jfloat JNICALL Java_space_kscience_kmath_noa_JNoa_getFloat
        (JNIEnv *env, jclass, jlong tensor_handle, jintArray index) {
    const auto res =
            jnoa::safe_run<float>(env,
                                  jnoa::getter<float>,
                                  jnoa::cast_tensor(tensor_handle),
                                  env->GetIntArrayElements(index, nullptr));
    return res.has_value() ? res.value() : 0.f;
}

JNIEXPORT jlong JNICALL Java_space_kscience_kmath_noa_JNoa_getLong
        (JNIEnv *env, jclass, jlong tensor_handle, jintArray index) {
    const auto res =
            jnoa::safe_run<long>(env,
                                 jnoa::getter<long>,
                                 jnoa::cast_tensor(tensor_handle),
                                 env->GetIntArrayElements(index, nullptr));
    return res.has_value() ? res.value() : 0L;
}

JNIEXPORT jint JNICALL Java_space_kscience_kmath_noa_JNoa_getInt
        (JNIEnv *env, jclass, jlong tensor_handle, jintArray index) {
    const auto res =
            jnoa::safe_run<int>(env,
                                jnoa::getter<int>,
                                jnoa::cast_tensor(tensor_handle),
                                env->GetIntArrayElements(index, nullptr));
    return res.has_value() ? res.value() : 0;
}

JNIEXPORT void JNICALL Java_space_kscience_kmath_noa_JNoa_setDouble
        (JNIEnv *env, jclass, jlong tensor_handle, jintArray index, jdouble value) {
    jnoa::safe_run(env,
                   jnoa::setter<double>,
                   jnoa::cast_tensor(tensor_handle),
                   env->GetIntArrayElements(index, nullptr),
                   value);
}

JNIEXPORT void JNICALL Java_space_kscience_kmath_noa_JNoa_setFloat
        (JNIEnv *env, jclass, jlong tensor_handle, jintArray index, jfloat value) {
    jnoa::safe_run(env,
                   jnoa::setter<float>,
                   jnoa::cast_tensor(tensor_handle),
                   env->GetIntArrayElements(index, nullptr),
                   value);
}

JNIEXPORT void JNICALL Java_space_kscience_kmath_noa_JNoa_setLong
        (JNIEnv *env, jclass, jlong tensor_handle, jintArray index, jlong value) {
    jnoa::safe_run(env,
                   jnoa::setter<long>,
                   jnoa::cast_tensor(tensor_handle),
                   env->GetIntArrayElements(index, nullptr),
                   value);
}

JNIEXPORT void JNICALL Java_space_kscience_kmath_noa_JNoa_setInt
        (JNIEnv *env, jclass, jlong tensor_handle, jintArray index, jint value) {
    jnoa::safe_run(env,
                   jnoa::setter<int>,
                   jnoa::cast_tensor(tensor_handle),
                   env->GetIntArrayElements(index, nullptr),
                   value);
}

JNIEXPORT jlong JNICALL Java_space_kscience_kmath_noa_JNoa_randDouble
        (JNIEnv *env, jclass, jintArray shape, jint device) {
    const auto res =
            jnoa::safe_run<jnoa::Tensor>(env,
                                         jnoa::rand<double>,
                                         jnoa::to_shape(env, shape),
                                         jnoa::int_to_device(device));
    return res.has_value() ? (long) new jnoa::Tensor(res.value()) : 0L;
}

JNIEXPORT jlong JNICALL Java_space_kscience_kmath_noa_JNoa_randnDouble
        (JNIEnv *env, jclass, jintArray shape, jint device) {
    const auto res =
            jnoa::safe_run<jnoa::Tensor>(env,
                                         jnoa::randn<double>,
                                         jnoa::to_shape(env, shape),
                                         jnoa::int_to_device(device));
    return res.has_value() ? (long) new jnoa::Tensor(res.value()) : 0L;
}

JNIEXPORT jlong JNICALL Java_space_kscience_kmath_noa_JNoa_randFloat
        (JNIEnv *env, jclass, jintArray shape, jint device) {
    const auto res =
            jnoa::safe_run<jnoa::Tensor>(env,
                                         jnoa::rand<float>,
                                         jnoa::to_shape(env, shape),
                                         jnoa::int_to_device(device));
    return res.has_value() ? (long) new jnoa::Tensor(res.value()) : 0L;
}

JNIEXPORT jlong JNICALL Java_space_kscience_kmath_noa_JNoa_randnFloat
        (JNIEnv *env, jclass, jintArray shape, jint device) {
    const auto res =
            jnoa::safe_run<jnoa::Tensor>(env,
                                         jnoa::randn<float>,
                                         jnoa::to_shape(env, shape),
                                         jnoa::int_to_device(device));
    return res.has_value() ? (long) new jnoa::Tensor(res.value()) : 0L;
}

JNIEXPORT jlong JNICALL Java_space_kscience_kmath_noa_JNoa_randintDouble
        (JNIEnv *env, jclass, jlong low, jlong high, jintArray shape, jint device) {
    const auto res =
            jnoa::safe_run<jnoa::Tensor>(env,
                                         jnoa::randint<double>,
                                         low, high,
                                         jnoa::to_shape(env, shape),
                                         jnoa::int_to_device(device)
            );
    return res.has_value() ? (long) new jnoa::Tensor(res.value()) : 0L;
}

JNIEXPORT jlong JNICALL Java_space_kscience_kmath_noa_JNoa_randintFloat
        (JNIEnv *env, jclass, jlong low, jlong high, jintArray shape, jint device) {
    const auto res =
            jnoa::safe_run<jnoa::Tensor>(env,
                                         jnoa::randint<float>,
                                         low, high,
                                         jnoa::to_shape(env, shape),
                                         jnoa::int_to_device(device)
            );
    return res.has_value() ? (long) new jnoa::Tensor(res.value()) : 0L;
}

JNIEXPORT jlong JNICALL Java_space_kscience_kmath_noa_JNoa_randintLong
        (JNIEnv *env, jclass, jlong low, jlong high, jintArray shape, jint device) {
    const auto res =
            jnoa::safe_run<jnoa::Tensor>(env,
                                         jnoa::randint<long>,
                                         low, high,
                                         jnoa::to_shape(env, shape),
                                         jnoa::int_to_device(device)
            );
    return res.has_value() ? (long) new jnoa::Tensor(res.value()) : 0L;
}

JNIEXPORT jlong JNICALL Java_space_kscience_kmath_noa_JNoa_randintInt
        (JNIEnv *env, jclass, jlong low, jlong high, jintArray shape, jint device) {
    const auto res =
            jnoa::safe_run<jnoa::Tensor>(env,
                                         jnoa::randint<int>,
                                         low, high,
                                         jnoa::to_shape(env, shape),
                                         jnoa::int_to_device(device)
            );
    return res.has_value() ? (long) new jnoa::Tensor(res.value()) : 0L;
}

JNIEXPORT jlong JNICALL Java_space_kscience_kmath_noa_JNoa_randLike
        (JNIEnv *, jclass, jlong tensor_handle) {
    return (long) new jnoa::Tensor(torch::rand_like(jnoa::cast_tensor(tensor_handle)));
}

JNIEXPORT void JNICALL Java_space_kscience_kmath_noa_JNoa_randLikeAssign
        (JNIEnv *, jclass, jlong tensor_handle) {
    jnoa::cast_tensor(tensor_handle) = torch::rand_like(jnoa::cast_tensor(tensor_handle));
}

JNIEXPORT jlong JNICALL Java_space_kscience_kmath_noa_JNoa_randnLike
        (JNIEnv *, jclass, jlong tensor_handle) {
    return (long) new jnoa::Tensor(torch::randn_like(jnoa::cast_tensor(tensor_handle)));
}

JNIEXPORT void JNICALL Java_space_kscience_kmath_noa_JNoa_randnLikeAssign
        (JNIEnv *, jclass, jlong tensor_handle) {
    jnoa::cast_tensor(tensor_handle) = torch::randn_like(jnoa::cast_tensor(tensor_handle));
}

JNIEXPORT jlong JNICALL Java_space_kscience_kmath_noa_JNoa_randintLike
        (JNIEnv *, jclass, jlong low, jlong high, jlong tensor_handle) {
    return (long) new jnoa::Tensor(torch::randint_like(jnoa::cast_tensor(tensor_handle), low, high));
}

JNIEXPORT void JNICALL Java_space_kscience_kmath_noa_JNoa_randintLikeAssign
        (JNIEnv *, jclass, jlong low, jlong high, jlong tensor_handle) {
    jnoa::cast_tensor(tensor_handle) = torch::randint_like(jnoa::cast_tensor(tensor_handle), low, high);
}

JNIEXPORT jlong JNICALL Java_space_kscience_kmath_noa_JNoa_fullDouble
        (JNIEnv *env, jclass, jdouble value, jintArray shape, jint device) {
    const auto res =
            jnoa::safe_run<jnoa::Tensor>(env,
                                         jnoa::full<double>,
                                         value,
                                         jnoa::to_shape(env, shape),
                                         jnoa::int_to_device(device));
    return res.has_value() ? (long) new jnoa::Tensor(res.value()) : 0L;
}

JNIEXPORT jlong JNICALL Java_space_kscience_kmath_noa_JNoa_fullFloat
        (JNIEnv *env, jclass, jfloat value, jintArray shape, jint device) {
    const auto res =
            jnoa::safe_run<jnoa::Tensor>(env,
                                         jnoa::full<float>,
                                         value,
                                         jnoa::to_shape(env, shape),
                                         jnoa::int_to_device(device));
    return res.has_value() ? (long) new jnoa::Tensor(res.value()) : 0L;
}

JNIEXPORT jlong JNICALL Java_space_kscience_kmath_noa_JNoa_fullLong
        (JNIEnv *env, jclass, jlong value, jintArray shape, jint device) {
    const auto res =
            jnoa::safe_run<jnoa::Tensor>(env,
                                         jnoa::full<long>,
                                         value,
                                         jnoa::to_shape(env, shape),
                                         jnoa::int_to_device(device));
    return res.has_value() ? (long) new jnoa::Tensor(res.value()) : 0L;
}

JNIEXPORT jlong JNICALL Java_space_kscience_kmath_noa_JNoa_fullInt
        (JNIEnv *env, jclass, jint value, jintArray shape, jint device) {
    const auto res =
            jnoa::safe_run<jnoa::Tensor>(env,
                                         jnoa::full<int>,
                                         value,
                                         jnoa::to_shape(env, shape),
                                         jnoa::int_to_device(device));
    return res.has_value() ? (long) new jnoa::Tensor(res.value()) : 0L;
}

JNIEXPORT jlong JNICALL Java_space_kscience_kmath_noa_JNoa_timesDouble
        (JNIEnv *, jclass, jdouble value, jlong other) {
    return (long) new jnoa::Tensor(value * jnoa::cast_tensor(other));
}

JNIEXPORT jlong JNICALL Java_space_kscience_kmath_noa_JNoa_timesFloat
        (JNIEnv *, jclass, jfloat value, jlong other) {
    return (long) new jnoa::Tensor(value * jnoa::cast_tensor(other));
}

JNIEXPORT jlong JNICALL Java_space_kscience_kmath_noa_JNoa_timesLong
        (JNIEnv *, jclass, jlong value, jlong other) {
    return (long) new jnoa::Tensor(value * jnoa::cast_tensor(other));
}

JNIEXPORT jlong JNICALL Java_space_kscience_kmath_noa_JNoa_timesInt
        (JNIEnv *, jclass, jint value, jlong other) {
    return (long) new jnoa::Tensor(value * jnoa::cast_tensor(other));
}

JNIEXPORT void JNICALL
Java_space_kscience_kmath_noa_JNoa_timesDoubleAssign
        (JNIEnv *, jclass, jdouble value, jlong other) {
    jnoa::cast_tensor(other) *= value;
}

JNIEXPORT void JNICALL
Java_space_kscience_kmath_noa_JNoa_timesFloatAssign
        (JNIEnv *, jclass, jfloat value, jlong other) {
    jnoa::cast_tensor(other) *= value;
}

JNIEXPORT void JNICALL Java_space_kscience_kmath_noa_JNoa_timesLongAssign
        (JNIEnv *, jclass, jlong value, jlong other) {
    jnoa::cast_tensor(other) *= value;
}

JNIEXPORT void JNICALL Java_space_kscience_kmath_noa_JNoa_timesIntAssign
        (JNIEnv *, jclass, jint value, jlong other) {
    jnoa::cast_tensor(other) *= value;
}

JNIEXPORT jlong JNICALL Java_space_kscience_kmath_noa_JNoa_plusDouble
        (JNIEnv *, jclass, jdouble value, jlong other) {
    return (long) new jnoa::Tensor(value + jnoa::cast_tensor(other));
}

JNIEXPORT jlong JNICALL Java_space_kscience_kmath_noa_JNoa_plusFloat
        (JNIEnv *, jclass, jfloat value, jlong other) {
    return (long) new jnoa::Tensor(value + jnoa::cast_tensor(other));
}

JNIEXPORT jlong JNICALL Java_space_kscience_kmath_noa_JNoa_plusLong
        (JNIEnv *, jclass, jlong value, jlong other) {
    return (long) new jnoa::Tensor(value + jnoa::cast_tensor(other));
}

JNIEXPORT jlong JNICALL Java_space_kscience_kmath_noa_JNoa_plusInt
        (JNIEnv *, jclass, jint value, jlong other) {
    return (long) new jnoa::Tensor(value + jnoa::cast_tensor(other));
}

JNIEXPORT void JNICALL Java_space_kscience_kmath_noa_JNoa_plusDoubleAssign
        (JNIEnv *, jclass, jdouble value, jlong other) {
    jnoa::cast_tensor(other) += value;
}

JNIEXPORT void JNICALL Java_space_kscience_kmath_noa_JNoa_plusFloatAssign
        (JNIEnv *, jclass, jfloat value, jlong other) {
    jnoa::cast_tensor(other) += value;
}

JNIEXPORT void JNICALL Java_space_kscience_kmath_noa_JNoa_plusLongAssign
        (JNIEnv *, jclass, jlong value, jlong other) {
    jnoa::cast_tensor(other) += value;
}

JNIEXPORT void JNICALL Java_space_kscience_kmath_noa_JNoa_plusIntAssign
        (JNIEnv *, jclass, jint value, jlong other) {
    jnoa::cast_tensor(other) += value;
}

JNIEXPORT jlong JNICALL Java_space_kscience_kmath_noa_JNoa_timesTensor
        (JNIEnv *env, jclass, jlong lhs, jlong rhs) {
    const auto res =
            jnoa::safe_run<jnoa::Tensor>(env,
                                         [](const auto &lhs, const auto &rhs) {
                                             return lhs * rhs;
                                         },
                                         jnoa::cast_tensor(lhs),
                                         jnoa::cast_tensor(rhs));
    return res.has_value() ? (long) new jnoa::Tensor(res.value()) : 0L;
}

JNIEXPORT void JNICALL Java_space_kscience_kmath_noa_JNoa_timesTensorAssign
        (JNIEnv *env, jclass, jlong lhs, jlong rhs) {
    jnoa::safe_run(env,
                   [](auto &lhs, const auto &rhs) {
                       lhs *= rhs;
                   },
                   jnoa::cast_tensor(lhs),
                   jnoa::cast_tensor(rhs));
}

JNIEXPORT jlong JNICALL Java_space_kscience_kmath_noa_JNoa_divTensor
        (JNIEnv *env, jclass, jlong lhs, jlong rhs) {
    const auto res =
            jnoa::safe_run<jnoa::Tensor>(env,
                                         [](const auto &lhs, const auto &rhs) {
                                             return lhs / rhs;
                                         },
                                         jnoa::cast_tensor(lhs),
                                         jnoa::cast_tensor(rhs));
    return res.has_value() ? (long) new jnoa::Tensor(res.value()) : 0L;
}

JNIEXPORT void JNICALL Java_space_kscience_kmath_noa_JNoa_divTensorAssign
        (JNIEnv *env, jclass, jlong lhs, jlong rhs) {
    jnoa::safe_run(env,
                   [](auto &lhs, const auto &rhs) {
                       lhs /= rhs;
                   },
                   jnoa::cast_tensor(lhs),
                   jnoa::cast_tensor(rhs));
}

JNIEXPORT jlong JNICALL Java_space_kscience_kmath_noa_JNoa_plusTensor
        (JNIEnv *env, jclass, jlong lhs, jlong rhs) {
    const auto res =
            jnoa::safe_run<jnoa::Tensor>(env,
                                         [](const auto &lhs, const auto &rhs) {
                                             return lhs + rhs;
                                         },
                                         jnoa::cast_tensor(lhs),
                                         jnoa::cast_tensor(rhs));
    return res.has_value() ? (long) new jnoa::Tensor(res.value()) : 0L;
}

JNIEXPORT void JNICALL Java_space_kscience_kmath_noa_JNoa_plusTensorAssign
        (JNIEnv *env, jclass, jlong lhs, jlong rhs) {
    jnoa::safe_run(env,
                   [](auto &lhs, const auto &rhs) {
                       lhs += rhs;
                   },
                   jnoa::cast_tensor(lhs),
                   jnoa::cast_tensor(rhs));
}

JNIEXPORT jlong JNICALL Java_space_kscience_kmath_noa_JNoa_minusTensor
        (JNIEnv *env, jclass, jlong lhs, jlong rhs) {
    const auto res =
            jnoa::safe_run<jnoa::Tensor>(env,
                                         [](const auto &lhs, const auto &rhs) {
                                             return lhs - rhs;
                                         },
                                         jnoa::cast_tensor(lhs),
                                         jnoa::cast_tensor(rhs));
    return res.has_value() ? (long) new jnoa::Tensor(res.value()) : 0L;
}

JNIEXPORT void JNICALL Java_space_kscience_kmath_noa_JNoa_minusTensorAssign
        (JNIEnv *env, jclass, jlong lhs, jlong rhs) {
    jnoa::safe_run(env,
                   [](auto &lhs, const auto &rhs) {
                       lhs -= rhs;
                   },
                   jnoa::cast_tensor(lhs),
                   jnoa::cast_tensor(rhs));
}

JNIEXPORT jlong JNICALL Java_space_kscience_kmath_noa_JNoa_unaryMinus
        (JNIEnv *, jclass, jlong tensor_handle) {
    return (long) new jnoa::Tensor(-jnoa::cast_tensor(tensor_handle));
}

JNIEXPORT jlong JNICALL Java_space_kscience_kmath_noa_JNoa_transposeTensor
        (JNIEnv *env, jclass, jlong tensor_handle, jint i, jint j) {
    const auto res =
            jnoa::safe_run<jnoa::Tensor>(env,
                                         [](const auto &tensor, const int i, const int j) {
                                             return tensor.transpose(i, j);
                                         },
                                         jnoa::cast_tensor(tensor_handle),
                                         i, j);
    return res.has_value() ? (long) new jnoa::Tensor(res.value()) : 0L;
}

JNIEXPORT jlong JNICALL Java_space_kscience_kmath_noa_JNoa_absTensor
        (JNIEnv *, jclass, jlong tensor_handle) {
    return (long) new jnoa::Tensor(jnoa::cast_tensor(tensor_handle).abs());
}

JNIEXPORT jlong JNICALL Java_space_kscience_kmath_noa_JNoa_expTensor
        (JNIEnv *, jclass, jlong tensor_handle) {
    return (long) new jnoa::Tensor(jnoa::cast_tensor(tensor_handle).exp());
}

JNIEXPORT jlong JNICALL Java_space_kscience_kmath_noa_JNoa_lnTensor
        (JNIEnv *, jclass, jlong tensor_handle) {
    return (long) new jnoa::Tensor(jnoa::cast_tensor(tensor_handle).log());
}

JNIEXPORT jlong JNICALL Java_space_kscience_kmath_noa_JNoa_sqrtTensor
        (JNIEnv *, jclass, jlong tensor_handle) {
    return (long) new jnoa::Tensor(jnoa::cast_tensor(tensor_handle).sqrt());
}

JNIEXPORT jlong JNICALL Java_space_kscience_kmath_noa_JNoa_cosTensor
        (JNIEnv *, jclass, jlong tensor_handle) {
    return (long) new jnoa::Tensor(jnoa::cast_tensor(tensor_handle).cos());
}

JNIEXPORT jlong JNICALL Java_space_kscience_kmath_noa_JNoa_acosTensor
        (JNIEnv *, jclass, jlong tensor_handle) {
    return (long) new jnoa::Tensor(jnoa::cast_tensor(tensor_handle).acos());
}

JNIEXPORT jlong JNICALL Java_space_kscience_kmath_noa_JNoa_coshTensor
        (JNIEnv *, jclass, jlong tensor_handle) {
    return (long) new jnoa::Tensor(jnoa::cast_tensor(tensor_handle).cosh());
}

JNIEXPORT jlong JNICALL Java_space_kscience_kmath_noa_JNoa_acoshTensor
        (JNIEnv *, jclass, jlong tensor_handle) {
    return (long) new jnoa::Tensor(jnoa::cast_tensor(tensor_handle).acosh());
}

JNIEXPORT jlong JNICALL Java_space_kscience_kmath_noa_JNoa_sinTensor
        (JNIEnv *, jclass, jlong tensor_handle) {
    return (long) new jnoa::Tensor(jnoa::cast_tensor(tensor_handle).sin());
}

JNIEXPORT jlong JNICALL Java_space_kscience_kmath_noa_JNoa_sinhTensor
        (JNIEnv *, jclass, jlong tensor_handle) {
    return (long) new jnoa::Tensor(jnoa::cast_tensor(tensor_handle).sinh());
}

JNIEXPORT jlong JNICALL Java_space_kscience_kmath_noa_JNoa_asinhTensor
        (JNIEnv *, jclass, jlong tensor_handle) {
    return (long) new jnoa::Tensor(jnoa::cast_tensor(tensor_handle).asinh());
}

JNIEXPORT jlong JNICALL Java_space_kscience_kmath_noa_JNoa_tanTensor
        (JNIEnv *, jclass, jlong tensor_handle) {
    return (long) new jnoa::Tensor(jnoa::cast_tensor(tensor_handle).tan());
}

JNIEXPORT jlong JNICALL Java_space_kscience_kmath_noa_JNoa_atanTensor
        (JNIEnv *, jclass, jlong tensor_handle) {
    return (long) new jnoa::Tensor(jnoa::cast_tensor(tensor_handle).atan());
}

JNIEXPORT jlong JNICALL Java_space_kscience_kmath_noa_JNoa_tanhTensor
        (JNIEnv *, jclass, jlong tensor_handle) {
    return (long) new jnoa::Tensor(jnoa::cast_tensor(tensor_handle).tanh());
}

JNIEXPORT jlong JNICALL Java_space_kscience_kmath_noa_JNoa_atanhTensor
        (JNIEnv *, jclass, jlong tensor_handle) {
    return (long) new jnoa::Tensor(jnoa::cast_tensor(tensor_handle).atanh());
}

JNIEXPORT jlong JNICALL Java_space_kscience_kmath_noa_JNoa_ceilTensor
        (JNIEnv *, jclass, jlong tensor_handle) {
    return (long) new jnoa::Tensor(jnoa::cast_tensor(tensor_handle).ceil());
}

JNIEXPORT jlong JNICALL Java_space_kscience_kmath_noa_JNoa_floorTensor
        (JNIEnv *, jclass, jlong tensor_handle) {
    return (long) new jnoa::Tensor(jnoa::cast_tensor(tensor_handle).floor());
}

JNIEXPORT jlong JNICALL Java_space_kscience_kmath_noa_JNoa_sumTensor
        (JNIEnv *, jclass, jlong tensor_handle) {
    return (long) new jnoa::Tensor(jnoa::cast_tensor(tensor_handle).sum());
}

JNIEXPORT jlong JNICALL Java_space_kscience_kmath_noa_JNoa_sumDimTensor
        (JNIEnv *env, jclass, jlong tensor_handle, jint dim, jboolean keep) {
    const auto res =
            jnoa::safe_run<jnoa::Tensor>(env,
                                         [](const auto &tensor, const int i, const bool keep) {
                                             return tensor.sum(i, keep);
                                         },
                                         jnoa::cast_tensor(tensor_handle),
                                         dim, keep);
    return res.has_value() ? (long) new jnoa::Tensor(res.value()) : 0L;
}


JNIEXPORT jlong JNICALL Java_space_kscience_kmath_noa_JNoa_minTensor
        (JNIEnv *, jclass, jlong tensor_handle) {
    return (long) new jnoa::Tensor(jnoa::cast_tensor(tensor_handle).min());
}

JNIEXPORT jlong JNICALL Java_space_kscience_kmath_noa_JNoa_minDimTensor
        (JNIEnv *env, jclass, jlong tensor_handle, jint dim, jboolean keep) {
    const auto res =
            jnoa::safe_run<jnoa::Tensor>(env,
                                         [](const auto &tensor, const int i, const bool keep) {
                                             return std::get<0>(torch::min(tensor, i, keep));
                                         },
                                         jnoa::cast_tensor(tensor_handle),
                                         dim, keep);
    return res.has_value() ? (long) new jnoa::Tensor(res.value()) : 0L;
}

JNIEXPORT jlong JNICALL Java_space_kscience_kmath_noa_JNoa_maxTensor
        (JNIEnv *, jclass, jlong tensor_handle) {
    return (long) new jnoa::Tensor(jnoa::cast_tensor(tensor_handle).max());
}

JNIEXPORT jlong JNICALL Java_space_kscience_kmath_noa_JNoa_maxDimTensor
        (JNIEnv *env, jclass, jlong tensor_handle, jint dim, jboolean keep) {
    const auto res =
            jnoa::safe_run<jnoa::Tensor>(env,
                                         [](const auto &tensor, const int i, const bool keep) {
                                             return std::get<0>(torch::max(tensor, i, keep));
                                         },
                                         jnoa::cast_tensor(tensor_handle),
                                         dim, keep);
    return res.has_value() ? (long) new jnoa::Tensor(res.value()) : 0L;
}

JNIEXPORT jlong JNICALL Java_space_kscience_kmath_noa_JNoa_meanTensor
        (JNIEnv *, jclass, jlong tensor_handle) {
    return (long) new jnoa::Tensor(jnoa::cast_tensor(tensor_handle).mean());
}

JNIEXPORT jlong JNICALL Java_space_kscience_kmath_noa_JNoa_meanDimTensor
        (JNIEnv *env, jclass, jlong tensor_handle, jint dim, jboolean keep) {
    const auto res =
            jnoa::safe_run<jnoa::Tensor>(env,
                                         [](const auto &tensor, const int i, const bool keep) {
                                             return tensor.mean(i, keep);
                                         },
                                         jnoa::cast_tensor(tensor_handle),
                                         dim, keep);
    return res.has_value() ? (long) new jnoa::Tensor(res.value()) : 0L;
}

JNIEXPORT jlong JNICALL Java_space_kscience_kmath_noa_JNoa_stdTensor
        (JNIEnv *, jclass, jlong tensor_handle) {
    return (long) new jnoa::Tensor(jnoa::cast_tensor(tensor_handle).std());
}

JNIEXPORT jlong JNICALL Java_space_kscience_kmath_noa_JNoa_stdDimTensor
        (JNIEnv *env, jclass, jlong tensor_handle, jint dim, jboolean keep) {
    const auto res =
            jnoa::safe_run<jnoa::Tensor>(env,
                                         [](const auto &tensor, const int i, const bool keep) {
                                             return tensor.std(i, keep);
                                         },
                                         jnoa::cast_tensor(tensor_handle),
                                         dim, keep);
    return res.has_value() ? (long) new jnoa::Tensor(res.value()) : 0L;
}


JNIEXPORT jlong JNICALL Java_space_kscience_kmath_noa_JNoa_varTensor
        (JNIEnv *, jclass, jlong tensor_handle) {
    return (long) new jnoa::Tensor(jnoa::cast_tensor(tensor_handle).var());
}

JNIEXPORT jlong JNICALL Java_space_kscience_kmath_noa_JNoa_varDimTensor
        (JNIEnv *env, jclass, jlong tensor_handle, jint dim, jboolean keep) {
    const auto res =
            jnoa::safe_run<jnoa::Tensor>(env,
                                         [](const auto &tensor, const int i, const bool keep) {
                                             return tensor.var(i, keep);
                                         },
                                         jnoa::cast_tensor(tensor_handle),
                                         dim, keep);
    return res.has_value() ? (long) new jnoa::Tensor(res.value()) : 0L;
}

JNIEXPORT jlong JNICALL Java_space_kscience_kmath_noa_JNoa_argMaxTensor
        (JNIEnv *env, jclass, jlong tensor_handle, jint dim, jboolean keep) {
    const auto res =
            jnoa::safe_run<jnoa::Tensor>(env,
                                         [](const auto &tensor, const int i, const bool keep) {
                                             return tensor.argmax(i, keep);
                                         },
                                         jnoa::cast_tensor(tensor_handle),
                                         dim, keep);
    return res.has_value() ? (long) new jnoa::Tensor(res.value()) : 0L;
}

JNIEXPORT jlong JNICALL Java_space_kscience_kmath_noa_JNoa_flattenTensor
        (JNIEnv *env, jclass, jlong tensor_handle, jint i, jint j) {
    const auto res =
            jnoa::safe_run<jnoa::Tensor>(env,
                                         [](const auto &tensor, const int i, const int j) {
                                             return tensor.flatten(i, j);
                                         },
                                         jnoa::cast_tensor(tensor_handle),
                                         i, j);
    return res.has_value() ? (long) new jnoa::Tensor(res.value()) : 0L;
}

JNIEXPORT jlong JNICALL Java_space_kscience_kmath_noa_JNoa_matmul
        (JNIEnv *env, jclass, jlong lhs, jlong rhs) {
    const auto res =
            jnoa::safe_run<jnoa::Tensor>(env,
                                         [](const auto &lhs, const auto &rhs) {
                                             return lhs.matmul(rhs);
                                         },
                                         jnoa::cast_tensor(lhs),
                                         jnoa::cast_tensor(rhs));
    return res.has_value() ? (long) new jnoa::Tensor(res.value()) : 0L;
}

JNIEXPORT void JNICALL Java_space_kscience_kmath_noa_JNoa_matmulAssign
        (JNIEnv *env, jclass, jlong lhs, jlong rhs) {
    jnoa::safe_run(env,
                   [](auto &lhs, const auto &rhs) {
                       lhs = lhs.matmul(rhs);
                   },
                   jnoa::cast_tensor(lhs),
                   jnoa::cast_tensor(rhs));
}

JNIEXPORT void JNICALL Java_space_kscience_kmath_noa_JNoa_matmulRightAssign
        (JNIEnv *env, jclass, jlong lhs, jlong rhs) {
    jnoa::safe_run(env,
                   [](const auto &lhs, auto &rhs) {
                       rhs = lhs.matmul(rhs);
                   },
                   jnoa::cast_tensor(lhs),
                   jnoa::cast_tensor(rhs));
}

JNIEXPORT jlong JNICALL Java_space_kscience_kmath_noa_JNoa_diagEmbed
        (JNIEnv *env, jclass, jlong diags_handle, jint offset, jint dim1, jint dim2){
    const auto res =
            jnoa::safe_run<jnoa::Tensor>(env,
                                         [](const auto &diag_tensor, const int offset,
                                                 const int dim1, const int dim2) {
                                             return torch::diag_embed(diag_tensor, offset, dim1, dim2);
                                         },
                                         jnoa::cast_tensor(diags_handle),
                                         offset, dim1, dim2);
    return res.has_value() ? (long) new jnoa::Tensor(res.value()) : 0L;
}

JNIEXPORT jlong JNICALL Java_space_kscience_kmath_noa_JNoa_detTensor
        (JNIEnv *env, jclass, jlong tensor_handle){
    const auto res =
            jnoa::safe_run<jnoa::Tensor>(env,
                                         [](const auto &tensor) {
                                             return torch::linalg::det(tensor);
                                         },
                                         jnoa::cast_tensor(tensor_handle));
    return res.has_value() ? (long) new jnoa::Tensor(res.value()) : 0L;
}

JNIEXPORT jlong JNICALL Java_space_kscience_kmath_noa_JNoa_invTensor
        (JNIEnv *env, jclass, jlong tensor_handle){
    const auto res =
            jnoa::safe_run<jnoa::Tensor>(env,
                                         [](const auto &tensor) {
                                             return torch::linalg::inv(tensor);
                                         },
                                         jnoa::cast_tensor(tensor_handle));
    return res.has_value() ? (long) new jnoa::Tensor(res.value()) : 0L;
}

JNIEXPORT jlong JNICALL Java_space_kscience_kmath_noa_JNoa_choleskyTensor
        (JNIEnv *env, jclass, jlong tensor_handle){
    const auto res =
            jnoa::safe_run<jnoa::Tensor>(env,
                                         [](const auto &tensor) {
                                             return torch::linalg::cholesky(tensor);
                                         },
                                         jnoa::cast_tensor(tensor_handle));
    return res.has_value() ? (long) new jnoa::Tensor(res.value()) : 0L;
}

