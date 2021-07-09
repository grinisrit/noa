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
    return (long) new jnoa::Tensor(
            jnoa::from_blob<double>(
                    env->GetDoubleArrayElements(data, nullptr),
                    jnoa::to_vec_int(env->GetIntArrayElements(shape, nullptr), env->GetArrayLength(shape)),
                    jnoa::int_to_device(device)));
}

JNIEXPORT jlong JNICALL Java_space_kscience_kmath_noa_JNoa_fromBlobFloat
        (JNIEnv *env, jclass, jfloatArray data, jintArray shape, jint device) {
    return (long) new jnoa::Tensor(
            jnoa::from_blob<float>(
                    env->GetFloatArrayElements(data, nullptr),
                    jnoa::to_vec_int(env->GetIntArrayElements(shape, nullptr), env->GetArrayLength(shape)),
                    jnoa::int_to_device(device)));
}

JNIEXPORT jlong JNICALL Java_space_kscience_kmath_noa_JNoa_fromBlobLong
        (JNIEnv *env, jclass, jlongArray data, jintArray shape, jint device) {
    return (long) new jnoa::Tensor(
            jnoa::from_blob<long>(
                    env->GetLongArrayElements(data, nullptr),
                    jnoa::to_vec_int(env->GetIntArrayElements(shape, nullptr), env->GetArrayLength(shape)),
                    jnoa::int_to_device(device)));
}

JNIEXPORT jlong JNICALL Java_space_kscience_kmath_noa_JNoa_fromBlobInt
        (JNIEnv *env, jclass, jintArray data, jintArray shape, jint device) {
    return (long) new jnoa::Tensor(
            jnoa::from_blob<int>(
                    env->GetIntArrayElements(data, nullptr),
                    jnoa::to_vec_int(env->GetIntArrayElements(shape, nullptr), env->GetArrayLength(shape)),
                    jnoa::int_to_device(device)));
}

JNIEXPORT jlong JNICALL Java_space_kscience_kmath_noa_JNoa_copyTensor
        (JNIEnv *, jclass, jlong tensor_handle) {
    return (long) new jnoa::Tensor(jnoa::cast_tensor(tensor_handle).clone());
}

JNIEXPORT jlong JNICALL Java_space_kscience_kmath_noa_JNoa_copyToDevice
        (JNIEnv *, jclass, jlong tensor_handle, jint device) {
    return (long) new jnoa::Tensor(
            jnoa::cast_tensor(tensor_handle)
                    .to(jnoa::int_to_device(device), false, true));
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
    return (long) new jnoa::Tensor(
            jnoa::cast_tensor(tensor_handle)
                    .view(jnoa::to_vec_int(env->GetIntArrayElements(shape, nullptr), env->GetArrayLength(shape))));
}

JNIEXPORT jlong JNICALL Java_space_kscience_kmath_noa_JNoa_viewAsTensor
        (JNIEnv *, jclass, jlong tensor_handle, jlong as_tensor_handle) {
    return (long) new jnoa::Tensor(
            jnoa::cast_tensor(tensor_handle)
                    .view_as(jnoa::cast_tensor(as_tensor_handle)));
}

JNIEXPORT jstring JNICALL Java_space_kscience_kmath_noa_JNoa_tensorToString
        (JNIEnv *env, jclass, jlong tensor_handle){
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
        (JNIEnv *, jclass, jlong tensor_handle, jint d){
    return jnoa::cast_tensor(tensor_handle).size(d);
}

JNIEXPORT jint JNICALL Java_space_kscience_kmath_noa_JNoa_getStrideAt
        (JNIEnv *, jclass, jlong tensor_handle, jint d){
    return jnoa::cast_tensor(tensor_handle).stride(d);
}

JNIEXPORT jint JNICALL Java_space_kscience_kmath_noa_JNoa_getDevice
        (JNIEnv *, jclass, jlong tensor_handle){
    return jnoa::device_to_int(jnoa::cast_tensor(tensor_handle));
}

