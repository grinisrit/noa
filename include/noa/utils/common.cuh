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

namespace noa::utils::cuda {

    constexpr int MIN_BLOCK = 32;
    constexpr int MAX_BLOCK = 1024;

    inline std::tuple<int, int> get_grid(int num_tasks) {
        int thread_blocks = (num_tasks + MIN_BLOCK - 1) / MIN_BLOCK;
        int num_threads = std::min(MIN_BLOCK * thread_blocks, MAX_BLOCK);
        return std::make_tuple((num_tasks + num_threads - 1) / num_threads, num_threads);
    }

    template<typename Kernel>
    __global__ void launch_kernel(const Kernel kernel, const int num_tasks) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < num_tasks) kernel(i);
    }

    template<typename Dtype, typename Lambda>
    inline void vmapi(const torch::Tensor &values, const Lambda &lambda, const torch::Tensor &result) {
        const Dtype *pvals = values.data_ptr<Dtype>();
        Dtype *pres = result.data_ptr<Dtype>();

        auto n = result.numel();
        auto [blocks, threads] = get_grid(n);

        auto lambda_kernel = [pres, lambda, pvals] __device__(const int i){
            pres[i] = lambda(i, pvals[i]);
        };
        launch_kernel<<<blocks, threads>>>(lambda_kernel, n);
    }



}