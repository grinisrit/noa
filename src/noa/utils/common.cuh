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

#include <torch/types.h>

namespace noa::utils::cuda {

    constexpr uint32_t MIN_BLOCK = 32;
    constexpr uint32_t MAX_BLOCK = 1024;

    inline std::tuple<uint32_t, uint32_t> get_grid(const uint32_t num_tasks) {
        const uint32_t thread_blocks = (num_tasks + MIN_BLOCK - 1) / MIN_BLOCK;
        const uint32_t num_threads = std::min(MIN_BLOCK * thread_blocks, MAX_BLOCK);
        return std::make_tuple((num_tasks + num_threads - 1) / num_threads, num_threads);
    }

    template<typename Kernel>
    __global__ void launch_kernel(const Kernel kernel, const uint32_t num_tasks) {
        uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < num_tasks) kernel(i);
    }

    template<typename Dtype, typename Lambda>
    inline void for_eachi(const Lambda &lambda, const torch::Tensor &result) {
        Dtype *pres = result.data_ptr<Dtype>();

        const auto n = result.numel();
        const auto [blocks, threads] = get_grid(n);

        const auto lambda_kernel = [pres, lambda] __device__(const int i){
            lambda(i, pres[i]);
        };
        launch_kernel<<<blocks, threads>>>(lambda_kernel, n);
    }

    template<typename Dtype, typename Lambda>
    inline void for_each(const Lambda &lambda, const torch::Tensor &result) {
        for_eachi<Dtype>([lambda] __device__(const int, Dtype &k){lambda(k);}, result);
    }

    template<typename Dtype, typename Lambda>
    inline void vmapi(const torch::Tensor &values, const Lambda &lambda, const torch::Tensor &result) {
        const Dtype *pvals = values.data_ptr<Dtype>();
        for_eachi<Dtype>([lambda, pvals] __device__(const int i, Dtype &k){k = lambda(i, pvals[i]);}, result);
    }

    template<typename Dtype, typename Lambda>
    inline void vmap(const torch::Tensor &values, const Lambda &lambda, const torch::Tensor &result) {
        vmapi<Dtype>(values,
                     [lambda] __device__(const int, const Dtype &k){ return lambda(k);},
                     result);
    }

    template<typename Dtype, typename Lambda>
    inline torch::Tensor vmapi(const torch::Tensor &values, const Lambda &lambda) {
        const auto result = torch::zeros_like(values);
        vmapi<Dtype>(values, lambda, result);
        return result;
    }

    template<typename Dtype, typename Lambda>
    inline torch::Tensor vmap(const torch::Tensor &values, const Lambda &lambda) {
        const auto result = torch::zeros_like(values);
        vmap<Dtype>(values, lambda, result);
        return result;
    }

}