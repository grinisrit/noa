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

#include <filesystem>
#include <fstream>
#include <iostream>
#include <optional>
#include <regex>

#include <torch/torch.h>

namespace noa::utils
{

    using Path = std::filesystem::path;
    using Status = bool;
    using Line = std::string;
    using TensorOpt = std::optional<torch::Tensor>;

    constexpr double TOLERANCE = 1E-6;
    constexpr int SEED = 987654;

    inline const auto num_pattern = std::regex{"[0-9.E\\+-]+"};

    inline Status check_path_exists(const Path &path)
    {
        if (!std::filesystem::exists(path))
        {
            std::cerr << "Cannot find " << path << std::endl;
            return false;
        }
        return true;
    }

    inline TensorOpt load_tensor(const Path &path)
    {
        if (check_path_exists(path))
        {
            auto tensor = torch::Tensor{};
            try
            { 
                torch::load(tensor, path);
            }
            catch (...)
            {
                std::cerr << "Failed to load tensor from " << path << "\n";
                return std::nullopt;
            }
            return std::make_optional(tensor);
        }
        else
        {
            return std::nullopt;
        }
    }

    inline std::optional<Line> find_line(
        std::ifstream &stream, const std::regex &line_pattern)
    {
        auto line = Line{};
        while (std::getline(stream, line))
            if (std::regex_search(line, line_pattern))
                return line;
        return std::nullopt;
    }

    template <typename Dtype>
    inline std::optional<std::vector<Dtype>> get_numerics(
        const std::string &line, int size)
    {
        auto no_data = std::sregex_iterator();
        auto nums = std::sregex_iterator(line.begin(), line.end(), num_pattern);
        if (std::distance(nums, no_data) != size)
            return std::nullopt;

        auto vec = std::vector<Dtype>{};
        vec.reserve(size);
        for (auto &num = nums; num != no_data; num++)
            vec.emplace_back(std::stod(num->str()));
        return vec;
    }

    template <typename Dtype, typename Lambda>
    inline torch::Tensor vmap(const torch::Tensor &values, const Lambda &lambda)
    {
        const Dtype *pvals = values.data_ptr<Dtype>();
        auto result = torch::zeros_like(values);
        Dtype *pres = result.data_ptr<Dtype>();
        const int n = result.numel();
        for (int i = 0; i < n; i++)
            pres[i] = lambda(pvals[i]);
        return result;
    }

    template <typename Dtype, typename Lambda>
    inline void vmap(const torch::Tensor &values, const Lambda &lambda, const torch::Tensor &result)
    {
        const Dtype *pvals = values.data_ptr<Dtype>();
        Dtype *pres = result.data_ptr<Dtype>();
        const int n = result.numel();
        for (int i = 0; i < n; i++)
            pres[i] = lambda(pvals[i]);
    }

    template <typename Dtype, typename Lambda>
    inline torch::Tensor vmapi(const torch::Tensor &values, const Lambda &lambda)
    {
        const Dtype *pvals = values.data_ptr<Dtype>();
        auto result = torch::zeros_like(values);
        Dtype *pres = result.data_ptr<Dtype>();
        const int n = result.numel();
        for (int i = 0; i < n; i++)
            pres[i] = lambda(i, pvals[i]);
        return result;
    }

    template <typename Dtype, typename Lambda>
    inline void vmapi(const torch::Tensor &values, const Lambda &lambda, const torch::Tensor &result)
    {
        const Dtype *pvals = values.data_ptr<Dtype>();
        Dtype *pres = result.data_ptr<Dtype>();
        const int n = result.numel();
        for (int i = 0; i < n; i++)
            pres[i] = lambda(i, pvals[i]);
    }

    template <typename Dtype, typename Lambda>
    inline void for_each(const torch::Tensor &values, const Lambda &lambda)
    {
        const Dtype *pvals = values.data_ptr<Dtype>();
        const int n = values.numel();
        for (int i = 0; i < n; i++)
            lambda(pvals[i]);
    }

    template <typename Dtype, typename Lambda>
    inline void for_eachi(const torch::Tensor &values, const Lambda &lambda)
    {
        const Dtype *pvals = values.data_ptr<Dtype>();
        const int n = values.numel();
        for (int i = 0; i < n; i++)
            lambda(i, pvals[i]);
    }

    template <typename Dtype>
    inline torch::Tensor relative_error(const torch::Tensor &computed, const torch::Tensor &expected)
    {
        return torch::abs(
                   (computed - expected) / (computed + std::numeric_limits<Dtype>::min()))
                   .sum() /
               computed.numel();
    }

    template <typename Dtype>
    inline torch::Tensor mean_error(const torch::Tensor &computed, const torch::Tensor &expected)
    {
        return torch::abs(computed - expected).sum() / computed.numel();
    }

} // namespace noa::utils