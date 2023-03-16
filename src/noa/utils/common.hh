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
 * \file common.hh
 * Implemented by: Roland Grinis
 */

#pragma once

#include <filesystem>
#include <fstream>
#include <iostream>
#include <optional>
#include <regex>

#include <torch/torch.h>
#include <torch/script.h>

/// NOA utility namespace
namespace noa::utils {

    /// Path type
    using Path = std::filesystem::path;
    using Status = bool;
    using Line = std::string;
    using Tensor = torch::Tensor;
    using TensorOpt = std::optional<Tensor>;
    using Tensors = std::vector<Tensor>;
    using TensorsOpt = std::optional<Tensors>;
    using NamedTensors = std::unordered_map<std::string, Tensor>;
    using ScriptModule = torch::jit::Module;
    using ScriptModuleOpt = std::optional<ScriptModule>;
    using OutputLeaf = Tensor;
    using InputLeaves = Tensors;
    using ADGraph = std::tuple<OutputLeaf, InputLeaves>;

    constexpr double_t TOLERANCE = 1E-6;
    constexpr int32_t SEED = 987654;

    inline const auto num_pattern = std::regex{"[0-9.E\\+-]+"};

    [[nodiscard]] inline Status check_path_exists(const Path &path) {
        if (!std::filesystem::exists(path)) {
            std::cerr << "Cannot find " << path << std::endl;
            return false;
        }
        return true;
    }

    /// Return type for \ref check_directory()
    enum class DirectoryStatus {
            /// Nothing at the specified path
            NOT_FOUND,
            /// Path points to a file that is not a directory
            NOT_A_DIR,
            /// Path points to an existing directory, OK
            EXISTS
    };

    /// Checks if specified path points to an existing directory in the filesystem
    ///
    /// \returns See DirectoryStatus
    [[nodiscard]] inline DirectoryStatus check_directory(const Path &path) {
            if (!check_path_exists(path)) return DirectoryStatus::NOT_FOUND;

            if (!std::filesystem::is_directory(path)) {
                    std::cerr << path << ": not a directory" << std::endl;
                    return DirectoryStatus::NOT_A_DIR;
            }

            return DirectoryStatus::EXISTS;
    }

    /// Bitwise comparison of files
    ///
    /// \returns `true` if files are equal, `false` if differ
    [[nodiscard]] inline Status compare_files(const Path& file1, const Path& file2) {
        std::ifstream file1_stream(file1, std::ios::binary), file2_stream(file2, std::ios::binary);

        if (file1_stream.fail() || file2_stream.fail()) return false;

        char c1, c2;
        while (bool(file1_stream.read(&c1, 1)) & bool(file2_stream.read(&c2, 1))) // & instead of && to ensure both sides get executed
                if (c1 != c2) return false;

        return file1_stream.eof() && file2_stream.eof(); // Check if reached EOF in both files
    }

    inline TensorOpt load_tensor(const Path &path) {
        if (check_path_exists(path)) {
            auto tensor = Tensor{};
            try {
                torch::load(tensor, path);
            }
            catch (const std::exception &exc) {
                std::cerr << "Failed to load tensor from " << path << "\n" << exc.what() << "\n";
                return std::nullopt;
            }
            return std::make_optional(tensor);
        } else {
            return std::nullopt;
        }
    }

    inline std::optional<Line> find_line(
            std::ifstream &stream, const std::regex &line_pattern) {
        auto line = Line{};
        while (std::getline(stream, line))
            if (std::regex_search(line, line_pattern))
                return line;
        return std::nullopt;
    }

    template<typename Dtype>
    inline std::optional<std::vector<Dtype>> get_numerics(
            const std::string &line, int64_t size) {
        const auto no_data = std::sregex_iterator();
        auto nums = std::sregex_iterator(line.begin(), line.end(), num_pattern);
        if (std::distance(nums, no_data) != size)
            return std::nullopt;

        auto vec = std::vector<Dtype>{};
        vec.reserve(size);
        for (auto &num = nums; num != no_data; num++)
            vec.emplace_back(std::stod(num->str()));
        return vec;
    }

    template<typename Dtype, typename Lambda>
    inline void for_eachi(const Lambda &lambda, const Tensor &result) {
        Dtype *pres = result.data_ptr<Dtype>();
        const int64_t n = result.numel();
        for (int64_t i = 0; i < n; i++)
            lambda(i, pres[i]);
    }

    template<typename Dtype, typename Lambda>
    inline void pfor_eachi(const Lambda &lambda, const Tensor &result) {
        Dtype *pres = result.data_ptr<Dtype>();
        const int64_t n = result.numel();
#pragma omp parallel for default(none) shared(n, lambda, pres)
        for (int64_t i = 0; i < n; i++)
            lambda(i, pres[i]);
    }

    template<typename Dtype, typename Lambda>
    inline void for_each(const Lambda &lambda, const Tensor &result) {
        for_eachi<Dtype>([&lambda](const int64_t, Dtype &k) { lambda(k); }, result);
    }

    template<typename Dtype, typename Lambda>
    inline void pfor_each(const Lambda &lambda, const Tensor &result) {
        pfor_eachi<Dtype>([&lambda](const int64_t, Dtype &k) { lambda(k); }, result);
    }

    template<typename Dtype, typename Lambda>
    inline void vmapi(const Tensor &values, const Lambda &lambda, const Tensor &result) {
        const Dtype *pvals = values.data_ptr<Dtype>();
        for_eachi<Dtype>([&lambda, &pvals](const int64_t i, Dtype &k) { k = lambda(i, pvals[i]); }, result);
    }

    template<typename Dtype, typename Lambda>
    inline void pvmapi(const Tensor &values, const Lambda &lambda, const Tensor &result) {
        const Dtype *pvals = values.data_ptr<Dtype>();
        pfor_eachi<Dtype>([&lambda, &pvals](const int64_t i, Dtype &k) { k = lambda(i, pvals[i]); }, result);
    }


    template<typename Dtype, typename Lambda>
    inline void vmap(const Tensor &values, const Lambda &lambda, const Tensor &result) {
        vmapi<Dtype>(values,
                     [&lambda](const int64_t, const Dtype &k) { return lambda(k); },
                     result);
    }

    template<typename Dtype, typename Lambda>
    inline void pvmap(const Tensor &values, const Lambda &lambda, const Tensor &result) {
        pvmapi<Dtype>(values,
                      [&lambda](const int64_t, const Dtype &k) { return lambda(k); },
                      result);
    }

    template<typename Dtype, typename Lambda>
    inline Tensor vmapi(const Tensor &values, const Lambda &lambda) {
        const auto result = torch::zeros_like(values);
        vmapi<Dtype>(values, lambda, result);
        return result;
    }

    template<typename Dtype, typename Lambda>
    inline Tensor pvmapi(const Tensor &values, const Lambda &lambda) {
        const auto result = torch::zeros_like(values);
        pvmapi<Dtype>(values, lambda, result);
        return result;
    }

    template<typename Dtype, typename Lambda>
    inline Tensor vmap(const Tensor &values, const Lambda &lambda) {
        const auto result = torch::zeros_like(values);
        vmap<Dtype>(values, lambda, result);
        return result;
    }

    template<typename Dtype, typename Lambda>
    inline Tensor pvmap(const Tensor &values, const Lambda &lambda) {
        const auto result = torch::zeros_like(values);
        pvmap<Dtype>(values, lambda, result);
        return result;
    }

    inline Tensor relative_error(const Tensor &computed, const Tensor &expected) {
        auto res = Tensor{};
        AT_DISPATCH_FLOATING_TYPES(computed.scalar_type(), "relative_error", [&] {
            res = torch::abs(
                    (computed - expected) / (computed + std::numeric_limits<scalar_t>::min()))
                          .mean();
        });
        return res;
    }

    inline Tensor mean_error(const Tensor &computed, const Tensor &expected) {
        return torch::abs(computed - expected).mean();
    }

    template<typename NetData>
    inline Tensors to_tensors(const NetData &net_data, bool copy) {
        auto res = Tensors{};
        for (const auto &val : net_data)
            res.push_back(copy ? val.detach().clone() : val);
        return res;
    }


    template<typename NetNamedData>
    inline NamedTensors to_named_tensors(const NetNamedData &net_named_data, bool copy) {
        auto res = NamedTensors{};
        for (const auto &[name, val] : net_named_data)
            res[name] = copy ? val.detach().clone() : val;
        return res;
    }

    template<typename Net>
    inline Tensors parameters(const Net &net, bool copy = false) {
        return to_tensors(net.parameters(), copy);
    }

    template<typename Net>
    inline Tensors buffers(const Net &net, bool copy = false) {
        return to_tensors(net.buffers(), copy);
    }

    template<typename Net>
    inline NamedTensors named_parameters(const Net &net, bool copy = false) {
        return to_named_tensors(net.named_parameters(), copy);
    }

    template<typename Net>
    inline NamedTensors named_buffers(const Net &net, bool copy = false) {
        return to_named_tensors(net.named_buffers(), copy);
    }

    template<typename NetData>
    inline Tensor flat_data(const NetData &net_data, bool detach) {
        auto res = Tensors{};
        for (const auto &val : net_data)
            res.push_back((detach ? val.detach() : val).flatten());
        return torch::cat(res);
    }

    template<typename Net>
    inline Tensor flat_parameters(const Net &net, bool detach = true) {
        return flat_data(net.parameters(), detach);
    }

    template<typename Net>
    inline Tensor flat_buffers(const Net &net, bool detach = true) {
        return flat_data(net.buffers(), detach);
    }

    template<typename NetData>
    inline bool set_data(const NetData &net_data, const Tensors &data, bool copy) {
        uint32_t i = 0;
        for (const auto &val : net_data) {
            val.set_data(copy ? data.at(i).detach().clone() : data.at(i));
            i++;
        }
        return true;
    }

    template<typename Net>
    inline bool set_parameters(Net &net, const Tensors &parameters, bool copy = false) {
        return set_data(net.parameters(), parameters, copy);
    }

    template<typename Net>
    inline bool set_buffers(Net &net, const Tensors &buffers, bool copy = false) {
        return set_data(net.buffers(), buffers, copy);
    }

    template<typename NetData>
    inline bool set_flat_data(const NetData &net_data, const Tensor &data, bool copy) {
        if (data.dim() > 1) {
            std::cerr << "Invalid arguments to noa::utils::set_flat_data : "
                      << "expecting 1D parameters\n";
            return false;
        }
        int64_t i = 0;
        for (const auto &val : net_data) {
            const auto i0 = i;
            i += val.numel();
            val.set_data(copy ? data.slice(0, i0, i).detach().clone().view_as(val)
                              : data.slice(0, i0, i).view_as(val));
        }
        return true;
    }

    template<typename Net>
    inline bool set_flat_parameters(Net &net, const Tensor &parameters, bool copy = false) {
        return set_flat_data(net.parameters(), parameters, copy);
    }

    template<typename Net>
    inline bool set_flat_buffers(Net &net, const Tensor &buffers, bool copy = false) {
        return set_flat_data(net.buffers(), buffers, copy);
    }

    inline ScriptModuleOpt load_module(const Path &jit_module_pt) {
        if (check_path_exists(jit_module_pt)) {
            try {
                return torch::jit::load(jit_module_pt);
            }
            catch (const std::exception &exc) {
                std::cerr << "Failed to load JIT module from " << jit_module_pt << "\n"<< exc.what() << "\n";
                return std::nullopt;
            }
        } else {
            return std::nullopt;
        }
    }

    inline Tensor stack(const std::vector<Tensors> &vec_tensors) {
        auto result = Tensors{};
        result.reserve(vec_tensors.size());
        for (const auto &tensors: vec_tensors) {
            auto tensors_flat = Tensors{};
            tensors_flat.reserve(tensors.size());
            for (const auto &tensor: tensors)
                tensors_flat.push_back(tensor.flatten());
            result.push_back(torch::cat(tensors_flat));
        }
        return torch::stack(result);
    }

    inline Tensors zeros_like(const Tensors &tensors, bool detach = false){
        auto res = Tensors{};
        res.reserve(tensors.size());
        for(const auto &tensor: tensors)
            res.push_back( detach ? tensor.detach() : tensor);
        return res;
    }

} // namespace noa::utils
