#pragma once

#include <noa/3rdparty/TNL/Containers/Vector.h>
#include <noa/3rdparty/TNL/Algorithms/reduce.h>
#include <noa/3rdparty/TNL/Matrices/MatrixWrapping.h>

#include <torch/torch.h>
#include <gtest/gtest.h>

using namespace noa::TNL::Containers;
using namespace noa::TNL::Algorithms;
using namespace noa::TNL::Matrices;

template<typename Dtype, typename Device>
Dtype map_reduce(const VectorView<Dtype, Device> &u_view) {
    auto fetch = [=] __cuda_callable__(int i) -> Dtype {
        return u_view[2 * i];
    };
    auto reduction = [] __cuda_callable__(const Dtype &a, const Dtype &b) { return a + b; };
    return reduce<Device>(0, u_view.getSize() / 2, fetch, reduction, 0.0);
}

template<typename Dtype, typename Device>
void map_reduce_test(const torch::TensorOptions &tensor_opts) {
    const int n = 100000;
    const auto tensor = torch::ones(n, tensor_opts);
    const auto vector = VectorView<Dtype, Device>{tensor.data_ptr<Dtype>(), n};
    ASSERT_EQ(map_reduce<Dtype>(vector), 50000);
}

template<typename Dtype, typename Device>
void create_csr_matrix(const torch::TensorOptions &tensor_opts) {
    const auto crow_indices = torch::tensor({0, 2, 4},
                                            torch::dtype(torch::kInt32).device(tensor_opts.device()));
    const auto col_indices = torch::tensor({0, 1, 0, 1},
                                           torch::dtype(torch::kInt32).device(tensor_opts.device()));
    const auto values = torch::tensor({1, 2, 3, 4}, tensor_opts);
    const auto csr = wrapCSRMatrix<Device>(2, 2,
                                           crow_indices.data_ptr<int>(),
                                           values.data_ptr<Dtype>(),
                                           col_indices.data_ptr<int>());

    ASSERT_EQ(csr.getElement(0, 1), 2.0);
}