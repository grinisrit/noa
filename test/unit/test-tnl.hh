#pragma once

#include <noa/3rdparty/TNL/Containers/Vector.h>
#include <noa/3rdparty/TNL/Algorithms/reduce.h>
#include <noa/3rdparty/TNL/Matrices/MatrixWrapping.h>
#include <noa/3rdparty/TNL/Matrices/SparseMatrix.h>
#include <noa/3rdparty/TNL/Solvers/Linear/Jacobi.h>

#include <torch/torch.h>
#include <gtest/gtest.h>

using namespace noa::TNL::Containers;
using namespace noa::TNL::Algorithms;
using namespace noa::TNL::Matrices;
using namespace noa::TNL::Solvers::Linear;

template<typename Dtype, typename Device>
void tensor_blob_test(const torch::TensorOptions &tensor_opt) {
    // Array
    Array<Dtype, Device> tnl_array(3);
    tnl_array = 5.0;

    const auto tensor = torch::from_blob(tnl_array.getData(), {3}, tensor_opt);

    ArrayView<Dtype, Device> tnl_view(tensor.template data_ptr<Dtype>(), 3);
    tensor[1] = (Dtype) 10.0;

    ASSERT_EQ(tnl_view.getElement(1), 10.0);
    ASSERT_EQ(tnl_view.getElement(2), 5.0);
}

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
void create_dense_matrix(const torch::TensorOptions &tensor_opts) {

    const auto tensor = torch::randn({3, 3}, tensor_opts);
    const auto matrix = wrapDenseMatrix<Device>(3, 3, tensor.data_ptr<Dtype>());
    tensor[1][1] = (Dtype) 10.0;
    ASSERT_EQ(matrix.getElement(1, 1), (Dtype) 10.0);
}

template<typename Dtype, typename Device>
void create_csr_matrix(const torch::TensorOptions &tensor_opts) {
    const auto crow_indices = torch::tensor({0, 2, 4},
                                            torch::dtype(torch::kInt32).device(tensor_opts.device()));
    const auto col_indices = torch::tensor({0, 1, 0, 1},
                                           torch::dtype(torch::kInt32).device(tensor_opts.device()));
    const auto values = torch::tensor({1, 2, 3, 4}, tensor_opts);
    auto csr = wrapCSRMatrix<Device>(2, 2,
                                     crow_indices.data_ptr<int>(),
                                     values.data_ptr<Dtype>(),
                                     col_indices.data_ptr<int>());

    ASSERT_EQ(csr.getElement(0, 1), (Dtype) 2.0);

    values[1] = (Dtype) 10.0;
    ASSERT_EQ(csr.getElement(0, 1), (Dtype) 10.0);
}

template<typename Dtype, typename Device>
void jacobi_test(const torch::TensorOptions &tensor_opts) {

    const int size = 5;

    const auto crow_indices = torch::tensor({0, 2, 5, 8, 11, 13},
                                            torch::dtype(torch::kInt32).device(tensor_opts.device()));
    const auto col_indices = torch::tensor({0, 1, 0, 1, 2, 1, 2, 3, 2, 3, 4, 3, 4},
                                           torch::dtype(torch::kInt32).device(tensor_opts.device()));
    const auto values =
            torch::tensor({2.5000, -1.5000, -0.5000, 2.5000, -1.5000, -0.5000,
                           2.5000, -1.5000, -0.5000, 2.5000, -1.5000, -0.5000,
                           2.5000}, tensor_opts);
    auto matrix = std::make_shared<SparseMatrixView<Dtype, Device>>(
            wrapCSRMatrix<Device>(size, size,
                                  crow_indices.data_ptr<int>(),
                                  values.data_ptr<Dtype>(),
                                  col_indices.data_ptr<int>()));


    const auto tensor_x0 = torch::ones({size}, tensor_opts);
    const auto tensor_x = torch::zeros({size}, tensor_opts);
    const auto tensor_b = torch::zeros({size}, tensor_opts);

    const auto x0 = VectorView<Dtype, Device>{tensor_x0.data_ptr<Dtype>(), size};
    const auto x = VectorView<Dtype, Device>{tensor_x.data_ptr<Dtype>(), size};
    auto b = VectorView<Dtype, Device>{tensor_b.data_ptr<Dtype>(), size};

    matrix->vectorProduct(x0, b);

    auto solver = Jacobi<SparseMatrixView<Dtype, Device>>{};
    solver.setMatrix(matrix);
    solver.solve(b, x);

    ASSERT_NEAR(torch::dist(tensor_x0, tensor_x).item<Dtype>(), 0.f, 1e-5f);

}