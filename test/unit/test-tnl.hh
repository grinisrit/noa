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
    const auto csr = wrapCSRMatrix<Device>(2, 2,
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

    auto matrix_ptr = std::make_shared< SparseMatrix< Dtype, Device>>();
    matrix_ptr->setDimensions( size, size );
    const auto rowCapacities_t = torch::tensor({ 1, 2, 3, 4, 5 }, tensor_opts);
    const auto rowCapacities = VectorView<Dtype, Device>{rowCapacities_t.data_ptr<Dtype>(), size};
    matrix_ptr->setRowCapacities( rowCapacities );

    auto f = [=] __cuda_callable__ ( typename SparseMatrix< Dtype, Device>::RowView& row ) mutable {
        const int rowIdx = row.getRowIndex();
        if( rowIdx == 0 )
        {
            row.setElement( 0, rowIdx,    2.5 );    // diagonal element
            row.setElement( 1, rowIdx+1, -1 );      // element above the diagonal
        }
        else if( rowIdx == size - 1 )
        {
            row.setElement( 0, rowIdx-1, -1.0 );    // element below the diagonal
            row.setElement( 1, rowIdx,    2.5 );    // diagonal element
        }
        else
        {
            row.setElement( 0, rowIdx-1, -1.0 );    // element below the diagonal
            row.setElement( 1, rowIdx,    2.5 );    // diagonal element
            row.setElement( 2, rowIdx+1, -1.0 );    // element above the diagonal
        }
    };

    matrix_ptr->forAllRows( f );

    const auto tensor_x0 = torch::ones({5}, tensor_opts);
    const auto tensor_x = torch::zeros({5}, tensor_opts);
    const auto tensor_b = torch::zeros({5}, tensor_opts);

    const auto x0 = VectorView<Dtype, Device>{tensor_x0.data_ptr<Dtype>(), 5};
    const auto x = VectorView<Dtype, Device>{tensor_x.data_ptr<Dtype>(), 5};
    auto b = VectorView<Dtype, Device>{tensor_b.data_ptr<Dtype>(), 5};

    matrix_ptr->vectorProduct(x0, b);

    auto solver = Jacobi<SparseMatrix< Dtype, Device>>{};
    solver.setMatrix(matrix_ptr);
    solver.solve( b, x );

    ASSERT_NEAR(torch::dist(tensor_x0,tensor_x).item<Dtype>(), 0.f, 1e-5f);

}