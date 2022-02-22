#define HAVE_OPENMP

#include "test-tnl.hh"

#include <torch/torch.h>

#include <gtest/gtest.h>

using namespace noa::TNL;
using namespace noa::TNL::Containers;

TEST(TNL, TensorBlob){
    tensor_blob_test<float, Devices::Host>(torch::dtype<float>());
}

TEST(TNL, MapReduce){
    map_reduce_test<float, Devices::Host>(torch::dtype<float>());
}

TEST(TNL, DenseMatTransfer){
    create_dense_matrix<float, Devices::Host>(torch::dtype<float>());
}

TEST(TNL, SparseCSRTransfer){
    create_csr_matrix<float, Devices::Host>(torch::dtype<float>());
}

TEST(TNL, JacobiSolver){
    jacobi_test<float, Devices::Host>(torch::dtype<float>());
}