#define HAVE_CUDA

#include "test-tnl.hh"

#include <torch/torch.h>

#include <gtest/gtest.h>

using namespace noa::TNL;
using namespace noa::TNL::Containers;

TEST(TNL, TensorBlobCUDA) {
    tensor_blob_test<float, Devices::Cuda>(torch::dtype<float>().device(torch::kCUDA,0));
}

TEST(TNL, MapReduceCUDA) {
    map_reduce_test<float, Devices::Cuda>(torch::dtype<float>().device(torch::kCUDA,0));
}

TEST(TNL, DenseMatTransferCUDA){
    create_dense_matrix<float, Devices::Cuda>(torch::dtype<float>().device(torch::kCUDA,0));
}

TEST(TNL, SparseCSRTransferCUDA){
    create_csr_matrix<float, Devices::Cuda>(torch::dtype<float>().device(torch::kCUDA,0));
}
