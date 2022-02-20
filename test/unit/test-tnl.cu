#define HAVE_CUDA

#include "test-tnl.hh"

#include <noa/3rdparty/TNL/Containers/Array.h>
#include <noa/3rdparty/TNL/Containers/ArrayView.h>

#include <torch/torch.h>

#include <gtest/gtest.h>

using namespace noa::TNL;
using namespace noa::TNL::Containers;

TEST(TNL, TensorBlobCUDA) {

    Array<float, Devices::Cuda> tnl_array(3);
    tnl_array = 5.f;

    const auto tensor = torch::from_blob(tnl_array.getData(), {3},
                                         torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));

    ArrayView<float, Devices::Cuda> tnl_view(tensor.data_ptr<float>(), 3);
    tensor[1] = 10.f;

    ASSERT_EQ(tnl_view.getElement(1), 10.f);
    ASSERT_EQ(tnl_view.getElement(2), 5.f);
}

TEST(TNL, MapReduceCUDA) {
    map_reduce_test<float, Devices::Cuda>(torch::dtype<float>().device(torch::kCUDA,0));
}

TEST(TNL, SparseCSRCUDA){
    create_csr_matrix<float, Devices::Cuda>(torch::dtype<float>().device(torch::kCUDA,0));
}