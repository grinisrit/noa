#define HAVE_CUDA

#include <TNL/Containers/Array.h>
#include <TNL/Containers/ArrayView.h>
#include <torch/torch.h>

#include <gtest/gtest.h>

using namespace TNL;
using namespace TNL::Containers;

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