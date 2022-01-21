#include <TNL/Containers/Array.h>
#include <TNL/Containers/ArrayView.h>
#include <torch/torch.h>

#include <gtest/gtest.h>

using namespace TNL;
using namespace TNL::Containers;

TEST(TNL, TensorBlob){

    Array< double > tnl_array( 3 );
    tnl_array = 5.0;

    const auto tensor = torch::from_blob(tnl_array.getData(), {3}, torch::kDouble);

    ArrayView<double> tnl_view(tensor.data_ptr<double>(),3);
    tensor[1] = 10.0;

    ASSERT_EQ(tnl_view.getElement(1), 10.0);
    ASSERT_EQ(tnl_view.getElement(2), 5.0);
}
