#include "test-tnl.hh"

#include <noa/3rdparty/TNL/Containers/Array.h>
#include <noa/3rdparty/TNL/Containers/ArrayView.h>


#include <torch/torch.h>

#include <gtest/gtest.h>

using namespace noa::TNL;
using namespace noa::TNL::Containers;

TEST(TNL, TensorBlob){

    Array< double > tnl_array( 3 );
    tnl_array = 5.0;

    const auto tensor = torch::from_blob(tnl_array.getData(), {3}, torch::kDouble);

    ArrayView<double> tnl_view(tensor.data_ptr<double>(),3);
    tensor[1] = 10.0;

    ASSERT_EQ(tnl_view.getElement(1), 10.0);
    ASSERT_EQ(tnl_view.getElement(2), 5.0);
}

TEST(TNL, MapReduce){
    map_reduce_test<float, Devices::Host>(torch::dtype<float>());
}

TEST(TNL, SparseCSR){
    create_csr_matrix<float, Devices::Host>(torch::dtype<float>());
}