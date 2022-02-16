#include <noa/3rdparty/TNL/Containers/Vector.h>
#include <noa/3rdparty/TNL/Algorithms/reduce.h>
#include <noa/3rdparty/TNL/Timer.h>

#include <torch/torch.h>
#include <gtest/gtest.h>

using namespace noa::TNL::Containers;
using namespace noa::TNL::Algorithms;

template< typename Dtype, typename Device >
Dtype map_reduce(const VectorView< Dtype, Device >& u_view )
{
    auto fetch = [=] __cuda_callable__ ( int i )-> Dtype {
            return u_view[ 2 * i ]; };
    auto reduction = [] __cuda_callable__ ( const Dtype& a, const Dtype& b ) { return a + b; };
    return reduce< Device >( 0, u_view.getSize() / 2, fetch, reduction, 0.0 );
}

template< typename Dtype, typename Device >
void map_reduce_test(const torch::TensorOptions &tensor_opts)
{
    const int n = 100000;
    const auto tensor = torch::ones(n, tensor_opts);
    const auto vector = VectorView< Dtype, Device>{tensor.data_ptr<float>(),n};
    ASSERT_EQ(map_reduce<Dtype>( vector), 50000);
}
