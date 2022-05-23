#ifdef HAVE_GTEST
#include "gtest/gtest.h"

#include <TNL/Containers/NDArray.h>

#include <TNL/Algorithms/ParallelFor.h>

using namespace TNL;
using namespace TNL::Containers;
using std::index_sequence;

template< typename Array >
void expect_identity( const Array& a )
{
    Array identity;
    identity.setSize( a.getSize() );
    for( int i = 0; i < identity.getSize(); i++ )
        identity.setElement( i, i );
    EXPECT_EQ( a, identity );
}

// nvcc fuck-up: __host__ __device__ lambdas cannot be inside protected/private class methods
void __test_SetThroughView()
{
    constexpr int I = 3, J = 5;
    using ViewType = typename StaticNDArray< int, SizesHolder< int, I, J > >::ViewType;
    NDArray< int,
             SizesHolder< int, I, J >,
             std::make_index_sequence< 2 >,
             TNL::Devices::Cuda > a;
    a.setSizes( 0, 0 );
    ViewType a_view( a.getStorageArray().getData(), SizesHolder< int, I, J >{} );

    auto kernel = [] __cuda_callable__ ( int, ViewType a ) {
        int v = 0;
        for( int i = 0; i < I; i++ )
            for( int j = 0; j < J; j++ )
                a( i, j ) = v++;
    };

    a.setValue(0);
    Algorithms::ParallelFor< TNL::Devices::Cuda >::exec( 0, 1, kernel, a_view );
    expect_identity( a.getStorageArray() );
}
TEST( StaticNDArrayCudaTest, SetThroughView )
{
    __test_SetThroughView();
}

// nvcc fuck-up: __host__ __device__ lambdas cannot be inside protected/private class methods
void __test_CopyFromArray()
{
    constexpr int I = 3, J = 5;
    using ViewType = typename StaticNDArray< int, SizesHolder< int, I, J > >::ViewType;
    NDArray< int,
             SizesHolder< int, I, J >,
             std::make_index_sequence< 2 >,
             TNL::Devices::Cuda > a;
    a.setSizes( 0, 0 );
    ViewType a_view( a.getStorageArray().getData(), SizesHolder< int, I, J >{} );

    auto kernel = [] __cuda_callable__ ( int, ViewType a ) {
        StaticNDArray< int, SizesHolder< int, I, J > > b;
        int v = 0;
        for( int i = 0; i < I; i++ )
            for( int j = 0; j < J; j++ )
                b( i, j ) = v++;
        a = b.getView();
        a( 0, 0 ) = a != b.getView();
    };

    a.setValue(0);
    Algorithms::ParallelFor< TNL::Devices::Cuda >::exec( 0, 1, kernel, a_view );
    expect_identity( a.getStorageArray() );
}
TEST( StaticNDArrayCudaTest, CopyFromArray )
{
    __test_CopyFromArray();
}
#endif // HAVE_GTEST


#include "../../main.h"
