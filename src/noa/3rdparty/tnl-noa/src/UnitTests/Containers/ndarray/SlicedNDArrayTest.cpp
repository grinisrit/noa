#ifdef HAVE_GTEST
#include "gtest/gtest.h"

#include <TNL/Containers/NDArray.h>

using namespace TNL::Containers;
using std::index_sequence;

template< typename Array >
void expect_identity( const Array& a )
{
    Array identity;
    identity.setLike( a );
    for( int i = 0; i < identity.getSize(); i++ )
        identity[ i ] = i;
    EXPECT_EQ( a, identity );
}

template< typename Array, typename Seq >
void expect_seq( const Array& a, const Seq& seq )
{
    for( int i = 0; i < a.getSize(); i++ )
        EXPECT_EQ( a[ i ], seq[ i ] );
}

TEST( SlicedNDArrayTest, 2D_Static_Identity )
{
    constexpr int I = 3, J = 5;
    SlicedNDArray< int, SizesHolder< int, I, J > > a;
    a.setSizes( 0, 0 );

    int v = 0;
    for( int i = 0; i < I; i++ )
        for( int j = 0; j < J; j++ )
            a( i, j ) = v++;

    expect_identity( a.getStorageArray() );
}

TEST( SlicedNDArrayTest, 2D_Static_Permuted )
{
    constexpr int I = 3, J = 5;
    SlicedNDArray< int,
                   SizesHolder< int, I, J >,
                   index_sequence< 1, 0 > > a;
    a.setSizes( 0, 0 );

    int v = 0;
    for( int j = 0; j < J; j++ )
        for( int i = 0; i < I; i++ )
            a( i, j ) = v++;

    expect_identity( a.getStorageArray() );
}

TEST( SlicedNDArrayTest, 6D_Dynamic )
{
    int I = 2, J = 2, K = 2, L = 2, M = 2, N = 2;
    SlicedNDArray< int,
                   SizesHolder< int, 0, 0, 0, 0, 0, 0 >,
                   index_sequence< 5, 3, 4, 2, 0, 1 > > a;
    a.setSizes( I, J, K, L, M, N );

    int v = 0;
    for( int n = 0; n < N; n++ )
        for( int l = 0; l < L; l++ )
            for( int m = 0; m < M; m++ )
                for( int k = 0; k < K; k++ )
                    for( int i = 0; i < I; i++ )
                        for( int j = 0; j < J; j++ )
                            a( i, j, k, l, m, n ) = v++;

    expect_identity( a.getStorageArray() );
}


TEST( SlicedNDArrayTest, Sliced2D_Dynamic_Identity )
{
    const int I = 3, J = 5;
    SlicedNDArray< int,
                   SizesHolder< int, 0, 0 >,
                   index_sequence< 0, 1 >,
                   SliceInfo< 1, 2 > > a;  // J is sliced
    a.setSizes( I, J );

    a.getStorageArray().setValue(-1);
    int v = 0;
    for( int i = 0; i < I; i++ )
        for( int j = 0; j < J; j++ )
            a( i, j ) = v++;

    const std::vector< int > seq({
            // first slice
            0, 1,
            5, 6,
            10, 11,
            // second slice
            2, 3,
            7, 8,
            12, 13,
            // third slice
            4, -1,
            9, -1,
            14, -1,
        });
    expect_seq( a.getStorageArray(), seq );
}

TEST( SlicedNDArrayTest, Sliced2D_HalfStatic_Identity )
{
    constexpr int I = 3;
    const int J = 5;
    SlicedNDArray< int,
                   SizesHolder< int, I, 0 >,
                   index_sequence< 0, 1 >,
                   SliceInfo< 1, 2 > > a;  // J is sliced
    a.setSizes( 0, J );

    a.getStorageArray().setValue(-1);
    int v = 0;
    for( int i = 0; i < I; i++ )
        for( int j = 0; j < J; j++ )
            a( i, j ) = v++;

    const std::vector< int > seq({
            // first slice
            0, 1,
            5, 6,
            10, 11,
            // second slice
            2, 3,
            7, 8,
            12, 13,
            // third slice
            4, -1,
            9, -1,
            14, -1,
        });
    expect_seq( a.getStorageArray(), seq );
}

TEST( SlicedNDArrayTest, Sliced2D_Dynamic_Permuted )
{
    const int I = 3, J = 5;
    SlicedNDArray< int,
                   SizesHolder< int, 0, 0 >,
                   index_sequence< 1, 0 >,
                   SliceInfo< 0, 2 > > a;  // I is sliced
    a.setSizes( I, J );

    a.getStorageArray().setValue(-1);
    int v = 0;
    for( int j = 0; j < J; j++ )
        for( int i = 0; i < I; i++ )
            a( i, j ) = v++;

    const std::vector< int > seq({
            // first slice (transposed)
            0, 1,
            3, 4,
            6, 7,
            9, 10,
            12, 13,
            // second slice (transposed)
            2, -1,
            5, -1,
            8, -1,
            11, -1,
            14, -1,
        });
    expect_seq( a.getStorageArray(), seq );
}

TEST( SlicedNDArrayTest, Sliced2D_HalfStatic_Permuted )
{
    const int I = 3;
    constexpr int J = 5;
    SlicedNDArray< int,
                   SizesHolder< int, 0, J >,
                   index_sequence< 1, 0 >,
                   SliceInfo< 0, 2 > > a;  // I is sliced
    a.setSizes( I, 0 );

    a.getStorageArray().setValue(-1);
    int v = 0;
    for( int j = 0; j < J; j++ )
        for( int i = 0; i < I; i++ )
            a( i, j ) = v++;

    const std::vector< int > seq({
            // first slice (transposed)
            0, 1,
            3, 4,
            6, 7,
            9, 10,
            12, 13,
            // second slice (transposed)
            2, -1,
            5, -1,
            8, -1,
            11, -1,
            14, -1,
        });
    expect_seq( a.getStorageArray(), seq );
}


TEST( SlicedNDArrayTest, CopySemantics )
{
    const int I = 3, J = 4;
    SlicedNDArray< int,
                   SizesHolder< int, 0, 0 >,
                   index_sequence< 0, 1 >,
                   SliceInfo< 1, 2 > > a, b, c;  // J is sliced
    a.setSizes( I, J );

    int v = 0;
    for( int i = 0; i < I; i++ )
    for( int j = 0; j < J; j++ )
        a( i, j ) = v++;

    b = a;
    EXPECT_EQ( a, b );

    auto a_view = a.getView();
    auto b_view = b.getView();
    EXPECT_EQ( a_view, b_view );
    EXPECT_EQ( a_view.getView(), b_view );
    EXPECT_EQ( a_view.getConstView(), b_view.getConstView() );
    EXPECT_EQ( a.getConstView(), b.getConstView() );
    EXPECT_EQ( a.getConstView(), b_view.getConstView() );

    c.setSizes( I, J );
    auto c_view = c.getView();
    c_view = b_view;
    EXPECT_EQ( a_view, c_view );
    EXPECT_EQ( a_view.getView(), c_view );
    EXPECT_EQ( a_view.getConstView(), c_view.getConstView() );
    EXPECT_EQ( a.getConstView(), c.getConstView() );
    EXPECT_EQ( a.getConstView(), c_view.getConstView() );
}
#endif // HAVE_GTEST


#include "../../main.h"
