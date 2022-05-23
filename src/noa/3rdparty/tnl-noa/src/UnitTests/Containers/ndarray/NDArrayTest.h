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
    int last = 0;
    for( int i = 0; i < identity.getSize(); i++ ) {
        // skip negative/invalid entries due to alignment
        if( a[ i ] < 0 )
            identity[ i ] = a[ i ];
        else
            identity[ i ] = last++;
    }
    EXPECT_EQ( a, identity );
}

TEST( NDArrayTest, setLike )
{
    int I = 2, J = 2, K = 2, L = 2, M = 2, N = 2;
    NDArray< int,
             SizesHolder< int, 0, 0, 0, 0, 0, 0 >,
             index_sequence< 5, 3, 4, 2, 0, 1 > > a;
    a.setSizes( I, J, K, L, M, N );

    decltype(a) b;
    EXPECT_EQ( b.template getSize< 0 >(), 0 );
    EXPECT_EQ( b.template getSize< 1 >(), 0 );
    EXPECT_EQ( b.template getSize< 2 >(), 0 );
    EXPECT_EQ( b.template getSize< 3 >(), 0 );
    EXPECT_EQ( b.template getSize< 4 >(), 0 );
    EXPECT_EQ( b.template getSize< 5 >(), 0 );
    b.setLike( a );
    EXPECT_EQ( b.template getSize< 0 >(), I );
    EXPECT_EQ( b.template getSize< 1 >(), J );
    EXPECT_EQ( b.template getSize< 2 >(), K );
    EXPECT_EQ( b.template getSize< 3 >(), L );
    EXPECT_EQ( b.template getSize< 4 >(), M );
    EXPECT_EQ( b.template getSize< 5 >(), N );
}

TEST( NDArrayTest, reset )
{
    int I = 2, J = 2, K = 2, L = 2, M = 2, N = 2;
    NDArray< int,
             SizesHolder< int, 0, 0, 0, 0, 0, 0 >,
             index_sequence< 5, 3, 4, 2, 0, 1 > > a;
    a.setSizes( I, J, K, L, M, N );
    EXPECT_EQ( a.template getSize< 0 >(), I );
    EXPECT_EQ( a.template getSize< 1 >(), J );
    EXPECT_EQ( a.template getSize< 2 >(), K );
    EXPECT_EQ( a.template getSize< 3 >(), L );
    EXPECT_EQ( a.template getSize< 4 >(), M );
    EXPECT_EQ( a.template getSize< 5 >(), N );

    a.reset();
    EXPECT_EQ( a.template getSize< 0 >(), 0 );
    EXPECT_EQ( a.template getSize< 1 >(), 0 );
    EXPECT_EQ( a.template getSize< 2 >(), 0 );
    EXPECT_EQ( a.template getSize< 3 >(), 0 );
    EXPECT_EQ( a.template getSize< 4 >(), 0 );
    EXPECT_EQ( a.template getSize< 5 >(), 0 );
}

TEST( NDArrayTest, Static_1D )
{
    constexpr int I = 3;
    NDArray< int, SizesHolder< int, I > > a;
    a.setSizes( 0 );

    int v = 0;
    for( int i = 0; i < I; i++ ) {
        a( i ) = v++;
        EXPECT_EQ( a[ i ], a( i ) );
    }

    expect_identity( a.getStorageArray() );
}

TEST( NDArrayTest, Static_2D_Identity )
{
    constexpr int I = 3, J = 5;
    NDArray< int, SizesHolder< int, I, J > > a;
    a.setSizes( 0, 0 );

    int v = 0;
    for( int i = 0; i < I; i++ )
        for( int j = 0; j < J; j++ )
            a( i, j ) = v++;

    expect_identity( a.getStorageArray() );
}

TEST( NDArrayTest, Static_2D_Permuted )
{
    constexpr int I = 3, J = 5;
    NDArray< int,
             SizesHolder< int, I, J >,
             index_sequence< 1, 0 > > a;
    a.setSizes( 0, 0 );

    int v = 0;
    for( int j = 0; j < J; j++ )
        for( int i = 0; i < I; i++ )
            a( i, j ) = v++;

    expect_identity( a.getStorageArray() );
}

TEST( NDArrayTest, Dynamic_6D )
{
    int I = 2, J = 2, K = 2, L = 2, M = 2, N = 2;
    NDArray< int,
             SizesHolder< int, 0, 0, 0, 0, 0, 0 >,
             index_sequence< 5, 3, 4, 2, 0, 1 > > a;
    a.setSizes( I, J, K, L, M, N );

    // initialize entries invalid due to alignment to -1
    a.getStorageArray().setValue( -1 );

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

TEST( NDArrayTest, CopySemantics )
{
    constexpr int I = 3, J = 4;
    NDArray< int, SizesHolder< int, 0, 0 > > a;
    a.setSizes( I, J );

    auto a_view = a.getView();

    int v = 0;
    for( int i = 0; i < I; i++ )
    for( int j = 0; j < J; j++ )
        a( i, j ) = v++;

    expect_identity( a.getStorageArray() );

    // assignment with zero sizes
    NDArray< int, SizesHolder< int, 0, 0 > > b;
    b = a;
    auto b_view = b.getView();
    EXPECT_EQ( a, b );
    EXPECT_EQ( a_view, b_view );
    EXPECT_EQ( a_view.getView(), b_view );
    EXPECT_EQ( a_view.getConstView(), b_view.getConstView() );
    EXPECT_EQ( a.getConstView(), b.getConstView() );
    EXPECT_EQ( a.getConstView(), b_view.getConstView() );

    // assignment between views
    NDArray< int, SizesHolder< int, 0, 0 > > c;
    c.setSizes( I, J );
    auto c_view = c.getView();
    c_view = a_view;
    EXPECT_EQ( a, c );
    EXPECT_EQ( a_view, c_view );
    EXPECT_EQ( a_view.getView(), c_view );
    EXPECT_EQ( a_view.getConstView(), c_view.getConstView() );
    EXPECT_EQ( a.getConstView(), c.getConstView() );
    EXPECT_EQ( a.getConstView(), c_view.getConstView() );

    // move-assignment between views should do a deep copy
    b_view = a.getView();
    EXPECT_EQ( a_view, b_view );
    EXPECT_EQ( a, b );
    EXPECT_NE( &b_view( 0, 0 ), &a_view( 0, 0 ) );

    // assignment of view to array
    c.setValue( 0 );
    c = a_view;
    EXPECT_EQ( a, c );
    EXPECT_EQ( a_view, c_view );
    EXPECT_EQ( a_view.getView(), c_view );
    EXPECT_EQ( a_view.getConstView(), c_view.getConstView() );
    EXPECT_EQ( a.getConstView(), c.getConstView() );
    EXPECT_EQ( a.getConstView(), c_view.getConstView() );

    // assignment of array to view
    c.setValue( 0 );
    c_view = a;
    EXPECT_EQ( a, c );
    EXPECT_EQ( a_view, c_view );
    EXPECT_EQ( a_view.getView(), c_view );
    EXPECT_EQ( a_view.getConstView(), c_view.getConstView() );
    EXPECT_EQ( a.getConstView(), c.getConstView() );
    EXPECT_EQ( a.getConstView(), c_view.getConstView() );

    // assignment with different ValueType
    NDArray< double, SizesHolder< int, 0, 0 > > d;
    d = a;
    expect_identity( d.getStorageArray() );

    // assignment with different SizesHolder
    NDArray< double, SizesHolder< int, I, J > > e;
    e = a;
    expect_identity( e.getStorageArray() );

    // assignment with different IndexType
    NDArray< double, SizesHolder< short int, 0, 0 > > f;
    f = a;
    expect_identity( f.getStorageArray() );

    // assignment with different Permutation
    // TODO
}

#ifdef HAVE_CUDA
TEST( NDArrayTest, CopySemanticsCrossDevice )
{
    constexpr int I = 3, J = 4;
    NDArray< int, SizesHolder< int, 0, 0 > > a;
    NDArray< int, SizesHolder< int, 0, 0 >,
             std::index_sequence< 0, 1 >,
             TNL::Devices::Cuda > da;
    a.setSizes( I, J );
    da.setSizes( I, J );

    auto a_view = a.getView();
    auto da_view = da.getView();

    int v = 0;
    for( int i = 0; i < I; i++ )
    for( int j = 0; j < J; j++ )
        a( i, j ) = v++;

    expect_identity( a.getStorageArray() );

    // copy to the device, simple check
    da = a;
    EXPECT_EQ( da.getStorageArray(), a.getStorageArray() );

    // assignment with zero sizes
    NDArray< int, SizesHolder< int, 0, 0 > > b;
    b = da;
    auto b_view = b.getView();
    EXPECT_EQ( a, b );
    EXPECT_EQ( a_view, b_view );
    EXPECT_EQ( a_view.getView(), b_view );
    EXPECT_EQ( a_view.getConstView(), b_view.getConstView() );
    EXPECT_EQ( a.getConstView(), b.getConstView() );
    EXPECT_EQ( a.getConstView(), b_view.getConstView() );

    // assignment between views
    NDArray< int, SizesHolder< int, 0, 0 > > c;
    c.setSizes( I, J );
    auto c_view = c.getView();
    c_view = da_view;
    EXPECT_EQ( a, c );
    EXPECT_EQ( a_view, c_view );
    EXPECT_EQ( a_view.getView(), c_view );
    EXPECT_EQ( a_view.getConstView(), c_view.getConstView() );
    EXPECT_EQ( a.getConstView(), c.getConstView() );
    EXPECT_EQ( a.getConstView(), c_view.getConstView() );

    // move-assignment between views should do a deep copy
    b_view = da.getView();
    EXPECT_EQ( a_view, b_view );
    EXPECT_EQ( a, b );
    EXPECT_NE( &b_view( 0, 0 ), &a_view( 0, 0 ) );

    // assignment of view to array
    c.setValue( 0 );
    c = da_view;
    EXPECT_EQ( a, c );
    EXPECT_EQ( a_view, c_view );
    EXPECT_EQ( a_view.getView(), c_view );
    EXPECT_EQ( a_view.getConstView(), c_view.getConstView() );
    EXPECT_EQ( a.getConstView(), c.getConstView() );
    EXPECT_EQ( a.getConstView(), c_view.getConstView() );

    // assignment of array to view
    c.setValue( 0 );
    c_view = da;
    EXPECT_EQ( a, c );
    EXPECT_EQ( a_view, c_view );
    EXPECT_EQ( a_view.getView(), c_view );
    EXPECT_EQ( a_view.getConstView(), c_view.getConstView() );
    EXPECT_EQ( a.getConstView(), c.getConstView() );
    EXPECT_EQ( a.getConstView(), c_view.getConstView() );

    // assignment with different ValueType
    NDArray< double, SizesHolder< int, 0, 0 > > d;
    d = da;
    expect_identity( d.getStorageArray() );

    // assignment with different SizesHolder
    NDArray< double, SizesHolder< int, I, J > > e;
    e = da;
    expect_identity( e.getStorageArray() );

    // assignment with different IndexType
    NDArray< double, SizesHolder< short int, 0, 0 > > f;
    f = da;
    expect_identity( f.getStorageArray() );

    // assignment with different Permutation
    // TODO
}
#endif

TEST( NDArrayTest, SizesHolderPrinter )
{
   SizesHolder< int, 0, 1, 2 > holder;
   holder.setSize< 0 >( 3 );

   std::stringstream str;
   str << holder;
   EXPECT_EQ( str.str(), "SizesHolder< 0, 1, 2 >( 3, 1, 2 )" );
}

TEST( NDArrayTest, forAll_dynamic_1D )
{
    int I = 2;
    NDArray< int,
             SizesHolder< int, 0 >,
             index_sequence< 0 > > a;
    a.setSizes( I );
    a.setValue( 0 );

    auto setter = [&] ( int i )
    {
       a( i ) += 1;
    };

    a.forAll( setter );

    for( int i = 0; i < I; i++ )
        EXPECT_EQ( a( i ), 1 );
}

TEST( NDArrayTest, forAll_dynamic_2D )
{
    int I = 2, J = 3;
    NDArray< int,
             SizesHolder< int, 0, 0 >,
             index_sequence< 1, 0 > > a;
    a.setSizes( I, J );
    a.setValue( 0 );

    auto setter = [&] ( int i, int j )
    {
       a( i, j ) += 1;
    };

    a.forAll( setter );

    for( int j = 0; j < J; j++ )
    for( int i = 0; i < I; i++ )
        EXPECT_EQ( a( i, j ), 1 );
}

TEST( NDArrayTest, forAll_dynamic_3D )
{
    int I = 2, J = 3, K = 4;
    NDArray< int,
             SizesHolder< int, 0, 0, 0 >,
             index_sequence< 2, 0, 1 > > a;
    a.setSizes( I, J, K );
    a.setValue( 0 );

    auto setter = [&] ( int i, int j, int k )
    {
       a( i, j, k ) += 1;
    };

    a.forAll( setter );

    for( int k = 0; k < K; k++ )
    for( int i = 0; i < I; i++ )
    for( int j = 0; j < J; j++ )
        EXPECT_EQ( a( i, j, k ), 1 );
}

TEST( NDArrayTest, forAll_dynamic_4D )
{
    int I = 2, J = 3, K = 4, L = 5;
    NDArray< int,
             SizesHolder< int, 0, 0, 0, 0 >,
             index_sequence< 3, 2, 0, 1 > > a;
    a.setSizes( I, J, K, L );
    a.setValue( 0 );

    auto setter = [&] ( int i, int j, int k, int l )
    {
       a( i, j, k, l ) += 1;
    };

    a.forAll( setter );

    for( int l = 0; l < L; l++ )
    for( int k = 0; k < K; k++ )
    for( int i = 0; i < I; i++ )
    for( int j = 0; j < J; j++ )
        EXPECT_EQ( a( i, j, k, l ), 1 );
}

TEST( NDArrayTest, forAll_dynamic_5D )
{
    int I = 2, J = 3, K = 4, L = 5, M = 6;
    NDArray< int,
             SizesHolder< int, 0, 0, 0, 0, 0 >,
             index_sequence< 3, 4, 2, 0, 1 > > a;
    a.setSizes( I, J, K, L, M );
    a.setValue( 0 );

    auto setter = [&] ( int i, int j, int k, int l, int m )
    {
       a( i, j, k, l, m ) += 1;
    };

    a.forAll( setter );

    for( int l = 0; l < L; l++ )
    for( int m = 0; m < M; m++ )
    for( int k = 0; k < K; k++ )
    for( int i = 0; i < I; i++ )
    for( int j = 0; j < J; j++ )
        EXPECT_EQ( a( i, j, k, l, m ), 1 );
}

TEST( NDArrayTest, forAll_dynamic_6D )
{
    int I = 2, J = 3, K = 4, L = 5, M = 6, N = 7;
    NDArray< int,
             SizesHolder< int, 0, 0, 0, 0, 0, 0 >,
             index_sequence< 5, 3, 4, 2, 0, 1 > > a;
    a.setSizes( I, J, K, L, M, N );
    a.setValue( 0 );

    auto setter = [&] ( int i, int j, int k, int l, int m, int n )
    {
       a( i, j, k, l, m, n ) += 1;
    };

    a.forAll( setter );

    for( int n = 0; n < N; n++ )
    for( int l = 0; l < L; l++ )
    for( int m = 0; m < M; m++ )
    for( int k = 0; k < K; k++ )
    for( int i = 0; i < I; i++ )
    for( int j = 0; j < J; j++ )
        EXPECT_EQ( a( i, j, k, l, m, n ), 1 );
}

TEST( NDArrayTest, forAll_static_1D )
{
    constexpr int I = 3;
    StaticNDArray< int, SizesHolder< int, I > > a;
//    a.setSizes( 0 );
    a.setValue( 0 );

    auto setter = [&] ( int i )
    {
       a( i ) += 1;
    };

    a.forAll( setter );

    for( int i = 0; i < I; i++ )
        EXPECT_EQ( a( i ), 1 );
}

TEST( NDArrayTest, forAll_static_2D )
{
    constexpr int I = 3, J = 4;
    StaticNDArray< int, SizesHolder< int, I, J > > a;
//    a.setSizes( 0, 0 );
    a.setValue( 0 );

    auto setter = [&] ( int i, int j )
    {
       a( i, j ) += 1;
    };

    a.forAll( setter );

    for( int i = 0; i < I; i++ )
    for( int j = 0; j < J; j++ )
        EXPECT_EQ( a( i, j ), 1 );
}

TEST( NDArrayTest, forAll_static_3D )
{
    constexpr int I = 3, J = 4, K = 5;
    StaticNDArray< int, SizesHolder< int, I, J, K > > a;
//    a.setSizes( 0, 0, 0 );
    a.setValue( 0 );

    auto setter = [&] ( int i, int j, int k )
    {
       a( i, j, k ) += 1;
    };

    a.forAll( setter );

    for( int i = 0; i < I; i++ )
    for( int j = 0; j < J; j++ )
    for( int k = 0; k < K; k++ )
        EXPECT_EQ( a( i, j, k ), 1 );
}

TEST( NDArrayTest, forAll_static_4D )
{
    constexpr int I = 3, J = 4, K = 5, L = 6;
    StaticNDArray< int, SizesHolder< int, I, J, K, L > > a;
//    a.setSizes( 0, 0, 0, 0 );
    a.setValue( 0 );

    auto setter = [&] ( int i, int j, int k, int l )
    {
       a( i, j, k, l ) += 1;
    };

    a.forAll( setter );

    for( int i = 0; i < I; i++ )
    for( int j = 0; j < J; j++ )
    for( int k = 0; k < K; k++ )
    for( int l = 0; l < L; l++ )
        EXPECT_EQ( a( i, j, k, l ), 1 );
}

TEST( NDArrayTest, forAll_static_5D )
{
    constexpr int I = 3, J = 4, K = 5, L = 6, M = 7;
    StaticNDArray< int, SizesHolder< int, I, J, K, L, M > > a;
//    a.setSizes( 0, 0, 0, 0, 0 );
    a.setValue( 0 );

    auto setter = [&] ( int i, int j, int k, int l, int m )
    {
       a( i, j, k, l, m ) += 1;
    };

    a.forAll( setter );

    for( int i = 0; i < I; i++ )
    for( int j = 0; j < J; j++ )
    for( int k = 0; k < K; k++ )
    for( int l = 0; l < L; l++ )
    for( int m = 0; m < M; m++ )
        EXPECT_EQ( a( i, j, k, l, m ), 1 );
}

TEST( NDArrayTest, forAll_static_6D )
{
    constexpr int I = 3, J = 4, K = 5, L = 6, M = 7, N = 8;
    StaticNDArray< int, SizesHolder< int, I, J, K, L, M, N > > a;
//    a.setSizes( 0, 0, 0, 0, 0, 0 );
    a.setValue( 0 );

    auto setter = [&] ( int i, int j, int k, int l, int m, int n )
    {
       a( i, j, k, l, m, n ) += 1;
    };

    a.forAll( setter );

    for( int i = 0; i < I; i++ )
    for( int j = 0; j < J; j++ )
    for( int k = 0; k < K; k++ )
    for( int l = 0; l < L; l++ )
    for( int m = 0; m < M; m++ )
    for( int n = 0; n < N; n++ )
        EXPECT_EQ( a( i, j, k, l, m, n ), 1 );
}

TEST( NDArrayTest, forInternal_dynamic_1D )
{
    int I = 3;
    NDArray< int,
             SizesHolder< int, 0 >,
             index_sequence< 0 > > a;
    a.setSizes( I );
    a.setValue( 0 );

    auto setter = [&] ( int i )
    {
       a( i ) += 1;
    };

    a.forInternal( setter );

    for( int i = 0; i < I; i++ )
    {
        if( i == 0 || i == I - 1 )
            EXPECT_EQ( a( i ), 0 )
               << "i = " << i;
        else
            EXPECT_EQ( a( i ), 1 )
               << "i = " << i;
    }
}

TEST( NDArrayTest, forInternal_dynamic_2D )
{
    int I = 3, J = 4;
    NDArray< int,
             SizesHolder< int, 0, 0 >,
             index_sequence< 1, 0 > > a;
    a.setSizes( I, J );
    a.setValue( 0 );

    auto setter = [&] ( int i, int j )
    {
       a( i, j ) += 1;
    };

    a.forInternal( setter );

    for( int j = 0; j < J; j++ )
    for( int i = 0; i < I; i++ )
    {
        if( i == 0 || i == I - 1 ||
            j == 0 || j == J - 1 )
            EXPECT_EQ( a( i, j ), 0 )
               << "i = " << i << ", j = " << j;
        else
            EXPECT_EQ( a( i, j ), 1 )
               << "i = " << i << ", j = " << j;
    }
}

TEST( NDArrayTest, forInternal_dynamic_3D )
{
    int I = 3, J = 4, K = 5;
    NDArray< int,
             SizesHolder< int, 0, 0, 0 >,
             index_sequence< 2, 0, 1 > > a;
    a.setSizes( I, J, K );
    a.setValue( 0 );

    auto setter = [&] ( int i, int j, int k )
    {
       a( i, j, k ) += 1;
    };

    a.forInternal( setter );

    for( int k = 0; k < K; k++ )
    for( int i = 0; i < I; i++ )
    for( int j = 0; j < J; j++ )
    {
        if( i == 0 || i == I - 1 ||
            j == 0 || j == J - 1 ||
            k == 0 || k == K - 1 )
            EXPECT_EQ( a( i, j, k ), 0 )
               << "i = " << i << ", j = " << j << ", k = " << k;
        else
            EXPECT_EQ( a( i, j, k ), 1 )
               << "i = " << i << ", j = " << j << ", k = " << k;
    }
}

TEST( NDArrayTest, forInternal_dynamic_4D )
{
    int I = 3, J = 4, K = 5, L = 6;
    NDArray< int,
             SizesHolder< int, 0, 0, 0, 0 >,
             index_sequence< 3, 2, 0, 1 > > a;
    a.setSizes( I, J, K, L );
    a.setValue( 0 );

    auto setter = [&] ( int i, int j, int k, int l )
    {
       a( i, j, k, l ) += 1;
    };

    a.forInternal( setter );

    for( int l = 0; l < L; l++ )
    for( int k = 0; k < K; k++ )
    for( int i = 0; i < I; i++ )
    for( int j = 0; j < J; j++ )
    {
        if( i == 0 || i == I - 1 ||
            j == 0 || j == J - 1 ||
            k == 0 || k == K - 1 ||
            l == 0 || l == L - 1 )
            EXPECT_EQ( a( i, j, k, l ), 0 )
               << "i = " << i << ", j = " << j << ", k = " << k << ", l = " << l;
        else
            EXPECT_EQ( a( i, j, k, l ), 1 )
               << "i = " << i << ", j = " << j << ", k = " << k << ", l = " << l;
    }
}

TEST( NDArrayTest, forInternal_dynamic_5D )
{
    int I = 3, J = 4, K = 5, L = 6, M = 7;
    NDArray< int,
             SizesHolder< int, 0, 0, 0, 0, 0 >,
             index_sequence< 3, 4, 2, 0, 1 > > a;
    a.setSizes( I, J, K, L, M );
    a.setValue( 0 );

    auto setter = [&] ( int i, int j, int k, int l, int m )
    {
       a( i, j, k, l, m ) += 1;
    };

    a.forInternal( setter );

    for( int l = 0; l < L; l++ )
    for( int m = 0; m < M; m++ )
    for( int k = 0; k < K; k++ )
    for( int i = 0; i < I; i++ )
    for( int j = 0; j < J; j++ )
    {
        if( i == 0 || i == I - 1 ||
            j == 0 || j == J - 1 ||
            k == 0 || k == K - 1 ||
            l == 0 || l == L - 1 ||
            m == 0 || m == M - 1 )
            EXPECT_EQ( a( i, j, k, l, m ), 0 )
               << "i = " << i << ", j = " << j << ", k = " << k << ", l = " << l << ", m = " << m;
        else
            EXPECT_EQ( a( i, j, k, l, m ), 1 )
               << "i = " << i << ", j = " << j << ", k = " << k << ", l = " << l << ", m = " << m;
    }
}

TEST( NDArrayTest, forInternal_dynamic_6D )
{
    int I = 3, J = 4, K = 5, L = 6, M = 7, N = 8;
    NDArray< int,
             SizesHolder< int, 0, 0, 0, 0, 0, 0 >,
             index_sequence< 5, 3, 4, 2, 0, 1 > > a;
    a.setSizes( I, J, K, L, M, N );
    a.setValue( 0 );

    auto setter = [&] ( int i, int j, int k, int l, int m, int n )
    {
       a( i, j, k, l, m, n ) += 1;
    };

    a.forInternal( setter );

    for( int n = 0; n < N; n++ )
    for( int l = 0; l < L; l++ )
    for( int m = 0; m < M; m++ )
    for( int k = 0; k < K; k++ )
    for( int i = 0; i < I; i++ )
    for( int j = 0; j < J; j++ )
    {
        if( i == 0 || i == I - 1 ||
            j == 0 || j == J - 1 ||
            k == 0 || k == K - 1 ||
            l == 0 || l == L - 1 ||
            m == 0 || m == M - 1 ||
            n == 0 || n == N - 1 )
            EXPECT_EQ( a( i, j, k, l, m, n ), 0 )
               << "i = " << i << ", j = " << j << ", k = " << k << ", l = " << l << ", m = " << m << ", n = " << n;
        else
            EXPECT_EQ( a( i, j, k, l, m, n ), 1 )
               << "i = " << i << ", j = " << j << ", k = " << k << ", l = " << l << ", m = " << m << ", n = " << n;
    }
}

TEST( NDArrayTest, forInternal_static_1D )
{
    constexpr int I = 3;
    StaticNDArray< int, SizesHolder< int, I > > a;
//    a.setSizes( 0 );
    a.setValue( 0 );

    auto setter = [&] ( int i )
    {
       a( i ) += 1;
    };

    a.forInternal( setter );

    for( int i = 0; i < I; i++ )
    {
        if( i == 0 || i == I - 1 )
            EXPECT_EQ( a( i ), 0 )
               << "i = " << i;
        else
            EXPECT_EQ( a( i ), 1 )
               << "i = " << i;
    }
}

TEST( NDArrayTest, forInternal_static_2D )
{
    constexpr int I = 3, J = 4;
    StaticNDArray< int, SizesHolder< int, I, J > > a;
//    a.setSizes( 0, 0 );
    a.setValue( 0 );

    auto setter = [&] ( int i, int j )
    {
       a( i, j ) += 1;
    };

    a.forInternal( setter );

    for( int j = 0; j < J; j++ )
    for( int i = 0; i < I; i++ )
    {
        if( i == 0 || i == I - 1 ||
            j == 0 || j == J - 1 )
            EXPECT_EQ( a( i, j ), 0 )
               << "i = " << i << ", j = " << j;
        else
            EXPECT_EQ( a( i, j ), 1 )
               << "i = " << i << ", j = " << j;
    }
}

TEST( NDArrayTest, forInternal_static_3D )
{
    constexpr int I = 3, J = 4, K = 5;
    StaticNDArray< int, SizesHolder< int, I, J, K > > a;
//    a.setSizes( 0, 0, 0 );
    a.setValue( 0 );

    auto setter = [&] ( int i, int j, int k )
    {
       a( i, j, k ) += 1;
    };

    a.forInternal( setter );

    for( int k = 0; k < K; k++ )
    for( int i = 0; i < I; i++ )
    for( int j = 0; j < J; j++ )
    {
        if( i == 0 || i == I - 1 ||
            j == 0 || j == J - 1 ||
            k == 0 || k == K - 1 )
            EXPECT_EQ( a( i, j, k ), 0 )
               << "i = " << i << ", j = " << j << ", k = " << k;
        else
            EXPECT_EQ( a( i, j, k ), 1 )
               << "i = " << i << ", j = " << j << ", k = " << k;
    }
}

TEST( NDArrayTest, forInternal_static_4D )
{
    constexpr int I = 3, J = 4, K = 5, L = 6;
    StaticNDArray< int, SizesHolder< int, I, J, K, L > > a;
//    a.setSizes( 0, 0, 0, 0 );
    a.setValue( 0 );

    auto setter = [&] ( int i, int j, int k, int l )
    {
       a( i, j, k, l ) += 1;
    };

    a.forInternal( setter );

    for( int l = 0; l < L; l++ )
    for( int k = 0; k < K; k++ )
    for( int i = 0; i < I; i++ )
    for( int j = 0; j < J; j++ )
    {
        if( i == 0 || i == I - 1 ||
            j == 0 || j == J - 1 ||
            k == 0 || k == K - 1 ||
            l == 0 || l == L - 1 )
            EXPECT_EQ( a( i, j, k, l ), 0 )
               << "i = " << i << ", j = " << j << ", k = " << k << ", l = " << l;
        else
            EXPECT_EQ( a( i, j, k, l ), 1 )
               << "i = " << i << ", j = " << j << ", k = " << k << ", l = " << l;
    }
}

TEST( NDArrayTest, forInternal_static_5D )
{
    constexpr int I = 3, J = 4, K = 5, L = 6, M = 7;
    StaticNDArray< int, SizesHolder< int, I, J, K, L, M > > a;
//    a.setSizes( 0, 0, 0, 0, 0 );
    a.setValue( 0 );

    auto setter = [&] ( int i, int j, int k, int l, int m )
    {
       a( i, j, k, l, m ) += 1;
    };

    a.forInternal( setter );

    for( int l = 0; l < L; l++ )
    for( int m = 0; m < M; m++ )
    for( int k = 0; k < K; k++ )
    for( int i = 0; i < I; i++ )
    for( int j = 0; j < J; j++ )
    {
        if( i == 0 || i == I - 1 ||
            j == 0 || j == J - 1 ||
            k == 0 || k == K - 1 ||
            l == 0 || l == L - 1 ||
            m == 0 || m == M - 1 )
            EXPECT_EQ( a( i, j, k, l, m ), 0 )
               << "i = " << i << ", j = " << j << ", k = " << k << ", l = " << l << ", m = " << m;
        else
            EXPECT_EQ( a( i, j, k, l, m ), 1 )
               << "i = " << i << ", j = " << j << ", k = " << k << ", l = " << l << ", m = " << m;
    }
}

TEST( NDArrayTest, forInternal_static_6D )
{
    constexpr int I = 3, J = 4, K = 5, L = 6, M = 7, N = 8;
    StaticNDArray< int, SizesHolder< int, I, J, K, L, M, N > > a;
//    a.setSizes( 0, 0, 0, 0, 0, 0 );
    a.setValue( 0 );

    auto setter = [&] ( int i, int j, int k, int l, int m, int n )
    {
       a( i, j, k, l, m, n ) += 1;
    };

    a.forInternal( setter );

    for( int n = 0; n < N; n++ )
    for( int l = 0; l < L; l++ )
    for( int m = 0; m < M; m++ )
    for( int k = 0; k < K; k++ )
    for( int i = 0; i < I; i++ )
    for( int j = 0; j < J; j++ )
    {
        if( i == 0 || i == I - 1 ||
            j == 0 || j == J - 1 ||
            k == 0 || k == K - 1 ||
            l == 0 || l == L - 1 ||
            m == 0 || m == M - 1 ||
            n == 0 || n == N - 1 )
            EXPECT_EQ( a( i, j, k, l, m, n ), 0 )
               << "i = " << i << ", j = " << j << ", k = " << k << ", l = " << l << ", m = " << m << ", n = " << n;
        else
            EXPECT_EQ( a( i, j, k, l, m, n ), 1 )
               << "i = " << i << ", j = " << j << ", k = " << k << ", l = " << l << ", m = " << m << ", n = " << n;
    }
}

TEST( NDArrayTest, forBoundary_dynamic_1D )
{
    int I = 3;
    NDArray< int,
             SizesHolder< int, 0 >,
             index_sequence< 0 > > a;
    a.setSizes( I );
    a.setValue( 0 );

    auto setter = [&] ( int i )
    {
       a( i ) += 1;
    };

    a.forBoundary( setter );

    for( int i = 0; i < I; i++ )
    {
        if( i == 0 || i == I - 1 )
            EXPECT_EQ( a( i ), 1 )
               << "i = " << i;
        else
            EXPECT_EQ( a( i ), 0 )
               << "i = " << i;
    }
}

TEST( NDArrayTest, forBoundary_dynamic_2D )
{
    int I = 3, J = 4;
    NDArray< int,
             SizesHolder< int, 0, 0 >,
             index_sequence< 1, 0 > > a;
    a.setSizes( I, J );
    a.setValue( 0 );

    auto setter = [&] ( int i, int j )
    {
       a( i, j ) += 1;
    };

    a.forBoundary( setter );

    for( int j = 0; j < J; j++ )
    for( int i = 0; i < I; i++ )
    {
        if( i == 0 || i == I - 1 ||
            j == 0 || j == J - 1 )
            EXPECT_EQ( a( i, j ), 1 )
               << "i = " << i << ", j = " << j;
        else
            EXPECT_EQ( a( i, j ), 0 )
               << "i = " << i << ", j = " << j;
    }
}

TEST( NDArrayTest, forBoundary_dynamic_3D )
{
    int I = 3, J = 4, K = 5;
    NDArray< int,
             SizesHolder< int, 0, 0, 0 >,
             index_sequence< 2, 0, 1 > > a;
    a.setSizes( I, J, K );
    a.setValue( 0 );

    auto setter = [&] ( int i, int j, int k )
    {
       a( i, j, k ) += 1;
    };

    a.forBoundary( setter );

    for( int k = 0; k < K; k++ )
    for( int i = 0; i < I; i++ )
    for( int j = 0; j < J; j++ )
    {
        if( i == 0 || i == I - 1 ||
            j == 0 || j == J - 1 ||
            k == 0 || k == K - 1 )
            EXPECT_EQ( a( i, j, k ), 1 )
               << "i = " << i << ", j = " << j << ", k = " << k;
        else
            EXPECT_EQ( a( i, j, k ), 0 )
               << "i = " << i << ", j = " << j << ", k = " << k;
    }
}

// TODO: implement general ParallelBoundaryExecutor
//TEST( NDArrayTest, forBoundary_dynamic_4D )
//{
//    int I = 3, J = 4, K = 5, L = 6;
//    NDArray< int,
//             SizesHolder< int, 0, 0, 0, 0 >,
//             index_sequence< 3, 2, 0, 1 > > a;
//    a.setSizes( I, J, K, L );
//    a.setValue( 0 );
//
//    auto setter = [&] ( int i, int j, int k, int l )
//    {
//       a( i, j, k, l ) += 1;
//    };
//
//    a.forBoundary( setter );
//
//    for( int l = 0; l < L; l++ )
//    for( int k = 0; k < K; k++ )
//    for( int i = 0; i < I; i++ )
//    for( int j = 0; j < J; j++ )
//    {
//        if( i == 0 || i == I - 1 ||
//            j == 0 || j == J - 1 ||
//            k == 0 || k == K - 1 ||
//            l == 0 || l == L - 1 )
//            EXPECT_EQ( a( i, j, k, l ), 1 )
//               << "i = " << i << ", j = " << j << ", k = " << k << ", l = " << l;
//        else
//            EXPECT_EQ( a( i, j, k, l ), 0 )
//               << "i = " << i << ", j = " << j << ", k = " << k << ", l = " << l;
//    }
//}
//
//TEST( NDArrayTest, forBoundary_dynamic_5D )
//{
//    int I = 3, J = 4, K = 5, L = 6, M = 7;
//    NDArray< int,
//             SizesHolder< int, 0, 0, 0, 0, 0 >,
//             index_sequence< 3, 4, 2, 0, 1 > > a;
//    a.setSizes( I, J, K, L, M );
//    a.setValue( 0 );
//
//    auto setter = [&] ( int i, int j, int k, int l, int m )
//    {
//       a( i, j, k, l, m ) += 1;
//    };
//
//    a.forBoundary( setter );
//
//    for( int l = 0; l < L; l++ )
//    for( int m = 0; m < M; m++ )
//    for( int k = 0; k < K; k++ )
//    for( int i = 0; i < I; i++ )
//    for( int j = 0; j < J; j++ )
//    {
//        if( i == 0 || i == I - 1 ||
//            j == 0 || j == J - 1 ||
//            k == 0 || k == K - 1 ||
//            l == 0 || l == L - 1 ||
//            m == 0 || m == M - 1 )
//            EXPECT_EQ( a( i, j, k, l, m ), 1 )
//               << "i = " << i << ", j = " << j << ", k = " << k << ", l = " << l << ", m = " << m;
//        else
//            EXPECT_EQ( a( i, j, k, l, m ), 0 )
//               << "i = " << i << ", j = " << j << ", k = " << k << ", l = " << l << ", m = " << m;
//    }
//}
//
//TEST( NDArrayTest, forBoundary_dynamic_6D )
//{
//    int I = 3, J = 4, K = 5, L = 6, M = 7, N = 8;
//    NDArray< int,
//             SizesHolder< int, 0, 0, 0, 0, 0, 0 >,
//             index_sequence< 5, 3, 4, 2, 0, 1 > > a;
//    a.setSizes( I, J, K, L, M, N );
//    a.setValue( 0 );
//
//    auto setter = [&] ( int i, int j, int k, int l, int m, int n )
//    {
//       a( i, j, k, l, m, n ) += 1;
//    };
//
//    a.forBoundary( setter );
//
//    for( int n = 0; n < N; n++ )
//    for( int l = 0; l < L; l++ )
//    for( int m = 0; m < M; m++ )
//    for( int k = 0; k < K; k++ )
//    for( int i = 0; i < I; i++ )
//    for( int j = 0; j < J; j++ )
//    {
//        if( i == 0 || i == I - 1 ||
//            j == 0 || j == J - 1 ||
//            k == 0 || k == K - 1 ||
//            l == 0 || l == L - 1 ||
//            m == 0 || m == M - 1 ||
//            n == 0 || n == N - 1 )
//            EXPECT_EQ( a( i, j, k, l, m, n ), 1 )
//               << "i = " << i << ", j = " << j << ", k = " << k << ", l = " << l << ", m = " << m << ", n = " << n;
//        else
//            EXPECT_EQ( a( i, j, k, l, m, n ), 0 )
//               << "i = " << i << ", j = " << j << ", k = " << k << ", l = " << l << ", m = " << m << ", n = " << n;
//    }
//}

TEST( NDArrayTest, forBoundary_static_1D )
{
    constexpr int I = 3;
    StaticNDArray< int, SizesHolder< int, I > > a;
//    a.setSizes( 0 );
    a.setValue( 0 );

    auto setter = [&] ( int i )
    {
       a( i ) += 1;
    };

    a.forBoundary( setter );

    for( int i = 0; i < I; i++ )
    {
        if( i == 0 || i == I - 1 )
            EXPECT_EQ( a( i ), 1 )
               << "i = " << i;
        else
            EXPECT_EQ( a( i ), 0 )
               << "i = " << i;
    }
}

TEST( NDArrayTest, forBoundary_static_2D )
{
    constexpr int I = 3, J = 4;
    StaticNDArray< int, SizesHolder< int, I, J > > a;
//    a.setSizes( 0, 0 );
    a.setValue( 0 );

    auto setter = [&] ( int i, int j )
    {
       a( i, j ) += 1;
    };

    a.forBoundary( setter );

    for( int j = 0; j < J; j++ )
    for( int i = 0; i < I; i++ )
    {
        if( i == 0 || i == I - 1 ||
            j == 0 || j == J - 1 )
            EXPECT_EQ( a( i, j ), 1 )
               << "i = " << i << ", j = " << j;
        else
            EXPECT_EQ( a( i, j ), 0 )
               << "i = " << i << ", j = " << j;
    }
}

TEST( NDArrayTest, forBoundary_static_3D )
{
    constexpr int I = 3, J = 4, K = 5;
    StaticNDArray< int, SizesHolder< int, I, J, K > > a;
//    a.setSizes( 0, 0, 0 );
    a.setValue( 0 );

    auto setter = [&] ( int i, int j, int k )
    {
       a( i, j, k ) += 1;
    };

    a.forBoundary( setter );

    for( int k = 0; k < K; k++ )
    for( int i = 0; i < I; i++ )
    for( int j = 0; j < J; j++ )
    {
        if( i == 0 || i == I - 1 ||
            j == 0 || j == J - 1 ||
            k == 0 || k == K - 1 )
            EXPECT_EQ( a( i, j, k ), 1 )
               << "i = " << i << ", j = " << j << ", k = " << k;
        else
            EXPECT_EQ( a( i, j, k ), 0 )
               << "i = " << i << ", j = " << j << ", k = " << k;
    }
}

TEST( NDArrayTest, forBoundary_static_4D )
{
    constexpr int I = 3, J = 4, K = 5, L = 6;
    StaticNDArray< int, SizesHolder< int, I, J, K, L > > a;
//    a.setSizes( 0, 0, 0, 0 );
    a.setValue( 0 );

    auto setter = [&] ( int i, int j, int k, int l )
    {
       a( i, j, k, l ) += 1;
    };

    a.forBoundary( setter );

    for( int l = 0; l < L; l++ )
    for( int k = 0; k < K; k++ )
    for( int i = 0; i < I; i++ )
    for( int j = 0; j < J; j++ )
    {
        if( i == 0 || i == I - 1 ||
            j == 0 || j == J - 1 ||
            k == 0 || k == K - 1 ||
            l == 0 || l == L - 1 )
            EXPECT_EQ( a( i, j, k, l ), 1 )
               << "i = " << i << ", j = " << j << ", k = " << k << ", l = " << l;
        else
            EXPECT_EQ( a( i, j, k, l ), 0 )
               << "i = " << i << ", j = " << j << ", k = " << k << ", l = " << l;
    }
}

TEST( NDArrayTest, forBoundary_static_5D )
{
    constexpr int I = 3, J = 4, K = 5, L = 6, M = 7;
    StaticNDArray< int, SizesHolder< int, I, J, K, L, M > > a;
//    a.setSizes( 0, 0, 0, 0, 0 );
    a.setValue( 0 );

    auto setter = [&] ( int i, int j, int k, int l, int m )
    {
       a( i, j, k, l, m ) += 1;
    };

    a.forBoundary( setter );

    for( int l = 0; l < L; l++ )
    for( int m = 0; m < M; m++ )
    for( int k = 0; k < K; k++ )
    for( int i = 0; i < I; i++ )
    for( int j = 0; j < J; j++ )
    {
        if( i == 0 || i == I - 1 ||
            j == 0 || j == J - 1 ||
            k == 0 || k == K - 1 ||
            l == 0 || l == L - 1 ||
            m == 0 || m == M - 1 )
            EXPECT_EQ( a( i, j, k, l, m ), 1 )
               << "i = " << i << ", j = " << j << ", k = " << k << ", l = " << l << ", m = " << m;
        else
            EXPECT_EQ( a( i, j, k, l, m ), 0 )
               << "i = " << i << ", j = " << j << ", k = " << k << ", l = " << l << ", m = " << m;
    }
}

TEST( NDArrayTest, forBoundary_static_6D )
{
    constexpr int I = 3, J = 4, K = 5, L = 6, M = 7, N = 8;
    StaticNDArray< int, SizesHolder< int, I, J, K, L, M, N > > a;
//    a.setSizes( 0, 0, 0, 0, 0, 0 );
    a.setValue( 0 );

    auto setter = [&] ( int i, int j, int k, int l, int m, int n )
    {
       a( i, j, k, l, m, n ) += 1;
    };

    a.forBoundary( setter );

    for( int n = 0; n < N; n++ )
    for( int l = 0; l < L; l++ )
    for( int m = 0; m < M; m++ )
    for( int k = 0; k < K; k++ )
    for( int i = 0; i < I; i++ )
    for( int j = 0; j < J; j++ )
    {
        if( i == 0 || i == I - 1 ||
            j == 0 || j == J - 1 ||
            k == 0 || k == K - 1 ||
            l == 0 || l == L - 1 ||
            m == 0 || m == M - 1 ||
            n == 0 || n == N - 1 )
            EXPECT_EQ( a( i, j, k, l, m, n ), 1 )
               << "i = " << i << ", j = " << j << ", k = " << k << ", l = " << l << ", m = " << m << ", n = " << n;
        else
            EXPECT_EQ( a( i, j, k, l, m, n ), 0 )
               << "i = " << i << ", j = " << j << ", k = " << k << ", l = " << l << ", m = " << m << ", n = " << n;
    }
}
#endif // HAVE_GTEST


#include "../../main.h"
