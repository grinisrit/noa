#ifdef HAVE_GTEST
#include "gtest/gtest.h"

#include <TNL/Containers/NDArray.h>

using namespace TNL::Containers;
using std::index_sequence;

// wrapper around static_assert to get the type names in the error message
template< typename Permutation, typename ExpectedPermutation >
void check_permutation()
{
    static_assert( std::is_same< Permutation, ExpectedPermutation >::value,
                   "The permutation is not the same as the expected permutation." );
}

TEST( NDArraySubarrayTest, StaticAsserts )
{
    using namespace TNL::Containers::__ndarray_impl;

//    auto is_even = [](int _in) {return _in % 2 == 0;};
    using expected_type = std::integer_sequence<int, 0, 2, 4, 6, 8>;
    using test_type = std::integer_sequence<int, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9>;
//    constexpr auto result = filter_sequence(test_type{}, is_even);
    constexpr auto result = filter_sequence< expected_type >(test_type{});
    using result_type = std::decay_t<decltype(result)>;
    static_assert(std::is_same<expected_type, result_type>::value, "Integer sequences should be equal");



    using Permutation = std::integer_sequence< std::size_t, 5, 3, 1, 4, 2, 6, 0 >;
    {
        using Dimensions = std::integer_sequence< std::size_t, 3, 4, 6 >;
        using Subpermutation = typename SubpermutationGetter< Dimensions, Permutation >::Subpermutation;
        check_permutation< Subpermutation,
                           std::integer_sequence< std::size_t, 0, 1, 2 > >();
    }
    {
        using Dimensions = std::integer_sequence< std::size_t, 1, 4, 2 >;
        using Subpermutation = typename SubpermutationGetter< Dimensions, Permutation >::Subpermutation;
        check_permutation< Subpermutation,
                           std::integer_sequence< std::size_t, 0, 2, 1 > >();
    }
    {
        using Dimensions = std::integer_sequence< std::size_t, 5, 1, 6 >;
        using Subpermutation = typename SubpermutationGetter< Dimensions, Permutation >::Subpermutation;
        check_permutation< Subpermutation,
                           std::integer_sequence< std::size_t, 1, 0, 2 > >();
    }
    {
        using Dimensions = std::integer_sequence< std::size_t, 5, 1, 2 >;
        using Subpermutation = typename SubpermutationGetter< Dimensions, Permutation >::Subpermutation;
        check_permutation< Subpermutation,
                           std::integer_sequence< std::size_t, 2, 0, 1 > >();
    }
    {
        using Dimensions = std::integer_sequence< std::size_t, 2, 3, 4 >;
        using Subpermutation = typename SubpermutationGetter< Dimensions, Permutation >::Subpermutation;
        check_permutation< Subpermutation,
                           std::integer_sequence< std::size_t, 1, 2, 0 > >();
    }
    {
        using Dimensions = std::integer_sequence< std::size_t, 0, 1, 5 >;
        using Subpermutation = typename SubpermutationGetter< Dimensions, Permutation >::Subpermutation;
        check_permutation< Subpermutation,
                           std::integer_sequence< std::size_t, 2, 1, 0 > >();
    }

    static_assert( is_increasing_sequence( {0, 1, 2, 3, 4} ), "bug" );
    static_assert( ! is_increasing_sequence( {0, 1, 2, 0, 4} ), "bug" );
    static_assert( ! is_increasing_sequence( {1, 0, 2, 3, 4} ), "bug" );
}

TEST( NDArraySubarrayTest, Dynamic_6D )
{
    int I = 2, J = 3, K = 4, L = 5, M = 6, N = 7;
    NDArray< int,
             SizesHolder< int, 0, 0, 0, 0, 0, 0 >,
             index_sequence< 5, 3, 4, 2, 0, 1 > > a;
    a.setSizes( I, J, K, L, M, N );
    a.setValue( 0 );

    auto v = a.getView();

    auto s1 = v.template getSubarrayView< 0 >( 0, 0, 0, 0, 0, 0 );
    const int size1 = s1.template getSize< 0 >();
    const int stride1 = s1.template getStride< 0 >();
    EXPECT_EQ( size1, I );
    EXPECT_EQ( stride1, J );
    for( int i = 0; i < I; i++ ) {
        s1( i ) = 1 + i;
        EXPECT_EQ( v( i, 0, 0, 0, 0, 0 ), 1 + i );
    }
    a.setValue( 0 );

    auto s2 = v.template getSubarrayView< 1 >( 0, 0, 0, 0, 0, 0 );
    const int size2 = s2.template getSize< 0 >();
    const int stride2 = s2.template getStride< 0 >();
    EXPECT_EQ( size2, J );
    EXPECT_EQ( stride2, 1 );
    for( int j = 0; j < J; j++ ) {
        s2( j ) = 1 + j;
        EXPECT_EQ( v( 0, j, 0, 0, 0, 0 ), 1 + j );
    }
    a.setValue( 0 );

    auto s3 = v.template getSubarrayView< 2 >( 0, 0, 0, 0, 0, 0 );
    const int size3 = s3.template getSize< 0 >();
    const int stride3 = s3.template getStride< 0 >();
    EXPECT_EQ( size3, K );
    EXPECT_EQ( stride3, I*J );
    for( int k = 0; k < K; k++ ) {
        s3( k ) = 1 + k;
        EXPECT_EQ( v( 0, 0, k, 0, 0, 0 ), 1 + k );
    }
    a.setValue( 0 );

    auto s4 = v.template getSubarrayView< 3 >( 0, 0, 0, 0, 0, 0 );
    const int size4 = s4.template getSize< 0 >();
    const int stride4 = s4.template getStride< 0 >();
    EXPECT_EQ( size4, L );
    EXPECT_EQ( stride4, I*J*K*M );
    for( int l = 0; l < L; l++ ) {
        s4( l ) = 1 + l;
        EXPECT_EQ( v( 0, 0, 0, l, 0, 0 ), 1 + l );
    }
    a.setValue( 0 );

    auto s5 = v.template getSubarrayView< 4 >( 0, 0, 0, 0, 0, 0 );
    const int size5 = s5.template getSize< 0 >();
    const int stride5 = s5.template getStride< 0 >();
    EXPECT_EQ( size5, M );
    EXPECT_EQ( stride5, I*J*K );
    for( int m = 0; m < M; m++ ) {
        s5( m ) = 1 + m;
        EXPECT_EQ( v( 0, 0, 0, 0, m, 0 ), 1 + m );
    }
    a.setValue( 0 );

    auto s6 = v.template getSubarrayView< 5 >( 0, 0, 0, 0, 0, 0 );
    const int size6 = s6.template getSize< 0 >();
    const int stride6 = s6.template getStride< 0 >();
    EXPECT_EQ( size6, N );
    EXPECT_EQ( stride6, I*J*K*L*M );
    for( int n = 0; n < N; n++ ) {
        s6( n ) = 1 + n;
        EXPECT_EQ( v( 0, 0, 0, 0, 0, n ), 1 + n );
    }
    a.setValue( 0 );


    auto s_ij = v.template getSubarrayView< 0, 1 >( 0, 0, 0, 0, 0, 0 );
    const int size_ij_0 = s_ij.template getSize< 0 >();
    const int size_ij_1 = s_ij.template getSize< 1 >();
    const int stride_ij_0 = s_ij.template getStride< 0 >();
    const int stride_ij_1 = s_ij.template getStride< 1 >();
    EXPECT_EQ( size_ij_0, I );
    EXPECT_EQ( size_ij_1, J );
    EXPECT_EQ( stride_ij_0, 1 );
    EXPECT_EQ( stride_ij_1, 1 );
    for( int i = 0; i < I; i++ )
    for( int j = 0; j < J; j++ ) {
        s_ij( i, j ) = 1;
        EXPECT_EQ( v( i, j, 0, 0, 0, 0 ), 1 );
    }
    a.setValue( 0 );

    auto s_ik = v.template getSubarrayView< 0, 2 >( 0, 0, 0, 0, 0, 0 );
    const int size_ik_0 = s_ik.template getSize< 0 >();
    const int size_ik_1 = s_ik.template getSize< 1 >();
    const int stride_ik_0 = s_ik.template getStride< 0 >();
    const int stride_ik_1 = s_ik.template getStride< 1 >();
    EXPECT_EQ( size_ik_0, I );
    EXPECT_EQ( size_ik_1, K );
    EXPECT_EQ( stride_ik_0, J );
    EXPECT_EQ( stride_ik_1, 1 );
    for( int i = 0; i < I; i++ )
    for( int k = 0; k < K; k++ ) {
        s_ik( i, k ) = 1 + k;
        EXPECT_EQ( v( i, 0, k, 0, 0, 0 ), 1 + k );
    }
    a.setValue( 0 );

    auto s_il = v.template getSubarrayView< 0, 3 >( 0, 0, 0, 0, 0, 0 );
    const int size_il_0 = s_il.template getSize< 0 >();
    const int size_il_1 = s_il.template getSize< 1 >();
    const int stride_il_0 = s_il.template getStride< 0 >();
    const int stride_il_1 = s_il.template getStride< 1 >();
    EXPECT_EQ( size_il_0, I );
    EXPECT_EQ( size_il_1, L );
    EXPECT_EQ( stride_il_0, J );
    EXPECT_EQ( stride_il_1, K*M );
    for( int i = 0; i < I; i++ )
    for( int l = 0; l < L; l++ ) {
        s_il( i, l ) = 1 + l;
        EXPECT_EQ( v( i, 0, 0, l, 0, 0 ), 1 + l );
    }
    a.setValue( 0 );

    auto s_im = v.template getSubarrayView< 0, 4 >( 0, 0, 0, 0, 0, 0 );
    const int size_im_0 = s_im.template getSize< 0 >();
    const int size_im_1 = s_im.template getSize< 1 >();
    const int stride_im_0 = s_im.template getStride< 0 >();
    const int stride_im_1 = s_im.template getStride< 1 >();
    EXPECT_EQ( size_im_0, I );
    EXPECT_EQ( size_im_1, M );
    EXPECT_EQ( stride_im_0, J );
    EXPECT_EQ( stride_im_1, K );
    for( int i = 0; i < I; i++ )
    for( int m = 0; m < M; m++ ) {
        s_im( i, m ) = 1 + m;
        EXPECT_EQ( v( i, 0, 0, 0, m, 0 ), 1 + m );
    }
    a.setValue( 0 );

    auto s_in = v.template getSubarrayView< 0, 5 >( 0, 0, 0, 0, 0, 0 );
    const int size_in_0 = s_in.template getSize< 0 >();
    const int size_in_1 = s_in.template getSize< 1 >();
    const int stride_in_0 = s_in.template getStride< 0 >();
    const int stride_in_1 = s_in.template getStride< 1 >();
    EXPECT_EQ( size_in_0, I );
    EXPECT_EQ( size_in_1, N );
    EXPECT_EQ( stride_in_0, J );
    EXPECT_EQ( stride_in_1, K*L*M );
    for( int i = 0; i < I; i++ )
    for( int n = 0; n < N; n++ ) {
        s_in( i, n ) = 1 + n;
        EXPECT_EQ( v( i, 0, 0, 0, 0, n ), 1 + n );
    }
    a.setValue( 0 );


    auto s_jk = v.template getSubarrayView< 1, 2 >( 0, 0, 0, 0, 0, 0 );
    const int size_jk_0 = s_jk.template getSize< 0 >();
    const int size_jk_1 = s_jk.template getSize< 1 >();
    const int stride_jk_0 = s_jk.template getStride< 0 >();
    const int stride_jk_1 = s_jk.template getStride< 1 >();
    EXPECT_EQ( size_jk_0, J );
    EXPECT_EQ( size_jk_1, K );
    EXPECT_EQ( stride_jk_0, 1 );
    EXPECT_EQ( stride_jk_1, I );
    for( int j = 0; j < J; j++ )
    for( int k = 0; k < K; k++ ) {
        s_jk( j, k ) = 1 + k;
        EXPECT_EQ( v( 0, j, k, 0, 0, 0 ), 1 + k );
    }
    a.setValue( 0 );

    auto s_jl = v.template getSubarrayView< 1, 3 >( 0, 0, 0, 0, 0, 0 );
    const int size_jl_0 = s_jl.template getSize< 0 >();
    const int size_jl_1 = s_jl.template getSize< 1 >();
    const int stride_jl_0 = s_jl.template getStride< 0 >();
    const int stride_jl_1 = s_jl.template getStride< 1 >();
    EXPECT_EQ( size_jl_0, J );
    EXPECT_EQ( size_jl_1, L );
    EXPECT_EQ( stride_jl_0, 1 );
    EXPECT_EQ( stride_jl_1, I*K*M );
    for( int j = 0; j < J; j++ )
    for( int l = 0; l < L; l++ ) {
        s_jl( j, l ) = 1 + l;
        EXPECT_EQ( v( 0, j, 0, l, 0, 0 ), 1 + l );
    }
    a.setValue( 0 );

    auto s_jm = v.template getSubarrayView< 1, 4 >( 0, 0, 0, 0, 0, 0 );
    const int size_jm_0 = s_jm.template getSize< 0 >();
    const int size_jm_1 = s_jm.template getSize< 1 >();
    const int stride_jm_0 = s_jm.template getStride< 0 >();
    const int stride_jm_1 = s_jm.template getStride< 1 >();
    EXPECT_EQ( size_jm_0, J );
    EXPECT_EQ( size_jm_1, M );
    EXPECT_EQ( stride_jm_0, 1 );
    EXPECT_EQ( stride_jm_1, I*K );
    for( int j = 0; j < J; j++ )
    for( int m = 0; m < M; m++ ) {
        s_jm( j, m ) = 1 + m;
        EXPECT_EQ( v( 0, j, 0, 0, m, 0 ), 1 + m );
    }
    a.setValue( 0 );

    auto s_jn = v.template getSubarrayView< 1, 5 >( 0, 0, 0, 0, 0, 0 );
    const int size_jn_0 = s_jn.template getSize< 0 >();
    const int size_jn_1 = s_jn.template getSize< 1 >();
    const int stride_jn_0 = s_jn.template getStride< 0 >();
    const int stride_jn_1 = s_jn.template getStride< 1 >();
    EXPECT_EQ( size_jn_0, J );
    EXPECT_EQ( size_jn_1, N );
    EXPECT_EQ( stride_jn_0, 1 );
    EXPECT_EQ( stride_jn_1, I*K*L*M );
    for( int j = 0; j < J; j++ )
    for( int n = 0; n < N; n++ ) {
        s_jn( j, n ) = 1 + n;
        EXPECT_EQ( v( 0, j, 0, 0, 0, n ), 1 + n );
    }
    a.setValue( 0 );


    auto s_kl = v.template getSubarrayView< 2, 3 >( 0, 0, 0, 0, 0, 0 );
    const int size_kl_0 = s_kl.template getSize< 0 >();
    const int size_kl_1 = s_kl.template getSize< 1 >();
    const int stride_kl_0 = s_kl.template getStride< 0 >();
    const int stride_kl_1 = s_kl.template getStride< 1 >();
    EXPECT_EQ( size_kl_0, K );
    EXPECT_EQ( size_kl_1, L );
    EXPECT_EQ( stride_kl_0, I*J );
    EXPECT_EQ( stride_kl_1, M );
    for( int k = 0; k < K; k++ )
    for( int l = 0; l < L; l++ ) {
        s_kl( k, l ) = 1 + l;
        EXPECT_EQ( v( 0, 0, k, l, 0, 0 ), 1 + l );
    }
    a.setValue( 0 );

    auto s_km = v.template getSubarrayView< 2, 4 >( 0, 0, 0, 0, 0, 0 );
    const int size_km_0 = s_km.template getSize< 0 >();
    const int size_km_1 = s_km.template getSize< 1 >();
    const int stride_km_0 = s_km.template getStride< 0 >();
    const int stride_km_1 = s_km.template getStride< 1 >();
    EXPECT_EQ( size_km_0, K );
    EXPECT_EQ( size_km_1, M );
    EXPECT_EQ( stride_km_0, I*J );
    EXPECT_EQ( stride_km_1, 1 );
    for( int k = 0; k < K; k++ )
    for( int m = 0; m < M; m++ ) {
        s_km( k, m ) = 1 + m;
        EXPECT_EQ( v( 0, 0, k, 0, m, 0 ), 1 + m );
    }
    a.setValue( 0 );

    auto s_kn = v.template getSubarrayView< 2, 5 >( 0, 0, 0, 0, 0, 0 );
    const int size_kn_0 = s_kn.template getSize< 0 >();
    const int size_kn_1 = s_kn.template getSize< 1 >();
    const int stride_kn_0 = s_kn.template getStride< 0 >();
    const int stride_kn_1 = s_kn.template getStride< 1 >();
    EXPECT_EQ( size_kn_0, K );
    EXPECT_EQ( size_kn_1, N );
    EXPECT_EQ( stride_kn_0, I*J );
    EXPECT_EQ( stride_kn_1, L*M );
    for( int k = 0; k < K; k++ )
    for( int n = 0; n < N; n++ ) {
        s_kn( k, n ) = 1 + n;
        EXPECT_EQ( v( 0, 0, k, 0, 0, n ), 1 + n );
    }
    a.setValue( 0 );


    auto s_lm = v.template getSubarrayView< 3, 4 >( 0, 0, 0, 0, 0, 0 );
    const int size_lm_0 = s_lm.template getSize< 0 >();
    const int size_lm_1 = s_lm.template getSize< 1 >();
    const int stride_lm_0 = s_lm.template getStride< 0 >();
    const int stride_lm_1 = s_lm.template getStride< 1 >();
    EXPECT_EQ( size_lm_0, L );
    EXPECT_EQ( size_lm_1, M );
    EXPECT_EQ( stride_lm_0, 1 );
    EXPECT_EQ( stride_lm_1, I*J*K );
    for( int l = 0; l < L; l++ )
    for( int m = 0; m < M; m++ ) {
        s_lm( l, m ) = 1 + m;
        EXPECT_EQ( v( 0, 0, 0, l, m, 0 ), 1 + m );
    }
    a.setValue( 0 );

    auto s_ln = v.template getSubarrayView< 3, 5 >( 0, 0, 0, 0, 0, 0 );
    const int size_ln_0 = s_ln.template getSize< 0 >();
    const int size_ln_1 = s_ln.template getSize< 1 >();
    const int stride_ln_0 = s_ln.template getStride< 0 >();
    const int stride_ln_1 = s_ln.template getStride< 1 >();
    EXPECT_EQ( size_ln_0, L );
    EXPECT_EQ( size_ln_1, N );
    EXPECT_EQ( stride_ln_0, I*J*K*M );
    EXPECT_EQ( stride_ln_1, 1 );
    for( int l = 0; l < L; l++ )
    for( int n = 0; n < N; n++ ) {
        s_ln( l, n ) = 1 + n;
        EXPECT_EQ( v( 0, 0, 0, l, 0, n ), 1 + n );
    }
    a.setValue( 0 );


    auto s_mn = v.template getSubarrayView< 4, 5 >( 0, 0, 0, 0, 0, 0 );
    const int size_mn_0 = s_mn.template getSize< 0 >();
    const int size_mn_1 = s_mn.template getSize< 1 >();
    const int stride_mn_0 = s_mn.template getStride< 0 >();
    const int stride_mn_1 = s_mn.template getStride< 1 >();
    EXPECT_EQ( size_mn_0, M );
    EXPECT_EQ( size_mn_1, N );
    EXPECT_EQ( stride_mn_0, I*J*K );
    EXPECT_EQ( stride_mn_1, L );
    for( int m = 0; m < M; m++ )
    for( int n = 0; n < N; n++ ) {
        s_mn( m, n ) = 1 + n;
        EXPECT_EQ( v( 0, 0, 0, 0, m, n ), 1 + n );
    }
    a.setValue( 0 );
}
#endif // HAVE_GTEST


#include "../../main.h"
