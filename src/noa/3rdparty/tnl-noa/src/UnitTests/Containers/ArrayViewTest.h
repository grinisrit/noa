#pragma once

#ifdef HAVE_GTEST
#include <type_traits>

#include <TNL/Containers/Array.h>
#include <TNL/Containers/Vector.h>
#include <TNL/Containers/ArrayView.h>
#include <TNL/Containers/VectorView.h>

#include "gtest/gtest.h"

using namespace TNL;
using namespace TNL::Containers;

// minimal custom data structure usable as ValueType in Array
struct MyData
{
   double data;

   __cuda_callable__
   MyData() : data(0) {}

   template< typename T >
   __cuda_callable__
   MyData( T v ) : data(v) {}

   __cuda_callable__
   bool operator==( const MyData& v ) const { return data == v.data; }

   // operator used in tests, not necessary for Array to work
   template< typename T >
   bool operator==( T v ) const { return data == v; }
};

std::ostream& operator<<( std::ostream& str, const MyData& v )
{
   return str << v.data;
}

// test fixture for typed tests
template< typename View >
class ArrayViewTest : public ::testing::Test
{
protected:
   using ViewType = View;
   using ArrayType = Array< typename View::ValueType, typename View::DeviceType, typename View::IndexType >;
};

// types for which ArrayViewTest is instantiated
using ViewTypes = ::testing::Types<
#ifndef __CUDACC__
   // we can't test all types because the argument list would be too long...
//    ArrayView< int,    Devices::Sequential, short >
//   ,ArrayView< long,   Devices::Sequential, short >
//   ,ArrayView< float,  Devices::Sequential, short >
//   ,ArrayView< double, Devices::Sequential, short >
//   ,ArrayView< MyData, Devices::Sequential, short >
//   ,ArrayView< int,    Devices::Sequential, int >
//   ,ArrayView< long,   Devices::Sequential, int >
//   ,ArrayView< float,  Devices::Sequential, int >
//   ,ArrayView< double, Devices::Sequential, int >
//   ,ArrayView< MyData, Devices::Sequential, int >
    ArrayView< int,    Devices::Sequential, long >
   ,ArrayView< long,   Devices::Sequential, long >
   ,ArrayView< float,  Devices::Sequential, long >
   ,ArrayView< double, Devices::Sequential, long >
   ,ArrayView< MyData, Devices::Sequential, long >

   ,ArrayView< int,    Devices::Host, short >
   ,ArrayView< long,   Devices::Host, short >
   ,ArrayView< float,  Devices::Host, short >
   ,ArrayView< double, Devices::Host, short >
   ,ArrayView< MyData, Devices::Host, short >
   ,ArrayView< int,    Devices::Host, int >
   ,ArrayView< long,   Devices::Host, int >
   ,ArrayView< float,  Devices::Host, int >
   ,ArrayView< double, Devices::Host, int >
   ,ArrayView< MyData, Devices::Host, int >
   ,ArrayView< int,    Devices::Host, long >
   ,ArrayView< long,   Devices::Host, long >
   ,ArrayView< float,  Devices::Host, long >
   ,ArrayView< double, Devices::Host, long >
   ,ArrayView< MyData, Devices::Host, long >
#endif
#ifdef __CUDACC__
    ArrayView< int,    Devices::Cuda, short >
   ,ArrayView< long,   Devices::Cuda, short >
   ,ArrayView< float,  Devices::Cuda, short >
   ,ArrayView< double, Devices::Cuda, short >
   ,ArrayView< MyData, Devices::Cuda, short >
   ,ArrayView< int,    Devices::Cuda, int >
   ,ArrayView< long,   Devices::Cuda, int >
   ,ArrayView< float,  Devices::Cuda, int >
   ,ArrayView< double, Devices::Cuda, int >
   ,ArrayView< MyData, Devices::Cuda, int >
   ,ArrayView< int,    Devices::Cuda, long >
   ,ArrayView< long,   Devices::Cuda, long >
   ,ArrayView< float,  Devices::Cuda, long >
   ,ArrayView< double, Devices::Cuda, long >
   ,ArrayView< MyData, Devices::Cuda, long >
#endif

   // all ArrayView tests should also work with VectorView
   // (but we can't test all types because the argument list would be too long...)
#ifndef __CUDACC__
   ,
   VectorView< float,  Devices::Sequential, long >,
   VectorView< double, Devices::Sequential, long >,
   VectorView< float,  Devices::Host, long >,
   VectorView< double, Devices::Host, long >
#endif
#ifdef __CUDACC__
   ,
   VectorView< float,  Devices::Cuda, long >,
   VectorView< double, Devices::Cuda, long >
#endif
>;

TYPED_TEST_SUITE( ArrayViewTest, ViewTypes );


TYPED_TEST( ArrayViewTest, constructors )
{
   using ArrayType = typename TestFixture::ArrayType;
   using ViewType = typename TestFixture::ViewType;
   using ConstViewType = typename ViewType::ConstViewType;

   ArrayType a( 10 );
   EXPECT_EQ( a.getSize(), 10 );

   ViewType v = a.getView();
   EXPECT_EQ( v.getSize(), 10 );
   EXPECT_EQ( v.getData(), a.getData() );

   if( std::is_same< typename ArrayType::DeviceType, Devices::Host >::value ) {
      typename ArrayType::ValueType data[ 10 ];
      ViewType w( data, 10 );
      EXPECT_EQ( w.getData(), data );

      ViewType z( w );
      EXPECT_EQ( z.getData(), data );
      EXPECT_EQ( z.getSize(), 10 );
   }

   // test initialization by const reference
   const ArrayType& b = a;
   ConstViewType b_view = b.getConstView();
   EXPECT_EQ( b_view.getData(), b.getData() );
   ConstViewType const_a_view = a.getConstView();
   EXPECT_EQ( const_a_view.getData(), a.getData() );
   EXPECT_EQ( const_a_view.getSize(), a.getSize() );

   // test initialization of const view by non-const view
   ConstViewType const_b_view( b_view );
   EXPECT_EQ( const_b_view.getData(), b.getData() );
   EXPECT_EQ( const_b_view.getSize(), b.getSize() );
}

TYPED_TEST( ArrayViewTest, bind )
{
   using ArrayType = typename TestFixture::ArrayType;
   using ViewType = typename TestFixture::ViewType;

   ArrayType a( 10 );
   ViewType v;
   v.bind( a );
   EXPECT_EQ( v.getSize(), a.getSize() );
   EXPECT_EQ( v.getData(), a.getData() );

   // setting values
   a.setValue( 27 );
   EXPECT_EQ( a.getElement( 0 ), 27 );
   v.setValue( 50 );
   EXPECT_EQ( a.getElement( 0 ), 50 );
   a.reset();
   EXPECT_EQ( a.getSize(), 0 );
   EXPECT_EQ( v.getSize(), 10 );

   ArrayType b = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
   EXPECT_EQ( b.getSize(), 10 );
   EXPECT_EQ( b.getElement( 1 ), 2 );
   v.bind( b );
   EXPECT_EQ( v.getElement( 1 ), 2 );
   v.setElement( 1, 3 );
   EXPECT_EQ( b.getElement( 1 ), 3 );
}

TYPED_TEST( ArrayViewTest, swap )
{
   using ArrayType = typename TestFixture::ArrayType;
   using ViewType = typename TestFixture::ViewType;

   ArrayType a( 10 ), b( 20 );
   a.setValue( 0 );
   b.setValue( 1 );

   ViewType u = a.getView();
   ViewType v = b.getView();
   u.swap( v );
   EXPECT_EQ( u.getSize(), 20 );
   EXPECT_EQ( v.getSize(), 10 );
   for( int i = 0; i < 20; i++ )
      EXPECT_EQ( u.getElement( i ), 1 );
   for( int i = 0; i < 10; i++ )
      EXPECT_EQ( v.getElement( i ), 0 );
}

TYPED_TEST( ArrayViewTest, reset )
{
   using ArrayType = typename TestFixture::ArrayType;
   using ViewType = typename TestFixture::ViewType;

   ArrayType a;
   a.setSize( 100 );
   ViewType u = a.getView();
   EXPECT_EQ( u.getSize(), 100 );
   EXPECT_NE( u.getData(), nullptr );
   u.reset();
   EXPECT_EQ( u.getSize(), 0 );
   EXPECT_EQ( u.getData(), nullptr );
   u.bind( a );
   EXPECT_EQ( u.getSize(), 100 );
   EXPECT_NE( u.getData(), nullptr );
   u.reset();
   EXPECT_EQ( u.getSize(), 0 );
   EXPECT_EQ( u.getData(), nullptr );
}

template< typename Value, typename Index >
void testArrayViewElementwiseAccess( Array< Value, Devices::Sequential, Index >&& a )
{
   a.setSize( 10 );
   using ViewType = ArrayView< Value, Devices::Sequential, Index >;
   ViewType u( a );
   for( int i = 0; i < 10; i++ ) {
      u.setElement( i, i );
      EXPECT_EQ( u.getData()[ i ], i );
      EXPECT_EQ( u.getElement( i ), i );
      EXPECT_EQ( u[ i ], i );
      EXPECT_EQ( u( i ), i );
   }
}

template< typename Value, typename Index >
void testArrayViewElementwiseAccess( Array< Value, Devices::Host, Index >&& a )
{
   a.setSize( 10 );
   using ViewType = ArrayView< Value, Devices::Host, Index >;
   ViewType u( a );
   for( int i = 0; i < 10; i++ ) {
      u.setElement( i, i );
      EXPECT_EQ( u.getData()[ i ], i );
      EXPECT_EQ( u.getElement( i ), i );
      EXPECT_EQ( u[ i ], i );
      EXPECT_EQ( u( i ), i );
   }
}

#ifdef __CUDACC__
template< typename ValueType, typename IndexType >
__global__ void testSetGetElementKernel( ArrayView< ValueType, Devices::Cuda, IndexType > u,
                                         ArrayView< ValueType, Devices::Cuda, IndexType > v )
{
   if( threadIdx.x < v.getSize() )
      u[ threadIdx.x ] = v( threadIdx.x ) = threadIdx.x;
}
#endif // __CUDACC__

template< typename Value, typename Index >
void testArrayViewElementwiseAccess( Array< Value, Devices::Cuda, Index >&& a )
{
#ifdef __CUDACC__
   using ArrayType = Array< Value, Devices::Cuda, Index >;
   using ViewType = ArrayView< Value, Devices::Cuda, Index >;
   a.setSize( 10 );
   ArrayType b( 10 );
   ViewType u( a ), v( b );
   testSetGetElementKernel<<< 1, 16 >>>( u, v );
   TNL_CHECK_CUDA_DEVICE;
   for( int i = 0; i < 10; i++ ) {
      EXPECT_EQ( a.getElement( i ), i );
      EXPECT_EQ( b.getElement( i ), i );
   }
#endif
}

TYPED_TEST( ArrayViewTest, elementwiseAccess )
{
   using ArrayType = typename TestFixture::ArrayType;

   testArrayViewElementwiseAccess( ArrayType() );
}

template< typename ArrayType >
void ArrayViewEvaluateTest( ArrayType& u )
{
   using ValueType = typename ArrayType::ValueType;
   using DeviceType = typename ArrayType::DeviceType;
   using IndexType = typename ArrayType::IndexType;
   using ViewType = ArrayView< ValueType, DeviceType, IndexType >;
   ViewType v( u );

   v.forAllElements( [] __cuda_callable__ ( IndexType i, ValueType& value ) { value = 3 * i % 4; } );

   for( int i = 0; i < 10; i++ )
   {
      EXPECT_EQ( u.getElement( i ), 3 * i % 4 );
      EXPECT_EQ( v.getElement( i ), 3 * i % 4 );
   }
}

template< typename ArrayType >
void test_setElement()
{
   ArrayType a( 10, 0 ), b( 10, 0 );
   auto a_view = a.getView();
   auto b_view = b.getView();
   auto set = [=] __cuda_callable__ ( int i ) mutable {
      a_view.setElement( i, i );
      b_view.setElement( i, a_view.getElement( i ) );
   };
   Algorithms::ParallelFor< typename ArrayType::DeviceType >::exec( 0, 10, set );
   for( int i = 0; i < 10; i++ )
   {
      EXPECT_EQ( a.getElement( i ), i );
      EXPECT_EQ( b.getElement( i ), i );
   }
}

TYPED_TEST( ArrayViewTest, setElement )
{
   using ArrayType = typename TestFixture::ArrayType;

   ArrayType a( 10 );
   auto a_view = a.getView();
   for( int i = 0; i < 10; i++ )
      a_view.setElement( i, i );

   for( int i = 0; i < 10; i++ )
      EXPECT_EQ( a_view.getElement( i ), i );

   test_setElement< ArrayType >();
}


TYPED_TEST( ArrayViewTest, evaluate )
{
   using ArrayType = typename TestFixture::ArrayType;
   ArrayType u( 10 );
   ArrayViewEvaluateTest( u );
}

TYPED_TEST( ArrayViewTest, comparisonOperator )
{
   using ArrayType = typename TestFixture::ArrayType;
   using ViewType = typename TestFixture::ViewType;
   using HostArrayType = typename ArrayType::template Self< typename ArrayType::ValueType, Devices::Sequential >;

   ArrayType a( 10 ), b( 10 );
   HostArrayType a_host( 10 );
   for( int i = 0; i < 10; i ++ ) {
      a.setElement( i, i );
      a_host.setElement( i, i );
      b.setElement( i, 2 * i );
   }

   ViewType u = a.getView();
   ViewType v = a.getView();
   ViewType w = b.getView();

   EXPECT_TRUE( u == u );
   EXPECT_TRUE( u == v );
   EXPECT_TRUE( v == u );
   EXPECT_FALSE( u != v );
   EXPECT_FALSE( v != u );
   EXPECT_TRUE( u != w );
   EXPECT_TRUE( w != u );
   EXPECT_FALSE( u == w );
   EXPECT_FALSE( w == u );

   // comparison with arrays
   EXPECT_TRUE( a == u );
   EXPECT_FALSE( a != u );
   EXPECT_TRUE( u == a );
   EXPECT_FALSE( u != a );
   EXPECT_TRUE( a != w );
   EXPECT_FALSE( a == w );

   // comparison with different device
   EXPECT_TRUE( u == a_host );
   EXPECT_TRUE( a_host == u );
   EXPECT_TRUE( w != a_host );
   EXPECT_TRUE( a_host != w );

   v.reset();
   EXPECT_FALSE( u == v );
   u.reset();
   EXPECT_TRUE( u == v );
}

TYPED_TEST( ArrayViewTest, comparisonOperatorWithDifferentType )
{
   using DeviceType = typename TestFixture::ArrayType::DeviceType;
   using ArrayType1 = Array< short, DeviceType >;
   using ArrayType2 = Array< float, Devices::Host >;
   using ViewType1 = ArrayView< short, DeviceType >;
   using ViewType2 = ArrayView< float, Devices::Host >;

   ArrayType1 a( 10 );
   ArrayType2 b( 10 );
   for( int i = 0; i < 10; i++ ) {
      a.setElement( i, i );
      b.setElement( i, i );
   }

   ViewType1 u = a.getView();
   ViewType2 v = b.getView();

   EXPECT_TRUE( u == v );
   EXPECT_FALSE( u != v );

   // the comparison will be in floats
   v.setElement( 0, 0.1f );
   EXPECT_FALSE( u == v );
   EXPECT_TRUE( u != v );
}

TYPED_TEST( ArrayViewTest, assignmentOperator )
{
   using ArrayType = typename TestFixture::ArrayType;
   using ViewType = typename TestFixture::ViewType;
   using ConstViewType = VectorView< const typename ArrayType::ValueType, typename ArrayType::DeviceType, typename ArrayType::IndexType >;
   using HostArrayType = typename ArrayType::template Self< typename ArrayType::ValueType, Devices::Sequential >;
   using HostViewType = typename HostArrayType::ViewType;

   ArrayType a( 10 ), b( 10 );
   HostArrayType a_host( 10 );
   for( int i = 0; i < 10; i++ ) {
      a.setElement( i, i );
      a_host.setElement( i, i );
   }

   ViewType u = a.getView();
   ViewType v = b.getView();
   HostViewType u_host = a_host.getView();

   v.setValue( 0 );
   v = u;
   EXPECT_TRUE( u == v );
   EXPECT_EQ( v.getData(), b.getData() );

   // assignment from host to device
   v.setValue( 0 );
   v = u_host;
   EXPECT_EQ( u, v );
   EXPECT_EQ( v.getData(), b.getData() );

   // assignment from device to host
   u_host.setValue( 0 );
   u_host = u;
   EXPECT_EQ( u_host, u );
   EXPECT_EQ( u_host.getData(), a_host.getData() );

   // assignment of const view to non-const view
   v.setValue( 0 );
   ConstViewType c( u );
   v = c;

   // assignment of a scalar
   u = 42;
   for( int i = 0; i < 10; i++ )
      EXPECT_EQ( u.getElement( i ), 42 );
   EXPECT_EQ( u.getData(), a.getData() );
}

// test works only for arithmetic types
template< typename ArrayType,
          typename = typename std::enable_if< std::is_arithmetic< typename ArrayType::ValueType >::value >::type >
void testArrayAssignmentWithDifferentType()
{
   using HostArrayType = typename ArrayType::template Self< typename ArrayType::ValueType, Devices::Sequential >;

   ArrayType a( 10 );
   Array< short, typename ArrayType::DeviceType, short > b( 10 );
   Array< short, Devices::Sequential, short > b_host( 10 );
   HostArrayType a_host( 10 );
   for( int i = 0; i < 10; i++ ) {
      a.setElement( i, i );
      a_host.setElement( i, i );
   }

   using ViewType = ArrayView< typename ArrayType::ValueType, typename ArrayType::DeviceType, typename ArrayType::IndexType >;
   using HostViewType = typename ViewType::template Self< typename ViewType::ValueType, Devices::Sequential >;
   ViewType u = a.getView();
   HostViewType u_host( a_host );
   using ShortViewType = ArrayView< short, typename ArrayType::DeviceType, short >;
   using HostShortViewType = ArrayView< short, Devices::Sequential, short >;
   ShortViewType v( b );
   HostShortViewType v_host( b_host );

   v.setValue( 0 );
   v = u;
   EXPECT_EQ( v, u );
   EXPECT_EQ( v.getData(), b.getData() );

   // assignment from host to device
   v.setValue( 0 );
   v = u_host;
   EXPECT_EQ( v, u_host );
   EXPECT_EQ( v.getData(), b.getData() );

   // assignment from device to host
   v_host.setValue( 0 );
   v_host = u;
   EXPECT_EQ( v_host, u );
   EXPECT_EQ( v_host.getData(), b_host.getData() );
}

template< typename ArrayType,
          typename = typename std::enable_if< ! std::is_arithmetic< typename ArrayType::ValueType >::value >::type,
          typename = void >
void testArrayAssignmentWithDifferentType()
{
}

TYPED_TEST( ArrayViewTest, assignmentOperatorWithDifferentType )
{
   using ArrayType = typename TestFixture::ArrayType;

   testArrayAssignmentWithDifferentType< ArrayType >();
}

// TODO: test all __cuda_callable__ methods from a CUDA kernel

#endif // HAVE_GTEST


#include "../main.h"
