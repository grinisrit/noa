#pragma once

#ifdef HAVE_GTEST
#include <type_traits>

#include <TNL/Containers/Array.h>
#include <TNL/Containers/Vector.h>
#include <TNL/Pointers/DevicePointer.h>

#include "gtest/gtest.h"

using namespace TNL;
using namespace TNL::Containers;

#ifdef HAVE_CUDA
static const char* TEST_FILE_NAME = "test_ArrayTestCuda.tnl";
#else
static const char* TEST_FILE_NAME = "test_ArrayTest.tnl";
#endif

// minimal custom data structure usable as ValueType in Array
struct MyData
{
   double data;

   __cuda_callable__
   MyData() : data(0) {}

   __cuda_callable__
   MyData( double v ) : data(v) {}

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
template< typename Array >
class ArrayTest : public ::testing::Test
{
protected:
   using ArrayType = Array;
};

// types for which ArrayTest is instantiated
using ArrayTypes = ::testing::Types<
#ifndef HAVE_CUDA
   // we can't test all types because the argument list would be too long...
//   Array< int,    Devices::Sequential, short >,
//   Array< long,   Devices::Sequential, short >,
//   Array< float,  Devices::Sequential, short >,
//   Array< double, Devices::Sequential, short >,
//   Array< MyData, Devices::Sequential, short >,
//   Array< int,    Devices::Sequential, int >,
//   Array< long,   Devices::Sequential, int >,
//   Array< float,  Devices::Sequential, int >,
//   Array< double, Devices::Sequential, int >,
//   Array< MyData, Devices::Sequential, int >,
   Array< int,    Devices::Sequential, long >,
   Array< long,   Devices::Sequential, long >,
   Array< float,  Devices::Sequential, long >,
   Array< double, Devices::Sequential, long >,
   Array< MyData, Devices::Sequential, long >,

   Array< int,    Devices::Host, short >,
   Array< long,   Devices::Host, short >,
   Array< float,  Devices::Host, short >,
   Array< double, Devices::Host, short >,
   Array< MyData, Devices::Host, short >,
   Array< int,    Devices::Host, int >,
   Array< long,   Devices::Host, int >,
   Array< float,  Devices::Host, int >,
   Array< double, Devices::Host, int >,
   Array< MyData, Devices::Host, int >,
   Array< int,    Devices::Host, long >,
   Array< long,   Devices::Host, long >,
   Array< float,  Devices::Host, long >,
   Array< double, Devices::Host, long >,
   Array< MyData, Devices::Host, long >
#endif
#ifdef HAVE_CUDA
   Array< int,    Devices::Cuda, short >,
   Array< long,   Devices::Cuda, short >,
   Array< float,  Devices::Cuda, short >,
   Array< double, Devices::Cuda, short >,
   Array< MyData, Devices::Cuda, short >,
   Array< int,    Devices::Cuda, int >,
   Array< long,   Devices::Cuda, int >,
   Array< float,  Devices::Cuda, int >,
   Array< double, Devices::Cuda, int >,
   Array< MyData, Devices::Cuda, int >,
   Array< int,    Devices::Cuda, long >,
   Array< long,   Devices::Cuda, long >,
   Array< float,  Devices::Cuda, long >,
   Array< double, Devices::Cuda, long >,
   Array< MyData, Devices::Cuda, long >
#endif

   // all array tests should also work with Vector
   // (but we can't test all types because the argument list would be too long...)
#ifndef HAVE_CUDA
   ,
   Vector< float,  Devices::Sequential, long >,
   Vector< double, Devices::Sequential, long >,
   Vector< float,  Devices::Host, long >,
   Vector< double, Devices::Host, long >
#endif
#ifdef HAVE_CUDA
   ,
   Vector< float,  Devices::Cuda, long >,
   Vector< double, Devices::Cuda, long >
#endif
>;

TYPED_TEST_SUITE( ArrayTest, ArrayTypes );


TYPED_TEST( ArrayTest, constructors )
{
   using ArrayType = typename TestFixture::ArrayType;

   ArrayType empty_u;
   ArrayType empty_v( empty_u );
   EXPECT_EQ( empty_u.getSize(), 0 );
   EXPECT_EQ( empty_v.getSize(), 0 );

   ArrayType u;
   EXPECT_EQ( u.getSize(), 0 );
   u.setSize( 10 );
   EXPECT_EQ( u.getSize(), 10 );

   ArrayType v( 10 );
   EXPECT_EQ( v.getSize(), 10 );
   v = 0;
   EXPECT_EQ( v.getSize(), 10 );

   ArrayType vv( 10, 4 );
   EXPECT_EQ( vv.getSize(), 10 );
   for( int i = 0; i < 10; i++ )
      EXPECT_EQ( vv.getElement( i ), 4 );

   // deep copy
   ArrayType w( v );
   EXPECT_NE( w.getData(), v.getData() );
   EXPECT_EQ( w.getSize(), v.getSize() );
   for( int i = 0; i < 10; i++ )
      EXPECT_EQ( v.getElement( i ), w.getElement( i ) );
   v.reset();
   EXPECT_EQ( w.getSize(), 10 );

   Containers::Array< int > int_array( 10, 1 );
   ArrayType int_array_copy( int_array );
   for( int i = 0; i < 10; i++ )
      EXPECT_EQ( int_array_copy.getElement( i ), 1 );

   ArrayType a1 { 1, 2, 3 };
   EXPECT_EQ( a1.getElement( 0 ), 1 );
   EXPECT_EQ( a1.getElement( 1 ), 2 );
   EXPECT_EQ( a1.getElement( 2 ), 3 );

   std::list< int > l = { 4, 5, 6 };
   ArrayType a2( l );
   EXPECT_EQ( a2.getElement( 0 ), 4 );
   EXPECT_EQ( a2.getElement( 1 ), 5 );
   EXPECT_EQ( a2.getElement( 2 ), 6 );

   std::vector< int > q = { 7, 8, 9 };
   ArrayType a3( q );
   EXPECT_EQ( a3.getElement( 0 ), 7 );
   EXPECT_EQ( a3.getElement( 1 ), 8 );
   EXPECT_EQ( a3.getElement( 2 ), 9 );
}

TYPED_TEST( ArrayTest, constructorsWithAllocators )
{
   using ArrayType = typename TestFixture::ArrayType;
   using AllocatorType = typename ArrayType::AllocatorType;

   AllocatorType allocator;

   ArrayType u( allocator );
   EXPECT_EQ( u.getAllocator(), allocator );
   u.setSize( 10 );
   EXPECT_EQ( u.getSize(), 10 );

   ArrayType v( 10, allocator );
   EXPECT_EQ( v.getSize(), 10 );
   EXPECT_EQ( v.getAllocator(), allocator );
   v.reset();
   EXPECT_EQ( v.getAllocator(), allocator );

   // deep copy
   u = 0;   // floating-point values have to be initialized before comparison, because nan != nan
   ArrayType w( u, allocator );
   EXPECT_NE( w.getData(), u.getData() );
   EXPECT_EQ( w.getSize(), u.getSize() );
   for( int i = 0; i < 10; i++ )
      EXPECT_EQ( u.getElement( i ), w.getElement( i ) );
   EXPECT_EQ( w.getAllocator(), allocator );
   u.reset();
   EXPECT_EQ( w.getSize(), 10 );

   ArrayType a1( { 1, 2, 3 }, allocator );
   EXPECT_EQ( a1.getElement( 0 ), 1 );
   EXPECT_EQ( a1.getElement( 1 ), 2 );
   EXPECT_EQ( a1.getElement( 2 ), 3 );
   EXPECT_EQ( a1.getAllocator(), allocator );

   std::list< int > l = { 4, 5, 6 };
   ArrayType a2( l, allocator );
   EXPECT_EQ( a2.getElement( 0 ), 4 );
   EXPECT_EQ( a2.getElement( 1 ), 5 );
   EXPECT_EQ( a2.getElement( 2 ), 6 );
   EXPECT_EQ( a2.getAllocator(), allocator );

   std::vector< int > q = { 7, 8, 9 };
   ArrayType a3( q, allocator );
   EXPECT_EQ( a3.getElement( 0 ), 7 );
   EXPECT_EQ( a3.getElement( 1 ), 8 );
   EXPECT_EQ( a3.getElement( 2 ), 9 );
   EXPECT_EQ( a3.getAllocator(), allocator );

   // test value-initialization of non-fundamental types
   if( ! std::is_fundamental< typename ArrayType::ValueType >::value )
   {
      const typename ArrayType::ValueType init{};
      ArrayType a( 42 );
      ASSERT_EQ( a.getSize(), 42 );
      for( int i = 0; i < a.getSize(); i++ )
         EXPECT_EQ( a.getElement( i ), init ) << "i = " << i;
   }
}

TYPED_TEST( ArrayTest, resize )
{
   using ArrayType = typename TestFixture::ArrayType;

   ArrayType u( 42 );
   ASSERT_EQ( u.getSize(), 42 );
   for( int i = 0; i < u.getSize(); i++ )
      u.setElement( i, i );

   // no change test
   const typename ArrayType::ValueType* old_data = u.getData();
   u.resize( u.getSize() );
   EXPECT_EQ( u.getData(), old_data );

   // shrink test
   u.resize( 20 );
   ASSERT_EQ( u.getSize(), 20 );
   EXPECT_NE( u.getData(), old_data );
   for( int i = 0; i < u.getSize(); i++ )
      EXPECT_EQ( u.getElement( i ), i );

   // expand test
   const typename ArrayType::IndexType old_size = u.getSize();
   old_data = u.getData();
   u.resize( old_size * 2 );
   ASSERT_EQ( u.getSize(), old_size * 2 );
   EXPECT_NE( u.getData(), old_data );
   for( int i = 0; i < old_size; i++ )
      EXPECT_EQ( u.getElement( i ), i );

   // expand test with initial value
   const typename ArrayType::ValueType init = 3;
   ArrayType v( 10 );
   v.setValue( 0 );
   v.resize( 42, init );
   ASSERT_EQ( v.getSize(), 42 );
   for( int i = 0; i < 10; i++ )
      EXPECT_EQ( v.getElement( i ), 0 ) << "i = " << i;
   for( int i = 10; i < v.getSize(); i++ )
      EXPECT_EQ( v.getElement( i ), init ) << "i = " << i;

   // test value-initialization of non-fundamental types
   if( ! std::is_fundamental< typename ArrayType::ValueType >::value )
   {
      const typename ArrayType::ValueType init{};
      ArrayType w;
      w.resize( 42 );
      ASSERT_EQ( w.getSize(), 42 );
      for( int i = 0; i < w.getSize(); i++ )
         EXPECT_EQ( w.getElement( i ), init ) << "i = " << i;
   }
}

TYPED_TEST( ArrayTest, setSize )
{
   using ArrayType = typename TestFixture::ArrayType;

   ArrayType u;
   const int maxSize = 10;
   for( int i = 0; i <= maxSize; i ++ ) {
      u.setSize( i );
      EXPECT_EQ( u.getSize(), i );
   }
}

TYPED_TEST( ArrayTest, empty )
{
   using ArrayType = typename TestFixture::ArrayType;
   ArrayType u( 10 );

   EXPECT_FALSE( u.empty() );
   u.reset();
   EXPECT_TRUE( u.empty() );
}

TYPED_TEST( ArrayTest, setLike )
{
   using ArrayType = typename TestFixture::ArrayType;

   ArrayType u( 10 );
   EXPECT_EQ( u.getSize(), 10 );

   ArrayType v;
   v.setLike( u );
   EXPECT_EQ( v.getSize(), u.getSize() );
   EXPECT_NE( v.getData(), u.getData() );
}

TYPED_TEST( ArrayTest, swap )
{
   using ArrayType = typename TestFixture::ArrayType;

   ArrayType u( 10 ), v( 20 );
   u.setValue( 0 );
   v.setValue( 1 );
   u.swap( v );
   EXPECT_EQ( u.getSize(), 20 );
   EXPECT_EQ( v.getSize(), 10 );
   for( int i = 0; i < 20; i++ )
      EXPECT_EQ( u.getElement( i ), 1 );
   for( int i = 0; i < 10; i++ )
      EXPECT_EQ( v.getElement( i ), 0 );
}

TYPED_TEST( ArrayTest, reset )
{
   using ArrayType = typename TestFixture::ArrayType;

   ArrayType u;
   u.setSize( 100 );
   EXPECT_EQ( u.getSize(), 100 );
   EXPECT_FALSE( u.empty() );
   EXPECT_NE( u.getData(), nullptr );
   u.reset();
   EXPECT_EQ( u.getSize(), 0 );
   EXPECT_TRUE( u.empty() );
   EXPECT_EQ( u.getData(), nullptr );
   u.setSize( 100 );
   EXPECT_EQ( u.getSize(), 100 );
   EXPECT_FALSE( u.empty() );
   EXPECT_NE( u.getData(), nullptr );
   u.reset();
   EXPECT_EQ( u.getSize(), 0 );
   EXPECT_TRUE( u.empty() );
   EXPECT_EQ( u.getData(), nullptr );
}

template< typename Value, typename Index >
void testArrayElementwiseAccess( Array< Value, Devices::Sequential, Index >&& u )
{
   u.setSize( 10 );
   for( int i = 0; i < 10; i++ ) {
      u.setElement( i, i );
      EXPECT_EQ( u.getData()[ i ], i );
      EXPECT_EQ( u.getElement( i ), i );
      EXPECT_EQ( u[ i ], i );
      EXPECT_EQ( u( i ), i );
   }
}

template< typename Value, typename Index >
void testArrayElementwiseAccess( Array< Value, Devices::Host, Index >&& u )
{
   u.setSize( 10 );
   for( int i = 0; i < 10; i++ ) {
      u.setElement( i, i );
      EXPECT_EQ( u.getData()[ i ], i );
      EXPECT_EQ( u.getElement( i ), i );
      EXPECT_EQ( u[ i ], i );
      EXPECT_EQ( u( i ), i );
   }
}

#ifdef HAVE_CUDA
template< typename ValueType, typename IndexType >
__global__ void testSetGetElementKernel( Array< ValueType, Devices::Cuda, IndexType >* u,
                                         Array< ValueType, Devices::Cuda, IndexType >* v )
{
   if( threadIdx.x < u->getSize() )
      ( *u )[ threadIdx.x ] = ( *v )( threadIdx.x ) = threadIdx.x;
}
#endif /* HAVE_CUDA */

template< typename Value, typename Index >
void testArrayElementwiseAccess( Array< Value, Devices::Cuda, Index >&& u )
{
#ifdef HAVE_CUDA
   using ArrayType = Array< Value, Devices::Cuda, Index >;
   u.setSize( 10 );
   ArrayType v( 10 );
   Pointers::DevicePointer< ArrayType > kernel_u( u ), kernel_v( v );
   testSetGetElementKernel<<< 1, 16 >>>( &kernel_u.template modifyData< Devices::Cuda >(), &kernel_v.template modifyData< Devices::Cuda >() );
   cudaDeviceSynchronize();
   TNL_CHECK_CUDA_DEVICE;
   for( int i = 0; i < 10; i++ ) {
      EXPECT_EQ( u.getElement( i ), i );
      EXPECT_EQ( v.getElement( i ), i );
   }
#endif
}

TYPED_TEST( ArrayTest, elementwiseAccess )
{
   using ArrayType = typename TestFixture::ArrayType;

   testArrayElementwiseAccess( ArrayType() );
}

template< typename Value, typename Index >
void test_setElement_on_device( const Array< Value, Devices::Sequential, Index >& )
{
}

template< typename Value, typename Index >
void test_setElement_on_device( const Array< Value, Devices::Host, Index >& )
{
}

#ifdef HAVE_CUDA
template< typename ValueType, typename IndexType >
__global__ void test_setElement_on_device_kernel( Array< ValueType, Devices::Cuda, IndexType >* a,
                                                  Array< ValueType, Devices::Cuda, IndexType >* b )
{
   if( threadIdx.x < a->getSize() ) {
      a->setElement( threadIdx.x, threadIdx.x );
      b->setElement( threadIdx.x, a->getElement( threadIdx.x ) );
   }
}
#endif /* HAVE_CUDA */

template< typename Value, typename Index >
void test_setElement_on_device( const Array< Value, Devices::Cuda, Index >& )
{
#ifdef HAVE_CUDA
   using ArrayType = Array< Value, Devices::Cuda, Index >;
   ArrayType a( 10, 0 ), b( 10, 0 );
   Pointers::DevicePointer< ArrayType > kernel_a( a );
   Pointers::DevicePointer< ArrayType > kernel_b( b );
   test_setElement_on_device_kernel<<< 1, 16 >>>( &kernel_a.template modifyData< Devices::Cuda >(),
                                                  &kernel_b.template modifyData< Devices::Cuda >() );
   cudaDeviceSynchronize();
   TNL_CHECK_CUDA_DEVICE;
   for( int i = 0; i < 10; i++ ) {
      EXPECT_EQ( a.getElement( i ), i );
      EXPECT_EQ( b.getElement( i ), i );
   }
#endif
}

TYPED_TEST( ArrayTest, setElement )
{
   using ArrayType = typename TestFixture::ArrayType;

   ArrayType a( 10 );
   for( int i = 0; i < 10; i++ )
      a.setElement( i, i );

   for( int i = 0; i < 10; i++ )
      EXPECT_EQ( a.getElement( i ), i );

   test_setElement_on_device( a );
}

// test must be in a plain function because nvcc sucks (extended lambdas are
// not allowed to be defined in protected class member functions)
template< typename ArrayType >
void testArrayForEachElement()
{
   using IndexType = typename ArrayType::IndexType;
   using ValueType = typename ArrayType::ValueType;

   ArrayType a( 10 );
   a.forAllElements( [] __cuda_callable__ ( IndexType i, ValueType& v ) mutable { v = i; } );

   for( int i = 0; i < 10; i++ )
      EXPECT_EQ( a.getElement( i ), i );
}
TYPED_TEST( ArrayTest, forElements )
{
   testArrayForEachElement< typename TestFixture::ArrayType >();
}

TYPED_TEST( ArrayTest, comparisonOperator )
{
   using ArrayType = typename TestFixture::ArrayType;
   using HostArrayType = typename ArrayType::template Self< typename ArrayType::ValueType, Devices::Sequential >;

   ArrayType u( 10 ), v( 10 ), w( 10 );
   HostArrayType u_host( 10 );
   for( int i = 0; i < 10; i ++ ) {
      u.setElement( i, i );
      u_host.setElement( i, i );
      v.setElement( i, i );
      w.setElement( i, 2 * i );
   }
   EXPECT_TRUE( u == u );
   EXPECT_TRUE( u == v );
   EXPECT_TRUE( v == u );
   EXPECT_FALSE( u != v );
   EXPECT_FALSE( v != u );
   EXPECT_TRUE( u != w );
   EXPECT_TRUE( w != u );
   EXPECT_FALSE( u == w );
   EXPECT_FALSE( w == u );

   // comparison with different device
   EXPECT_TRUE( u == u_host );
   EXPECT_TRUE( u_host == u );
   EXPECT_TRUE( w != u_host );
   EXPECT_TRUE( u_host != w );

   v.setSize( 0 );
   EXPECT_FALSE( u == v );
   u.setSize( 0 );
   EXPECT_TRUE( u == v );
}

TYPED_TEST( ArrayTest, comparisonOperatorWithDifferentType )
{
   using DeviceType = typename TestFixture::ArrayType::DeviceType;
   using ArrayType1 = Array< short, DeviceType >;
   using ArrayType2 = Array< float, Devices::Host >;

   ArrayType1 u( 10 );
   ArrayType2 v( 10 );
   for( int i = 0; i < 10; i++ ) {
      u.setElement( i, i );
      v.setElement( i, i );
   }
   EXPECT_TRUE( u == v );
   EXPECT_FALSE( u != v );

   // the comparison will be in floats
   v.setElement( 0, 0.1f );
   EXPECT_FALSE( u == v );
   EXPECT_TRUE( u != v );
}

TYPED_TEST( ArrayTest, assignmentOperator )
{
   using ArrayType = typename TestFixture::ArrayType;
   using HostArrayType = typename ArrayType::template Self< typename ArrayType::ValueType, Devices::Sequential >;

   ArrayType u( 10 ), v( 10 );
   HostArrayType u_host( 10 );
   for( int i = 0; i < 10; i++ ) {
      u.setElement( i, i );
      u_host.setElement( i, i );
   }

   // assignment from host to device
   v.setValue( 0 );
   v = u_host;
   EXPECT_EQ( u, v );

   // assignment from device to host
   u_host.setValue( 0 );
   u_host = u;
   EXPECT_EQ( u_host, u );

   // assignment of a scalar
   u = 42;
   for( int i = 0; i < 10; i++ )
      EXPECT_EQ( u.getElement( i ), 42 );
}

// test works only for arithmetic types
template< typename ArrayType,
          typename = typename std::enable_if< std::is_arithmetic< typename ArrayType::ValueType >::value >::type >
void testArrayAssignmentWithDifferentType()
{
   using HostArrayType = typename ArrayType::template Self< typename ArrayType::ValueType, Devices::Sequential >;

   ArrayType u( 10 );
   Array< short, typename ArrayType::DeviceType, short > v( 10 );
   Array< short, Devices::Host, short > v_host( 10 );
   HostArrayType u_host( 10 );
   for( int i = 0; i < 10; i++ ) {
      u.setElement( i, i );
      u_host.setElement( i, i );
   }

   v.setValue( 0 );
   v = u;
   EXPECT_EQ( v, u );

   // assignment from host to device
   v.setValue( 0 );
   v = u_host;
   EXPECT_EQ( v, u_host );

   // assignment from device to host
   v_host.setValue( 0 );
   v_host = u;
   EXPECT_EQ( v_host, u );
}

template< typename ArrayType,
          typename = typename std::enable_if< ! std::is_arithmetic< typename ArrayType::ValueType >::value >::type,
          typename = void >
void testArrayAssignmentWithDifferentType()
{
}

TYPED_TEST( ArrayTest, assignmentOperatorWithDifferentType )
{
   using ArrayType = typename TestFixture::ArrayType;

   testArrayAssignmentWithDifferentType< ArrayType >();
}

TYPED_TEST( ArrayTest, SaveAndLoad )
{
   using ArrayType = typename TestFixture::ArrayType;

   ArrayType u, v;
   v.setSize( 100 );
   for( int i = 0; i < 100; i ++ )
      v.setElement( i, 42 );
   ASSERT_NO_THROW( File( TEST_FILE_NAME, std::ios_base::out ) << v );
   ASSERT_NO_THROW( File( TEST_FILE_NAME, std::ios_base::in ) >> u );
   EXPECT_EQ( u, v );

   EXPECT_EQ( std::remove( TEST_FILE_NAME ), 0 );
}

TYPED_TEST( ArrayTest, LoadViaView )
{
   using ArrayType = typename TestFixture::ArrayType;

   ArrayType v, w;
   v.setSize( 100 );
   for( int i = 0; i < 100; i ++ )
      v.setElement( i, 42 );
   ASSERT_NO_THROW( File( TEST_FILE_NAME, std::ios_base::out ) << v );

   w.setSize( 100 );
   auto u = w.getView();
   ASSERT_NO_THROW( File( TEST_FILE_NAME, std::ios_base::in ) >> u );
   EXPECT_EQ( u, v );
   EXPECT_EQ( u.getData(), w.getData() );

   ArrayType z( 50 );
   File file;
   ASSERT_NO_THROW( file.open( TEST_FILE_NAME, std::ios_base::in ) );
   EXPECT_THROW( file >> z.getView(), Exceptions::FileDeserializationError );

   EXPECT_EQ( std::remove( TEST_FILE_NAME ), 0 );
}

// TODO: test all __cuda_callable__ methods from a CUDA kernel

#endif // HAVE_GTEST


#include "../main.h"
