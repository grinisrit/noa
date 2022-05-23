#pragma once

#ifdef HAVE_GTEST
#include <gtest/gtest.h>

#include <TNL/Arithmetics/Quad.h>
#include <TNL/Containers/Array.h>
#include <TNL/Algorithms/reduce.h>
#include "../CustomScalar.h"

using namespace TNL;
using namespace TNL::Containers;
using namespace TNL::Arithmetics;
using namespace TNL::Algorithms;
using namespace TNL::Algorithms::detail;

// test fixture for typed tests
template< typename Array >
class ReduceTest : public ::testing::Test
{
protected:
   using ArrayType = Array;
};

// types for which ReduceTest is instantiated
// TODO: Quad must be fixed
using ArrayTypes = ::testing::Types<
#ifndef HAVE_CUDA
   Array< CustomScalar< int >, Devices::Sequential, int >,
   Array< int,            Devices::Sequential, int >,
   Array< long,           Devices::Sequential, int >,
   Array< double,         Devices::Sequential, int >,
   //Array< Quad< float >,  Devices::Sequential, int >,
   //Array< Quad< double >, Devices::Sequential, int >,
   Array< CustomScalar< int >, Devices::Sequential, long >,
   Array< int,            Devices::Sequential, long >,
   Array< long,           Devices::Sequential, long >,
   Array< double,         Devices::Sequential, long >,
   //Array< Quad< float >,  Devices::Sequential, long >,
   //Array< Quad< double >, Devices::Sequential, long >,

   Array< CustomScalar< int >, Devices::Host, int >,
   Array< int,            Devices::Host, int >,
   Array< long,           Devices::Host, int >,
   Array< double,         Devices::Host, int >,
   //Array< Quad< float >,  Devices::Host, int >,
   //Array< Quad< double >, Devices::Host, int >,
   Array< CustomScalar< int >, Devices::Host, long >,
   Array< int,            Devices::Host, long >,
   Array< long,           Devices::Host, long >,
   Array< double,         Devices::Host, long >
   //Array< Quad< float >,  Devices::Host, long >,
   //Array< Quad< double >, Devices::Host, long >
#endif
#ifdef HAVE_CUDA
   Array< CustomScalar< int >, Devices::Cuda, int >,  // the reduction kernel for CustomScalar is not specialized with __shfl instructions
   Array< int,            Devices::Cuda, int >,
   Array< long,           Devices::Cuda, int >,
   Array< double,         Devices::Cuda, int >,
   //Array< Quad< float >,  Devices::Cuda, int >,
   //Array< Quad< double >, Devices::Cuda, int >,
   Array< CustomScalar< int >, Devices::Cuda, long >,  // the reduction kernel for CustomScalar is not specialized with __shfl instructions
   Array< int,            Devices::Cuda, long >,
   Array< long,           Devices::Cuda, long >,
   Array< double,         Devices::Cuda, long >
   //Array< Quad< float >,  Devices::Cuda, long >,
   //Array< Quad< double >, Devices::Cuda, long >
#endif
>;

TYPED_TEST_SUITE( ReduceTest, ArrayTypes );

template< typename Array >
void iota( Array& array, typename Array::ValueType start = 0 )
{
   array.forAllElements( [start] __cuda_callable__
                         ( typename Array::IndexType idx, typename Array::ValueType& value )
                         { value = idx + start; }
                       );
}

template< typename Array >
void mod( Array& array, typename Array::IndexType mod = 0 )
{
   array.forAllElements( [mod] __cuda_callable__
                         ( typename Array::IndexType idx, typename Array::ValueType& value )
                         { value = idx % mod; }
                       );
}

TYPED_TEST( ReduceTest, sum )
{
   using ArrayType = typename TestFixture::ArrayType;
   ArrayType a;
   for( int size = 1; size <= 100000; size *= 10 )
   {
      a.setSize( size );
      a.setValue( 1 );

      auto res = reduce< typename ArrayType::DeviceType >( 0, size, a.getConstView(), TNL::Plus{} );
      EXPECT_EQ( res, size );

      res = reduce( a, TNL::Plus{} );
      EXPECT_EQ( res, size );
   }

   const int size = 9377;
   a.setSize( size );
   iota( a );
   auto res = reduce( a, TNL::Plus{} );
   EXPECT_EQ( res, (size * (size - 1)) / 2 );
}

TYPED_TEST( ReduceTest, product )
{
   using ArrayType = typename TestFixture::ArrayType;
   ArrayType a;
   a.setSize( 10 );
   a.setValue( 2 );

   int result = 1;
   for( int size = 0; size < a.getSize(); size++ )
   {
      auto res = reduce< typename ArrayType::DeviceType >( 0, size, a.getConstView(), TNL::Multiplies{} );
      EXPECT_EQ( res, result );
      result *= 2;
   }
}

TYPED_TEST( ReduceTest, min )
{
   using ArrayType = typename TestFixture::ArrayType;
   ArrayType a;
   for( int size = 1; size <= 100000; size *= 10 )
   {
      a.setSize( size );
      iota( a, 1 );

      auto res = reduce< typename ArrayType::DeviceType >( 0, size, a.getConstView(), TNL::Min{} );
      EXPECT_EQ( res, 1 );
   }
}

TYPED_TEST( ReduceTest, max )
{
   using ArrayType = typename TestFixture::ArrayType;
   ArrayType a;
   for( int size = 1; size <= 100000; size *= 10 )
   {
      a.setSize( size );
      iota( a, 1 );

      auto res = reduce< typename ArrayType::DeviceType >( 0, size, a.getConstView(), TNL::Max{} );
      EXPECT_EQ( res, size );
   }
}

TYPED_TEST( ReduceTest, minWithArg )
{
   using ArrayType = typename TestFixture::ArrayType;
   ArrayType a;
   for( int size = 1; size <= 100000; size *= 10 )
   {
      a.setSize( size );
      iota( a, 1 );

      auto res = reduceWithArgument< typename ArrayType::DeviceType >( 0, size, a.getConstView(), TNL::MinWithArg{} );
      EXPECT_EQ( res.first, 1 );
      EXPECT_EQ( res.second, 0 );
   }
}

TYPED_TEST( ReduceTest, maxWithArg )
{
   using ArrayType = typename TestFixture::ArrayType;
   ArrayType a;
   for( int size = 1; size <= 100000; size *= 10 )
   {
      a.setSize( size );
      iota( a, 1 );

      auto res = reduceWithArgument< typename ArrayType::DeviceType >( 0, size, a.getConstView(), TNL::MaxWithArg{} );
      EXPECT_EQ( res.first, size );
      EXPECT_EQ( res.second, size - 1 );
   }
}

TYPED_TEST( ReduceTest, logicalAnd )
{
   using ArrayType = typename TestFixture::ArrayType;
   ArrayType a;
   for( int size = 1; size <= 100000; size *= 10 )
   {
      a.setSize( size );

      mod( a, 2 );
      auto res = reduce< typename ArrayType::DeviceType >( 0, size, a.getConstView(), TNL::LogicalAnd{} );
      EXPECT_EQ( res, false );

      a.setValue( 1 );
      res = reduce< typename ArrayType::DeviceType >( 0, size, a.getConstView(), TNL::LogicalAnd{} );
      EXPECT_EQ( res, true );
   }
}

TYPED_TEST( ReduceTest, logicalOr )
{
   using ArrayType = typename TestFixture::ArrayType;
   ArrayType a;
   for( int size = 1; size <= 100000; size *= 10 )
   {
      a.setSize( size );

      if( size == 1 )
         a.setValue( 1 );
      else
         mod( a, 2 );
      auto res = reduce< typename ArrayType::DeviceType >( 0, size, a.getConstView(), TNL::LogicalOr{} );
      EXPECT_EQ( res, true );

      a.setValue( 0 );
      res = reduce< typename ArrayType::DeviceType >( 0, size, a.getConstView(), TNL::LogicalOr{} );
      EXPECT_EQ( res, false );
   }
}

// bitwise AND (&) is not defined for floating-point types
template< typename ArrayType >
std::enable_if_t< std::is_integral< typename ArrayType::ValueType >::value >
test_bitAnd( ArrayType& a )
{
   for( int size = 1; size <= 100000; size *= 10 )
   {
      a.setSize( size );
      a.forAllElements( [] __cuda_callable__ ( typename ArrayType::IndexType idx, typename ArrayType::ValueType& value ) { value = 1 | ( 1 << ( idx % 8 ) ); } );

      auto res = reduce< typename ArrayType::DeviceType >( 0, size, a.getConstView(), TNL::BitAnd{} );
      EXPECT_EQ( res, 1 );
   }
}

template< typename ArrayType >
std::enable_if_t< ! std::is_integral< typename ArrayType::ValueType >::value >
test_bitAnd( ArrayType& a )
{
}

TYPED_TEST( ReduceTest, bitAnd )
{
   using ArrayType = typename TestFixture::ArrayType;
   ArrayType a;
   test_bitAnd( a );
}

// bitwise OR (|) is not defined for floating-point types
template< typename ArrayType >
std::enable_if_t< std::is_integral< typename ArrayType::ValueType >::value >
test_bitOr( ArrayType& a )
{
   for( int size = 10; size <= 100000; size *= 10 )
   {
      a.setSize( size );
      a.forAllElements( [] __cuda_callable__ ( typename ArrayType::IndexType idx, typename ArrayType::ValueType& value ) { value = 1 << ( idx % 8 );} );

      auto res = reduce< typename ArrayType::DeviceType >( 0, size, a.getConstView(), TNL::BitOr{} );
      EXPECT_EQ( res, 255 );
   }
}

template< typename ArrayType >
std::enable_if_t< ! std::is_integral< typename ArrayType::ValueType >::value >
test_bitOr( ArrayType& a )
{
}

TYPED_TEST( ReduceTest, bitOr )
{
   using ArrayType = typename TestFixture::ArrayType;
   ArrayType a;
   test_bitOr( a );
}

#endif

#include "../main.h"
