#pragma once

#ifdef HAVE_GTEST

#include <TNL/Arithmetics/Quad.h>
#include <TNL/Containers/Array.h>
#include <TNL/Algorithms/SegmentedScan.h>

#include "gtest/gtest.h"

using namespace TNL;
using namespace TNL::Containers;
using namespace TNL::Arithmetics;
using namespace TNL::Algorithms;

// should be small enough to have fast tests, but larger than minGPUReductionDataSize
// and large enough to require multiple CUDA blocks for reduction
constexpr int ARRAY_TEST_SIZE = 10000;

// test fixture for typed tests
template< typename Array >
class SegmentedScanTest : public ::testing::Test
{
protected:
   using ArrayType = Array;
   using ViewType = ArrayView< typename Array::ValueType, typename Array::DeviceType, typename Array::IndexType >;
};

// types for which SegmentedScanTest is instantiated
// TODO: Quad must be fixed
using ArrayTypes = ::testing::Types<
#ifndef __CUDACC__
   Array< int,            Devices::Sequential, short >,
   Array< long,           Devices::Sequential, short >,
   Array< float,          Devices::Sequential, short >,
   Array< double,         Devices::Sequential, short >,
   //Array< Quad< float >,  Devices::Sequential, short >,
   //Array< Quad< double >, Devices::Sequential, short >,
   Array< int,            Devices::Sequential, int >,
   Array< long,           Devices::Sequential, int >,
   Array< float,          Devices::Sequential, int >,
   Array< double,         Devices::Sequential, int >,
   //Array< Quad< float >,  Devices::Sequential, int >,
   //Array< Quad< double >, Devices::Sequential, int >,
   Array< int,            Devices::Sequential, long >,
   Array< long,           Devices::Sequential, long >,
   Array< float,          Devices::Sequential, long >,
   Array< double,         Devices::Sequential, long >,
   //Array< Quad< float >,  Devices::Sequential, long >,
   //Array< Quad< double >, Devices::Sequential, long >,

   Array< int,            Devices::Host, short >,
   Array< long,           Devices::Host, short >,
   Array< float,          Devices::Host, short >,
   Array< double,         Devices::Host, short >,
   //Array< Quad< float >,  Devices::Host, short >,
   //Array< Quad< double >, Devices::Host, short >,
   Array< int,            Devices::Host, int >,
   Array< long,           Devices::Host, int >,
   Array< float,          Devices::Host, int >,
   Array< double,         Devices::Host, int >,
   //Array< Quad< float >,  Devices::Host, int >,
   //Array< Quad< double >, Devices::Host, int >,
   Array< int,            Devices::Host, long >,
   Array< long,           Devices::Host, long >,
   Array< float,          Devices::Host, long >,
   Array< double,         Devices::Host, long >
   //Array< Quad< float >,  Devices::Host, long >,
   //Array< Quad< double >, Devices::Host, long >
#endif
// TODO: segmented scan for CUDA is not implemented yet
//#ifdef __CUDACC__
//   Array< int,            Devices::Cuda, short >,
//   Array< long,           Devices::Cuda, short >,
//   Array< float,          Devices::Cuda, short >,
//   Array< double,         Devices::Cuda, short >,
//   //Array< Quad< float >,  Devices::Cuda, short >,
//   //Array< Quad< double >, Devices::Cuda, short >,
//   Array< int,            Devices::Cuda, int >,
//   Array< long,           Devices::Cuda, int >,
//   Array< float,          Devices::Cuda, int >,
//   Array< double,         Devices::Cuda, int >,
//   //Array< Quad< float >,  Devices::Cuda, int >,
//   //Array< Quad< double >, Devices::Cuda, int >,
//   Array< int,            Devices::Cuda, long >,
//   Array< long,           Devices::Cuda, long >,
//   Array< float,          Devices::Cuda, long >,
//   Array< double,         Devices::Cuda, long >
//   //Array< Quad< float >,  Devices::Cuda, long >,
//   //Array< Quad< double >, Devices::Cuda, long >
//#endif
>;

TYPED_TEST_SUITE( SegmentedScanTest, ArrayTypes );

template< typename Array >
void setLinearSequence( Array& array )
{
   using Value = typename Array::ValueType;
   using Index = typename Array::IndexType;
   auto f1 = [] __cuda_callable__ ( Index i, Value& value ) { value = i; };
   array.forAllElements( f1 );
}

template< typename FlagsView >
void setupFlags( FlagsView& flags )
{
   using Value = typename FlagsView::ValueType;
   using Index = typename FlagsView::IndexType;
   auto f1 = [] __cuda_callable__ ( Index i, Value& value ) { value = ( i % 5 == 0 ); };
   flags.forAllElements( f1 );
}

TYPED_TEST( SegmentedScanTest, inclusive )
{
   using ArrayType = typename TestFixture::ArrayType;
   using ViewType = typename TestFixture::ViewType;
   using ValueType = typename ArrayType::ValueType;
   using DeviceType = typename ArrayType::DeviceType;
   using IndexType = typename ArrayType::IndexType;
   using FlagsArrayType = Array< bool, DeviceType, IndexType >;
   using FlagsViewType = ArrayView< bool, DeviceType, IndexType >;
   const int size = ARRAY_TEST_SIZE;

   ArrayType v( size );
   ViewType v_view( v );

   FlagsArrayType flags( size ), flags_copy( size );
   FlagsViewType flags_view( flags );
   setupFlags( flags_view );
   flags_copy = flags_view;

   v = 0;
   SegmentedScan< DeviceType >::perform( v, flags_view, 0, size, TNL::Plus{}, TNL::Plus::template getIdentity< ValueType >() );
   for( int i = 0; i < size; i++ )
      EXPECT_EQ( v.getElement( i ), 0 );
   flags_view = flags_copy;

   v = 1;
   SegmentedScan< DeviceType >::perform( v, flags_view, 0, size, TNL::Plus{}, TNL::Plus::template getIdentity< ValueType >() );
   for( int i = 0; i < size; i++ )
      EXPECT_EQ( v.getElement( i ), ( i % 5 ) + 1 );
   flags_view = flags_copy;

   setLinearSequence( v );
   SegmentedScan< DeviceType >::perform( v, flags_view, 0, size, TNL::Plus{}, TNL::Plus::template getIdentity< ValueType >() );
   for( int i = 1; i < size; i++ )
   {
      if( flags.getElement( i ) )
         EXPECT_EQ( v.getElement( i ), i );
      else
         EXPECT_EQ( v.getElement( i ) - v.getElement( i - 1 ), i );
   }
   flags_view = flags_copy;

   v_view = 0;
   SegmentedScan< DeviceType >::perform( v_view, flags_view, 0, size, TNL::Plus{}, TNL::Plus::template getIdentity< ValueType >() );
   for( int i = 0; i < size; i++ )
      EXPECT_EQ( v_view.getElement( i ), 0 );
   flags_view = flags_copy;

   v_view = 1;
   SegmentedScan< DeviceType >::perform( v_view, flags_view, 0, size, TNL::Plus{}, TNL::Plus::template getIdentity< ValueType >() );
   for( int i = 0; i < size; i++ )
      EXPECT_EQ( v_view.getElement( i ), ( i % 5 ) + 1 );
   flags_view = flags_copy;

   setLinearSequence( v );
   SegmentedScan< DeviceType >::perform( v_view, flags_view, 0, size, TNL::Plus{}, TNL::Plus::template getIdentity< ValueType >() );
   for( int i = 1; i < size; i++ )
   {
      if( flags.getElement( i ) )
         EXPECT_EQ( v_view.getElement( i ), i );
      else
         EXPECT_EQ( v_view.getElement( i ) - v_view.getElement( i - 1 ), i );
   }
}

// TODO: test exclusive segmented scan

#endif // HAVE_GTEST

#include "../main.h"
