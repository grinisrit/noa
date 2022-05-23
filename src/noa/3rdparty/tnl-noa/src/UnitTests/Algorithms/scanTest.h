#pragma once

#ifdef HAVE_GTEST
#include <gtest/gtest.h>

#include <TNL/Arithmetics/Quad.h>
#include <TNL/Containers/Array.h>
#include <TNL/Containers/VectorView.h>
#include <TNL/Algorithms/scan.h>
#include "../CustomScalar.h"

using namespace TNL;
using namespace TNL::Containers;
using namespace TNL::Arithmetics;
using namespace TNL::Algorithms;
using namespace TNL::Algorithms::detail;

// test fixture for typed tests
template< typename Array >
class ScanTest : public ::testing::Test
{
protected:
   using ArrayType = Array;
   using ValueType = typename ArrayType::ValueType;
   using DeviceType = typename ArrayType::DeviceType;
   using IndexType = typename ArrayType::IndexType;
   using ArrayView = Containers::ArrayView< ValueType, DeviceType, IndexType >;
   using VectorView = Containers::VectorView< ValueType, DeviceType, IndexType >;
   using HostArrayType = typename ArrayType::template Self< ValueType, Devices::Sequential >;

   ArrayType a, b, c;
   ArrayView a_view, b_view, c_view;
   VectorView av_view, bv_view, cv_view;
   HostArrayType array_host, input_host, expected_host;

   // should be small enough to have fast tests, but larger than minGPUReductionDataSize
   // and large enough to require multiple CUDA blocks for reduction
   // also should be a prime number to cause non-uniform distribution of the work
   const int size = 9377;

   ScanTest()
   {
      resetWorkingArrays();
      input_host = expected_host = a;
   }

   void resetWorkingArrays()
   {
      a.setSize( size );
      a.setValue( -1 );
      c = b = a;
      a_view.bind( a );
      b_view.bind( b );
      c_view.bind( c );
      av_view.bind( a );
      bv_view.bind( b );
      cv_view.bind( c );

      // make sure that we perform tests with multiple CUDA grids
#ifdef HAVE_CUDA
      if( std::is_same< DeviceType, Devices::Cuda >::value )
      {
         CudaScanKernelLauncher< ScanType::Inclusive, ScanPhaseType::WriteInFirstPhase, ValueType >::resetMaxGridSize();
         CudaScanKernelLauncher< ScanType::Inclusive, ScanPhaseType::WriteInFirstPhase, ValueType >::maxGridSize() = 3;
         CudaScanKernelLauncher< ScanType::Exclusive, ScanPhaseType::WriteInFirstPhase, ValueType >::resetMaxGridSize();
         CudaScanKernelLauncher< ScanType::Exclusive, ScanPhaseType::WriteInFirstPhase, ValueType >::maxGridSize() = 3;
         CudaScanKernelLauncher< ScanType::Inclusive, ScanPhaseType::WriteInSecondPhase, ValueType >::resetMaxGridSize();
         CudaScanKernelLauncher< ScanType::Inclusive, ScanPhaseType::WriteInSecondPhase, ValueType >::maxGridSize() = 3;
         CudaScanKernelLauncher< ScanType::Exclusive, ScanPhaseType::WriteInSecondPhase, ValueType >::resetMaxGridSize();
         CudaScanKernelLauncher< ScanType::Exclusive, ScanPhaseType::WriteInSecondPhase, ValueType >::maxGridSize() = 3;
      }
#endif
   }

   template< Algorithms::detail::ScanType ScanType >
   void checkResult( const ArrayType& array )
   {
#ifdef HAVE_CUDA
      // skip the check for too small arrays
      if( array.getSize() > 256 ) {
         // we don't care which kernel launcher was actually used
         const auto gridsCount = TNL::max( CudaScanKernelLauncher< ScanType, ScanPhaseType::WriteInFirstPhase, ValueType >::gridsCount(),
                                           CudaScanKernelLauncher< ScanType, ScanPhaseType::WriteInSecondPhase, ValueType >::gridsCount() );
         EXPECT_GT( gridsCount, 1 );
      }
#endif

      array_host = array;

      for( int i = 0; i < array.getSize(); i++ )
         EXPECT_EQ( array_host[ i ], expected_host[ i ] ) << "arrays differ at index i = " << i;
   }
};

// types for which ScanTest is instantiated
// TODO: Quad must be fixed
using ArrayTypes = ::testing::Types<
#ifndef HAVE_CUDA
   Array< CustomScalar< int >, Devices::Sequential, short >,
   Array< int,            Devices::Sequential, short >,
   Array< long,           Devices::Sequential, short >,
   Array< double,         Devices::Sequential, short >,
   //Array< Quad< float >,  Devices::Sequential, short >,
   //Array< Quad< double >, Devices::Sequential, short >,
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

   Array< CustomScalar< int >, Devices::Host, short >,
   Array< int,            Devices::Host, short >,
   Array< long,           Devices::Host, short >,
   Array< double,         Devices::Host, short >,
   //Array< Quad< float >,  Devices::Host, short >,
   //Array< Quad< double >, Devices::Host, short >,
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
   Array< CustomScalar< int >, Devices::Cuda, short >,  // the scan kernel for CustomScalar is not specialized with __shfl instructions
   Array< int,            Devices::Cuda, short >,
   Array< long,           Devices::Cuda, short >,
   Array< double,         Devices::Cuda, short >,
   //Array< Quad< float >,  Devices::Cuda, short >,
   //Array< Quad< double >, Devices::Cuda, short >,
   Array< CustomScalar< int >, Devices::Cuda, int >,  // the scan kernel for CustomScalar is not specialized with __shfl instructions
   Array< int,            Devices::Cuda, int >,
   Array< long,           Devices::Cuda, int >,
   Array< double,         Devices::Cuda, int >,
   //Array< Quad< float >,  Devices::Cuda, int >,
   //Array< Quad< double >, Devices::Cuda, int >,
   Array< CustomScalar< int >, Devices::Cuda, long >,  // the scan kernel for CustomScalar is not specialized with __shfl instructions
   Array< int,            Devices::Cuda, long >,
   Array< long,           Devices::Cuda, long >,
   Array< double,         Devices::Cuda, long >
   //Array< Quad< float >,  Devices::Cuda, long >,
   //Array< Quad< double >, Devices::Cuda, long >
#endif
>;

TYPED_TEST_SUITE( ScanTest, ArrayTypes );

TYPED_TEST( ScanTest, inclusiveScan_zero_array )
{
   using ValueType = typename TestFixture::ValueType;

   this->input_host.setValue( 0 );
   this->expected_host.setValue( 0 );

   // general overload, array
   this->a = this->input_host;
   inclusiveScan( this->a, this->b, 0, this->size, 0, std::plus<>{}, (ValueType) 0 );
   this->template checkResult< ScanType::Inclusive >( this->b );
   EXPECT_EQ( this->a, this->input_host );

   this->resetWorkingArrays();

   // general overload, array view
   this->a = this->input_host;
   inclusiveScan( this->a_view, this->b_view, 0, this->size, 0, std::plus<>{}, (ValueType) 0 );
   this->template checkResult< ScanType::Inclusive >( this->b );
   EXPECT_EQ( this->a, this->input_host );

   this->resetWorkingArrays();

   // overload with TNL functional, array view
   this->a = this->input_host;
   inclusiveScan( this->a_view, this->b_view, 0, this->size, 0, TNL::Plus{} );
   this->template checkResult< ScanType::Inclusive >( this->b );
   EXPECT_EQ( this->a, this->input_host );

   this->resetWorkingArrays();

   // overload with default reduction operation, array
   this->a = this->input_host;
   inclusiveScan( this->a, this->b, 0, this->size, 0 );
   this->template checkResult< ScanType::Inclusive >( this->b );
   EXPECT_EQ( this->a, this->input_host );

   this->resetWorkingArrays();

   // overload with default reduction operation and default outputBegin, array view
   this->a = this->input_host;
   inclusiveScan( this->a_view, this->b_view, 0, this->size );
   this->template checkResult< ScanType::Inclusive >( this->b );
   EXPECT_EQ( this->a, this->input_host );

   // overload with default reduction operation and default end and outputBegin, array view
   this->a = this->input_host;
   inclusiveScan( this->a_view, this->b_view, 0 );
   this->template checkResult< ScanType::Inclusive >( this->b );
   EXPECT_EQ( this->a, this->input_host );

   this->resetWorkingArrays();

   // overload with default reduction operation and default begin, end and outputBegin, array
   this->a = this->input_host;
   inclusiveScan( this->a_view, this->b_view );
   this->template checkResult< ScanType::Inclusive >( this->b );
   EXPECT_EQ( this->a, this->input_host );
}

TYPED_TEST( ScanTest, inplaceInclusiveScan_zero_array )
{
   using ValueType = typename TestFixture::ValueType;

   this->input_host.setValue( 0 );
   this->expected_host.setValue( 0 );

   // general overload, array
   this->a = this->input_host;
   inplaceInclusiveScan( this->a, 0, this->size, std::plus<>{}, (ValueType) 0 );
   this->template checkResult< ScanType::Inclusive >( this->a );

   this->resetWorkingArrays();

   // general overload, array view
   this->a = this->input_host;
   inplaceInclusiveScan( this->a_view, 0, this->size, std::plus<>{}, (ValueType) 0 );
   this->template checkResult< ScanType::Inclusive >( this->a );

   this->resetWorkingArrays();

   // overload with TNL functional, array view
   this->a = this->input_host;
   inplaceInclusiveScan( this->a_view, 0, this->size, TNL::Plus{} );
   this->template checkResult< ScanType::Inclusive >( this->a );

   this->resetWorkingArrays();

   // overload with default reduction operation, array
   this->a = this->input_host;
   inplaceInclusiveScan( this->a, 0, this->size );
   this->template checkResult< ScanType::Inclusive >( this->a );

   this->resetWorkingArrays();

   // overload with default reduction operation and default end, array view
   this->a = this->input_host;
   inplaceInclusiveScan( this->a_view, 0 );
   this->template checkResult< ScanType::Inclusive >( this->a );

   this->resetWorkingArrays();

   // overload with default reduction operation and default begin and end, array
   this->a = this->input_host;
   inplaceInclusiveScan( this->a_view );
   this->template checkResult< ScanType::Inclusive >( this->a );
}

TYPED_TEST( ScanTest, inclusiveScan_constant_sequence )
{
   using ValueType = typename TestFixture::ValueType;

   this->input_host.setValue( 1 );
   for( int i = 0; i < this->size; i++ )
      this->expected_host[ i ] = i + 1;

   // general overload, array
   this->a = this->input_host;
   inclusiveScan( this->a, this->b, 0, this->size, 0, std::plus<>{}, (ValueType) 0 );
   this->template checkResult< ScanType::Inclusive >( this->b );
   EXPECT_EQ( this->a, this->input_host );

   this->resetWorkingArrays();

   // general overload, array view
   this->a = this->input_host;
   inclusiveScan( this->a_view, this->b_view, 0, this->size, 0, std::plus<>{}, (ValueType) 0 );
   this->template checkResult< ScanType::Inclusive >( this->b );
   EXPECT_EQ( this->a, this->input_host );

   this->resetWorkingArrays();

   // overload with TNL functional, array view
   this->a = this->input_host;
   inclusiveScan( this->a_view, this->b_view, 0, this->size, 0, TNL::Plus{} );
   this->template checkResult< ScanType::Inclusive >( this->b );
   EXPECT_EQ( this->a, this->input_host );

   this->resetWorkingArrays();

   // overload with default reduction operation, array
   this->a = this->input_host;
   inclusiveScan( this->a, this->b, 0, this->size, 0 );
   this->template checkResult< ScanType::Inclusive >( this->b );
   EXPECT_EQ( this->a, this->input_host );

   this->resetWorkingArrays();

   // overload with default reduction operation and default outputBegin, array view
   this->a = this->input_host;
   inclusiveScan( this->a_view, this->b_view, 0, this->size );
   this->template checkResult< ScanType::Inclusive >( this->b );
   EXPECT_EQ( this->a, this->input_host );

   // overload with default reduction operation and default end and outputBegin, array view
   this->a = this->input_host;
   inclusiveScan( this->a_view, this->b_view, 0 );
   this->template checkResult< ScanType::Inclusive >( this->b );
   EXPECT_EQ( this->a, this->input_host );

   this->resetWorkingArrays();

   // overload with default reduction operation and default begin, end and outputBegin, array
   this->a = this->input_host;
   inclusiveScan( this->a_view, this->b_view );
   this->template checkResult< ScanType::Inclusive >( this->b );
   EXPECT_EQ( this->a, this->input_host );
}

TYPED_TEST( ScanTest, inplaceInclusiveScan_constant_sequence )
{
   using ValueType = typename TestFixture::ValueType;

   this->input_host.setValue( 1 );
   for( int i = 0; i < this->size; i++ )
      this->expected_host[ i ] = i + 1;

   // general overload, array
   this->a = this->input_host;
   inplaceInclusiveScan( this->a, 0, this->size, std::plus<>{}, (ValueType) 0 );
   this->template checkResult< ScanType::Inclusive >( this->a );

   this->resetWorkingArrays();

   // general overload, array view
   this->a = this->input_host;
   inplaceInclusiveScan( this->a_view, 0, this->size, std::plus<>{}, (ValueType) 0 );
   this->template checkResult< ScanType::Inclusive >( this->a );

   this->resetWorkingArrays();

   // overload with TNL functional, array view
   this->a = this->input_host;
   inplaceInclusiveScan( this->a_view, 0, this->size, TNL::Plus{} );
   this->template checkResult< ScanType::Inclusive >( this->a );

   this->resetWorkingArrays();

   // overload with default reduction operation, array
   this->a = this->input_host;
   inplaceInclusiveScan( this->a, 0, this->size );
   this->template checkResult< ScanType::Inclusive >( this->a );

   this->resetWorkingArrays();

   // overload with default reduction operation and default end, array view
   this->a = this->input_host;
   inplaceInclusiveScan( this->a_view, 0 );
   this->template checkResult< ScanType::Inclusive >( this->a );

   this->resetWorkingArrays();

   // overload with default reduction operation and default begin and end, array
   this->a = this->input_host;
   inplaceInclusiveScan( this->a_view );
   this->template checkResult< ScanType::Inclusive >( this->a );
}

TYPED_TEST( ScanTest, inclusiveScan_linear_sequence )
{
   using ValueType = typename TestFixture::ValueType;

   for( int i = 0; i < this->size; i++ ) {
      this->input_host[ i ] = i;
      this->expected_host[ i ] = (i * (i + 1)) / 2;
   }

   this->a = this->input_host;
   inclusiveScan( this->a, this->b, 0, this->size, 0, std::plus<>{}, (ValueType) 0 );
   this->template checkResult< ScanType::Inclusive >( this->b );
   EXPECT_EQ( this->a, this->input_host );

   this->resetWorkingArrays();

   this->a = this->input_host;
   inclusiveScan( this->a_view, this->b_view, 0, this->size, 0, std::plus<>{}, (ValueType) 0 );
   this->template checkResult< ScanType::Inclusive >( this->b );
   EXPECT_EQ( this->a, this->input_host );
}

TYPED_TEST( ScanTest, inplaceInclusiveScan_linear_sequence )
{
   using ValueType = typename TestFixture::ValueType;

   for( int i = 0; i < this->size; i++ ) {
      this->input_host[ i ] = i;
      this->expected_host[ i ] = (i * (i + 1)) / 2;
   }

   this->a = this->input_host;
   inplaceInclusiveScan( this->a, 0, this->size, std::plus<>{}, (ValueType) 0 );
   this->template checkResult< ScanType::Inclusive >( this->a );

   this->resetWorkingArrays();

   this->a = this->input_host;
   inplaceInclusiveScan( this->a_view, 0, this->size, std::plus<>{}, (ValueType) 0 );
   this->template checkResult< ScanType::Inclusive >( this->a );
}

TYPED_TEST( ScanTest, exclusiveScan_zero_array )
{
   using ValueType = typename TestFixture::ValueType;

   this->input_host.setValue( 0 );
   this->expected_host.setValue( 0 );

   // general overload, array
   this->a = this->input_host;
   exclusiveScan( this->a, this->b, 0, this->size, 0, std::plus<>{}, (ValueType) 0 );
   this->template checkResult< ScanType::Exclusive >( this->b );
   EXPECT_EQ( this->a, this->input_host );

   this->resetWorkingArrays();

   // general overload, array view
   this->a = this->input_host;
   exclusiveScan( this->a_view, this->b_view, 0, this->size, 0, std::plus<>{}, (ValueType) 0 );
   this->template checkResult< ScanType::Exclusive >( this->b );
   EXPECT_EQ( this->a, this->input_host );

   this->resetWorkingArrays();

   // overload with TNL functional, array view
   this->a = this->input_host;
   exclusiveScan( this->a_view, this->b_view, 0, this->size, 0, TNL::Plus{} );
   this->template checkResult< ScanType::Exclusive >( this->b );
   EXPECT_EQ( this->a, this->input_host );

   this->resetWorkingArrays();

   // overload with default reduction operation, array
   this->a = this->input_host;
   exclusiveScan( this->a, this->b, 0, this->size, 0 );
   this->template checkResult< ScanType::Exclusive >( this->b );
   EXPECT_EQ( this->a, this->input_host );

   this->resetWorkingArrays();

   // overload with default reduction operation and default outputBegin, array view
   this->a = this->input_host;
   exclusiveScan( this->a_view, this->b_view, 0, this->size );
   this->template checkResult< ScanType::Exclusive >( this->b );
   EXPECT_EQ( this->a, this->input_host );

   // overload with default reduction operation and default end and outputBegin, array view
   this->a = this->input_host;
   exclusiveScan( this->a_view, this->b_view, 0 );
   this->template checkResult< ScanType::Exclusive >( this->b );
   EXPECT_EQ( this->a, this->input_host );

   this->resetWorkingArrays();

   // overload with default reduction operation and default begin, end and outputBegin, array
   this->a = this->input_host;
   exclusiveScan( this->a_view, this->b_view );
   this->template checkResult< ScanType::Exclusive >( this->b );
   EXPECT_EQ( this->a, this->input_host );
}

TYPED_TEST( ScanTest, inplaceExclusiveScan_zero_array )
{
   using ValueType = typename TestFixture::ValueType;

   this->input_host.setValue( 0 );
   this->expected_host.setValue( 0 );

   // general overload, array
   this->a = this->input_host;
   inplaceExclusiveScan( this->a, 0, this->size, std::plus<>{}, (ValueType) 0 );
   this->template checkResult< ScanType::Exclusive >( this->a );

   this->resetWorkingArrays();

   // general overload, array view
   this->a = this->input_host;
   inplaceExclusiveScan( this->a_view, 0, this->size, std::plus<>{}, (ValueType) 0 );
   this->template checkResult< ScanType::Exclusive >( this->a );

   this->resetWorkingArrays();

   // overload with TNL functional, array view
   this->a = this->input_host;
   inplaceExclusiveScan( this->a_view, 0, this->size, TNL::Plus{} );
   this->template checkResult< ScanType::Exclusive >( this->a );

   this->resetWorkingArrays();

   // overload with default reduction operation, array
   this->a = this->input_host;
   inplaceExclusiveScan( this->a, 0, this->size );
   this->template checkResult< ScanType::Exclusive >( this->a );

   this->resetWorkingArrays();

   // overload with default reduction operation and default end, array view
   this->a = this->input_host;
   inplaceExclusiveScan( this->a_view, 0 );
   this->template checkResult< ScanType::Exclusive >( this->a );

   this->resetWorkingArrays();

   // overload with default reduction operation and default begin and end, array
   this->a = this->input_host;
   inplaceExclusiveScan( this->a_view );
   this->template checkResult< ScanType::Exclusive >( this->a );
}

TYPED_TEST( ScanTest, exclusiveScan_constant_sequence )
{
   using ValueType = typename TestFixture::ValueType;

   this->input_host.setValue( 1 );
   for( int i = 0; i < this->size; i++ )
      this->expected_host[ i ] = i;

   // general overload, array
   this->a = this->input_host;
   exclusiveScan( this->a, this->b, 0, this->size, 0, std::plus<>{}, (ValueType) 0 );
   this->template checkResult< ScanType::Exclusive >( this->b );
   EXPECT_EQ( this->a, this->input_host );

   this->resetWorkingArrays();

   // general overload, array view
   this->a = this->input_host;
   exclusiveScan( this->a_view, this->b_view, 0, this->size, 0, std::plus<>{}, (ValueType) 0 );
   this->template checkResult< ScanType::Exclusive >( this->b );
   EXPECT_EQ( this->a, this->input_host );

   this->resetWorkingArrays();

   // overload with TNL functional, array view
   this->a = this->input_host;
   exclusiveScan( this->a_view, this->b_view, 0, this->size, 0, TNL::Plus{} );
   this->template checkResult< ScanType::Exclusive >( this->b );
   EXPECT_EQ( this->a, this->input_host );

   this->resetWorkingArrays();

   // overload with default reduction operation, array
   this->a = this->input_host;
   exclusiveScan( this->a, this->b, 0, this->size, 0 );
   this->template checkResult< ScanType::Exclusive >( this->b );
   EXPECT_EQ( this->a, this->input_host );

   this->resetWorkingArrays();

   // overload with default reduction operation and default outputBegin, array view
   this->a = this->input_host;
   exclusiveScan( this->a_view, this->b_view, 0, this->size );
   this->template checkResult< ScanType::Exclusive >( this->b );
   EXPECT_EQ( this->a, this->input_host );

   // overload with default reduction operation and default end and outputBegin, array view
   this->a = this->input_host;
   exclusiveScan( this->a_view, this->b_view, 0 );
   this->template checkResult< ScanType::Exclusive >( this->b );
   EXPECT_EQ( this->a, this->input_host );

   this->resetWorkingArrays();

   // overload with default reduction operation and default begin, end and outputBegin, array
   this->a = this->input_host;
   exclusiveScan( this->a_view, this->b_view );
   this->template checkResult< ScanType::Exclusive >( this->b );
   EXPECT_EQ( this->a, this->input_host );
}

TYPED_TEST( ScanTest, inplaceExclusiveScan_constant_sequence )
{
   using ValueType = typename TestFixture::ValueType;

   this->input_host.setValue( 1 );
   for( int i = 0; i < this->size; i++ )
      this->expected_host[ i ] = i;

   // general overload, array
   this->a = this->input_host;
   inplaceExclusiveScan( this->a, 0, this->size, std::plus<>{}, (ValueType) 0 );
   this->template checkResult< ScanType::Exclusive >( this->a );

   this->resetWorkingArrays();

   // general overload, array view
   this->a = this->input_host;
   inplaceExclusiveScan( this->a_view, 0, this->size, std::plus<>{}, (ValueType) 0 );
   this->template checkResult< ScanType::Exclusive >( this->a );

   this->resetWorkingArrays();

   // overload with TNL functional, array view
   this->a = this->input_host;
   inplaceExclusiveScan( this->a_view, 0, this->size, TNL::Plus{} );
   this->template checkResult< ScanType::Exclusive >( this->a );

   this->resetWorkingArrays();

   // overload with default reduction operation, array
   this->a = this->input_host;
   inplaceExclusiveScan( this->a, 0, this->size );
   this->template checkResult< ScanType::Exclusive >( this->a );

   this->resetWorkingArrays();

   // overload with default reduction operation and default end, array view
   this->a = this->input_host;
   inplaceExclusiveScan( this->a_view, 0 );
   this->template checkResult< ScanType::Exclusive >( this->a );

   this->resetWorkingArrays();

   // overload with default reduction operation and default begin and end, array
   this->a = this->input_host;
   inplaceExclusiveScan( this->a_view );
   this->template checkResult< ScanType::Exclusive >( this->a );
}

TYPED_TEST( ScanTest, exclusiveScan_linear_sequence )
{
   using ValueType = typename TestFixture::ValueType;

   for( int i = 0; i < this->size; i++ ) {
      this->input_host[ i ] = i;
      this->expected_host[ i ] = (i * (i - 1)) / 2;
   }

   this->a = this->input_host;
   exclusiveScan( this->a, this->b, 0, this->size, 0, std::plus<>{}, (ValueType) 0 );
   this->template checkResult< ScanType::Exclusive >( this->b );
   EXPECT_EQ( this->a, this->input_host );

   this->resetWorkingArrays();

   this->a = this->input_host;
   exclusiveScan( this->a_view, this->b_view, 0, this->size, 0, std::plus<>{}, (ValueType) 0 );
   this->template checkResult< ScanType::Exclusive >( this->b );
   EXPECT_EQ( this->a, this->input_host );
}

TYPED_TEST( ScanTest, inplaceExclusiveScan_linear_sequence )
{
   using ValueType = typename TestFixture::ValueType;

   for( int i = 0; i < this->size; i++ ) {
      this->input_host[ i ] = i;
      this->expected_host[ i ] = (i * (i - 1)) / 2;
   }

   this->a = this->input_host;
   inplaceExclusiveScan( this->a, 0, this->size, std::plus<>{}, (ValueType) 0 );
   this->template checkResult< ScanType::Exclusive >( this->a );

   this->resetWorkingArrays();

   this->a = this->input_host;
   inplaceExclusiveScan( this->a_view, 0, this->size, std::plus<>{}, (ValueType) 0 );
   this->template checkResult< ScanType::Exclusive >( this->a );
}


TYPED_TEST( ScanTest, multiplication )
{
   this->input_host.setSize( 10 );
   this->input_host.setValue( 2 );
   this->expected_host = this->input_host;

   // exclusive scan test
   int value = 1;
   for( int i = 0; i < this->expected_host.getSize(); i++ ) {
      this->expected_host[ i ] = value;
      value *= 2;
   }

   this->a = this->input_host;
   this->b = this->input_host;
   exclusiveScan( this->a, this->b, 0, this->a.getSize(), 0, TNL::Multiplies{} );
   this->template checkResult< ScanType::Exclusive >( this->b );
   EXPECT_EQ( this->a, this->input_host );
   inplaceExclusiveScan( this->a, 0, this->a.getSize(), TNL::Multiplies{} );
   this->template checkResult< ScanType::Exclusive >( this->a );

   // inclusive scan test
   for( int i = 0; i < this->expected_host.getSize(); i++ )
      this->expected_host[ i ] *= 2;

   this->a = this->input_host;
   this->b = this->input_host;
   inclusiveScan( this->a, this->b, 0, this->a.getSize(), 0, TNL::Multiplies{} );
   this->template checkResult< ScanType::Exclusive >( this->b );
   EXPECT_EQ( this->a, this->input_host );
   inplaceInclusiveScan( this->a, 0, this->a.getSize(), TNL::Multiplies{} );
   this->template checkResult< ScanType::Inclusive >( this->a );
}

TYPED_TEST( ScanTest, custom_begin_end )
{
   using IndexType = typename TestFixture::IndexType;

   const IndexType begin = 42;
   const IndexType end = this->size - begin;

   // exclusive scan test
   this->input_host.setValue( 1 );
   this->expected_host.setValue( 1 );
   for( int i = begin; i < end; i++ )
      this->expected_host[ i ] = i - begin;

   this->a = this->input_host;
   this->b = this->input_host;
   exclusiveScan( this->a, this->b, begin, end, begin );
   this->template checkResult< ScanType::Exclusive >( this->b );
   EXPECT_EQ( this->a, this->input_host );
   inplaceExclusiveScan( this->a, begin, end );
   this->template checkResult< ScanType::Exclusive >( this->a );

   // inclusive scan test
   for( int i = begin; i < end; i++ )
      this->expected_host[ i ]++;

   this->a = this->input_host;
   this->b = this->input_host;
   inclusiveScan( this->a, this->b, begin, end, begin );
   this->template checkResult< ScanType::Inclusive >( this->b );
   EXPECT_EQ( this->a, this->input_host );
   inplaceInclusiveScan( this->a, begin, end );
   this->template checkResult< ScanType::Inclusive >( this->a );
}

TYPED_TEST( ScanTest, empty_range )
{
   using IndexType = typename TestFixture::IndexType;

   this->input_host.setSize( 42 );
   this->input_host.setValue( 1 );
   this->expected_host = this->input_host;

   const IndexType begin = 2;
   const IndexType end = 1;

   // exclusive scan test
   this->a = this->input_host;
   this->b = this->input_host;
   exclusiveScan( this->a, this->b, begin, end, 0 );
   this->template checkResult< ScanType::Exclusive >( this->b );
   EXPECT_EQ( this->a, this->input_host );
   inplaceExclusiveScan( this->a, begin, end );
   this->template checkResult< ScanType::Exclusive >( this->a );

   // inclusive scan test
   this->a = this->input_host;
   this->b = this->input_host;
   inclusiveScan( this->a, this->b, begin, end, 0 );
   this->template checkResult< ScanType::Inclusive >( this->b );
   EXPECT_EQ( this->a, this->input_host );
   inplaceInclusiveScan( this->a, begin, end );
   this->template checkResult< ScanType::Inclusive >( this->a );
}

TYPED_TEST( ScanTest, vector_expression )
{
   this->a.setValue( 2 );
   this->b.setValue( 1 );

   // exclusive scan test
   for( int i = 0; i < this->size; i++ )
      this->expected_host[ i ] = i;

   this->c.setValue( 0 );
   exclusiveScan( this->av_view - this->bv_view, this->c, 0, this->a.getSize(), 0, TNL::Plus{} );
   this->template checkResult< ScanType::Exclusive >( this->c );

   // inclusive scan test
   for( int i = 0; i < this->expected_host.getSize(); i++ )
      this->expected_host[ i ]++;

   this->c.setValue( 0 );
   inclusiveScan( this->av_view - this->bv_view, this->c, 0, this->a.getSize(), 0, TNL::Plus{} );
   this->template checkResult< ScanType::Inclusive >( this->c );
}

#endif // HAVE_GTEST

#include "../main.h"
