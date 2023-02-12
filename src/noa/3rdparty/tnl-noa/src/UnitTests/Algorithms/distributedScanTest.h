#pragma once

#ifdef HAVE_GTEST
#include <gtest/gtest.h>

#include <TNL/Containers/DistributedArray.h>
#include <TNL/Containers/DistributedVectorView.h>
#include <TNL/Containers/Partitioner.h>
#include <TNL/Algorithms/distributedScan.h>

#define DISTRIBUTED_VECTOR
#include "../Containers/VectorHelperFunctions.h"

using namespace TNL;
using namespace TNL::Containers;
using namespace TNL::Algorithms;
using namespace TNL::Algorithms::detail;
using namespace TNL::MPI;

// this is a workaround for an nvcc 11.7 bug: it drops the scope of enum class members in template function calls
#ifdef __CUDACC__
static constexpr auto Inclusive = TNL::Algorithms::detail::ScanType::Inclusive;
static constexpr auto Exclusive = TNL::Algorithms::detail::ScanType::Exclusive;
#endif

/*
 * Light check of DistributedArray.
 *
 * - Number of processes is not limited.
 * - Global size is hardcoded as 97 to force non-uniform distribution.
 * - Communicator is hardcoded as MPI_COMM_WORLD -- it may be changed as needed.
 */
template< typename DistributedArray >
class DistributedScanTest
: public ::testing::Test
{
protected:
   using ValueType = typename DistributedArray::ValueType;
   using DeviceType = typename DistributedArray::DeviceType;
   using IndexType = typename DistributedArray::IndexType;
   using DistributedArrayType = DistributedArray;
   using DistributedArrayView = Containers::DistributedArrayView< ValueType, DeviceType, IndexType >;
   using DistributedVectorView = Containers::DistributedVectorView< ValueType, DeviceType, IndexType >;
   using HostDistributedArrayType = typename DistributedArrayType::template Self< ValueType, Devices::Sequential >;
   using LocalRangeType = typename DistributedArray::LocalRangeType;
   using Synchronizer = typename Partitioner< IndexType >::template ArraySynchronizer< DeviceType >;
   using HostSynchronizer = typename Partitioner< IndexType >::template ArraySynchronizer< Devices::Sequential >;

   const MPI_Comm communicator = MPI_COMM_WORLD;

   DistributedArrayType a, b, c;
   DistributedArrayView a_view, b_view, c_view;
   DistributedVectorView av_view, bv_view, cv_view;
   HostDistributedArrayType array_host, input_host, expected_host;

   const int rank = GetRank(communicator);
   const int nproc = GetSize(communicator);

   // should be small enough to have fast tests, but large enough to test
   // scan with multiple CUDA grids
   // also should be a prime number to cause non-uniform distribution of the work
   const int globalSize = 9377 * nproc;

   LocalRangeType localRange;

   // some arbitrary value (but must be 0 if not distributed)
   const int ghosts = (nproc > 1) ? 4 : 0;

   DistributedScanTest()
   {
      resetWorkingArrays();
      input_host = a;
      input_host.setSynchronizer( std::make_shared<HostSynchronizer>( a.getLocalRange(), ghosts / 2, communicator ) );
      expected_host = input_host;
   }

   void resetWorkingArrays()
   {
      localRange = Partitioner< IndexType >::splitRange( globalSize, communicator );
      a.setDistribution( localRange, ghosts, globalSize, communicator );
      a.setSynchronizer( std::make_shared<Synchronizer>( localRange, ghosts / 2, communicator ) );

      a.setValue( -1 );
      c = b = a;
      a_view.bind( a );
      b_view.bind( b );
      c_view.bind( c );
      av_view.bind( a );
      bv_view.bind( b );
      cv_view.bind( c );

      // make sure that we perform tests with multiple CUDA grids
#ifdef __CUDACC__
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
   void checkResult( const DistributedArrayType& array, bool check_cuda_grids = true )
   {
#ifdef __CUDACC__
      // skip the check for too small arrays
      if( check_cuda_grids && array.getLocalRange().getSize() > 256 ) {
         // we don't care which kernel launcher was actually used
         const auto gridsCount = TNL::max( CudaScanKernelLauncher< ScanType, ScanPhaseType::WriteInFirstPhase, ValueType >::gridsCount(),
                                           CudaScanKernelLauncher< ScanType, ScanPhaseType::WriteInSecondPhase, ValueType >::gridsCount() );
         EXPECT_GT( gridsCount, 1 );
      }
#endif

      array_host = array;

      for( int i = a.getLocalRange().getBegin(); i < a.getLocalRange().getEnd(); i++ )
         EXPECT_EQ( array_host[ i ], expected_host[ i ] ) << "arrays differ at index i = " << i;
   }
};

// types for which DistributedScanTest is instantiated
using DistributedArrayTypes = ::testing::Types<
#ifndef __CUDACC__
   DistributedArray< double, Devices::Sequential, int >,
   DistributedArray< double, Devices::Host, int >
#endif
#ifdef __CUDACC__
   DistributedArray< double, Devices::Cuda, int >
#endif
>;

TYPED_TEST_SUITE( DistributedScanTest, DistributedArrayTypes );

// TODO: test that horizontal operations are computed for ghost values without synchronization

TYPED_TEST( DistributedScanTest, distributedInclusiveScan_zero_array )
{
   using ValueType = typename TestFixture::ValueType;

   this->input_host.setValue( 0 );
   this->expected_host.setValue( 0 );

   // general overload, array
   this->a = this->input_host;
   distributedInclusiveScan( this->a, this->b, 0, this->globalSize, std::plus<>{}, (ValueType) 0 );
   this->template checkResult< ScanType::Inclusive >( this->b );
   EXPECT_EQ( this->a, this->input_host );

   this->resetWorkingArrays();

   // general overload, array view
   this->a = this->input_host;
   distributedInclusiveScan( this->a_view, this->b_view, 0, this->globalSize, std::plus<>{}, (ValueType) 0 );
   this->template checkResult< ScanType::Inclusive >( this->b );
   EXPECT_EQ( this->a, this->input_host );

   this->resetWorkingArrays();

   // overload with TNL functional, array view
   this->a = this->input_host;
   distributedInclusiveScan( this->a_view, this->b_view, 0, this->globalSize, TNL::Plus{} );
   this->template checkResult< ScanType::Inclusive >( this->b );
   EXPECT_EQ( this->a, this->input_host );

   this->resetWorkingArrays();

   // overload with default reduction operation, array
   this->a = this->input_host;
   distributedInclusiveScan( this->a, this->b, 0, this->globalSize );
   this->template checkResult< ScanType::Inclusive >( this->b );
   EXPECT_EQ( this->a, this->input_host );

   this->resetWorkingArrays();

   // overload with default reduction operation and default end, array view
   this->a = this->input_host;
   distributedInclusiveScan( this->a_view, this->b_view, 0 );
   this->template checkResult< ScanType::Inclusive >( this->b );
   EXPECT_EQ( this->a, this->input_host );

   this->resetWorkingArrays();

   // overload with default reduction operation and default begin and end, array
   this->a = this->input_host;
   distributedInclusiveScan( this->a_view, this->b_view );
   this->template checkResult< ScanType::Inclusive >( this->b );
   EXPECT_EQ( this->a, this->input_host );
}

TYPED_TEST( DistributedScanTest, distributedInplaceInclusiveScan_zero_array )
{
   using ValueType = typename TestFixture::ValueType;

   this->input_host.setValue( 0 );
   this->expected_host.setValue( 0 );

   // general overload, array
   this->a = this->input_host;
   distributedInplaceInclusiveScan( this->a, 0, this->globalSize, std::plus<>{}, (ValueType) 0 );
   this->template checkResult< ScanType::Inclusive >( this->a );

   this->resetWorkingArrays();

   // general overload, array view
   this->a = this->input_host;
   distributedInplaceInclusiveScan( this->a_view, 0, this->globalSize, std::plus<>{}, (ValueType) 0 );
   this->template checkResult< ScanType::Inclusive >( this->a );

   this->resetWorkingArrays();

   // overload with TNL functional, array view
   this->a = this->input_host;
   distributedInplaceInclusiveScan( this->a_view, 0, this->globalSize, TNL::Plus{} );
   this->template checkResult< ScanType::Inclusive >( this->a );

   this->resetWorkingArrays();

   // overload with default reduction operation, array
   this->a = this->input_host;
   distributedInplaceInclusiveScan( this->a, 0, this->globalSize );
   this->template checkResult< ScanType::Inclusive >( this->a );

   this->resetWorkingArrays();

   // overload with default reduction operation and default end, array view
   this->a = this->input_host;
   distributedInplaceInclusiveScan( this->a_view, 0 );
   this->template checkResult< ScanType::Inclusive >( this->a );

   this->resetWorkingArrays();

   // overload with default reduction operation and default begin and end, array
   this->a = this->input_host;
   distributedInplaceInclusiveScan( this->a_view );
   this->template checkResult< ScanType::Inclusive >( this->a );
}

TYPED_TEST( DistributedScanTest, distributedInclusiveScan_constant_sequence )
{
   using ValueType = typename TestFixture::ValueType;

   this->input_host.setValue( 1 );
   for( int i = this->localRange.getBegin(); i < this->localRange.getEnd(); i++ )
      this->expected_host[ i ] = i + 1;

   // general overload, array
   this->a = this->input_host;
   distributedInclusiveScan( this->a, this->b, 0, this->globalSize, std::plus<>{}, (ValueType) 0 );
   this->template checkResult< ScanType::Inclusive >( this->b );
   EXPECT_EQ( this->a, this->input_host );

   this->resetWorkingArrays();

   // general overload, array view
   this->a = this->input_host;
   distributedInclusiveScan( this->a_view, this->b_view, 0, this->globalSize, std::plus<>{}, (ValueType) 0 );
   this->template checkResult< ScanType::Inclusive >( this->b );
   EXPECT_EQ( this->a, this->input_host );

   this->resetWorkingArrays();

   // overload with TNL functional, array view
   this->a = this->input_host;
   distributedInclusiveScan( this->a_view, this->b_view, 0, this->globalSize, TNL::Plus{} );
   this->template checkResult< ScanType::Inclusive >( this->b );
   EXPECT_EQ( this->a, this->input_host );

   this->resetWorkingArrays();

   // overload with default reduction operation, array
   this->a = this->input_host;
   distributedInclusiveScan( this->a, this->b, 0, this->globalSize );
   this->template checkResult< ScanType::Inclusive >( this->b );
   EXPECT_EQ( this->a, this->input_host );

   this->resetWorkingArrays();

   // overload with default reduction operation and default end, array view
   this->a = this->input_host;
   distributedInclusiveScan( this->a_view, this->b_view, 0 );
   this->template checkResult< ScanType::Inclusive >( this->b );
   EXPECT_EQ( this->a, this->input_host );

   this->resetWorkingArrays();

   // overload with default reduction operation and default begin and end, array
   this->a = this->input_host;
   distributedInclusiveScan( this->a_view, this->b_view );
   this->template checkResult< ScanType::Inclusive >( this->b );
   EXPECT_EQ( this->a, this->input_host );
}

TYPED_TEST( DistributedScanTest, distributedInplaceInclusiveScan_constant_sequence )
{
   using ValueType = typename TestFixture::ValueType;

   this->input_host.setValue( 1 );
   for( int i = this->localRange.getBegin(); i < this->localRange.getEnd(); i++ )
      this->expected_host[ i ] = i + 1;

   // general overload, array
   this->a = this->input_host;
   distributedInplaceInclusiveScan( this->a, 0, this->globalSize, std::plus<>{}, (ValueType) 0 );
   this->template checkResult< ScanType::Inclusive >( this->a );

   this->resetWorkingArrays();

   // general overload, array view
   this->a = this->input_host;
   distributedInplaceInclusiveScan( this->a_view, 0, this->globalSize, std::plus<>{}, (ValueType) 0 );
   this->template checkResult< ScanType::Inclusive >( this->a );

   this->resetWorkingArrays();

   // overload with TNL functional, array view
   this->a = this->input_host;
   distributedInplaceInclusiveScan( this->a_view, 0, this->globalSize, TNL::Plus{} );
   this->template checkResult< ScanType::Inclusive >( this->a );

   this->resetWorkingArrays();

   // overload with default reduction operation, array
   this->a = this->input_host;
   distributedInplaceInclusiveScan( this->a, 0, this->globalSize );
   this->template checkResult< ScanType::Inclusive >( this->a );

   this->resetWorkingArrays();

   // overload with default reduction operation and default end, array view
   this->a = this->input_host;
   distributedInplaceInclusiveScan( this->a_view, 0 );
   this->template checkResult< ScanType::Inclusive >( this->a );

   this->resetWorkingArrays();

   // overload with default reduction operation and default begin and end, array
   this->a = this->input_host;
   distributedInplaceInclusiveScan( this->a_view );
   this->template checkResult< ScanType::Inclusive >( this->a );
}

TYPED_TEST( DistributedScanTest, distributedInclusiveScan_linear_sequence )
{
   using ValueType = typename TestFixture::ValueType;

   for( int i = this->localRange.getBegin(); i < this->localRange.getEnd(); i++ ) {
      this->input_host[ i ] = i;
      this->expected_host[ i ] = (i * (i + 1)) / 2;
   }

   this->a = this->input_host;
   distributedInclusiveScan( this->a, this->b, 0, this->globalSize, std::plus<>{}, (ValueType) 0 );
   this->template checkResult< ScanType::Inclusive >( this->b );
   EXPECT_EQ( this->a, this->input_host );

   this->resetWorkingArrays();

   this->a = this->input_host;
   distributedInclusiveScan( this->a_view, this->b_view, 0, this->globalSize, std::plus<>{}, (ValueType) 0 );
   this->template checkResult< ScanType::Inclusive >( this->b );
   EXPECT_EQ( this->a, this->input_host );
}

TYPED_TEST( DistributedScanTest, distributedInplaceInclusiveScan_linear_sequence )
{
   using ValueType = typename TestFixture::ValueType;

   for( int i = this->localRange.getBegin(); i < this->localRange.getEnd(); i++ ) {
      this->input_host[ i ] = i;
      this->expected_host[ i ] = (i * (i + 1)) / 2;
   }

   this->a = this->input_host;
   distributedInplaceInclusiveScan( this->a, 0, this->globalSize, std::plus<>{}, (ValueType) 0 );
   this->template checkResult< ScanType::Inclusive >( this->a );

   this->resetWorkingArrays();

   this->a = this->input_host;
   distributedInplaceInclusiveScan( this->a_view, 0, this->globalSize, std::plus<>{}, (ValueType) 0 );
   this->template checkResult< ScanType::Inclusive >( this->a );
}

TYPED_TEST( DistributedScanTest, distributedExclusiveScan_zero_array )
{
   using ValueType = typename TestFixture::ValueType;

   this->input_host.setValue( 0 );
   this->expected_host.setValue( 0 );

   // general overload, array
   this->a = this->input_host;
   distributedExclusiveScan( this->a, this->b, 0, this->globalSize, std::plus<>{}, (ValueType) 0 );
   this->template checkResult< ScanType::Exclusive >( this->b );
   EXPECT_EQ( this->a, this->input_host );

   this->resetWorkingArrays();

   // general overload, array view
   this->a = this->input_host;
   distributedExclusiveScan( this->a_view, this->b_view, 0, this->globalSize, std::plus<>{}, (ValueType) 0 );
   this->template checkResult< ScanType::Exclusive >( this->b );
   EXPECT_EQ( this->a, this->input_host );

   this->resetWorkingArrays();

   // overload with TNL functional, array view
   this->a = this->input_host;
   distributedExclusiveScan( this->a_view, this->b_view, 0, this->globalSize, TNL::Plus{} );
   this->template checkResult< ScanType::Exclusive >( this->b );
   EXPECT_EQ( this->a, this->input_host );

   this->resetWorkingArrays();

   // overload with default reduction operation, array
   this->a = this->input_host;
   distributedExclusiveScan( this->a, this->b, 0, this->globalSize );
   this->template checkResult< ScanType::Exclusive >( this->b );
   EXPECT_EQ( this->a, this->input_host );

   this->resetWorkingArrays();

   // overload with default reduction operation and default end, array view
   this->a = this->input_host;
   distributedExclusiveScan( this->a_view, this->b_view, 0 );
   this->template checkResult< ScanType::Exclusive >( this->b );
   EXPECT_EQ( this->a, this->input_host );

   this->resetWorkingArrays();

   // overload with default reduction operation and default begin and end, array
   this->a = this->input_host;
   distributedExclusiveScan( this->a_view, this->b_view );
   this->template checkResult< ScanType::Exclusive >( this->b );
   EXPECT_EQ( this->a, this->input_host );
}

TYPED_TEST( DistributedScanTest, distributedInplaceExclusiveScan_zero_array )
{
   using ValueType = typename TestFixture::ValueType;

   this->input_host.setValue( 0 );
   this->expected_host.setValue( 0 );

   // general overload, array
   this->a = this->input_host;
   distributedInplaceExclusiveScan( this->a, 0, this->globalSize, std::plus<>{}, (ValueType) 0 );
   this->template checkResult< ScanType::Exclusive >( this->a );

   this->resetWorkingArrays();

   // general overload, array view
   this->a = this->input_host;
   distributedInplaceExclusiveScan( this->a_view, 0, this->globalSize, std::plus<>{}, (ValueType) 0 );
   this->template checkResult< ScanType::Exclusive >( this->a );

   this->resetWorkingArrays();

   // overload with TNL functional, array view
   this->a = this->input_host;
   distributedInplaceExclusiveScan( this->a_view, 0, this->globalSize, TNL::Plus{} );
   this->template checkResult< ScanType::Exclusive >( this->a );

   this->resetWorkingArrays();

   // overload with default reduction operation, array
   this->a = this->input_host;
   distributedInplaceExclusiveScan( this->a, 0, this->globalSize );
   this->template checkResult< ScanType::Exclusive >( this->a );

   this->resetWorkingArrays();

   // overload with default reduction operation and default end, array view
   this->a = this->input_host;
   distributedInplaceExclusiveScan( this->a_view, 0 );
   this->template checkResult< ScanType::Exclusive >( this->a );

   this->resetWorkingArrays();

   // overload with default reduction operation and default begin and end, array
   this->a = this->input_host;
   distributedInplaceExclusiveScan( this->a_view );
   this->template checkResult< ScanType::Exclusive >( this->a );
}

TYPED_TEST( DistributedScanTest, distributedExclusiveScan_constant_sequence )
{
   using ValueType = typename TestFixture::ValueType;

   this->input_host.setValue( 1 );
   for( int i = this->localRange.getBegin(); i < this->localRange.getEnd(); i++ )
      this->expected_host[ i ] = i;

   // general overload, array
   this->a = this->input_host;
   distributedExclusiveScan( this->a, this->b, 0, this->globalSize, std::plus<>{}, (ValueType) 0 );
   this->template checkResult< ScanType::Exclusive >( this->b );
   EXPECT_EQ( this->a, this->input_host );

   this->resetWorkingArrays();

   // general overload, array view
   this->a = this->input_host;
   distributedExclusiveScan( this->a_view, this->b_view, 0, this->globalSize, std::plus<>{}, (ValueType) 0 );
   this->template checkResult< ScanType::Exclusive >( this->b );
   EXPECT_EQ( this->a, this->input_host );

   this->resetWorkingArrays();

   // overload with TNL functional, array view
   this->a = this->input_host;
   distributedExclusiveScan( this->a_view, this->b_view, 0, this->globalSize, TNL::Plus{} );
   this->template checkResult< ScanType::Exclusive >( this->b );
   EXPECT_EQ( this->a, this->input_host );

   this->resetWorkingArrays();

   // overload with default reduction operation, array
   this->a = this->input_host;
   distributedExclusiveScan( this->a, this->b, 0, this->globalSize );
   this->template checkResult< ScanType::Exclusive >( this->b );
   EXPECT_EQ( this->a, this->input_host );

   this->resetWorkingArrays();

   // overload with default reduction operation and default end, array view
   this->a = this->input_host;
   distributedExclusiveScan( this->a_view, this->b_view, 0 );
   this->template checkResult< ScanType::Exclusive >( this->b );
   EXPECT_EQ( this->a, this->input_host );

   this->resetWorkingArrays();

   // overload with default reduction operation and default begin and end, array
   this->a = this->input_host;
   distributedExclusiveScan( this->a_view, this->b_view );
   this->template checkResult< ScanType::Exclusive >( this->b );
   EXPECT_EQ( this->a, this->input_host );
}

TYPED_TEST( DistributedScanTest, distributedInplaceExclusiveScan_constant_sequence )
{
   using ValueType = typename TestFixture::ValueType;

   this->input_host.setValue( 1 );
   for( int i = this->localRange.getBegin(); i < this->localRange.getEnd(); i++ )
      this->expected_host[ i ] = i;

   // general overload, array
   this->a = this->input_host;
   distributedInplaceExclusiveScan( this->a, 0, this->globalSize, std::plus<>{}, (ValueType) 0 );
   this->template checkResult< ScanType::Exclusive >( this->a );

   this->resetWorkingArrays();

   // general overload, array view
   this->a = this->input_host;
   distributedInplaceExclusiveScan( this->a_view, 0, this->globalSize, std::plus<>{}, (ValueType) 0 );
   this->template checkResult< ScanType::Exclusive >( this->a );

   this->resetWorkingArrays();

   // overload with TNL functional, array view
   this->a = this->input_host;
   distributedInplaceExclusiveScan( this->a_view, 0, this->globalSize, TNL::Plus{} );
   this->template checkResult< ScanType::Exclusive >( this->a );

   this->resetWorkingArrays();

   // overload with default reduction operation, array
   this->a = this->input_host;
   distributedInplaceExclusiveScan( this->a, 0, this->globalSize );
   this->template checkResult< ScanType::Exclusive >( this->a );

   this->resetWorkingArrays();

   // overload with default reduction operation and default end, array view
   this->a = this->input_host;
   distributedInplaceExclusiveScan( this->a_view, 0 );
   this->template checkResult< ScanType::Exclusive >( this->a );

   this->resetWorkingArrays();

   // overload with default reduction operation and default begin and end, array
   this->a = this->input_host;
   distributedInplaceExclusiveScan( this->a_view );
   this->template checkResult< ScanType::Exclusive >( this->a );
}

TYPED_TEST( DistributedScanTest, distributedExclusiveScan_linear_sequence )
{
   using ValueType = typename TestFixture::ValueType;

   for( int i = this->localRange.getBegin(); i < this->localRange.getEnd(); i++ ) {
      this->input_host[ i ] = i;
      this->expected_host[ i ] = (i * (i - 1)) / 2;
   }

   this->a = this->input_host;
   distributedExclusiveScan( this->a, this->b, 0, this->globalSize, std::plus<>{}, (ValueType) 0 );
   this->template checkResult< ScanType::Exclusive >( this->b );
   EXPECT_EQ( this->a, this->input_host );

   this->resetWorkingArrays();

   this->a = this->input_host;
   distributedExclusiveScan( this->a_view, this->b_view, 0, this->globalSize, std::plus<>{}, (ValueType) 0 );
   this->template checkResult< ScanType::Exclusive >( this->b );
   EXPECT_EQ( this->a, this->input_host );
}

TYPED_TEST( DistributedScanTest, distributedInplaceExclusiveScan_linear_sequence )
{
   using ValueType = typename TestFixture::ValueType;

   for( int i = this->localRange.getBegin(); i < this->localRange.getEnd(); i++ ) {
      this->input_host[ i ] = i;
      this->expected_host[ i ] = (i * (i - 1)) / 2;
   }

   this->a = this->input_host;
   distributedInplaceExclusiveScan( this->a, 0, this->globalSize, std::plus<>{}, (ValueType) 0 );
   this->template checkResult< ScanType::Exclusive >( this->a );

   this->resetWorkingArrays();

   this->a = this->input_host;
   distributedInplaceExclusiveScan( this->a_view, 0, this->globalSize, std::plus<>{}, (ValueType) 0 );
   this->template checkResult< ScanType::Exclusive >( this->a );
}


TYPED_TEST( DistributedScanTest, multiplication )
{
   this->localRange = Partitioner< typename TestFixture::IndexType >::splitRange( 10, this->communicator );
   this->input_host.setDistribution( this->localRange, 0, 10, this->communicator );
   this->input_host.setValue( 2 );
   this->expected_host = this->input_host;

   // exclusive scan test
   int value = 1;
   for( int i = 0; i < this->localRange.getEnd(); i++ ) {
      if( this->localRange.getBegin() <= i )
         this->expected_host[ i ] = value;
      value *= 2;
   }

   this->a = this->input_host;
   this->b = this->input_host;
   distributedExclusiveScan( this->a, this->b, 0, this->a.getSize(), TNL::Multiplies{} );
   this->template checkResult< ScanType::Exclusive >( this->b );
   EXPECT_EQ( this->a, this->input_host );
   distributedInplaceExclusiveScan( this->a, 0, this->a.getSize(), TNL::Multiplies{} );
   this->template checkResult< ScanType::Exclusive >( this->a );

   // inclusive scan test
   for( int i = this->localRange.getBegin(); i < this->localRange.getEnd(); i++ )
      this->expected_host[ i ] *= 2;

   this->a.reset();
   this->a = this->input_host;
   this->b = this->input_host;
   distributedInclusiveScan( this->a, this->b, 0, this->a.getSize(), TNL::Multiplies{} );
   this->template checkResult< ScanType::Inclusive >( this->b );
   EXPECT_EQ( this->a, this->input_host );
   distributedInplaceInclusiveScan( this->a, 0, this->a.getSize(), TNL::Multiplies{} );
   this->template checkResult< ScanType::Inclusive >( this->a );
}

TYPED_TEST( DistributedScanTest, custom_begin_end )
{
   using IndexType = typename TestFixture::IndexType;

   // make it span multiple processes
   const IndexType begin = 42;
   const IndexType end = (this->nproc > 1) ? this->globalSize / this->nproc + begin : this->globalSize - begin;

   // exclusive scan test
   this->input_host.setValue( 1 );
   this->expected_host.setValue( 1 );
   int value = 0;
   for( int i = begin; i < end; i++ ) {
      if( this->localRange.getBegin() <= i && i < this->localRange.getEnd() )
         this->expected_host[ i ] = value;
      value++;
   }

   this->a = this->input_host;
   this->b = this->input_host;
   distributedExclusiveScan( this->a, this->b, begin, end );
   this->template checkResult< ScanType::Exclusive >( this->b, false );
   EXPECT_EQ( this->a, this->input_host );
   distributedInplaceExclusiveScan( this->a, begin, end );
   this->template checkResult< ScanType::Exclusive >( this->a, false );

   // inclusive scan test
   for( int i = begin; i < end; i++ )
      if( this->localRange.getBegin() <= i && i < this->localRange.getEnd() )
         this->expected_host[ i ]++;

   this->a.reset();
   this->a = this->input_host;
   this->b = this->input_host;
   distributedInclusiveScan( this->a, this->b, begin, end );
   this->template checkResult< ScanType::Inclusive >( this->b, false );
   EXPECT_EQ( this->a, this->input_host );
   distributedInplaceInclusiveScan( this->a, begin, end );
   this->template checkResult< ScanType::Inclusive >( this->a, false );
}

TYPED_TEST( DistributedScanTest, empty_range )
{
   using IndexType = typename TestFixture::IndexType;

   this->localRange = Partitioner< typename TestFixture::IndexType >::splitRange( 42, this->communicator );
   this->input_host.setDistribution( this->localRange, 0, 42, this->communicator );
   this->input_host.setValue( 1 );
   this->expected_host = this->input_host;

   const IndexType begin = 2;
   const IndexType end = 1;

   // exclusive scan test
   this->a = this->input_host;
   this->b = this->input_host;
   distributedExclusiveScan( this->a, this->b, begin, end );
   this->template checkResult< ScanType::Exclusive >( this->b, false );
   EXPECT_EQ( this->a, this->input_host );
   distributedInplaceExclusiveScan( this->a, begin, end );
   this->template checkResult< ScanType::Exclusive >( this->a, false );

   // inclusive scan test
   this->a.reset();
   this->a = this->input_host;
   this->b = this->input_host;
   distributedInclusiveScan( this->a, this->b, begin, end );
   this->template checkResult< ScanType::Inclusive >( this->b, false );
   EXPECT_EQ( this->a, this->input_host );
   distributedInplaceInclusiveScan( this->a, begin, end );
   this->template checkResult< ScanType::Inclusive >( this->a, false );
}

TYPED_TEST( DistributedScanTest, vector_expression )
{
   this->a.setValue( 2 );
   this->b.setValue( 1 );

   // exclusive scan test
   for( int i = this->localRange.getBegin(); i < this->localRange.getEnd(); i++ )
      this->expected_host[ i ] = i;

   this->c.setValue( 0 );
   distributedExclusiveScan( this->av_view - this->bv_view, this->c, 0, this->a.getSize(), TNL::Plus{} );
   this->template checkResult< ScanType::Exclusive >( this->c );

   // inclusive scan test
   for( int i = this->localRange.getBegin(); i < this->localRange.getEnd(); i++ )
      this->expected_host[ i ]++;

   this->c.setValue( 0 );
   distributedInclusiveScan( this->av_view - this->bv_view, this->c, 0, this->a.getSize(), TNL::Plus{} );
   this->template checkResult< ScanType::Inclusive >( this->c );
}

#endif  // HAVE_GTEST

#include "../main_mpi.h"
