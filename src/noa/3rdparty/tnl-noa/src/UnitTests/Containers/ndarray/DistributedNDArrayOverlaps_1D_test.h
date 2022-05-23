#ifdef HAVE_GTEST
#include <gtest/gtest.h>

#include <TNL/Containers/DistributedNDArray.h>
#include <TNL/Containers/DistributedNDArrayView.h>
#include <TNL/Containers/DistributedNDArraySynchronizer.h>
#include <TNL/Containers/ArrayView.h>
#include <TNL/Containers/Partitioner.h>

using namespace TNL;
using namespace TNL::Containers;

/*
 * Light check of DistributedNDArray.
 *
 * - Number of processes is not limited.
 * - Global size is hardcoded as 97 to force non-uniform distribution.
 * - Communicator is hardcoded as MPI_COMM_WORLD -- it may be changed as needed.
 */
template< typename DistributedNDArray >
class DistributedNDArrayOverlaps_1D_test
: public ::testing::Test
{
protected:
   using ValueType = typename DistributedNDArray::ValueType;
   using DeviceType = typename DistributedNDArray::DeviceType;
   using IndexType = typename DistributedNDArray::IndexType;
   using DistributedNDArrayType = DistributedNDArray;

   // TODO: use ndarray
   using LocalArrayType = Array< ValueType, DeviceType, IndexType >;
   using LocalArrayViewType = ArrayView< ValueType, DeviceType, IndexType >;

   const int globalSize = 97;  // prime number to force non-uniform distribution
   const int overlaps = __ndarray_impl::get< 0 >( typename DistributedNDArray::OverlapsType{} );

   const MPI_Comm communicator = MPI_COMM_WORLD;

   DistributedNDArrayType distributedNDArray;

   const int rank = TNL::MPI::GetRank(communicator);
   const int nproc = TNL::MPI::GetSize(communicator);

   DistributedNDArrayOverlaps_1D_test()
   {
      using LocalRangeType = typename DistributedNDArray::LocalRangeType;
      const LocalRangeType localRange = Partitioner< IndexType >::splitRange( globalSize, communicator );
      distributedNDArray.setSizes( globalSize );
      distributedNDArray.template setDistribution< 0 >( localRange.getBegin(), localRange.getEnd(), communicator );
      distributedNDArray.allocate();

      EXPECT_EQ( distributedNDArray.template getLocalRange< 0 >(), localRange );
      EXPECT_EQ( distributedNDArray.getCommunicator(), communicator );
   }
};

// types for which DistributedNDArrayOverlaps_1D_test is instantiated
using DistributedNDArrayTypes = ::testing::Types<
   DistributedNDArray< NDArray< double,
                                SizesHolder< int, 0 >,
                                std::index_sequence< 0 >,
                                Devices::Host,
                                int,
                                std::index_sequence< 2 > > >  // overlaps
#ifdef HAVE_CUDA
   ,
   DistributedNDArray< NDArray< double,
                                SizesHolder< int, 0 >,
                                std::index_sequence< 0 >,
                                Devices::Cuda,
                                int,
                                std::index_sequence< 2 > > >  // overlaps
#endif
>;

TYPED_TEST_SUITE( DistributedNDArrayOverlaps_1D_test, DistributedNDArrayTypes );

TYPED_TEST( DistributedNDArrayOverlaps_1D_test, checkSumOfLocalSizes )
{
   const auto localRange = this->distributedNDArray.template getLocalRange< 0 >();
   const int localSize = localRange.getEnd() - localRange.getBegin();
   int sumOfLocalSizes = 0;
   TNL::MPI::Allreduce( &localSize, &sumOfLocalSizes, 1, MPI_SUM, this->communicator );
   EXPECT_EQ( sumOfLocalSizes, this->globalSize );
   EXPECT_EQ( this->distributedNDArray.template getSize< 0 >(), this->globalSize );

   EXPECT_EQ( this->distributedNDArray.getLocalStorageSize(), 2 * this->overlaps + localSize );
}

// separate function because nvcc does not allow __cuda_callable__ lambdas inside
// private or protected methods (which are created by TYPED_TEST macro)
template< typename DistributedArray >
void test_helper_forLocalInternal( DistributedArray& a )
{
   using IndexType = typename DistributedArray::IndexType;

   const int overlaps = __ndarray_impl::get< 0 >( typename DistributedArray::OverlapsType{} );
   const auto localRange = a.template getLocalRange< 0 >();
   auto a_view = a.getView();

   auto setter = [=] __cuda_callable__ ( IndexType i ) mutable
   {
      a_view( i ) += 1;
   };

   a.setValue( 0 );
   a.forLocalInternal( setter );

   for( int gi = localRange.getBegin(); gi < localRange.getBegin() + overlaps; gi++ )
      EXPECT_EQ( a.getElement( gi ), 0 )
            << "gi = " << gi;
   for( int gi = localRange.getBegin() + overlaps; gi < localRange.getEnd() - overlaps; gi++ )
      EXPECT_EQ( a.getElement( gi ), 1 )
            << "gi = " << gi;
   for( int gi = localRange.getEnd() - overlaps; gi < localRange.getEnd(); gi++ )
      EXPECT_EQ( a.getElement( gi ), 0 )
            << "gi = " << gi;

   a.setValue( 0 );
   a_view.forLocalInternal( setter );

   for( int gi = localRange.getBegin(); gi < localRange.getBegin() + overlaps; gi++ )
      EXPECT_EQ( a.getElement( gi ), 0 )
            << "gi = " << gi;
   for( int gi = localRange.getBegin() + overlaps; gi < localRange.getEnd() - overlaps; gi++ )
      EXPECT_EQ( a.getElement( gi ), 1 )
            << "gi = " << gi;
   for( int gi = localRange.getEnd() - overlaps; gi < localRange.getEnd(); gi++ )
      EXPECT_EQ( a.getElement( gi ), 0 )
            << "gi = " << gi;
}

TYPED_TEST( DistributedNDArrayOverlaps_1D_test, forLocalInternal )
{
   test_helper_forLocalInternal( this->distributedNDArray );
}

// separate function because nvcc does not allow __cuda_callable__ lambdas inside
// private or protected methods (which are created by TYPED_TEST macro)
template< typename DistributedArray >
void test_helper_forLocalBoundary( DistributedArray& a )
{
   using IndexType = typename DistributedArray::IndexType;

   const int overlaps = __ndarray_impl::get< 0 >( typename DistributedArray::OverlapsType{} );
   const auto localRange = a.template getLocalRange< 0 >();
   auto a_view = a.getView();

   auto setter = [=] __cuda_callable__ ( IndexType i ) mutable
   {
      a_view( i ) += 1;
   };

   a.setValue( 0 );
   a.forLocalBoundary( setter );

   for( int gi = localRange.getBegin(); gi < localRange.getBegin() + overlaps; gi++ )
      EXPECT_EQ( a.getElement( gi ), 1 )
            << "gi = " << gi;
   for( int gi = localRange.getBegin() + overlaps; gi < localRange.getEnd() - overlaps; gi++ )
      EXPECT_EQ( a.getElement( gi ), 0 )
            << "gi = " << gi;
   for( int gi = localRange.getEnd() - overlaps; gi < localRange.getEnd(); gi++ )
      EXPECT_EQ( a.getElement( gi ), 1 )
            << "gi = " << gi;

   a.setValue( 0 );
   a_view.forLocalBoundary( setter );

   for( int gi = localRange.getBegin(); gi < localRange.getBegin() + overlaps; gi++ )
      EXPECT_EQ( a.getElement( gi ), 1 )
            << "gi = " << gi;
   for( int gi = localRange.getBegin() + overlaps; gi < localRange.getEnd() - overlaps; gi++ )
      EXPECT_EQ( a.getElement( gi ), 0 )
            << "gi = " << gi;
   for( int gi = localRange.getEnd() - overlaps; gi < localRange.getEnd(); gi++ )
      EXPECT_EQ( a.getElement( gi ), 1 )
            << "gi = " << gi;
}

TYPED_TEST( DistributedNDArrayOverlaps_1D_test, forLocalBoundary )
{
   test_helper_forLocalBoundary( this->distributedNDArray );
}

// separate function because nvcc does not allow __cuda_callable__ lambdas inside
// private or protected methods (which are created by TYPED_TEST macro)
template< typename DistributedArray >
void test_helper_forOverlaps( DistributedArray& a )
{
   using IndexType = typename DistributedArray::IndexType;

   const int overlaps = __ndarray_impl::get< 0 >( typename DistributedArray::OverlapsType{} );
   const auto localRange = a.template getLocalRange< 0 >();
   auto a_view = a.getView();

   auto setter = [=] __cuda_callable__ ( IndexType i ) mutable
   {
      a_view( i ) += 1;
   };

   a.setValue( 0 );
   a.forOverlaps( setter );

   for( int gi = localRange.getBegin() - overlaps; gi < localRange.getBegin(); gi++ )
      EXPECT_EQ( a.getElement( gi ), 1 )
            << "gi = " << gi;
   for( int gi = localRange.getBegin(); gi < localRange.getEnd(); gi++ )
      EXPECT_EQ( a.getElement( gi ), 0 )
            << "gi = " << gi;
   for( int gi = localRange.getEnd(); gi < localRange.getEnd() + overlaps; gi++ )
      EXPECT_EQ( a.getElement( gi ), 1 )
            << "gi = " << gi;

   a.setValue( 0 );
   a_view.forOverlaps( setter );

   for( int gi = localRange.getBegin() - overlaps; gi < localRange.getBegin(); gi++ )
      EXPECT_EQ( a.getElement( gi ), 1 )
            << "gi = " << gi;
   for( int gi = localRange.getBegin(); gi < localRange.getEnd(); gi++ )
      EXPECT_EQ( a.getElement( gi ), 0 )
            << "gi = " << gi;
   for( int gi = localRange.getEnd(); gi < localRange.getEnd() + overlaps; gi++ )
      EXPECT_EQ( a.getElement( gi ), 1 )
            << "gi = " << gi;
}

TYPED_TEST( DistributedNDArrayOverlaps_1D_test, forOverlaps )
{
   test_helper_forOverlaps( this->distributedNDArray );
}

// separate function because nvcc does not allow __cuda_callable__ lambdas inside
// private or protected methods (which are created by TYPED_TEST macro)
template< typename DistributedArray >
void test_helper_synchronize( DistributedArray& a, const int rank, const int nproc )
{
   using IndexType = typename DistributedArray::IndexType;

   const int overlaps = __ndarray_impl::get< 0 >( typename DistributedArray::OverlapsType{} );
   const auto localRange = a.template getLocalRange< 0 >();
   auto a_view = a.getView();

   auto setter = [=] __cuda_callable__ ( IndexType i ) mutable
   {
      a_view( i ) = i;
   };

   a.setValue( -1 );
   a.forAll( setter );
   DistributedNDArraySynchronizer< DistributedArray > s1;
   s1.synchronize( a );

   for( int gi = localRange.getBegin() - overlaps; gi < localRange.getBegin(); gi++ )
      EXPECT_EQ( a.getElement( gi ), gi + ((rank == 0) ? 97 : 0) );
   for( int gi = localRange.getBegin(); gi < localRange.getEnd(); gi++ )
      EXPECT_EQ( a.getElement( gi ), gi );
   for( int gi = localRange.getEnd(); gi < localRange.getEnd() + overlaps; gi++ )
      EXPECT_EQ( a.getElement( gi ), gi - ((rank == nproc-1) ? 97 : 0) );

   a.setValue( -1 );
   a_view.forAll( setter );
   DistributedNDArraySynchronizer< decltype(a_view) > s2;
   s2.synchronize( a_view );

   for( int gi = localRange.getBegin() - overlaps; gi < localRange.getBegin(); gi++ )
      EXPECT_EQ( a.getElement( gi ), gi + ((rank == 0) ? 97 : 0) );
   for( int gi = localRange.getBegin(); gi < localRange.getEnd(); gi++ )
      EXPECT_EQ( a.getElement( gi ), gi );
   for( int gi = localRange.getEnd(); gi < localRange.getEnd() + overlaps; gi++ )
      EXPECT_EQ( a.getElement( gi ), gi - ((rank == nproc-1) ? 97 : 0) );
}

TYPED_TEST( DistributedNDArrayOverlaps_1D_test, synchronize )
{
   test_helper_synchronize( this->distributedNDArray, this->rank, this->nproc );
}

#endif  // HAVE_GTEST


#include "../../main_mpi.h"
