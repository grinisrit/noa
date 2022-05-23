#ifdef HAVE_GTEST
#include <gtest/gtest.h>

#include <TNL/Containers/DistributedNDArray.h>
#include <TNL/Containers/DistributedNDArrayView.h>
#include <TNL/Containers/ArrayView.h>
#include <TNL/Containers/Partitioner.h>

using namespace TNL;
using namespace TNL::Containers;

static constexpr int Q = 9;

/*
 * Light check of DistributedNDArray.
 *
 * - Number of processes is not limited.
 * - Global size is hardcoded as 97 to force non-uniform distribution.
 * - Communicator is hardcoded as MPI_COMM_WORLD -- it may be changed as needed.
 */
template< typename DistributedNDArray >
class DistributedNDArray_semi1D_test
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

   const MPI_Comm communicator = MPI_COMM_WORLD;

   DistributedNDArrayType distributedNDArray;

   const int rank = TNL::MPI::GetRank(communicator);
   const int nproc = TNL::MPI::GetSize(communicator);

   DistributedNDArray_semi1D_test()
   {
      using LocalRangeType = typename DistributedNDArray::LocalRangeType;
      const LocalRangeType localRange = Partitioner< IndexType >::splitRange( globalSize, communicator );
      distributedNDArray.setSizes( 0, globalSize, globalSize / 2 );
      distributedNDArray.template setDistribution< 1 >( localRange.getBegin(), localRange.getEnd(), communicator );
      distributedNDArray.allocate();

      EXPECT_EQ( distributedNDArray.template getLocalRange< 1 >(), localRange );
      EXPECT_EQ( distributedNDArray.getCommunicator(), communicator );
   }
};

// types for which DistributedNDArray_semi1D_test is instantiated
using DistributedNDArrayTypes = ::testing::Types<
   DistributedNDArray< NDArray< double,
                                SizesHolder< int, Q, 0, 0 >,  // Q, X, Y, Z
                                std::index_sequence< 0, 1, 2 >,  // permutation - should not matter
                                Devices::Host > >
#ifdef HAVE_CUDA
   ,
   DistributedNDArray< NDArray< double,
                                SizesHolder< int, Q, 0, 0 >,  // Q, X, Y, Z
                                std::index_sequence< 0, 1, 2 >,  // permutation - should not matter
                                Devices::Cuda > >
#endif
>;

TYPED_TEST_SUITE( DistributedNDArray_semi1D_test, DistributedNDArrayTypes );

TYPED_TEST( DistributedNDArray_semi1D_test, checkSumOfLocalSizes )
{
   const auto localRange = this->distributedNDArray.template getLocalRange< 1 >();
   const int localSize = localRange.getEnd() - localRange.getBegin();
   int sumOfLocalSizes = 0;
   TNL::MPI::Allreduce( &localSize, &sumOfLocalSizes, 1, MPI_SUM, this->communicator );
   EXPECT_EQ( sumOfLocalSizes, this->globalSize );
   EXPECT_EQ( this->distributedNDArray.template getSize< 1 >(), this->globalSize );
}

TYPED_TEST( DistributedNDArray_semi1D_test, setLike )
{
   using DistributedNDArrayType = typename TestFixture::DistributedNDArrayType;

   const auto localRange = this->distributedNDArray.template getLocalRange< 1 >();
   EXPECT_EQ( this->distributedNDArray.getLocalStorageSize(), Q * (localRange.getEnd() - localRange.getBegin()) * (this->globalSize / 2) );
   DistributedNDArrayType copy;
   EXPECT_EQ( copy.getLocalStorageSize(), 0 );
   copy.setLike( this->distributedNDArray );
   EXPECT_EQ( copy.getLocalStorageSize(), Q * (localRange.getEnd() - localRange.getBegin()) * (this->globalSize / 2) );
}

TYPED_TEST( DistributedNDArray_semi1D_test, reset )
{
   const auto localRange = this->distributedNDArray.template getLocalRange< 1 >();
   EXPECT_EQ( this->distributedNDArray.getLocalStorageSize(), Q * (localRange.getEnd() - localRange.getBegin()) * (this->globalSize / 2) );
   this->distributedNDArray.reset();
   EXPECT_EQ( this->distributedNDArray.getLocalStorageSize(), 0 );
}

TYPED_TEST( DistributedNDArray_semi1D_test, elementwiseAccess )
{
//   using ArrayViewType = typename TestFixture::ArrayViewType;
   using IndexType = typename TestFixture::IndexType;

   this->distributedNDArray.setValue( 0 );
//   ArrayViewType localArrayView = this->distributedNDArray.getLocalArrayView();
   const auto localRange = this->distributedNDArray.template getLocalRange< 1 >();

   // check initial value
   for( int q = 0; q < Q; q++ )
   for( IndexType gi = localRange.getBegin(); gi < localRange.getEnd(); gi++ )
   for( int j = 0; j < this->distributedNDArray.template getSize< 2 >(); j++ ) {
//      EXPECT_EQ( localArrayView.getElement( i ), 0 );
      EXPECT_EQ( this->distributedNDArray.getElement( q, gi, j ), 0 );
   }

   // use operator()
   if( std::is_same< typename TestFixture::DeviceType, Devices::Host >::value ) {
      for( int q = 0; q < Q; q++ )
      for( IndexType gi = localRange.getBegin(); gi < localRange.getEnd(); gi++ )
      for( int j = 0; j < this->distributedNDArray.template getSize< 2 >(); j++ ) {
         this->distributedNDArray( q, gi, j ) = gi + 1;
      }

      // check set value
      for( int q = 0; q < Q; q++ )
      for( IndexType gi = localRange.getBegin(); gi < localRange.getEnd(); gi++ )
      for( int j = 0; j < this->distributedNDArray.template getSize< 2 >(); j++ ) {
         EXPECT_EQ( this->distributedNDArray.getElement( q, gi, j ), gi + 1 );
         EXPECT_EQ( this->distributedNDArray( q, gi, j ), gi + 1 );
      }
   }
}

TYPED_TEST( DistributedNDArray_semi1D_test, copyAssignment )
{
   using DistributedNDArrayType = typename TestFixture::DistributedNDArrayType;

   this->distributedNDArray.setValue( 1 );
   DistributedNDArrayType copy;
   copy = this->distributedNDArray;
   // no binding, but deep copy
//   EXPECT_NE( copy.getLocalArrayView().getData(), this->distributedNDArray.getLocalArrayView().getData() );
//   EXPECT_EQ( copy.getLocalArrayView(), this->distributedNDArray.getLocalArrayView() );
}

// separate function because nvcc does not allow __cuda_callable__ lambdas inside
// private or protected methods (which are created by TYPED_TEST macro)
template< typename DistributedArray >
void test_helper_comparisonOperators( DistributedArray& u, DistributedArray& v, DistributedArray& w )
{
   using DeviceType = typename DistributedArray::DeviceType;
   using IndexType = typename DistributedArray::IndexType;

   const auto localRange = u.template getLocalRange< 1 >();
   auto u_view = u.getView();
   auto v_view = v.getView();
   auto w_view = w.getView();

   auto kernel = [=] __cuda_callable__ ( IndexType q, IndexType gi, IndexType j ) mutable
   {
      u_view( q, gi, j ) = gi;
      v_view( q, gi, j ) = gi;
      w_view( q, gi, j ) = 2 * gi;
   };
   Algorithms::ParallelFor3D< DeviceType >::exec( (IndexType) 0, localRange.getBegin(), (IndexType) 0,
                                      Q, localRange.getEnd(), u.template getSize< 2 >(),
                                      kernel );
}

TYPED_TEST( DistributedNDArray_semi1D_test, comparisonOperators )
{
   using DistributedNDArrayType = typename TestFixture::DistributedNDArrayType;

   DistributedNDArrayType& u = this->distributedNDArray;
   DistributedNDArrayType v, w;
   v.setLike( u );
   w.setLike( u );

   test_helper_comparisonOperators( u, v, w );

   EXPECT_TRUE( u == u );
   EXPECT_TRUE( u == v );
   EXPECT_TRUE( v == u );
   EXPECT_FALSE( u != v );
   EXPECT_FALSE( v != u );
   EXPECT_TRUE( u != w );
   EXPECT_TRUE( w != u );
   EXPECT_FALSE( u == w );
   EXPECT_FALSE( w == u );

   v.reset();
   EXPECT_FALSE( u == v );
   u.reset();
   EXPECT_TRUE( u == v );
}

// separate function because nvcc does not allow __cuda_callable__ lambdas inside
// private or protected methods (which are created by TYPED_TEST macro)
template< typename DistributedArray >
void test_helper_forAll( DistributedArray& a )
{
   using IndexType = typename DistributedArray::IndexType;

   const auto localRange = a.template getLocalRange< 1 >();
   auto a_view = a.getView();

   auto setter = [=] __cuda_callable__ ( IndexType q, IndexType i, IndexType j ) mutable
   {
      a_view( q, i, j ) += 1;
   };

   a.setValue( 0 );
   a.forAll( setter );

   for( int q = 0; q < Q; q++ )
   for( int gi = localRange.getBegin(); gi < localRange.getEnd(); gi++ )
   for( int j = 0; j < a.template getSize< 2 >(); j++ )
      EXPECT_EQ( a.getElement( q, gi, j ), 1 );

   a.setValue( 0 );
   a_view.forAll( setter );

   for( int q = 0; q < Q; q++ )
   for( int gi = localRange.getBegin(); gi < localRange.getEnd(); gi++ )
   for( int j = 0; j < a.template getSize< 2 >(); j++ )
      EXPECT_EQ( a.getElement( q, gi, j ), 1 );
}

TYPED_TEST( DistributedNDArray_semi1D_test, forAll )
{
   test_helper_forAll( this->distributedNDArray );
}

// separate function because nvcc does not allow __cuda_callable__ lambdas inside
// private or protected methods (which are created by TYPED_TEST macro)
template< typename DistributedArray >
void test_helper_forInternal( DistributedArray& a )
{
   using IndexType = typename DistributedArray::IndexType;

   const auto localRange = a.template getLocalRange< 1 >();
   auto a_view = a.getView();

   auto setter = [=] __cuda_callable__ ( IndexType q, IndexType i, IndexType j ) mutable
   {
      a_view( q, i, j ) += 1;
   };

   a.setValue( 0 );
   a.forInternal( setter );

   for( int q = 0; q < Q; q++ )
   for( int gi = localRange.getBegin(); gi < localRange.getEnd(); gi++ )
   for( int j = 0; j < a.template getSize< 2 >(); j++ )
   {
      if( q == 0 || q == 8 ||
          gi == 0 || gi == a.template getSize< 1 >() - 1 ||
          j == 0 || j == a.template getSize< 2 >() - 1 )
         EXPECT_EQ( a.getElement( q, gi, j ), 0 )
            << "gi = " << gi;
      else
         EXPECT_EQ( a.getElement( q, gi, j ), 1 )
            << "gi = " << gi;
   }

   a.setValue( 0 );
   a_view.forInternal( setter );

   for( int q = 0; q < Q; q++ )
   for( int gi = localRange.getBegin(); gi < localRange.getEnd(); gi++ )
   for( int j = 0; j < a.template getSize< 2 >(); j++ )
   {
      if( q == 0 || q == 8 ||
          gi == 0 || gi == a.template getSize< 1 >() - 1 ||
          j == 0 || j == a.template getSize< 2 >() - 1 )
         EXPECT_EQ( a.getElement( q, gi, j ), 0 )
            << "gi = " << gi;
      else
         EXPECT_EQ( a.getElement( q, gi, j ), 1 )
            << "gi = " << gi;
   }
}

TYPED_TEST( DistributedNDArray_semi1D_test, forInternal )
{
   test_helper_forInternal( this->distributedNDArray );
}

// separate function because nvcc does not allow __cuda_callable__ lambdas inside
// private or protected methods (which are created by TYPED_TEST macro)
template< typename DistributedArray >
void test_helper_forLocalInternal( DistributedArray& a )
{
   using IndexType = typename DistributedArray::IndexType;

   const auto localRange = a.template getLocalRange< 1 >();
   auto a_view = a.getView();

   auto setter = [=] __cuda_callable__ ( IndexType q, IndexType i, IndexType j ) mutable
   {
      a_view( q, i, j ) += 1;
   };

   a.setValue( 0 );
   // equivalent to forAll because all overlaps are 0
   a.forLocalInternal( setter );

   for( int q = 0; q < Q; q++ )
   for( int gi = localRange.getBegin(); gi < localRange.getEnd(); gi++ )
   for( int j = 0; j < a.template getSize< 2 >(); j++ )
      EXPECT_EQ( a.getElement( q, gi, j ), 1 );

   a.setValue( 0 );
   // equivalent to forAll because all overlaps are 0
   a_view.forLocalInternal( setter );

   for( int q = 0; q < Q; q++ )
   for( int gi = localRange.getBegin(); gi < localRange.getEnd(); gi++ )
   for( int j = 0; j < a.template getSize< 2 >(); j++ )
      EXPECT_EQ( a.getElement( q, gi, j ), 1 );
}

TYPED_TEST( DistributedNDArray_semi1D_test, forLocalInternal )
{
   test_helper_forLocalInternal( this->distributedNDArray );
}

// separate function because nvcc does not allow __cuda_callable__ lambdas inside
// private or protected methods (which are created by TYPED_TEST macro)
template< typename DistributedArray >
void test_helper_forBoundary( DistributedArray& a )
{
   using IndexType = typename DistributedArray::IndexType;

   const auto localRange = a.template getLocalRange< 1 >();
   auto a_view = a.getView();

   auto setter = [=] __cuda_callable__ ( IndexType q, IndexType i, IndexType j ) mutable
   {
      a_view( q, i, j ) += 1;
   };

   a.setValue( 0 );
   a.forBoundary( setter );

   for( int q = 0; q < Q; q++ )
   for( int gi = localRange.getBegin(); gi < localRange.getEnd(); gi++ )
   for( int j = 0; j < a.template getSize< 2 >(); j++ )
   {
      if( q == 0 || q == 8 ||
          gi == 0 || gi == a.template getSize< 1 >() - 1 ||
          j == 0 || j == a.template getSize< 2 >() - 1 )
         EXPECT_EQ( a.getElement( q, gi, j ), 1 )
            << "gi = " << gi;
      else
         EXPECT_EQ( a.getElement( q, gi, j ), 0 )
            << "gi = " << gi;
   }

   a.setValue( 0 );
   a_view.forBoundary( setter );

   for( int q = 0; q < Q; q++ )
   for( int gi = localRange.getBegin(); gi < localRange.getEnd(); gi++ )
   for( int j = 0; j < a.template getSize< 2 >(); j++ )
   {
      if( q == 0 || q == 8 ||
          gi == 0 || gi == a.template getSize< 1 >() - 1 ||
          j == 0 || j == a.template getSize< 2 >() - 1 )
         EXPECT_EQ( a.getElement( q, gi, j ), 1 )
            << "gi = " << gi;
      else
         EXPECT_EQ( a.getElement( q, gi, j ), 0 )
            << "gi = " << gi;
   }
}

TYPED_TEST( DistributedNDArray_semi1D_test, forBoundary )
{
   test_helper_forBoundary( this->distributedNDArray );
}

// separate function because nvcc does not allow __cuda_callable__ lambdas inside
// private or protected methods (which are created by TYPED_TEST macro)
template< typename DistributedArray >
void test_helper_forLocalBoundary( DistributedArray& a )
{
   using IndexType = typename DistributedArray::IndexType;

   const auto localRange = a.template getLocalRange< 1 >();
   auto a_view = a.getView();

   auto setter = [=] __cuda_callable__ ( IndexType q, IndexType i, IndexType j ) mutable
   {
      a_view( q, i, j ) += 1;
   };

   a.setValue( 0 );
   // empty set because all overlaps are 0
   a.forLocalBoundary( setter );

   for( int q = 0; q < Q; q++ )
   for( int gi = localRange.getBegin(); gi < localRange.getEnd(); gi++ )
   for( int j = 0; j < a.template getSize< 2 >(); j++ )
      EXPECT_EQ( a.getElement( q, gi, j ), 0 );

   a.setValue( 0 );
   // empty set because all overlaps are 0
   a_view.forLocalBoundary( setter );

   for( int q = 0; q < Q; q++ )
   for( int gi = localRange.getBegin(); gi < localRange.getEnd(); gi++ )
   for( int j = 0; j < a.template getSize< 2 >(); j++ )
      EXPECT_EQ( a.getElement( q, gi, j ), 0 );
}

TYPED_TEST( DistributedNDArray_semi1D_test, forLocalBoundary )
{
   test_helper_forLocalBoundary( this->distributedNDArray );
}

// separate function because nvcc does not allow __cuda_callable__ lambdas inside
// private or protected methods (which are created by TYPED_TEST macro)
template< typename DistributedArray >
void test_helper_forOverlaps( DistributedArray& a )
{
   using IndexType = typename DistributedArray::IndexType;

   const auto localRange = a.template getLocalRange< 1 >();
   auto a_view = a.getView();

   auto setter = [=] __cuda_callable__ ( IndexType q, IndexType i, IndexType j ) mutable
   {
      a_view( q, i, j ) += 1;
   };

   a.setValue( 0 );
   // empty set because all overlaps are 0
   a.forOverlaps( setter );

   for( int q = 0; q < Q; q++ )
   for( int gi = localRange.getBegin(); gi < localRange.getEnd(); gi++ )
   for( int j = 0; j < a.template getSize< 2 >(); j++ )
      EXPECT_EQ( a.getElement( q, gi, j ), 0 );

   a.setValue( 0 );
   // empty set because all overlaps are 0
   a_view.forOverlaps( setter );

   for( int q = 0; q < Q; q++ )
   for( int gi = localRange.getBegin(); gi < localRange.getEnd(); gi++ )
   for( int j = 0; j < a.template getSize< 2 >(); j++ )
      EXPECT_EQ( a.getElement( q, gi, j ), 0 );
}

TYPED_TEST( DistributedNDArray_semi1D_test, forOverlaps )
{
   test_helper_forOverlaps( this->distributedNDArray );
}

#endif  // HAVE_GTEST


#include "../../main_mpi.h"
