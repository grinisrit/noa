#ifdef HAVE_GTEST
#include <gtest/gtest.h>

#include <TNL/Containers/DistributedArray.h>
#include <TNL/Containers/Partitioner.h>

#include "VectorHelperFunctions.h"

using namespace TNL;
using namespace TNL::Containers;
using namespace TNL::MPI;

/*
 * Light check of DistributedArray.
 *
 * - Number of processes is not limited.
 * - Global size is hardcoded as 97 to force non-uniform distribution.
 * - Communicator is hardcoded as MPI_COMM_WORLD -- it may be changed as needed.
 */
template< typename DistributedArray >
class DistributedArrayTest
: public ::testing::Test
{
protected:
   using ValueType = typename DistributedArray::ValueType;
   using DeviceType = typename DistributedArray::DeviceType;
   using IndexType = typename DistributedArray::IndexType;
   using DistributedArrayType = DistributedArray;
   using ArrayViewType = typename DistributedArrayType::LocalViewType;
   using ArrayType = Array< typename ArrayViewType::ValueType, typename ArrayViewType::DeviceType, typename ArrayViewType::IndexType >;

   const int globalSize = 97;  // prime number to force non-uniform distribution

   const MPI_Comm communicator = MPI_COMM_WORLD;

   DistributedArrayType distributedArray;

   const int rank = GetRank(communicator);
   const int nproc = GetSize(communicator);

   // some arbitrary even value (but must be 0 if not distributed)
   const int ghosts = (nproc > 1) ? 4 : 0;

   DistributedArrayTest()
   {
      using LocalRangeType = typename DistributedArray::LocalRangeType;
      const LocalRangeType localRange = Partitioner< IndexType >::splitRange( globalSize, communicator );
      distributedArray.setDistribution( localRange, ghosts, globalSize, communicator );

      using Synchronizer = typename Partitioner< IndexType >::template ArraySynchronizer< DeviceType >;
      distributedArray.setSynchronizer( std::make_shared<Synchronizer>( localRange, ghosts / 2, communicator ) );

      EXPECT_EQ( distributedArray.getLocalRange(), localRange );
      EXPECT_EQ( distributedArray.getGhosts(), ghosts );
      EXPECT_EQ( distributedArray.getCommunicator(), communicator );
   }
};

// types for which DistributedArrayTest is instantiated
using DistributedArrayTypes = ::testing::Types<
   DistributedArray< double, Devices::Host, int >
#ifdef HAVE_CUDA
   ,
   DistributedArray< double, Devices::Cuda, int >
#endif
>;

TYPED_TEST_SUITE( DistributedArrayTest, DistributedArrayTypes );

TYPED_TEST( DistributedArrayTest, checkLocalSizes )
{
   EXPECT_EQ( this->distributedArray.getLocalView().getSize(), this->distributedArray.getLocalRange().getSize() );
   EXPECT_EQ( this->distributedArray.getConstLocalView().getSize(), this->distributedArray.getLocalRange().getSize() );
   EXPECT_EQ( this->distributedArray.getLocalViewWithGhosts().getSize(), this->distributedArray.getLocalRange().getSize() + this->ghosts );
   EXPECT_EQ( this->distributedArray.getConstLocalViewWithGhosts().getSize(), this->distributedArray.getLocalRange().getSize() + this->ghosts );
}

TYPED_TEST( DistributedArrayTest, checkSumOfLocalSizes )
{
   const int localSize = this->distributedArray.getLocalView().getSize();
   int sumOfLocalSizes = 0;
   Allreduce( &localSize, &sumOfLocalSizes, 1, MPI_SUM, this->communicator );
   EXPECT_EQ( sumOfLocalSizes, this->globalSize );
   EXPECT_EQ( this->distributedArray.getSize(), this->globalSize );
}

TYPED_TEST( DistributedArrayTest, copyFromGlobal )
{
   using ArrayViewType = typename TestFixture::ArrayViewType;
   using ArrayType = typename TestFixture::ArrayType;

   this->distributedArray.setValue( 0.0 );
   ArrayType globalArray( this->globalSize );
   setLinearSequence( globalArray );
   this->distributedArray.copyFromGlobal( globalArray );
   this->distributedArray.waitForSynchronization();

   const auto localRange = this->distributedArray.getLocalRange();
   ArrayViewType localArrayView;
   localArrayView.bind( this->distributedArray.getLocalView().getData(), localRange.getSize() );
   auto globalView = globalArray.getConstView();
   globalView.bind( &globalArray.getData()[ localRange.getBegin() ], localRange.getSize() );
   EXPECT_EQ( localArrayView, globalView );

   // check ghost values
   for( int o = 0; o < this->ghosts / 2; o++ ) {
      const int left_i = localRange.getSize() + o;
      const int left_gi = ((this->rank > 0) ? localRange.getBegin() : this->globalSize) - this->ghosts / 2 + o;
      EXPECT_EQ( this->distributedArray.getConstLocalViewWithGhosts().getElement( left_i ), globalArray.getElement( left_gi ) );
      const int right_i = localRange.getSize() + this->ghosts / 2 + o;
      const int right_gi = ((this->rank < this->nproc - 1) ? localRange.getEnd() : 0) + o;
      EXPECT_EQ( this->distributedArray.getConstLocalViewWithGhosts().getElement( right_i ), globalArray.getElement( right_gi ) );
   }
}

TYPED_TEST( DistributedArrayTest, setLike )
{
   using DistributedArrayType = typename TestFixture::DistributedArrayType;

   EXPECT_EQ( this->distributedArray.getSize(), this->globalSize );
   DistributedArrayType copy;
   EXPECT_EQ( copy.getSize(), 0 );
   copy.setLike( this->distributedArray );
   EXPECT_EQ( copy.getSize(), this->globalSize );
}

TYPED_TEST( DistributedArrayTest, reset )
{
   EXPECT_EQ( this->distributedArray.getSize(), this->globalSize );
   EXPECT_GT( this->distributedArray.getLocalView().getSize(), 0 );
   this->distributedArray.reset();
   EXPECT_EQ( this->distributedArray.getSize(), 0 );
   EXPECT_EQ( this->distributedArray.getLocalView().getSize(), 0 );
}

// TODO: swap

TYPED_TEST( DistributedArrayTest, setValue )
{
   using ArrayViewType = typename TestFixture::ArrayViewType;
   using ArrayType = typename TestFixture::ArrayType;

   this->distributedArray.setValue( 1.0 );
   this->distributedArray.waitForSynchronization();
   ArrayViewType localArrayView = this->distributedArray.getLocalView();
   ArrayType expected( localArrayView.getSize() );
   expected.setValue( 1.0 );
   EXPECT_EQ( localArrayView, expected );
}

TYPED_TEST( DistributedArrayTest, setValueGhosts )
{
   using ArrayViewType = typename TestFixture::ArrayViewType;
   using ArrayType = typename TestFixture::ArrayType;

   this->distributedArray.setValue( this->rank );
   this->distributedArray.waitForSynchronization();
   ArrayViewType localArrayView = this->distributedArray.getLocalViewWithGhosts();
   ArrayType expected( localArrayView.getSize() );
   expected.setValue( this->rank );

   // set expected ghost values
   const int left = (this->rank > 0) ? this->rank - 1 : this->nproc - 1;
   const int right = (this->rank < this->nproc - 1) ? this->rank + 1 : 0;
   for( int o = 0; o < this->ghosts / 2; o++ ) {
      expected.setElement( this->distributedArray.getLocalRange().getSize() + o, left );
      expected.setElement( this->distributedArray.getLocalRange().getSize() + this->ghosts / 2 + o, right );
   }

   EXPECT_EQ( localArrayView, expected );
}

TYPED_TEST( DistributedArrayTest, elementwiseAccess )
{
   using ArrayViewType = typename TestFixture::ArrayViewType;
   using IndexType = typename TestFixture::IndexType;

   this->distributedArray.setValue( 0 );
   this->distributedArray.waitForSynchronization();
   ArrayViewType localArrayView = this->distributedArray.getLocalView();
   const auto localRange = this->distributedArray.getLocalRange();

   // check initial value
   for( IndexType i = 0; i < localRange.getSize(); i++ ) {
      const IndexType gi = localRange.getGlobalIndex( i );
      EXPECT_EQ( localArrayView.getElement( i ), 0 );
      EXPECT_EQ( this->distributedArray.getElement( gi ), 0 );
      if( std::is_same< typename TestFixture::DeviceType, Devices::Host >::value ) {
         EXPECT_EQ( this->distributedArray[ gi ], 0 );
      }
   }

   // use setValue
   for( IndexType i = 0; i < localRange.getSize(); i++ ) {
      const IndexType gi = localRange.getGlobalIndex( i );
      this->distributedArray.setElement( gi, i + 1 );
   }

   // check set value
   for( IndexType i = 0; i < localRange.getSize(); i++ ) {
      const IndexType gi = localRange.getGlobalIndex( i );
      EXPECT_EQ( localArrayView.getElement( i ), i + 1 );
      EXPECT_EQ( this->distributedArray.getElement( gi ), i + 1 );
      if( std::is_same< typename TestFixture::DeviceType, Devices::Host >::value ) {
         EXPECT_EQ( this->distributedArray[ gi ], i + 1 );
      }
   }

   this->distributedArray.setValue( 0 );
   this->distributedArray.waitForSynchronization();

   // use operator[]
   if( std::is_same< typename TestFixture::DeviceType, Devices::Host >::value ) {
      for( IndexType i = 0; i < localRange.getSize(); i++ ) {
         const IndexType gi = localRange.getGlobalIndex( i );
         this->distributedArray[ gi ] = i + 1;
      }

      // check set value
      for( IndexType i = 0; i < localRange.getSize(); i++ ) {
         const IndexType gi = localRange.getGlobalIndex( i );
         EXPECT_EQ( localArrayView.getElement( i ), i + 1 );
         EXPECT_EQ( this->distributedArray.getElement( gi ), i + 1 );
         EXPECT_EQ( this->distributedArray[ gi ], i + 1 );
      }
   }
}

TYPED_TEST( DistributedArrayTest, copyConstructor )
{
   using DistributedArrayType = typename TestFixture::DistributedArrayType;

   this->distributedArray.setValue( 1 );
   DistributedArrayType copy( this->distributedArray );
   // no binding, but deep copy
   EXPECT_NE( copy.getLocalView().getData(), this->distributedArray.getLocalView().getData() );
   EXPECT_EQ( copy.getLocalView(), this->distributedArray.getLocalView() );
}

TYPED_TEST( DistributedArrayTest, copyAssignment )
{
   using DistributedArrayType = typename TestFixture::DistributedArrayType;

   this->distributedArray.setValue( 1 );
   DistributedArrayType copy;
   copy = this->distributedArray;
   // no binding, but deep copy
   EXPECT_NE( copy.getLocalView().getData(), this->distributedArray.getLocalView().getData() );
   EXPECT_EQ( copy.getLocalView(), this->distributedArray.getLocalView() );
}

TYPED_TEST( DistributedArrayTest, comparisonOperators )
{
   using DistributedArrayType = typename TestFixture::DistributedArrayType;
   using IndexType = typename TestFixture::IndexType;

   const auto localRange = this->distributedArray.getLocalRange();
   DistributedArrayType& u = this->distributedArray;
   DistributedArrayType v, w;
   v.setLike( u );
   w.setLike( u );

   for( int i = 0; i < localRange.getSize(); i ++ ) {
      const IndexType gi = localRange.getGlobalIndex( i );
      u.setElement( gi, i );
      v.setElement( gi, i );
      w.setElement( gi, 2 * i );
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

   v.reset();
   EXPECT_FALSE( u == v );
   u.reset();
   EXPECT_TRUE( u == v );
}

TYPED_TEST( DistributedArrayTest, empty )
{
   EXPECT_GT( this->distributedArray.getSize(), 0 );
   EXPECT_FALSE( this->distributedArray.empty() );
   this->distributedArray.reset();
   EXPECT_EQ( this->distributedArray.getSize(), 0 );
   EXPECT_TRUE( this->distributedArray.empty() );
}

#endif  // HAVE_GTEST

#include "../main_mpi.h"
