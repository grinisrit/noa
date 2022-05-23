#ifdef HAVE_GTEST
#include <gtest/gtest.h>

#include <TNL/Containers/Array.h>
#include <TNL/String.h>
#include <TNL/MPI/Utils.h>

#include "../Containers/VectorHelperFunctions.h"

using namespace TNL;
using namespace TNL::Containers;
using namespace TNL::MPI;

template< typename T1, typename T2 >
struct Pair
{
   using Left = T1;
   using Right = T2;
};

template< typename Pair >
class ArrayCommunicationTest
: public ::testing::Test
{
protected:
   using SrcArrayType = typename Pair::Left;
   using DestArrayType = typename Pair::Right;
   using ValueType = std::decay_t< decltype(SrcArrayType{}[0]) >;

   const MPI_Comm communicator = MPI_COMM_WORLD;

   // source array or view
   SrcArrayType srcArray;
   // source array
   DestArrayType _src;
   // destination array
   DestArrayType destArray;

   const int rank = GetRank(communicator);
   const int nproc = GetSize(communicator);

   ArrayCommunicationTest()
   {
      _src.setSize( rank );
      for( int i = 0; i < rank; i++ )
         _src[ i ] = ValueType( rank );
      bindOrAssign( srcArray, _src );
      EXPECT_EQ( srcArray.getSize(), rank );
   }
};

// types for which ArrayCommunicationTest is instantiated
using ArrayTypes = ::testing::Types<
   Pair< Array< int >, Array< int > >,
   Pair< ArrayView< int >, Array< int > >,
   Pair< String, String >
>;

TYPED_TEST_SUITE( ArrayCommunicationTest, ArrayTypes );


TYPED_TEST( ArrayCommunicationTest, send_recv )
{
   using DestArrayType = typename TestFixture::DestArrayType;
   using ValueType = typename TestFixture::ValueType;

   const int src  = (this->rank - 1 + this->nproc) % this->nproc;
   const int dest = (this->rank + 1 + this->nproc) % this->nproc;

   // NOTE: condition avoids a deadlock due to blocking communication
   if( this->rank % 2 ) {
      send( this->srcArray, dest, 0, this->communicator );
      recv( this->destArray, src, 0, this->communicator );
   }
   else {
      recv( this->destArray, src, 0, this->communicator );
      send( this->srcArray, dest, 0, this->communicator );
   }

   EXPECT_EQ( this->destArray.getSize(), src );
   DestArrayType expected;
   expected.setSize( src );
   for( int i = 0; i < src; i++ )
      expected[ i ] = ValueType( src );
   EXPECT_EQ( this->destArray, expected );
}

TYPED_TEST( ArrayCommunicationTest, sendrecv )
{
   using DestArrayType = typename TestFixture::DestArrayType;
   using ValueType = typename TestFixture::ValueType;

   const int src  = (this->rank - 1 + this->nproc) % this->nproc;
   const int dest = (this->rank + 1 + this->nproc) % this->nproc;

   sendrecv( this->srcArray, dest, 0, this->destArray, src, 0, this->communicator );

   EXPECT_EQ( this->destArray.getSize(), src );
   DestArrayType expected;
   expected.setSize( src );
   for( int i = 0; i < src; i++ )
      expected[ i ] = ValueType( src );
   EXPECT_EQ( this->destArray, expected );
}

TYPED_TEST( ArrayCommunicationTest, bcast )
{
   using DestArrayType = typename TestFixture::DestArrayType;
   using ValueType = typename TestFixture::ValueType;

   for( int root = 0; root < this->nproc; root++ ) {
      // reset the array on rank
      this->destArray.setSize( this->rank );
      for( int i = 0; i < this->rank; i++ )
         this->destArray[ i ] = ValueType( this->rank );

      // broadcast the dest array (bcast does not make sense for views)
      bcast( this->destArray, root, this->communicator );

      EXPECT_EQ( this->destArray.getSize(), root );
      DestArrayType expected;
      expected.setSize( root );
      for( int i = 0; i < root; i++ )
         expected[ i ] = ValueType( root );
      EXPECT_EQ( this->destArray, expected );

      Barrier( this->communicator );
   }
}

#endif  // HAVE_GTEST

#include "../main_mpi.h"
