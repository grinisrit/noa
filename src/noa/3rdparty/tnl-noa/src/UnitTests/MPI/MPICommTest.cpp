#ifdef HAVE_GTEST
#include <gtest/gtest.h>

#include <TNL/MPI/Comm.h>

using namespace TNL;
using namespace TNL::MPI;

TEST( CommTest, COMM_WORLD )
{
   const Comm c = MPI_COMM_WORLD;
   EXPECT_EQ( c.rank(), GetRank( MPI_COMM_WORLD ) );
   EXPECT_EQ( c.size(), GetSize( MPI_COMM_WORLD ) );
   EXPECT_EQ( MPI_Comm( c ), MPI_COMM_WORLD );
   EXPECT_EQ( c.compare( MPI_COMM_WORLD ), MPI_IDENT );
}

TEST( CommTest, duplicate_static )
{
   const Comm c = Comm::duplicate( MPI_COMM_WORLD );
   EXPECT_EQ( c.rank(), GetRank( MPI_COMM_WORLD ) );
   EXPECT_EQ( c.size(), GetSize( MPI_COMM_WORLD ) );
   EXPECT_NE( MPI_Comm( c ), MPI_COMM_WORLD );
   EXPECT_EQ( c.compare( MPI_COMM_WORLD ), MPI_CONGRUENT );
}

TEST( CommTest, duplicate )
{
   const Comm c = Comm( MPI_COMM_WORLD ).duplicate();
   EXPECT_EQ( c.rank(), GetRank( MPI_COMM_WORLD ) );
   EXPECT_EQ( c.size(), GetSize( MPI_COMM_WORLD ) );
   EXPECT_NE( MPI_Comm( c ), MPI_COMM_WORLD );
   EXPECT_EQ( c.compare( MPI_COMM_WORLD ), MPI_CONGRUENT );
}

TEST( CommTest, split_static_odd_even )
{
   const int rank = GetRank( MPI_COMM_WORLD );
   const int size = GetSize( MPI_COMM_WORLD );
   // split into two groups: odd and even based on the original rank
   const Comm c = Comm::split( MPI_COMM_WORLD, rank % 2, rank );
   const int my_size = ( size % 2 == 0 || rank % 2 == 1 ) ? size / 2 : size / 2 + 1;
   EXPECT_EQ( c.rank(), rank / 2 );
   EXPECT_EQ( c.size(), my_size );
   EXPECT_NE( MPI_Comm( c ), MPI_COMM_WORLD );
   if( size == 1 ) {
      EXPECT_EQ( c.compare( MPI_COMM_WORLD ), MPI_CONGRUENT );
   }
   else {
      EXPECT_EQ( c.compare( MPI_COMM_WORLD ), MPI_UNEQUAL );
   }
}

TEST( CommTest, split_odd_even )
{
   const int rank = GetRank( MPI_COMM_WORLD );
   const int size = GetSize( MPI_COMM_WORLD );
   // split into two groups: odd and even based on the original rank
   const Comm c = Comm( MPI_COMM_WORLD ).split( rank % 2, rank );
   const int my_size = ( size % 2 == 0 || rank % 2 == 1 ) ? size / 2 : size / 2 + 1;
   EXPECT_EQ( c.rank(), rank / 2 );
   EXPECT_EQ( c.size(), my_size );
   EXPECT_NE( MPI_Comm( c ), MPI_COMM_WORLD );
   if( size == 1 ) {
      EXPECT_EQ( c.compare( MPI_COMM_WORLD ), MPI_CONGRUENT );
   }
   else {
      EXPECT_EQ( c.compare( MPI_COMM_WORLD ), MPI_UNEQUAL );
   }
}

TEST( CommTest, split_static_renumber )
{
   const int rank = GetRank( MPI_COMM_WORLD );
   const int size = GetSize( MPI_COMM_WORLD );
   // same group, but different ranks
   const Comm c = Comm::split( MPI_COMM_WORLD, 0, size - 1 - rank );
   EXPECT_EQ( c.rank(), size - 1 - rank );
   EXPECT_EQ( c.size(), size );
   EXPECT_NE( MPI_Comm( c ), MPI_COMM_WORLD );
   if( size == 1 ) {
      EXPECT_EQ( c.compare( MPI_COMM_WORLD ), MPI_CONGRUENT );
   }
   else {
      EXPECT_EQ( c.compare( MPI_COMM_WORLD ), MPI_SIMILAR );
   }
}

TEST( CommTest, split_renumber )
{
   const int rank = GetRank( MPI_COMM_WORLD );
   const int size = GetSize( MPI_COMM_WORLD );
   // same group, but different ranks
   const Comm c = Comm( MPI_COMM_WORLD ).split( 0, size - 1 - rank );
   EXPECT_EQ( c.rank(), size - 1 - rank );
   EXPECT_EQ( c.size(), size );
   EXPECT_NE( MPI_Comm( c ), MPI_COMM_WORLD );
   if( size == 1 ) {
      EXPECT_EQ( c.compare( MPI_COMM_WORLD ), MPI_CONGRUENT );
   }
   else {
      EXPECT_EQ( c.compare( MPI_COMM_WORLD ), MPI_SIMILAR );
   }
}

#ifdef HAVE_MPI
TEST( CommTest, split_type_static )
{
   // tests are run on a single node, so the resulting communicator is congruent to MPI_COMM_WORLD
   const int rank = GetRank( MPI_COMM_WORLD );
   const Comm local_comm = Comm::split_type( MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL );
   EXPECT_EQ( local_comm.compare( MPI_COMM_WORLD ), MPI_CONGRUENT );
}

TEST( CommTest, split_type )
{
   // tests are run on a single node, so the resulting communicator is congruent to MPI_COMM_WORLD
   const int rank = GetRank( MPI_COMM_WORLD );
   const Comm local_comm = Comm( MPI_COMM_WORLD ).split_type( MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL );
   EXPECT_EQ( local_comm.compare( MPI_COMM_WORLD ), MPI_CONGRUENT );
}
#endif

#endif  // HAVE_GTEST

#include "../main_mpi.h"
