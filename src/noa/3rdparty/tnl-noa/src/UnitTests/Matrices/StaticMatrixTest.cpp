#ifdef HAVE_GTEST
#include "gtest/gtest.h"

#include <TNL/Matrices/StaticMatrix.h>

using namespace TNL::Containers;
using namespace TNL::Matrices;

TEST( StaticNDArrayTest, 3x4_row_major )
{
   constexpr int I = 3, J = 4;
   StaticMatrix< int, I, J > M;
   StaticVector< I, int > a, row_sums;
   StaticVector< J, int > b;

   row_sums.setValue( 0 );
   a.setValue( 0 );
   b.setValue( 1 );

   int v = 0;
   for( int i = 0; i < I; i++ )
   for( int j = 0; j < J; j++ )
   {
      M( i, j ) = v;
      row_sums[ i ] += v;
      v++;
   }

   a = M * b;

   EXPECT_EQ( a, row_sums );
}

TEST( StaticNDArrayTest, 4x3_row_major )
{
   constexpr int I = 4, J = 3;
   StaticMatrix< int, I, J > M;
   StaticVector< I, int > a, row_sums;
   StaticVector< J, int > b;

   row_sums.setValue( 0 );
   a.setValue( 0 );
   b.setValue( 1 );

   int v = 0;
   for( int i = 0; i < I; i++ )
   for( int j = 0; j < J; j++ )
   {
      M( i, j ) = v;
      row_sums[ i ] += v;
      v++;
   }

   a = M * b;

   EXPECT_EQ( a, row_sums );
}

TEST( StaticNDArrayTest, 3x4_column_major )
{
   constexpr int I = 3, J = 4;
   using Permutation = std::index_sequence< 1, 0 >;
   StaticMatrix< int, I, J, Permutation > M;
   StaticVector< I, int > a, row_sums;
   StaticVector< J, int > b;

   row_sums.setValue( 0 );
   a.setValue( 0 );
   b.setValue( 1 );

   int v = 0;
   for( int i = 0; i < I; i++ )
   for( int j = 0; j < J; j++ )
   {
      M( i, j ) = v;
      row_sums[ i ] += v;
      v++;
   }

   a = M * b;

   EXPECT_EQ( a, row_sums );
}

TEST( StaticNDArrayTest, 4x3_column_major )
{
   constexpr int I = 4, J = 3;
   using Permutation = std::index_sequence< 1, 0 >;
   StaticMatrix< int, I, J, Permutation > M;
   StaticVector< I, int > a, row_sums;
   StaticVector< J, int > b;

   row_sums.setValue( 0 );
   a.setValue( 0 );
   b.setValue( 1 );

   int v = 0;
   for( int i = 0; i < I; i++ )
   for( int j = 0; j < J; j++ )
   {
      M( i, j ) = v;
      row_sums[ i ] += v;
      v++;
   }

   a = M * b;

   EXPECT_EQ( a, row_sums );
}
#endif // HAVE_GTEST


#include "../main.h"
