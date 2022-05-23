#ifdef HAVE_GTEST
#include <gtest/gtest.h>

#include "CustomScalar.h"

using scalar = TNL::CustomScalar< int >;

TEST( CustomScalarTest, comparison )
{
   scalar a = 1;
   EXPECT_EQ( a, 1 );
   EXPECT_EQ( 1, a );
   EXPECT_NE( a, 2 );
   EXPECT_NE( 2, a );
   EXPECT_LE( a, 1 );
   EXPECT_LE( 1, a );
   EXPECT_GE( a, 1 );
   EXPECT_GE( 1, a );
   EXPECT_LT( a, 2 );
   EXPECT_LT( 0, a );
   EXPECT_GT( a, 0 );
   EXPECT_GT( 2, a );

   scalar b = 1.0;
   EXPECT_EQ( b, 1.0 );
   EXPECT_EQ( 1.0, b );
   EXPECT_NE( b, 2.0 );
   EXPECT_NE( 2.0, b );
   EXPECT_LE( b, 1.0 );
   EXPECT_LE( 1.0, b );
   EXPECT_GE( b, 1.0 );
   EXPECT_GE( 1.0, b );
   EXPECT_LT( b, 2.0 );
   EXPECT_LT( 0.0, b );
   EXPECT_GT( b, 0.0 );
   EXPECT_GT( 2.0, b );
}

// TODO: test the other operators

#endif

#include "main.h"
