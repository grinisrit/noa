#ifdef HAVE_GTEST
#include <gtest/gtest.h>
#endif

#include <TNL/Math.h>

#ifdef HAVE_GTEST
TEST( MathTest, variadic_min )
{
   using TNL::min;

   EXPECT_EQ( min(1, 2, 3, 4), 1 );
   EXPECT_EQ( min(1, 2, 4, 3), 1 );
   EXPECT_EQ( min(1, 3, 2, 4), 1 );
   EXPECT_EQ( min(1, 3, 4, 2), 1 );
   EXPECT_EQ( min(1, 4, 2, 3), 1 );
   EXPECT_EQ( min(1, 4, 3, 2), 1 );
   EXPECT_EQ( min(2, 1, 3, 4), 1 );
   EXPECT_EQ( min(2, 1, 4, 3), 1 );
   EXPECT_EQ( min(2, 3, 1, 4), 1 );
   EXPECT_EQ( min(2, 3, 4, 1), 1 );
   EXPECT_EQ( min(2, 4, 1, 3), 1 );
   EXPECT_EQ( min(2, 4, 3, 1), 1 );
   EXPECT_EQ( min(3, 1, 2, 4), 1 );
   EXPECT_EQ( min(3, 1, 4, 2), 1 );
   EXPECT_EQ( min(3, 2, 1, 4), 1 );
   EXPECT_EQ( min(3, 2, 4, 1), 1 );
   EXPECT_EQ( min(3, 4, 1, 2), 1 );
   EXPECT_EQ( min(3, 4, 2, 1), 1 );
   EXPECT_EQ( min(4, 1, 2, 3), 1 );
   EXPECT_EQ( min(4, 1, 3, 2), 1 );
   EXPECT_EQ( min(4, 2, 1, 3), 1 );
   EXPECT_EQ( min(4, 2, 3, 1), 1 );
   EXPECT_EQ( min(4, 3, 1, 2), 1 );
   EXPECT_EQ( min(4, 3, 2, 1), 1 );
}

TEST( MathTest, variadic_max )
{
   using TNL::max;

   EXPECT_EQ( max(1, 2, 3, 4), 4 );
   EXPECT_EQ( max(1, 2, 4, 3), 4 );
   EXPECT_EQ( max(1, 3, 2, 4), 4 );
   EXPECT_EQ( max(1, 3, 4, 2), 4 );
   EXPECT_EQ( max(1, 4, 2, 3), 4 );
   EXPECT_EQ( max(1, 4, 3, 2), 4 );
   EXPECT_EQ( max(2, 1, 3, 4), 4 );
   EXPECT_EQ( max(2, 1, 4, 3), 4 );
   EXPECT_EQ( max(2, 3, 1, 4), 4 );
   EXPECT_EQ( max(2, 3, 4, 1), 4 );
   EXPECT_EQ( max(2, 4, 1, 3), 4 );
   EXPECT_EQ( max(2, 4, 3, 1), 4 );
   EXPECT_EQ( max(3, 1, 2, 4), 4 );
   EXPECT_EQ( max(3, 1, 4, 2), 4 );
   EXPECT_EQ( max(3, 2, 1, 4), 4 );
   EXPECT_EQ( max(3, 2, 4, 1), 4 );
   EXPECT_EQ( max(3, 4, 1, 2), 4 );
   EXPECT_EQ( max(3, 4, 2, 1), 4 );
   EXPECT_EQ( max(4, 1, 2, 3), 4 );
   EXPECT_EQ( max(4, 1, 3, 2), 4 );
   EXPECT_EQ( max(4, 2, 1, 3), 4 );
   EXPECT_EQ( max(4, 2, 3, 1), 4 );
   EXPECT_EQ( max(4, 3, 1, 2), 4 );
   EXPECT_EQ( max(4, 3, 2, 1), 4 );
}
#endif

#include "main.h"
