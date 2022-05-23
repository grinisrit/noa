#ifdef NDEBUG
   #undef NDEBUG
#endif

#include <TNL/Assert.h>

#ifdef HAVE_GTEST
#include <gtest/gtest.h>

using namespace TNL;

TEST( AssertTest, basicTest )
{
   const bool tr = true;
   const bool fa = false;
   const int two = 2;
   const int ten = 10;

   // true statements:
   EXPECT_NO_THROW( TNL_ASSERT_TRUE( true, "true is true" ); );
   EXPECT_NO_THROW( TNL_ASSERT_TRUE( tr, "true is true" ); );
   EXPECT_NO_THROW( TNL_ASSERT_FALSE( false, "false is false" ); );
   EXPECT_NO_THROW( TNL_ASSERT_FALSE( fa, "false is false" ); );
   EXPECT_NO_THROW( TNL_ASSERT_EQ( two, 2, "two is 2" ); );
   EXPECT_NO_THROW( TNL_ASSERT_NE( ten, 2, "ten is not 2" ); );
   EXPECT_NO_THROW( TNL_ASSERT_LT( two, 10, "two < 10" ); );
   EXPECT_NO_THROW( TNL_ASSERT_LE( two, 10, "two <= 10" ); );
   EXPECT_NO_THROW( TNL_ASSERT_LE( two, 2, "two <= 2" ); );
   EXPECT_NO_THROW( TNL_ASSERT_GT( ten, 2, "ten > 2" ); );
   EXPECT_NO_THROW( TNL_ASSERT_GE( ten, 10, "ten >= 10" ); );
   EXPECT_NO_THROW( TNL_ASSERT_GE( ten, 2, "ten >= 2" ); );

   // errors:
   EXPECT_ANY_THROW( TNL_ASSERT_TRUE( false, "false is true" ); );
   EXPECT_ANY_THROW( TNL_ASSERT_TRUE( fa, "false is true" ); );
   EXPECT_ANY_THROW( TNL_ASSERT_FALSE( true, "true is false" ); );
   EXPECT_ANY_THROW( TNL_ASSERT_FALSE( tr, "true is false" ); );
   EXPECT_ANY_THROW( TNL_ASSERT_NE( two, 2, "two != 2" ); );
   EXPECT_ANY_THROW( TNL_ASSERT_EQ( ten, 2, "ten == 2" ); );
   EXPECT_ANY_THROW( TNL_ASSERT_GE( two, 10, "two >= 10" ); );
   EXPECT_ANY_THROW( TNL_ASSERT_GT( two, 10, "two > 10" ); );
   EXPECT_ANY_THROW( TNL_ASSERT_GT( two, 2, "two > 2" ); );
   EXPECT_ANY_THROW( TNL_ASSERT_LE( ten, 2, "ten <= 2" ); );
   EXPECT_ANY_THROW( TNL_ASSERT_LT( ten, 10, "ten < 10" ); );
   EXPECT_ANY_THROW( TNL_ASSERT_LT( ten, 2, "ten < 2" ); );

   // pointers
   const double* data_null = nullptr;
   const double** data_full = &data_null;

   // true statements:
   EXPECT_NO_THROW( TNL_ASSERT_FALSE( data_null, "nullptr is false" ); );
   EXPECT_NO_THROW( TNL_ASSERT_TRUE( data_full, "non-nullptr is true" ); );

   // errors
   EXPECT_ANY_THROW( TNL_ASSERT_TRUE( data_null, "nullptr is true" ); );
   EXPECT_ANY_THROW( TNL_ASSERT_FALSE( data_full, "non-nullptr is false" ); );
}
#endif

#include "main.h"
