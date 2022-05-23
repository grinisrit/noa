#include <cstdlib>
#include <TNL/Devices/Host.h>
#include <TNL/Pointers/SharedPointer.h>
#include <TNL/Containers/StaticArray.h>

#ifdef HAVE_GTEST
#include <gtest/gtest.h>
#endif

using namespace TNL;

#ifdef HAVE_GTEST
TEST( SharedPointerHostTest, ConstructorTest )
{
   typedef TNL::Containers::StaticArray< 2, int  > TestType;
   Pointers::SharedPointer< TestType, Devices::Host > ptr1;

   ptr1->x() = 0;
   ptr1->y() = 0;
   ASSERT_EQ( ptr1->x(), 0 );
   ASSERT_EQ( ptr1->y(), 0 );

   Pointers::SharedPointer< TestType, Devices::Host > ptr2( 1, 2 );
   ASSERT_EQ( ptr2->x(), 1 );
   ASSERT_EQ( ptr2->y(), 2 );

   ptr1 = ptr2;
   ASSERT_EQ( ptr1->x(), 1 );
   ASSERT_EQ( ptr1->y(), 2 );
};

TEST( SharedPointerCudaTest, nullptrAssignement )
{
   using TestType = Pointers::SharedPointer< double, Devices::Host >;
   TestType p1( 5 ), p2( nullptr );

   // This should not crash
   p1 = p2;

   ASSERT_FALSE( p1 );
   ASSERT_FALSE( p2 );
}

TEST( SharedPointerCudaTest, swap )
{
   using TestType = Pointers::SharedPointer< double, Devices::Host >;
   TestType p1( 1 ), p2( 2 );

   p1.swap( p2 );

   ASSERT_EQ( *p1, 2 );
   ASSERT_EQ( *p2, 1 );
}
#endif

#include "../main.h"
