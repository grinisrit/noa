#include <cstdlib>
#include <TNL/Devices/Host.h>
#include <TNL/Pointers/UniquePointer.h>
#include <TNL/Containers/StaticArray.h>

#ifdef HAVE_GTEST
#include <gtest/gtest.h>
#endif

using namespace TNL;
using namespace TNL::Pointers;

#ifdef HAVE_GTEST
TEST( UniquePointerTest, ConstructorTest )
{
   typedef TNL::Containers::StaticArray< 2, int  > TestType;
   UniquePointer< TestType, Devices::Host > ptr1;

   ptr1->x() = 0;
   ptr1->y() = 0;
   ASSERT_EQ( ptr1->x(), 0 );
   ASSERT_EQ( ptr1->y(), 0 );

   UniquePointer< TestType, Devices::Host > ptr2( 1, 2 );
   ASSERT_EQ( ptr2->x(), 1 );
   ASSERT_EQ( ptr2->y(), 2 );

   ptr1 = ptr2;
   ASSERT_EQ( ptr1->x(), 1 );
   ASSERT_EQ( ptr1->y(), 2 );
}
#endif

#include "../main.h"
