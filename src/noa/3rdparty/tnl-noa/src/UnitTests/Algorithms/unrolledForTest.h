#pragma once

#include <array>

#include <TNL/Containers/Array.h>
#include <TNL/Algorithms/ParallelFor.h>
#include <TNL/Algorithms/unrolledFor.h>

#ifdef HAVE_GTEST
#include <gtest/gtest.h>
#endif

using namespace TNL;
using namespace TNL::Algorithms;

#ifdef HAVE_GTEST
template< int N >
void test_host()
{
   std::array< int, N > a;
   a.fill( 0 );

   unrolledFor< int, 0, N >(
      [&a] ( auto i ) {
         a[ i ] += 1;
      }
   );

   std::array< int, N > expected;
   expected.fill( 1 );
   EXPECT_EQ( a, expected );
}

TEST( unrolledForTest, host_size_8 )
{
   test_host<8>();
}

TEST( unrolledForTest, host_size_97 )
{
   test_host<97>();
}

TEST( unrolledForTest, host_size_5000 )
{
   test_host<5000>();
}

TEST( unrolledForTest, host_empty )
{
   bool called = false;

   unrolledFor< int, 0, 0 >(
      [&called] ( auto i ) {
         called = true;
      }
   );
   EXPECT_FALSE( called );

   unrolledFor< int, 0, -1 >(
      [&called] ( auto i ) {
         called = true;
      }
   );
   EXPECT_FALSE( called );
}

#ifdef HAVE_CUDA
template< int N >
void test_cuda()
{
   using Array = Containers::Array< int, Devices::Cuda >;
   using ArrayHost = Containers::Array< int, Devices::Host >;
   Array a( N );
   a.setValue( 0 );
   auto view = a.getView();

   auto kernel = [=] __cuda_callable__ (int j) mutable
   {
      unrolledFor< int, 0, N >(
         [&view] ( auto i ) {
            view[ i ] += 1;
         }
      );
   };
   ParallelFor< Devices::Cuda >::exec( 0, 1, kernel );

   ArrayHost expected;
   expected.setSize( N );
   expected.setValue( 1 );

   ArrayHost ah;
   ah = a;
   EXPECT_EQ( ah, expected );
}

TEST( unrolledForTest, cuda_size_8 )
{
   test_cuda<8>();
}

TEST( unrolledForTest, cuda_size_97 )
{
   test_cuda<97>();
}

TEST( unrolledForTest, cuda_size_5000 )
{
   test_cuda<5000>();
}
#endif
#endif

#include "../main.h"
