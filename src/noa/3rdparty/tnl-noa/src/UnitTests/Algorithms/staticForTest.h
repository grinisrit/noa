#pragma once

#include <array>

#include <TNL/Containers/Array.h>
#include <TNL/Algorithms/ParallelFor.h>
#include <TNL/Algorithms/staticFor.h>

#ifdef HAVE_GTEST
#include <gtest/gtest.h>
#endif

using namespace TNL;
using namespace TNL::Algorithms;

#ifdef HAVE_GTEST
TEST( staticForTest, host_dynamic )
{
   constexpr int N = 5;
   std::array< int, N > a;
   a.fill( 0 );

   staticFor< int, 0, N >(
      [&a] ( auto i ) {
         a[ i ] += 1;
      }
   );

   std::array< int, N > expected;
   expected.fill( 1 );
   EXPECT_EQ( a, expected );
}

TEST( staticForTest, host_static )
{
   constexpr int N = 5;
   std::array< int, N > a;
   a.fill( 0 );

   staticFor< int, 0, N >(
      [&a] ( auto i ) {
         std::get< i >( a ) += 1;
      }
   );

   std::array< int, N > expected;
   expected.fill( 1 );
   EXPECT_EQ( a, expected );
}

TEST( staticForTest, host_empty )
{
   bool called = false;

   staticFor< int, 0, 0 >(
      [&called] ( auto i ) {
         called = true;
      }
   );
   EXPECT_FALSE( called );

   staticFor< int, 0, -1 >(
      [&called] ( auto i ) {
         called = true;
      }
   );
   EXPECT_FALSE( called );
}

#ifdef HAVE_CUDA
// nvcc does not allow __cuda_callable__ lambdas inside private regions
void test_cuda_dynamic()
{
   using Array = Containers::Array< int, Devices::Cuda >;
   using ArrayHost = Containers::Array< int, Devices::Host >;
   constexpr int N = 5;
   Array a( N );
   a.setValue( 0 );
   auto view = a.getView();

   auto kernel = [=] __cuda_callable__ (int j) mutable
   {
      staticFor< int, 0, N >(
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

TEST( staticForTest, cuda_dynamic )
{
   test_cuda_dynamic();
}

template< int i, typename View >
__cuda_callable__
void static_helper( View& view )
{
   view[ i ] += 1;
}

// nvcc does not allow __cuda_callable__ lambdas inside private regions
void test_cuda_static()
{
   using Array = Containers::Array< int, Devices::Cuda >;
   using ArrayHost = Containers::Array< int, Devices::Host >;
   constexpr int N = 5;
   Array a( N );
   a.setValue( 0 );
   auto view = a.getView();

   auto kernel = [=] __cuda_callable__ (int j) mutable
   {
      staticFor< int, 0, N >(
         [&view] ( auto i ) {
            static_helper< i >( view );
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

TEST( staticForTest, cuda_static )
{
   test_cuda_static();
}
#endif
#endif

#include "../main.h"
