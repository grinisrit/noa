#pragma once

#ifdef HAVE_GTEST
#include <TNL/Allocators/Host.h>
#include <TNL/Allocators/Cuda.h>
#include <TNL/Allocators/CudaHost.h>
#include <TNL/Allocators/CudaManaged.h>
#include <TNL/Algorithms/MemoryOperations.h>

#include "gtest/gtest.h"

using namespace TNL;

constexpr int ARRAY_TEST_SIZE = 5000;

// test fixture for typed tests
template< typename Value >
class AllocatorsTest : public ::testing::Test
{
protected:
   using ValueType = Value;
};

// types for which ArrayTest is instantiated
using ValueTypes = ::testing::Types< short int, int, long, float, double >;

TYPED_TEST_SUITE( AllocatorsTest, ValueTypes );

TYPED_TEST( AllocatorsTest, Host )
{
   using ValueType = typename TestFixture::ValueType;
   using Allocator = Allocators::Host< ValueType >;

   Allocator allocator;
   ValueType* data = allocator.allocate( ARRAY_TEST_SIZE );
   ASSERT_NE( data, nullptr );

   // do something useful with the data
   for (int i = 0; i < ARRAY_TEST_SIZE; i++) {
      data[i] = 0;
      EXPECT_EQ(data[i], 0);
   }

   allocator.deallocate( data, ARRAY_TEST_SIZE );
}

#ifdef __CUDACC__
TYPED_TEST( AllocatorsTest, CudaHost )
{
   using ValueType = typename TestFixture::ValueType;
   using Allocator = Allocators::CudaHost< ValueType >;

   Allocator allocator;
   ValueType* data = allocator.allocate( ARRAY_TEST_SIZE );
   ASSERT_NE( data, nullptr );

   // do something useful with the data
   for (int i = 0; i < ARRAY_TEST_SIZE; i++) {
      data[i] = 0;
      EXPECT_EQ(data[i], 0);
   }

   allocator.deallocate( data, ARRAY_TEST_SIZE );
}

TYPED_TEST( AllocatorsTest, CudaManaged )
{
   using ValueType = typename TestFixture::ValueType;
   using Allocator = Allocators::CudaManaged< ValueType >;

   Allocator allocator;
   ValueType* data = allocator.allocate( ARRAY_TEST_SIZE );
   ASSERT_NE( data, nullptr );

   // set data on the device
   Algorithms::MemoryOperations< Devices::Cuda >::set( data, (ValueType) 0, ARRAY_TEST_SIZE );
   ASSERT_NO_THROW( TNL_CHECK_CUDA_DEVICE );

   // check values on the host
   for (int i = 0; i < ARRAY_TEST_SIZE; i++)
      EXPECT_EQ(data[i], 0);

   allocator.deallocate( data, ARRAY_TEST_SIZE );
}

TYPED_TEST( AllocatorsTest, Cuda )
{
   using ValueType = typename TestFixture::ValueType;
   using Allocator = Allocators::CudaHost< ValueType >;

   Allocator allocator;
   ValueType* data = allocator.allocate( ARRAY_TEST_SIZE );
   ASSERT_NE( data, nullptr );

   // set data on the device
   Algorithms::MemoryOperations< Devices::Cuda >::set( data, (ValueType) 0, ARRAY_TEST_SIZE );
   ASSERT_NO_THROW( TNL_CHECK_CUDA_DEVICE );

   allocator.deallocate( data, ARRAY_TEST_SIZE );
}
#endif // __CUDACC__
#endif // HAVE_GTEST


#include "main.h"
