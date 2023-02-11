#pragma once

#ifdef HAVE_GTEST
#include <TNL/Allocators/Host.h>
#include <TNL/Allocators/Cuda.h>
#include <TNL/Algorithms/MemoryOperations.h>
#include <TNL/Algorithms/MultiDeviceMemoryOperations.h>

#include "gtest/gtest.h"

using namespace TNL;
using namespace TNL::Algorithms;

constexpr int ARRAY_TEST_SIZE = 5000;

// test fixture for typed tests
template< typename Value >
class MemoryOperationsTest : public ::testing::Test
{
protected:
   using ValueType = Value;
};

// types for which ArrayTest is instantiated
using ValueTypes = ::testing::Types< short int, int, long, float, double >;

TYPED_TEST_SUITE( MemoryOperationsTest, ValueTypes );

TYPED_TEST( MemoryOperationsTest, allocateMemory_host )
{
   using ValueType = typename TestFixture::ValueType;
   using Allocator = Allocators::Host< ValueType >;

   Allocator allocator;
   ValueType* data = allocator.allocate( ARRAY_TEST_SIZE );
   ASSERT_NE( data, nullptr );

   allocator.deallocate( data, ARRAY_TEST_SIZE );
}

TYPED_TEST( MemoryOperationsTest, setElement_host )
{
   using ValueType = typename TestFixture::ValueType;
   using Allocator = Allocators::Host< ValueType >;

   Allocator allocator;
   ValueType* data = allocator.allocate( ARRAY_TEST_SIZE );
   for( int i = 0; i < ARRAY_TEST_SIZE; i++ ) {
      MemoryOperations< Devices::Host >::setElement( data + i, (ValueType) i );
      EXPECT_EQ( data[ i ], i );
      EXPECT_EQ( MemoryOperations< Devices::Host >::getElement( data + i ), i );
   }
   allocator.deallocate( data, ARRAY_TEST_SIZE );
}

TYPED_TEST( MemoryOperationsTest, set_host )
{
   using ValueType = typename TestFixture::ValueType;
   using Allocator = Allocators::Host< ValueType >;

   Allocator allocator;
   ValueType* data = allocator.allocate( ARRAY_TEST_SIZE );
   MemoryOperations< Devices::Host >::set( data, (ValueType) 13, ARRAY_TEST_SIZE );
   for( int i = 0; i < ARRAY_TEST_SIZE; i ++ )
      EXPECT_EQ( data[ i ], 13 );
   allocator.deallocate( data, ARRAY_TEST_SIZE );
}

TYPED_TEST( MemoryOperationsTest, copy_host )
{
   using ValueType = typename TestFixture::ValueType;
   using Allocator = Allocators::Host< ValueType >;

   Allocator allocator;
   ValueType* data1 = allocator.allocate( ARRAY_TEST_SIZE );
   ValueType* data2 = allocator.allocate( ARRAY_TEST_SIZE );
   MemoryOperations< Devices::Host >::set( data1, (ValueType) 13, ARRAY_TEST_SIZE );
   MemoryOperations< Devices::Host >::copy< ValueType, ValueType >( data2, data1, ARRAY_TEST_SIZE );
   for( int i = 0; i < ARRAY_TEST_SIZE; i ++ )
      EXPECT_EQ( data1[ i ], data2[ i ]);
   allocator.deallocate( data1, ARRAY_TEST_SIZE );
   allocator.deallocate( data2, ARRAY_TEST_SIZE );
}

TYPED_TEST( MemoryOperationsTest, copyWithConversion_host )
{
   using Allocator1 = Allocators::Host< int >;
   using Allocator2 = Allocators::Host< float >;

   Allocator1 allocator1;
   Allocator2 allocator2;
   int* data1 = allocator1.allocate( ARRAY_TEST_SIZE );
   float* data2 = allocator2.allocate( ARRAY_TEST_SIZE );
   MemoryOperations< Devices::Host >::set( data1, 13, ARRAY_TEST_SIZE );
   MemoryOperations< Devices::Host >::copy< float, int >( data2, data1, ARRAY_TEST_SIZE );
   for( int i = 0; i < ARRAY_TEST_SIZE; i ++ )
      EXPECT_EQ( data1[ i ], data2[ i ] );
   allocator1.deallocate( data1, ARRAY_TEST_SIZE );
   allocator2.deallocate( data2, ARRAY_TEST_SIZE );
}

TYPED_TEST( MemoryOperationsTest, compare_host )
{
   using ValueType = typename TestFixture::ValueType;
   using Allocator = Allocators::Host< ValueType >;

   Allocator allocator;
   ValueType* data1 = allocator.allocate( ARRAY_TEST_SIZE );
   ValueType* data2 = allocator.allocate( ARRAY_TEST_SIZE );
   MemoryOperations< Devices::Host >::set( data1, (ValueType) 7, ARRAY_TEST_SIZE );
   MemoryOperations< Devices::Host >::set( data2, (ValueType) 0, ARRAY_TEST_SIZE );
   EXPECT_FALSE( ( MemoryOperations< Devices::Host >::compare< ValueType, ValueType >( data1, data2, ARRAY_TEST_SIZE ) ) );
   MemoryOperations< Devices::Host >::set( data2, (ValueType) 7, ARRAY_TEST_SIZE );
   EXPECT_TRUE( ( MemoryOperations< Devices::Host >::compare< ValueType, ValueType >( data1, data2, ARRAY_TEST_SIZE ) ) );
   allocator.deallocate( data1, ARRAY_TEST_SIZE );
   allocator.deallocate( data2, ARRAY_TEST_SIZE );
}

TYPED_TEST( MemoryOperationsTest, compareWithConversion_host )
{
   using Allocator1 = Allocators::Host< int >;
   using Allocator2 = Allocators::Host< float >;

   Allocator1 allocator1;
   Allocator2 allocator2;
   int* data1 = allocator1.allocate( ARRAY_TEST_SIZE );
   float* data2 = allocator2.allocate( ARRAY_TEST_SIZE );
   MemoryOperations< Devices::Host >::set( data1, 7, ARRAY_TEST_SIZE );
   MemoryOperations< Devices::Host >::set( data2, (float) 0.0, ARRAY_TEST_SIZE );
   EXPECT_FALSE( ( MemoryOperations< Devices::Host >::compare< int, float >( data1, data2, ARRAY_TEST_SIZE ) ) );
   MemoryOperations< Devices::Host >::set( data2, (float) 7.0, ARRAY_TEST_SIZE );
   EXPECT_TRUE( ( MemoryOperations< Devices::Host >::compare< int, float >( data1, data2, ARRAY_TEST_SIZE ) ) );
   allocator1.deallocate( data1, ARRAY_TEST_SIZE );
   allocator2.deallocate( data2, ARRAY_TEST_SIZE );
}


#ifdef __CUDACC__
TYPED_TEST( MemoryOperationsTest, allocateMemory_cuda )
{
   using ValueType = typename TestFixture::ValueType;
   using Allocator = Allocators::Cuda< ValueType >;

   Allocator allocator;
   ValueType* data = allocator.allocate( ARRAY_TEST_SIZE );
   ASSERT_NO_THROW( TNL_CHECK_CUDA_DEVICE );
   ASSERT_NE( data, nullptr );

   allocator.deallocate( data, ARRAY_TEST_SIZE );
   ASSERT_NO_THROW( TNL_CHECK_CUDA_DEVICE );
}

TYPED_TEST( MemoryOperationsTest, setElement_cuda )
{
   using ValueType = typename TestFixture::ValueType;
   using Allocator = Allocators::Cuda< ValueType >;

   Allocator allocator;
   ValueType* data = allocator.allocate( ARRAY_TEST_SIZE );
   ASSERT_NO_THROW( TNL_CHECK_CUDA_DEVICE );

   for( int i = 0; i < ARRAY_TEST_SIZE; i++ )
      MemoryOperations< Devices::Cuda >::setElement( &data[ i ], (ValueType) i );

   for( int i = 0; i < ARRAY_TEST_SIZE; i++ )
   {
      ValueType d;
      ASSERT_EQ( cudaMemcpy( &d, &data[ i ], sizeof( ValueType ), cudaMemcpyDeviceToHost ), cudaSuccess );
      EXPECT_EQ( d, i );
      EXPECT_EQ( MemoryOperations< Devices::Cuda >::getElement( &data[ i ] ), i );
   }

   allocator.deallocate( data, ARRAY_TEST_SIZE );
   ASSERT_NO_THROW( TNL_CHECK_CUDA_DEVICE );
}

TYPED_TEST( MemoryOperationsTest, set_cuda )
{
   using ValueType = typename TestFixture::ValueType;
   using HostAllocator = Allocators::Host< ValueType >;
   using CudaAllocator = Allocators::Cuda< ValueType >;

   HostAllocator hostAllocator;
   CudaAllocator cudaAllocator;
   ValueType* hostData = hostAllocator.allocate( ARRAY_TEST_SIZE );
   ValueType* deviceData = cudaAllocator.allocate( ARRAY_TEST_SIZE );
   MemoryOperations< Devices::Host >::set( hostData, (ValueType) 0, ARRAY_TEST_SIZE );
   MemoryOperations< Devices::Cuda >::set( deviceData, (ValueType) 13, ARRAY_TEST_SIZE );
   ASSERT_NO_THROW( TNL_CHECK_CUDA_DEVICE );
   MultiDeviceMemoryOperations< Devices::Host, Devices::Cuda >::copy< ValueType, ValueType >( hostData, deviceData, ARRAY_TEST_SIZE );
   ASSERT_NO_THROW( TNL_CHECK_CUDA_DEVICE );
   for( int i = 0; i < ARRAY_TEST_SIZE; i++ )
      EXPECT_EQ( hostData[ i ], 13 );
   hostAllocator.deallocate( hostData, ARRAY_TEST_SIZE );
   cudaAllocator.deallocate( deviceData, ARRAY_TEST_SIZE );
}

TYPED_TEST( MemoryOperationsTest, copy_cuda )
{
   using ValueType = typename TestFixture::ValueType;
   using HostAllocator = Allocators::Host< ValueType >;
   using CudaAllocator = Allocators::Cuda< ValueType >;

   HostAllocator hostAllocator;
   CudaAllocator cudaAllocator;
   ValueType* hostData = hostAllocator.allocate( ARRAY_TEST_SIZE );
   ValueType* hostData2 = hostAllocator.allocate( ARRAY_TEST_SIZE );
   ValueType* deviceData = cudaAllocator.allocate( ARRAY_TEST_SIZE );
   ValueType* deviceData2 = cudaAllocator.allocate( ARRAY_TEST_SIZE );
   MemoryOperations< Devices::Host >::set( hostData, (ValueType) 13, ARRAY_TEST_SIZE );
   MultiDeviceMemoryOperations< Devices::Cuda, Devices::Host >::copy< ValueType >( deviceData, hostData, ARRAY_TEST_SIZE );
   MemoryOperations< Devices::Cuda >::copy< ValueType, ValueType >( deviceData2, deviceData, ARRAY_TEST_SIZE );
   MultiDeviceMemoryOperations< Devices::Host, Devices::Cuda >::copy< ValueType, ValueType >( hostData2, deviceData2, ARRAY_TEST_SIZE );
   EXPECT_TRUE( ( MemoryOperations< Devices::Host >::compare< ValueType, ValueType >( hostData, hostData2, ARRAY_TEST_SIZE) ) );
   hostAllocator.deallocate( hostData, ARRAY_TEST_SIZE );
   hostAllocator.deallocate( hostData2, ARRAY_TEST_SIZE );
   cudaAllocator.deallocate( deviceData, ARRAY_TEST_SIZE );
   cudaAllocator.deallocate( deviceData2, ARRAY_TEST_SIZE );
}

TYPED_TEST( MemoryOperationsTest, copyWithConversions_cuda )
{
   using HostAllocator1 = Allocators::Host< int >;
   using HostAllocator2 = Allocators::Host< double >;
   using CudaAllocator1 = Allocators::Cuda< long >;
   using CudaAllocator2 = Allocators::Cuda< float >;

   HostAllocator1 hostAllocator1;
   HostAllocator2 hostAllocator2;
   CudaAllocator1 cudaAllocator1;
   CudaAllocator2 cudaAllocator2;
   int* hostData = hostAllocator1.allocate( ARRAY_TEST_SIZE );
   double* hostData2 = hostAllocator2.allocate( ARRAY_TEST_SIZE );
   long* deviceData = cudaAllocator1.allocate( ARRAY_TEST_SIZE );
   float* deviceData2 = cudaAllocator2.allocate( ARRAY_TEST_SIZE );
   MemoryOperations< Devices::Host >::set( hostData, 13, ARRAY_TEST_SIZE );
   MultiDeviceMemoryOperations< Devices::Cuda, Devices::Host >::copy< long, int >( deviceData, hostData, ARRAY_TEST_SIZE );
   MemoryOperations< Devices::Cuda >::copy< float, long >( deviceData2, deviceData, ARRAY_TEST_SIZE );
   MultiDeviceMemoryOperations< Devices::Host, Devices::Cuda >::copy< double, float >( hostData2, deviceData2, ARRAY_TEST_SIZE );
   for( int i = 0; i < ARRAY_TEST_SIZE; i ++ )
      EXPECT_EQ( hostData[ i ], hostData2[ i ] );
   hostAllocator1.deallocate( hostData, ARRAY_TEST_SIZE );
   hostAllocator2.deallocate( hostData2, ARRAY_TEST_SIZE );
   cudaAllocator1.deallocate( deviceData, ARRAY_TEST_SIZE );
   cudaAllocator2.deallocate( deviceData2, ARRAY_TEST_SIZE );
}

TYPED_TEST( MemoryOperationsTest, compare_cuda )
{
   using ValueType = typename TestFixture::ValueType;
   using HostAllocator = Allocators::Host< ValueType >;
   using CudaAllocator = Allocators::Cuda< ValueType >;

   HostAllocator hostAllocator;
   CudaAllocator cudaAllocator;
   ValueType* hostData = hostAllocator.allocate( ARRAY_TEST_SIZE );
   ValueType* deviceData = cudaAllocator.allocate( ARRAY_TEST_SIZE );
   ValueType* deviceData2 = cudaAllocator.allocate( ARRAY_TEST_SIZE );

   MemoryOperations< Devices::Host >::set( hostData, (ValueType) 7, ARRAY_TEST_SIZE );
   MemoryOperations< Devices::Cuda >::set( deviceData, (ValueType) 8, ARRAY_TEST_SIZE );
   MemoryOperations< Devices::Cuda >::set( deviceData2, (ValueType) 9, ARRAY_TEST_SIZE );
   EXPECT_FALSE(( MultiDeviceMemoryOperations< Devices::Host, Devices::Cuda >::compare< ValueType, ValueType >( hostData, deviceData, ARRAY_TEST_SIZE ) ));
   EXPECT_FALSE(( MultiDeviceMemoryOperations< Devices::Cuda, Devices::Host >::compare< ValueType, ValueType >( deviceData, hostData, ARRAY_TEST_SIZE ) ));
   EXPECT_FALSE(( MemoryOperations< Devices::Cuda >::compare< ValueType, ValueType >( deviceData, deviceData2, ARRAY_TEST_SIZE ) ));

   MemoryOperations< Devices::Cuda >::set( deviceData, (ValueType) 7, ARRAY_TEST_SIZE );
   MemoryOperations< Devices::Cuda >::set( deviceData2, (ValueType) 7, ARRAY_TEST_SIZE );
   EXPECT_TRUE(( MultiDeviceMemoryOperations< Devices::Host, Devices::Cuda >::compare< ValueType, ValueType >( hostData, deviceData, ARRAY_TEST_SIZE ) ));
   EXPECT_TRUE(( MultiDeviceMemoryOperations< Devices::Cuda, Devices::Host >::compare< ValueType, ValueType >( deviceData, hostData, ARRAY_TEST_SIZE ) ));
   EXPECT_TRUE(( MemoryOperations< Devices::Cuda >::compare< ValueType, ValueType >( deviceData, deviceData2, ARRAY_TEST_SIZE ) ));

   hostAllocator.deallocate( hostData, ARRAY_TEST_SIZE );
   cudaAllocator.deallocate( deviceData, ARRAY_TEST_SIZE );
   cudaAllocator.deallocate( deviceData2, ARRAY_TEST_SIZE );
}

TYPED_TEST( MemoryOperationsTest, compareWithConversions_cuda )
{
   using HostAllocator = Allocators::Host< int >;
   using CudaAllocator1 = Allocators::Cuda< float >;
   using CudaAllocator2 = Allocators::Cuda< double >;

   HostAllocator hostAllocator;
   CudaAllocator1 cudaAllocator1;
   CudaAllocator2 cudaAllocator2;
   int* hostData = hostAllocator.allocate( ARRAY_TEST_SIZE );
   float* deviceData = cudaAllocator1.allocate( ARRAY_TEST_SIZE );
   double* deviceData2 = cudaAllocator2.allocate( ARRAY_TEST_SIZE );

   MemoryOperations< Devices::Host >::set( hostData, 7, ARRAY_TEST_SIZE );
   MemoryOperations< Devices::Cuda >::set( deviceData, (float) 8, ARRAY_TEST_SIZE );
   MemoryOperations< Devices::Cuda >::set( deviceData2, (double) 9, ARRAY_TEST_SIZE );
   EXPECT_FALSE(( MultiDeviceMemoryOperations< Devices::Host, Devices::Cuda >::compare< int, float >( hostData, deviceData, ARRAY_TEST_SIZE ) ));
   EXPECT_FALSE(( MultiDeviceMemoryOperations< Devices::Cuda, Devices::Host >::compare< float, int >( deviceData, hostData, ARRAY_TEST_SIZE ) ));
   EXPECT_FALSE(( MemoryOperations< Devices::Cuda >::compare< float, double >( deviceData, deviceData2, ARRAY_TEST_SIZE ) ));

   MemoryOperations< Devices::Cuda >::set( deviceData, (float) 7, ARRAY_TEST_SIZE );
   MemoryOperations< Devices::Cuda >::set( deviceData2, (double) 7, ARRAY_TEST_SIZE );
   EXPECT_TRUE(( MultiDeviceMemoryOperations< Devices::Host, Devices::Cuda >::compare< int, float >( hostData, deviceData, ARRAY_TEST_SIZE ) ));
   EXPECT_TRUE(( MultiDeviceMemoryOperations< Devices::Cuda, Devices::Host >::compare< float, int >( deviceData, hostData, ARRAY_TEST_SIZE ) ));
   EXPECT_TRUE(( MemoryOperations< Devices::Cuda >::compare< float, double >( deviceData, deviceData2, ARRAY_TEST_SIZE ) ));

   hostAllocator.deallocate( hostData, ARRAY_TEST_SIZE );
   cudaAllocator1.deallocate( deviceData, ARRAY_TEST_SIZE );
   cudaAllocator2.deallocate( deviceData2, ARRAY_TEST_SIZE );
}
#endif // __CUDACC__
#endif // HAVE_GTEST


#include "../main.h"
