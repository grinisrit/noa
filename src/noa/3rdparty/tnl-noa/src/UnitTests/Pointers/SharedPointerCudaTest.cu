#include <cstdlib>
#include <TNL/Devices/Host.h>
#include <TNL/Pointers/SharedPointer.h>
#include <TNL/Containers/StaticArray.h>
#include <TNL/Containers/Array.h>

#ifdef HAVE_GTEST
#include <gtest/gtest.h>
#endif

#include <TNL/Devices/Cuda.h>

using namespace TNL;

#ifdef HAVE_GTEST
TEST( SharedPointerCudaTest, ConstructorTest )
{
#ifdef HAVE_CUDA
   typedef TNL::Containers::StaticArray< 2, int  > TestType;
   Pointers::SharedPointer< TestType, Devices::Cuda > ptr1;

   ptr1->x() = 0;
   ptr1->y() = 0;
   ASSERT_EQ( ptr1->x(), 0 );
   ASSERT_EQ( ptr1->y(), 0 );

   Pointers::SharedPointer< TestType, Devices::Cuda > ptr2( 1, 2 );
   ASSERT_EQ( ptr2->x(), 1 );
   ASSERT_EQ( ptr2->y(), 2 );

   ptr1 = ptr2;
   ASSERT_EQ( ptr1->x(), 1 );
   ASSERT_EQ( ptr1->y(), 2 );
#endif
};

TEST( SharedPointerCudaTest, getDataTest )
{
#ifdef HAVE_CUDA
   typedef TNL::Containers::StaticArray< 2, int  > TestType;
   Pointers::SharedPointer< TestType, Devices::Cuda > ptr1( 1, 2 );

   Pointers::synchronizeSmartPointersOnDevice< Devices::Cuda >();

   TestType aux;

   cudaMemcpy( ( void*) &aux, &ptr1.getData< Devices::Cuda >(), sizeof( TestType ), cudaMemcpyDeviceToHost );

   ASSERT_EQ( aux[ 0 ], 1 );
   ASSERT_EQ( aux[ 1 ], 2 );
#endif  // HAVE_CUDA
};

#ifdef HAVE_CUDA
__global__ void copyArrayKernel( const TNL::Containers::Array< int, Devices::Cuda >* inArray,
                                 int* outArray )
{
   if( threadIdx.x < 2 )
   {
      outArray[ threadIdx.x ] = ( *inArray )[ threadIdx.x ];
   }
}

__global__ void copyArrayKernel2( const Pointers::SharedPointer< TNL::Containers::Array< int, Devices::Cuda > > inArray,
                                  int* outArray )
{
   if( threadIdx.x < 2 )
   {
      outArray[ threadIdx.x ] = ( *inArray )[ threadIdx.x ];
   }
}
#endif

TEST( SharedPointerCudaTest, getDataArrayTest )
{
#ifdef HAVE_CUDA
   typedef TNL::Containers::Array< int, Devices::Cuda  > TestType;
   Pointers::SharedPointer< TestType > ptr;

   ptr->setSize( 2 );
   ptr->setElement( 0, 1 );
   ptr->setElement( 1, 2 );

   Pointers::synchronizeSmartPointersOnDevice< Devices::Cuda >();

   int *testArray_device, *testArray_host;
   cudaMalloc( ( void** ) &testArray_device, 2 * sizeof( int ) );
   copyArrayKernel<<< 1, 2 >>>( &ptr.getData< Devices::Cuda >(), testArray_device );
   testArray_host = new int [ 2 ];
   cudaMemcpy( testArray_host, testArray_device, 2 * sizeof( int ), cudaMemcpyDeviceToHost );

   ASSERT_EQ( testArray_host[ 0 ], 1 );
   ASSERT_EQ( testArray_host[ 1 ], 2 );

   copyArrayKernel2<<< 1, 2 >>>( ptr, testArray_device );
   cudaMemcpy( testArray_host, testArray_device, 2 * sizeof( int ), cudaMemcpyDeviceToHost );

   ASSERT_EQ( testArray_host[ 0 ], 1 );
   ASSERT_EQ( testArray_host[ 1 ], 2 );

   delete[] testArray_host;
   cudaFree( testArray_device );

#endif
};

TEST( SharedPointerCudaTest, nullptrAssignement )
{
#ifdef HAVE_CUDA
   using TestType = Pointers::SharedPointer< double, Devices::Cuda >;
   TestType p1( 5 ), p2( nullptr );

   // This should not crash
   p1 = p2;

   ASSERT_FALSE( p1 );
   ASSERT_FALSE( p2 );
#endif
}

TEST( SharedPointerCudaTest, swap )
{
#ifdef HAVE_CUDA
   using TestType = Pointers::SharedPointer< double, Devices::Cuda >;
   TestType p1( 1 ), p2( 2 );

   p1.swap( p2 );

   ASSERT_EQ( *p1, 2 );
   ASSERT_EQ( *p2, 1 );
#endif
}

#endif


#include "../main.h"
