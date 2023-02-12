#ifdef NDEBUG
   #undef NDEBUG
#endif

#include <TNL/Assert.h>
#include <TNL/Cuda/CheckDevice.h>
#include <TNL/Exceptions/CudaRuntimeError.h>

#ifdef HAVE_GTEST
#include <gtest/gtest.h>

using namespace TNL;

#ifdef __CUDACC__

// ignore useless nvcc warning: https://stackoverflow.com/a/49997636
#pragma push
#pragma diag_suppress = declared_but_not_referenced

#define WRAP_ASSERT( suffix, statement, not_failing )             \
__global__                                                        \
void kernel_##suffix( int* output )                               \
{                                                                 \
   const bool tr = true;                                          \
   const bool fa = false;                                         \
   const int two = 2;                                             \
   const int ten = 10;                                            \
   /* pointers */                                                 \
   const double* data_null = nullptr;                             \
   const double** data_full = &data_null;                         \
                                                                  \
   /* ignore warnings due to potentially unused variables */      \
   (void) tr;                                                     \
   (void) fa;                                                     \
   (void) two;                                                    \
   (void) ten;                                                    \
   (void) data_null;                                              \
   (void) data_full;                                              \
                                                                  \
   statement                                                      \
                                                                  \
   /* actually do something to test if the execution control gets here */ \
   output[0] = 1;                                                 \
}                                                                 \
                                                                  \
TEST( AssertCudaTest, suffix )                                    \
{                                                                 \
   int* output_cuda;                                              \
   int output_host = 0;                                           \
                                                                  \
   cudaMalloc( (void**)&output_cuda, sizeof(int) );               \
   cudaMemcpy( output_cuda, &output_host, sizeof(int), cudaMemcpyHostToDevice );    \
                                                                  \
   kernel_##suffix<<<1, 1>>>(output_cuda);                        \
   cudaDeviceSynchronize();                                       \
   if( not_failing ) {                                            \
      EXPECT_NO_THROW( TNL_CHECK_CUDA_DEVICE; );                  \
      cudaMemcpy( &output_host, output_cuda, sizeof(int), cudaMemcpyDeviceToHost );    \
      cudaFree( output_cuda );                                    \
      EXPECT_EQ( output_host, 1 );                                \
   }                                                              \
   else                                                           \
      EXPECT_THROW( TNL_CHECK_CUDA_DEVICE;,                       \
                    TNL::Exceptions::CudaRuntimeError );          \
}                                                                 \


// not failing statements:
WRAP_ASSERT( test1,  TNL_ASSERT_TRUE( true, "true is true" );, true );
WRAP_ASSERT( test2,  TNL_ASSERT_TRUE( tr, "true is true" );, true );
WRAP_ASSERT( test3,  TNL_ASSERT_FALSE( false, "false is false" );, true );
WRAP_ASSERT( test4,  TNL_ASSERT_FALSE( fa, "false is false" );, true );

WRAP_ASSERT( test5,  TNL_ASSERT_EQ( two, 2, "two is 2" );, true );
WRAP_ASSERT( test6,  TNL_ASSERT_NE( ten, 2, "ten is not 2" );, true );
WRAP_ASSERT( test7,  TNL_ASSERT_LT( two, 10, "two < 10" );, true );
WRAP_ASSERT( test8,  TNL_ASSERT_LE( two, 10, "two <= 10" );, true );
WRAP_ASSERT( test9,  TNL_ASSERT_LE( two, 2, "two <= 2" );, true );
WRAP_ASSERT( test10, TNL_ASSERT_GT( ten, 2, "ten > 2" );, true );
WRAP_ASSERT( test11, TNL_ASSERT_GE( ten, 10, "ten >= 10" );, true );
WRAP_ASSERT( test12, TNL_ASSERT_GE( ten, 2, "ten >= 2" );, true );

WRAP_ASSERT( test13, TNL_ASSERT_FALSE( data_null, "nullptr is false" );, true );
WRAP_ASSERT( test14, TNL_ASSERT_TRUE( data_full, "non-nullptr is true" );, true );

// errors:
WRAP_ASSERT( test15, TNL_ASSERT_TRUE( false, "false is true" );, false );
WRAP_ASSERT( test16, TNL_ASSERT_TRUE( fa, "false is true" );, false );
WRAP_ASSERT( test17, TNL_ASSERT_FALSE( true, "true is false" );, false );
WRAP_ASSERT( test18, TNL_ASSERT_FALSE( tr, "true is false" );, false );

WRAP_ASSERT( test19, TNL_ASSERT_NE( two, 2, "two != 2" );, false );
WRAP_ASSERT( test20, TNL_ASSERT_EQ( ten, 2, "ten == 2" );, false );
WRAP_ASSERT( test21, TNL_ASSERT_GE( two, 10, "two >= 10" );, false );
WRAP_ASSERT( test22, TNL_ASSERT_GT( two, 10, "two > 10" );, false );
WRAP_ASSERT( test23, TNL_ASSERT_GT( two, 2, "two > 2" );, false );
WRAP_ASSERT( test24, TNL_ASSERT_LE( ten, 2, "ten <= 2" );, false );
WRAP_ASSERT( test25, TNL_ASSERT_LT( ten, 10, "ten < 10" );, false );
WRAP_ASSERT( test26, TNL_ASSERT_LT( ten, 2, "ten < 2" );, false );

WRAP_ASSERT( test27, TNL_ASSERT_TRUE( data_null, "nullptr is true" );, false );
WRAP_ASSERT( test28, TNL_ASSERT_FALSE( data_full, "non-nullptr is false" );, false );

#pragma pop

#endif
#endif

#include "main.h"
