#include <TNL/File.h>

#ifdef HAVE_GTEST
#include <gtest/gtest.h>

using namespace TNL;

static const char* TEST_FILE_NAME = "test_FileTest.tnl";

TEST( FileTest, OpenInvalid )
{
   File file;
   EXPECT_THROW( file.open( "invalid-file.tnl", std::ios_base::in ), std::ios_base::failure );
}

TEST( FileTest, WriteAndRead )
{
   File file;
   ASSERT_NO_THROW( file.open( String( TEST_FILE_NAME ), std::ios_base::out ) );

   int intData( 5 );
   double doubleData[ 3 ] = { 1.0, 2.0, 3.0 };
   const double constDoubleData = 3.14;
   ASSERT_NO_THROW( file.save( &intData ) );
   ASSERT_NO_THROW( file.save( doubleData, 3 ) );
   ASSERT_NO_THROW( file.save( &constDoubleData ) );
   ASSERT_NO_THROW( file.close() );

   ASSERT_NO_THROW( file.open( String( TEST_FILE_NAME ), std::ios_base::in ) );
   int newIntData;
   double newDoubleData[ 3 ];
   double newConstDoubleData;
   ASSERT_NO_THROW( file.load( &newIntData, 1 ) );
   ASSERT_NO_THROW( file.load( newDoubleData, 3 ) );
   ASSERT_NO_THROW( file.load( &newConstDoubleData, 1 ) );

   EXPECT_EQ( newIntData, intData );
   for( int i = 0; i < 3; i ++ )
      EXPECT_EQ( newDoubleData[ i ], doubleData[ i ] );
   EXPECT_EQ( newConstDoubleData, constDoubleData );

   EXPECT_EQ( std::remove( TEST_FILE_NAME ), 0 );
};

TEST( FileTest, WriteAndReadWithConversion )
{
   double doubleData[ 3 ] = {  3.1415926535897932384626433,
                               2.7182818284590452353602874,
                               1.6180339887498948482045868 };
   float floatData[ 3 ];
   int intData[ 3 ];
   File file;
   ASSERT_NO_THROW( file.open( TEST_FILE_NAME, std::ios_base::out | std::ios_base::trunc ) );
   file.save< double, float >( doubleData, 3 );
   ASSERT_NO_THROW( file.close() );

   ASSERT_NO_THROW( file.open( TEST_FILE_NAME, std::ios_base::in ) );
   file.load< float, float >( floatData, 3 );
   ASSERT_NO_THROW( file.close() );

   ASSERT_NO_THROW( file.open( TEST_FILE_NAME, std::ios_base::in ) );
   file.load< int, float >( intData, 3 );
   ASSERT_NO_THROW( file.close() );

   EXPECT_NEAR( floatData[ 0 ], 3.14159, 0.0001 );
   EXPECT_NEAR( floatData[ 1 ], 2.71828, 0.0001 );
   EXPECT_NEAR( floatData[ 2 ], 1.61803, 0.0001 );

   EXPECT_EQ( intData[ 0 ], 3 );
   EXPECT_EQ( intData[ 1 ], 2 );
   EXPECT_EQ( intData[ 2 ], 1 );

   EXPECT_EQ( std::remove( TEST_FILE_NAME ), 0 );
}

#ifdef HAVE_CUDA
TEST( FileTest, WriteAndReadCUDA )
{
   int intData( 5 );
   float floatData[ 3 ] = { 1.0, 2.0, 3.0 };
   const double constDoubleData = 3.14;

   int* cudaIntData;
   float* cudaFloatData;
   const double* cudaConstDoubleData;
   cudaMalloc( ( void** ) &cudaIntData, sizeof( int ) );
   cudaMalloc( ( void** ) &cudaFloatData, 3 * sizeof( float ) );
   cudaMalloc( ( void** ) &cudaConstDoubleData, sizeof( double ) );
   cudaMemcpy( cudaIntData,
               &intData,
               sizeof( int ),
               cudaMemcpyHostToDevice );
   cudaMemcpy( cudaFloatData,
               floatData,
               3 * sizeof( float ),
               cudaMemcpyHostToDevice );
   cudaMemcpy( (void*) cudaConstDoubleData,
               &constDoubleData,
               sizeof( double ),
               cudaMemcpyHostToDevice );

   File file;
   ASSERT_NO_THROW( file.open( String( TEST_FILE_NAME ), std::ios_base::out ) );

   file.save< int, int, Allocators::Cuda<int> >( cudaIntData );
   file.save< float, float, Allocators::Cuda<float> >( cudaFloatData, 3 );
   file.save< const double, double, Allocators::Cuda<const double> >( cudaConstDoubleData );
   ASSERT_NO_THROW( file.close() );

   ASSERT_NO_THROW( file.open( String( TEST_FILE_NAME ), std::ios_base::in ) );
   int newIntData;
   float newFloatData[ 3 ];
   double newDoubleData;
   int* newCudaIntData;
   float* newCudaFloatData;
   double* newCudaDoubleData;
   cudaMalloc( ( void** ) &newCudaIntData, sizeof( int ) );
   cudaMalloc( ( void** ) &newCudaFloatData, 3 * sizeof( float ) );
   cudaMalloc( ( void** ) &newCudaDoubleData, sizeof( double ) );
   file.load< int, int, Allocators::Cuda<int> >( newCudaIntData, 1 );
   file.load< float, float, Allocators::Cuda<float> >( newCudaFloatData, 3 );
   file.load< double, double, Allocators::Cuda<double> >( newCudaDoubleData, 1 );
   cudaMemcpy( &newIntData,
               newCudaIntData,
               sizeof( int ),
               cudaMemcpyDeviceToHost );
   cudaMemcpy( newFloatData,
               newCudaFloatData,
               3 * sizeof( float ),
               cudaMemcpyDeviceToHost );
   cudaMemcpy( &newDoubleData,
               newCudaDoubleData,
               sizeof( double ),
               cudaMemcpyDeviceToHost );

   EXPECT_EQ( newIntData, intData );
   for( int i = 0; i < 3; i ++ )
      EXPECT_EQ( newFloatData[ i ], floatData[ i ] );
   EXPECT_EQ( newDoubleData, constDoubleData );

   EXPECT_EQ( std::remove( TEST_FILE_NAME ), 0 );
}

TEST( FileTest, WriteAndReadCUDAWithConversion )
{
   const double constDoubleData[ 3 ] = {  3.1415926535897932384626433,
                                          2.7182818284590452353602874,
                                          1.6180339887498948482045868 };
   float floatData[ 3 ];
   int intData[ 3 ];

   int* cudaIntData;
   float* cudaFloatData;
   const double* cudaConstDoubleData;
   cudaMalloc( ( void** ) &cudaIntData, 3 * sizeof( int ) );
   cudaMalloc( ( void** ) &cudaFloatData, 3 * sizeof( float ) );
   cudaMalloc( ( void** ) &cudaConstDoubleData, 3 * sizeof( double ) );
   cudaMemcpy( (void*) cudaConstDoubleData,
               &constDoubleData,
               3 * sizeof( double ),
               cudaMemcpyHostToDevice );

   File file;
   ASSERT_NO_THROW( file.open( String( TEST_FILE_NAME ), std::ios_base::out | std::ios_base::trunc ) );
   file.save< double, float, Allocators::Cuda<double> >( cudaConstDoubleData, 3 );
   ASSERT_NO_THROW( file.close() );

   ASSERT_NO_THROW( file.open( String( TEST_FILE_NAME ), std::ios_base::in ) );
   file.load< float, float, Allocators::Cuda<float> >( cudaFloatData, 3 );
   ASSERT_NO_THROW( file.close() );

   ASSERT_NO_THROW( file.open( String( TEST_FILE_NAME ), std::ios_base::in ) );
   file.load< int, float, Allocators::Cuda<int> >( cudaIntData, 3 );
   ASSERT_NO_THROW( file.close() );

   cudaMemcpy( floatData,
               cudaFloatData,
               3 * sizeof( float ),
               cudaMemcpyDeviceToHost );
   cudaMemcpy( &intData,
               cudaIntData,
               3* sizeof( int ),
               cudaMemcpyDeviceToHost );


   EXPECT_NEAR( floatData[ 0 ], 3.14159, 0.0001 );
   EXPECT_NEAR( floatData[ 1 ], 2.71828, 0.0001 );
   EXPECT_NEAR( floatData[ 2 ], 1.61803, 0.0001 );

   EXPECT_EQ( intData[ 0 ], 3 );
   EXPECT_EQ( intData[ 1 ], 2 );
   EXPECT_EQ( intData[ 2 ], 1 );

   EXPECT_EQ( std::remove( TEST_FILE_NAME ), 0 );
}

#endif
#endif

#include "main.h"
