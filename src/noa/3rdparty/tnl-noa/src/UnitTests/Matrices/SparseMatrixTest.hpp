#pragma once

#include <functional>
#include <TNL/Containers/Vector.h>
#include <TNL/Containers/VectorView.h>
#include <TNL/Math.h>
#include <TNL/Algorithms/ParallelFor.h>
#include <iostream>
#include <sstream>

// Just for ChunkedEllpack vectorProduct test exception
#include <TNL/Algorithms/Segments/ChunkedEllpackView.h>

#ifdef HAVE_GTEST
#include <gtest/gtest.h>

template< typename MatrixHostFloat, typename MatrixHostInt >
void host_test_GetType()
{
   bool testRan = false;
   EXPECT_TRUE( testRan );
   std::cout << "\nTEST DID NOT RUN. NOT WORKING.\n\n";
   std::cerr << "This test has not been implemented properly yet.\n" << std::endl;
}

template< typename MatrixCudaFloat, typename MatrixCudaInt >
void cuda_test_GetType()
{
   bool testRan = false;
   EXPECT_TRUE( testRan );
   std::cout << "\nTEST DID NOT RUN. NOT WORKING.\n\n";
   std::cerr << "This test has not been implemented properly yet.\n" << std::endl;
}

template< typename Matrix >
void test_Constructors()
{
   using RealType = typename Matrix::RealType;
   using DeviceType = typename Matrix::DeviceType;
   using IndexType = typename Matrix::IndexType;

   Matrix m1( 5, 6 );
   EXPECT_EQ( m1.getRows(), 5 );
   EXPECT_EQ( m1.getColumns(), 6 );

   Matrix m2( {1, 2, 2, 2, 1 }, 5 );
   typename Matrix::RowsCapacitiesType v1, v2{ 1, 2, 2, 2, 1 };
   m2.setElement( 0, 0, 1 );   // 0th row
   m2.setElement( 1, 0, 1 );   // 1st row
   m2.setElement( 1, 1, 1 );
   m2.setElement( 2, 1, 1 );   // 2nd row
   m2.setElement( 2, 2, 1 );
   m2.setElement( 3, 2, 1 );   // 3rd row
   m2.setElement( 3, 3, 1 );
   m2.setElement( 4, 4, 1 );   // 4th row

   EXPECT_EQ( m2.getElement( 0, 0 ), 1 );   // 0th row
   EXPECT_EQ( m2.getElement( 1, 0 ), 1 );   // 1st row
   EXPECT_EQ( m2.getElement( 1, 1 ), 1 );
   EXPECT_EQ( m2.getElement( 2, 1 ), 1 );   // 2nd row
   EXPECT_EQ( m2.getElement( 2, 2 ), 1 );
   EXPECT_EQ( m2.getElement( 3, 2 ), 1 );   // 3rd row
   EXPECT_EQ( m2.getElement( 3, 3 ), 1 );
   EXPECT_EQ( m2.getElement( 4, 4 ), 1 );   // 4th row

   if( std::is_same< DeviceType, TNL::Devices::Host >::value )
   {
      EXPECT_EQ( m2.getRow( 0 ).getValue( 0 ), 1 );   // 0th row
      EXPECT_EQ( m2.getRow( 1 ).getValue( 0 ), 1 );   // 1st row
      EXPECT_EQ( m2.getRow( 1 ).getValue( 1 ), 1 );
      EXPECT_EQ( m2.getRow( 2 ).getValue( 0 ), 1 );   // 2nd row
      EXPECT_EQ( m2.getRow( 2 ).getValue( 1 ), 1 );
      EXPECT_EQ( m2.getRow( 3 ).getValue( 0 ), 1 );   // 3rd row
      EXPECT_EQ( m2.getRow( 3 ).getValue( 1 ), 1 );
      EXPECT_EQ( m2.getRow( 4 ).getValue( 0 ), 1 );   // 4th row

      const Matrix& mm = m2;
      EXPECT_EQ( mm.getRow( 0 ).getValue( 0 ), 1 );   // 0th row
      EXPECT_EQ( mm.getRow( 1 ).getValue( 0 ), 1 );   // 1st row
      EXPECT_EQ( mm.getRow( 1 ).getValue( 1 ), 1 );
      EXPECT_EQ( mm.getRow( 2 ).getValue( 0 ), 1 );   // 2nd row
      EXPECT_EQ( mm.getRow( 2 ).getValue( 1 ), 1 );
      EXPECT_EQ( mm.getRow( 3 ).getValue( 0 ), 1 );   // 3rd row
      EXPECT_EQ( mm.getRow( 3 ).getValue( 1 ), 1 );
      EXPECT_EQ( mm.getRow( 4 ).getValue( 0 ), 1 );   // 4th row
   }

   m2.getCompressedRowLengths( v1 );
   EXPECT_EQ( v1, v2 );

   /*
    * Sets up the following 6x5 sparse matrix:
    *
    *    /  1  2  3  0  0 \
    *    |  0  4  5  6  0 |
    *    |  0  0  7  8  9 |
    *    | 10  0  0  0  0 |
    *    |  0 11  0  0  0 |
    *    \  0  0  0 12  0 /
    */

   const Matrix m3( 6, 5, {
      { 0, 0,  1.0 }, { 0, 1, 2.0 }, { 0, 2, 3.0 },
      { 1, 1,  4.0 }, { 1, 2, 5.0 }, { 1, 3, 6.0 },
      { 2, 2,  7.0 }, { 2, 3, 8.0 }, { 2, 4, 9.0 },
      { 3, 0, 10.0 },
      { 4, 1, 11.0 },
      { 5, 3, 12.0 } } );

   // Check the set elements
   EXPECT_EQ( m3.getElement( 0, 0 ),  1 );
   EXPECT_EQ( m3.getElement( 0, 1 ),  2 );
   EXPECT_EQ( m3.getElement( 0, 2 ),  3 );
   EXPECT_EQ( m3.getElement( 0, 3 ),  0 );
   EXPECT_EQ( m3.getElement( 0, 4 ),  0 );

   EXPECT_EQ( m3.getElement( 1, 0 ),  0 );
   EXPECT_EQ( m3.getElement( 1, 1 ),  4 );
   EXPECT_EQ( m3.getElement( 1, 2 ),  5 );
   EXPECT_EQ( m3.getElement( 1, 3 ),  6 );
   EXPECT_EQ( m3.getElement( 1, 4 ),  0 );

   EXPECT_EQ( m3.getElement( 2, 0 ),  0 );
   EXPECT_EQ( m3.getElement( 2, 1 ),  0 );
   EXPECT_EQ( m3.getElement( 2, 2 ),  7 );
   EXPECT_EQ( m3.getElement( 2, 3 ),  8 );
   EXPECT_EQ( m3.getElement( 2, 4 ),  9 );

   EXPECT_EQ( m3.getElement( 3, 0 ), 10 );
   EXPECT_EQ( m3.getElement( 3, 1 ),  0 );
   EXPECT_EQ( m3.getElement( 3, 2 ),  0 );
   EXPECT_EQ( m3.getElement( 3, 3 ),  0 );
   EXPECT_EQ( m3.getElement( 3, 4 ),  0 );

   EXPECT_EQ( m3.getElement( 4, 0 ),  0 );
   EXPECT_EQ( m3.getElement( 4, 1 ), 11 );
   EXPECT_EQ( m3.getElement( 4, 2 ),  0 );
   EXPECT_EQ( m3.getElement( 4, 3 ),  0 );
   EXPECT_EQ( m3.getElement( 4, 4 ),  0 );

   EXPECT_EQ( m3.getElement( 5, 0 ),  0 );
   EXPECT_EQ( m3.getElement( 5, 1 ),  0 );
   EXPECT_EQ( m3.getElement( 5, 2 ),  0 );
   EXPECT_EQ( m3.getElement( 5, 3 ), 12 );
   EXPECT_EQ( m3.getElement( 5, 4 ),  0 );

   if( std::is_same< DeviceType, TNL::Devices::Host >::value )
   {
      EXPECT_EQ( m3.getRow( 0 ).getValue( 0 ),  1 );
      EXPECT_EQ( m3.getRow( 0 ).getValue( 1 ),  2 );
      EXPECT_EQ( m3.getRow( 0 ).getValue( 2 ),  3 );

      EXPECT_EQ( m3.getRow( 1 ).getValue( 0 ),  4 );
      EXPECT_EQ( m3.getRow( 1 ).getValue( 1 ),  5 );
      EXPECT_EQ( m3.getRow( 1 ).getValue( 2 ),  6 );

      EXPECT_EQ( m3.getRow( 2 ).getValue( 0 ),  7 );
      EXPECT_EQ( m3.getRow( 2 ).getValue( 1 ),  8 );
      EXPECT_EQ( m3.getRow( 2 ).getValue( 2 ),  9 );

      EXPECT_EQ( m3.getRow( 3 ).getValue( 0 ), 10 );

      EXPECT_EQ( m3.getRow( 4 ).getValue( 0 ), 11 );

      EXPECT_EQ( m3.getRow( 5 ).getValue( 0 ), 12 );
   }

   std::map< std::pair< int, int >, float > map;
   map[ { 0, 0 } ] = 1.0;
   map[ { 0, 1 } ] = 2.0;
   map[ { 0, 2 } ] = 3.0;
   map[ { 1, 1 } ] = 4.0;
   map[ { 1, 2 } ] = 5.0;
   map[ { 1, 3 } ] = 6.0;
   map[ { 2, 2 } ] = 7.0;
   map[ { 2, 3 } ] = 8.0;
   map[ { 2, 4 } ] = 9.0;
   map[ { 3, 0 } ] = 10.0;
   map[ { 4, 1 } ] = 11.0;
   map[ { 5, 3 } ] = 12.0;
   Matrix m4( 6, 5, map );

   // Check the matrix elements
   EXPECT_EQ( m4.getElement( 0, 0 ),  1 );
   EXPECT_EQ( m4.getElement( 0, 1 ),  2 );
   EXPECT_EQ( m4.getElement( 0, 2 ),  3 );
   EXPECT_EQ( m4.getElement( 0, 3 ),  0 );
   EXPECT_EQ( m4.getElement( 0, 4 ),  0 );

   EXPECT_EQ( m4.getElement( 1, 0 ),  0 );
   EXPECT_EQ( m4.getElement( 1, 1 ),  4 );
   EXPECT_EQ( m4.getElement( 1, 2 ),  5 );
   EXPECT_EQ( m4.getElement( 1, 3 ),  6 );
   EXPECT_EQ( m4.getElement( 1, 4 ),  0 );

   EXPECT_EQ( m4.getElement( 2, 0 ),  0 );
   EXPECT_EQ( m4.getElement( 2, 1 ),  0 );
   EXPECT_EQ( m4.getElement( 2, 2 ),  7 );
   EXPECT_EQ( m4.getElement( 2, 3 ),  8 );
   EXPECT_EQ( m4.getElement( 2, 4 ),  9 );

   EXPECT_EQ( m4.getElement( 3, 0 ), 10 );
   EXPECT_EQ( m4.getElement( 3, 1 ),  0 );
   EXPECT_EQ( m4.getElement( 3, 2 ),  0 );
   EXPECT_EQ( m4.getElement( 3, 3 ),  0 );
   EXPECT_EQ( m4.getElement( 3, 4 ),  0 );

   EXPECT_EQ( m4.getElement( 4, 0 ),  0 );
   EXPECT_EQ( m4.getElement( 4, 1 ), 11 );
   EXPECT_EQ( m4.getElement( 4, 2 ),  0 );
   EXPECT_EQ( m4.getElement( 4, 3 ),  0 );
   EXPECT_EQ( m4.getElement( 4, 4 ),  0 );

   EXPECT_EQ( m4.getElement( 5, 0 ),  0 );
   EXPECT_EQ( m4.getElement( 5, 1 ),  0 );
   EXPECT_EQ( m4.getElement( 5, 2 ),  0 );
   EXPECT_EQ( m4.getElement( 5, 3 ), 12 );
   EXPECT_EQ( m4.getElement( 5, 4 ),  0 );
}

template< typename Matrix >
void test_SetDimensions()
{
   using RealType = typename Matrix::RealType;
   using DeviceType = typename Matrix::DeviceType;
   using IndexType = typename Matrix::IndexType;

   const IndexType rows = 9;
   const IndexType cols = 8;

   Matrix m;
   m.setDimensions( rows, cols );

   EXPECT_EQ( m.getRows(), 9 );
   EXPECT_EQ( m.getColumns(), 8 );

   // test empty matrix
   m.setDimensions( 0, 0 );
}

template< typename Matrix >
void test_SetRowCapacities()
{
   using RealType = typename Matrix::RealType;
   using DeviceType = typename Matrix::DeviceType;
   using IndexType = typename Matrix::IndexType;

   const IndexType rows = 10;
   const IndexType cols = 11;

   Matrix m( rows, cols );
   typename Matrix::RowsCapacitiesType rowLengths( rows, 3 );

   IndexType rowLength = 1;
   for( IndexType i = 2; i < rows; i++ )
      rowLengths.setElement( i, rowLength++ );

   m.setRowCapacities( rowLengths );

   // Insert values into the rows.
   RealType value = 1;

   for( IndexType i = 0; i < 3; i++ )      // 0th row
      m.setElement( 0, i, value++ );

   for( IndexType i = 0; i < 3; i++ )      // 1st row
      m.setElement( 1, i, value++ );

   for( IndexType i = 0; i < 1; i++ )      // 2nd row
      m.setElement( 2, i, value++ );

   for( IndexType i = 0; i < 2; i++ )      // 3rd row
      m.setElement( 3, i, value++ );

   for( IndexType i = 0; i < 3; i++ )      // 4th row
      m.setElement( 4, i, value++ );

   for( IndexType i = 0; i < 4; i++ )      // 5th row
      m.setElement( 5, i, value++ );

   for( IndexType i = 0; i < 5; i++ )      // 6th row
      m.setElement( 6, i, value++ );

   for( IndexType i = 0; i < 6; i++ )      // 7th row
      m.setElement( 7, i, value++ );

   for( IndexType i = 0; i < 7; i++ )      // 8th row
      m.setElement( 8, i, value++ );

   for( IndexType i = 0; i < 8; i++ )      // 9th row
      m.setElement( 9, i, value++ );

   rowLengths = 0;
   m.getCompressedRowLengths( rowLengths );
   typename Matrix::RowsCapacitiesType correctRowLengths{ 3, 3, 1, 2, 3, 4, 5, 6, 7, 8 };
   EXPECT_EQ( rowLengths, correctRowLengths );
}

template< typename Matrix1, typename Matrix2 >
void test_SetLike()
{
   using RealType = typename Matrix1::RealType;
   using DeviceType = typename Matrix1::DeviceType;
   using IndexType = typename Matrix1::IndexType;

   const IndexType rows = 8;
   const IndexType cols = 7;

   Matrix1 m1( rows + 1, cols + 2 );
   Matrix2 m2( rows, cols );

   m1.setLike( m2 );

   EXPECT_EQ( m1.getRows(), m2.getRows() );
   EXPECT_EQ( m1.getColumns(), m2.getColumns() );
}

template< typename Matrix >
void test_GetNonzeroElementsCount()
{
   using RealType = typename Matrix::RealType;
   using DeviceType = typename Matrix::DeviceType;
   using IndexType = typename Matrix::IndexType;

   /*
    * Sets up the following 10x10 sparse matrix:
    *
    *    /  1  0  2  0  3  0  4  0  0  0  \
    *    |  5  6  7  0  0  0  0  0  0  0  |
    *    |  8  9 10 11 12 13 14 15  0  0  |
    *    | 16 17  0  0  0  0  0  0  0  0  |
    *    | 18  0  0  0  0  0  0  0  0  0  |
    *    | 19  0  0  0  0  0  0  0  0  0  |
    *    | 20  0  0  0  0  0  0  0  0  0  |
    *    | 21  0  0  0  0  0  0  0  0  0  |
    *    | 22 23 24 25 26 27 28 29 30 31  |
    *    \ 32 33 34 35 36 37 38 39 40 41 /
    */

   const IndexType rows = 10;
   const IndexType cols = 10;

   Matrix m( rows, cols );

   typename Matrix::RowsCapacitiesType rowLengths{ 4, 3, 8, 2, 1, 1, 1, 1, 10, 10 };
   m.setRowCapacities( rowLengths );

   RealType value = 1;
   for( IndexType i = 0; i < 4; i++ )
      m.setElement( 0, 2 * i, value++ );

   for( IndexType i = 0; i < 3; i++ )
      m.setElement( 1, i, value++ );

   for( IndexType i = 0; i < 8; i++ )
      m.setElement( 2, i, value++ );

   for( IndexType i = 0; i < 2; i++ )
      m.setElement( 3, i, value++ );

   for( IndexType i = 4; i < 8; i++ )
      m.setElement( i, 0, value++ );

   for( IndexType j = 8; j < rows; j++)
      for( IndexType i = 0; i < cols; i++ )
         m.setElement( j, i, value++ );

   EXPECT_EQ( m.getNonzeroElementsCount(), 41 );
}

template< typename Matrix >
void test_Reset()
{
   using RealType = typename Matrix::RealType;
   using DeviceType = typename Matrix::DeviceType;
   using IndexType = typename Matrix::IndexType;

   /*
    * Sets up the following 5x4 sparse matrix:
    *
    *    /  0  0  0  0 \
    *    |  0  0  0  0 |
    *    |  0  0  0  0 |
    *    |  0  0  0  0 |
    *    \  0  0  0  0 /
    */

   const IndexType rows = 5;
   const IndexType cols = 4;

   Matrix m( rows, cols );
   m.reset();

   EXPECT_EQ( m.getRows(), 0 );
   EXPECT_EQ( m.getColumns(), 0 );
}

template< typename Matrix >
void test_GetRow()
{
   using RealType = typename Matrix::RealType;
   using DeviceType = typename Matrix::DeviceType;
   using IndexType = typename Matrix::IndexType;

   Matrix m2( {1, 2, 2, 2, 1 }, 5 );
   typename Matrix::RowsCapacitiesType v1, v2{ 1, 2, 2, 2, 1 };
   m2.setElement( 0, 0, 1 );   // 0th row
   m2.setElement( 1, 0, 1 );   // 1st row
   m2.setElement( 1, 1, 1 );
   m2.setElement( 2, 1, 1 );   // 2nd row
   m2.setElement( 2, 2, 1 );
   m2.setElement( 3, 2, 1 );   // 3rd row
   m2.setElement( 3, 3, 1 );
   m2.setElement( 4, 4, 1 );   // 4th row

   EXPECT_EQ( m2.getElement( 0, 0 ), 1 );   // 0th row
   EXPECT_EQ( m2.getElement( 1, 0 ), 1 );   // 1st row
   EXPECT_EQ( m2.getElement( 1, 1 ), 1 );
   EXPECT_EQ( m2.getElement( 2, 1 ), 1 );   // 2nd row
   EXPECT_EQ( m2.getElement( 2, 2 ), 1 );
   EXPECT_EQ( m2.getElement( 3, 2 ), 1 );   // 3rd row
   EXPECT_EQ( m2.getElement( 3, 3 ), 1 );
   EXPECT_EQ( m2.getElement( 4, 4 ), 1 );   // 4th row

   if( std::is_same< DeviceType, TNL::Devices::Host >::value )
   {
      EXPECT_EQ( m2.getRow( 0 ).getValue( 0 ), 1 );   // 0th row
      EXPECT_EQ( m2.getRow( 1 ).getValue( 0 ), 1 );   // 1st row
      EXPECT_EQ( m2.getRow( 1 ).getValue( 1 ), 1 );
      EXPECT_EQ( m2.getRow( 2 ).getValue( 0 ), 1 );   // 2nd row
      EXPECT_EQ( m2.getRow( 2 ).getValue( 1 ), 1 );
      EXPECT_EQ( m2.getRow( 3 ).getValue( 0 ), 1 );   // 3rd row
      EXPECT_EQ( m2.getRow( 3 ).getValue( 1 ), 1 );
      EXPECT_EQ( m2.getRow( 4 ).getValue( 0 ), 1 );   // 4th row
   }

   m2.getCompressedRowLengths( v1 );
   EXPECT_EQ( v1, v2 );

   /*
    * Sets up the following 6x5 sparse matrix:
    *
    *    /  1  2  3  0  0 \
    *    |  0  4  5  6  0 |
    *    |  0  0  7  8  9 |
    *    | 10  0  0  0  0 |
    *    |  0 11  0  0  0 |
    *    \  0  0  0 12  0 /
    */

   const Matrix m3( 6, 5, {
      { 0, 0,  1.0 }, { 0, 1, 2.0 }, { 0, 2, 3.0 },
      { 1, 1,  4.0 }, { 1, 2, 5.0 }, { 1, 3, 6.0 },
      { 2, 2,  7.0 }, { 2, 3, 8.0 }, { 2, 4, 9.0 },
      { 3, 0, 10.0 },
      { 4, 1, 11.0 },
      { 5, 3, 12.0 } } );

   // Check the set elements
   EXPECT_EQ( m3.getElement( 0, 0 ),  1 );
   EXPECT_EQ( m3.getElement( 0, 1 ),  2 );
   EXPECT_EQ( m3.getElement( 0, 2 ),  3 );
   EXPECT_EQ( m3.getElement( 0, 3 ),  0 );
   EXPECT_EQ( m3.getElement( 0, 4 ),  0 );

   EXPECT_EQ( m3.getElement( 1, 0 ),  0 );
   EXPECT_EQ( m3.getElement( 1, 1 ),  4 );
   EXPECT_EQ( m3.getElement( 1, 2 ),  5 );
   EXPECT_EQ( m3.getElement( 1, 3 ),  6 );
   EXPECT_EQ( m3.getElement( 1, 4 ),  0 );

   EXPECT_EQ( m3.getElement( 2, 0 ),  0 );
   EXPECT_EQ( m3.getElement( 2, 1 ),  0 );
   EXPECT_EQ( m3.getElement( 2, 2 ),  7 );
   EXPECT_EQ( m3.getElement( 2, 3 ),  8 );
   EXPECT_EQ( m3.getElement( 2, 4 ),  9 );

   EXPECT_EQ( m3.getElement( 3, 0 ), 10 );
   EXPECT_EQ( m3.getElement( 3, 1 ),  0 );
   EXPECT_EQ( m3.getElement( 3, 2 ),  0 );
   EXPECT_EQ( m3.getElement( 3, 3 ),  0 );
   EXPECT_EQ( m3.getElement( 3, 4 ),  0 );

   EXPECT_EQ( m3.getElement( 4, 0 ),  0 );
   EXPECT_EQ( m3.getElement( 4, 1 ), 11 );
   EXPECT_EQ( m3.getElement( 4, 2 ),  0 );
   EXPECT_EQ( m3.getElement( 4, 3 ),  0 );
   EXPECT_EQ( m3.getElement( 4, 4 ),  0 );

   EXPECT_EQ( m3.getElement( 5, 0 ),  0 );
   EXPECT_EQ( m3.getElement( 5, 1 ),  0 );
   EXPECT_EQ( m3.getElement( 5, 2 ),  0 );
   EXPECT_EQ( m3.getElement( 5, 3 ), 12 );
   EXPECT_EQ( m3.getElement( 5, 4 ),  0 );

   if( std::is_same< DeviceType, TNL::Devices::Host >::value )
   {
      EXPECT_EQ( m3.getRow( 0 ).getValue( 0 ),  1 );
      EXPECT_EQ( m3.getRow( 0 ).getValue( 1 ),  2 );
      EXPECT_EQ( m3.getRow( 0 ).getValue( 2 ),  3 );

      EXPECT_EQ( m3.getRow( 1 ).getValue( 0 ),  4 );
      EXPECT_EQ( m3.getRow( 1 ).getValue( 1 ),  5 );
      EXPECT_EQ( m3.getRow( 1 ).getValue( 2 ),  6 );

      EXPECT_EQ( m3.getRow( 2 ).getValue( 0 ),  7 );
      EXPECT_EQ( m3.getRow( 2 ).getValue( 1 ),  8 );
      EXPECT_EQ( m3.getRow( 2 ).getValue( 2 ),  9 );

      EXPECT_EQ( m3.getRow( 3 ).getValue( 0 ), 10 );

      EXPECT_EQ( m3.getRow( 4 ).getValue( 0 ), 11 );

      EXPECT_EQ( m3.getRow( 5 ).getValue( 0 ), 12 );
   }

   /*
    * Sets up the following 10x10 sparse matrix:
    *
    *    /  1  0  2  0  3  0  4  0  0  0  \
    *    |  5  6  7  0  0  0  0  0  0  0  |
    *    |  8  9 10 11 12 13 14 15  0  0  |
    *    | 16 17  0  0  0  0  0  0  0  0  |
    *    | 18  0  0  0  0  0  0  0  0  0  |
    *    | 19  0  0  0  0  0  0  0  0  0  |
    *    | 20  0  0  0  0  0  0  0  0  0  |
    *    | 21  0  0  0  0  0  0  0  0  0  |
    *    | 22 23 24 25 26 27 28 29 30 31  |
    *    \ 32 33 34 35 36 37 38 39 40 41 /
    */

   const IndexType rows = 10;
   const IndexType cols = 10;

   Matrix m( rows, cols );

   typename Matrix::RowsCapacitiesType rowLengths{ 4, 3, 8, 2, 1, 1, 1, 1, 10, 10 };
   m.setRowCapacities( rowLengths );

   auto matrixView = m.getView();
   auto f = [=] __cuda_callable__ ( const IndexType rowIdx ) mutable {
      auto row = matrixView.getRow( rowIdx );
      RealType val;
      switch( rowIdx )
      {
         case 0:
            val = 1;
            for( IndexType i = 0; i < 4; i++ )
               row.setElement( i, 2 * i, val++ );
            break;
         case 1:
            val = 5;
            for( IndexType i = 0; i < 3; i++ )
               row.setElement( i, i, val++ );
            break;
         case 2:
            val = 8;
            for( IndexType i = 0; i < 8; i++ )
               row.setElement( i, i, val++ );
            break;
         case 3:
            val = 16;
            for( IndexType i = 0; i < 2; i++ )
               row.setElement( i, i, val++ );
            break;
         case 4:
            row.setElement( 0, 0, 18 );
            break;
         case 5:
            row.setElement( 0, 0, 19 );
            break;
         case 6:
            row.setElement( 0, 0, 20 );
            break;
         case 7:
            row.setElement( 0, 0, 21 );
            break;
         case 8:
            val = 22;
            for( IndexType i = 0; i < rows; i++ )
               row.setElement( i, i, val++ );
            break;
         case 9:
            val = 32;
            for( IndexType i = 0; i < rows; i++ )
               row.setElement( i, i, val++ );
            break;
      }
   };
   TNL::Algorithms::ParallelFor< DeviceType >::exec( ( IndexType ) 0, rows, f );

   EXPECT_EQ( m.getElement( 0, 0 ),  1 );
   EXPECT_EQ( m.getElement( 0, 1 ),  0 );
   EXPECT_EQ( m.getElement( 0, 2 ),  2 );
   EXPECT_EQ( m.getElement( 0, 3 ),  0 );
   EXPECT_EQ( m.getElement( 0, 4 ),  3 );
   EXPECT_EQ( m.getElement( 0, 5 ),  0 );
   EXPECT_EQ( m.getElement( 0, 6 ),  4 );
   EXPECT_EQ( m.getElement( 0, 7 ),  0 );
   EXPECT_EQ( m.getElement( 0, 8 ),  0 );
   EXPECT_EQ( m.getElement( 0, 9 ),  0 );

   EXPECT_EQ( m.getElement( 1, 0 ),  5 );
   EXPECT_EQ( m.getElement( 1, 1 ),  6 );
   EXPECT_EQ( m.getElement( 1, 2 ),  7 );
   EXPECT_EQ( m.getElement( 1, 3 ),  0 );
   EXPECT_EQ( m.getElement( 1, 4 ),  0 );
   EXPECT_EQ( m.getElement( 1, 5 ),  0 );
   EXPECT_EQ( m.getElement( 1, 6 ),  0 );
   EXPECT_EQ( m.getElement( 1, 7 ),  0 );
   EXPECT_EQ( m.getElement( 1, 8 ),  0 );
   EXPECT_EQ( m.getElement( 1, 9 ),  0 );

   EXPECT_EQ( m.getElement( 2, 0 ),  8 );
   EXPECT_EQ( m.getElement( 2, 1 ),  9 );
   EXPECT_EQ( m.getElement( 2, 2 ), 10 );
   EXPECT_EQ( m.getElement( 2, 3 ), 11 );
   EXPECT_EQ( m.getElement( 2, 4 ), 12 );
   EXPECT_EQ( m.getElement( 2, 5 ), 13 );
   EXPECT_EQ( m.getElement( 2, 6 ), 14 );
   EXPECT_EQ( m.getElement( 2, 7 ), 15 );
   EXPECT_EQ( m.getElement( 2, 8 ),  0 );
   EXPECT_EQ( m.getElement( 2, 9 ),  0 );

   EXPECT_EQ( m.getElement( 3, 0 ), 16 );
   EXPECT_EQ( m.getElement( 3, 1 ), 17 );
   EXPECT_EQ( m.getElement( 3, 2 ),  0 );
   EXPECT_EQ( m.getElement( 3, 3 ),  0 );
   EXPECT_EQ( m.getElement( 3, 4 ),  0 );
   EXPECT_EQ( m.getElement( 3, 5 ),  0 );
   EXPECT_EQ( m.getElement( 3, 6 ),  0 );
   EXPECT_EQ( m.getElement( 3, 7 ),  0 );
   EXPECT_EQ( m.getElement( 3, 8 ),  0 );
   EXPECT_EQ( m.getElement( 3, 9 ),  0 );

   EXPECT_EQ( m.getElement( 4, 0 ), 18 );
   EXPECT_EQ( m.getElement( 4, 1 ),  0 );
   EXPECT_EQ( m.getElement( 4, 2 ),  0 );
   EXPECT_EQ( m.getElement( 4, 3 ),  0 );
   EXPECT_EQ( m.getElement( 4, 4 ),  0 );
   EXPECT_EQ( m.getElement( 4, 5 ),  0 );
   EXPECT_EQ( m.getElement( 4, 6 ),  0 );
   EXPECT_EQ( m.getElement( 4, 7 ),  0 );
   EXPECT_EQ( m.getElement( 4, 8 ),  0 );
   EXPECT_EQ( m.getElement( 4, 9 ),  0 );

   EXPECT_EQ( m.getElement( 5, 0 ), 19 );
   EXPECT_EQ( m.getElement( 5, 1 ),  0 );
   EXPECT_EQ( m.getElement( 5, 2 ),  0 );
   EXPECT_EQ( m.getElement( 5, 3 ),  0 );
   EXPECT_EQ( m.getElement( 5, 4 ),  0 );
   EXPECT_EQ( m.getElement( 5, 5 ),  0 );
   EXPECT_EQ( m.getElement( 5, 6 ),  0 );
   EXPECT_EQ( m.getElement( 5, 7 ),  0 );
   EXPECT_EQ( m.getElement( 5, 8 ),  0 );
   EXPECT_EQ( m.getElement( 5, 9 ),  0 );

   EXPECT_EQ( m.getElement( 6, 0 ), 20 );
   EXPECT_EQ( m.getElement( 6, 1 ),  0 );
   EXPECT_EQ( m.getElement( 6, 2 ),  0 );
   EXPECT_EQ( m.getElement( 6, 3 ),  0 );
   EXPECT_EQ( m.getElement( 6, 4 ),  0 );
   EXPECT_EQ( m.getElement( 6, 5 ),  0 );
   EXPECT_EQ( m.getElement( 6, 6 ),  0 );
   EXPECT_EQ( m.getElement( 6, 7 ),  0 );
   EXPECT_EQ( m.getElement( 6, 8 ),  0 );
   EXPECT_EQ( m.getElement( 6, 9 ),  0 );

   EXPECT_EQ( m.getElement( 7, 0 ), 21 );
   EXPECT_EQ( m.getElement( 7, 1 ),  0 );
   EXPECT_EQ( m.getElement( 7, 2 ),  0 );
   EXPECT_EQ( m.getElement( 7, 3 ),  0 );
   EXPECT_EQ( m.getElement( 7, 4 ),  0 );
   EXPECT_EQ( m.getElement( 7, 5 ),  0 );
   EXPECT_EQ( m.getElement( 7, 6 ),  0 );
   EXPECT_EQ( m.getElement( 7, 7 ),  0 );
   EXPECT_EQ( m.getElement( 7, 8 ),  0 );
   EXPECT_EQ( m.getElement( 7, 9 ),  0 );

   EXPECT_EQ( m.getElement( 8, 0 ), 22 );
   EXPECT_EQ( m.getElement( 8, 1 ), 23 );
   EXPECT_EQ( m.getElement( 8, 2 ), 24 );
   EXPECT_EQ( m.getElement( 8, 3 ), 25 );
   EXPECT_EQ( m.getElement( 8, 4 ), 26 );
   EXPECT_EQ( m.getElement( 8, 5 ), 27 );
   EXPECT_EQ( m.getElement( 8, 6 ), 28 );
   EXPECT_EQ( m.getElement( 8, 7 ), 29 );
   EXPECT_EQ( m.getElement( 8, 8 ), 30 );
   EXPECT_EQ( m.getElement( 8, 9 ), 31 );

   EXPECT_EQ( m.getElement( 9, 0 ), 32 );
   EXPECT_EQ( m.getElement( 9, 1 ), 33 );
   EXPECT_EQ( m.getElement( 9, 2 ), 34 );
   EXPECT_EQ( m.getElement( 9, 3 ), 35 );
   EXPECT_EQ( m.getElement( 9, 4 ), 36 );
   EXPECT_EQ( m.getElement( 9, 5 ), 37 );
   EXPECT_EQ( m.getElement( 9, 6 ), 38 );
   EXPECT_EQ( m.getElement( 9, 7 ), 39 );
   EXPECT_EQ( m.getElement( 9, 8 ), 40 );
   EXPECT_EQ( m.getElement( 9, 9 ), 41 );
}


template< typename Matrix >
void test_SetElement()
{
   using RealType = typename Matrix::RealType;
   using DeviceType = typename Matrix::DeviceType;
   using IndexType = typename Matrix::IndexType;

   /*
    * Sets up the following 10x10 sparse matrix:
    *
    *    /  1  0  2  0  3  0  4  0  0  0  \
    *    |  5  6  7  0  0  0  0  0  0  0  |
    *    |  8  9 10 11 12 13 14 15  0  0  |
    *    | 16 17  0  0  0  0  0  0  0  0  |
    *    | 18  0  0  0  0  0  0  0  0  0  |
    *    | 19  0  0  0  0  0  0  0  0  0  |
    *    | 20  0  0  0  0  0  0  0  0  0  |
    *    | 21  0  0  0  0  0  0  0  0  0  |
    *    | 22 23 24 25 26 27 28 29 30 31  |
    *    \ 32 33 34 35 36 37 38 39 40 41 /
    */

   const IndexType rows = 10;
   const IndexType cols = 10;

   Matrix m;
   m.reset();

   m.setDimensions( rows, cols );

   typename Matrix::RowsCapacitiesType rowLengths { 4, 3, 8, 2, 1, 1, 1, 1, 10, 10 };
   m.setRowCapacities( rowLengths );

   RealType value = 1;
   for( IndexType i = 0; i < 4; i++ )
      m.setElement( 0, 2 * i, value++ );

   for( IndexType i = 0; i < 3; i++ )
      m.setElement( 1, i, value++ );

   for( IndexType i = 0; i < 8; i++ )
      m.setElement( 2, i, value++ );

   for( IndexType i = 0; i < 2; i++ )
      m.setElement( 3, i, value++ );

   for( IndexType i = 4; i < 8; i++ )
      m.setElement( i, 0, value++ );

   for( IndexType j = 8; j < rows; j++)
      for( IndexType i = 0; i < cols; i++ )
         m.setElement( j, i, value++ );

   EXPECT_EQ( m.getElement( 0, 0 ),  1 );
   EXPECT_EQ( m.getElement( 0, 1 ),  0 );
   EXPECT_EQ( m.getElement( 0, 2 ),  2 );
   EXPECT_EQ( m.getElement( 0, 3 ),  0 );
   EXPECT_EQ( m.getElement( 0, 4 ),  3 );
   EXPECT_EQ( m.getElement( 0, 5 ),  0 );
   EXPECT_EQ( m.getElement( 0, 6 ),  4 );
   EXPECT_EQ( m.getElement( 0, 7 ),  0 );
   EXPECT_EQ( m.getElement( 0, 8 ),  0 );
   EXPECT_EQ( m.getElement( 0, 9 ),  0 );

   EXPECT_EQ( m.getElement( 1, 0 ),  5 );
   EXPECT_EQ( m.getElement( 1, 1 ),  6 );
   EXPECT_EQ( m.getElement( 1, 2 ),  7 );
   EXPECT_EQ( m.getElement( 1, 3 ),  0 );
   EXPECT_EQ( m.getElement( 1, 4 ),  0 );
   EXPECT_EQ( m.getElement( 1, 5 ),  0 );
   EXPECT_EQ( m.getElement( 1, 6 ),  0 );
   EXPECT_EQ( m.getElement( 1, 7 ),  0 );
   EXPECT_EQ( m.getElement( 1, 8 ),  0 );
   EXPECT_EQ( m.getElement( 1, 9 ),  0 );

   EXPECT_EQ( m.getElement( 2, 0 ),  8 );
   EXPECT_EQ( m.getElement( 2, 1 ),  9 );
   EXPECT_EQ( m.getElement( 2, 2 ), 10 );
   EXPECT_EQ( m.getElement( 2, 3 ), 11 );
   EXPECT_EQ( m.getElement( 2, 4 ), 12 );
   EXPECT_EQ( m.getElement( 2, 5 ), 13 );
   EXPECT_EQ( m.getElement( 2, 6 ), 14 );
   EXPECT_EQ( m.getElement( 2, 7 ), 15 );
   EXPECT_EQ( m.getElement( 2, 8 ),  0 );
   EXPECT_EQ( m.getElement( 2, 9 ),  0 );

   EXPECT_EQ( m.getElement( 3, 0 ), 16 );
   EXPECT_EQ( m.getElement( 3, 1 ), 17 );
   EXPECT_EQ( m.getElement( 3, 2 ),  0 );
   EXPECT_EQ( m.getElement( 3, 3 ),  0 );
   EXPECT_EQ( m.getElement( 3, 4 ),  0 );
   EXPECT_EQ( m.getElement( 3, 5 ),  0 );
   EXPECT_EQ( m.getElement( 3, 6 ),  0 );
   EXPECT_EQ( m.getElement( 3, 7 ),  0 );
   EXPECT_EQ( m.getElement( 3, 8 ),  0 );
   EXPECT_EQ( m.getElement( 3, 9 ),  0 );

   EXPECT_EQ( m.getElement( 4, 0 ), 18 );
   EXPECT_EQ( m.getElement( 4, 1 ),  0 );
   EXPECT_EQ( m.getElement( 4, 2 ),  0 );
   EXPECT_EQ( m.getElement( 4, 3 ),  0 );
   EXPECT_EQ( m.getElement( 4, 4 ),  0 );
   EXPECT_EQ( m.getElement( 4, 5 ),  0 );
   EXPECT_EQ( m.getElement( 4, 6 ),  0 );
   EXPECT_EQ( m.getElement( 4, 7 ),  0 );
   EXPECT_EQ( m.getElement( 4, 8 ),  0 );
   EXPECT_EQ( m.getElement( 4, 9 ),  0 );

   EXPECT_EQ( m.getElement( 5, 0 ), 19 );
   EXPECT_EQ( m.getElement( 5, 1 ),  0 );
   EXPECT_EQ( m.getElement( 5, 2 ),  0 );
   EXPECT_EQ( m.getElement( 5, 3 ),  0 );
   EXPECT_EQ( m.getElement( 5, 4 ),  0 );
   EXPECT_EQ( m.getElement( 5, 5 ),  0 );
   EXPECT_EQ( m.getElement( 5, 6 ),  0 );
   EXPECT_EQ( m.getElement( 5, 7 ),  0 );
   EXPECT_EQ( m.getElement( 5, 8 ),  0 );
   EXPECT_EQ( m.getElement( 5, 9 ),  0 );

   EXPECT_EQ( m.getElement( 6, 0 ), 20 );
   EXPECT_EQ( m.getElement( 6, 1 ),  0 );
   EXPECT_EQ( m.getElement( 6, 2 ),  0 );
   EXPECT_EQ( m.getElement( 6, 3 ),  0 );
   EXPECT_EQ( m.getElement( 6, 4 ),  0 );
   EXPECT_EQ( m.getElement( 6, 5 ),  0 );
   EXPECT_EQ( m.getElement( 6, 6 ),  0 );
   EXPECT_EQ( m.getElement( 6, 7 ),  0 );
   EXPECT_EQ( m.getElement( 6, 8 ),  0 );
   EXPECT_EQ( m.getElement( 6, 9 ),  0 );

   EXPECT_EQ( m.getElement( 7, 0 ), 21 );
   EXPECT_EQ( m.getElement( 7, 1 ),  0 );
   EXPECT_EQ( m.getElement( 7, 2 ),  0 );
   EXPECT_EQ( m.getElement( 7, 3 ),  0 );
   EXPECT_EQ( m.getElement( 7, 4 ),  0 );
   EXPECT_EQ( m.getElement( 7, 5 ),  0 );
   EXPECT_EQ( m.getElement( 7, 6 ),  0 );
   EXPECT_EQ( m.getElement( 7, 7 ),  0 );
   EXPECT_EQ( m.getElement( 7, 8 ),  0 );
   EXPECT_EQ( m.getElement( 7, 9 ),  0 );

   EXPECT_EQ( m.getElement( 8, 0 ), 22 );
   EXPECT_EQ( m.getElement( 8, 1 ), 23 );
   EXPECT_EQ( m.getElement( 8, 2 ), 24 );
   EXPECT_EQ( m.getElement( 8, 3 ), 25 );
   EXPECT_EQ( m.getElement( 8, 4 ), 26 );
   EXPECT_EQ( m.getElement( 8, 5 ), 27 );
   EXPECT_EQ( m.getElement( 8, 6 ), 28 );
   EXPECT_EQ( m.getElement( 8, 7 ), 29 );
   EXPECT_EQ( m.getElement( 8, 8 ), 30 );
   EXPECT_EQ( m.getElement( 8, 9 ), 31 );

   EXPECT_EQ( m.getElement( 9, 0 ), 32 );
   EXPECT_EQ( m.getElement( 9, 1 ), 33 );
   EXPECT_EQ( m.getElement( 9, 2 ), 34 );
   EXPECT_EQ( m.getElement( 9, 3 ), 35 );
   EXPECT_EQ( m.getElement( 9, 4 ), 36 );
   EXPECT_EQ( m.getElement( 9, 5 ), 37 );
   EXPECT_EQ( m.getElement( 9, 6 ), 38 );
   EXPECT_EQ( m.getElement( 9, 7 ), 39 );
   EXPECT_EQ( m.getElement( 9, 8 ), 40 );
   EXPECT_EQ( m.getElement( 9, 9 ), 41 );
}

template< typename Matrix >
void test_AddElement()
{
   using RealType = typename Matrix::RealType;
   using DeviceType = typename Matrix::DeviceType;
   using IndexType = typename Matrix::IndexType;

   /*
    * Sets up the following 6x5 sparse matrix:
    *
    *    /  1  2  3  0  0 \
    *    |  0  4  5  6  0 |
    *    |  0  0  7  8  9 |
    *    | 10  1  1  0  0 |
    *    |  0 11  1  1  0 |
    *    \  0  0  1 12  1 /
    */

   const IndexType rows = 6;
   const IndexType cols = 5;

   Matrix m( rows, cols, {
      { 0, 0,  1 }, { 0, 1,  2 }, { 0, 2, 3 },
                    { 1, 1,  4 }, { 1, 2, 5 }, { 1, 3,  6 },
                                  { 2, 2, 7 }, { 2, 3,  8 }, { 2, 4, 9 },
      { 3, 0, 10 }, { 3, 1,  1 }, { 3, 2, 1 },
                    { 4, 1, 11 }, { 4, 2, 1 }, { 4, 3,  1 },
                                  { 5, 2, 1 }, { 5, 3, 12 }, { 5, 4, 1 } } );
   /*typename Matrix::RowsCapacitiesType rowLengths( rows, 3 );
   m.setRowCapacities( rowLengths );

   RealType value = 1;
   for( IndexType i = 0; i < cols - 2; i++ )     // 0th row
      m.setElement( 0, i, value++ );

   for( IndexType i = 1; i < cols - 1; i++ )     // 1st row
      m.setElement( 1, i, value++ );

   for( IndexType i = 2; i < cols; i++ )         // 2nd row
      m.setElement( 2, i, value++ );

   m.setElement( 3, 0, value++ );      // 3rd row

   m.setElement( 4, 1, value++ );      // 4th row

   m.setElement( 5, 3, value++ );      // 5th row*/


   // Check the set elements
   EXPECT_EQ( m.getElement( 0, 0 ),  1 );
   EXPECT_EQ( m.getElement( 0, 1 ),  2 );
   EXPECT_EQ( m.getElement( 0, 2 ),  3 );
   EXPECT_EQ( m.getElement( 0, 3 ),  0 );
   EXPECT_EQ( m.getElement( 0, 4 ),  0 );

   EXPECT_EQ( m.getElement( 1, 0 ),  0 );
   EXPECT_EQ( m.getElement( 1, 1 ),  4 );
   EXPECT_EQ( m.getElement( 1, 2 ),  5 );
   EXPECT_EQ( m.getElement( 1, 3 ),  6 );
   EXPECT_EQ( m.getElement( 1, 4 ),  0 );

   EXPECT_EQ( m.getElement( 2, 0 ),  0 );
   EXPECT_EQ( m.getElement( 2, 1 ),  0 );
   EXPECT_EQ( m.getElement( 2, 2 ),  7 );
   EXPECT_EQ( m.getElement( 2, 3 ),  8 );
   EXPECT_EQ( m.getElement( 2, 4 ),  9 );

   EXPECT_EQ( m.getElement( 3, 0 ), 10 );
   EXPECT_EQ( m.getElement( 3, 1 ),  1 );
   EXPECT_EQ( m.getElement( 3, 2 ),  1 );
   EXPECT_EQ( m.getElement( 3, 3 ),  0 );
   EXPECT_EQ( m.getElement( 3, 4 ),  0 );

   EXPECT_EQ( m.getElement( 4, 0 ),  0 );
   EXPECT_EQ( m.getElement( 4, 1 ), 11 );
   EXPECT_EQ( m.getElement( 4, 2 ),  1 );
   EXPECT_EQ( m.getElement( 4, 3 ),  1 );
   EXPECT_EQ( m.getElement( 4, 4 ),  0 );

   EXPECT_EQ( m.getElement( 5, 0 ),  0 );
   EXPECT_EQ( m.getElement( 5, 1 ),  0 );
   EXPECT_EQ( m.getElement( 5, 2 ),  1 );
   EXPECT_EQ( m.getElement( 5, 3 ), 12 );
   EXPECT_EQ( m.getElement( 5, 4 ),  1 );

   // Add new elements to the old elements with a multiplying factor applied to the old elements.
   /*
    * The following setup results in the following 6x5 sparse matrix:
    *
    *    /  3  6  9  0  0 \
    *    |  0 12 15 18  0 |
    *    |  0  0 21 24 27 |
    *    | 30 13 14  0  0 |
    *    |  0 35 16 17  0 |
    *    \  0  0 18 41 20 /
    */

   RealType newValue = 1;
   for( IndexType i = 0; i < cols - 2; i++ )         // 0th row
      m.addElement( 0, i, newValue++, 2.0 );

   for( IndexType i = 1; i < cols - 1; i++ )         // 1st row
      m.addElement( 1, i, newValue++, 2.0 );

   for( IndexType i = 2; i < cols; i++ )             // 2nd row
      m.addElement( 2, i, newValue++, 2.0 );

   for( IndexType i = 0; i < cols - 2; i++ )         // 3rd row
      m.addElement( 3, i, newValue++, 2.0 );

   for( IndexType i = 1; i < cols - 1; i++ )         // 4th row
      m.addElement( 4, i, newValue++, 2.0 );

   for( IndexType i = 2; i < cols; i++ )             // 5th row
      m.addElement( 5, i, newValue++, 2.0 );


   EXPECT_EQ( m.getElement( 0, 0 ),  3 );
   EXPECT_EQ( m.getElement( 0, 1 ),  6 );
   EXPECT_EQ( m.getElement( 0, 2 ),  9 );
   EXPECT_EQ( m.getElement( 0, 3 ),  0 );
   EXPECT_EQ( m.getElement( 0, 4 ),  0 );

   EXPECT_EQ( m.getElement( 1, 0 ),  0 );
   EXPECT_EQ( m.getElement( 1, 1 ), 12 );
   EXPECT_EQ( m.getElement( 1, 2 ), 15 );
   EXPECT_EQ( m.getElement( 1, 3 ), 18 );
   EXPECT_EQ( m.getElement( 1, 4 ),  0 );

   EXPECT_EQ( m.getElement( 2, 0 ),  0 );
   EXPECT_EQ( m.getElement( 2, 1 ),  0 );
   EXPECT_EQ( m.getElement( 2, 2 ), 21 );
   EXPECT_EQ( m.getElement( 2, 3 ), 24 );
   EXPECT_EQ( m.getElement( 2, 4 ), 27 );

   EXPECT_EQ( m.getElement( 3, 0 ), 30 );
   EXPECT_EQ( m.getElement( 3, 1 ), 13 );
   EXPECT_EQ( m.getElement( 3, 2 ), 14 );
   EXPECT_EQ( m.getElement( 3, 3 ),  0 );
   EXPECT_EQ( m.getElement( 3, 4 ),  0 );

   EXPECT_EQ( m.getElement( 4, 0 ),  0 );
   EXPECT_EQ( m.getElement( 4, 1 ), 35 );
   EXPECT_EQ( m.getElement( 4, 2 ), 16 );
   EXPECT_EQ( m.getElement( 4, 3 ), 17 );
   EXPECT_EQ( m.getElement( 4, 4 ),  0 );

   EXPECT_EQ( m.getElement( 5, 0 ),  0 );
   EXPECT_EQ( m.getElement( 5, 1 ),  0 );
   EXPECT_EQ( m.getElement( 5, 2 ), 18 );
   EXPECT_EQ( m.getElement( 5, 3 ), 41 );
   EXPECT_EQ( m.getElement( 5, 4 ), 20 );
}

template< typename Matrix >
void test_ForElements()
{
   using RealType = typename Matrix::RealType;
   using DeviceType = typename Matrix::DeviceType;
   using IndexType = typename Matrix::IndexType;

   /*
    * Sets up the following 8x3 sparse matrix:
    *
    *    /  1  1  1  \
    *    |  2  2  2  |
    *    |  3  3  3  |
    *    |  4  4  4  |
    *    |  5  5  5  |
    *    |  6  6  6  |
    *    |  7  7  7  |
    *    \  8  8  8  /
    */

   const IndexType cols = 3;
   const IndexType rows = 8;

   Matrix m( { 3, 3, 3, 3, 3, 3, 3, 3, 3 }, cols  );
   m.forAllElements( [] __cuda_callable__ ( IndexType rowIdx, IndexType localIdx, IndexType& columnIdx, RealType& value ) mutable {
      value = rowIdx + 1.0;
      columnIdx = localIdx;
   } );

   for( IndexType rowIdx = 0; rowIdx < rows; rowIdx++ )
      for( IndexType colIdx = 0; colIdx < cols; colIdx++ )
         EXPECT_EQ( m.getElement( rowIdx, colIdx ), rowIdx + 1.0 );
}

template< typename Matrix >
void test_ForRows()
{
   using RealType = typename Matrix::RealType;
   using DeviceType = typename Matrix::DeviceType;
   using IndexType = typename Matrix::IndexType;

   /////
   // Setup lower triangular matrix
   const IndexType cols = 8;
   const IndexType rows = 8;

   /////
   // Test without iterator
   Matrix m( { 1, 2, 3, 4, 5, 6, 7, 8, 9 }, cols  );
   using RowView = typename Matrix::RowView;
   m.forAllRows( [] __cuda_callable__ ( RowView& row ) mutable {
      for( IndexType localIdx = 0; localIdx <= row.getRowIndex(); localIdx++ )
      {
         row.setValue( localIdx, row.getRowIndex() - localIdx + 1.0 );
         row.setColumnIndex( localIdx, localIdx );
      }
   } );

   for( IndexType rowIdx = 0; rowIdx < rows; rowIdx++ )
      for( IndexType colIdx = 0; colIdx < cols; colIdx++ )
      {
         if( colIdx <= rowIdx )
            EXPECT_EQ( m.getElement( rowIdx, colIdx ), rowIdx - colIdx + 1.0 );
         else
            EXPECT_EQ( m.getElement( rowIdx, colIdx ), 0.0 );
      }

   ////
   // Test with iterator
   m.getValues() = 0.0;
   m.forAllRows( [] __cuda_callable__ ( RowView& row ) mutable {
      for( auto element : row )
      {
         if( element.localIndex() <= element.rowIndex() )
         {
            element.value() = element.rowIndex() - element.localIndex() + 1.0;
            element.columnIndex() = element.localIndex();
         }
      }
   } );

   for( IndexType rowIdx = 0; rowIdx < rows; rowIdx++ )
      for( IndexType colIdx = 0; colIdx < cols; colIdx++ )
      {
         if( colIdx <= rowIdx )
            EXPECT_EQ( m.getElement( rowIdx, colIdx ), rowIdx - colIdx + 1.0 );
         else
            EXPECT_EQ( m.getElement( rowIdx, colIdx ), 0.0 );
      }
}

template< typename Matrix >
void test_reduceRows()
{
   using RealType = typename Matrix::RealType;
   using DeviceType = typename Matrix::DeviceType;
   using IndexType = typename Matrix::IndexType;

   /*
    * Sets up the following 8x8 sparse matrix:
    *
    *    /  1  2  3  0  4  5  0  1 \   6
    *    |  0  6  0  7  0  0  0  1 |   3
    *    |  0  8  9  0 10  0  0  1 |   4
    *    |  0 11 12 13 14  0  0  1 |   5
    *    |  0 15  0  0  0  0  0  1 |   2
    *    |  0 16 17 18 19 20 21  1 |   7
    *    | 22 23 24 25 26 27 28  1 |   8
    *    \ 29 30 31 32 33 34 35 36 /   8
    */

   const IndexType rows = 8;
   const IndexType cols = 8;

   Matrix m( rows, cols );
   typename Matrix::RowsCapacitiesType rowsCapacities{ 6, 3, 4, 5, 2, 7, 8, 8 };
   m.setRowCapacities( rowsCapacities );

   RealType value = 1;
   for( IndexType i = 0; i < 3; i++ )   // 0th row
      m.setElement( 0, i, value++ );

   m.setElement( 0, 4, value++ );       // 0th row
   m.setElement( 0, 5, value++ );

   m.setElement( 1, 1, value++ );       // 1st row
   m.setElement( 1, 3, value++ );

   for( IndexType i = 1; i < 3; i++ )   // 2nd row
      m.setElement( 2, i, value++ );

   m.setElement( 2, 4, value++ );       // 2nd row

   for( IndexType i = 1; i < 5; i++ )   // 3rd row
      m.setElement( 3, i, value++ );

   m.setElement( 4, 1, value++ );       // 4th row

   for( IndexType i = 1; i < 7; i++ )   // 5th row
      m.setElement( 5, i, value++ );

   for( IndexType i = 0; i < 7; i++ )   // 6th row
      m.setElement( 6, i, value++ );

   for( IndexType i = 0; i < 8; i++ )   // 7th row
       m.setElement( 7, i, value++ );

   for( IndexType i = 0; i < 7; i++ )   // 1s at the end of rows
      m.setElement( i, 7, 1);

   ////
   // Compute number of non-zero elements in rows.
   typename Matrix::RowsCapacitiesType rowLengths( rows );
   auto rowLengths_view = rowLengths.getView();
   auto fetch = [] __cuda_callable__ ( IndexType row, IndexType column, const RealType& value ) -> IndexType {
      return ( value != 0.0 );
   };
   auto keep = [=] __cuda_callable__ ( const IndexType rowIdx, const IndexType value ) mutable {
      rowLengths_view[ rowIdx ] = value;
   };
   m.reduceAllRows( fetch, std::plus<>{}, keep, 0 );
   EXPECT_EQ( rowsCapacities, rowLengths );
   m.getCompressedRowLengths( rowLengths );
   EXPECT_EQ( rowsCapacities, rowLengths );

   ////
   // Compute max norm
   TNL::Containers::Vector< RealType, DeviceType, IndexType > rowSums( rows );
   auto rowSums_view = rowSums.getView();
   auto max_fetch = [] __cuda_callable__ ( IndexType row, IndexType column, const RealType& value ) -> IndexType {
      return TNL::abs( value );
   };
   auto max_keep = [=] __cuda_callable__ ( const IndexType rowIdx, const IndexType value ) mutable {
      rowSums_view[ rowIdx ] = value;
   };
   m.reduceAllRows( max_fetch, std::plus<>{}, max_keep, 0 );
   const RealType maxNorm = TNL::max( rowSums );
   EXPECT_EQ( maxNorm, 260 ) ; // 29+30+31+32+33+34+35+36
}

template< typename Matrix >
void test_PerformSORIteration()
{
   using RealType = typename Matrix::RealType;
   using DeviceType = typename Matrix::DeviceType;
   using IndexType = typename Matrix::IndexType;

   /*
    * Sets up the following 4x4 sparse matrix:
    *
    *    /  4  1  0  0 \
    *    |  1  4  1  0 |
    *    |  0  1  4  1 |
    *    \  0  0  1  4 /
    */

   const IndexType m_rows = 4;
   const IndexType m_cols = 4;

   Matrix m( m_rows, m_cols );
   typename Matrix::RowsCapacitiesType rowLengths( m_rows, 3 );
   m.setRowCapacities( rowLengths );

   m.setElement( 0, 0, 4.0 );        // 0th row
   m.setElement( 0, 1, 1.0);

   m.setElement( 1, 0, 1.0 );        // 1st row
   m.setElement( 1, 1, 4.0 );
   m.setElement( 1, 2, 1.0 );

   m.setElement( 2, 1, 1.0 );        // 2nd row
   m.setElement( 2, 2, 4.0 );
   m.setElement( 2, 3, 1.0 );

   m.setElement( 3, 2, 1.0 );        // 3rd row
   m.setElement( 3, 3, 4.0 );

   RealType bVector [ 4 ] = { 1, 1, 1, 1 };
   RealType xVector [ 4 ] = { 1, 1, 1, 1 };

   IndexType row = 0;
   RealType omega = 1;

   m.performSORIteration( bVector, row++, xVector, omega);

   EXPECT_EQ( xVector[ 0 ], 0.0 );
   EXPECT_EQ( xVector[ 1 ], 1.0 );
   EXPECT_EQ( xVector[ 2 ], 1.0 );
   EXPECT_EQ( xVector[ 3 ], 1.0 );

   m.performSORIteration( bVector, row++, xVector, omega);

   EXPECT_EQ( xVector[ 0 ], 0.0 );
   EXPECT_EQ( xVector[ 1 ], 0.0 );
   EXPECT_EQ( xVector[ 2 ], 1.0 );
   EXPECT_EQ( xVector[ 3 ], 1.0 );

   m.performSORIteration( bVector, row++, xVector, omega);

   EXPECT_EQ( xVector[ 0 ], 0.0 );
   EXPECT_EQ( xVector[ 1 ], 0.0 );
   EXPECT_EQ( xVector[ 2 ], 0.0 );
   EXPECT_EQ( xVector[ 3 ], 1.0 );

   m.performSORIteration( bVector, row++, xVector, omega);

   EXPECT_EQ( xVector[ 0 ], 0.0 );
   EXPECT_EQ( xVector[ 1 ], 0.0 );
   EXPECT_EQ( xVector[ 2 ], 0.0 );
   EXPECT_EQ( xVector[ 3 ], 0.25 );
}

template< typename Matrix >
void test_SaveAndLoad( const char* filename )
{
   using RealType = typename Matrix::RealType;
   using DeviceType = typename Matrix::DeviceType;
   using IndexType = typename Matrix::IndexType;

   /*
    * Sets up the following 4x4 sparse matrix:
    *
    *    /  1  2  3  0 \
    *    |  0  4  0  5 |
    *    |  6  7  8  0 |
    *    \  0  9 10 11 /
    */

   const IndexType m_rows = 4;
   const IndexType m_cols = 4;

   Matrix savedMatrix( m_rows, m_cols );
   typename Matrix::RowsCapacitiesType rowLengths( m_rows, 3 );
   savedMatrix.setRowCapacities( rowLengths );

   RealType value = 1;
   for( IndexType i = 0; i < m_cols - 1; i++ )   // 0th row
      savedMatrix.setElement( 0, i, value++ );

   savedMatrix.setElement( 1, 1, value++ );
   savedMatrix.setElement( 1, 3, value++ );      // 1st row

   for( IndexType i = 0; i < m_cols - 1; i++ )   // 2nd row
      savedMatrix.setElement( 2, i, value++ );

   for( IndexType i = 1; i < m_cols; i++ )       // 3rd row
      savedMatrix.setElement( 3, i, value++ );

   ASSERT_NO_THROW( savedMatrix.save( filename ) );

   Matrix loadedMatrix;

   ASSERT_NO_THROW( loadedMatrix.load( filename ) );

   EXPECT_EQ( savedMatrix.getElement( 0, 0 ), loadedMatrix.getElement( 0, 0 ) );
   EXPECT_EQ( savedMatrix.getElement( 0, 1 ), loadedMatrix.getElement( 0, 1 ) );
   EXPECT_EQ( savedMatrix.getElement( 0, 2 ), loadedMatrix.getElement( 0, 2 ) );
   EXPECT_EQ( savedMatrix.getElement( 0, 3 ), loadedMatrix.getElement( 0, 3 ) );

   EXPECT_EQ( savedMatrix.getElement( 1, 0 ), loadedMatrix.getElement( 1, 0 ) );
   EXPECT_EQ( savedMatrix.getElement( 1, 1 ), loadedMatrix.getElement( 1, 1 ) );
   EXPECT_EQ( savedMatrix.getElement( 1, 2 ), loadedMatrix.getElement( 1, 2 ) );
   EXPECT_EQ( savedMatrix.getElement( 1, 3 ), loadedMatrix.getElement( 1, 3 ) );

   EXPECT_EQ( savedMatrix.getElement( 2, 0 ), loadedMatrix.getElement( 2, 0 ) );
   EXPECT_EQ( savedMatrix.getElement( 2, 1 ), loadedMatrix.getElement( 2, 1 ) );
   EXPECT_EQ( savedMatrix.getElement( 2, 2 ), loadedMatrix.getElement( 2, 2 ) );
   EXPECT_EQ( savedMatrix.getElement( 2, 3 ), loadedMatrix.getElement( 2, 3 ) );

   EXPECT_EQ( savedMatrix.getElement( 3, 0 ), loadedMatrix.getElement( 3, 0 ) );
   EXPECT_EQ( savedMatrix.getElement( 3, 1 ), loadedMatrix.getElement( 3, 1 ) );
   EXPECT_EQ( savedMatrix.getElement( 3, 2 ), loadedMatrix.getElement( 3, 2 ) );
   EXPECT_EQ( savedMatrix.getElement( 3, 3 ), loadedMatrix.getElement( 3, 3 ) );

   EXPECT_EQ( savedMatrix.getElement( 0, 0 ),  1 );
   EXPECT_EQ( savedMatrix.getElement( 0, 1 ),  2 );
   EXPECT_EQ( savedMatrix.getElement( 0, 2 ),  3 );
   EXPECT_EQ( savedMatrix.getElement( 0, 3 ),  0 );

   EXPECT_EQ( savedMatrix.getElement( 1, 0 ),  0 );
   EXPECT_EQ( savedMatrix.getElement( 1, 1 ),  4 );
   EXPECT_EQ( savedMatrix.getElement( 1, 2 ),  0 );
   EXPECT_EQ( savedMatrix.getElement( 1, 3 ),  5 );

   EXPECT_EQ( savedMatrix.getElement( 2, 0 ),  6 );
   EXPECT_EQ( savedMatrix.getElement( 2, 1 ),  7 );
   EXPECT_EQ( savedMatrix.getElement( 2, 2 ),  8 );
   EXPECT_EQ( savedMatrix.getElement( 2, 3 ),  0 );

   EXPECT_EQ( savedMatrix.getElement( 3, 0 ),  0 );
   EXPECT_EQ( savedMatrix.getElement( 3, 1 ),  9 );
   EXPECT_EQ( savedMatrix.getElement( 3, 2 ), 10 );
   EXPECT_EQ( savedMatrix.getElement( 3, 3 ), 11 );

   EXPECT_EQ( std::remove( filename ), 0 );
}

#endif
