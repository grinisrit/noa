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

template< typename Matrix >
void test_VectorProduct_smallMatrix1()
{
   using RealType = typename Matrix::RealType;
   using DeviceType = typename Matrix::DeviceType;
   using IndexType = typename Matrix::IndexType;
   using VectorType = TNL::Containers::Vector< RealType, DeviceType, IndexType >;

   /*
    * Sets up the following 4x4 sparse matrix:
    *
    *    /  1  0  0  0 \
    *    |  0  2  0  3 |
    *    |  0  4  0  0 |
    *    \  0  0  5  0 /
    */

   const IndexType m_rows_1 = 4;
   const IndexType m_cols_1 = 4;

   Matrix m_1;
   m_1.reset();
   m_1.setDimensions( m_rows_1, m_cols_1 );
   typename Matrix::RowsCapacitiesType rowLengths_1{ 1, 2, 1, 1 };
   m_1.setRowCapacities( rowLengths_1 );

   RealType value_1 = 1;
   m_1.setElement( 0, 0, value_1++ );      // 0th row

   m_1.setElement( 1, 1, value_1++ );      // 1st row
   m_1.setElement( 1, 3, value_1++ );

   m_1.setElement( 2, 1, value_1++ );      // 2nd row

   m_1.setElement( 3, 2, value_1++ );      // 3rd row

   VectorType inVector_1;
   inVector_1.setSize( m_cols_1 );
   for( IndexType i = 0; i < inVector_1.getSize(); i++ )
       inVector_1.setElement( i, 2 );

   VectorType outVector_1;
   outVector_1.setSize( m_rows_1 );
   for( IndexType j = 0; j < outVector_1.getSize(); j++ )
       outVector_1.setElement( j, 0 );

   m_1.vectorProduct( inVector_1, outVector_1 );
   EXPECT_EQ( outVector_1.getElement( 0 ),  2 );
   EXPECT_EQ( outVector_1.getElement( 1 ), 10 );
   EXPECT_EQ( outVector_1.getElement( 2 ),  8 );
   EXPECT_EQ( outVector_1.getElement( 3 ), 10 );
}

template< typename Matrix >
void test_VectorProduct_smallMatrix2()
{
   using RealType = typename Matrix::RealType;
   using DeviceType = typename Matrix::DeviceType;
   using IndexType = typename Matrix::IndexType;
   using VectorType = TNL::Containers::Vector< RealType, DeviceType, IndexType >;

   /*
    * Sets up the following 4x4 sparse matrix:
    *
    *    /  1  2  3  0 \
    *    |  0  0  0  4 |
    *    |  5  6  7  0 |
    *    \  0  8  0  0 /
    */

   const IndexType m_rows_2 = 4;
   const IndexType m_cols_2 = 4;

   Matrix m_2( m_rows_2, m_cols_2 );
   typename Matrix::RowsCapacitiesType rowLengths_2{ 3, 1, 3, 1 };
   m_2.setRowCapacities( rowLengths_2 );

   RealType value_2 = 1;
   for( IndexType i = 0; i < 3; i++ )      // 0th row
      m_2.setElement( 0, i, value_2++ );

   m_2.setElement( 1, 3, value_2++ );      // 1st row

   for( IndexType i = 0; i < 3; i++ )      // 2nd row
      m_2.setElement( 2, i, value_2++ );

   for( IndexType i = 1; i < 2; i++ )      // 3rd row
      m_2.setElement( 3, i, value_2++ );

   VectorType inVector_2;
   inVector_2.setSize( m_cols_2 );
   for( IndexType i = 0; i < inVector_2.getSize(); i++ )
      inVector_2.setElement( i, 2 );

   VectorType outVector_2;
   outVector_2.setSize( m_rows_2 );
   for( IndexType j = 0; j < outVector_2.getSize(); j++ )
      outVector_2.setElement( j, 0 );

   m_2.vectorProduct( inVector_2, outVector_2 );

   EXPECT_EQ( outVector_2.getElement( 0 ), 12 );
   EXPECT_EQ( outVector_2.getElement( 1 ),  8 );
   EXPECT_EQ( outVector_2.getElement( 2 ), 36 );
   EXPECT_EQ( outVector_2.getElement( 3 ), 16 );
}

template< typename Matrix >
void test_VectorProduct_smallMatrix3()
{
   using RealType = typename Matrix::RealType;
   using DeviceType = typename Matrix::DeviceType;
   using IndexType = typename Matrix::IndexType;
   using VectorType = TNL::Containers::Vector< RealType, DeviceType, IndexType >;

   /*
    * Sets up the following 4x4 sparse matrix:
    *
    *    /  1  2  3  0 \
    *    |  0  4  5  6 |
    *    |  7  8  9  0 |
    *    \  0 10 11 12 /
    */

   const IndexType m_rows_3 = 4;
   const IndexType m_cols_3 = 4;

   Matrix m_3( m_rows_3, m_cols_3 );
   typename Matrix::RowsCapacitiesType rowLengths_3{ 3, 3, 3, 3 };
   m_3.setRowCapacities( rowLengths_3 );

   RealType value_3 = 1;
   for( IndexType i = 0; i < 3; i++ )          // 0th row
      m_3.setElement( 0, i, value_3++ );

   for( IndexType i = 1; i < 4; i++ )
      m_3.setElement( 1, i, value_3++ );      // 1st row

   for( IndexType i = 0; i < 3; i++ )          // 2nd row
      m_3.setElement( 2, i, value_3++ );

   for( IndexType i = 1; i < 4; i++ )          // 3rd row
      m_3.setElement( 3, i, value_3++ );

   VectorType inVector_3;
   inVector_3.setSize( m_cols_3 );
   for( IndexType i = 0; i < inVector_3.getSize(); i++ )
      inVector_3.setElement( i, 2 );

   VectorType outVector_3;
   outVector_3.setSize( m_rows_3 );
   for( IndexType j = 0; j < outVector_3.getSize(); j++ )
      outVector_3.setElement( j, 0 );

   m_3.vectorProduct( inVector_3, outVector_3 );

   EXPECT_EQ( outVector_3.getElement( 0 ), 12 );
   EXPECT_EQ( outVector_3.getElement( 1 ), 30 );
   EXPECT_EQ( outVector_3.getElement( 2 ), 48 );
   EXPECT_EQ( outVector_3.getElement( 3 ), 66 );
}

template< typename Matrix >
void test_VectorProduct_mediumSizeMatrix1()
{
   using RealType = typename Matrix::RealType;
   using DeviceType = typename Matrix::DeviceType;
   using IndexType = typename Matrix::IndexType;
   using VectorType = TNL::Containers::Vector< RealType, DeviceType, IndexType >;

   /*
    * Sets up the following 8x8 sparse matrix:
    *
    *    /  1  2  3  0  0  4  0  0 \
    *    |  0  5  6  7  8  0  0  0 |
    *    |  9 10 11 12 13  0  0  0 |
    *    |  0 14 15 16 17  0  0  0 |
    *    |  0  0 18 19 20 21  0  0 |
    *    |  0  0  0 22 23 24 25  0 |
    *    | 26 27 28 29 30  0  0  0 |
    *    \ 31 32 33 34 35  0  0  0 /
    */

   const IndexType m_rows_4 = 8;
   const IndexType m_cols_4 = 8;

   Matrix m_4( m_rows_4, m_cols_4 );
   typename Matrix::RowsCapacitiesType rowLengths_4{ 4, 4, 5, 4, 4, 4, 5, 5 };
   m_4.setRowCapacities( rowLengths_4 );

   RealType value_4 = 1;
   for( IndexType i = 0; i < 3; i++ )       // 0th row
      m_4.setElement( 0, i, value_4++ );

   m_4.setElement( 0, 5, value_4++ );

   for( IndexType i = 1; i < 5; i++ )       // 1st row
      m_4.setElement( 1, i, value_4++ );

   for( IndexType i = 0; i < 5; i++ )       // 2nd row
      m_4.setElement( 2, i, value_4++ );

   for( IndexType i = 1; i < 5; i++ )       // 3rd row
      m_4.setElement( 3, i, value_4++ );

   for( IndexType i = 2; i < 6; i++ )       // 4th row
      m_4.setElement( 4, i, value_4++ );

   for( IndexType i = 3; i < 7; i++ )       // 5th row
      m_4.setElement( 5, i, value_4++ );

   for( IndexType i = 0; i < 5; i++ )       // 6th row
      m_4.setElement( 6, i, value_4++ );

   for( IndexType i = 0; i < 5; i++ )       // 7th row
      m_4.setElement( 7, i, value_4++ );

   VectorType inVector_4;
   inVector_4.setSize( m_cols_4 );
   for( IndexType i = 0; i < inVector_4.getSize(); i++ )
      inVector_4.setElement( i, 2 );

   VectorType outVector_4;
   outVector_4.setSize( m_rows_4 );
   for( IndexType j = 0; j < outVector_4.getSize(); j++ )
      outVector_4.setElement( j, 0 );

   m_4.vectorProduct( inVector_4, outVector_4 );

   EXPECT_EQ( outVector_4.getElement( 0 ),  20 );
   EXPECT_EQ( outVector_4.getElement( 1 ),  52 );
   EXPECT_EQ( outVector_4.getElement( 2 ), 110 );
   EXPECT_EQ( outVector_4.getElement( 3 ), 124 );
   EXPECT_EQ( outVector_4.getElement( 4 ), 156 );
   EXPECT_EQ( outVector_4.getElement( 5 ), 188 );
   EXPECT_EQ( outVector_4.getElement( 6 ), 280 );
   EXPECT_EQ( outVector_4.getElement( 7 ), 330 );
}

template< typename Matrix >
void test_VectorProduct_mediumSizeMatrix2()
{
   using RealType = typename Matrix::RealType;
   using DeviceType = typename Matrix::DeviceType;
   using IndexType = typename Matrix::IndexType;
   using VectorType = TNL::Containers::Vector< RealType, DeviceType, IndexType >;

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

   const IndexType m_rows_5 = 8;
   const IndexType m_cols_5 = 8;

   Matrix m_5( m_rows_5, m_cols_5 );
   typename Matrix::RowsCapacitiesType rowLengths_5{ 6, 3, 4, 5, 2, 7, 8, 8 };
   m_5.setRowCapacities( rowLengths_5 );

   RealType value_5 = 1;
   for( IndexType i = 0; i < 3; i++ )   // 0th row
      m_5.setElement( 0, i, value_5++ );

   m_5.setElement( 0, 4, value_5++ );           // 0th row
   m_5.setElement( 0, 5, value_5++ );

   m_5.setElement( 1, 1, value_5++ );           // 1st row
   m_5.setElement( 1, 3, value_5++ );

   for( IndexType i = 1; i < 3; i++ )            // 2nd row
      m_5.setElement( 2, i, value_5++ );

   m_5.setElement( 2, 4, value_5++ );           // 2nd row

   for( IndexType i = 1; i < 5; i++ )            // 3rd row
      m_5.setElement( 3, i, value_5++ );

   m_5.setElement( 4, 1, value_5++ );           // 4th row

   for( IndexType i = 1; i < 7; i++ )            // 5th row
      m_5.setElement( 5, i, value_5++ );

   for( IndexType i = 0; i < 7; i++ )            // 6th row
      m_5.setElement( 6, i, value_5++ );

   for( IndexType i = 0; i < 8; i++ )            // 7th row
      m_5.setElement( 7, i, value_5++ );

   for( IndexType i = 0; i < 7; i++ )            // 1s at the end of rows
      m_5.setElement( i, 7, 1);

   VectorType inVector_5;
   inVector_5.setSize( m_cols_5 );
   for( IndexType i = 0; i < inVector_5.getSize(); i++ )
       inVector_5.setElement( i, 2 );

   VectorType outVector_5;
   outVector_5.setSize( m_rows_5 );
   for( IndexType j = 0; j < outVector_5.getSize(); j++ )
       outVector_5.setElement( j, 0 );

   m_5.vectorProduct( inVector_5, outVector_5 );

   EXPECT_EQ( outVector_5.getElement( 0 ),  32 );
   EXPECT_EQ( outVector_5.getElement( 1 ),  28 );
   EXPECT_EQ( outVector_5.getElement( 2 ),  56 );
   EXPECT_EQ( outVector_5.getElement( 3 ), 102 );
   EXPECT_EQ( outVector_5.getElement( 4 ),  32 );
   EXPECT_EQ( outVector_5.getElement( 5 ), 224 );
   EXPECT_EQ( outVector_5.getElement( 6 ), 352 );
   EXPECT_EQ( outVector_5.getElement( 7 ), 520 );
}


template< typename Matrix >
void test_VectorProduct_largeMatrix()
{
   using RealType = typename Matrix::RealType;
   using DeviceType = typename Matrix::DeviceType;
   using IndexType = typename Matrix::IndexType;
   using VectorType = TNL::Containers::Vector< RealType, DeviceType, IndexType >;

   /////
   // Large test
   const IndexType size( 1051 );
   //for( int size = 1; size < 1000; size++ )
   {
      //std::cerr << " size = " << size << std::endl;
      // Test with large diagonal matrix
      Matrix m1( size, size );
      TNL::Containers::Vector< IndexType, DeviceType, IndexType > rowCapacities( size );
      rowCapacities.forAllElements( [] __cuda_callable__ ( IndexType i, IndexType& value ) { value = 1; } );
      m1.setRowCapacities( rowCapacities );
      auto f1 = [=] __cuda_callable__ ( IndexType row, IndexType localIdx, IndexType& column, RealType& value ) {
         if( localIdx == 0  )
         {
            value = row + 1;
            column = row;
         }
      };
      m1.forAllElements( f1 );
      // check that the matrix was initialized
      m1.getCompressedRowLengths( rowCapacities );
      EXPECT_EQ( rowCapacities, 1 );

      TNL::Containers::Vector< double, DeviceType, IndexType > in( size, 1.0 ), out( size, 0.0 );
      m1.vectorProduct( in, out );
      //std::cerr << out << std::endl;
      for( IndexType i = 0; i < size; i++ )
         EXPECT_EQ( out.getElement( i ), i + 1 );

      // Test with large triangular matrix
      const int rows( size ), columns( size );
      Matrix m2( rows, columns );
      rowCapacities.setSize( rows );
      rowCapacities.forAllElements( [=] __cuda_callable__ ( IndexType i, IndexType& value ) { value = i + 1; } );
      m2.setRowCapacities( rowCapacities );
      auto f2 = [=] __cuda_callable__ ( IndexType row, IndexType localIdx, IndexType& column, RealType& value ) {
         if( localIdx <= row )
         {
            value = localIdx + 1;
            column = localIdx;
         }
      };
      m2.forAllElements( f2 );
      // check that the matrix was initialized
      TNL::Containers::Vector< IndexType, DeviceType, IndexType > rowLengths( rows );
      m2.getCompressedRowLengths( rowLengths );
      EXPECT_EQ( rowLengths, rowCapacities );

      out.setSize( rows );
      out = 0.0;
      m2.vectorProduct( in, out );
      for( IndexType i = 0; i < rows; i++ )
         EXPECT_EQ( out.getElement( i ), ( i + 1 ) * ( i + 2 ) / 2 );
   }
}

template< typename Matrix >
void test_VectorProduct_longRowsMatrix()
{
   using RealType = typename Matrix::RealType;
   using DeviceType = typename Matrix::DeviceType;
   using IndexType = typename Matrix::IndexType;
   using VectorType = TNL::Containers::Vector< RealType, DeviceType, IndexType >;

   /**
    * Long row test
    */
   using MatrixSegmentsType = typename Matrix::SegmentsType;
   constexpr TNL::Algorithms::Segments::ElementsOrganization organization = MatrixSegmentsType::getOrganization();
   using ChunkedEllpackView_ = TNL::Algorithms::Segments::ChunkedEllpackView< DeviceType, IndexType, organization >;
   for( auto columns : { 64, 65, 128, 129, 256, 257, 512, 513, 1024, 1025, 2048, 2049, 3000 } )
   {
      const int rows = 33;
      Matrix m3( rows, columns );
      TNL::Containers::Vector< IndexType, DeviceType, IndexType > rowsCapacities( rows );
      rowsCapacities = columns;
      m3.setRowCapacities( rowsCapacities );
      auto f = [] __cuda_callable__ ( IndexType row, IndexType localIdx, IndexType& column, RealType& value ) {
         column = localIdx;
         value = localIdx + row;
      };
      m3.forAllElements( f );
      TNL::Containers::Vector< double, DeviceType, IndexType > in( columns, 1.0 ), out( rows, 0.0 );
      m3.vectorProduct( in, out );
      for( IndexType rowIdx = 0; rowIdx < rows; rowIdx++ )
         EXPECT_EQ( out.getElement( rowIdx ), ( double ) columns * ( double ) (columns - 1 ) / 2.0 + columns * rowIdx );
   }
}

#endif
