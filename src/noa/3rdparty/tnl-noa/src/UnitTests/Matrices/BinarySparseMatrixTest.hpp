#include <TNL/Containers/Vector.h>
#include <TNL/Containers/VectorView.h>
#include <TNL/Math.h>
#include <TNL/Algorithms/ParallelFor.h>
#include <iostream>
#include <sstream>

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

   Matrix m2( rows, cols );
   EXPECT_EQ( m2.getRows(), 9 );
   EXPECT_EQ( m2.getColumns(), 8 );
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
   typename Matrix::RowsCapacitiesType rowLengths;
   rowLengths.setSize( rows );
   rowLengths.setValue( 3 );

   IndexType rowLength = 1;
   for( IndexType i = 2; i < rows; i++ )
      rowLengths.setElement( i, rowLength++ );

   m.setRowCapacities( rowLengths );

   // Insert values into the rows.
   for( IndexType i = 0; i < 3; i++ )      // 0th row
      m.setElement( 0, i, 1 );

   for( IndexType i = 0; i < 3; i++ )      // 1st row
      m.setElement( 1, i, 1 );

   for( IndexType i = 0; i < 1; i++ )      // 2nd row
      m.setElement( 2, i, 1 );

   for( IndexType i = 0; i < 2; i++ )      // 3rd row
      m.setElement( 3, i, 1 );

   for( IndexType i = 0; i < 3; i++ )      // 4th row
      m.setElement( 4, i, 1 );

   for( IndexType i = 0; i < 4; i++ )      // 5th row
      m.setElement( 5, i, 1 );

   for( IndexType i = 0; i < 5; i++ )      // 6th row
      m.setElement( 6, i, 1 );

   for( IndexType i = 0; i < 6; i++ )      // 7th row
      m.setElement( 7, i, 1 );

   for( IndexType i = 0; i < 7; i++ )      // 8th row
      m.setElement( 8, i, 1 );

   for( IndexType i = 0; i < 8; i++ )      // 9th row
      m.setElement( 9, i, 1 );

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
void test_GetNumberOfNonzeroMatrixElements()
{
   using RealType = typename Matrix::RealType;
   using DeviceType = typename Matrix::DeviceType;
   using IndexType = typename Matrix::IndexType;

   /*
    * Sets up the following 10x10 sparse matrix:
    *
    *    /  1  0  1  0  1  0  1  0  0  0  \
    *    |  1  1  1  0  0  0  0  0  0  0  |
    *    |  1  1  1  1  1  1  1  1  0  0  |
    *    |  1  1  0  0  0  0  0  0  0  0  |
    *    |  1  0  0  0  0  0  0  0  0  0  |
    *    |  1  0  0  0  0  0  0  0  0  0  |
    *    |  1  0  0  0  0  0  0  0  0  0  |
    *    |  1  0  0  0  0  0  0  0  0  0  |
    *    |  1  1  1  1  1  1  1  1  1  1  |
    *    \  1  1  1  1  1  1  1  1  1  1 /
    */

   const IndexType rows = 10;
   const IndexType cols = 10;

   Matrix m( rows, cols );

   typename Matrix::RowsCapacitiesType rowLengths;
   rowLengths.setSize( rows );
   rowLengths.setElement( 0, 1 );
   rowLengths.setElement( 1, 1 );
   rowLengths.setElement( 2, 1 );
   rowLengths.setElement( 3, 1 );
   for( IndexType i = 4; i < rows - 2; i++ )
      rowLengths.setElement( i, 1 );

   rowLengths.setElement( 8, 1 );
   rowLengths.setElement( 9, 1 );
   m.setRowCapacities( rowLengths );

   for( IndexType i = 0; i < 4; i++ )
      m.setElement( 0, 2 * i, 1 );

   for( IndexType i = 0; i < 3; i++ )
      m.setElement( 1, i, 1 );

   for( IndexType i = 0; i < 8; i++ )
      m.setElement( 2, i, 1 );

   for( IndexType i = 0; i < 2; i++ )
      m.setElement( 3, i, 1 );

   for( IndexType i = 4; i < 8; i++ )
      m.setElement( i, 0, 1 );

   for( IndexType j = 8; j < rows; j++)
      for( IndexType i = 0; i < cols; i++ )
         m.setElement( j, i, 1 );

   EXPECT_EQ( m.getNumberOfNonzeroMatrixElements(), 41 );
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

   /*
    * Sets up the following 10x10 sparse matrix:
    *
    *    /  1  0  1  0  1  0  1  0  0  0  \
    *    |  1  1  1  0  0  0  0  0  0  0  |
    *    |  1  1  1  1  1  1  1  1  0  0  |
    *    |  1  1  0  0  0  0  0  0  0  0  |
    *    |  1  0  0  0  0  0  0  0  0  0  |
    *    |  1  0  0  0  0  0  0  0  0  0  |
    *    |  1  0  0  0  0  0  0  0  0  0  |
    *    |  1  0  0  0  0  0  0  0  0  0  |
    *    |  1  1  1  1  1  1  1  1  1  1  |
    *    \  1  1  1  1  1  1  1  1  1  1 /
    */

   const IndexType rows = 10;
   const IndexType cols = 10;

   Matrix m( rows, cols );

   typename Matrix::RowsCapacitiesType rowLengths;
   rowLengths.setSize( rows );
   rowLengths.setElement( 0, 4 );
   rowLengths.setElement( 1, 3 );
   rowLengths.setElement( 2, 8 );
   rowLengths.setElement( 3, 2 );
   for( IndexType i = 4; i < rows - 2; i++ )
       rowLengths.setElement( i, 1 );

   rowLengths.setElement( 8, 10 );
   rowLengths.setElement( 9, 10 );
   m.setRowCapacities( rowLengths );

   auto matrixView = m.getView();
   auto f = [=] __cuda_callable__ ( const IndexType rowIdx ) mutable {
      auto row = matrixView.getRow( rowIdx );
      switch( rowIdx )
      {
         case 0:
           for( IndexType i = 0; i < 4; i++ )
              row.setElement( i, 2 * i, 1 );
           break;
        case 1:
           for( IndexType i = 0; i < 3; i++ )
              row.setElement( i, i, 1 );
           break;
        case 2:
           for( IndexType i = 0; i < 8; i++ )
              row.setElement( i, i, 1 );
           break;
        case 3:
           for( IndexType i = 0; i < 2; i++ )
              row.setElement( i, i, 1 );
           break;
        case 4:
           row.setElement( 0, 0, 1 );
           break;
        case 5:
           row.setElement( 0, 0, 1 );
           break;
        case 6:
           row.setElement( 0, 0, 1 );
           break;
        case 7:
           row.setElement( 0, 0, 1 );
           break;
        case 8:
            for( IndexType i = 0; i < rows; i++ )
               row.setElement( i, i, 1 );
            break;
        case 9:
            for( IndexType i = 0; i < rows; i++ )
               row.setElement( i, i, 1 );
            break;
      }
   };
   TNL::Algorithms::ParallelFor< DeviceType >::exec( ( IndexType ) 0, rows, f );

   EXPECT_EQ( m.getElement( 0, 0 ), 1 );
   EXPECT_EQ( m.getElement( 0, 1 ), 0 );
   EXPECT_EQ( m.getElement( 0, 2 ), 1 );
   EXPECT_EQ( m.getElement( 0, 3 ), 0 );
   EXPECT_EQ( m.getElement( 0, 4 ), 1 );
   EXPECT_EQ( m.getElement( 0, 5 ), 0 );
   EXPECT_EQ( m.getElement( 0, 6 ), 1 );
   EXPECT_EQ( m.getElement( 0, 7 ), 0 );
   EXPECT_EQ( m.getElement( 0, 8 ), 0 );
   EXPECT_EQ( m.getElement( 0, 9 ), 0 );

   EXPECT_EQ( m.getElement( 1, 0 ), 1 );
   EXPECT_EQ( m.getElement( 1, 1 ), 1 );
   EXPECT_EQ( m.getElement( 1, 2 ), 1 );
   EXPECT_EQ( m.getElement( 1, 3 ), 0 );
   EXPECT_EQ( m.getElement( 1, 4 ), 0 );
   EXPECT_EQ( m.getElement( 1, 5 ), 0 );
   EXPECT_EQ( m.getElement( 1, 6 ), 0 );
   EXPECT_EQ( m.getElement( 1, 7 ), 0 );
   EXPECT_EQ( m.getElement( 1, 8 ), 0 );
   EXPECT_EQ( m.getElement( 1, 9 ), 0 );

   EXPECT_EQ( m.getElement( 2, 0 ), 1 );
   EXPECT_EQ( m.getElement( 2, 1 ), 1 );
   EXPECT_EQ( m.getElement( 2, 2 ), 1 );
   EXPECT_EQ( m.getElement( 2, 3 ), 1 );
   EXPECT_EQ( m.getElement( 2, 4 ), 1 );
   EXPECT_EQ( m.getElement( 2, 5 ), 1 );
   EXPECT_EQ( m.getElement( 2, 6 ), 1 );
   EXPECT_EQ( m.getElement( 2, 7 ), 1 );
   EXPECT_EQ( m.getElement( 2, 8 ), 0 );
   EXPECT_EQ( m.getElement( 2, 9 ), 0 );

   EXPECT_EQ( m.getElement( 3, 0 ), 1 );
   EXPECT_EQ( m.getElement( 3, 1 ), 1 );
   EXPECT_EQ( m.getElement( 3, 2 ), 0 );
   EXPECT_EQ( m.getElement( 3, 3 ), 0 );
   EXPECT_EQ( m.getElement( 3, 4 ), 0 );
   EXPECT_EQ( m.getElement( 3, 5 ), 0 );
   EXPECT_EQ( m.getElement( 3, 6 ), 0 );
   EXPECT_EQ( m.getElement( 3, 7 ), 0 );
   EXPECT_EQ( m.getElement( 3, 8 ), 0 );
   EXPECT_EQ( m.getElement( 3, 9 ), 0 );

   EXPECT_EQ( m.getElement( 4, 0 ), 1 );
   EXPECT_EQ( m.getElement( 4, 1 ), 0 );
   EXPECT_EQ( m.getElement( 4, 2 ), 0 );
   EXPECT_EQ( m.getElement( 4, 3 ), 0 );
   EXPECT_EQ( m.getElement( 4, 4 ), 0 );
   EXPECT_EQ( m.getElement( 4, 5 ), 0 );
   EXPECT_EQ( m.getElement( 4, 6 ), 0 );
   EXPECT_EQ( m.getElement( 4, 7 ), 0 );
   EXPECT_EQ( m.getElement( 4, 8 ), 0 );
   EXPECT_EQ( m.getElement( 4, 9 ), 0 );

   EXPECT_EQ( m.getElement( 5, 0 ), 1 );
   EXPECT_EQ( m.getElement( 5, 1 ), 0 );
   EXPECT_EQ( m.getElement( 5, 2 ), 0 );
   EXPECT_EQ( m.getElement( 5, 3 ), 0 );
   EXPECT_EQ( m.getElement( 5, 4 ), 0 );
   EXPECT_EQ( m.getElement( 5, 5 ), 0 );
   EXPECT_EQ( m.getElement( 5, 6 ), 0 );
   EXPECT_EQ( m.getElement( 5, 7 ), 0 );
   EXPECT_EQ( m.getElement( 5, 8 ), 0 );
   EXPECT_EQ( m.getElement( 5, 9 ), 0 );

   EXPECT_EQ( m.getElement( 6, 0 ), 1 );
   EXPECT_EQ( m.getElement( 6, 1 ), 0 );
   EXPECT_EQ( m.getElement( 6, 2 ), 0 );
   EXPECT_EQ( m.getElement( 6, 3 ), 0 );
   EXPECT_EQ( m.getElement( 6, 4 ), 0 );
   EXPECT_EQ( m.getElement( 6, 5 ), 0 );
   EXPECT_EQ( m.getElement( 6, 6 ), 0 );
   EXPECT_EQ( m.getElement( 6, 7 ), 0 );
   EXPECT_EQ( m.getElement( 6, 8 ), 0 );
   EXPECT_EQ( m.getElement( 6, 9 ), 0 );

   EXPECT_EQ( m.getElement( 7, 0 ), 1 );
   EXPECT_EQ( m.getElement( 7, 1 ), 0 );
   EXPECT_EQ( m.getElement( 7, 2 ), 0 );
   EXPECT_EQ( m.getElement( 7, 3 ), 0 );
   EXPECT_EQ( m.getElement( 7, 4 ), 0 );
   EXPECT_EQ( m.getElement( 7, 5 ), 0 );
   EXPECT_EQ( m.getElement( 7, 6 ), 0 );
   EXPECT_EQ( m.getElement( 7, 7 ), 0 );
   EXPECT_EQ( m.getElement( 7, 8 ), 0 );
   EXPECT_EQ( m.getElement( 7, 9 ), 0 );

   EXPECT_EQ( m.getElement( 8, 0 ), 1 );
   EXPECT_EQ( m.getElement( 8, 1 ), 1 );
   EXPECT_EQ( m.getElement( 8, 2 ), 1 );
   EXPECT_EQ( m.getElement( 8, 3 ), 1 );
   EXPECT_EQ( m.getElement( 8, 4 ), 1 );
   EXPECT_EQ( m.getElement( 8, 5 ), 1 );
   EXPECT_EQ( m.getElement( 8, 6 ), 1 );
   EXPECT_EQ( m.getElement( 8, 7 ), 1 );
   EXPECT_EQ( m.getElement( 8, 8 ), 1 );
   EXPECT_EQ( m.getElement( 8, 9 ), 1 );

   EXPECT_EQ( m.getElement( 9, 0 ), 1 );
   EXPECT_EQ( m.getElement( 9, 1 ), 1 );
   EXPECT_EQ( m.getElement( 9, 2 ), 1 );
   EXPECT_EQ( m.getElement( 9, 3 ), 1 );
   EXPECT_EQ( m.getElement( 9, 4 ), 1 );
   EXPECT_EQ( m.getElement( 9, 5 ), 1 );
   EXPECT_EQ( m.getElement( 9, 6 ), 1 );
   EXPECT_EQ( m.getElement( 9, 7 ), 1 );
   EXPECT_EQ( m.getElement( 9, 8 ), 1 );
   EXPECT_EQ( m.getElement( 9, 9 ), 1 );
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
    *    /  1  0  1  0  1  0  1  0  0  0  \
    *    |  1  1  1  0  0  0  0  0  0  0  |
    *    |  1  1  1  1  1  1  1  1  0  0  |
    *    |  1  1  0  0  0  0  0  0  0  0  |
    *    |  1  0  0  0  0  0  0  0  0  0  |
    *    |  1  0  0  0  0  0  0  0  0  0  |
    *    |  1  0  0  0  0  0  0  0  0  0  |
    *    |  1  0  0  0  0  0  0  0  0  0  |
    *    |  1  1  1  1  1  1  1  1  1  1  |
    *    \  1  1  1  1  1  1  1  1  1  1 /
    */

   const IndexType rows = 10;
   const IndexType cols = 10;

   Matrix m( rows, cols );

   typename Matrix::RowsCapacitiesType rowLengths;
   rowLengths.setSize( rows );
   rowLengths.setElement( 0, 4 );
   rowLengths.setElement( 1, 3 );
   rowLengths.setElement( 2, 8 );
   rowLengths.setElement( 3, 2 );
   for( IndexType i = 4; i < rows - 2; i++ )
       rowLengths.setElement( i, 1 );

   rowLengths.setElement( 8, 10 );
   rowLengths.setElement( 9, 10 );
   m.setRowCapacities( rowLengths );

   for( IndexType i = 0; i < 4; i++ )
       m.setElement( 0, 2 * i, 1 );

   for( IndexType i = 0; i < 3; i++ )
       m.setElement( 1, i, 1 );

   for( IndexType i = 0; i < 8; i++ )
       m.setElement( 2, i, 1 );

   for( IndexType i = 0; i < 2; i++ )
       m.setElement( 3, i, 1 );

   for( IndexType i = 4; i < 8; i++ )
       m.setElement( i, 0, 1 );

   for( IndexType j = 8; j < rows; j++)
       for( IndexType i = 0; i < cols; i++ )
           m.setElement( j, i, 1 );

   EXPECT_EQ( m.getElement( 0, 0 ), 1 );
   EXPECT_EQ( m.getElement( 0, 1 ), 0 );
   EXPECT_EQ( m.getElement( 0, 2 ), 1 );
   EXPECT_EQ( m.getElement( 0, 3 ), 0 );
   EXPECT_EQ( m.getElement( 0, 4 ), 1 );
   EXPECT_EQ( m.getElement( 0, 5 ), 0 );
   EXPECT_EQ( m.getElement( 0, 6 ), 1 );
   EXPECT_EQ( m.getElement( 0, 7 ), 0 );
   EXPECT_EQ( m.getElement( 0, 8 ), 0 );
   EXPECT_EQ( m.getElement( 0, 9 ), 0 );

   EXPECT_EQ( m.getElement( 1, 0 ), 1 );
   EXPECT_EQ( m.getElement( 1, 1 ), 1 );
   EXPECT_EQ( m.getElement( 1, 2 ), 1 );
   EXPECT_EQ( m.getElement( 1, 3 ), 0 );
   EXPECT_EQ( m.getElement( 1, 4 ), 0 );
   EXPECT_EQ( m.getElement( 1, 5 ), 0 );
   EXPECT_EQ( m.getElement( 1, 6 ), 0 );
   EXPECT_EQ( m.getElement( 1, 7 ), 0 );
   EXPECT_EQ( m.getElement( 1, 8 ), 0 );
   EXPECT_EQ( m.getElement( 1, 9 ), 0 );

   EXPECT_EQ( m.getElement( 2, 0 ), 1 );
   EXPECT_EQ( m.getElement( 2, 1 ), 1 );
   EXPECT_EQ( m.getElement( 2, 2 ), 1 );
   EXPECT_EQ( m.getElement( 2, 3 ), 1 );
   EXPECT_EQ( m.getElement( 2, 4 ), 1 );
   EXPECT_EQ( m.getElement( 2, 5 ), 1 );
   EXPECT_EQ( m.getElement( 2, 6 ), 1 );
   EXPECT_EQ( m.getElement( 2, 7 ), 1 );
   EXPECT_EQ( m.getElement( 2, 8 ), 0 );
   EXPECT_EQ( m.getElement( 2, 9 ), 0 );

   EXPECT_EQ( m.getElement( 3, 0 ), 1 );
   EXPECT_EQ( m.getElement( 3, 1 ), 1 );
   EXPECT_EQ( m.getElement( 3, 2 ), 0 );
   EXPECT_EQ( m.getElement( 3, 3 ), 0 );
   EXPECT_EQ( m.getElement( 3, 4 ), 0 );
   EXPECT_EQ( m.getElement( 3, 5 ), 0 );
   EXPECT_EQ( m.getElement( 3, 6 ), 0 );
   EXPECT_EQ( m.getElement( 3, 7 ), 0 );
   EXPECT_EQ( m.getElement( 3, 8 ), 0 );
   EXPECT_EQ( m.getElement( 3, 9 ), 0 );

   EXPECT_EQ( m.getElement( 4, 0 ), 1 );
   EXPECT_EQ( m.getElement( 4, 1 ), 0 );
   EXPECT_EQ( m.getElement( 4, 2 ), 0 );
   EXPECT_EQ( m.getElement( 4, 3 ), 0 );
   EXPECT_EQ( m.getElement( 4, 4 ), 0 );
   EXPECT_EQ( m.getElement( 4, 5 ), 0 );
   EXPECT_EQ( m.getElement( 4, 6 ), 0 );
   EXPECT_EQ( m.getElement( 4, 7 ), 0 );
   EXPECT_EQ( m.getElement( 4, 8 ), 0 );
   EXPECT_EQ( m.getElement( 4, 9 ), 0 );

   EXPECT_EQ( m.getElement( 5, 0 ), 1 );
   EXPECT_EQ( m.getElement( 5, 1 ), 0 );
   EXPECT_EQ( m.getElement( 5, 2 ), 0 );
   EXPECT_EQ( m.getElement( 5, 3 ), 0 );
   EXPECT_EQ( m.getElement( 5, 4 ), 0 );
   EXPECT_EQ( m.getElement( 5, 5 ), 0 );
   EXPECT_EQ( m.getElement( 5, 6 ), 0 );
   EXPECT_EQ( m.getElement( 5, 7 ), 0 );
   EXPECT_EQ( m.getElement( 5, 8 ), 0 );
   EXPECT_EQ( m.getElement( 5, 9 ), 0 );

   EXPECT_EQ( m.getElement( 6, 0 ), 1 );
   EXPECT_EQ( m.getElement( 6, 1 ), 0 );
   EXPECT_EQ( m.getElement( 6, 2 ), 0 );
   EXPECT_EQ( m.getElement( 6, 3 ), 0 );
   EXPECT_EQ( m.getElement( 6, 4 ), 0 );
   EXPECT_EQ( m.getElement( 6, 5 ), 0 );
   EXPECT_EQ( m.getElement( 6, 6 ), 0 );
   EXPECT_EQ( m.getElement( 6, 7 ), 0 );
   EXPECT_EQ( m.getElement( 6, 8 ), 0 );
   EXPECT_EQ( m.getElement( 6, 9 ), 0 );

   EXPECT_EQ( m.getElement( 7, 0 ), 1 );
   EXPECT_EQ( m.getElement( 7, 1 ), 0 );
   EXPECT_EQ( m.getElement( 7, 2 ), 0 );
   EXPECT_EQ( m.getElement( 7, 3 ), 0 );
   EXPECT_EQ( m.getElement( 7, 4 ), 0 );
   EXPECT_EQ( m.getElement( 7, 5 ), 0 );
   EXPECT_EQ( m.getElement( 7, 6 ), 0 );
   EXPECT_EQ( m.getElement( 7, 7 ), 0 );
   EXPECT_EQ( m.getElement( 7, 8 ), 0 );
   EXPECT_EQ( m.getElement( 7, 9 ), 0 );

   EXPECT_EQ( m.getElement( 8, 0 ), 1 );
   EXPECT_EQ( m.getElement( 8, 1 ), 1 );
   EXPECT_EQ( m.getElement( 8, 2 ), 1 );
   EXPECT_EQ( m.getElement( 8, 3 ), 1 );
   EXPECT_EQ( m.getElement( 8, 4 ), 1 );
   EXPECT_EQ( m.getElement( 8, 5 ), 1 );
   EXPECT_EQ( m.getElement( 8, 6 ), 1 );
   EXPECT_EQ( m.getElement( 8, 7 ), 1 );
   EXPECT_EQ( m.getElement( 8, 8 ), 1 );
   EXPECT_EQ( m.getElement( 8, 9 ), 1 );

   EXPECT_EQ( m.getElement( 9, 0 ), 1 );
   EXPECT_EQ( m.getElement( 9, 1 ), 1 );
   EXPECT_EQ( m.getElement( 9, 2 ), 1 );
   EXPECT_EQ( m.getElement( 9, 3 ), 1 );
   EXPECT_EQ( m.getElement( 9, 4 ), 1 );
   EXPECT_EQ( m.getElement( 9, 5 ), 1 );
   EXPECT_EQ( m.getElement( 9, 6 ), 1 );
   EXPECT_EQ( m.getElement( 9, 7 ), 1 );
   EXPECT_EQ( m.getElement( 9, 8 ), 1 );
   EXPECT_EQ( m.getElement( 9, 9 ), 1 );
}

template< typename Matrix >
void test_VectorProduct()
{
   using RealType = typename Matrix::RealType;
   using ComputeRealType = typename Matrix::ComputeRealType;
   using DeviceType = typename Matrix::DeviceType;
   using IndexType = typename Matrix::IndexType;
   using VectorType = TNL::Containers::Vector< ComputeRealType, DeviceType, IndexType >;

   /*
    * Sets up the following 4x4 sparse matrix:
    *
    *    /  1  0  0  0 \
    *    |  0  1  0  1 |
    *    |  0  1  0  0 |
    *    \  0  0  1  0 /
    */

   const IndexType m_rows_1 = 4;
   const IndexType m_cols_1 = 4;

   Matrix m_1( m_rows_1, m_cols_1 );
   typename Matrix::RowsCapacitiesType rowLengths_1;
   rowLengths_1.setSize( m_rows_1 );
   rowLengths_1.setElement( 0, 1 );
   rowLengths_1.setElement( 1, 2 );
   rowLengths_1.setElement( 2, 1 );
   rowLengths_1.setElement( 3, 1 );
   m_1.setRowCapacities( rowLengths_1 );

   m_1.setElement( 0, 0, 1 );      // 0th row

   m_1.setElement( 1, 1, 1 );      // 1st row
   m_1.setElement( 1, 3, 1 );

   m_1.setElement( 2, 1, 1 );      // 2nd row

   m_1.setElement( 3, 2, 1 );      // 3rd row

   VectorType inVector_1( m_cols_1 );
   inVector_1 = 2.0;

   VectorType outVector_1( m_rows_1 );
   outVector_1 = 0.0;

   m_1.vectorProduct( inVector_1, outVector_1 );


   EXPECT_EQ( outVector_1.getElement( 0 ), 2 );
   EXPECT_EQ( outVector_1.getElement( 1 ), 4 );
   EXPECT_EQ( outVector_1.getElement( 2 ), 2 );
   EXPECT_EQ( outVector_1.getElement( 3 ), 2 );


   /*
    * Sets up the following 4x4 sparse matrix:
    *
    *    /  1  1  1  0 \
    *    |  0  0  0  1 |
    *    |  1  1  1  0 |
    *    \  0  1  0  0 /
    */
   const IndexType m_rows_2 = 4;
   const IndexType m_cols_2 = 4;

   Matrix m_2( m_rows_2, m_cols_2 );
   typename Matrix::RowsCapacitiesType rowLengths_2;
   rowLengths_2.setSize( m_rows_2 );
   rowLengths_2.setValue( 3 );
   rowLengths_2.setElement( 1, 1 );
   rowLengths_2.setElement( 3, 1 );
   m_2.setRowCapacities( rowLengths_2 );

   for( IndexType i = 0; i < 3; i++ )   // 0th row
      m_2.setElement( 0, i, 1 );

   m_2.setElement( 1, 3, 1 );           // 1st row

   for( IndexType i = 0; i < 3; i++ )   // 2nd row
      m_2.setElement( 2, i, 1 );

   for( IndexType i = 1; i < 2; i++ )   // 3rd row
      m_2.setElement( 3, i, 1 );

   VectorType inVector_2( m_cols_2 );
   inVector_2 = 2.0;

   VectorType outVector_2( m_rows_2 );
   outVector_2 = 0.0;

   m_2.vectorProduct( inVector_2, outVector_2 );

   EXPECT_EQ( outVector_2.getElement( 0 ), 6 );
   EXPECT_EQ( outVector_2.getElement( 1 ), 2 );
   EXPECT_EQ( outVector_2.getElement( 2 ), 6 );
   EXPECT_EQ( outVector_2.getElement( 3 ), 2 );

   /*
    * Sets up the following 4x4 sparse matrix:
    *
    *    /  1  1  1  0 \
    *    |  0  1  1  1 |
    *    |  1  1  1  0 |
    *    \  0  1  1  1 /
    */
   const IndexType m_rows_3 = 4;
   const IndexType m_cols_3 = 4;

   Matrix m_3( m_rows_3, m_cols_3 );
   typename Matrix::RowsCapacitiesType rowLengths_3;
   rowLengths_3.setSize( m_rows_3 );
   rowLengths_3.setValue( 3 );
   m_3.setRowCapacities( rowLengths_3 );

   for( IndexType i = 0; i < 3; i++ )          // 0th row
      m_3.setElement( 0, i, 1 );

   for( IndexType i = 1; i < 4; i++ )
      m_3.setElement( 1, i, 1 );      // 1st row

   for( IndexType i = 0; i < 3; i++ )          // 2nd row
      m_3.setElement( 2, i, 1 );

   for( IndexType i = 1; i < 4; i++ )          // 3rd row
      m_3.setElement( 3, i, 1 );

   VectorType inVector_3( m_cols_3 );
   inVector_3 = 2.0;

   VectorType outVector_3( m_rows_3 );
   outVector_3 = 0.0;

   m_3.vectorProduct( inVector_3, outVector_3 );


   EXPECT_EQ( outVector_3.getElement( 0 ), 6 );
   EXPECT_EQ( outVector_3.getElement( 1 ), 6 );
   EXPECT_EQ( outVector_3.getElement( 2 ), 6 );
   EXPECT_EQ( outVector_3.getElement( 3 ), 6 );

   /*
    * Sets up the following 8x8 sparse matrix:
    *
    *    /  1  1  1  0  0  1  0  0 \
    *    |  0  1  1  1  1  0  0  0 |
    *    |  1  1  1  1  1  0  0  0 |
    *    |  0  1  1  1  1  0  0  0 |
    *    |  0  0  1  1  1  1  0  0 |
    *    |  0  0  0  1  1  1  1  0 |
    *    |  1  1  1  1  1  0  0  0 |
    *    \  1  1  1  1  1  0  0  0 /
    */
   const IndexType m_rows_4 = 8;
   const IndexType m_cols_4 = 8;

   Matrix m_4( m_rows_4, m_cols_4 );
   typename Matrix::RowsCapacitiesType rowLengths_4;
   rowLengths_4.setSize( m_rows_4 );
   rowLengths_4.setValue( 4 );
   rowLengths_4.setElement( 2, 5 );
   rowLengths_4.setElement( 6, 5 );
   rowLengths_4.setElement( 7, 5 );
   m_4.setRowCapacities( rowLengths_4 );

   for( IndexType i = 0; i < 3; i++ )       // 0th row
      m_4.setElement( 0, i, 1 );

   m_4.setElement( 0, 5, 1 );

   for( IndexType i = 1; i < 5; i++ )       // 1st row
      m_4.setElement( 1, i, 1 );

   for( IndexType i = 0; i < 5; i++ )       // 2nd row
      m_4.setElement( 2, i, 1 );

   for( IndexType i = 1; i < 5; i++ )       // 3rd row
      m_4.setElement( 3, i, 1 );

   for( IndexType i = 2; i < 6; i++ )       // 4th row
      m_4.setElement( 4, i, 1 );

   for( IndexType i = 3; i < 7; i++ )       // 5th row
      m_4.setElement( 5, i, 1 );

   for( IndexType i = 0; i < 5; i++ )       // 6th row
      m_4.setElement( 6, i, 1 );

   for( IndexType i = 0; i < 5; i++ )       // 7th row
      m_4.setElement( 7, i, 1 );

   VectorType inVector_4( m_cols_4 );
   inVector_4 = 2.0;

   VectorType outVector_4( m_rows_4 );
   outVector_4 = 0.0;

   m_4.vectorProduct( inVector_4, outVector_4 );


   EXPECT_EQ( outVector_4.getElement( 0 ),  8 );
   EXPECT_EQ( outVector_4.getElement( 1 ),  8 );
   EXPECT_EQ( outVector_4.getElement( 2 ), 10 );
   EXPECT_EQ( outVector_4.getElement( 3 ),  8 );
   EXPECT_EQ( outVector_4.getElement( 4 ),  8 );
   EXPECT_EQ( outVector_4.getElement( 5 ),  8 );
   EXPECT_EQ( outVector_4.getElement( 6 ), 10 );
   EXPECT_EQ( outVector_4.getElement( 7 ), 10 );


   /*
    * Sets up the following 8x8 sparse matrix:
    *
    *    /  1  1  1  0  1  1  0  1 \   6
    *    |  0  1  0  1  0  0  0  1 |   3
    *    |  0  1  1  0  1  0  0  1 |   4
    *    |  0  1  1  1  1  0  0  1 |   5
    *    |  0  1  0  0  0  0  0  1 |   2
    *    |  0  1  1  1  1  1  1  1 |   7
    *    |  1  1  1  1  1  1  1  1 |   8
    *    \  1  1  1  1  1  1  1  1 /   8
    */

   const IndexType m_rows_5 = 8;
   const IndexType m_cols_5 = 8;

   Matrix m_5( m_rows_5, m_cols_5 );
   typename Matrix::RowsCapacitiesType rowLengths_5;
   rowLengths_5.setSize( m_rows_5 );
   rowLengths_5.setElement(0, 6);
   rowLengths_5.setElement(1, 3);
   rowLengths_5.setElement(2, 4);
   rowLengths_5.setElement(3, 5);
   rowLengths_5.setElement(4, 2);
   rowLengths_5.setElement(5, 7);
   rowLengths_5.setElement(6, 8);
   rowLengths_5.setElement(7, 8);
   m_5.setRowCapacities( rowLengths_5 );

   for( IndexType i = 0; i < 3; i++ )   // 0th row
      m_5.setElement( 0, i, 1 );

   m_5.setElement( 0, 4, 1 );           // 0th row
   m_5.setElement( 0, 5, 1 );

   m_5.setElement( 1, 1, 1 );           // 1st row
   m_5.setElement( 1, 3, 1 );

   for( IndexType i = 1; i < 3; i++ )            // 2nd row
      m_5.setElement( 2, i, 1 );

   m_5.setElement( 2, 4, 1 );           // 2nd row

   for( IndexType i = 1; i < 5; i++ )            // 3rd row
      m_5.setElement( 3, i, 1 );

   m_5.setElement( 4, 1, 1 );           // 4th row

   for( IndexType i = 1; i < 7; i++ )            // 5th row
      m_5.setElement( 5, i, 1 );

   for( IndexType i = 0; i < 7; i++ )            // 6th row
      m_5.setElement( 6, i, 1 );

   for( IndexType i = 0; i < 8; i++ )            // 7th row
      m_5.setElement( 7, i, 1 );

   for( IndexType i = 0; i < 7; i++ )            // 1s at the end of rows
      m_5.setElement( i, 7, 1);

   VectorType inVector_5( m_cols_5 );
   inVector_5 = 2.0;

   VectorType outVector_5( m_rows_5 );
   outVector_5 = 0.0;

   m_5.vectorProduct( inVector_5, outVector_5 );

   EXPECT_EQ( outVector_5.getElement( 0 ), 12 );
   EXPECT_EQ( outVector_5.getElement( 1 ),  6 );
   EXPECT_EQ( outVector_5.getElement( 2 ),  8 );
   EXPECT_EQ( outVector_5.getElement( 3 ), 10 );
   EXPECT_EQ( outVector_5.getElement( 4 ),  4 );
   EXPECT_EQ( outVector_5.getElement( 5 ), 14 );
   EXPECT_EQ( outVector_5.getElement( 6 ), 16 );
   EXPECT_EQ( outVector_5.getElement( 7 ), 16 );
}

template< typename Matrix >
void test_reduceRows()
{
   using RealType = typename Matrix::RealType;
   using ComputeRealType = typename Matrix::ComputeRealType;
   using DeviceType = typename Matrix::DeviceType;
   using IndexType = typename Matrix::IndexType;

   /*
    * Sets up the following 8x8 sparse matrix:
    *
    *    /  1  1  1  0  1  1  0  1 \   6
    *    |  0  1  0  1  0  0  0  1 |   3
    *    |  0  1  1  0  1  0  0  1 |   4
    *    |  0  1  1  1  1  0  0  1 |   5
    *    |  0  1  0  0  0  0  0  1 |   2
    *    |  0  1  1  1  1  1  1  1 |   7
    *    |  1  1  1  1  1  1  1  1 |   8
    *    \  1  1  1  1  1  1  1  1 /   8
    */

   const IndexType rows = 8;
   const IndexType cols = 8;

   Matrix m( rows, cols );
   typename Matrix::RowsCapacitiesType rowsCapacities( rows );
   //rowLengths.setSize( rows );
   rowsCapacities.setElement(0, 6);
   rowsCapacities.setElement(1, 3);
   rowsCapacities.setElement(2, 4);
   rowsCapacities.setElement(3, 5);
   rowsCapacities.setElement(4, 2);
   rowsCapacities.setElement(5, 7);
   rowsCapacities.setElement(6, 8);
   rowsCapacities.setElement(7, 8);
   m.setRowCapacities( rowsCapacities );

   for( IndexType i = 0; i < 3; i++ )   // 0th row
      m.setElement( 0, i, 1 );

   m.setElement( 0, 4, 1 );             // 0th row
   m.setElement( 0, 5, 1 );

   m.setElement( 1, 1, 1 );             // 1st row
   m.setElement( 1, 3, 1 );

   for( IndexType i = 1; i < 3; i++ )   // 2nd row
      m.setElement( 2, i, 1 );

   m.setElement( 2, 4, 1 );             // 2nd row

   for( IndexType i = 1; i < 5; i++ )   // 3rd row
      m.setElement( 3, i, 1 );

   m.setElement( 4, 1, 1 );             // 4th row

   for( IndexType i = 1; i < 7; i++ )   // 5th row
      m.setElement( 5, i, 1 );

   for( IndexType i = 0; i < 7; i++ )   // 6th row
      m.setElement( 6, i, 1 );

   for( IndexType i = 0; i < 8; i++ )   // 7th row
       m.setElement( 7, i, 1 );

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
   TNL::Containers::Vector< ComputeRealType, DeviceType, IndexType > rowSums( rows );
   auto rowSums_view = rowSums.getView();
   auto max_fetch = [] __cuda_callable__ ( IndexType row, IndexType column, const RealType& value ) -> IndexType {
      return TNL::abs( value );
   };
   auto max_keep = [=] __cuda_callable__ ( const IndexType rowIdx, const IndexType value ) mutable {
      rowSums_view[ rowIdx ] = value;
   };
   m.reduceAllRows( max_fetch, std::plus<>{}, max_keep, 0 );
   const auto maxNorm = TNL::max( rowSums );
   EXPECT_EQ( maxNorm, 8 ) ; // 29+30+31+32+33+34+35+36
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
    *    /  1  1  0  0 \
    *    |  1  1  1  0 |
    *    |  0  1  1  1 |
    *    \  0  0  1  1 /
    */

   const IndexType m_rows = 4;
   const IndexType m_cols = 4;

   Matrix m( m_rows, m_cols );
   typename Matrix::RowsCapacitiesType rowLengths;
   rowLengths.setSize( m_rows );
   rowLengths.setValue( 3 );
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
    *    /  1  1  1  0 \
    *    |  0  1  0  1 |
    *    |  1  1  1  0 |
    *    \  0  1  1  1 /
    */

   const IndexType m_rows = 4;
   const IndexType m_cols = 4;

   Matrix savedMatrix( m_rows, m_cols );
   typename Matrix::RowsCapacitiesType rowLengths( m_rows, 3 );
   savedMatrix.setRowCapacities( rowLengths );

   for( IndexType i = 0; i < m_cols - 1; i++ )   // 0th row
       savedMatrix.setElement( 0, i, 1 );

   savedMatrix.setElement( 1, 1, 1 );
   savedMatrix.setElement( 1, 3, 1 );            // 1st row

   for( IndexType i = 0; i < m_cols - 1; i++ )   // 2nd row
       savedMatrix.setElement( 2, i, 1 );

   for( IndexType i = 1; i < m_cols; i++ )       // 3rd row
       savedMatrix.setElement( 3, i, 1 );

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
   EXPECT_EQ( savedMatrix.getElement( 0, 1 ),  1 );
   EXPECT_EQ( savedMatrix.getElement( 0, 2 ),  1 );
   EXPECT_EQ( savedMatrix.getElement( 0, 3 ),  0 );

   EXPECT_EQ( savedMatrix.getElement( 1, 0 ),  0 );
   EXPECT_EQ( savedMatrix.getElement( 1, 1 ),  1 );
   EXPECT_EQ( savedMatrix.getElement( 1, 2 ),  0 );
   EXPECT_EQ( savedMatrix.getElement( 1, 3 ),  1 );

   EXPECT_EQ( savedMatrix.getElement( 2, 0 ),  1 );
   EXPECT_EQ( savedMatrix.getElement( 2, 1 ),  1 );
   EXPECT_EQ( savedMatrix.getElement( 2, 2 ),  1 );
   EXPECT_EQ( savedMatrix.getElement( 2, 3 ),  0 );

   EXPECT_EQ( savedMatrix.getElement( 3, 0 ),  0 );
   EXPECT_EQ( savedMatrix.getElement( 3, 1 ),  1 );
   EXPECT_EQ( savedMatrix.getElement( 3, 2 ),  1 );
   EXPECT_EQ( savedMatrix.getElement( 3, 3 ),  1 );

   EXPECT_EQ( std::remove( filename ), 0 );
}

#endif
