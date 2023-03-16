#include <TNL/Containers/Vector.h>
#include <TNL/Containers/VectorView.h>
#include <TNL/Math.h>
#include <TNL/Algorithms/ParallelFor.h>
#include <TNL/Algorithms/AtomicOperations.h>
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
   using IndexType = typename Matrix::IndexType;

   const IndexType rows = 9;
   const IndexType cols = 8;

   Matrix m;
   m.setDimensions( rows, cols );

   EXPECT_EQ( m.getRows(), 9 );
   EXPECT_EQ( m.getColumns(), 8 );
}

template< typename Matrix >
void test_SetRowCapacities()
{
   using RealType = typename Matrix::RealType;
   using IndexType = typename Matrix::IndexType;

   /*
    /  1  2  4  7                       \
    |  2  3  5  8 10 13 16 19           |
    |  4  5  6    11 14       21 24 27  |
    |  7  8     9       17 20           |
    |    10 11    12          22 25     |
    |    13 14       15             28  |
    |    16    17       18              |
    |    19    20          21           |
    |       21    22          23        |
    |       24    25             26     |
    \       27       28             30  /
    */
   const IndexType rows = 11;
   const IndexType cols = 11;

   Matrix m( rows, cols );
   typename Matrix::RowsCapacitiesType rowLengths { 1, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3  };
   m.setRowCapacities( rowLengths );

   // Insert values into the rows.
   RealType value = 1;

   // 0th row - lower part
   m.setElement( 0, 0, value++ );

   // 1st row - lower part
   m.setElement( 1, 0, value++ );
   m.setElement( 1, 1, value++ );

   // 2nd row - lower part
   m.setElement( 2, 0, value++ );
   m.setElement( 2, 1, value++ );
   m.setElement( 2, 2, value++ );

   // 3rd row - lower part
   m.setElement( 3, 0, value++ );
   m.setElement( 3, 1, value++ );
   m.setElement( 3, 3, value++ );

   // 4th row - lower part
   m.setElement( 4, 1, value++ );
   m.setElement( 4, 2, value++ );
   m.setElement( 4, 4, value++ );

   // 5th row - lower part
   m.setElement( 5, 1, value++ );
   m.setElement( 5, 2, value++ );
   m.setElement( 5, 5, value++ );

   // 6th row - lower part
   m.setElement( 6, 1, value++ );
   m.setElement( 6, 3, value++ );
   m.setElement( 6, 6, value++ );

   // 7th row - lower part
   m.setElement( 7, 1, value++ );
   m.setElement( 7, 3, value++ );
   m.setElement( 7, 7, value++ );

   // 8th row - lower part
   m.setElement( 8, 2, value++ );
   m.setElement( 8, 4, value++ );
   m.setElement( 8, 8, value++ );

   // 8th row - lower part
   m.setElement( 9, 2, value++ );
   m.setElement( 9, 4, value++ );
   m.setElement( 9, 9, value++ );

   // 8th row - lower part
   m.setElement( 10,  2, value++ );
   m.setElement( 10,  5, value++ );
   m.setElement( 10, 10, value++ );

   rowLengths = 0;
   m.getCompressedRowLengths( rowLengths );

   typename Matrix::RowsCapacitiesType correctRowLengths{ 1, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3 };
   EXPECT_EQ( rowLengths, correctRowLengths );
}

template< typename Matrix1, typename Matrix2 >
void test_SetLike()
{
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
   using IndexType = typename Matrix::IndexType;

   /*
    * Sets up the following 10x10 sparse matrix:
    *
    /  1  2  4  7                       \ -> 4
    |  2  3  5  8 10 13 16 19           | -> 8
    |  4  5  6    11 14       21 25 28  | -> 8
    |  7  8     9       17 20           | -> 5
    |    10 11    12          23 26     | -> 5
    |    13 14       15             29  | -> 4
    |    16    17       18              | -> 3
    |    19    20          21           | -> 3
    |       22    23          24        | -> 3
    |       25    26             27     | -> 3
    \       28       29             30  / -> 3
                                          ----
                                            49
    */

   const IndexType rows = 11;
   const IndexType cols = 11;

   Matrix m( rows, cols, {
      { 0, 0,  1 },
      { 1, 0,  2 }, { 1, 1,  3 },
      { 2, 0,  4 }, { 2, 1,  5 }, {  2, 2,  6 },
      { 3, 0,  7 }, { 3, 1,  8 },                { 3, 3,  9 },
                    { 4, 1, 10 }, {  4, 2, 11 },               { 4, 4, 12 },
                    { 5, 1, 13 }, {  5, 2, 14 },                              {  5, 5, 15 },
                    { 6, 1, 16 },                { 6, 3, 17 },                              { 6, 6, 18 },
                    { 7, 1, 19 },                { 7, 3, 20 },                                            { 7, 7, 21 },
                                  {  8, 2, 22 },               { 8, 4, 23 },                                           { 8, 8, 24 },
                                  {  9, 2, 25 },               { 9, 4, 26 },                                                         { 9, 9, 27 },
                                  { 10, 2, 28 },                              { 10, 4, 29 },                                                      { 10, 10, 30 }
   } );

   EXPECT_EQ( m.getNonzeroElementsCount(), 49 );
}

template< typename Matrix >
void test_Reset()
{
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
   using DeviceType = typename Matrix::DeviceType;
   using IndexType = typename Matrix::IndexType;
   /*
    * Sets up the following 11x11 sparse matrix:
    *
    /  1  2  4  7                       \
    |  2  3  5  8 10 13 16 19           |
    |  4  5  6    11 14       22 25 28  |
    |  7  8     9       17 20           |
    |    10 11    12          23 26     |
    |    13 14       15             29  |
    |    16    17       18              |
    |    19    20          21           |
    |       22    23          24        |
    |       25    26             27     |
    \       28       29             30  /
    */

   Matrix m( { 1, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3 }, 11 );

   auto matrixView = m.getView();
   auto f = [=] __cuda_callable__ ( const IndexType rowIdx ) mutable {
      auto row = matrixView.getRow( rowIdx );
      switch( rowIdx )
      {
         case  0: row.setElement( 0, 0,  1 ); break;
         case  1: row.setElement( 0, 0,  2 ); row.setElement( 1, 1,  3 ); break;
         case  2: row.setElement( 0, 0,  4 ); row.setElement( 1, 1,  5 ); row.setElement( 2,  2,  6 ); break;
         case  3: row.setElement( 0, 0,  7 ); row.setElement( 1, 1,  8 ); row.setElement( 2,  3,  9 ); break;
         case  4: row.setElement( 0, 1, 10 ); row.setElement( 1, 2, 11 ); row.setElement( 2,  4, 12 ); break;
         case  5: row.setElement( 0, 1, 13 ); row.setElement( 1, 2, 14 ); row.setElement( 2,  5, 15 ); break;
         case  6: row.setElement( 0, 1, 16 ); row.setElement( 1, 3, 17 ); row.setElement( 2,  6, 18 ); break;
         case  7: row.setElement( 0, 1, 19 ); row.setElement( 1, 3, 20 ); row.setElement( 2,  7, 21 ); break;
         case  8: row.setElement( 0, 2, 22 ); row.setElement( 1, 4, 23 ); row.setElement( 2,  8, 24 ); break;
         case  9: row.setElement( 0, 2, 25 ); row.setElement( 1, 4, 26 ); row.setElement( 2,  9, 27 ); break;
         case 10: row.setElement( 0, 2, 28 ); row.setElement( 1, 5, 29 ); row.setElement( 2, 10, 30 ); break;
      }
   };
   TNL::Algorithms::ParallelFor< DeviceType >::exec( ( IndexType ) 0, m.getRows(), f );

   EXPECT_EQ( m.getElement( 0,  0 ),  1 );
   EXPECT_EQ( m.getElement( 0,  1 ),  2 );
   EXPECT_EQ( m.getElement( 0,  2 ),  4 );
   EXPECT_EQ( m.getElement( 0,  3 ),  7 );
   EXPECT_EQ( m.getElement( 0,  4 ),  0 );
   EXPECT_EQ( m.getElement( 0,  5 ),  0 );
   EXPECT_EQ( m.getElement( 0,  6 ),  0 );
   EXPECT_EQ( m.getElement( 0,  7 ),  0 );
   EXPECT_EQ( m.getElement( 0,  8 ),  0 );
   EXPECT_EQ( m.getElement( 0,  9 ),  0 );
   EXPECT_EQ( m.getElement( 0, 10 ),  0 );

   EXPECT_EQ( m.getElement( 1,  0 ),  2 );
   EXPECT_EQ( m.getElement( 1,  1 ),  3 );
   EXPECT_EQ( m.getElement( 1,  2 ),  5 );
   EXPECT_EQ( m.getElement( 1,  3 ),  8 );
   EXPECT_EQ( m.getElement( 1,  4 ), 10 );
   EXPECT_EQ( m.getElement( 1,  5 ), 13 );
   EXPECT_EQ( m.getElement( 1,  6 ), 16 );
   EXPECT_EQ( m.getElement( 1,  7 ), 19 );
   EXPECT_EQ( m.getElement( 1,  8 ),  0 );
   EXPECT_EQ( m.getElement( 1,  9 ),  0 );
   EXPECT_EQ( m.getElement( 1, 10 ),  0 );

   EXPECT_EQ( m.getElement( 2,  0 ),  4 );
   EXPECT_EQ( m.getElement( 2,  1 ),  5 );
   EXPECT_EQ( m.getElement( 2,  2 ),  6 );
   EXPECT_EQ( m.getElement( 2,  3 ),  0 );
   EXPECT_EQ( m.getElement( 2,  4 ), 11 );
   EXPECT_EQ( m.getElement( 2,  5 ), 14 );
   EXPECT_EQ( m.getElement( 2,  6 ),  0 );
   EXPECT_EQ( m.getElement( 2,  7 ),  0 );
   EXPECT_EQ( m.getElement( 2,  8 ), 22 );
   EXPECT_EQ( m.getElement( 2,  9 ), 25 );
   EXPECT_EQ( m.getElement( 2, 10 ), 28 );

   EXPECT_EQ( m.getElement( 3,  0 ),  7 );
   EXPECT_EQ( m.getElement( 3,  1 ),  8 );
   EXPECT_EQ( m.getElement( 3,  2 ),  0 );
   EXPECT_EQ( m.getElement( 3,  3 ),  9 );
   EXPECT_EQ( m.getElement( 3,  4 ),  0 );
   EXPECT_EQ( m.getElement( 3,  5 ),  0 );
   EXPECT_EQ( m.getElement( 3,  6 ), 17 );
   EXPECT_EQ( m.getElement( 3,  7 ), 20 );
   EXPECT_EQ( m.getElement( 3,  8 ),  0 );
   EXPECT_EQ( m.getElement( 3,  9 ),  0 );
   EXPECT_EQ( m.getElement( 3, 10 ),  0 );

   EXPECT_EQ( m.getElement( 4,  0 ),  0 );
   EXPECT_EQ( m.getElement( 4,  1 ), 10 );
   EXPECT_EQ( m.getElement( 4,  2 ), 11 );
   EXPECT_EQ( m.getElement( 4,  3 ),  0 );
   EXPECT_EQ( m.getElement( 4,  4 ), 12 );
   EXPECT_EQ( m.getElement( 4,  5 ),  0 );
   EXPECT_EQ( m.getElement( 4,  6 ),  0 );
   EXPECT_EQ( m.getElement( 4,  7 ),  0 );
   EXPECT_EQ( m.getElement( 4,  8 ), 23 );
   EXPECT_EQ( m.getElement( 4,  9 ), 26 );
   EXPECT_EQ( m.getElement( 4, 10 ),  0 );

   EXPECT_EQ( m.getElement( 5,  0 ),  0 );
   EXPECT_EQ( m.getElement( 5,  1 ), 13 );
   EXPECT_EQ( m.getElement( 5,  2 ), 14 );
   EXPECT_EQ( m.getElement( 5,  3 ),  0 );
   EXPECT_EQ( m.getElement( 5,  4 ),  0 );
   EXPECT_EQ( m.getElement( 5,  5 ), 15 );
   EXPECT_EQ( m.getElement( 5,  6 ),  0 );
   EXPECT_EQ( m.getElement( 5,  7 ),  0 );
   EXPECT_EQ( m.getElement( 5,  8 ),  0 );
   EXPECT_EQ( m.getElement( 5,  9 ),  0 );
   EXPECT_EQ( m.getElement( 5, 10 ), 29 );

   EXPECT_EQ( m.getElement( 6,  0 ),  0 );
   EXPECT_EQ( m.getElement( 6,  1 ), 16 );
   EXPECT_EQ( m.getElement( 6,  2 ),  0 );
   EXPECT_EQ( m.getElement( 6,  3 ), 17 );
   EXPECT_EQ( m.getElement( 6,  4 ),  0 );
   EXPECT_EQ( m.getElement( 6,  5 ),  0 );
   EXPECT_EQ( m.getElement( 6,  6 ), 18 );
   EXPECT_EQ( m.getElement( 6,  7 ),  0 );
   EXPECT_EQ( m.getElement( 6,  8 ),  0 );
   EXPECT_EQ( m.getElement( 6,  9 ),  0 );
   EXPECT_EQ( m.getElement( 6, 10 ),  0 );

   EXPECT_EQ( m.getElement( 7,  0 ),  0 );
   EXPECT_EQ( m.getElement( 7,  1 ), 19 );
   EXPECT_EQ( m.getElement( 7,  2 ),  0 );
   EXPECT_EQ( m.getElement( 7,  3 ), 20 );
   EXPECT_EQ( m.getElement( 7,  4 ),  0 );
   EXPECT_EQ( m.getElement( 7,  5 ),  0 );
   EXPECT_EQ( m.getElement( 7,  6 ),  0 );
   EXPECT_EQ( m.getElement( 7,  7 ), 21 );
   EXPECT_EQ( m.getElement( 7,  8 ),  0 );
   EXPECT_EQ( m.getElement( 7,  9 ),  0 );
   EXPECT_EQ( m.getElement( 7, 10 ),  0 );

   EXPECT_EQ( m.getElement( 8,  0 ),  0 );
   EXPECT_EQ( m.getElement( 8,  1 ),  0 );
   EXPECT_EQ( m.getElement( 8,  2 ), 22 );
   EXPECT_EQ( m.getElement( 8,  3 ),  0 );
   EXPECT_EQ( m.getElement( 8,  4 ), 23 );
   EXPECT_EQ( m.getElement( 8,  5 ),  0 );
   EXPECT_EQ( m.getElement( 8,  6 ),  0 );
   EXPECT_EQ( m.getElement( 8,  7 ),  0 );
   EXPECT_EQ( m.getElement( 8,  8 ), 24 );
   EXPECT_EQ( m.getElement( 8,  9 ),  0 );
   EXPECT_EQ( m.getElement( 8, 10 ),  0 );

   EXPECT_EQ( m.getElement( 9,  0 ),  0 );
   EXPECT_EQ( m.getElement( 9,  1 ),  0 );
   EXPECT_EQ( m.getElement( 9,  2 ), 25 );
   EXPECT_EQ( m.getElement( 9,  3 ),  0 );
   EXPECT_EQ( m.getElement( 9,  4 ), 26 );
   EXPECT_EQ( m.getElement( 9,  5 ),  0 );
   EXPECT_EQ( m.getElement( 9,  6 ),  0 );
   EXPECT_EQ( m.getElement( 9,  7 ),  0 );
   EXPECT_EQ( m.getElement( 9,  8 ),  0 );
   EXPECT_EQ( m.getElement( 9,  9 ), 27 );
   EXPECT_EQ( m.getElement( 9, 10 ),  0 );

   EXPECT_EQ( m.getElement( 10,  0 ),  0 );
   EXPECT_EQ( m.getElement( 10,  1 ),  0 );
   EXPECT_EQ( m.getElement( 10,  2 ), 28 );
   EXPECT_EQ( m.getElement( 10,  3 ),  0 );
   EXPECT_EQ( m.getElement( 10,  4 ),  0 );
   EXPECT_EQ( m.getElement( 10,  5 ), 29 );
   EXPECT_EQ( m.getElement( 10,  6 ),  0 );
   EXPECT_EQ( m.getElement( 10,  7 ),  0 );
   EXPECT_EQ( m.getElement( 10,  8 ),  0 );
   EXPECT_EQ( m.getElement( 10,  9 ),  0 );
   EXPECT_EQ( m.getElement( 10, 10 ), 30 );
}


template< typename Matrix >
void test_SetElement()
{
   using RealType = typename Matrix::RealType;
   using IndexType = typename Matrix::IndexType;

   /*
    * Sets up the following 10x10 sparse matrix:
    *
    *    /  1  0  0  4  0  0 10  0  0  0  \
    *    |  0  2  0  5  0  0 11  0  0  0  |
    *    |  0  0  3  6  0  0 12  0  0  0  |
    *    |  4  5  6  7  0  0 13  0  0  0  |
    *    |  0  0  0  0  8  0 14  0  0  0  |
    *    |  0  0  0  0  0  9 15  0  0  0  |
    *    | 10 11 12 13 14 15 16  0  0  0  |
    *    |  0  0  0  0  0  0  0 17  0  0  |
    *    |  0  0  0  0  0  0  0  0 18  0  |
    *    \  0  0  0  0  0  0  0  0  0 19 /
    */

   Matrix m( { 1, 1, 1, 4, 1, 1, 7, 1, 1, 1 }, 10 );

   RealType value = 1;
   for( IndexType i = 0; i < 3; i++ )
      m.setElement( i, i, value++ );

   for( IndexType i = 0; i < 4; i++ )
      m.setElement( 3, i, value++ );

   for( IndexType i = 4; i < 6; i++ )
      m.setElement( i, i, value++ );

   for( IndexType i = 0; i < 7; i++ )
      m.setElement( 6, i, value++ );

   for( IndexType i = 7; i < 10; i++ )
      m.setElement( i, i, value++ );

   EXPECT_EQ( m.getElement( 0, 0 ),  1 );
   EXPECT_EQ( m.getElement( 0, 1 ),  0 );
   EXPECT_EQ( m.getElement( 0, 2 ),  0 );
   EXPECT_EQ( m.getElement( 0, 3 ),  4 );
   EXPECT_EQ( m.getElement( 0, 4 ),  0 );
   EXPECT_EQ( m.getElement( 0, 5 ),  0 );
   EXPECT_EQ( m.getElement( 0, 6 ), 10 );
   EXPECT_EQ( m.getElement( 0, 7 ),  0 );
   EXPECT_EQ( m.getElement( 0, 8 ),  0 );
   EXPECT_EQ( m.getElement( 0, 9 ),  0 );

   EXPECT_EQ( m.getElement( 1, 0 ),  0 );
   EXPECT_EQ( m.getElement( 1, 1 ),  2 );
   EXPECT_EQ( m.getElement( 1, 2 ),  0 );
   EXPECT_EQ( m.getElement( 1, 3 ),  5 );
   EXPECT_EQ( m.getElement( 1, 4 ),  0 );
   EXPECT_EQ( m.getElement( 1, 5 ),  0 );
   EXPECT_EQ( m.getElement( 1, 6 ), 11 );
   EXPECT_EQ( m.getElement( 1, 7 ),  0 );
   EXPECT_EQ( m.getElement( 1, 8 ),  0 );
   EXPECT_EQ( m.getElement( 1, 9 ),  0 );

   EXPECT_EQ( m.getElement( 2, 0 ),  0 );
   EXPECT_EQ( m.getElement( 2, 1 ),  0 );
   EXPECT_EQ( m.getElement( 2, 2 ),  3 );
   EXPECT_EQ( m.getElement( 2, 3 ),  6 );
   EXPECT_EQ( m.getElement( 2, 4 ),  0 );
   EXPECT_EQ( m.getElement( 2, 5 ),  0 );
   EXPECT_EQ( m.getElement( 2, 6 ), 12 );
   EXPECT_EQ( m.getElement( 2, 7 ),  0 );
   EXPECT_EQ( m.getElement( 2, 8 ),  0 );
   EXPECT_EQ( m.getElement( 2, 9 ),  0 );

   EXPECT_EQ( m.getElement( 3, 0 ),  4 );
   EXPECT_EQ( m.getElement( 3, 1 ),  5 );
   EXPECT_EQ( m.getElement( 3, 2 ),  6 );
   EXPECT_EQ( m.getElement( 3, 3 ),  7 );
   EXPECT_EQ( m.getElement( 3, 4 ),  0 );
   EXPECT_EQ( m.getElement( 3, 5 ),  0 );
   EXPECT_EQ( m.getElement( 3, 6 ), 13 );
   EXPECT_EQ( m.getElement( 3, 7 ),  0 );
   EXPECT_EQ( m.getElement( 3, 8 ),  0 );
   EXPECT_EQ( m.getElement( 3, 9 ),  0 );

   EXPECT_EQ( m.getElement( 4, 0 ),  0 );
   EXPECT_EQ( m.getElement( 4, 1 ),  0 );
   EXPECT_EQ( m.getElement( 4, 2 ),  0 );
   EXPECT_EQ( m.getElement( 4, 3 ),  0 );
   EXPECT_EQ( m.getElement( 4, 4 ),  8 );
   EXPECT_EQ( m.getElement( 4, 5 ),  0 );
   EXPECT_EQ( m.getElement( 4, 6 ), 14 );
   EXPECT_EQ( m.getElement( 4, 7 ),  0 );
   EXPECT_EQ( m.getElement( 4, 8 ),  0 );
   EXPECT_EQ( m.getElement( 4, 9 ),  0 );

   EXPECT_EQ( m.getElement( 5, 0 ),  0 );
   EXPECT_EQ( m.getElement( 5, 1 ),  0 );
   EXPECT_EQ( m.getElement( 5, 2 ),  0 );
   EXPECT_EQ( m.getElement( 5, 3 ),  0 );
   EXPECT_EQ( m.getElement( 5, 4 ),  0 );
   EXPECT_EQ( m.getElement( 5, 5 ),  9 );
   EXPECT_EQ( m.getElement( 5, 6 ), 15 );
   EXPECT_EQ( m.getElement( 5, 7 ),  0 );
   EXPECT_EQ( m.getElement( 5, 8 ),  0 );
   EXPECT_EQ( m.getElement( 5, 9 ),  0 );

   EXPECT_EQ( m.getElement( 6, 0 ), 10 );
   EXPECT_EQ( m.getElement( 6, 1 ), 11 );
   EXPECT_EQ( m.getElement( 6, 2 ), 12 );
   EXPECT_EQ( m.getElement( 6, 3 ), 13 );
   EXPECT_EQ( m.getElement( 6, 4 ), 14 );
   EXPECT_EQ( m.getElement( 6, 5 ), 15 );
   EXPECT_EQ( m.getElement( 6, 6 ), 16 );
   EXPECT_EQ( m.getElement( 6, 7 ),  0 );
   EXPECT_EQ( m.getElement( 6, 8 ),  0 );
   EXPECT_EQ( m.getElement( 6, 9 ),  0 );

   EXPECT_EQ( m.getElement( 7, 0 ),  0 );
   EXPECT_EQ( m.getElement( 7, 1 ),  0 );
   EXPECT_EQ( m.getElement( 7, 2 ),  0 );
   EXPECT_EQ( m.getElement( 7, 3 ),  0 );
   EXPECT_EQ( m.getElement( 7, 4 ),  0 );
   EXPECT_EQ( m.getElement( 7, 5 ),  0 );
   EXPECT_EQ( m.getElement( 7, 6 ),  0 );
   EXPECT_EQ( m.getElement( 7, 7 ), 17 );
   EXPECT_EQ( m.getElement( 7, 8 ),  0 );
   EXPECT_EQ( m.getElement( 7, 9 ),  0 );

   EXPECT_EQ( m.getElement( 8, 0 ),  0 );
   EXPECT_EQ( m.getElement( 8, 1 ),  0 );
   EXPECT_EQ( m.getElement( 8, 2 ),  0 );
   EXPECT_EQ( m.getElement( 8, 3 ),  0 );
   EXPECT_EQ( m.getElement( 8, 4 ),  0 );
   EXPECT_EQ( m.getElement( 8, 5 ),  0 );
   EXPECT_EQ( m.getElement( 8, 6 ),  0 );
   EXPECT_EQ( m.getElement( 8, 7 ),  0 );
   EXPECT_EQ( m.getElement( 8, 8 ), 18 );
   EXPECT_EQ( m.getElement( 8, 9 ),  0 );

   EXPECT_EQ( m.getElement( 9, 0 ),  0 );
   EXPECT_EQ( m.getElement( 9, 1 ),  0 );
   EXPECT_EQ( m.getElement( 9, 2 ),  0 );
   EXPECT_EQ( m.getElement( 9, 3 ),  0 );
   EXPECT_EQ( m.getElement( 9, 4 ),  0 );
   EXPECT_EQ( m.getElement( 9, 5 ),  0 );
   EXPECT_EQ( m.getElement( 9, 6 ),  0 );
   EXPECT_EQ( m.getElement( 9, 7 ),  0 );
   EXPECT_EQ( m.getElement( 9, 8 ),  0 );
   EXPECT_EQ( m.getElement( 9, 9 ), 19 );
}

template< typename Matrix >
void test_AddElement()
{
   using IndexType = typename Matrix::IndexType;

   /*
    * Sets up the following 6x5 sparse matrix:
    *
    *    /  1  2  0  0  0 \
    *    |  2  3  4  0  0 |
    *    |  0  4  5  6  0 |
    *    |  0  0  6  7  8 |
    *    |  0  0  0  8  9 |
    *    \  0  0  0  0 10 /
    */

   const IndexType rows = 6;
   const IndexType cols = 5;

   Matrix m( 6, 5, {
      { 0, 0, 1 },
      { 1, 0, 2 }, { 1, 1, 3 },
                   { 2, 1, 4 }, { 2, 2, 5 },
                                { 3, 2, 6 }, { 3, 3, 7 },
                                             { 4, 3, 8 }, { 4, 4,  9 },
                                                          { 5, 4, 10 } } );

   // Check the set elements
   EXPECT_EQ( m.getElement( 0, 0 ),  1 );
   EXPECT_EQ( m.getElement( 0, 1 ),  2 );
   EXPECT_EQ( m.getElement( 0, 2 ),  0 );
   EXPECT_EQ( m.getElement( 0, 3 ),  0 );
   EXPECT_EQ( m.getElement( 0, 4 ),  0 );

   EXPECT_EQ( m.getElement( 1, 0 ),  2 );
   EXPECT_EQ( m.getElement( 1, 1 ),  3 );
   EXPECT_EQ( m.getElement( 1, 2 ),  4 );
   EXPECT_EQ( m.getElement( 1, 3 ),  0 );
   EXPECT_EQ( m.getElement( 1, 4 ),  0 );

   EXPECT_EQ( m.getElement( 2, 0 ),  0 );
   EXPECT_EQ( m.getElement( 2, 1 ),  4 );
   EXPECT_EQ( m.getElement( 2, 2 ),  5 );
   EXPECT_EQ( m.getElement( 2, 3 ),  6 );
   EXPECT_EQ( m.getElement( 2, 4 ),  0 );

   EXPECT_EQ( m.getElement( 3, 0 ),  0 );
   EXPECT_EQ( m.getElement( 3, 1 ),  0 );
   EXPECT_EQ( m.getElement( 3, 2 ),  6 );
   EXPECT_EQ( m.getElement( 3, 3 ),  7 );
   EXPECT_EQ( m.getElement( 3, 4 ),  8 );

   EXPECT_EQ( m.getElement( 4, 0 ),  0 );
   EXPECT_EQ( m.getElement( 4, 1 ),  0 );
   EXPECT_EQ( m.getElement( 4, 2 ),  0 );
   EXPECT_EQ( m.getElement( 4, 3 ),  8 );
   EXPECT_EQ( m.getElement( 4, 4 ),  9 );

   EXPECT_EQ( m.getElement( 5, 0 ),  0 );
   EXPECT_EQ( m.getElement( 5, 1 ),  0 );
   EXPECT_EQ( m.getElement( 5, 2 ),  0 );
   EXPECT_EQ( m.getElement( 5, 3 ),  0 );
   EXPECT_EQ( m.getElement( 5, 4 ), 10 );

   // Add new elements to the old elements with a multiplying factor applied to the old elements.
   /*
    * The following setup results in the following 6x5 sparse matrix:
    *
    *    /  1  2  0  0  0 \   /  0  1  0  0  0 \   /  2  5  0  0  0 \
    *    |  2  3  4  0  0 |   |  1  0  1  0  0 |   |  5  6  9  0  0 |
    * 2  |  0  4  5  6  0 | + |  0  1  0  1  0 | = |  0  9 10 13  0 |
    *    |  0  0  6  7  8 |   |  0  0  1  0  1 |   |  0  0 13 14 17 |
    *    |  0  0  0  8  9 |   |  0  0  0  1  0 |   |  0  0  0 17 18 |
    *    \  0  0  0  0 10 /   \  0  0  0  0  1 /   \  0  0  0  0 21 /
    */

   for( IndexType i = 0; i < rows; i++ )
   {
      if( i > 0 )
         m.addElement( i, i - 1, 1.0, 2.0 );
      if( i < cols )
         m.addElement( i, i, 0.0, 2.0 );
   }

   EXPECT_EQ( m.getElement( 0, 0 ),  2 );
   EXPECT_EQ( m.getElement( 0, 1 ),  5 );
   EXPECT_EQ( m.getElement( 0, 2 ),  0 );
   EXPECT_EQ( m.getElement( 0, 3 ),  0 );
   EXPECT_EQ( m.getElement( 0, 4 ),  0 );

   EXPECT_EQ( m.getElement( 1, 0 ),  5 );
   EXPECT_EQ( m.getElement( 1, 1 ),  6 );
   EXPECT_EQ( m.getElement( 1, 2 ),  9 );
   EXPECT_EQ( m.getElement( 1, 3 ),  0 );
   EXPECT_EQ( m.getElement( 1, 4 ),  0 );

   EXPECT_EQ( m.getElement( 2, 0 ),  0 );
   EXPECT_EQ( m.getElement( 2, 1 ),  9 );
   EXPECT_EQ( m.getElement( 2, 2 ), 10 );
   EXPECT_EQ( m.getElement( 2, 3 ), 13 );
   EXPECT_EQ( m.getElement( 2, 4 ),  0 );

   EXPECT_EQ( m.getElement( 3, 0 ),  0 );
   EXPECT_EQ( m.getElement( 3, 1 ),  0 );
   EXPECT_EQ( m.getElement( 3, 2 ), 13 );
   EXPECT_EQ( m.getElement( 3, 3 ), 14 );
   EXPECT_EQ( m.getElement( 3, 4 ), 17 );

   EXPECT_EQ( m.getElement( 4, 0 ),  0 );
   EXPECT_EQ( m.getElement( 4, 1 ),  0 );
   EXPECT_EQ( m.getElement( 4, 2 ),  0 );
   EXPECT_EQ( m.getElement( 4, 3 ), 17 );
   EXPECT_EQ( m.getElement( 4, 4 ), 18 );

   EXPECT_EQ( m.getElement( 5, 0 ),  0 );
   EXPECT_EQ( m.getElement( 5, 1 ),  0 );
   EXPECT_EQ( m.getElement( 5, 2 ),  0 );
   EXPECT_EQ( m.getElement( 5, 3 ),  0 );
   EXPECT_EQ( m.getElement( 5, 4 ), 21 );
}

template< typename Matrix >
void test_VectorProduct()
{
   using RealType = typename Matrix::RealType;
   using DeviceType = typename Matrix::DeviceType;
   using IndexType = typename Matrix::IndexType;
   using VectorType = TNL::Containers::Vector< RealType, DeviceType, IndexType >;

   /**
    * Sets up the following 4x4 sparse matrix:
    *
    *    /  1  0  0  0 \
    *    |  0  2  3  4 |
    *    |  0  3  0  5 |
    *    \  0  4  5  0 /
    */

   const IndexType m_rows_1 = 4;
   const IndexType m_cols_1 = 4;

   Matrix m_1( m_rows_1, m_cols_1, {
      { 0, 0, 1 },
                   { 1, 1, 2 },
                   { 2, 1, 3 },
                   { 3, 1, 4 }, { 3, 2, 5 } } );

   VectorType inVector_1( m_cols_1, 2.0 );
   VectorType outVector_1( m_rows_1, 0.0 );
   m_1.vectorProduct( inVector_1, outVector_1 );

   EXPECT_EQ( outVector_1.getElement( 0 ),  2 );
   EXPECT_EQ( outVector_1.getElement( 1 ), 18 );
   EXPECT_EQ( outVector_1.getElement( 2 ), 16 );
   EXPECT_EQ( outVector_1.getElement( 3 ), 18 );

   /**
    * Sets up the following 4x4 sparse matrix:
    *
    *    /  1  2  3  0 \
    *    |  2  0  6  8 |
    *    |  3  6  7  0 |
    *    \  0  8  0  9 /
    */

   const IndexType m_rows_2 = 4;
   const IndexType m_cols_2 = 4;

   Matrix m_2( m_rows_2, m_cols_2, {
      { 0, 0, 1 },
      { 1, 0, 2 },
      { 2, 0, 3 }, { 2, 1, 6 }, { 2, 2, 7 },
                   { 3, 1, 8 },              { 3, 3, 9 } } );

   VectorType inVector_2( m_cols_2, 2 );
   VectorType outVector_2( m_rows_2, 0 );
   m_2.vectorProduct( inVector_2, outVector_2 );

   EXPECT_EQ( outVector_2.getElement( 0 ), 12 );
   EXPECT_EQ( outVector_2.getElement( 1 ), 32 );
   EXPECT_EQ( outVector_2.getElement( 2 ), 32 );
   EXPECT_EQ( outVector_2.getElement( 3 ), 34 );

   /*
    * Sets up the following 4x4 sparse matrix:
    *
    *    /  1  2  3  4 \
    *    |  2  5  0  0 |
    *    |  3  0  6  0 |
    *    \  4  0  0  7 /
    */

   const IndexType m_rows_3 = 4;
   const IndexType m_cols_3 = 4;

   Matrix m_3( m_rows_3, m_cols_3, {
      { 0, 0, 1 }, { 0, 1, 2 }, { 0, 2, 3 }, { 0, 3, 4 },
      { 1, 0, 2 }, { 1, 1, 5 },
      { 2, 0, 3 }, { 2, 2, 6 },
      { 3, 0, 4 }, { 3, 3, 7 }
   } );

   VectorType inVector_3( { 0, 1, 2, 3 } );
   VectorType outVector_3( m_rows_3, 0 );
   m_3.vectorProduct( inVector_3, outVector_3 );

   EXPECT_EQ( outVector_3.getElement( 0 ), 20 );
   EXPECT_EQ( outVector_3.getElement( 1 ),  5 );
   EXPECT_EQ( outVector_3.getElement( 2 ), 12 );
   EXPECT_EQ( outVector_3.getElement( 3 ), 21 );

   /*
    * Sets up the following 8x8 sparse matrix:
    *
    *    /  1  0  3  0  9  0 15  0 \
    *    |  0  2  0  6  0 12  0 19 |
    *    |  3  0  5  0 10  0 16  0 |
    *    |  0  6  0  8  0 13  0 20 |
    *    |  9  0 10  0 11  0 17  0 |
    *    |  0 12  0 13  0 14  0 21 |
    *    | 15  0 16  0 17  0 18  0 |
    *    \  0 19  0 20  0 21  0 22 /
    */

   const IndexType m_rows_4 = 8;
   const IndexType m_cols_4 = 8;

   Matrix m_4( m_rows_4, m_cols_4, {
      { 0, 0,  1 },
                    { 1, 1,  2 },
      { 2, 0,  3 },               { 2, 2, 5 },
                    { 3, 1,  6 },               { 3, 3, 8 },
      { 4, 0,  9 },               { 4, 2, 10 },               { 4, 4, 11 },
                    { 5, 1, 12 },               { 5, 3, 13 },               { 5, 5, 14 },
      { 6, 0, 15 },               { 6, 2, 16 },               { 6, 4, 17 },               { 6, 6, 18 },
                    { 7, 1, 19 },               { 7, 3, 20 },               { 7, 5, 21 },               { 7, 7, 22 }
   } );

   VectorType inVector_4 { 1, 2, 1, 2, 1, 2, 1, 2 };
   VectorType outVector_4( m_rows_4, 0 );
   m_4.vectorProduct( inVector_4, outVector_4 );

   EXPECT_EQ( outVector_4.getElement( 0 ),  28 );
   EXPECT_EQ( outVector_4.getElement( 1 ),  78 );
   EXPECT_EQ( outVector_4.getElement( 2 ),  34 );
   EXPECT_EQ( outVector_4.getElement( 3 ),  94 );
   EXPECT_EQ( outVector_4.getElement( 4 ),  47 );
   EXPECT_EQ( outVector_4.getElement( 5 ), 120 );
   EXPECT_EQ( outVector_4.getElement( 6 ),  66 );
   EXPECT_EQ( outVector_4.getElement( 7 ), 164 );

   /*
    * Sets up the following 8x8 sparse matrix:
    *
    *    /  1  0  0  0  0  0  0  0 \
    *    |  0  2  0  0  0  0  0  0 |
    *    |  0  0  3  4  6  9  0  0 |
    *    |  0  0  4  5  7 10  0  0 |
    *    |  0  0  6  7  8 11  0  0 |
    *    |  0  0  9 10 11 12  0  0 |
    *    |  0  0  0  0  0  0 13  0 |
    *    \  0  0  0  0  0  0  0 14 /
    */

   const IndexType m_rows_5 = 8;
   const IndexType m_cols_5 = 8;

   Matrix m_5( m_rows_5, m_cols_5,{
      { 0, 0, 1 },
                   { 1, 1, 2, },
                                 { 2, 2, 3 },
                                 { 3, 2, 4 }, { 3, 3,  5 },
                                 { 4, 2, 6 }, { 4, 3,  7 }, { 4, 4,  8 },
                                 { 5, 2, 9 }, { 5, 3, 10 }, { 5, 4, 11 }, { 5, 5, 12 },
                                                                                        { 6, 6, 13 },
                                                                                                      { 7, 7, 14 }
   } );

   VectorType inVector_5( { 1, 2, 3, 4, 5, 6, 7, 8 } );
   VectorType outVector_5( m_rows_5, 0.0 );
   m_5.vectorProduct( inVector_5, outVector_5 );

   EXPECT_EQ( outVector_5.getElement( 0 ), 1*1 );
   EXPECT_EQ( outVector_5.getElement( 1 ), 2*2 );
   EXPECT_EQ( outVector_5.getElement( 2 ), 3*3 + 4*4  + 5*6  + 6*9 );
   EXPECT_EQ( outVector_5.getElement( 3 ), 3*4 + 4*5  + 5*7  + 6*10 );
   EXPECT_EQ( outVector_5.getElement( 4 ), 3*6 + 4*7  + 5*8  + 6*11 );
   EXPECT_EQ( outVector_5.getElement( 5 ), 3*9 + 4*10 + 5*11 + 6*12 );
   EXPECT_EQ( outVector_5.getElement( 6 ), 7*13 );
   EXPECT_EQ( outVector_5.getElement( 7 ), 8*14 );
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
    *    /  1  0  0  0  0  0  0  0 \
    *    |  0  2  0  0  0  0  0  0 |
    *    |  0  0  3  4  6  9  0  0 |
    *    |  0  0  4  5  7 10  0  0 |
    *    |  0  0  6  7  8 11  0  0 |
    *    |  0  0  9 10 11 12  0  0 |
    *    |  0  0  0  0  0  0 13  0 |
    *    \  0  0  0  0  0  0  0 14 /
    */

   const IndexType m_rows_5 = 8;
   const IndexType m_cols_5 = 8;

   Matrix m_5( m_rows_5, m_cols_5,{
      { 0, 0, 1 },
                   { 1, 1, 2, },
                                 { 2, 2, 3 },
                                 { 3, 2, 4 }, { 3, 3,  5 },
                                 { 4, 2, 6 }, { 4, 3,  7 }, { 4, 4,  8 },
                                 { 5, 2, 9 }, { 5, 3, 10 }, { 5, 4, 11 }, { 5, 5, 12 },
                                                                                        { 6, 6, 13 },
                                                                                                      { 7, 7, 14 }
   } );

   ////
   // Compute number of non-zero elements in rows.
   typename Matrix::RowsCapacitiesType rowLengths( m_rows_5 );
   typename Matrix::RowsCapacitiesType rowLengths_true( { 1, 1, 4, 4, 4, 4, 1, 1 } );
   auto rowLengths_view = rowLengths.getView();
   rowLengths_view = 0;
   auto fetch = [=] __cuda_callable__ ( IndexType row, IndexType column, const RealType& value ) mutable -> IndexType {
      if( value != 0.0 && row != column)
         TNL::Algorithms::AtomicOperations< DeviceType >::add( rowLengths_view[ column ], ( IndexType ) 1 );
      return ( value != 0.0 );
   };
   auto keep = [=] __cuda_callable__ ( const IndexType rowIdx, const IndexType value ) mutable {
      rowLengths_view[ rowIdx ] += value;
   };
   m_5.reduceAllRows( fetch, std::plus<>{}, keep, 0 );

   EXPECT_EQ( rowLengths_true, rowLengths );
   m_5.getCompressedRowLengths( rowLengths );
   typename Matrix::RowsCapacitiesType rowLengths_symmetric( { 1, 1, 1, 2, 3, 4, 1, 1 } );
   EXPECT_EQ( rowLengths_symmetric, rowLengths );

   ////
   // Compute max norm
   /*TNL::Containers::Vector< RealType, DeviceType, IndexType > rowSums( m_5.getRows() );
   auto rowSums_view = rowSums.getView();
   auto max_fetch = [] __cuda_callable__ ( IndexType row, IndexType column, IndexType globalIdx, const RealType& value ) -> IndexType {
      return TNL::abs( value );
   };
   auto max_reduce = [] __cuda_callable__ ( IndexType& aux, const IndexType a ) {
      aux += a;
   };
   auto max_keep = [=] __cuda_callable__ ( const IndexType rowIdx, const IndexType value ) mutable {
      rowSums_view[ rowIdx ] = value;
   };
   m_5.reduceAllRows( max_fetch, max_reduce, max_keep, 0 );
   const RealType maxNorm = TNL::max( rowSums );
   EXPECT_EQ( maxNorm, 260 ) ; // 29+30+31+32+33+34+35+36*/
}

template< typename Matrix >
void test_SaveAndLoad( const char* filename )
{
   /*
    * Sets up the following 6x5 sparse matrix:
    *
    *    /  1  2  0  0  0 \
    *    |  2  3  4  0  0 |
    *    |  0  4  5  6  0 |
    *    |  0  0  6  7  8 |
    *    |  0  0  0  8  9 |
    *    \  0  0  0  0 10 /
    */

   //const IndexType rows = 6;
   //const IndexType cols = 5;

   Matrix savedMatrix( 6, 5, {
      { 0, 0, 1 },
      { 1, 0, 2 }, { 1, 1, 3 },
                   { 2, 1, 4 }, { 2, 2, 5 },
                                { 3, 2, 6 }, { 3, 3, 7 },
                                             { 4, 3, 8 }, { 4, 4,  9 },
                                                          { 5, 4, 10 } } );

   // Check the set elements
   EXPECT_EQ( savedMatrix.getElement( 0, 0 ),  1 );
   EXPECT_EQ( savedMatrix.getElement( 0, 1 ),  2 );
   EXPECT_EQ( savedMatrix.getElement( 0, 2 ),  0 );
   EXPECT_EQ( savedMatrix.getElement( 0, 3 ),  0 );
   EXPECT_EQ( savedMatrix.getElement( 0, 4 ),  0 );

   EXPECT_EQ( savedMatrix.getElement( 1, 0 ),  2 );
   EXPECT_EQ( savedMatrix.getElement( 1, 1 ),  3 );
   EXPECT_EQ( savedMatrix.getElement( 1, 2 ),  4 );
   EXPECT_EQ( savedMatrix.getElement( 1, 3 ),  0 );
   EXPECT_EQ( savedMatrix.getElement( 1, 4 ),  0 );

   EXPECT_EQ( savedMatrix.getElement( 2, 0 ),  0 );
   EXPECT_EQ( savedMatrix.getElement( 2, 1 ),  4 );
   EXPECT_EQ( savedMatrix.getElement( 2, 2 ),  5 );
   EXPECT_EQ( savedMatrix.getElement( 2, 3 ),  6 );
   EXPECT_EQ( savedMatrix.getElement( 2, 4 ),  0 );

   EXPECT_EQ( savedMatrix.getElement( 3, 0 ),  0 );
   EXPECT_EQ( savedMatrix.getElement( 3, 1 ),  0 );
   EXPECT_EQ( savedMatrix.getElement( 3, 2 ),  6 );
   EXPECT_EQ( savedMatrix.getElement( 3, 3 ),  7 );
   EXPECT_EQ( savedMatrix.getElement( 3, 4 ),  8 );

   EXPECT_EQ( savedMatrix.getElement( 4, 0 ),  0 );
   EXPECT_EQ( savedMatrix.getElement( 4, 1 ),  0 );
   EXPECT_EQ( savedMatrix.getElement( 4, 2 ),  0 );
   EXPECT_EQ( savedMatrix.getElement( 4, 3 ),  8 );
   EXPECT_EQ( savedMatrix.getElement( 4, 4 ),  9 );

   EXPECT_EQ( savedMatrix.getElement( 5, 0 ),  0 );
   EXPECT_EQ( savedMatrix.getElement( 5, 1 ),  0 );
   EXPECT_EQ( savedMatrix.getElement( 5, 2 ),  0 );
   EXPECT_EQ( savedMatrix.getElement( 5, 3 ),  0 );
   EXPECT_EQ( savedMatrix.getElement( 5, 4 ), 10 );

   ASSERT_NO_THROW( savedMatrix.save( filename ) );

   Matrix loadedMatrix;

   ASSERT_NO_THROW( loadedMatrix.load( filename ) );

   EXPECT_EQ( savedMatrix.getElement( 0, 0 ), loadedMatrix.getElement( 0, 0 ) );
   EXPECT_EQ( savedMatrix.getElement( 0, 1 ), loadedMatrix.getElement( 0, 1 ) );
   EXPECT_EQ( savedMatrix.getElement( 0, 2 ), loadedMatrix.getElement( 0, 2 ) );
   EXPECT_EQ( savedMatrix.getElement( 0, 3 ), loadedMatrix.getElement( 0, 3 ) );
   EXPECT_EQ( savedMatrix.getElement( 0, 4 ), loadedMatrix.getElement( 0, 4 ) );

   EXPECT_EQ( savedMatrix.getElement( 1, 0 ), loadedMatrix.getElement( 1, 0 ) );
   EXPECT_EQ( savedMatrix.getElement( 1, 1 ), loadedMatrix.getElement( 1, 1 ) );
   EXPECT_EQ( savedMatrix.getElement( 1, 2 ), loadedMatrix.getElement( 1, 2 ) );
   EXPECT_EQ( savedMatrix.getElement( 1, 3 ), loadedMatrix.getElement( 1, 3 ) );
   EXPECT_EQ( savedMatrix.getElement( 1, 4 ), loadedMatrix.getElement( 1, 4 ) );

   EXPECT_EQ( savedMatrix.getElement( 2, 0 ), loadedMatrix.getElement( 2, 0 ) );
   EXPECT_EQ( savedMatrix.getElement( 2, 1 ), loadedMatrix.getElement( 2, 1 ) );
   EXPECT_EQ( savedMatrix.getElement( 2, 2 ), loadedMatrix.getElement( 2, 2 ) );
   EXPECT_EQ( savedMatrix.getElement( 2, 3 ), loadedMatrix.getElement( 2, 3 ) );
   EXPECT_EQ( savedMatrix.getElement( 2, 4 ), loadedMatrix.getElement( 2, 4 ) );

   EXPECT_EQ( savedMatrix.getElement( 3, 0 ), loadedMatrix.getElement( 3, 0 ) );
   EXPECT_EQ( savedMatrix.getElement( 3, 1 ), loadedMatrix.getElement( 3, 1 ) );
   EXPECT_EQ( savedMatrix.getElement( 3, 2 ), loadedMatrix.getElement( 3, 2 ) );
   EXPECT_EQ( savedMatrix.getElement( 3, 3 ), loadedMatrix.getElement( 3, 3 ) );
   EXPECT_EQ( savedMatrix.getElement( 3, 4 ), loadedMatrix.getElement( 3, 4 ) );

   EXPECT_EQ( savedMatrix.getElement( 4, 0 ), loadedMatrix.getElement( 4, 0 ) );
   EXPECT_EQ( savedMatrix.getElement( 4, 1 ), loadedMatrix.getElement( 4, 1 ) );
   EXPECT_EQ( savedMatrix.getElement( 4, 2 ), loadedMatrix.getElement( 4, 2 ) );
   EXPECT_EQ( savedMatrix.getElement( 4, 3 ), loadedMatrix.getElement( 4, 3 ) );
   EXPECT_EQ( savedMatrix.getElement( 4, 4 ), loadedMatrix.getElement( 4, 4 ) );

   EXPECT_EQ( savedMatrix.getElement( 5, 0 ), loadedMatrix.getElement( 5, 0 ) );
   EXPECT_EQ( savedMatrix.getElement( 5, 1 ), loadedMatrix.getElement( 5, 1 ) );
   EXPECT_EQ( savedMatrix.getElement( 5, 2 ), loadedMatrix.getElement( 5, 2 ) );
   EXPECT_EQ( savedMatrix.getElement( 5, 3 ), loadedMatrix.getElement( 5, 3 ) );
   EXPECT_EQ( savedMatrix.getElement( 5, 4 ), loadedMatrix.getElement( 5, 4 ) );
   EXPECT_EQ( std::remove( filename ), 0 );
}

template< typename Matrix >
void test_Print()
{
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

   Matrix m( m_rows, m_cols, {
      { 0, 0, 4 },
      { 1, 0, 1 }, { 1, 1, 4 },
                   { 2, 1, 1 }, { 2, 2, 4 },
                                { 3, 2, 1 }, { 3, 3, 4 }
   } );

   std::stringstream printed;
   std::stringstream couted;

   //change the underlying buffer and save the old buffer
   auto old_buf = std::cout.rdbuf(printed.rdbuf());

   m.print( std::cout ); //all the std::cout goes to ss

   std::cout.rdbuf(old_buf); //reset

   couted << "Row: 0 ->  Col:0->4	 Col:1->1\t\n"
             "Row: 1 ->  Col:0->1	 Col:1->4	 Col:2->1\t\n"
             "Row: 2 ->  Col:1->1	 Col:2->4	 Col:3->1\t\n"
             "Row: 3 ->  Col:2->1	 Col:3->4\t\n";

   EXPECT_EQ( printed.str(), couted.str() );
}

#endif
