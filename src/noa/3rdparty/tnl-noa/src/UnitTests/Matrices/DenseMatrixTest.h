#include <iostream>
#include <functional>
#include <TNL/Devices/Host.h>
#include <TNL/Matrices/Matrix.h>
#include <TNL/Matrices/DenseMatrix.h>
#include <TNL/Containers/Array.h>
#include <TNL/Containers/Vector.h>
#include <TNL/Containers/VectorView.h>
#include <TNL/Algorithms/ParallelFor.h>
#include <TNL/Algorithms/reduce.h>
#include <TNL/Math.h>

using Dense_host_float = TNL::Matrices::DenseMatrix< float, TNL::Devices::Host, int >;
using Dense_host_int = TNL::Matrices::DenseMatrix< int, TNL::Devices::Host, int >;

using Dense_cuda_float = TNL::Matrices::DenseMatrix< float, TNL::Devices::Cuda, int >;
using Dense_cuda_int = TNL::Matrices::DenseMatrix< int, TNL::Devices::Cuda, int >;

static const char* TEST_FILE_NAME = "test_DenseMatrixTest.tnl";

#ifdef HAVE_GTEST
#include <type_traits>

#include <gtest/gtest.h>

void test_GetSerializationType()
{
   using namespace TNL::Algorithms::Segments;
   EXPECT_EQ( ( TNL::Matrices::DenseMatrix< float, TNL::Devices::Host, int, RowMajorOrder >::getSerializationType() ), TNL::String( "Matrices::DenseMatrix< float, [any_device], int, RowMajorOrder >" ) );
   EXPECT_EQ( ( TNL::Matrices::DenseMatrix< int,   TNL::Devices::Host, int, RowMajorOrder >::getSerializationType() ), TNL::String( "Matrices::DenseMatrix< int, [any_device], int, RowMajorOrder >" ) );
   EXPECT_EQ( ( TNL::Matrices::DenseMatrix< float, TNL::Devices::Cuda, int, RowMajorOrder >::getSerializationType() ), TNL::String( "Matrices::DenseMatrix< float, [any_device], int, RowMajorOrder >" ) );
   EXPECT_EQ( ( TNL::Matrices::DenseMatrix< int,   TNL::Devices::Cuda, int, RowMajorOrder >::getSerializationType() ), TNL::String( "Matrices::DenseMatrix< int, [any_device], int, RowMajorOrder >" ) );
   EXPECT_EQ( ( TNL::Matrices::DenseMatrix< float, TNL::Devices::Host, int, ColumnMajorOrder >::getSerializationType() ), TNL::String( "Matrices::DenseMatrix< float, [any_device], int, ColumnMajorOrder >" ) );
   EXPECT_EQ( ( TNL::Matrices::DenseMatrix< int,   TNL::Devices::Host, int, ColumnMajorOrder >::getSerializationType() ), TNL::String( "Matrices::DenseMatrix< int, [any_device], int, ColumnMajorOrder >" ) );
   EXPECT_EQ( ( TNL::Matrices::DenseMatrix< float, TNL::Devices::Cuda, int, ColumnMajorOrder >::getSerializationType() ), TNL::String( "Matrices::DenseMatrix< float, [any_device], int, ColumnMajorOrder >" ) );
   EXPECT_EQ( ( TNL::Matrices::DenseMatrix< int,   TNL::Devices::Cuda, int, ColumnMajorOrder >::getSerializationType() ), TNL::String( "Matrices::DenseMatrix< int, [any_device], int, ColumnMajorOrder >" ) );
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
}

template< typename Matrix1, typename Matrix2 >
void test_SetLike()
{
   using RealType = typename Matrix1::RealType;
   using DeviceType = typename Matrix1::DeviceType;
   using IndexType = typename Matrix1::IndexType;

   const IndexType rows = 8;
   const IndexType cols = 7;

   Matrix1 m1;
   m1.reset();
   m1.setDimensions( rows + 1, cols + 2 );

   Matrix2 m2;
   m2.reset();
   m2.setDimensions( rows, cols );

   m1.setLike( m2 );

   EXPECT_EQ( m1.getRows(), m2.getRows() );
   EXPECT_EQ( m1.getColumns(), m2.getColumns() );
}

template< typename Matrix >
void test_SetElements()
{
   using RealType = typename Matrix::RealType;
   using DeviceType = typename Matrix::DeviceType;
   using IndexType = typename Matrix::IndexType;

   Matrix m( {
      { 1, 2, 3 },
      { 4, 5, 6 },
      { 7, 8, 9 },
   } );

   EXPECT_EQ( m.getRows(), 3 );
   EXPECT_EQ( m.getColumns(), 3 );
   EXPECT_EQ( m.getElement( 0, 0 ), 1 );
   EXPECT_EQ( m.getElement( 0, 1 ), 2 );
   EXPECT_EQ( m.getElement( 0, 2 ), 3 );
   EXPECT_EQ( m.getElement( 1, 0 ), 4 );
   EXPECT_EQ( m.getElement( 1, 1 ), 5 );
   EXPECT_EQ( m.getElement( 1, 2 ), 6 );
   EXPECT_EQ( m.getElement( 2, 0 ), 7 );
   EXPECT_EQ( m.getElement( 2, 1 ), 8 );
   EXPECT_EQ( m.getElement( 2, 2 ), 9 );
}

template< typename Matrix >
void test_GetCompressedRowLengths()
{
   using RealType = typename Matrix::RealType;
   using DeviceType = typename Matrix::DeviceType;
   using IndexType = typename Matrix::IndexType;

   const IndexType rows = 10;
   const IndexType cols = 11;

    Matrix m( rows, cols );

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

   typename Matrix::RowsCapacitiesType rowLengths;
   rowLengths = 0;
   m.getCompressedRowLengths( rowLengths );
   typename Matrix::RowsCapacitiesType correctRowLengths{ 3, 3, 1, 2, 3, 4, 5, 6, 7, 8 };
   EXPECT_EQ( rowLengths, correctRowLengths );
}

template< typename Matrix >
void test_GetAllocatedElementsCount()
{
    using RealType = typename Matrix::RealType;
    using DeviceType = typename Matrix::DeviceType;
    using IndexType = typename Matrix::IndexType;

    const IndexType rows = 7;
    const IndexType cols = 6;

    Matrix m;
    m.reset();
    m.setDimensions( rows, cols );

    EXPECT_EQ( m.getAllocatedElementsCount(), 42 );
}

template< typename Matrix >
void test_GetNonzeroElementsCount()
{
    using RealType = typename Matrix::RealType;
    using DeviceType = typename Matrix::DeviceType;
    using IndexType = typename Matrix::IndexType;

/*
 * Sets up the following 7x6 dense matrix:
 *
 *    /  0  2  3  4  5  6 \
 *    |  7  8  9 10 11 12 |
 *    | 13 14 15 16 17 18 |
 *    | 19 20 21 22 23 24 |
 *    | 25 26 27 28 29 30 |
 *    | 31 32 33 34 35 36 |
 *    \ 37 38 39 40 41  0 /
 */
    const IndexType rows = 7;
    const IndexType cols = 6;

    Matrix m;
    m.reset();
    m.setDimensions( rows, cols );

    RealType value = 1;
    for( IndexType i = 0; i < rows; i++ )
        for( IndexType j = 0; j < cols; j++ )
            m.setElement( i, j, value++ );

    m.setElement( 0, 0, 0); // Set the first element of the diagonal to 0.
    m.setElement( 6, 5, 0); // Set the last element of the diagonal to 0.

    EXPECT_EQ( m.getNonzeroElementsCount(), 40 );
}

template< typename Matrix >
void test_Reset()
{
    using RealType = typename Matrix::RealType;
    using DeviceType = typename Matrix::DeviceType;
    using IndexType = typename Matrix::IndexType;

/*
 * Sets up the following 5x4 dense matrix:
 *
 *    /  0  0  0  0 \
 *    |  0  0  0  0 |
 *    |  0  0  0  0 |
 *    |  0  0  0  0 |
 *    \  0  0  0  0 /
 */
    const IndexType rows = 5;
    const IndexType cols = 4;

    Matrix m;
    m.setDimensions( rows, cols );

    m.reset();

    EXPECT_EQ( m.getRows(), 0 );
    EXPECT_EQ( m.getColumns(), 0 );
}

template< typename Matrix >
void test_SetValue()
{
    using RealType = typename Matrix::RealType;
    using DeviceType = typename Matrix::DeviceType;
    using IndexType = typename Matrix::IndexType;
/*
 * Sets up the following 7x6 dense matrix:
 *
 *    /  1  2  3  4  5  6 \
 *    |  7  8  9 10 11 12 |
 *    | 13 14 15 16 17 18 |
 *    | 19 20 21 22 23 24 |
 *    | 25 26 27 28 29 30 |
 *    | 31 32 33 34 35 36 |
 *    \ 37 38 39 40 41 42 /
 */
    const IndexType rows = 7;
    const IndexType cols = 6;

    Matrix m;
    m.reset();
    m.setDimensions( rows, cols );

    RealType value = 1;
    for( IndexType i = 0; i < rows; i++ )
        for( IndexType j = 0; j < cols; j++ )
            m.setElement( i, j, value++ );

    EXPECT_EQ( m.getElement( 0, 0 ),  1 );
    EXPECT_EQ( m.getElement( 0, 1 ),  2 );
    EXPECT_EQ( m.getElement( 0, 2 ),  3 );
    EXPECT_EQ( m.getElement( 0, 3 ),  4 );
    EXPECT_EQ( m.getElement( 0, 4 ),  5 );
    EXPECT_EQ( m.getElement( 0, 5 ),  6 );

    EXPECT_EQ( m.getElement( 1, 0 ),  7 );
    EXPECT_EQ( m.getElement( 1, 1 ),  8 );
    EXPECT_EQ( m.getElement( 1, 2 ),  9 );
    EXPECT_EQ( m.getElement( 1, 3 ), 10 );
    EXPECT_EQ( m.getElement( 1, 4 ), 11 );
    EXPECT_EQ( m.getElement( 1, 5 ), 12 );

    EXPECT_EQ( m.getElement( 2, 0 ), 13 );
    EXPECT_EQ( m.getElement( 2, 1 ), 14 );
    EXPECT_EQ( m.getElement( 2, 2 ), 15 );
    EXPECT_EQ( m.getElement( 2, 3 ), 16 );
    EXPECT_EQ( m.getElement( 2, 4 ), 17 );
    EXPECT_EQ( m.getElement( 2, 5 ), 18 );

    EXPECT_EQ( m.getElement( 3, 0 ), 19 );
    EXPECT_EQ( m.getElement( 3, 1 ), 20 );
    EXPECT_EQ( m.getElement( 3, 2 ), 21 );
    EXPECT_EQ( m.getElement( 3, 3 ), 22 );
    EXPECT_EQ( m.getElement( 3, 4 ), 23 );
    EXPECT_EQ( m.getElement( 3, 5 ), 24 );

    EXPECT_EQ( m.getElement( 4, 0 ), 25 );
    EXPECT_EQ( m.getElement( 4, 1 ), 26 );
    EXPECT_EQ( m.getElement( 4, 2 ), 27 );
    EXPECT_EQ( m.getElement( 4, 3 ), 28 );
    EXPECT_EQ( m.getElement( 4, 4 ), 29 );
    EXPECT_EQ( m.getElement( 4, 5 ), 30 );

    EXPECT_EQ( m.getElement( 5, 0 ), 31 );
    EXPECT_EQ( m.getElement( 5, 1 ), 32 );
    EXPECT_EQ( m.getElement( 5, 2 ), 33 );
    EXPECT_EQ( m.getElement( 5, 3 ), 34 );
    EXPECT_EQ( m.getElement( 5, 4 ), 35 );
    EXPECT_EQ( m.getElement( 5, 5 ), 36 );

    EXPECT_EQ( m.getElement( 6, 0 ), 37 );
    EXPECT_EQ( m.getElement( 6, 1 ), 38 );
    EXPECT_EQ( m.getElement( 6, 2 ), 39 );
    EXPECT_EQ( m.getElement( 6, 3 ), 40 );
    EXPECT_EQ( m.getElement( 6, 4 ), 41 );
    EXPECT_EQ( m.getElement( 6, 5 ), 42 );

    // Set the values of all elements to a certain number
    m.setValue( 42 );

    EXPECT_EQ( m.getElement( 0, 0 ), 42 );
    EXPECT_EQ( m.getElement( 0, 1 ), 42 );
    EXPECT_EQ( m.getElement( 0, 2 ), 42 );
    EXPECT_EQ( m.getElement( 0, 3 ), 42 );
    EXPECT_EQ( m.getElement( 0, 4 ), 42 );
    EXPECT_EQ( m.getElement( 0, 5 ), 42 );

    EXPECT_EQ( m.getElement( 1, 0 ), 42 );
    EXPECT_EQ( m.getElement( 1, 1 ), 42 );
    EXPECT_EQ( m.getElement( 1, 2 ), 42 );
    EXPECT_EQ( m.getElement( 1, 3 ), 42 );
    EXPECT_EQ( m.getElement( 1, 4 ), 42 );
    EXPECT_EQ( m.getElement( 1, 5 ), 42 );

    EXPECT_EQ( m.getElement( 2, 0 ), 42 );
    EXPECT_EQ( m.getElement( 2, 1 ), 42 );
    EXPECT_EQ( m.getElement( 2, 2 ), 42 );
    EXPECT_EQ( m.getElement( 2, 3 ), 42 );
    EXPECT_EQ( m.getElement( 2, 4 ), 42 );
    EXPECT_EQ( m.getElement( 2, 5 ), 42 );

    EXPECT_EQ( m.getElement( 3, 0 ), 42 );
    EXPECT_EQ( m.getElement( 3, 1 ), 42 );
    EXPECT_EQ( m.getElement( 3, 2 ), 42 );
    EXPECT_EQ( m.getElement( 3, 3 ), 42 );
    EXPECT_EQ( m.getElement( 3, 4 ), 42 );
    EXPECT_EQ( m.getElement( 3, 5 ), 42 );

    EXPECT_EQ( m.getElement( 4, 0 ), 42 );
    EXPECT_EQ( m.getElement( 4, 1 ), 42 );
    EXPECT_EQ( m.getElement( 4, 2 ), 42 );
    EXPECT_EQ( m.getElement( 4, 3 ), 42 );
    EXPECT_EQ( m.getElement( 4, 4 ), 42 );
    EXPECT_EQ( m.getElement( 4, 5 ), 42 );

    EXPECT_EQ( m.getElement( 5, 0 ), 42 );
    EXPECT_EQ( m.getElement( 5, 1 ), 42 );
    EXPECT_EQ( m.getElement( 5, 2 ), 42 );
    EXPECT_EQ( m.getElement( 5, 3 ), 42 );
    EXPECT_EQ( m.getElement( 5, 4 ), 42 );
    EXPECT_EQ( m.getElement( 5, 5 ), 42 );

    EXPECT_EQ( m.getElement( 6, 0 ), 42 );
    EXPECT_EQ( m.getElement( 6, 1 ), 42 );
    EXPECT_EQ( m.getElement( 6, 2 ), 42 );
    EXPECT_EQ( m.getElement( 6, 3 ), 42 );
    EXPECT_EQ( m.getElement( 6, 4 ), 42 );
    EXPECT_EQ( m.getElement( 6, 5 ), 42 );
}

template< typename Matrix >
void test_SetElement()
{
   using RealType = typename Matrix::RealType;
   using DeviceType = typename Matrix::DeviceType;
   using IndexType = typename Matrix::IndexType;

   /*
    * Sets up the following 5x5 dense matrix:
    *
    *    /  1  2  3  4  5 \
    *    |  6  7  8  9 10 |
    *    | 11 12 13 14 15 |
    *    | 16 17 18 19 20 |
    *    \ 21 22 23 24 25 /
    */
   const IndexType rows = 5;
   const IndexType cols = 5;

   Matrix m( rows, cols );

   RealType value = 1;
   for( IndexType i = 0; i < rows; i++ )
      for( IndexType j = 0; j < cols; j++ )
         m.setElement( i, j, value++ );

   EXPECT_EQ( m.getElement( 0, 0 ),  1 );
   EXPECT_EQ( m.getElement( 0, 1 ),  2 );
   EXPECT_EQ( m.getElement( 0, 2 ),  3 );
   EXPECT_EQ( m.getElement( 0, 3 ),  4 );
   EXPECT_EQ( m.getElement( 0, 4 ),  5 );

   EXPECT_EQ( m.getElement( 1, 0 ),  6 );
   EXPECT_EQ( m.getElement( 1, 1 ),  7 );
   EXPECT_EQ( m.getElement( 1, 2 ),  8 );
   EXPECT_EQ( m.getElement( 1, 3 ),  9 );
   EXPECT_EQ( m.getElement( 1, 4 ), 10 );

   EXPECT_EQ( m.getElement( 2, 0 ), 11 );
   EXPECT_EQ( m.getElement( 2, 1 ), 12 );
   EXPECT_EQ( m.getElement( 2, 2 ), 13 );
   EXPECT_EQ( m.getElement( 2, 3 ), 14 );
   EXPECT_EQ( m.getElement( 2, 4 ), 15 );

   EXPECT_EQ( m.getElement( 3, 0 ), 16 );
   EXPECT_EQ( m.getElement( 3, 1 ), 17 );
   EXPECT_EQ( m.getElement( 3, 2 ), 18 );
   EXPECT_EQ( m.getElement( 3, 3 ), 19 );
   EXPECT_EQ( m.getElement( 3, 4 ), 20 );

   EXPECT_EQ( m.getElement( 4, 0 ), 21 );
   EXPECT_EQ( m.getElement( 4, 1 ), 22 );
   EXPECT_EQ( m.getElement( 4, 2 ), 23 );
   EXPECT_EQ( m.getElement( 4, 3 ), 24 );
   EXPECT_EQ( m.getElement( 4, 4 ), 25 );

   TNL::Containers::Vector< RealType, DeviceType, IndexType > v( m.getRows(), 0 );
   auto v_view = v.getView();
   auto m_view = m.getView();
   auto f1 = [=] __cuda_callable__ ( IndexType i ) mutable {
      v_view[ i ] = m_view.getElement( i, i );
   };
   TNL::Algorithms::ParallelFor< DeviceType >::exec( ( IndexType ) 0, m.getRows(), f1 );

   for( IndexType i = 0; i < m.getRows(); i++ )
      EXPECT_EQ( v.getElement( i ), m.getElement( i, i ) );
   auto fetch = [=] __cuda_callable__ ( IndexType i ) -> bool {
      return ( v_view[ i ] == m_view.getElement( i, i ) );
   };
   EXPECT_TRUE( TNL::Algorithms::reduce< DeviceType >( ( IndexType ) 0, m.getRows(), fetch, std::logical_and<>{}, true ) );

}

template< typename Matrix >
void test_AddElement()
{
   using RealType = typename Matrix::RealType;
   using DeviceType = typename Matrix::DeviceType;
   using IndexType = typename Matrix::IndexType;

   /*
    * Sets up the following 6x5 dense matrix:
    *
    *    /  1  2  3  4  5 \
    *    |  6  7  8  9 10 |
    *    | 11 12 13 14 15 |
    *    | 16 17 18 19 20 |
    *    | 21 22 23 24 25 |
    *    \ 26 27 28 29 30 /
    */
   const IndexType rows = 6;
   const IndexType cols = 5;

   Matrix m( rows, cols );

   RealType value = 1;
   for( IndexType i = 0; i < rows; i++ )
      for( IndexType j = 0; j < cols; j++ )
         m.setElement( i, j, value++ );

   // Check the added elements
   EXPECT_EQ( m.getElement( 0, 0 ),  1 );
   EXPECT_EQ( m.getElement( 0, 1 ),  2 );
   EXPECT_EQ( m.getElement( 0, 2 ),  3 );
   EXPECT_EQ( m.getElement( 0, 3 ),  4 );
   EXPECT_EQ( m.getElement( 0, 4 ),  5 );

   EXPECT_EQ( m.getElement( 1, 0 ),  6 );
   EXPECT_EQ( m.getElement( 1, 1 ),  7 );
   EXPECT_EQ( m.getElement( 1, 2 ),  8 );
   EXPECT_EQ( m.getElement( 1, 3 ),  9 );
   EXPECT_EQ( m.getElement( 1, 4 ), 10 );

   EXPECT_EQ( m.getElement( 2, 0 ), 11 );
   EXPECT_EQ( m.getElement( 2, 1 ), 12 );
   EXPECT_EQ( m.getElement( 2, 2 ), 13 );
   EXPECT_EQ( m.getElement( 2, 3 ), 14 );
   EXPECT_EQ( m.getElement( 2, 4 ), 15 );

   EXPECT_EQ( m.getElement( 3, 0 ), 16 );
   EXPECT_EQ( m.getElement( 3, 1 ), 17 );
   EXPECT_EQ( m.getElement( 3, 2 ), 18 );
   EXPECT_EQ( m.getElement( 3, 3 ), 19 );
   EXPECT_EQ( m.getElement( 3, 4 ), 20 );

   EXPECT_EQ( m.getElement( 4, 0 ), 21 );
   EXPECT_EQ( m.getElement( 4, 1 ), 22 );
   EXPECT_EQ( m.getElement( 4, 2 ), 23 );
   EXPECT_EQ( m.getElement( 4, 3 ), 24 );
   EXPECT_EQ( m.getElement( 4, 4 ), 25 );

   EXPECT_EQ( m.getElement( 5, 0 ), 26 );
   EXPECT_EQ( m.getElement( 5, 1 ), 27 );
   EXPECT_EQ( m.getElement( 5, 2 ), 28 );
   EXPECT_EQ( m.getElement( 5, 3 ), 29 );
   EXPECT_EQ( m.getElement( 5, 4 ), 30 );

   // Add new elements to the old elements with a multiplying factor applied to the old elements.
   /*
    * The following setup results in the following 6x5 dense matrix:
    *
    *    /  3  6  9 12 15 \
    *    | 18 21 24 27 30 |
    *    | 33 36 39 42 45 |
    *    | 48 51 54 57 60 |
    *    | 63 66 69 72 75 |
    *    \ 78 81 84 87 90 /
    */
   RealType newValue = 1;
   RealType multiplicator = 2;
   for( IndexType i = 0; i < rows; i++ )
      for( IndexType j = 0; j < cols; j++ )
         m.addElement( i, j, newValue++, multiplicator );

   EXPECT_EQ( m.getElement( 0, 0 ),  3 );
   EXPECT_EQ( m.getElement( 0, 1 ),  6 );
   EXPECT_EQ( m.getElement( 0, 2 ),  9 );
   EXPECT_EQ( m.getElement( 0, 3 ), 12 );
   EXPECT_EQ( m.getElement( 0, 4 ), 15 );

   EXPECT_EQ( m.getElement( 1, 0 ), 18 );
   EXPECT_EQ( m.getElement( 1, 1 ), 21 );
   EXPECT_EQ( m.getElement( 1, 2 ), 24 );
   EXPECT_EQ( m.getElement( 1, 3 ), 27 );
   EXPECT_EQ( m.getElement( 1, 4 ), 30 );

   EXPECT_EQ( m.getElement( 2, 0 ), 33 );
   EXPECT_EQ( m.getElement( 2, 1 ), 36 );
   EXPECT_EQ( m.getElement( 2, 2 ), 39 );
   EXPECT_EQ( m.getElement( 2, 3 ), 42 );
   EXPECT_EQ( m.getElement( 2, 4 ), 45 );

   EXPECT_EQ( m.getElement( 3, 0 ), 48 );
   EXPECT_EQ( m.getElement( 3, 1 ), 51 );
   EXPECT_EQ( m.getElement( 3, 2 ), 54 );
   EXPECT_EQ( m.getElement( 3, 3 ), 57 );
   EXPECT_EQ( m.getElement( 3, 4 ), 60 );

   EXPECT_EQ( m.getElement( 4, 0 ), 63 );
   EXPECT_EQ( m.getElement( 4, 1 ), 66 );
   EXPECT_EQ( m.getElement( 4, 2 ), 69 );
   EXPECT_EQ( m.getElement( 4, 3 ), 72 );
   EXPECT_EQ( m.getElement( 4, 4 ), 75 );

   EXPECT_EQ( m.getElement( 5, 0 ), 78 );
   EXPECT_EQ( m.getElement( 5, 1 ), 81 );
   EXPECT_EQ( m.getElement( 5, 2 ), 84 );
   EXPECT_EQ( m.getElement( 5, 3 ), 87 );
   EXPECT_EQ( m.getElement( 5, 4 ), 90 );
}

template< typename Matrix >
void test_SetRow()
{
   using RealType = typename Matrix::RealType;
   using DeviceType = typename Matrix::DeviceType;
   using IndexType = typename Matrix::IndexType;

   /*
    * Sets up the following 3x7 dense matrix:
    *
    *    / 11 11 11 11 11  6  7 \
    *    | 22 22 22 22 22 13 14 |
    *    \ 15 16 33 33 33 33 33 /
    */
   const IndexType rows = 3;
   const IndexType cols = 7;

   Matrix m;
   m.reset();
   m.setDimensions( rows, cols );

   RealType value = 1;
   for( IndexType i = 0; i < rows; i++ )
      for( IndexType j = 0; j < cols; j++ )
         m.setElement( i, j, value++ );

   auto matrix_view = m.getView();
   auto f = [=] __cuda_callable__ ( IndexType rowIdx ) mutable {
      RealType values[ 3 ][ 5 ] {
         { 11, 11, 11, 11, 11 },
         { 22, 22, 22, 22, 22 },
         { 33, 33, 33, 33, 33 } };
      IndexType columnIndexes[ 3 ][ 5 ] {
         { 0, 1, 2, 3, 4 },
         { 0, 1, 2, 3, 4 },
         { 2, 3, 4, 5, 6 } };
      auto row = matrix_view.getRow( rowIdx );
      for( IndexType i = 0; i < 5; i++ )
        row.setValue( columnIndexes[ rowIdx ][ i ], values[ rowIdx ][ i ] );
   };
   TNL::Algorithms::ParallelFor< DeviceType >::exec( 0, 3, f );

   EXPECT_EQ( m.getElement( 0, 0 ), 11 );
   EXPECT_EQ( m.getElement( 0, 1 ), 11 );
   EXPECT_EQ( m.getElement( 0, 2 ), 11 );
   EXPECT_EQ( m.getElement( 0, 3 ), 11 );
   EXPECT_EQ( m.getElement( 0, 4 ), 11 );
   EXPECT_EQ( m.getElement( 0, 5 ),  6 );
   EXPECT_EQ( m.getElement( 0, 6 ),  7 );

   EXPECT_EQ( m.getElement( 1, 0 ), 22 );
   EXPECT_EQ( m.getElement( 1, 1 ), 22 );
   EXPECT_EQ( m.getElement( 1, 2 ), 22 );
   EXPECT_EQ( m.getElement( 1, 3 ), 22 );
   EXPECT_EQ( m.getElement( 1, 4 ), 22 );
   EXPECT_EQ( m.getElement( 1, 5 ), 13 );
   EXPECT_EQ( m.getElement( 1, 6 ), 14 );

   EXPECT_EQ( m.getElement( 2, 0 ), 15 );
   EXPECT_EQ( m.getElement( 2, 1 ), 16 );
   EXPECT_EQ( m.getElement( 2, 2 ), 33 );
   EXPECT_EQ( m.getElement( 2, 3 ), 33 );
   EXPECT_EQ( m.getElement( 2, 4 ), 33 );
   EXPECT_EQ( m.getElement( 2, 5 ), 33 );
   EXPECT_EQ( m.getElement( 2, 6 ), 33 );
}

template< typename Matrix >
void test_AddRow()
{
   using RealType = typename Matrix::RealType;
   using DeviceType = typename Matrix::DeviceType;
   using IndexType = typename Matrix::IndexType;
   /*
    * Sets up the following 6x5 dense matrix:
    *
    *    /  1  2  3  4  5 \
    *    |  6  7  8  9 10 |
    *    | 11 12 13 14 15 |
    *    | 16 17 18 19 20 |
    *    | 21 22 23 24 25 |
    *    \ 26 27 28 29 30 /
    */

   const IndexType rows = 6;
   const IndexType cols = 5;

   Matrix m( rows, cols );

   RealType value = 1;
   for( IndexType i = 0; i < rows; i++ )
      for( IndexType j = 0; j < cols; j++ )
         m.setElement( i, j, value++ );

   // Check the added elements
   EXPECT_EQ( m.getElement( 0, 0 ),  1 );
   EXPECT_EQ( m.getElement( 0, 1 ),  2 );
   EXPECT_EQ( m.getElement( 0, 2 ),  3 );
   EXPECT_EQ( m.getElement( 0, 3 ),  4 );
   EXPECT_EQ( m.getElement( 0, 4 ),  5 );

   EXPECT_EQ( m.getElement( 1, 0 ),  6 );
   EXPECT_EQ( m.getElement( 1, 1 ),  7 );
   EXPECT_EQ( m.getElement( 1, 2 ),  8 );
   EXPECT_EQ( m.getElement( 1, 3 ),  9 );
   EXPECT_EQ( m.getElement( 1, 4 ), 10 );

   EXPECT_EQ( m.getElement( 2, 0 ), 11 );
   EXPECT_EQ( m.getElement( 2, 1 ), 12 );
   EXPECT_EQ( m.getElement( 2, 2 ), 13 );
   EXPECT_EQ( m.getElement( 2, 3 ), 14 );
   EXPECT_EQ( m.getElement( 2, 4 ), 15 );

   EXPECT_EQ( m.getElement( 3, 0 ), 16 );
   EXPECT_EQ( m.getElement( 3, 1 ), 17 );
   EXPECT_EQ( m.getElement( 3, 2 ), 18 );
   EXPECT_EQ( m.getElement( 3, 3 ), 19 );
   EXPECT_EQ( m.getElement( 3, 4 ), 20 );

   EXPECT_EQ( m.getElement( 4, 0 ), 21 );
   EXPECT_EQ( m.getElement( 4, 1 ), 22 );
   EXPECT_EQ( m.getElement( 4, 2 ), 23 );
   EXPECT_EQ( m.getElement( 4, 3 ), 24 );
   EXPECT_EQ( m.getElement( 4, 4 ), 25 );

   EXPECT_EQ( m.getElement( 5, 0 ), 26 );
   EXPECT_EQ( m.getElement( 5, 1 ), 27 );
   EXPECT_EQ( m.getElement( 5, 2 ), 28 );
   EXPECT_EQ( m.getElement( 5, 3 ), 29 );
   EXPECT_EQ( m.getElement( 5, 4 ), 30 );

   // Add new elements to the old elements with a multiplying factor applied to the old elements.
   /*
    * The following setup results in the following 6x5 sparse matrix:
    *
    *    /  3  6  9 12 15 \
    *    | 18 21 24 27 30 |
    *    | 33 36 39 42 45 |
    *    | 48 51 54 57 60 |
    *    | 63 66 69 72 75 |
    *    \ 78 81 84 87 90 /
    */

   auto matrix_view = m.getView();
   auto f = [=] __cuda_callable__ ( IndexType rowIdx ) mutable {
      RealType values[ 6 ][ 5 ] {
         { 11, 11, 11, 11, 0 },
         { 22, 22, 22, 22, 0 },
         { 33, 33, 33, 33, 0 },
         { 44, 44, 44, 44, 0 },
         { 55, 55, 55, 55, 0 },
         { 66, 66, 66, 66, 0 } };
      auto row = matrix_view.getRow( rowIdx );
      for( IndexType i = 0; i < 5; i++ )
      {
         RealType& val = row.getValue( i );
         val = rowIdx * val + values[ rowIdx ][ i ];
      }
   };
   TNL::Algorithms::ParallelFor< DeviceType >::exec( 0, 6, f );


    EXPECT_EQ( m.getElement( 0, 0 ),  11 );
    EXPECT_EQ( m.getElement( 0, 1 ),  11 );
    EXPECT_EQ( m.getElement( 0, 2 ),  11 );
    EXPECT_EQ( m.getElement( 0, 3 ),  11 );
    EXPECT_EQ( m.getElement( 0, 4 ),   0 );

    EXPECT_EQ( m.getElement( 1, 0 ),  28 );
    EXPECT_EQ( m.getElement( 1, 1 ),  29 );
    EXPECT_EQ( m.getElement( 1, 2 ),  30 );
    EXPECT_EQ( m.getElement( 1, 3 ),  31 );
    EXPECT_EQ( m.getElement( 1, 4 ),  10 );

    EXPECT_EQ( m.getElement( 2, 0 ),  55 );
    EXPECT_EQ( m.getElement( 2, 1 ),  57 );
    EXPECT_EQ( m.getElement( 2, 2 ),  59 );
    EXPECT_EQ( m.getElement( 2, 3 ),  61 );
    EXPECT_EQ( m.getElement( 2, 4 ),  30 );

    EXPECT_EQ( m.getElement( 3, 0 ),  92 );
    EXPECT_EQ( m.getElement( 3, 1 ),  95 );
    EXPECT_EQ( m.getElement( 3, 2 ),  98 );
    EXPECT_EQ( m.getElement( 3, 3 ), 101 );
    EXPECT_EQ( m.getElement( 3, 4 ),  60 );

    EXPECT_EQ( m.getElement( 4, 0 ), 139 );
    EXPECT_EQ( m.getElement( 4, 1 ), 143 );
    EXPECT_EQ( m.getElement( 4, 2 ), 147 );
    EXPECT_EQ( m.getElement( 4, 3 ), 151 );
    EXPECT_EQ( m.getElement( 4, 4 ), 100 );

    EXPECT_EQ( m.getElement( 5, 0 ), 196 );
    EXPECT_EQ( m.getElement( 5, 1 ), 201 );
    EXPECT_EQ( m.getElement( 5, 2 ), 206 );
    EXPECT_EQ( m.getElement( 5, 3 ), 211 );
    EXPECT_EQ( m.getElement( 5, 4 ), 150 );
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

   Matrix m( rows, cols  );
   m.forAllElements( [] __cuda_callable__ ( IndexType rowIdx, IndexType localIdx, const IndexType& columnIdx, RealType& value ) mutable {
      value = rowIdx + 1.0;
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
   Matrix m( rows, cols  );
   using RowView = typename Matrix::RowView;
   m.forAllRows( [] __cuda_callable__ ( RowView& row ) mutable {
      for( IndexType localIdx = 0; localIdx <= row.getRowIndex(); localIdx++ )
         row.setValue( localIdx, row.getRowIndex() - localIdx + 1.0 );
   } );

   for( IndexType rowIdx = 0; rowIdx < rows; rowIdx++ )
      for( IndexType colIdx = 0; colIdx < cols; colIdx++ )
      {
         if( colIdx <= rowIdx )
            EXPECT_EQ( m.getElement( rowIdx, colIdx ), rowIdx - colIdx + 1.0 );
         else
            EXPECT_EQ( m.getElement( rowIdx, colIdx ), 0.0 );
      }

   /////
   // Test without iterator
   m.getValues() = 0.0;
   m.forAllRows( [] __cuda_callable__ ( RowView& row ) mutable {
      for( auto element : row )
         if( element.columnIndex() <= element.rowIndex() )
            element.value() = element.rowIndex() - element.columnIndex() + 1.0;
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

   Matrix m( {
        {  1,  2,  3,  0,  4,  5,  0,  1 },
        {  0,  6,  0,  7,  0,  0,  0,  1 },
        {  0,  8,  9,  0, 10,  0,  0,  1 },
        {  0, 11, 12, 13, 14,  0,  0,  1 },
        {  0, 15,  0,  0,  0,  0,  0,  1 },
        {  0, 16, 17, 18, 19, 20, 21,  1 },
        { 22, 23, 24, 25, 26, 27, 28,  1 },
        { 29, 30, 31, 32, 33, 34, 35, 36 } } );
    typename Matrix::RowsCapacitiesType rowsCapacities{ 6, 3, 4, 5, 2, 7, 8, 8 };

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
void test_VectorProduct()
{
    using RealType = typename Matrix::RealType;
    using DeviceType = typename Matrix::DeviceType;
    using IndexType = typename Matrix::IndexType;
/*
 * Sets up the following 5x4 dense matrix:
 *
 *    /  1  2  3  4 \
 *    |  5  6  7  8 |
 *    |  9 10 11 12 |
 *    | 13 14 15 16 |
 *    \ 17 18 19 20 /
 */
    const IndexType rows = 5;
    const IndexType cols = 4;

    Matrix m;
    m.reset();
    m.setDimensions( rows, cols );

    RealType value = 1;
    for( IndexType i = 0; i < rows; i++ )
        for( IndexType j = 0; j < cols; j++)
            m.setElement( i, j, value++ );

    using VectorType = TNL::Containers::Vector< RealType, DeviceType, IndexType >;

    VectorType inVector;
    inVector.setSize( 4 );
    for( IndexType i = 0; i < inVector.getSize(); i++ )
        inVector.setElement( i, 2 );

    VectorType outVector;
    outVector.setSize( 5 );
    for( IndexType j = 0; j < outVector.getSize(); j++ )
        outVector.setElement( j, 0 );


    m.vectorProduct( inVector, outVector);

    EXPECT_EQ( outVector.getElement( 0 ),  20 );
    EXPECT_EQ( outVector.getElement( 1 ),  52 );
    EXPECT_EQ( outVector.getElement( 2 ),  84 );
    EXPECT_EQ( outVector.getElement( 3 ), 116 );
    EXPECT_EQ( outVector.getElement( 4 ), 148 );
}

template< typename Matrix >
void test_LargeVectorProduct()
{
   using RealType = typename Matrix::RealType;
   using DeviceType = typename Matrix::DeviceType;
   using IndexType = typename Matrix::IndexType;

   if( std::is_same< IndexType, short >::value )
      return;

   const IndexType rows = 997;
   const IndexType cols = 997;

   Matrix m( rows, cols );
   m.forAllElements(
      [] __cuda_callable__ ( IndexType rowIdx, IndexType columnIdx, IndexType columnIdx_, RealType& value ) {
         value = columnIdx + 1.0;
      }
   );


   using VectorType = TNL::Containers::Vector< RealType, DeviceType, IndexType >;

   VectorType inVector( cols );
   inVector.forAllElements( [] __cuda_callable__ ( IndexType i, RealType& value ) {
      value = 1.0;
   } );

   VectorType outVector( rows, 0.0 );

   m.vectorProduct( inVector, outVector);

   for( IndexType i = 0; i < rows; i++ )
   {
      //RealType diag = ( i % 2 == 1 ? cols - 1 : -cols + 1 );
      //RealType non_diag = ( cols % 2 == 0 ? 0.0 : 1.0 );
      RealType rcols = cols;
      EXPECT_EQ( outVector.getElement( i ),  ( 0.5 * rcols ) * ( rcols + 1.0 ) );
   }
}


template< typename Matrix >
void test_AddMatrix()
{
    using RealType = typename Matrix::RealType;
    using DeviceType = typename Matrix::DeviceType;
    using IndexType = typename Matrix::IndexType;
/*
 * Sets up the following 5x4 dense matrix:
 *
 *    /  1  2  3  4 \
 *    |  5  6  7  8 |
 *    |  9 10 11 12 |
 *    | 13 14 15 16 |
 *    \ 17 18 19 20 /
 */
    const IndexType rows = 5;
    const IndexType cols = 4;

    Matrix m;
    m.reset();
    m.setDimensions( rows, cols );

    RealType value = 1;
    for( IndexType i = 0; i < rows; i++ )
        for( IndexType j = 0; j < cols; j++)
            m.setElement( i, j, value++ );

/*
 * Sets up the following 5x4 dense matrix:
 *
 *    /  1  2  3  4 \
 *    |  5  6  7  8 |
 *    |  9 10 11 12 |
 *    | 13 14 15 16 |
 *    \ 17 18 19 20 /
 */

    Matrix m2;
    m2.reset();
    m2.setDimensions( rows, cols );

    RealType newValue = 1;
    for( IndexType i = 0; i < rows; i++ )
        for( IndexType j = 0; j < cols; j++)
            m2.setElement( i, j, newValue++ );

    /*
 * Sets up the following 5x4 dense matrix:
 *
 *    /  1  2  3  4 \
 *    |  5  6  7  8 |
 *    |  9 10 11 12 |
 *    | 13 14 15 16 |
 *    \ 17 18 19 20 /
 */

    Matrix mResult;
    mResult.reset();
    mResult.setDimensions( rows, cols );

    mResult = m;

    RealType matrixMultiplicator = 2;
    RealType thisMatrixMultiplicator = 1;

    mResult.addMatrix( m2, matrixMultiplicator, thisMatrixMultiplicator );

    EXPECT_EQ( mResult.getElement( 0, 0 ), matrixMultiplicator * m2.getElement( 0, 0 ) + thisMatrixMultiplicator * m.getElement( 0, 0 ) );
    EXPECT_EQ( mResult.getElement( 0, 1 ), matrixMultiplicator * m2.getElement( 0, 1 ) + thisMatrixMultiplicator * m.getElement( 0, 1 ) );
    EXPECT_EQ( mResult.getElement( 0, 2 ), matrixMultiplicator * m2.getElement( 0, 2 ) + thisMatrixMultiplicator * m.getElement( 0, 2 ) );
    EXPECT_EQ( mResult.getElement( 0, 3 ), matrixMultiplicator * m2.getElement( 0, 3 ) + thisMatrixMultiplicator * m.getElement( 0, 3 ) );

    EXPECT_EQ( mResult.getElement( 1, 0 ), matrixMultiplicator * m2.getElement( 1, 0 ) + thisMatrixMultiplicator * m.getElement( 1, 0 ) );
    EXPECT_EQ( mResult.getElement( 1, 1 ), matrixMultiplicator * m2.getElement( 1, 1 ) + thisMatrixMultiplicator * m.getElement( 1, 1 ) );
    EXPECT_EQ( mResult.getElement( 1, 2 ), matrixMultiplicator * m2.getElement( 1, 2 ) + thisMatrixMultiplicator * m.getElement( 1, 2 ) );
    EXPECT_EQ( mResult.getElement( 1, 3 ), matrixMultiplicator * m2.getElement( 1, 3 ) + thisMatrixMultiplicator * m.getElement( 1, 3 ) );

    EXPECT_EQ( mResult.getElement( 2, 0 ), matrixMultiplicator * m2.getElement( 2, 0 ) + thisMatrixMultiplicator * m.getElement( 2, 0 ) );
    EXPECT_EQ( mResult.getElement( 2, 1 ), matrixMultiplicator * m2.getElement( 2, 1 ) + thisMatrixMultiplicator * m.getElement( 2, 1 ) );
    EXPECT_EQ( mResult.getElement( 2, 2 ), matrixMultiplicator * m2.getElement( 2, 2 ) + thisMatrixMultiplicator * m.getElement( 2, 2 ) );
    EXPECT_EQ( mResult.getElement( 2, 3 ), matrixMultiplicator * m2.getElement( 2, 3 ) + thisMatrixMultiplicator * m.getElement( 2, 3 ) );

    EXPECT_EQ( mResult.getElement( 3, 0 ), matrixMultiplicator * m2.getElement( 3, 0 ) + thisMatrixMultiplicator * m.getElement( 3, 0 ) );
    EXPECT_EQ( mResult.getElement( 3, 1 ), matrixMultiplicator * m2.getElement( 3, 1 ) + thisMatrixMultiplicator * m.getElement( 3, 1 ) );
    EXPECT_EQ( mResult.getElement( 3, 2 ), matrixMultiplicator * m2.getElement( 3, 2 ) + thisMatrixMultiplicator * m.getElement( 3, 2 ) );
    EXPECT_EQ( mResult.getElement( 3, 3 ), matrixMultiplicator * m2.getElement( 3, 3 ) + thisMatrixMultiplicator * m.getElement( 3, 3 ) );

    EXPECT_EQ( mResult.getElement( 4, 0 ), matrixMultiplicator * m2.getElement( 4, 0 ) + thisMatrixMultiplicator * m.getElement( 4, 0 ) );
    EXPECT_EQ( mResult.getElement( 4, 1 ), matrixMultiplicator * m2.getElement( 4, 1 ) + thisMatrixMultiplicator * m.getElement( 4, 1 ) );
    EXPECT_EQ( mResult.getElement( 4, 2 ), matrixMultiplicator * m2.getElement( 4, 2 ) + thisMatrixMultiplicator * m.getElement( 4, 2 ) );
    EXPECT_EQ( mResult.getElement( 4, 3 ), matrixMultiplicator * m2.getElement( 4, 3 ) + thisMatrixMultiplicator * m.getElement( 4, 3 ) );

    EXPECT_EQ( mResult.getElement( 0, 0 ),  3 );
    EXPECT_EQ( mResult.getElement( 0, 1 ),  6 );
    EXPECT_EQ( mResult.getElement( 0, 2 ),  9 );
    EXPECT_EQ( mResult.getElement( 0, 3 ), 12 );

    EXPECT_EQ( mResult.getElement( 1, 0 ), 15 );
    EXPECT_EQ( mResult.getElement( 1, 1 ), 18 );
    EXPECT_EQ( mResult.getElement( 1, 2 ), 21 );
    EXPECT_EQ( mResult.getElement( 1, 3 ), 24 );

    EXPECT_EQ( mResult.getElement( 2, 0 ), 27 );
    EXPECT_EQ( mResult.getElement( 2, 1 ), 30 );
    EXPECT_EQ( mResult.getElement( 2, 2 ), 33 );
    EXPECT_EQ( mResult.getElement( 2, 3 ), 36 );

    EXPECT_EQ( mResult.getElement( 3, 0 ), 39 );
    EXPECT_EQ( mResult.getElement( 3, 1 ), 42 );
    EXPECT_EQ( mResult.getElement( 3, 2 ), 45 );
    EXPECT_EQ( mResult.getElement( 3, 3 ), 48 );

    EXPECT_EQ( mResult.getElement( 4, 0 ), 51 );
    EXPECT_EQ( mResult.getElement( 4, 1 ), 54 );
    EXPECT_EQ( mResult.getElement( 4, 2 ), 57 );
    EXPECT_EQ( mResult.getElement( 4, 3 ), 60 );
}

template< typename Matrix >
void test_GetMatrixProduct()
{
    using RealType = typename Matrix::RealType;
    using DeviceType = typename Matrix::DeviceType;
    using IndexType = typename Matrix::IndexType;
/*
 * Sets up the following 5x4 dense matrix:
 *
 *    /  1  2  3  4 \
 *    |  5  6  7  8 |
 *    |  9 10 11 12 |
 *    | 13 14 15 16 |
 *    \ 17 18 19 20 /
 */
    const IndexType leftRows = 5;
    const IndexType leftCols = 4;

    Matrix leftMatrix;
    leftMatrix.reset();
    leftMatrix.setDimensions( leftRows, leftCols );

    RealType value = 1;
    for( IndexType i = 0; i < leftRows; i++ )
        for( IndexType j = 0; j < leftCols; j++)
            leftMatrix.setElement( i, j, value++ );

/*
 * Sets up the following 4x5 dense matrix:
 *
 *    /  1  2  3  4  5 \
 *    |  6  7  8  9 10 |
 *    | 11 12 13 14 15 |
 *    \ 16 17 18 19 20 /
 */
    const IndexType rightRows = 4;
    const IndexType rightCols = 5;

    Matrix rightMatrix;
    rightMatrix.reset();
    rightMatrix.setDimensions( rightRows, rightCols );

    RealType newValue = 1;
    for( IndexType i = 0; i < rightRows; i++ )
        for( IndexType j = 0; j < rightCols; j++)
            rightMatrix.setElement( i, j, newValue++ );

/*
 * Sets up the following 5x5 resulting dense matrix:
 *
 *    /  0  0  0  0 \
 *    |  0  0  0  0 |
 *    |  0  0  0  0 |
 *    |  0  0  0  0 |
 *    \  0  0  0  0 /
 */

    Matrix mResult;
    mResult.reset();
    mResult.setDimensions( leftRows, rightCols );
    mResult.setValue( 0 );

    RealType leftMatrixMultiplicator = 1;
    RealType rightMatrixMultiplicator = 2;
/*
 *      /  1  2  3  4 \                            /  220  240  260  280  300 \
 *      |  5  6  7  8 |       /  1  2  3  4  5 \   |  492  544  596  648  700 |
 *  1 * |  9 10 11 12 | * 2 * |  6  7  8  9 10 | = |  764  848  932 1016 1100 |
 *      | 13 14 15 16 |       | 11 12 13 14 15 |   | 1036 1152 1268 1384 1500 |
 *      \ 17 18 19 20 /       \ 16 17 18 19 20 /   \ 1308 1456 1604 1752 1900 /
 */

    mResult.getMatrixProduct( leftMatrix, rightMatrix, leftMatrixMultiplicator, rightMatrixMultiplicator );

    EXPECT_EQ( mResult.getElement( 0, 0 ),  220 );
    EXPECT_EQ( mResult.getElement( 0, 1 ),  240 );
    EXPECT_EQ( mResult.getElement( 0, 2 ),  260 );
    EXPECT_EQ( mResult.getElement( 0, 3 ),  280 );
    EXPECT_EQ( mResult.getElement( 0, 4 ),  300 );

    EXPECT_EQ( mResult.getElement( 1, 0 ),  492 );
    EXPECT_EQ( mResult.getElement( 1, 1 ),  544 );
    EXPECT_EQ( mResult.getElement( 1, 2 ),  596 );
    EXPECT_EQ( mResult.getElement( 1, 3 ),  648 );
    EXPECT_EQ( mResult.getElement( 1, 4 ),  700 );

    EXPECT_EQ( mResult.getElement( 2, 0 ),  764 );
    EXPECT_EQ( mResult.getElement( 2, 1 ),  848 );
    EXPECT_EQ( mResult.getElement( 2, 2 ),  932 );
    EXPECT_EQ( mResult.getElement( 2, 3 ), 1016 );
    EXPECT_EQ( mResult.getElement( 2, 4 ), 1100 );

    EXPECT_EQ( mResult.getElement( 3, 0 ), 1036 );
    EXPECT_EQ( mResult.getElement( 3, 1 ), 1152 );
    EXPECT_EQ( mResult.getElement( 3, 2 ), 1268 );
    EXPECT_EQ( mResult.getElement( 3, 3 ), 1384 );
    EXPECT_EQ( mResult.getElement( 3, 4 ), 1500 );

    EXPECT_EQ( mResult.getElement( 4, 0 ), 1308 );
    EXPECT_EQ( mResult.getElement( 4, 1 ), 1456 );
    EXPECT_EQ( mResult.getElement( 4, 2 ), 1604 );
    EXPECT_EQ( mResult.getElement( 4, 3 ), 1752 );
    EXPECT_EQ( mResult.getElement( 4, 4 ), 1900 );
}

template< typename Matrix >
void test_GetTransposition()
{
    using RealType = typename Matrix::RealType;
    using DeviceType = typename Matrix::DeviceType;
    using IndexType = typename Matrix::IndexType;
/*
 * Sets up the following 3x2 dense matrix:
 *
 *    /  1  2 \
 *    |  3  4 |
 *    \  5  6 /
 */
    const IndexType rows = 3;
    const IndexType cols = 2;

    Matrix m;
    m.reset();
    m.setDimensions( rows, cols );

    RealType value = 1;
    for( IndexType i = 0; i < rows; i++ )
        for( IndexType j = 0; j < cols; j++ )
            m.setElement( i, j, value++ );

    m.print( std::cout );

/*
 * Sets up the following 2x3 dense matrix:
 *
 *    /  0  0  0 \
 *    \  0  0  0 /
 */
    Matrix mTransposed;
    mTransposed.reset();
    mTransposed.setDimensions( cols, rows );

    mTransposed.print( std::cout );

    RealType matrixMultiplicator = 1;

    mTransposed.getTransposition( m, matrixMultiplicator );

    mTransposed.print( std::cout );

/*
 * Should result in the following 2x3 dense matrix:
 *
 *    /  1  3  5 \
 *    \  2  4  6 /
 */

    EXPECT_EQ( mTransposed.getElement( 0, 0 ), 1 );
    EXPECT_EQ( mTransposed.getElement( 0, 1 ), 3 );
    EXPECT_EQ( mTransposed.getElement( 0, 2 ), 5 );

    EXPECT_EQ( mTransposed.getElement( 1, 0 ), 2 );
    EXPECT_EQ( mTransposed.getElement( 1, 1 ), 4 );
    EXPECT_EQ( mTransposed.getElement( 1, 2 ), 6 );
}


template< typename Matrix >
void test_PerformSORIteration()
{
    using RealType = typename Matrix::RealType;
    using DeviceType = typename Matrix::DeviceType;
    using IndexType = typename Matrix::IndexType;
/*
 * Sets up the following 4x4 dense matrix:
 *
 *    /  4  1  1  1 \
 *    |  1  4  1  1 |
 *    |  1  1  4  1 |
 *    \  1  1  1  4 /
 */
    const IndexType rows = 4;
    const IndexType cols = 4;

    Matrix m;
    m.reset();
    m.setDimensions( rows, cols );

    m.setElement( 0, 0, 4.0 );        // 0th row
    m.setElement( 0, 1, 1.0 );
    m.setElement( 0, 2, 1.0 );
    m.setElement( 0, 3, 1.0 );

    m.setElement( 1, 0, 1.0 );        // 1st row
    m.setElement( 1, 1, 4.0 );
    m.setElement( 1, 2, 1.0 );
    m.setElement( 1, 3, 1.0 );

    m.setElement( 2, 0, 1.0 );
    m.setElement( 2, 1, 1.0 );        // 2nd row
    m.setElement( 2, 2, 4.0 );
    m.setElement( 2, 3, 1.0 );

    m.setElement( 3, 0, 1.0 );        // 3rd row
    m.setElement( 3, 1, 1.0 );
    m.setElement( 3, 2, 1.0 );
    m.setElement( 3, 3, 4.0 );

    RealType bVector [ 4 ] = { 1.0, 1.0, 1.0, 1.0 };
    RealType xVector [ 4 ] = { 1.0, 1.0, 1.0, 1.0 };

    IndexType row = 0;
    RealType omega = 1;

    m.performSORIteration( bVector, row++, xVector, omega);

    EXPECT_EQ( xVector[ 0 ], -0.5 );
    EXPECT_EQ( xVector[ 1 ],  1.0 );
    EXPECT_EQ( xVector[ 2 ],  1.0 );
    EXPECT_EQ( xVector[ 3 ],  1.0 );

    m.performSORIteration( bVector, row++, xVector, omega);

    EXPECT_EQ( xVector[ 0 ], -0.5 );
    EXPECT_EQ( xVector[ 1 ], -0.125 );
    EXPECT_EQ( xVector[ 2 ],  1.0 );
    EXPECT_EQ( xVector[ 3 ],  1.0 );

    m.performSORIteration( bVector, row++, xVector, omega);

    EXPECT_EQ( xVector[ 0 ], -0.5 );
    EXPECT_EQ( xVector[ 1 ], -0.125 );
    EXPECT_EQ( xVector[ 2 ],  0.15625 );
    EXPECT_EQ( xVector[ 3 ],  1.0 );

    m.performSORIteration( bVector, row++, xVector, omega);

    EXPECT_EQ( xVector[ 0 ], -0.5 );
    EXPECT_EQ( xVector[ 1 ], -0.125 );
    EXPECT_EQ( xVector[ 2 ], 0.15625 );
    EXPECT_EQ( xVector[ 3 ], 0.3671875 );
}

template< typename Matrix >
void test_AssignmentOperator()
{
   using RealType = typename Matrix::RealType;
   using DeviceType = typename Matrix::DeviceType;
   using IndexType = typename Matrix::IndexType;

   using DenseHost = TNL::Matrices::DenseMatrix< RealType, TNL::Devices::Host, IndexType >;
   using DenseCuda = TNL::Matrices::DenseMatrix< RealType, TNL::Devices::Cuda, IndexType >;

   const IndexType rows( 10 ), columns( 10 );
   DenseHost hostMatrix( rows, columns );
   for( IndexType i = 0; i < columns; i++ )
      for( IndexType j = 0; j <= i; j++ )
         hostMatrix( i, j ) = i + j;

   Matrix matrix( rows, columns );
   matrix.getValues() = 0.0;
   matrix = hostMatrix;
   for( IndexType i = 0; i < columns; i++ )
      for( IndexType j = 0; j < rows; j++ )
      {
         if( j > i )
            EXPECT_EQ( matrix.getElement( i, j ), 0.0 );
         else
            EXPECT_EQ( matrix.getElement( i, j ), i + j );
      }

#ifdef HAVE_CUDA
   DenseCuda cudaMatrix( rows, columns );
   for( IndexType i = 0; i < columns; i++ )
      for( IndexType j = 0; j <= i; j++ )
         cudaMatrix.setElement( i, j, i + j );

   matrix.getValues() = 0.0;
   matrix = cudaMatrix;
   for( IndexType i = 0; i < columns; i++ )
      for( IndexType j = 0; j < rows; j++ )
      {
         if( j > i )
            EXPECT_EQ( matrix.getElement( i, j ), 0.0 );
         else
            EXPECT_EQ( matrix.getElement( i, j ), i + j );
      }
#endif
}


template< typename Matrix >
void test_SaveAndLoad()
{
    using RealType = typename Matrix::RealType;
    using DeviceType = typename Matrix::DeviceType;
    using IndexType = typename Matrix::IndexType;
/*
 * Sets up the following 4x4 dense matrix:
 *
 *    /  1  2  3  4 \
 *    |  5  6  7  8 |
 *    |  9 10 11 12 |
 *    \ 13 14 15 16 /
 */
    const IndexType rows = 4;
    const IndexType cols = 4;

    Matrix savedMatrix;
    savedMatrix.reset();
    savedMatrix.setDimensions( rows, cols );

    RealType value = 1;
    for( IndexType i = 0; i < rows; i++ )
        for( IndexType j = 0; j < cols; j++ )
            savedMatrix.setElement( i, j, value++ );

    ASSERT_NO_THROW( savedMatrix.save( TEST_FILE_NAME ) );

    Matrix loadedMatrix;

    ASSERT_NO_THROW( loadedMatrix.load( TEST_FILE_NAME ) );

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
    EXPECT_EQ( savedMatrix.getElement( 0, 3 ),  4 );

    EXPECT_EQ( savedMatrix.getElement( 1, 0 ),  5 );
    EXPECT_EQ( savedMatrix.getElement( 1, 1 ),  6 );
    EXPECT_EQ( savedMatrix.getElement( 1, 2 ),  7 );
    EXPECT_EQ( savedMatrix.getElement( 1, 3 ),  8 );

    EXPECT_EQ( savedMatrix.getElement( 2, 0 ),  9 );
    EXPECT_EQ( savedMatrix.getElement( 2, 1 ), 10 );
    EXPECT_EQ( savedMatrix.getElement( 2, 2 ), 11 );
    EXPECT_EQ( savedMatrix.getElement( 2, 3 ), 12 );

    EXPECT_EQ( savedMatrix.getElement( 3, 0 ), 13 );
    EXPECT_EQ( savedMatrix.getElement( 3, 1 ), 14 );
    EXPECT_EQ( savedMatrix.getElement( 3, 2 ), 15 );
    EXPECT_EQ( savedMatrix.getElement( 3, 3 ), 16 );
}

// test fixture for typed tests
template< typename Matrix >
class MatrixTest : public ::testing::Test
{
protected:
   using MatrixType = Matrix;
};

// types for which MatrixTest is instantiated
using MatrixTypes = ::testing::Types
<
    TNL::Matrices::DenseMatrix< int,    TNL::Devices::Host, short >,
    TNL::Matrices::DenseMatrix< long,   TNL::Devices::Host, short >,
    TNL::Matrices::DenseMatrix< float,  TNL::Devices::Host, short >,
    TNL::Matrices::DenseMatrix< double, TNL::Devices::Host, short >,
    TNL::Matrices::DenseMatrix< int,    TNL::Devices::Host, int >,
    TNL::Matrices::DenseMatrix< long,   TNL::Devices::Host, int >,
    TNL::Matrices::DenseMatrix< float,  TNL::Devices::Host, int >,
    TNL::Matrices::DenseMatrix< double, TNL::Devices::Host, int >,
    TNL::Matrices::DenseMatrix< int,    TNL::Devices::Host, long >,
    TNL::Matrices::DenseMatrix< long,   TNL::Devices::Host, long >,
    TNL::Matrices::DenseMatrix< float,  TNL::Devices::Host, long >,
    TNL::Matrices::DenseMatrix< double, TNL::Devices::Host, long >
#ifdef HAVE_CUDA
    ,TNL::Matrices::DenseMatrix< int,    TNL::Devices::Cuda, short >,
    TNL::Matrices::DenseMatrix< long,   TNL::Devices::Cuda, short >,
    TNL::Matrices::DenseMatrix< float,  TNL::Devices::Cuda, short >,
    TNL::Matrices::DenseMatrix< double, TNL::Devices::Cuda, short >,
    TNL::Matrices::DenseMatrix< int,    TNL::Devices::Cuda, int >,
    TNL::Matrices::DenseMatrix< long,   TNL::Devices::Cuda, int >,
    TNL::Matrices::DenseMatrix< float,  TNL::Devices::Cuda, int >,
    TNL::Matrices::DenseMatrix< double, TNL::Devices::Cuda, int >,
    TNL::Matrices::DenseMatrix< int,    TNL::Devices::Cuda, long >,
    TNL::Matrices::DenseMatrix< long,   TNL::Devices::Cuda, long >,
    TNL::Matrices::DenseMatrix< float,  TNL::Devices::Cuda, long >,
    TNL::Matrices::DenseMatrix< double, TNL::Devices::Cuda, long >
#endif
>;

TYPED_TEST_SUITE( MatrixTest, MatrixTypes );

TYPED_TEST( MatrixTest, getSerializationType )
{
   test_GetSerializationType();
}

TYPED_TEST( MatrixTest, setDimensionsTest )
{
    using MatrixType = typename TestFixture::MatrixType;

    test_SetDimensions< MatrixType >();
}

TYPED_TEST( MatrixTest, setLikeTest )
{
    using MatrixType = typename TestFixture::MatrixType;

    test_SetLike< MatrixType, MatrixType >();
}

TYPED_TEST( MatrixTest, setElementsTest )
{
    using MatrixType = typename TestFixture::MatrixType;

    test_SetElements< MatrixType >();
}

TYPED_TEST( MatrixTest, getAllocatedElementsCountTest )
{
    using MatrixType = typename TestFixture::MatrixType;

    test_GetAllocatedElementsCount< MatrixType >();
}

TYPED_TEST( MatrixTest, getNonzeroElementsCountTest )
{
    using MatrixType = typename TestFixture::MatrixType;

    test_GetNonzeroElementsCount< MatrixType >();
}

TYPED_TEST( MatrixTest, resetTest )
{
    using MatrixType = typename TestFixture::MatrixType;

    test_Reset< MatrixType >();
}

TYPED_TEST( MatrixTest, setValueTest )
{
    using MatrixType = typename TestFixture::MatrixType;

    test_SetValue< MatrixType >();
}

TYPED_TEST( MatrixTest, setElementTest )
{
    using MatrixType = typename TestFixture::MatrixType;

    test_SetElement< MatrixType >();
}

TYPED_TEST( MatrixTest, addElementTest )
{
    using MatrixType = typename TestFixture::MatrixType;

    test_AddElement< MatrixType >();
}

TYPED_TEST( MatrixTest, setRowTest )
{
    using MatrixType = typename TestFixture::MatrixType;

    test_SetRow< MatrixType >();
}

TYPED_TEST( MatrixTest, addRowTest )
{
    using MatrixType = typename TestFixture::MatrixType;

    test_AddRow< MatrixType >();
}

TYPED_TEST( MatrixTest, forElementsTest )
{
    using MatrixType = typename TestFixture::MatrixType;

    test_ForElements< MatrixType >();
}

TYPED_TEST( MatrixTest, forRowsTest )
{
    using MatrixType = typename TestFixture::MatrixType;

    test_ForRows< MatrixType >();
}

TYPED_TEST( MatrixTest, vectorProductTest )
{
    using MatrixType = typename TestFixture::MatrixType;

    test_VectorProduct< MatrixType >();
}

TYPED_TEST( MatrixTest, largeVectorProductTest )
{
    using MatrixType = typename TestFixture::MatrixType;

    test_LargeVectorProduct< MatrixType >();
}

TYPED_TEST( MatrixTest, addMatrixTest )
{
    using MatrixType = typename TestFixture::MatrixType;

    test_AddMatrix< MatrixType >();
}

TYPED_TEST( MatrixTest, assignmentOperatorTest )
{
    using MatrixType = typename TestFixture::MatrixType;

    test_AssignmentOperator< MatrixType >();
}

TYPED_TEST( MatrixTest, saveAndLoadTest )
{
    using MatrixType = typename TestFixture::MatrixType;

    test_SaveAndLoad< MatrixType >();
}

//// test_getType is not general enough yet. DO NOT TEST IT YET.

//TEST( DenseMatrixTest, Dense_GetTypeTest_Host )
//{
//    host_test_GetType< Dense_host_float, Dense_host_int >();
//}
//
//#ifdef HAVE_CUDA
//TEST( DenseMatrixTest, Dense_GetTypeTest_Cuda )
//{
//    cuda_test_GetType< Dense_cuda_float, Dense_cuda_int >();
//}
//#endif

/*TEST( DenseMatrixTest, Dense_getMatrixProductTest_Host )
{
    bool testRan = false;
    EXPECT_TRUE( testRan );
    std::cout << "\nTEST DID NOT RUN. NOT WORKING.\n\n";
    std::cout << "If launched on CPU, this test will not build, but will print the following message: \n";
    std::cout << "      /home/lukas/tnl-dev/src/TNL/Matrices/Dense_impl.h(609): error: no instance of function template \"TNL::Matrices::DenseMatrixProductKernel\" matches the argument list\n";
    std::cout << "              argument types are: (TNL::Matrices::Dense<int, TNL::Devices::Host, int> *, Dense_host_int *, Dense_host_int *, const int, const int, int, int)\n";
    std::cout << "          detected during:\n";
    std::cout << "              instantiation of \"void TNL::Matrices::Dense<Real, Device, Index>::getMatrixProduct(const Matrix1 &, const Matrix2 &, const TNL::Matrices::Dense<Real, Device, Index>::RealType &, const TNL::Matrices::Dense<Real, Device, Index>::RealType &) [with Real=int, Device=TNL::Devices::Host, Index=int, Matrix1=Dense_host_int, Matrix2=Dense_host_int, tileDim=32]\"\n";
    std::cout << "              /home/lukas/tnl-dev/src/UnitTests/Matrices/DenseMatrixTest.h(901): here\n";
    std::cout << "                  instantiation of \"void test_GetMatrixProduct<Matrix>() [with Matrix=Dense_host_int]\"\n";
    std::cout << "              /home/lukas/tnl-dev/src/UnitTests/Matrices/DenseMatrixTest.h(1315): here\n\n";
}

#ifdef HAVE_CUDA
TEST( DenseMatrixTest, Dense_getMatrixProductTest_Cuda )
{
    bool testRan = false;
    EXPECT_TRUE( testRan );
    std::cout << "\nTEST DID NOT RUN. NOT WORKING.\n\n";
    std::cout << "If launched on GPU, this test will not build, but will print the following message: \n";
    std::cout << "      /home/lukas/tnl-dev/src/TNL/Matrices/Dense_impl.h(510): error: identifier \"tnlCudaMin\" is undefined\n";
    std::cout << "          detected during:\n";
    std::cout << "              instantiation of \"void TNL::Matrices::DenseMatrixProductKernel<Real,Index,Matrix1,Matrix2,tileDim,tileRowBlockSize>(TNL::Matrices::Dense<Real, TNL::Devices::Cuda, Index> *, const Matrix1 *, const Matrix2 *, Real, Real, Index, Index) [with Real=int, Index=int, Matrix1=Dense_cuda_int, Matrix2=Dense_cuda_int, tileDim=32, tileRowBlockSize=8]\"\n";
    std::cout << "              instantiation of \"void TNL::Matrices::Dense<Real, Device, Index>::getMatrixProduct(const Matrix1 &, const Matrix2 &, const TNL::Matrices::Dense<Real, Device, Index>::RealType &, const TNL::Matrices::Dense<Real, Device, Index>::RealType &) [with Real=int, Device=TNL::Devices::Cuda, Index=int, Matrix1=Dense_cuda_int, Matrix2=Dense_cuda_int, tileDim=32]\"\n";
    std::cout << "              /home/lukas/tnl-dev/src/UnitTests/Matrices/DenseMatrixTest.h(901): here\n";
    std::cout << "                  instantiation of \"void test_GetMatrixProduct<Matrix>() [with Matrix=Dense_cuda_int]\"\n";
    std::cout << "              /home/lukas/tnl-dev/src/UnitTests/Matrices/DenseMatrixTest.h(1332): here\n\n";
}
#endif

TEST( DenseMatrixTest, Dense_getTranspositionTest_Host )
{
//    test_GetTransposition< Dense_host_int >();
    bool testRan = false;
    EXPECT_TRUE( testRan );
    std::cout << "\nTEST DID NOT RUN. NOT WORKING.\n\n";
    std::cout << "If launched on CPU, this test will not build, but will print the following message: \n";
    std::cout << "      /home/lukas/tnl-dev/src/TNL/Matrices/Dense_impl.h(836): error: no instance of function template \"TNL::Matrices::DenseTranspositionAlignedKernel\" matches the argument list\n";
    std::cout << "              argument types are: (TNL::Matrices::Dense<int, TNL::Devices::Host, int> *, Dense_host_int *, const int, int, int)\n";
    std::cout << "          detected during:\n";
    std::cout << "              instantiation of \"void TNL::Matrices::Dense<Real, Device, Index>::getTransposition(const Matrix &, const TNL::Matrices::Dense<Real, Device, Index>::RealType &) [with Real=int, Device=TNL::Devices::Host, Index=int, Matrix=Dense_host_int, tileDim=32]\"\n";
    std::cout << "              /home/lukas/tnl-dev/src/UnitTests/Matrices/DenseMatrixTest.h(977): here\n";
    std::cout << "                  instantiation of \"void test_GetTransposition<Matrix>() [with Matrix=Dense_host_int]\"\n";
    std::cout << "              /home/lukas/tnl-dev/src/UnitTests/Matrices/DenseMatrixTest.h(1420): here\n\n";
    std::cout << "AND this message: \n";
    std::cout << "      /home/lukas/tnl-dev/src/TNL/Matrices/Dense_impl.h(852): error: no instance of function template \"TNL::Matrices::DenseTranspositionNonAlignedKernel\" matches the argument list\n";
    std::cout << "              argument types are: (TNL::Matrices::Dense<int, TNL::Devices::Host, int> *, Dense_host_int *, const int, int, int)\n";
    std::cout << "          detected during:\n";
    std::cout << "              instantiation of \"void TNL::Matrices::Dense<Real, Device, Index>::getTransposition(const Matrix &, const TNL::Matrices::Dense<Real, Device, Index>::RealType &) [with Real=int, Device=TNL::Devices::Host, Index=int, Matrix=Dense_host_int, tileDim=32]\"\n";
    std::cout << "              /home/lukas/tnl-dev/src/UnitTests/Matrices/DenseMatrixTest.h(977): here\n";
    std::cout << "                  instantiation of \"void test_GetTransposition<Matrix>() [with Matrix=Dense_host_int]\"\n";
    std::cout << "              /home/lukas/tnl-dev/src/UnitTests/Matrices/DenseMatrixTest.h(1420): here\n\n";
}

#ifdef HAVE_CUDA
TEST( DenseMatrixTest, Dense_getTranspositionTest_Cuda )
{
//    test_GetTransposition< Dense_cuda_int >();
    bool testRan = false;
    EXPECT_TRUE( testRan );
    std::cout << "\nTEST DID NOT RUN. NOT WORKING.\n\n";
    std::cout << "If launched on GPU, this test throws the following message: \n";
    std::cout << "  Assertion 'row >= 0 && row < this->getRows() && column >= 0 && column < this->getColumns()' failed !!!\n";
    std::cout << "      File: /home/lukas/tnl-dev/src/TNL/Matrices/Dense_impl.h \n";
    std::cout << "      Line: 329 \n";
    std::cout << "      Diagnostics: Not supported with CUDA.\n";
    std::cout << "  Assertion 'row >= 0 && row < this->getRows() && column >= 0 && column < this->getColumns()' failed !!! \n";
    std::cout << "      File: /home/lukas/tnl-dev/src/TNL/Matrices/Dense_impl.h \n";
    std::cout << "      Line: 329 \n";
    std::cout << "      Diagnostics: Not supported with CUDA.\n";
    std::cout << "  Assertion 'row >= 0 && row < this->getRows() && column >= 0 && column < this->getColumns()' failed !!! \n";
    std::cout << "      File: /home/lukas/tnl-dev/src/TNL/Matrices/Dense_impl.h \n";
    std::cout << "      Line: 329 \n";
    std::cout << "      Diagnostics: Not supported with CUDA.\n";
    std::cout << "  Assertion 'row >= 0 && row < this->getRows() && column >= 0 && column < this->getColumns()' failed !!! \n";
    std::cout << "      File: /home/lukas/tnl-dev/src/TNL/Matrices/Dense_impl.h \n";
    std::cout << "      Line: 329 \n";
    std::cout << "      Diagnostics: Not supported with CUDA.\n";
    std::cout << "  terminate called after throwing an instance of 'TNL::Exceptions::CudaRuntimeError'\n";
    std::cout << "          what():  CUDA ERROR 4 (cudaErrorLaunchFailure): unspecified launch failure.\n";
    std::cout << "  Source: line 57 in /home/lukas/tnl-dev/src/TNL/Containers/Algorithms/ArrayOperationsCuda_impl.h: unspecified launch failure\n";
    std::cout << "  [1]    4003 abort (core dumped)  ./DenseMatrixTest-dbg\n";
}
#endif

TEST( DenseMatrixTest, Dense_performSORIterationTest_Host )
{
    test_PerformSORIteration< Dense_host_float >();
}

#ifdef HAVE_CUDA
TEST( DenseMatrixTest, Dense_performSORIterationTest_Cuda )
{
//    test_PerformSORIteration< Dense_cuda_float >();
    bool testRan = false;
    EXPECT_TRUE( testRan );
    std::cout << "\nTEST DID NOT RUN. NOT WORKING.\n\n";
    std::cout << "If launched, this test throws the following message: \n";
    std::cout << "      [1]    6992 segmentation fault (core dumped)  ./SparseMatrixTest-dbg\n\n";
    std::cout << "\n THIS IS NOT IMPLEMENTED FOR CUDA YET!!\n\n";
}
#endif
 * */

#endif // HAVE_GTEST

#include "../main.h"
