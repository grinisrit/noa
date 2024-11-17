#include <Benchmarks/SpMV/ReferenceFormats/Legacy/CSR.h>
#include <Benchmarks/SpMV/ReferenceFormats/Legacy/Ellpack.h>
#include <Benchmarks/SpMV/ReferenceFormats/Legacy/SlicedEllpack.h>

#include <TNL/Matrices/SparseMatrix.h>
#include <TNL/Matrices/MatrixType.h>
#include <TNL/Matrices/DenseMatrix.h>
#include <TNL/Matrices/TridiagonalMatrix.h>
#include <TNL/Matrices/MultidiagonalMatrix.h>
#include <TNL/Algorithms/Segments/CSR.h>
#include <TNL/Algorithms/Segments/Ellpack.h>
#include <TNL/Algorithms/Segments/SlicedEllpack.h>
#include <TNL/Algorithms/contains.h>

template< typename Device, typename Index, typename IndexAllocator >
using EllpackSegments = TNL::Algorithms::Segments::Ellpack< Device, Index, IndexAllocator >;

template< typename Device, typename Index, typename IndexAllocator >
using SlicedEllpackSegments = TNL::Algorithms::Segments::SlicedEllpack< Device, Index, IndexAllocator >;

using CSR_host = TNL::Matrices::SparseMatrix< int, TNL::Devices::Host, int, TNL::Matrices::GeneralMatrix, TNL::Algorithms::Segments::CSRDefault >;
using CSR_cuda = TNL::Matrices::SparseMatrix< int, TNL::Devices::Cuda, int, TNL::Matrices::GeneralMatrix, TNL::Algorithms::Segments::CSRDefault >;
using E_host   = TNL::Matrices::SparseMatrix< int, TNL::Devices::Host, int, TNL::Matrices::GeneralMatrix, EllpackSegments >;
using E_cuda   = TNL::Matrices::SparseMatrix< int, TNL::Devices::Cuda, int, TNL::Matrices::GeneralMatrix, EllpackSegments >;
using SE_host  = TNL::Matrices::SparseMatrix< int, TNL::Devices::Host, int, TNL::Matrices::GeneralMatrix, SlicedEllpackSegments >;
using SE_cuda  = TNL::Matrices::SparseMatrix< int, TNL::Devices::Cuda, int, TNL::Matrices::GeneralMatrix, SlicedEllpackSegments >;
using Dense_host               = TNL::Matrices::DenseMatrix< int, TNL::Devices::Host, int, TNL::Algorithms::Segments::ColumnMajorOrder >;
using Dense_host_RowMajorOrder = TNL::Matrices::DenseMatrix< int, TNL::Devices::Host, int, TNL::Algorithms::Segments::RowMajorOrder >;
using Dense_cuda               = TNL::Matrices::DenseMatrix< int, TNL::Devices::Cuda, int, TNL::Algorithms::Segments::ColumnMajorOrder >;
using Dense_cuda_RowMajorOrder = TNL::Matrices::DenseMatrix< int, TNL::Devices::Cuda, int, TNL::Algorithms::Segments::RowMajorOrder >;


#ifdef HAVE_GTEST
#include <gtest/gtest.h>

/*
 * Sets up the following 10x6 sparse matrix:
 *
 *    /  1  2             \
 *    |           3  4  5 |
 *    |  6  7  8          |
 *    |     9 10 11 12 13 |
 *    | 14 15 16 17 18    |
 *    | 19 20             |
 *    | 21                |
 *    | 22                |
 *    | 23 24 25 26 27    |
 *    \                28 /
 */
template< typename Matrix >
void setupUnevenRowSizeMatrix( Matrix& m )
{
   const int rows = 10;
   const int cols = 6;
   m.setDimensions( rows, cols );
   typename Matrix::RowsCapacitiesType rowLengths;
   rowLengths.setSize( rows );
   rowLengths.setValue( 5 );
   rowLengths.setElement( 0, 2 );
   rowLengths.setElement( 1,  3 );
   rowLengths.setElement( 2,  3 );
   rowLengths.setElement( 5,  2 );
   rowLengths.setElement( 6,  1 );
   rowLengths.setElement( 7,  1 );
   rowLengths.setElement( 9,  1 );
   m.setRowCapacities( rowLengths );

    int value = 1;
    for( int i = 0; i < cols - 4; i++ )  // 0th row
        m.setElement( 0, i, value++ );

    for( int i = 3; i < cols; i++ )      // 1st row
        m.setElement( 1, i, value++ );

    for( int i = 0; i < cols - 3; i++ )  // 2nd row
        m.setElement( 2, i, value++ );

    for( int i = 1; i < cols; i++ )      // 3rd row
        m.setElement( 3, i, value++ );

    for( int i = 0; i < cols - 1; i++ )  // 4th row
        m.setElement( 4, i, value++ );

    for( int i = 0; i < cols - 4; i++ )  // 5th row
        m.setElement( 5, i, value++ );

    m.setElement( 6, 0, value++ );   // 6th row

    m.setElement( 7, 0, value++ );   // 7th row

    for( int i = 0; i < cols - 1; i++ )  // 8th row
        m.setElement( 8, i, value++ );

    m.setElement( 9, 5, value++ );   // 9th row
}

template< typename Matrix >
void checkUnevenRowSizeMatrix( Matrix& m )
{
   ASSERT_EQ( m.getRows(), 10 );
   ASSERT_EQ( m.getColumns(), 6 );

   EXPECT_EQ( m.getElement( 0, 0 ),  1 );
   EXPECT_EQ( m.getElement( 0, 1 ),  2 );
   EXPECT_EQ( m.getElement( 0, 2 ),  0 );
   EXPECT_EQ( m.getElement( 0, 3 ),  0 );
   EXPECT_EQ( m.getElement( 0, 4 ),  0 );
   EXPECT_EQ( m.getElement( 0, 5 ),  0);

   EXPECT_EQ( m.getElement( 1, 0 ),  0 );
   EXPECT_EQ( m.getElement( 1, 1 ),  0 );
   EXPECT_EQ( m.getElement( 1, 2 ),  0 );
   EXPECT_EQ( m.getElement( 1, 3 ),  3 );
   EXPECT_EQ( m.getElement( 1, 4 ),  4 );
   EXPECT_EQ( m.getElement( 1, 5 ),  5 );

   EXPECT_EQ( m.getElement( 2, 0 ),  6 );
   EXPECT_EQ( m.getElement( 2, 1 ),  7 );
   EXPECT_EQ( m.getElement( 2, 2 ),  8 );
   EXPECT_EQ( m.getElement( 2, 3 ),  0 );
   EXPECT_EQ( m.getElement( 2, 4 ),  0 );
   EXPECT_EQ( m.getElement( 2, 5 ),  0 );

   EXPECT_EQ( m.getElement( 3, 0 ),  0 );
   EXPECT_EQ( m.getElement( 3, 1 ),  9 );
   EXPECT_EQ( m.getElement( 3, 2 ), 10 );
   EXPECT_EQ( m.getElement( 3, 3 ), 11 );
   EXPECT_EQ( m.getElement( 3, 4 ), 12 );
   EXPECT_EQ( m.getElement( 3, 5 ), 13 );

   EXPECT_EQ( m.getElement( 4, 0 ), 14 );
   EXPECT_EQ( m.getElement( 4, 1 ), 15 );
   EXPECT_EQ( m.getElement( 4, 2 ), 16 );
   EXPECT_EQ( m.getElement( 4, 3 ), 17 );
   EXPECT_EQ( m.getElement( 4, 4 ), 18 );
   EXPECT_EQ( m.getElement( 4, 5 ),  0 );

   EXPECT_EQ( m.getElement( 5, 0 ), 19 );
   EXPECT_EQ( m.getElement( 5, 1 ), 20 );
   EXPECT_EQ( m.getElement( 5, 2 ),  0 );
   EXPECT_EQ( m.getElement( 5, 3 ),  0 );
   EXPECT_EQ( m.getElement( 5, 4 ),  0 );
   EXPECT_EQ( m.getElement( 5, 5 ),  0 );

   EXPECT_EQ( m.getElement( 6, 0 ), 21 );
   EXPECT_EQ( m.getElement( 6, 1 ),  0 );
   EXPECT_EQ( m.getElement( 6, 2 ),  0 );
   EXPECT_EQ( m.getElement( 6, 3 ),  0 );
   EXPECT_EQ( m.getElement( 6, 4 ),  0 );
   EXPECT_EQ( m.getElement( 6, 5 ),  0 );

   EXPECT_EQ( m.getElement( 7, 0 ), 22 );
   EXPECT_EQ( m.getElement( 7, 1 ),  0 );
   EXPECT_EQ( m.getElement( 7, 2 ),  0 );
   EXPECT_EQ( m.getElement( 7, 3 ),  0 );
   EXPECT_EQ( m.getElement( 7, 4 ),  0 );
   EXPECT_EQ( m.getElement( 7, 5 ),  0 );

   EXPECT_EQ( m.getElement( 8, 0 ), 23 );
   EXPECT_EQ( m.getElement( 8, 1 ), 24 );
   EXPECT_EQ( m.getElement( 8, 2 ), 25 );
   EXPECT_EQ( m.getElement( 8, 3 ), 26 );
   EXPECT_EQ( m.getElement( 8, 4 ), 27 );
   EXPECT_EQ( m.getElement( 8, 5 ),  0 );

   EXPECT_EQ( m.getElement( 9, 0 ),  0 );
   EXPECT_EQ( m.getElement( 9, 1 ),  0 );
   EXPECT_EQ( m.getElement( 9, 2 ),  0 );
   EXPECT_EQ( m.getElement( 9, 3 ),  0 );
   EXPECT_EQ( m.getElement( 9, 4 ),  0 );
   EXPECT_EQ( m.getElement( 9, 5 ), 28 );
}

/*
 * Sets up the following 7x6 sparse matrix:
 *
 *    /              2  1 \
 *    |           5  4  3 |
 *    |        8  7  6    |
 *    |    11 10  9       |
 *    | 14 13 12          |
 *    | 16 15             |
 *    \ 17                /
 */
template< typename Matrix >
void setupAntiTriDiagMatrix( Matrix& m )
{
   const int rows = 7;
   const int cols = 6;
   m.setDimensions( rows, cols );
   typename Matrix::RowsCapacitiesType rowLengths;
   rowLengths.setSize( rows );
   rowLengths.setValue( 3 );
   rowLengths.setElement( 0, 4);
   rowLengths.setElement( 1,  4 );
   m.setRowCapacities( rowLengths );

   int value = 1;
   for( int i = 0; i < rows; i++ )
      for( int j = cols - 1; j > 2; j-- )
         if( j - i + 1 < cols && j - i + 1 >= 0 )
            m.setElement( i, j - i + 1, value++ );
}

template< typename Matrix >
void checkAntiTriDiagMatrix( Matrix& m )
{
   ASSERT_EQ( m.getRows(), 7 );
   ASSERT_EQ( m.getColumns(), 6 );

   EXPECT_EQ( m.getElement( 0, 0 ),  0 );
   EXPECT_EQ( m.getElement( 0, 1 ),  0 );
   EXPECT_EQ( m.getElement( 0, 2 ),  0 );
   EXPECT_EQ( m.getElement( 0, 3 ),  0 );
   EXPECT_EQ( m.getElement( 0, 4 ),  2 );
   EXPECT_EQ( m.getElement( 0, 5 ),  1);

   EXPECT_EQ( m.getElement( 1, 0 ),  0 );
   EXPECT_EQ( m.getElement( 1, 1 ),  0 );
   EXPECT_EQ( m.getElement( 1, 2 ),  0 );
   EXPECT_EQ( m.getElement( 1, 3 ),  5 );
   EXPECT_EQ( m.getElement( 1, 4 ),  4 );
   EXPECT_EQ( m.getElement( 1, 5 ),  3 );

   EXPECT_EQ( m.getElement( 2, 0 ),  0 );
   EXPECT_EQ( m.getElement( 2, 1 ),  0 );
   EXPECT_EQ( m.getElement( 2, 2 ),  8 );
   EXPECT_EQ( m.getElement( 2, 3 ),  7 );
   EXPECT_EQ( m.getElement( 2, 4 ),  6 );
   EXPECT_EQ( m.getElement( 2, 5 ),  0 );

   EXPECT_EQ( m.getElement( 3, 0 ),  0 );
   EXPECT_EQ( m.getElement( 3, 1 ), 11 );
   EXPECT_EQ( m.getElement( 3, 2 ), 10 );
   EXPECT_EQ( m.getElement( 3, 3 ),  9 );
   EXPECT_EQ( m.getElement( 3, 4 ),  0 );
   EXPECT_EQ( m.getElement( 3, 5 ),  0 );

   EXPECT_EQ( m.getElement( 4, 0 ), 14 );
   EXPECT_EQ( m.getElement( 4, 1 ), 13 );
   EXPECT_EQ( m.getElement( 4, 2 ), 12 );
   EXPECT_EQ( m.getElement( 4, 3 ),  0 );
   EXPECT_EQ( m.getElement( 4, 4 ),  0 );
   EXPECT_EQ( m.getElement( 4, 5 ),  0 );

   EXPECT_EQ( m.getElement( 5, 0 ), 16 );
   EXPECT_EQ( m.getElement( 5, 1 ), 15 );
   EXPECT_EQ( m.getElement( 5, 2 ),  0 );
   EXPECT_EQ( m.getElement( 5, 3 ),  0 );
   EXPECT_EQ( m.getElement( 5, 4 ),  0 );
   EXPECT_EQ( m.getElement( 5, 5 ),  0 );

   EXPECT_EQ( m.getElement( 6, 0 ), 17 );
   EXPECT_EQ( m.getElement( 6, 1 ),  0 );
   EXPECT_EQ( m.getElement( 6, 2 ),  0 );
   EXPECT_EQ( m.getElement( 6, 3 ),  0 );
   EXPECT_EQ( m.getElement( 6, 4 ),  0 );
   EXPECT_EQ( m.getElement( 6, 5 ),  0 );
}

/*
 * Sets up the following 7x6 sparse matrix:
 *
 *    / 1  2             \
 *    | 3  4  5          |
 *    |    6  7  8       |
 *    |       9 10 11    |
 *    |         12 13 14 |
 *    |            15 16 |
 *    \               17 /
 */
template< typename Matrix >
void setupTriDiagMatrix( Matrix& m )
{
   const int rows = 7;
   const int cols = 6;
   m.setDimensions( rows, cols );
   typename Matrix::RowsCapacitiesType rowLengths;
   rowLengths.setSize( rows );
   rowLengths.setValue( 3 );
   rowLengths.setElement( 0 , 4 );
   rowLengths.setElement( 1,  4 );
   m.setRowCapacities( rowLengths );


   int value = 1;
   for( int i = 0; i < rows; i++ )
      for( int j = 0; j < 3; j++ )
         if( i + j - 1 >= 0 && i + j - 1 < cols )
            m.setElement( i, i + j - 1, value++ );
}

template< typename Matrix >
void checkTriDiagMatrix( Matrix& m )
{
   ASSERT_EQ( m.getRows(), 7 );
   ASSERT_EQ( m.getColumns(), 6 );

   EXPECT_EQ( m.getElement( 0, 0 ),  1 );
   EXPECT_EQ( m.getElement( 0, 1 ),  2 );
   EXPECT_EQ( m.getElement( 0, 2 ),  0 );
   EXPECT_EQ( m.getElement( 0, 3 ),  0 );
   EXPECT_EQ( m.getElement( 0, 4 ),  0 );
   EXPECT_EQ( m.getElement( 0, 5 ),  0 );

   EXPECT_EQ( m.getElement( 1, 0 ),  3 );
   EXPECT_EQ( m.getElement( 1, 1 ),  4 );
   EXPECT_EQ( m.getElement( 1, 2 ),  5 );
   EXPECT_EQ( m.getElement( 1, 3 ),  0 );
   EXPECT_EQ( m.getElement( 1, 4 ),  0 );
   EXPECT_EQ( m.getElement( 1, 5 ),  0 );

   EXPECT_EQ( m.getElement( 2, 0 ),  0 );
   EXPECT_EQ( m.getElement( 2, 1 ),  6 );
   EXPECT_EQ( m.getElement( 2, 2 ),  7 );
   EXPECT_EQ( m.getElement( 2, 3 ),  8 );
   EXPECT_EQ( m.getElement( 2, 4 ),  0 );
   EXPECT_EQ( m.getElement( 2, 5 ),  0 );

   EXPECT_EQ( m.getElement( 3, 0 ),  0 );
   EXPECT_EQ( m.getElement( 3, 1 ),  0 );
   EXPECT_EQ( m.getElement( 3, 2 ),  9 );
   EXPECT_EQ( m.getElement( 3, 3 ), 10 );
   EXPECT_EQ( m.getElement( 3, 4 ), 11 );
   EXPECT_EQ( m.getElement( 3, 5 ),  0 );

   EXPECT_EQ( m.getElement( 4, 0 ),  0 );
   EXPECT_EQ( m.getElement( 4, 1 ),  0 );
   EXPECT_EQ( m.getElement( 4, 2 ),  0 );
   EXPECT_EQ( m.getElement( 4, 3 ), 12 );
   EXPECT_EQ( m.getElement( 4, 4 ), 13 );
   EXPECT_EQ( m.getElement( 4, 5 ), 14 );

   EXPECT_EQ( m.getElement( 5, 0 ),  0 );
   EXPECT_EQ( m.getElement( 5, 1 ),  0 );
   EXPECT_EQ( m.getElement( 5, 2 ),  0 );
   EXPECT_EQ( m.getElement( 5, 3 ),  0 );
   EXPECT_EQ( m.getElement( 5, 4 ), 15 );
   EXPECT_EQ( m.getElement( 5, 5 ), 16 );

   EXPECT_EQ( m.getElement( 6, 0 ),  0 );
   EXPECT_EQ( m.getElement( 6, 1 ),  0 );
   EXPECT_EQ( m.getElement( 6, 2 ),  0 );
   EXPECT_EQ( m.getElement( 6, 3 ),  0 );
   EXPECT_EQ( m.getElement( 6, 4 ),  0 );
   EXPECT_EQ( m.getElement( 6, 5 ), 17 );
}

template< typename Matrix1, typename Matrix2 >
void testCopyAssignment()
{
   {
      SCOPED_TRACE("Tri Diagonal Matrix");

      Matrix1 triDiag1;
      setupTriDiagMatrix( triDiag1 );
      checkTriDiagMatrix( triDiag1 );

      Matrix2 triDiag2;
      triDiag2 = triDiag1;
      checkTriDiagMatrix( triDiag1 );
      checkTriDiagMatrix( triDiag2 );
   }
   {
      SCOPED_TRACE("Anti Tri Diagonal Matrix");
      Matrix1 antiTriDiag1;
      setupAntiTriDiagMatrix( antiTriDiag1 );
      checkAntiTriDiagMatrix( antiTriDiag1 );

      Matrix2 antiTriDiag2;
      antiTriDiag2 = antiTriDiag1;
      checkAntiTriDiagMatrix( antiTriDiag2 );
   }
   {
      SCOPED_TRACE("Uneven Row Size Matrix");
      Matrix1 unevenRowSize1;
      setupUnevenRowSizeMatrix( unevenRowSize1 );
      checkUnevenRowSizeMatrix( unevenRowSize1 );

      Matrix2 unevenRowSize2;
      unevenRowSize2 = unevenRowSize1;

      checkUnevenRowSizeMatrix( unevenRowSize2 );
   }
}

template< typename Matrix >
void tridiagonalMatrixAssignment()
{
   using RealType = typename Matrix::RealType;
   using DeviceType = typename Matrix::DeviceType;
   using IndexType = typename Matrix::IndexType;

   using TridiagonalHost = TNL::Matrices::TridiagonalMatrix< RealType, TNL::Devices::Host, IndexType >;

   const IndexType rows( 10 ), columns( 10 );
   TridiagonalHost hostMatrix( rows, columns );
   for( IndexType i = 0; i < rows; i++ )
      for( IndexType j = TNL::max( 0, i - 1 ); j < TNL::min( columns, i + 2 ); j++ )
         hostMatrix.setElement( i, j, i + j );

   Matrix matrix;
   matrix = hostMatrix;
   using RowCapacitiesType = TNL::Containers::Vector< IndexType, DeviceType, IndexType >;
   RowCapacitiesType rowCapacities;
   matrix.getCompressedRowLengths( rowCapacities );
   RowCapacitiesType exactRowLengths{ 1, 3, 3, 3, 3, 3, 3, 3, 3, 2 };
   EXPECT_EQ( rowCapacities, exactRowLengths );
   for( IndexType i = 0; i < rows; i++ )
      for( IndexType j = 0; j < columns; j++ )
      {
         if( abs( i - j ) > 1 )
            EXPECT_EQ( matrix.getElement( i, j ), 0.0 );
         else
            EXPECT_EQ( matrix.getElement( i, j ), i + j );
      }

#ifdef __CUDACC__
   using TridiagonalCuda = TNL::Matrices::TridiagonalMatrix< RealType, TNL::Devices::Cuda, IndexType >;
   TridiagonalCuda cudaMatrix( rows, columns );
   cudaMatrix = hostMatrix;
   matrix = cudaMatrix;
   matrix.getCompressedRowLengths( rowCapacities );
   EXPECT_EQ( rowCapacities, exactRowLengths );
   for( IndexType i = 0; i < rows; i++ )
      for( IndexType j = 0; j < columns; j++ )
      {
         if( abs( i - j ) > 1 )
            EXPECT_EQ( matrix.getElement( i, j ), 0.0 );
         else
            EXPECT_EQ( matrix.getElement( i, j ), i + j );
      }
#endif
}

template< typename Matrix >
void multidiagonalMatrixAssignment()
{
   using RealType = typename Matrix::RealType;
   using DeviceType = typename Matrix::DeviceType;
   using IndexType = typename Matrix::IndexType;

   using MultidiagonalHost = TNL::Matrices::MultidiagonalMatrix< RealType, TNL::Devices::Host, IndexType >;
   using DiagonalsOffsetsType = typename MultidiagonalHost::DiagonalsOffsetsType;
   DiagonalsOffsetsType diagonals{ -4, -2, 0, 1, 3, 5 };

   const IndexType rows( 10 ), columns( 10 );
   MultidiagonalHost hostMatrix( rows, columns, diagonals );
   for( IndexType i = 0; i < rows; i++ )
      for( IndexType j = 0; j < columns; j++ )
         if( TNL::Algorithms::contains( diagonals, j - i ) )
            hostMatrix.setElement( i, j, i + j );

   Matrix matrix;
   matrix = hostMatrix;
   using RowCapacitiesType = TNL::Containers::Vector< IndexType, DeviceType, IndexType >;
   RowCapacitiesType rowCapacities;
   matrix.getCompressedRowLengths( rowCapacities );
   RowCapacitiesType exactRowLengths{ 3, 4, 5, 5, 6, 5, 5, 4, 4, 3 };
   EXPECT_EQ( rowCapacities, exactRowLengths );
   for( IndexType i = 0; i < rows; i++ )
      for( IndexType j = 0; j < columns; j++ )
      {
         if( TNL::Algorithms::contains( diagonals, j - i ) )
            EXPECT_EQ( matrix.getElement( i, j ), i + j );
         else
            EXPECT_EQ( matrix.getElement( i, j ), 0.0 );
      }

#ifdef __CUDACC__
   using MultidiagonalCuda = TNL::Matrices::MultidiagonalMatrix< RealType, TNL::Devices::Cuda, IndexType >;
   MultidiagonalCuda cudaMatrix( rows, columns, diagonals );
   cudaMatrix = hostMatrix;
   matrix = cudaMatrix;
   matrix.getCompressedRowLengths( rowCapacities );
   EXPECT_EQ( rowCapacities, exactRowLengths );
   for( IndexType i = 0; i < rows; i++ )
      for( IndexType j = 0; j < columns; j++ )
      {
         if( TNL::Algorithms::contains( diagonals, j - i ) )
            EXPECT_EQ( matrix.getElement( i, j ), i + j );
         else
            EXPECT_EQ( matrix.getElement( i, j ), 0.0 );
      }
#endif
}

template< typename Matrix >
void denseMatrixAssignment()
{
   using RealType = typename Matrix::RealType;
   using DeviceType = typename Matrix::DeviceType;
   using IndexType = typename Matrix::IndexType;

   using DenseHost = TNL::Matrices::DenseMatrix< RealType, TNL::Devices::Host, IndexType >;

   const IndexType rows( 10 ), columns( 10 );
   DenseHost hostMatrix( rows, columns );
   for( IndexType i = 0; i < columns; i++ )
      for( IndexType j = 0; j <= i; j++ )
         hostMatrix( i, j ) = i + j;

   Matrix matrix;
   matrix = hostMatrix;
   using RowCapacitiesType = TNL::Containers::Vector< IndexType, DeviceType, IndexType >;
   RowCapacitiesType rowCapacities;
   matrix.getCompressedRowLengths( rowCapacities );
   RowCapacitiesType exactRowLengths{ 0, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
   EXPECT_EQ( rowCapacities, exactRowLengths );
   for( IndexType i = 0; i < columns; i++ )
      for( IndexType j = 0; j < rows; j++ )
      {
         if( j > i )
            EXPECT_EQ( matrix.getElement( i, j ), 0.0 );
         else
            EXPECT_EQ( matrix.getElement( i, j ), i + j );
      }

#ifdef __CUDACC__
   using DenseCuda = TNL::Matrices::DenseMatrix< RealType, TNL::Devices::Cuda, IndexType >;
   DenseCuda cudaMatrix( rows, columns );
   cudaMatrix = hostMatrix;
   matrix = cudaMatrix;
   matrix.getCompressedRowLengths( rowCapacities );
   EXPECT_EQ( rowCapacities, exactRowLengths );
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

TEST( DenseMatrixCopyTest, Dense_HostToDense_Host )
{
   testCopyAssignment< Dense_host,               Dense_host >();
   testCopyAssignment< Dense_host_RowMajorOrder, Dense_host >();
   testCopyAssignment< Dense_host,               Dense_host_RowMajorOrder >();
   testCopyAssignment< Dense_host_RowMajorOrder, Dense_host_RowMajorOrder >();
}

#ifdef __CUDACC__
TEST( DenseMatrixCopyTest, Dense_HostToDense_Cuda )
{
   testCopyAssignment< Dense_host,               Dense_cuda >();
   testCopyAssignment< Dense_host_RowMajorOrder, Dense_cuda >();
   testCopyAssignment< Dense_host,               Dense_cuda_RowMajorOrder >();
   testCopyAssignment< Dense_host_RowMajorOrder, Dense_cuda_RowMajorOrder >();
}

TEST( DenseMatrixCopyTest, Dense_CudaToDense_Host )
{
   testCopyAssignment< Dense_cuda,               Dense_host >();
   testCopyAssignment< Dense_cuda_RowMajorOrder, Dense_host >();
   testCopyAssignment< Dense_cuda,               Dense_host_RowMajorOrder >();
   testCopyAssignment< Dense_cuda_RowMajorOrder, Dense_host_RowMajorOrder >();
}

TEST( DenseMatrixCopyTest, Dense_CudaToDense_Cuda )
{
   testCopyAssignment< Dense_cuda,               Dense_cuda >();
   testCopyAssignment< Dense_cuda_RowMajorOrder, Dense_cuda >();
   testCopyAssignment< Dense_cuda,               Dense_cuda_RowMajorOrder >();
   testCopyAssignment< Dense_cuda_RowMajorOrder, Dense_cuda_RowMajorOrder >();
}
#endif // __CUDACC__


TEST( DenseMatrixCopyTest, CSR_HostToDense_Host )
{
   testCopyAssignment< CSR_host, Dense_host >();
   testCopyAssignment< CSR_host, Dense_host_RowMajorOrder >();
}

#ifdef __CUDACC__
TEST( DenseMatrixCopyTest, CSR_HostToDense_cuda )
{
   testCopyAssignment< CSR_host, Dense_cuda >();
   testCopyAssignment< CSR_host, Dense_cuda_RowMajorOrder >();
}

TEST( DenseMatrixCopyTest, CSR_CudaToDense_host )
{
   testCopyAssignment< CSR_cuda, Dense_host >();
   testCopyAssignment< CSR_cuda, Dense_host_RowMajorOrder >();
}

TEST( DenseMatrixCopyTest, CSR_CudaToDense_cuda )
{
   testCopyAssignment< CSR_cuda, Dense_cuda >();
   testCopyAssignment< CSR_cuda, Dense_cuda_RowMajorOrder >();
}
#endif

////
// Tridiagonal matrix assignment test
TEST( DenseMatrixCopyTest, TridiagonalMatrixAssignment_to_Dense_host )
{
   tridiagonalMatrixAssignment< Dense_host >();
   tridiagonalMatrixAssignment< Dense_host_RowMajorOrder >();
}

#ifdef __CUDACC__
TEST( DenseMatrixCopyTest, TridiagonalMatrixAssignment_to_Dense_cuda )
{
   tridiagonalMatrixAssignment< Dense_cuda >();
   tridiagonalMatrixAssignment< Dense_cuda_RowMajorOrder >();
}
#endif // __CUDACC__

////
// Multidiagonal matrix assignment test
TEST( DenseMatrixCopyTest, MultidiagonalMatrixAssignment_to_Dense_host )
{
   multidiagonalMatrixAssignment< Dense_host >();
   multidiagonalMatrixAssignment< Dense_host_RowMajorOrder >();
}

#ifdef __CUDACC__
TEST( DenseMatrixCopyTest, MultidiagonalMatrixAssignment_to_Dense_cuda )
{
   multidiagonalMatrixAssignment< Dense_cuda >();
   multidiagonalMatrixAssignment< Dense_cuda_RowMajorOrder >();
}
#endif // __CUDACC__

////
// Dense matrix assignment test
TEST( DenseMatrixCopyTest, DenseMatrixAssignment_to_Dense_host )
{
   denseMatrixAssignment< Dense_host >();
   denseMatrixAssignment< Dense_host_RowMajorOrder >();
}

#ifdef __CUDACC__
TEST( DenseMatrixCopyTest, DenseMatrixAssignment_to_Dense_cuda )
{
   denseMatrixAssignment< Dense_cuda >();
   denseMatrixAssignment< Dense_cuda_RowMajorOrder >();
}
#endif // __CUDACC__

#endif //HAVE_GTEST

#include "../main.h"
