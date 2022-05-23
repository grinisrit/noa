#include <iostream>
#include <sstream>

#ifdef HAVE_GTEST
#include <gtest/gtest.h>
#include <TNL/Matrices/DenseMatrix.h>

template< typename Matrix >
void test_Constructors()
{
   using RealType = typename Matrix::RealType;
   using DeviceType = typename Matrix::DeviceType;
   using IndexType = typename Matrix::IndexType;

   IndexType size = 5;
   auto rowLengths = [=] __cuda_callable__ ( const IndexType rows, const IndexType columns, const IndexType rowIdx ) -> IndexType { return 1; };
   auto matrixElements = [=] __cuda_callable__ ( const IndexType rows, const IndexType columns, const IndexType rowIdx, const IndexType localIdx, IndexType& columnIdx, RealType& value ) {
         columnIdx = rowIdx;
         value =  1.0;
   };

   using MatrixType = decltype( TNL::Matrices::LambdaMatrixFactory< RealType, DeviceType, IndexType >::create( matrixElements, rowLengths ) );

   MatrixType m( size, size, matrixElements, rowLengths );

   EXPECT_EQ( m.getRows(), size );
   EXPECT_EQ( m.getColumns(), size );
}

template< typename Matrix >
void test_SetDimensions()
{
   using RealType = typename Matrix::RealType;
   using DeviceType = typename Matrix::DeviceType;
   using IndexType = typename Matrix::IndexType;

   IndexType size = 5;
   auto rowLengths = [=] __cuda_callable__ ( const IndexType rows, const IndexType columns, const IndexType rowIdx ) -> IndexType { return 1; };
   auto matrixElements = [=] __cuda_callable__ ( const IndexType rows, const IndexType columns, const IndexType rowIdx, const IndexType localIdx, IndexType& columnIdx, RealType& value ) {
         columnIdx = rowIdx;
         value =  1.0;
   };

   using MatrixType = decltype( TNL::Matrices::LambdaMatrixFactory< RealType, DeviceType, IndexType >::create( matrixElements, rowLengths ) );

   MatrixType m( size, size, matrixElements, rowLengths );

   EXPECT_EQ( m.getRows(), size );
   EXPECT_EQ( m.getColumns(), size );

   m.setDimensions( 10, 10 );
   EXPECT_EQ( m.getRows(), 10 );
   EXPECT_EQ( m.getColumns(), 10 );

}

template< typename Matrix >
void test_GetCompressedRowLengths()
{
   using RealType = typename Matrix::RealType;
   using DeviceType = typename Matrix::DeviceType;
   using IndexType = typename Matrix::IndexType;

   IndexType size = 5;
   auto rowLengths = [=] __cuda_callable__ ( const IndexType rows, const IndexType columns, const IndexType rowIdx ) -> IndexType {
      if( rowIdx == 0 || rowIdx == size - 1 )
         return 1;
      return 3;
   };

   auto matrixElements = [=] __cuda_callable__ ( const IndexType rows, const IndexType columns, const IndexType rowIdx, const IndexType localIdx, IndexType& columnIdx, RealType& value ) {
      if( rowIdx == 0 || rowIdx == size -1 )
      {
         columnIdx = rowIdx;
         value =  1.0;
      }
      else
      {
         columnIdx = rowIdx + localIdx - 1;
         value = ( columnIdx == rowIdx ) ? -2.0 : 1.0;
      }
   };

   using MatrixType = decltype( TNL::Matrices::LambdaMatrixFactory< RealType, DeviceType, IndexType >::create( matrixElements, rowLengths ) );

   MatrixType m( size, size, matrixElements, rowLengths );
   TNL::Containers::Vector< IndexType > correctRowLengths{ 1, 3, 3, 3, 1 };
   TNL::Containers::Vector< IndexType, DeviceType > rowLengthsVector;
   m.getCompressedRowLengths( rowLengthsVector );
   for( int i = 0; i < size; i++ )
      EXPECT_EQ( correctRowLengths.getElement( i ), rowLengthsVector.getElement( i ) );
}

template< typename Matrix >
void test_GetElement()
{
   using RealType = typename Matrix::RealType;
   using DeviceType = typename Matrix::DeviceType;
   using IndexType = typename Matrix::IndexType;

   IndexType size = 5;
   auto rowLengths = [=] __cuda_callable__ ( const IndexType rows, const IndexType columns, const IndexType rowIdx ) -> IndexType {
      if( rowIdx == 0 || rowIdx == size - 1 )
         return 1;
      return 3;
   };

   auto matrixElements = [=] __cuda_callable__ ( const IndexType rows, const IndexType columns, const IndexType rowIdx, const IndexType localIdx, IndexType& columnIdx, RealType& value ) {
      if( rowIdx == 0 || rowIdx == size -1 )
      {
         columnIdx = rowIdx;
         value =  1.0;
      }
      else
      {
         columnIdx = rowIdx + localIdx - 1;
         value = ( columnIdx == rowIdx ) ? -2.0 : 1.0;
      }
   };

   using MatrixType = decltype( TNL::Matrices::LambdaMatrixFactory< RealType, DeviceType, IndexType >::create( matrixElements, rowLengths ) );

   MatrixType m( size, size, matrixElements, rowLengths );
   EXPECT_EQ( m.getElement( 0, 0 ),  1.0 );
   EXPECT_EQ( m.getElement( 0, 1 ),  0.0 );
   EXPECT_EQ( m.getElement( 0, 2 ),  0.0 );
   EXPECT_EQ( m.getElement( 0, 3 ),  0.0 );
   EXPECT_EQ( m.getElement( 0, 4 ),  0.0 );

   EXPECT_EQ( m.getElement( 1, 0 ),  1.0 );
   EXPECT_EQ( m.getElement( 1, 1 ), -2.0 );
   EXPECT_EQ( m.getElement( 1, 2 ),  1.0 );
   EXPECT_EQ( m.getElement( 1, 3 ),  0.0 );
   EXPECT_EQ( m.getElement( 1, 4 ),  0.0 );

   EXPECT_EQ( m.getElement( 2, 0 ),  0.0 );
   EXPECT_EQ( m.getElement( 2, 1 ),  1.0 );
   EXPECT_EQ( m.getElement( 2, 2 ), -2.0 );
   EXPECT_EQ( m.getElement( 2, 3 ),  1.0 );
   EXPECT_EQ( m.getElement( 2, 4 ),  0.0 );

   EXPECT_EQ( m.getElement( 3, 0 ),  0.0 );
   EXPECT_EQ( m.getElement( 3, 1 ),  0.0 );
   EXPECT_EQ( m.getElement( 3, 2 ),  1.0 );
   EXPECT_EQ( m.getElement( 3, 3 ), -2.0 );
   EXPECT_EQ( m.getElement( 3, 4 ),  1.0 );

   EXPECT_EQ( m.getElement( 4, 0 ),  0.0 );
   EXPECT_EQ( m.getElement( 4, 1 ),  0.0 );
   EXPECT_EQ( m.getElement( 4, 2 ),  0.0 );
   EXPECT_EQ( m.getElement( 4, 3 ),  0.0 );
   EXPECT_EQ( m.getElement( 4, 4 ),  1.0 );
}

template< typename Matrix >
void test_ForRows()
{
   using RealType = typename Matrix::RealType;
   using DeviceType = typename Matrix::DeviceType;
   using IndexType = typename Matrix::IndexType;

   /**
    * Prepare lambda matrix of the following form:
    *
    * /  1   0   0   0   0 \.
    * | -2   1  -2   0   0 |
    * |  0  -2   1  -2   0 |
    * |  0   0  -2   1  -2 |
    * \  0   0   0   0   1 /.
    */

   IndexType size = 5;
   auto rowLengths = [=] __cuda_callable__ ( const IndexType rows, const IndexType columns, const IndexType rowIdx ) -> IndexType {
      if( rowIdx == 0 || rowIdx == size - 1 )
         return 1;
      return 3;
   };

   auto matrixElements = [=] __cuda_callable__ ( const IndexType rows, const IndexType columns, const IndexType rowIdx, const IndexType localIdx, IndexType& columnIdx, RealType& value ) {
      if( rowIdx == 0 || rowIdx == size -1 )
      {
         columnIdx = rowIdx;
         value =  1.0;
      }
      else
      {
         columnIdx = rowIdx + localIdx - 1;
         value = ( columnIdx == rowIdx ) ? -2.0 : 1.0;
      }
   };

   using MatrixType = decltype( TNL::Matrices::LambdaMatrixFactory< RealType, DeviceType, IndexType >::create( matrixElements, rowLengths ) );

   MatrixType m( size, size, matrixElements, rowLengths );

   ////
   // Test without iterator
   TNL::Matrices::DenseMatrix< RealType, DeviceType, IndexType > denseMatrix( size, size );
   denseMatrix.setValue( 0.0 );
   auto dense_view = denseMatrix.getView();
   auto f = [=] __cuda_callable__ ( const typename MatrixType::RowView& row ) mutable {
      auto dense_row = dense_view.getRow( row.getRowIndex() );
      for( IndexType localIdx = 0; localIdx < row.getSize(); localIdx++ )
         dense_row.setValue( row.getColumnIndex( localIdx ), row.getValue( localIdx ) );
   };
   m.forAllRows( f );

   for( IndexType row = 0; row < size; row++ )
      for( IndexType column = 0; column < size; column++ )
         EXPECT_EQ( m.getElement( row, column ), denseMatrix.getElement( row, column ) );

   ////
   // Test with iterator
   denseMatrix.getValues() = 0.0;
   auto f_iter = [=] __cuda_callable__ ( const typename MatrixType::RowView& row ) mutable {
      auto dense_row = dense_view.getRow( row.getRowIndex() );
      for( const auto element : row )
         dense_row.setValue( element.columnIndex(), element.value() );
   };
   m.forAllRows( f_iter );

   for( IndexType row = 0; row < size; row++ )
      for( IndexType column = 0; column < size; column++ )
         EXPECT_EQ( m.getElement( row, column ), denseMatrix.getElement( row, column ) );

}

template< typename Matrix >
void test_VectorProduct()
{
   using RealType = typename Matrix::RealType;
   using DeviceType = typename Matrix::DeviceType;
   using IndexType = typename Matrix::IndexType;

   IndexType size = 5;
   auto rowLengths = [=] __cuda_callable__ ( const IndexType rows, const IndexType columns, const IndexType rowIdx ) -> IndexType {
      if( rowIdx == 0 || rowIdx == size - 1 )
         return 1;
      return 3;
   };

   auto matrixElements = [=] __cuda_callable__ ( const IndexType rows, const IndexType columns, const IndexType rowIdx, const IndexType localIdx, IndexType& columnIdx, RealType& value ) {
      if( rowIdx == 0 || rowIdx == size -1 )
      {
         columnIdx = rowIdx;
         value =  1.0;
      }
      else
      {
         columnIdx = rowIdx + localIdx - 1;
         value = ( columnIdx == rowIdx ) ? -2.0 : 1.0;
      }
   };

   using MatrixType = decltype( TNL::Matrices::LambdaMatrixFactory< RealType, DeviceType, IndexType >::create( matrixElements, rowLengths ) );

   MatrixType A( size, size, matrixElements, rowLengths );
   TNL::Containers::Vector< RealType, DeviceType, IndexType > x( size, 1.0 ), b( size, 5.0 );
   A.vectorProduct( x, b );
   EXPECT_EQ( b.getElement( 0 ),  1.0 );
   EXPECT_EQ( b.getElement( 1 ),  0.0 );
   EXPECT_EQ( b.getElement( 2 ),  0.0 );
   EXPECT_EQ( b.getElement( 3 ),  0.0 );
   EXPECT_EQ( b.getElement( 4 ),  1.0 );
}

template< typename Matrix >
void test_reduceRows()
{
   using RealType = typename Matrix::RealType;
   using DeviceType = typename Matrix::DeviceType;
   using IndexType = typename Matrix::IndexType;

   IndexType size = 5;
   auto rowLengths = [=] __cuda_callable__ ( const IndexType rows, const IndexType columns, const IndexType rowIdx ) -> IndexType {
      if( rowIdx == 0 || rowIdx == size - 1 )
         return 1;
      return 3;
   };

   auto matrixElements = [=] __cuda_callable__ ( const IndexType rows, const IndexType columns, const IndexType rowIdx, const IndexType localIdx, IndexType& columnIdx, RealType& value ) {
      if( rowIdx == 0 || rowIdx == size -1 )
      {
         columnIdx = rowIdx;
         value =  1.0;
      }
      else
      {
         columnIdx = rowIdx + localIdx - 1;
         value = ( columnIdx == rowIdx ) ? -2.0 : 1.0;
      }
   };

   using MatrixType = decltype( TNL::Matrices::LambdaMatrixFactory< RealType, DeviceType, IndexType >::create( matrixElements, rowLengths ) );

   MatrixType A( size, size, matrixElements, rowLengths );
   TNL::Containers::Vector< RealType, DeviceType, IndexType > v( size, -1.0 );
   auto vView = v.getView();

   auto fetch = [=] __cuda_callable__ ( IndexType row, IndexType columnIdx, const RealType& value ) mutable -> RealType {
      return value;
   };
   auto reduce = [] __cuda_callable__ ( RealType& sum, const RealType& value ) -> RealType {
      return sum + value;
   };
   auto keep = [=] __cuda_callable__ ( IndexType row, const RealType& value ) mutable {
      vView[ row ] = value;
   };
   A.reduceAllRows( fetch, reduce, keep, 0.0 );

   EXPECT_EQ( v.getElement( 0 ),  1.0 );
   EXPECT_EQ( v.getElement( 1 ),  0.0 );
   EXPECT_EQ( v.getElement( 2 ),  0.0 );
   EXPECT_EQ( v.getElement( 3 ),  0.0 );
   EXPECT_EQ( v.getElement( 4 ),  1.0 );
}

#endif // HAVE_GTEST
