#pragma once

#include <TNL/Containers/Vector.h>
#include <TNL/Containers/VectorView.h>
#include <TNL/Math.h>
#include <TNL/Matrices/DenseMatrix.h>
#include <TNL/Matrices/SparseMatrix.h>
#include <TNL/Matrices/MatrixWrapping.h>
#include <TNL/Algorithms/Segments/Ellpack.h>
#include <iostream>
#include <sstream>

#ifdef HAVE_GTEST
#include <gtest/gtest.h>

template< typename Device_, typename Index_, typename IndexAllocator_ >
using RowMajorEllpack = TNL::Algorithms::Segments::Ellpack< Device_, Index_, IndexAllocator_, TNL::Algorithms::Segments::RowMajorOrder, 1 >;

// test fixture for typed tests
template< typename Matrix >
class MatrixTest : public ::testing::Test
{
protected:
   using MatrixType = Matrix;
};


// types for which MatrixTest is instantiated
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
#ifdef __CUDACC__
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


TYPED_TEST_SUITE( MatrixTest, MatrixTypes);

TYPED_TEST( MatrixTest, WrapMatrix )
{
   using DenseMatrix = typename TestFixture::MatrixType;
   using RealType  = typename DenseMatrix::RealType;
   using DeviceType  = typename DenseMatrix::DeviceType;
   using IndexType  = typename DenseMatrix::IndexType;
   using CSRMatrix = TNL::Matrices::SparseMatrix< RealType, DeviceType, IndexType,  TNL::Matrices::GeneralMatrix, TNL::Algorithms::Segments::CSRScalar >;
   using EllpackMatrix = TNL::Matrices::SparseMatrix< RealType, DeviceType, IndexType, TNL::Matrices::GeneralMatrix, RowMajorEllpack >;

   DenseMatrix denseMatrix{
    { 1,  2,  0,  0 },
    { 0,  6,  0,  0 },
    { 9,  0,  0,  0 },
    { 0,  0, 15, 16 } };
   IndexType rows( 4 ), columns( 4 );
   CSRMatrix csrMatrix;
   EllpackMatrix ellpackMatrix;
   csrMatrix = ellpackMatrix = denseMatrix;

   auto denseMatrixValues  = denseMatrix.getValues().getData();

   auto csrMatrixValues = csrMatrix.getValues().getData();
   auto csrMatrixColumnIndexes = csrMatrix.getColumnIndexes().getData();
   auto csrMatrixRowPointers = csrMatrix.getSegments().getOffsets().getData();

   auto ellpackMatrixValues = ellpackMatrix.getValues().getData();
   auto ellpackMatrixColumnIndexes = ellpackMatrix.getColumnIndexes().getData();

   auto wrappedDenseMatrix   = TNL::Matrices::wrapDenseMatrix< DeviceType >( rows, columns, denseMatrixValues );
   auto wrappedCSRMatrix     = TNL::Matrices::wrapCSRMatrix< DeviceType >( rows, columns, csrMatrixRowPointers, csrMatrixValues, csrMatrixColumnIndexes );
   auto wrappedEllpackMatrix = TNL::Matrices::wrapEllpackMatrix< DeviceType, TNL::Algorithms::Segments::RowMajorOrder >( rows, columns, ( IndexType ) 2, ellpackMatrixValues, ellpackMatrixColumnIndexes );

   EXPECT_EQ( denseMatrix, wrappedDenseMatrix );
   EXPECT_EQ( csrMatrix, wrappedCSRMatrix );
   EXPECT_EQ( ellpackMatrix, wrappedEllpackMatrix );
}


#include "../main.h"

#endif
