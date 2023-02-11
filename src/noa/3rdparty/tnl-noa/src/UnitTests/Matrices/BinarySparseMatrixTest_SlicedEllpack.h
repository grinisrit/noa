#include <TNL/Algorithms/Segments/SlicedEllpack.h>
#include <TNL/Matrices/SparseMatrix.h>


#include "BinarySparseMatrixTest.hpp"
#include <iostream>

#ifdef HAVE_GTEST
#include <gtest/gtest.h>

// test fixture for typed tests
template< typename Matrix >
class BinaryMatrixTest_SlicedEllpack : public ::testing::Test
{
protected:
   using SlicedEllpackMatrixType = Matrix;
};

////
// Row-major format is used for the host system
template< typename Device, typename Index, typename IndexAllocator >
using RowMajorSlicedEllpack = TNL::Algorithms::Segments::SlicedEllpack< Device, Index, IndexAllocator, TNL::Algorithms::Segments::RowMajorOrder, 32 >;


////
// Column-major format is used for GPUs
template< typename Device, typename Index, typename IndexAllocator >
using ColumnMajorSlicedEllpack = TNL::Algorithms::Segments::SlicedEllpack< Device, Index, IndexAllocator, TNL::Algorithms::Segments::ColumnMajorOrder, 32 >;

// types for which MatrixTest is instantiated
using SlicedEllpackMatrixTypes = ::testing::Types
<
    TNL::Matrices::SparseMatrix< bool, TNL::Devices::Host, int,   TNL::Matrices::GeneralMatrix, RowMajorSlicedEllpack, int >,
    TNL::Matrices::SparseMatrix< bool, TNL::Devices::Host, long,  TNL::Matrices::GeneralMatrix, RowMajorSlicedEllpack, int >
#ifdef __CUDACC__
   ,TNL::Matrices::SparseMatrix< bool, TNL::Devices::Cuda, int,   TNL::Matrices::GeneralMatrix, ColumnMajorSlicedEllpack, int >,
    TNL::Matrices::SparseMatrix< bool, TNL::Devices::Cuda, long,  TNL::Matrices::GeneralMatrix, ColumnMajorSlicedEllpack, int >
#endif
>;

TYPED_TEST_SUITE( BinaryMatrixTest_SlicedEllpack, SlicedEllpackMatrixTypes);

TYPED_TEST( BinaryMatrixTest_SlicedEllpack, setDimensionsTest )
{
    using SlicedEllpackMatrixType = typename TestFixture::SlicedEllpackMatrixType;

    test_SetDimensions< SlicedEllpackMatrixType >();
}

TYPED_TEST( BinaryMatrixTest_SlicedEllpack, setRowCapacitiesTest )
{
    using SlicedEllpackMatrixType = typename TestFixture::SlicedEllpackMatrixType;

    test_SetRowCapacities< SlicedEllpackMatrixType >();
}

TYPED_TEST( BinaryMatrixTest_SlicedEllpack, setLikeTest )
{
    using SlicedEllpackMatrixType = typename TestFixture::SlicedEllpackMatrixType;

    test_SetLike< SlicedEllpackMatrixType, SlicedEllpackMatrixType >();
}

TYPED_TEST( BinaryMatrixTest_SlicedEllpack, resetTest )
{
    using SlicedEllpackMatrixType = typename TestFixture::SlicedEllpackMatrixType;

    test_Reset< SlicedEllpackMatrixType >();
}

TYPED_TEST( BinaryMatrixTest_SlicedEllpack, getRowTest )
{
    using SlicedEllpackMatrixType = typename TestFixture::SlicedEllpackMatrixType;

    test_GetRow< SlicedEllpackMatrixType >();
}


TYPED_TEST( BinaryMatrixTest_SlicedEllpack, setElementTest )
{
    using SlicedEllpackMatrixType = typename TestFixture::SlicedEllpackMatrixType;

    test_SetElement< SlicedEllpackMatrixType >();
}

TYPED_TEST( BinaryMatrixTest_SlicedEllpack, vectorProductTest )
{
    using SlicedEllpackMatrixType = typename TestFixture::SlicedEllpackMatrixType;

    test_VectorProduct< SlicedEllpackMatrixType >();
}

TYPED_TEST( BinaryMatrixTest_SlicedEllpack, reduceRows )
{
    using SlicedEllpackMatrixType = typename TestFixture::SlicedEllpackMatrixType;

    test_reduceRows< SlicedEllpackMatrixType >();
}

TYPED_TEST( BinaryMatrixTest_SlicedEllpack, saveAndLoadTest )
{
    using SlicedEllpackMatrixType = typename TestFixture::SlicedEllpackMatrixType;

    test_SaveAndLoad< SlicedEllpackMatrixType >( "test_BinarySparseMatrixTest" );
}

#endif

#include "../main.h"
