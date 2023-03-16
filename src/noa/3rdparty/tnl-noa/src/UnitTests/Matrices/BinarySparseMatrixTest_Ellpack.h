#include <TNL/Algorithms/Segments/Ellpack.h>
#include <TNL/Matrices/SparseMatrix.h>


#include "BinarySparseMatrixTest.hpp"
#include <iostream>

#ifdef HAVE_GTEST
#include <gtest/gtest.h>

// test fixture for typed tests
template< typename Matrix >
class BinaryMatrixTest_Ellpack : public ::testing::Test
{
protected:
   using EllpackMatrixType = Matrix;
};

////
// Row-major format is used for the host system
template< typename Device, typename Index, typename IndexAlocator >
using RowMajorEllpack = TNL::Algorithms::Segments::Ellpack< Device, Index, IndexAlocator, TNL::Algorithms::Segments::RowMajorOrder, 32 >;


////
// Column-major format is used for GPUs
template< typename Device, typename Index, typename IndexAllocator >
using ColumnMajorEllpack = TNL::Algorithms::Segments::Ellpack< Device, Index, IndexAllocator, TNL::Algorithms::Segments::ColumnMajorOrder, 32 >;

// types for which MatrixTest is instantiated
using EllpackMatrixTypes = ::testing::Types
<
    TNL::Matrices::SparseMatrix< bool, TNL::Devices::Host, int,   TNL::Matrices::GeneralMatrix, RowMajorEllpack, int >,
    TNL::Matrices::SparseMatrix< bool, TNL::Devices::Host, long,  TNL::Matrices::GeneralMatrix, RowMajorEllpack, int >
#ifdef __CUDACC__
   ,TNL::Matrices::SparseMatrix< bool, TNL::Devices::Cuda, int,   TNL::Matrices::GeneralMatrix, ColumnMajorEllpack, int >,
    TNL::Matrices::SparseMatrix< bool, TNL::Devices::Cuda, long,  TNL::Matrices::GeneralMatrix, ColumnMajorEllpack, int >
#endif
>;

TYPED_TEST_SUITE( BinaryMatrixTest_Ellpack, EllpackMatrixTypes);

TYPED_TEST( BinaryMatrixTest_Ellpack, setDimensionsTest )
{
    using EllpackMatrixType = typename TestFixture::EllpackMatrixType;

    test_SetDimensions< EllpackMatrixType >();
}

TYPED_TEST( BinaryMatrixTest_Ellpack, setRowCapacitiesTest )
{
    using EllpackMatrixType = typename TestFixture::EllpackMatrixType;

    test_SetRowCapacities< EllpackMatrixType >();
}

TYPED_TEST( BinaryMatrixTest_Ellpack, setLikeTest )
{
    using EllpackMatrixType = typename TestFixture::EllpackMatrixType;

    test_SetLike< EllpackMatrixType, EllpackMatrixType >();
}

TYPED_TEST( BinaryMatrixTest_Ellpack, resetTest )
{
    using EllpackMatrixType = typename TestFixture::EllpackMatrixType;

    test_Reset< EllpackMatrixType >();
}

TYPED_TEST( BinaryMatrixTest_Ellpack, getRowTest )
{
    using EllpackMatrixType = typename TestFixture::EllpackMatrixType;

    test_GetRow< EllpackMatrixType >();
}


TYPED_TEST( BinaryMatrixTest_Ellpack, setElementTest )
{
    using EllpackMatrixType = typename TestFixture::EllpackMatrixType;

    test_SetElement< EllpackMatrixType >();
}

TYPED_TEST( BinaryMatrixTest_Ellpack, vectorProductTest )
{
    using EllpackMatrixType = typename TestFixture::EllpackMatrixType;

    test_VectorProduct< EllpackMatrixType >();
}

TYPED_TEST( BinaryMatrixTest_Ellpack, reduceRows )
{
    using EllpackMatrixType = typename TestFixture::EllpackMatrixType;

    test_reduceRows< EllpackMatrixType >();
}

TYPED_TEST( BinaryMatrixTest_Ellpack, saveAndLoadTest )
{
    using EllpackMatrixType = typename TestFixture::EllpackMatrixType;

    test_SaveAndLoad< EllpackMatrixType >( "test_BinarySparseMatrixTest_Ellpack" );
}
#endif

#include "../main.h"
