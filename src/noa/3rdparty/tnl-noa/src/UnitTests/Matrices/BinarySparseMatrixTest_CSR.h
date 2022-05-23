#include <TNL/Algorithms/Segments/CSR.h>
#include <TNL/Matrices/SparseMatrix.h>


#include "BinarySparseMatrixTest.hpp"
#include <iostream>

#ifdef HAVE_GTEST
#include <gtest/gtest.h>

// test fixture for typed tests
template< typename Matrix >
class BinaryMatrixTest_CSR : public ::testing::Test
{
protected:
   using CSRMatrixType = Matrix;
};

// types for which MatrixTest is instantiated
using CSRMatrixTypes = ::testing::Types
<
    TNL::Matrices::SparseMatrix< bool, TNL::Devices::Host, int,   TNL::Matrices::GeneralMatrix, TNL::Algorithms::Segments::CSRDefault, int >,
    TNL::Matrices::SparseMatrix< bool, TNL::Devices::Host, long,  TNL::Matrices::GeneralMatrix, TNL::Algorithms::Segments::CSRDefault, int >
#ifdef HAVE_CUDA
   ,TNL::Matrices::SparseMatrix< bool, TNL::Devices::Cuda, int,   TNL::Matrices::GeneralMatrix, TNL::Algorithms::Segments::CSRDefault, int >,
    TNL::Matrices::SparseMatrix< bool, TNL::Devices::Cuda, long,  TNL::Matrices::GeneralMatrix, TNL::Algorithms::Segments::CSRDefault, int >
#endif
>;

TYPED_TEST_SUITE( BinaryMatrixTest_CSR, CSRMatrixTypes);

TYPED_TEST( BinaryMatrixTest_CSR, setDimensionsTest )
{
    using CSRMatrixType = typename TestFixture::CSRMatrixType;

    test_SetDimensions< CSRMatrixType >();
}

TYPED_TEST( BinaryMatrixTest_CSR, setRowCapacitiesTest )
{
    using CSRMatrixType = typename TestFixture::CSRMatrixType;

    test_SetRowCapacities< CSRMatrixType >();
}

TYPED_TEST( BinaryMatrixTest_CSR, setLikeTest )
{
    using CSRMatrixType = typename TestFixture::CSRMatrixType;

    test_SetLike< CSRMatrixType, CSRMatrixType >();
}

TYPED_TEST( BinaryMatrixTest_CSR, resetTest )
{
    using CSRMatrixType = typename TestFixture::CSRMatrixType;

    test_Reset< CSRMatrixType >();
}

TYPED_TEST( BinaryMatrixTest_CSR, getRowTest )
{
    using CSRMatrixType = typename TestFixture::CSRMatrixType;

    test_GetRow< CSRMatrixType >();
}


TYPED_TEST( BinaryMatrixTest_CSR, setElementTest )
{
    using CSRMatrixType = typename TestFixture::CSRMatrixType;

    test_SetElement< CSRMatrixType >();
}

TYPED_TEST( BinaryMatrixTest_CSR, vectorProductTest )
{
    using CSRMatrixType = typename TestFixture::CSRMatrixType;

    test_VectorProduct< CSRMatrixType >();
}

TYPED_TEST( BinaryMatrixTest_CSR, reduceRows )
{
    using CSRMatrixType = typename TestFixture::CSRMatrixType;

    test_reduceRows< CSRMatrixType >();
}

TYPED_TEST( BinaryMatrixTest_CSR, saveAndLoadTest )
{
    using CSRMatrixType = typename TestFixture::CSRMatrixType;

    test_SaveAndLoad< CSRMatrixType >( "test_BinarySparseMatrixTest_CSR" );
}
#endif

#include "../main.h"
