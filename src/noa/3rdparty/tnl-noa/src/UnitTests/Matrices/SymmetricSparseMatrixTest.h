#ifdef HAVE_GTEST
#include <gtest/gtest.h>
#include <iostream>
#include <TNL/Matrices/SparseMatrix.h>
#include "SymmetricSparseMatrixTest.hpp"

// test fixture for typed tests
template< typename Matrix >
class MatrixTest : public ::testing::Test
{
protected:
   using MatrixType = Matrix;
};

TYPED_TEST_SUITE( MatrixTest, MatrixTypes);

TYPED_TEST( MatrixTest, setDimensionsTest )
{
    using MatrixType = typename TestFixture::MatrixType;

    test_SetDimensions< MatrixType >();
}

TYPED_TEST( MatrixTest, setRowCapacitiesTest )
{
    using MatrixType = typename TestFixture::MatrixType;

    test_SetRowCapacities< MatrixType >();
}

TYPED_TEST( MatrixTest, setLikeTest )
{
    using MatrixType = typename TestFixture::MatrixType;

    test_SetLike< MatrixType, MatrixType >();
}

TYPED_TEST( MatrixTest, getNonzeroElementsCount )
{
    using MatrixType = typename TestFixture::MatrixType;

    test_GetNonzeroElementsCount< MatrixType >();
}

TYPED_TEST( MatrixTest, resetTest )
{
    using MatrixType = typename TestFixture::MatrixType;

    test_Reset< MatrixType >();
}

TYPED_TEST( MatrixTest, getRowTest )
{
    using MatrixType = typename TestFixture::MatrixType;

    test_GetRow< MatrixType >();
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

TYPED_TEST( MatrixTest, vectorProductTest )
{
    using MatrixType = typename TestFixture::MatrixType;

    test_VectorProduct< MatrixType >();
}

TYPED_TEST( MatrixTest, reduceRows )
{
    using MatrixType = typename TestFixture::MatrixType;

    test_reduceRows< MatrixType >();
}

TYPED_TEST( MatrixTest, saveAndLoadTest )
{
    using MatrixType = typename TestFixture::MatrixType;

    test_SaveAndLoad< MatrixType >( saveAndLoadTestFileName );
}

TYPED_TEST( MatrixTest, printTest )
{
    using MatrixType = typename TestFixture::MatrixType;

    test_Print< MatrixType >();
}

#endif
