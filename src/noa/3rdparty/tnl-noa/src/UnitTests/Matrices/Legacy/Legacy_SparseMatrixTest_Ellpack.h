#include <Benchmarks/SpMV/ReferenceFormats/Legacy/Ellpack.h>

#include "Legacy_SparseMatrixTest.hpp"
#include <iostream>

#ifdef HAVE_GTEST
#include <gtest/gtest.h>

// test fixture for typed tests
template< typename Matrix >
class EllpackMatrixTest : public ::testing::Test
{
protected:
   using EllpackMatrixType = Matrix;
};

using namespace TNL::Benchmarks::SpMV::ReferenceFormats;

// types for which MatrixTest is instantiated
using EllpackMatrixTypes = ::testing::Types
<
    Legacy::Ellpack< int,    TNL::Devices::Host, int >,
    Legacy::Ellpack< long,   TNL::Devices::Host, int >,
    Legacy::Ellpack< float,  TNL::Devices::Host, int >,
    Legacy::Ellpack< double, TNL::Devices::Host, int >,
    Legacy::Ellpack< int,    TNL::Devices::Host, long >,
    Legacy::Ellpack< long,   TNL::Devices::Host, long >,
    Legacy::Ellpack< float,  TNL::Devices::Host, long >,
    Legacy::Ellpack< double, TNL::Devices::Host, long >
#ifdef HAVE_CUDA
   ,Legacy::Ellpack< int,    TNL::Devices::Cuda, int >,
    Legacy::Ellpack< long,   TNL::Devices::Cuda, int >,
    Legacy::Ellpack< float,  TNL::Devices::Cuda, int >,
    Legacy::Ellpack< double, TNL::Devices::Cuda, int >,
    Legacy::Ellpack< int,    TNL::Devices::Cuda, long >,
    Legacy::Ellpack< long,   TNL::Devices::Cuda, long >,
    Legacy::Ellpack< float,  TNL::Devices::Cuda, long >,
    Legacy::Ellpack< double, TNL::Devices::Cuda, long >
#endif
>;

TYPED_TEST_SUITE( EllpackMatrixTest, EllpackMatrixTypes );

TYPED_TEST( EllpackMatrixTest, setDimensionsTest )
{
    using EllpackMatrixType = typename TestFixture::EllpackMatrixType;

    test_SetDimensions< EllpackMatrixType >();
}

//TYPED_TEST( EllpackMatrixTest, setCompressedRowLengthsTest )
//{
////    using EllpackMatrixType = typename TestFixture::EllpackMatrixType;
//
////    test_SetCompressedRowLengths< EllpackMatrixType >();
//
//    bool testRan = false;
//    EXPECT_TRUE( testRan );
//    std::cout << "\nTEST DID NOT RUN. NOT WORKING.\n\n";
//    std::cout << "      This test is dependent on the input format. \n";
//    std::cout << "      Almost every format allocates elements per row differently.\n\n";
//    std::cout << "\n    TODO: Finish implementation of getNonZeroRowLength (Only non-zero elements, not the number of allocated elements.)\n\n";
//}

TYPED_TEST( EllpackMatrixTest, setLikeTest )
{
    using EllpackMatrixType = typename TestFixture::EllpackMatrixType;

    test_SetLike< EllpackMatrixType, EllpackMatrixType >();
}

TYPED_TEST( EllpackMatrixTest, resetTest )
{
    using EllpackMatrixType = typename TestFixture::EllpackMatrixType;

    test_Reset< EllpackMatrixType >();
}

TYPED_TEST( EllpackMatrixTest, setElementTest )
{
    using EllpackMatrixType = typename TestFixture::EllpackMatrixType;

    test_SetElement< EllpackMatrixType >();
}

TYPED_TEST( EllpackMatrixTest, addElementTest )
{
    using EllpackMatrixType = typename TestFixture::EllpackMatrixType;

    test_AddElement< EllpackMatrixType >();
}

TYPED_TEST( EllpackMatrixTest, setRowTest )
{
    using EllpackMatrixType = typename TestFixture::EllpackMatrixType;

    test_SetRow< EllpackMatrixType >();
}

TYPED_TEST( EllpackMatrixTest, vectorProductTest )
{
    using EllpackMatrixType = typename TestFixture::EllpackMatrixType;

    test_VectorProduct< EllpackMatrixType >();
}

TYPED_TEST( EllpackMatrixTest, saveAndLoadTest )
{
    using EllpackMatrixType = typename TestFixture::EllpackMatrixType;

    test_SaveAndLoad< EllpackMatrixType >( "test_SparseMatrixTest_Ellpack" );
}

TYPED_TEST( EllpackMatrixTest, printTest )
{
    using EllpackMatrixType = typename TestFixture::EllpackMatrixType;

    test_Print< EllpackMatrixType >();
}

#endif

#include "../../main.h"
