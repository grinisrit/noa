#include <Benchmarks/SpMV/ReferenceFormats/Legacy/BiEllpack.h>

#include "Legacy_SparseMatrixTest.hpp"
#include <iostream>

#ifdef HAVE_GTEST
#include <gtest/gtest.h>

// test fixture for typed tests
template< typename Matrix >
class BiEllpackMatrixTest : public ::testing::Test
{
protected:
   using BiEllpackMatrixType = Matrix;
};

using namespace TNL::Benchmarks::SpMV::ReferenceFormats;
// types for which MatrixTest is instantiated
using BiEllpackMatrixTypes = ::testing::Types
<
    Legacy::BiEllpack< int,    TNL::Devices::Host, int >,
    Legacy::BiEllpack< long,   TNL::Devices::Host, int >,
    Legacy::BiEllpack< float,  TNL::Devices::Host, int >,
    Legacy::BiEllpack< double, TNL::Devices::Host, int >,
    Legacy::BiEllpack< int,    TNL::Devices::Host, long >,
    Legacy::BiEllpack< long,   TNL::Devices::Host, long >,
    Legacy::BiEllpack< float,  TNL::Devices::Host, long >,
    Legacy::BiEllpack< double, TNL::Devices::Host, long >
#ifdef HAVE_CUDA
   ,Legacy::BiEllpack< int,    TNL::Devices::Cuda, int >,
    Legacy::BiEllpack< long,   TNL::Devices::Cuda, int >,
    Legacy::BiEllpack< float,  TNL::Devices::Cuda, int >,
    Legacy::BiEllpack< double, TNL::Devices::Cuda, int >,
    Legacy::BiEllpack< int,    TNL::Devices::Cuda, long >,
    Legacy::BiEllpack< long,   TNL::Devices::Cuda, long >,
    Legacy::BiEllpack< float,  TNL::Devices::Cuda, long >,
    Legacy::BiEllpack< double, TNL::Devices::Cuda, long >
#endif
>;

TYPED_TEST_SUITE( BiEllpackMatrixTest, BiEllpackMatrixTypes);

TYPED_TEST( BiEllpackMatrixTest, setDimensionsTest )
{
    using BiEllpackMatrixType = typename TestFixture::BiEllpackMatrixType;

    test_SetDimensions< BiEllpackMatrixType >();
}

//TYPED_TEST( BiEllpackMatrixTest, setCompressedRowLengthsTest )
//{
////    using BiEllpackMatrixType = typename TestFixture::BiEllpackMatrixType;
//
////    test_SetCompressedRowLengths< BiEllpackMatrixType >();
//
//    bool testRan = false;
//    EXPECT_TRUE( testRan );
//    std::cout << "\nTEST DID NOT RUN. NOT WORKING.\n\n";
//    std::cout << "      This test is dependent on the input format. \n";
//    std::cout << "      Almost every format allocates elements per row differently.\n\n";
//    std::cout << "\n    TODO: Finish implementation of getNonZeroRowLength (Only non-zero elements, not the number of allocated elements.)\n\n";
//}

TYPED_TEST( BiEllpackMatrixTest, setLikeTest )
{
    using BiEllpackMatrixType = typename TestFixture::BiEllpackMatrixType;

    test_SetLike< BiEllpackMatrixType, BiEllpackMatrixType >();
}

TYPED_TEST( BiEllpackMatrixTest, resetTest )
{
    using BiEllpackMatrixType = typename TestFixture::BiEllpackMatrixType;

    test_Reset< BiEllpackMatrixType >();
}

TYPED_TEST( BiEllpackMatrixTest, setElementTest )
{
    using BiEllpackMatrixType = typename TestFixture::BiEllpackMatrixType;

    test_SetElement< BiEllpackMatrixType >();
}

TYPED_TEST( BiEllpackMatrixTest, addElementTest )
{
    using BiEllpackMatrixType = typename TestFixture::BiEllpackMatrixType;

    test_AddElement< BiEllpackMatrixType >();
}

TYPED_TEST( BiEllpackMatrixTest, setRowTest )
{
    using BiEllpackMatrixType = typename TestFixture::BiEllpackMatrixType;

    test_SetRow< BiEllpackMatrixType >();
}

TYPED_TEST( BiEllpackMatrixTest, vectorProductTest )
{
    using BiEllpackMatrixType = typename TestFixture::BiEllpackMatrixType;

    test_VectorProduct< BiEllpackMatrixType >();
}

//TYPED_TEST( BiEllpackMatrixTest, operatorEqualsTest )
//{
//    using BiEllpackMatrixType = typename TestFixture::BiEllpackMatrixType;
//
//    test_OperatorEquals< BiEllpackMatrixType >();
//}

TYPED_TEST( BiEllpackMatrixTest, saveAndLoadTest )
{
    using BiEllpackMatrixType = typename TestFixture::BiEllpackMatrixType;

    test_SaveAndLoad< BiEllpackMatrixType >( "test_SparseMatrixTest_BiEllpack" );
}

TYPED_TEST( BiEllpackMatrixTest, printTest )
{
    using BiEllpackMatrixType = typename TestFixture::BiEllpackMatrixType;

    test_Print< BiEllpackMatrixType >();
}
#endif // HAVE_GTEST

#include "../../main.h"
