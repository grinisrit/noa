#include <Benchmarks/SpMV/ReferenceFormats/Legacy/SlicedEllpack.h>


#include "Legacy_SparseMatrixTest.hpp"
#include <iostream>

#ifdef HAVE_GTEST
#include <gtest/gtest.h>

// test fixture for typed tests
template< typename Matrix >
class SlicedEllpackMatrixTest : public ::testing::Test
{
protected:
   using SlicedEllpackMatrixType = Matrix;
};

using namespace TNL::Benchmarks::SpMV::ReferenceFormats;

template< typename Real, typename Device, typename Index >
using SlicedEllpackType = Legacy::SlicedEllpack< Real, Device, Index, 32 >;


// types for which MatrixTest is instantiated
using SlicedEllpackMatrixTypes = ::testing::Types
<
    SlicedEllpackType< int,     TNL::Devices::Host, int   >,
    SlicedEllpackType< long,    TNL::Devices::Host, int   >,
    SlicedEllpackType< float,   TNL::Devices::Host, int   >,
    SlicedEllpackType< double,  TNL::Devices::Host, int   >,
    SlicedEllpackType< int,     TNL::Devices::Host, long  >,
    SlicedEllpackType< long,    TNL::Devices::Host, long  >,
    SlicedEllpackType< float,   TNL::Devices::Host, long  >,
    SlicedEllpackType< double,  TNL::Devices::Host, long  >
#ifdef HAVE_CUDA
   ,SlicedEllpackType< int,     TNL::Devices::Cuda, int   >,
    SlicedEllpackType< long,    TNL::Devices::Cuda, int   >,
    SlicedEllpackType< float,   TNL::Devices::Cuda, int   >,
    SlicedEllpackType< double,  TNL::Devices::Cuda, int   >,
    SlicedEllpackType< int,     TNL::Devices::Cuda, long  >,
    SlicedEllpackType< long,    TNL::Devices::Cuda, long  >,
    SlicedEllpackType< float,   TNL::Devices::Cuda, long  >,
    SlicedEllpackType< double,  TNL::Devices::Cuda, long  >
#endif
>;

TYPED_TEST_SUITE( SlicedEllpackMatrixTest, SlicedEllpackMatrixTypes);

TYPED_TEST( SlicedEllpackMatrixTest, setDimensionsTest )
{
    using SlicedEllpackMatrixType = typename TestFixture::SlicedEllpackMatrixType;

    test_SetDimensions< SlicedEllpackMatrixType >();
}

//TYPED_TEST( SlicedEllpackMatrixTest, setCompressedRowLengthsTest )
//{
////    using SlicedEllpackMatrixType = typename TestFixture::SlicedEllpackMatrixType;
//
////    test_SetCompressedRowLengths< SlicedEllpackMatrixType >();
//
//    bool testRan = false;
//    EXPECT_TRUE( testRan );
//    std::cout << "\nTEST DID NOT RUN. NOT WORKING.\n\n";
//    std::cout << "      This test is dependent on the input format. \n";
//    std::cout << "      Almost every format allocates elements per row differently.\n\n";
//    std::cout << "\n    TODO: Finish implementation of getNonZeroRowLength (Only non-zero elements, not the number of allocated elements.)\n\n";
//}

TYPED_TEST( SlicedEllpackMatrixTest, setLikeTest )
{
    using SlicedEllpackMatrixType = typename TestFixture::SlicedEllpackMatrixType;

    test_SetLike< SlicedEllpackMatrixType, SlicedEllpackMatrixType >();
}

TYPED_TEST( SlicedEllpackMatrixTest, resetTest )
{
    using SlicedEllpackMatrixType = typename TestFixture::SlicedEllpackMatrixType;

    test_Reset< SlicedEllpackMatrixType >();
}

TYPED_TEST( SlicedEllpackMatrixTest, setElementTest )
{
    using SlicedEllpackMatrixType = typename TestFixture::SlicedEllpackMatrixType;

    test_SetElement< SlicedEllpackMatrixType >();
}

TYPED_TEST( SlicedEllpackMatrixTest, addElementTest )
{
    using SlicedEllpackMatrixType = typename TestFixture::SlicedEllpackMatrixType;

    test_AddElement< SlicedEllpackMatrixType >();
}

TYPED_TEST( SlicedEllpackMatrixTest, setRowTest )
{
    using SlicedEllpackMatrixType = typename TestFixture::SlicedEllpackMatrixType;

    test_SetRow< SlicedEllpackMatrixType >();
}

TYPED_TEST( SlicedEllpackMatrixTest, vectorProductTest )
{
    using SlicedEllpackMatrixType = typename TestFixture::SlicedEllpackMatrixType;

    test_VectorProduct< SlicedEllpackMatrixType >();
}

TYPED_TEST( SlicedEllpackMatrixTest, saveAndLoadTest )
{
    using SlicedEllpackMatrixType = typename TestFixture::SlicedEllpackMatrixType;

    test_SaveAndLoad< SlicedEllpackMatrixType >( "test_SparseMatrixTest_SlicedEllpack_segments" );
}

TYPED_TEST( SlicedEllpackMatrixTest, printTest )
{
    using SlicedEllpackMatrixType = typename TestFixture::SlicedEllpackMatrixType;

    test_Print< SlicedEllpackMatrixType >();
}

#endif

#include "../../main.h"
