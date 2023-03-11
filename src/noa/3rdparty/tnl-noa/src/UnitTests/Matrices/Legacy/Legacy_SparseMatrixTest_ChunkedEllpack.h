#include <Benchmarks/SpMV/ReferenceFormats/Legacy/ChunkedEllpack.h>

#include "Legacy_SparseMatrixTest.hpp"
#include <iostream>

#ifdef HAVE_GTEST 
#include <gtest/gtest.h>

// test fixture for typed tests
template< typename Matrix >
class ChunkedEllpackMatrixTest : public ::testing::Test
{
protected:
   using ChunkedEllpackMatrixType = Matrix;
};

using namespace TNL::Benchmarks::SpMV::ReferenceFormats;

// types for which MatrixTest is instantiated
using ChEllpackMatrixTypes = ::testing::Types
<
    Legacy::ChunkedEllpack< int,    TNL::Devices::Host, int >,
    Legacy::ChunkedEllpack< long,   TNL::Devices::Host, int >,
    Legacy::ChunkedEllpack< float,  TNL::Devices::Host, int >,
    Legacy::ChunkedEllpack< double, TNL::Devices::Host, int >,
    Legacy::ChunkedEllpack< int,    TNL::Devices::Host, long >,
    Legacy::ChunkedEllpack< long,   TNL::Devices::Host, long >,
    Legacy::ChunkedEllpack< float,  TNL::Devices::Host, long >,
    Legacy::ChunkedEllpack< double, TNL::Devices::Host, long >
#ifdef __CUDACC__
   ,Legacy::ChunkedEllpack< int,    TNL::Devices::Cuda, int >,
    Legacy::ChunkedEllpack< long,   TNL::Devices::Cuda, int >,
    Legacy::ChunkedEllpack< float,  TNL::Devices::Cuda, int >,
    Legacy::ChunkedEllpack< double, TNL::Devices::Cuda, int >,
    Legacy::ChunkedEllpack< int,    TNL::Devices::Cuda, long >,
    Legacy::ChunkedEllpack< long,   TNL::Devices::Cuda, long >,
    Legacy::ChunkedEllpack< float,  TNL::Devices::Cuda, long >,
    Legacy::ChunkedEllpack< double, TNL::Devices::Cuda, long >
#endif
>;

TYPED_TEST_SUITE( ChunkedEllpackMatrixTest, ChEllpackMatrixTypes);

TYPED_TEST( ChunkedEllpackMatrixTest, setDimensionsTest )
{
    using ChunkedEllpackMatrixType = typename TestFixture::ChunkedEllpackMatrixType;
    
    test_SetDimensions< ChunkedEllpackMatrixType >();
}

//TYPED_TEST( ChunkedEllpackMatrixTest, setCompressedRowLengthsTest )
//{
////    using ChunkedEllpackMatrixType = typename TestFixture::ChunkedEllpackMatrixType;
//    
////    test_SetCompressedRowLengths< ChunkedEllpackMatrixType >();
//    
//    bool testRan = false;
//    EXPECT_TRUE( testRan );
//    std::cout << "\nTEST DID NOT RUN. NOT WORKING.\n\n";
//    std::cout << "      This test is dependent on the input format. \n";
//    std::cout << "      Almost every format allocates elements per row differently.\n\n";
//    std::cout << "\n    TODO: Finish implementation of getNonZeroRowLength (Only non-zero elements, not the number of allocated elements.)\n\n";
//}

TYPED_TEST( ChunkedEllpackMatrixTest, setLikeTest )
{
    using ChunkedEllpackMatrixType = typename TestFixture::ChunkedEllpackMatrixType;
    
    test_SetLike< ChunkedEllpackMatrixType, ChunkedEllpackMatrixType >();
}

TYPED_TEST( ChunkedEllpackMatrixTest, resetTest )
{
    using ChunkedEllpackMatrixType = typename TestFixture::ChunkedEllpackMatrixType;
    
    test_Reset< ChunkedEllpackMatrixType >();
}

TYPED_TEST( ChunkedEllpackMatrixTest, setElementTest )
{
    using ChunkedEllpackMatrixType = typename TestFixture::ChunkedEllpackMatrixType;
    
    test_SetElement< ChunkedEllpackMatrixType >();
}

TYPED_TEST( ChunkedEllpackMatrixTest, addElementTest )
{
    using ChunkedEllpackMatrixType = typename TestFixture::ChunkedEllpackMatrixType;
    
    test_AddElement< ChunkedEllpackMatrixType >();
}

TYPED_TEST( ChunkedEllpackMatrixTest, setRowTest )
{
    using ChunkedEllpackMatrixType = typename TestFixture::ChunkedEllpackMatrixType;
    
    test_SetRow< ChunkedEllpackMatrixType >();
}

TYPED_TEST( ChunkedEllpackMatrixTest, vectorProductTest )
{
    using ChunkedEllpackMatrixType = typename TestFixture::ChunkedEllpackMatrixType;
    
    test_VectorProduct< ChunkedEllpackMatrixType >();
}

//TYPED_TEST( ChunkedEllpackMatrixTest, operatorEqualsTest )
//{
//    using ChunkedEllpackMatrixType = typename TestFixture::ChunkedEllpackMatrixType;
//    
//    test_OperatorEquals< ChunkedEllpackMatrixType >();
//}

TYPED_TEST( ChunkedEllpackMatrixTest, saveAndLoadTest )
{
    using ChunkedEllpackMatrixType = typename TestFixture::ChunkedEllpackMatrixType;
    
    test_SaveAndLoad< ChunkedEllpackMatrixType >( "test_SparseMatrixTest_ChunkedEllpack" );
}

TYPED_TEST( ChunkedEllpackMatrixTest, printTest )
{
    using ChunkedEllpackMatrixType = typename TestFixture::ChunkedEllpackMatrixType;
    
    test_Print< ChunkedEllpackMatrixType >();
}

#endif

#include "../../main.h"
