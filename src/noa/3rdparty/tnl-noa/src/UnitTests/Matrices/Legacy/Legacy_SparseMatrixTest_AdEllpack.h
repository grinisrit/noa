#include <TNL/Matrices/Legacy/AdEllpack.h>

#include "Legacy_SparseMatrixTest.hpp"
#include <iostream>

#ifdef HAVE_GTEST 
#include <gtest/gtest.h>

// test fixture for typed tests
template< typename Matrix >
class AdEllpackMatrixTest : public ::testing::Test
{
protected:
   using AdEllpackMatrixType = Matrix;
};

// types for which MatrixTest is instantiated
using AdEllpackMatrixTypes = ::testing::Types
<
    TNL::Matrices::Legacy::AdEllpack< int,    TNL::Devices::Host, int >,
    TNL::Matrices::Legacy::AdEllpack< long,   TNL::Devices::Host, int >,
    TNL::Matrices::Legacy::AdEllpack< float,  TNL::Devices::Host, int >,
    TNL::Matrices::Legacy::AdEllpack< double, TNL::Devices::Host, int >,
    TNL::Matrices::Legacy::AdEllpack< int,    TNL::Devices::Host, long >,
    TNL::Matrices::Legacy::AdEllpack< long,   TNL::Devices::Host, long >,
    TNL::Matrices::Legacy::AdEllpack< float,  TNL::Devices::Host, long >,
    TNL::Matrices::Legacy::AdEllpack< double, TNL::Devices::Host, long >
#ifdef __CUDACC__
   ,TNL::Matrices::Legacy::AdEllpack< int,    TNL::Devices::Cuda, int >,
    TNL::Matrices::Legacy::AdEllpack< long,   TNL::Devices::Cuda, int >,
    TNL::Matrices::Legacy::AdEllpack< float,  TNL::Devices::Cuda, int >,
    TNL::Matrices::Legacy::AdEllpack< double, TNL::Devices::Cuda, int >,
    TNL::Matrices::Legacy::AdEllpack< int,    TNL::Devices::Cuda, long >,
    TNL::Matrices::Legacy::AdEllpack< long,   TNL::Devices::Cuda, long >,
    TNL::Matrices::Legacy::AdEllpack< float,  TNL::Devices::Cuda, long >,
    TNL::Matrices::Legacy::AdEllpack< double, TNL::Devices::Cuda, long >
#endif
>;

TYPED_TEST_SUITE( AdEllpackMatrixTest, AdEllpackMatrixTypes);

TYPED_TEST( AdEllpackMatrixTest, setDimensionsTest )
{
    using AdEllpackMatrixType = typename TestFixture::AdEllpackMatrixType;
    
    test_SetDimensions< AdEllpackMatrixType >();
}

TYPED_TEST( AdEllpackMatrixTest, setLikeTest )
{
    using AdEllpackMatrixType = typename TestFixture::AdEllpackMatrixType;
    
    test_SetLike< AdEllpackMatrixType, AdEllpackMatrixType >();
}

TYPED_TEST( AdEllpackMatrixTest, resetTest )
{
    using AdEllpackMatrixType = typename TestFixture::AdEllpackMatrixType;
    
    test_Reset< AdEllpackMatrixType >();
}

TYPED_TEST( AdEllpackMatrixTest, setElementTest )
{    
    using AdEllpackMatrixType = typename TestFixture::AdEllpackMatrixType;
    
    test_SetElement< AdEllpackMatrixType >();
}

TYPED_TEST( AdEllpackMatrixTest, addElementTest )
{
    using AdEllpackMatrixType = typename TestFixture::AdEllpackMatrixType;
    
    test_AddElement< AdEllpackMatrixType >();
}

TYPED_TEST( AdEllpackMatrixTest, setRowTest )
{
    using AdEllpackMatrixType = typename TestFixture::AdEllpackMatrixType;
    
    test_SetRow< AdEllpackMatrixType >();
}

TYPED_TEST( AdEllpackMatrixTest, vectorProductTest )
{
    using AdEllpackMatrixType = typename TestFixture::AdEllpackMatrixType;
    
    test_VectorProduct< AdEllpackMatrixType >();
}

TYPED_TEST( AdEllpackMatrixTest, operatorEqualsTest )
{
    using AdEllpackMatrixType = typename TestFixture::AdEllpackMatrixType;
    
    test_OperatorEquals< AdEllpackMatrixType >();
}

TYPED_TEST( AdEllpackMatrixTest, saveAndLoadTest )
{
    using AdEllpackMatrixType = typename TestFixture::AdEllpackMatrixType;
    
    test_SaveAndLoad< AdEllpackMatrixType >( "test_SparseMatrixTest_AdEllpack" );
}

TYPED_TEST( AdEllpackMatrixTest, printTest )
{
    using AdEllpackMatrixType = typename TestFixture::AdEllpackMatrixType;
    
    test_Print< AdEllpackMatrixType >();
}

#ifdef NOT_WORKING
#endif

#endif


#include "../../main.h"
