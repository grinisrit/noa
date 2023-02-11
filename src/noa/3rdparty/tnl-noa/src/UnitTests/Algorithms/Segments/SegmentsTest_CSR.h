#include <TNL/Algorithms/Segments/CSR.h>

#include "SegmentsTest.hpp"
#include <iostream>

#ifdef HAVE_GTEST
#include <gtest/gtest.h>

// test fixture for typed tests
template< typename Segments >
class CSRSegmentsTest : public ::testing::Test
{
protected:
   using CSRSegmentsType = Segments;
};

// types for which MatrixTest is instantiated
using CSRSegmentsTypes = ::testing::Types
<
    TNL::Algorithms::Segments::CSR< TNL::Devices::Host, int    >,
    TNL::Algorithms::Segments::CSR< TNL::Devices::Host, long   >
#ifdef __CUDACC__
   ,TNL::Algorithms::Segments::CSR< TNL::Devices::Cuda, int    >,
    TNL::Algorithms::Segments::CSR< TNL::Devices::Cuda, long   >
#endif
>;

TYPED_TEST_SUITE( CSRSegmentsTest, CSRSegmentsTypes );

TYPED_TEST( CSRSegmentsTest, setSegmentsSizes_EqualSizes )
{
    using CSRSegmentsType = typename TestFixture::CSRSegmentsType;

    test_SetSegmentsSizes_EqualSizes< CSRSegmentsType >();
}

TYPED_TEST( CSRSegmentsTest, reduceAllSegments_MaximumInSegments )
{
    using CSRSegmentsType = typename TestFixture::CSRSegmentsType;

    test_reduceAllSegments_MaximumInSegments< CSRSegmentsType >();
}

#endif

#include "../../main.h"
