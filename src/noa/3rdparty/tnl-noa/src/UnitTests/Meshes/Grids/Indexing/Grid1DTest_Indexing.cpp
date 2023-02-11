
#ifdef HAVE_GTEST

#include <gtest/gtest.h>

#include <TNL/Meshes/Grid.h>

#include "support.h"

using Implementations = ::testing::Types<
   TNL::Meshes::Grid<1, double, TNL::Devices::Sequential, int>,
   TNL::Meshes::Grid<1, float, TNL::Devices::Sequential, int>
>;

template< typename Grid >
class GridTestSuite: public ::testing::Test {
   protected:
      using GridType = Grid;
};

TYPED_TEST_SUITE(GridTestSuite, Implementations);

TYPED_TEST(GridTestSuite, TestGetEntityFromIndex_0D_Entity) {
   using GridType = typename TestFixture::GridType;

   GridType grid;

   testGetEntityFromIndex< GridType, 0 >( grid, { 10 } );
}

TYPED_TEST(GridTestSuite, TestGetEntityFromIndex_1D_Entity) {
   using GridType = typename TestFixture::GridType;

   GridType grid;

   testGetEntityFromIndex< GridType, 1 >( grid, { 10 } );
}


#endif // HAVE_GTEST

#include "../../../main.h"
