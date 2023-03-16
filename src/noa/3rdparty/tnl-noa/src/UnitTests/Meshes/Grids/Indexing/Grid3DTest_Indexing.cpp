
#ifdef HAVE_GTEST

#include <gtest/gtest.h>

#include <TNL/Meshes/Grid.h>

#include "support.h"

const int size = 3;

using Implementations = ::testing::Types<
   TNL::Meshes::Grid<3, double, TNL::Devices::Sequential, int>,
   TNL::Meshes::Grid<3, float, TNL::Devices::Sequential, int>
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

   testGetEntityFromIndex< GridType, 0 >( grid, { size, size, size } );
}

TYPED_TEST(GridTestSuite, TestGetEntityFromIndex_1D_Entity) {
   using GridType = typename TestFixture::GridType;

   GridType grid;

   testGetEntityFromIndex< GridType, 1 >( grid, { size, size, size } );
}

TYPED_TEST(GridTestSuite, TestGetEntityFromIndex_2D_Entity) {
   using GridType = typename TestFixture::GridType;

   GridType grid;

   testGetEntityFromIndex< GridType, 2 >( grid, { size, size, size } );
}

TYPED_TEST(GridTestSuite, TestGetEntityFromIndex_3D_Entity) {
   using GridType = typename TestFixture::GridType;

   GridType grid;

   testGetEntityFromIndex< GridType, 3 >( grid, { size, size, size } );
}


#endif // HAVE_GTEST

#include "../../../main.h"
