
#pragma once

#ifdef HAVE_GTEST

#include <gtest/gtest.h>

#include <TNL/Meshes/Grid.h>

#include "support.h"

#ifdef __CUDACC__
using Implementations = ::testing::Types<
   TNL::Meshes::Grid<3, double, TNL::Devices::Host, int>,
   TNL::Meshes::Grid<3, float, TNL::Devices::Host, int>,
   TNL::Meshes::Grid<3, double, TNL::Devices::Cuda, int>,
   TNL::Meshes::Grid<3, float, TNL::Devices::Cuda, int>
>;
#else
using Implementations = ::testing::Types<
   TNL::Meshes::Grid<3, double, TNL::Devices::Host, int>,
   TNL::Meshes::Grid<3, float, TNL::Devices::Host, int>
>;
#endif

template <class GridType>
class GridTestSuite: public ::testing::Test {
   protected:
      GridType grid;

#ifndef __CUDACC__
      void SetUp() override {
         if (std::is_same<typename GridType::DeviceType, TNL::Devices::Cuda>::value) {
            GTEST_SKIP() << "No CUDA available on host. Try to compile with CUDA instead";
         }
      }
#endif
};

TYPED_TEST_SUITE(GridTestSuite, Implementations);

TYPED_TEST(GridTestSuite, TestMeshDimensionGetter) {
   EXPECT_EQ(TypeParam::getMeshDimension(), 3) << "All grids must have dimension 3";
}

TYPED_TEST(GridTestSuite, TestSetWithParameterPack) {
   testDimensionSetByIndex<TypeParam>(this -> grid, 0, 0, 0);

   testDimensionSetByIndex<TypeParam>(this -> grid, 0, 0, 1);
   testDimensionSetByIndex<TypeParam>(this -> grid, 0, 1, 0);
   testDimensionSetByIndex<TypeParam>(this -> grid, 1, 0, 0);

   testDimensionSetByIndex<TypeParam>(this -> grid, 0, 2, 2);
   testDimensionSetByIndex<TypeParam>(this -> grid, 2, 0, 2);
   testDimensionSetByIndex<TypeParam>(this -> grid, 2, 2, 0);

   testDimensionSetByIndex<TypeParam>(this -> grid, 3, 3, 3);

   testDimensionSetByIndex<TypeParam>(this -> grid, 1, 1, 11211);
   testDimensionSetByIndex<TypeParam>(this -> grid, 232121, 21, 20);
   testDimensionSetByIndex<TypeParam>(this -> grid, 54544, 434343, 321341);
}

TYPED_TEST(GridTestSuite, TestSetWithCoordinates) {
   testDimensionSetByCoordinate<TypeParam>(this -> grid, { 0, 0, 0 });

   testDimensionSetByCoordinate<TypeParam>(this -> grid, { 0, 0, 1 });
   testDimensionSetByCoordinate<TypeParam>(this -> grid, { 0, 1, 0 });
   testDimensionSetByCoordinate<TypeParam>(this -> grid, { 1, 0, 0 });

   testDimensionSetByCoordinate<TypeParam>(this -> grid, { 0, 2, 2 });
   testDimensionSetByCoordinate<TypeParam>(this -> grid, { 2, 0, 2 });
   testDimensionSetByCoordinate<TypeParam>(this -> grid, { 2, 2, 0 });

   testDimensionSetByCoordinate<TypeParam>(this -> grid, { 3, 3, 3 });

   testDimensionSetByCoordinate<TypeParam>(this -> grid, { 1, 1, 11211 });
   testDimensionSetByCoordinate<TypeParam>(this -> grid, { 232121, 21, 20 });
   testDimensionSetByCoordinate<TypeParam>(this -> grid, { 54544, 434343, 321343 });
}

TYPED_TEST(GridTestSuite, TestEntitiesCount) {
   // GridType, Edges, Vertices | Edges
   testEntitiesCounts(this -> grid, { 0, 0, 0 }, { 0, 0, 0, 0 });

   testEntitiesCounts(this -> grid, { 1, 0, 0 }, { 0, 0, 0, 0 });
   testEntitiesCounts(this -> grid, { 0, 1, 0 }, { 0, 0, 0, 0 });
   testEntitiesCounts(this -> grid, { 0, 0, 1 }, { 0, 0, 0, 0 });

   testEntitiesCounts(this -> grid, { 1, 1, 1 }, { 8, 12, 6, 1 });

   testEntitiesCounts(this -> grid, { 2, 1, 1 }, { 12, 20, 11, 2 });
   testEntitiesCounts(this -> grid, { 1, 2, 1 }, { 12, 20, 11, 2 });
   testEntitiesCounts(this -> grid, { 1, 1, 2 }, { 12, 20, 11, 2 });

   testEntitiesCounts(this -> grid, { 2, 2, 2 }, { 27, 54, 36, 8 });

   testEntitiesCounts(this -> grid, { 2, 2, 3 }, { 36, 75, 52, 12 });
   testEntitiesCounts(this -> grid, { 2, 3, 2 }, { 36, 75, 52, 12 });
   testEntitiesCounts(this -> grid, { 3, 2, 2 }, { 36, 75, 52, 12 });

   testEntitiesCounts(this -> grid, { 50, 50, 50 }, { 51 * 51 * 51,
                                                      51 * 51 * 50 + 51 * 50 * 51 + 50 * 51 * 51,
                                                      51 * 50 * 50 + 50 * 51 * 50 + 50 * 50 * 51,
                                                      50 * 50 * 50 });
   testEntitiesCounts(this -> grid, { 50, 100, 150 }, { 51 * 101 * 151,
                                                        51 * 101 * 150 + 51 * 100 * 151 + 50 * 101 * 151,
                                                        51 * 100 * 150 + 50 * 101 * 150 + 50 * 100 * 151,
                                                        50 * 100 * 150 });
}

TYPED_TEST(GridTestSuite, TestOriginSet) {
   testOriginSetByCoordinate<TypeParam>(this -> grid, { 0.6, 1.2, 1.8 });
   testOriginSetByCoordinate<TypeParam>(this -> grid, { -1, 23232, -1 });
   testOriginSetByCoordinate<TypeParam>(this -> grid, { 100, -12132, 1231 });
   testOriginSetByCoordinate<TypeParam>(this -> grid, { -100000, 32112, 123 });
   testOriginSetByCoordinate<TypeParam>(this -> grid, { 323121, -100312, 1341231 });

   testOriginSetByIndex<TypeParam>(this -> grid, 0.6, 1.2, 1.8);
   testOriginSetByIndex<TypeParam>(this -> grid, -1, 23232, -1);
   testOriginSetByIndex<TypeParam>(this -> grid, 100, -12132, 1231);
   testOriginSetByIndex<TypeParam>(this -> grid, -100000, 32112, 123);
   testOriginSetByIndex<TypeParam>(this -> grid, 323121, -100312, 1341231);
}

TYPED_TEST(GridTestSuite, TestSpaceStepsGetter) {
   const int spaceStepsPowersSize = TypeParam::spaceStepsPowersSize;

   testSpaceStepsSetByCoordinate<TypeParam>(this -> grid, spaceStepsPowersSize, { 0.6, 1.2, 1.8 });
   testSpaceStepsSetByCoordinate<TypeParam>(this -> grid, spaceStepsPowersSize, { 1, 2, 1 });
   testSpaceStepsSetByCoordinate<TypeParam>(this -> grid, spaceStepsPowersSize, { 0.5, 3, 2 });
   testSpaceStepsSetByCoordinate<TypeParam>(this -> grid, spaceStepsPowersSize, { 5, 4, 3 });
   testSpaceStepsSetByCoordinate<TypeParam>(this -> grid, spaceStepsPowersSize, { 3, 2, 1 });

   testSpaceStepsSetByIndex<TypeParam>(this -> grid, spaceStepsPowersSize, 0.6, 1.2, 1.8);
   testSpaceStepsSetByIndex<TypeParam>(this -> grid, spaceStepsPowersSize, 1, 2, 1 );
   testSpaceStepsSetByIndex<TypeParam>(this -> grid, spaceStepsPowersSize, 0.5, 3, 2 );
   testSpaceStepsSetByIndex<TypeParam>(this -> grid, spaceStepsPowersSize, 5, 4, 3);
   testSpaceStepsSetByIndex<TypeParam>(this -> grid, spaceStepsPowersSize, 3, 2, 1);
}

TYPED_TEST(GridTestSuite, TestSpaceStepsPowersValues) {
   const int spaceStepsPowersSize = TypeParam::spaceStepsPowersSize;

   testSpaceStepsPowerValues(this -> grid, spaceStepsPowersSize, { 1., 1., 1.});
   testSpaceStepsPowerValues(this -> grid, spaceStepsPowersSize, { 0., 2., 4.});
   testSpaceStepsPowerValues(this -> grid, spaceStepsPowersSize, { 1., 2., 3.});
}

#endif

#include "../../../main.h"
