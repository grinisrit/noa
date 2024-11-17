
#pragma once

#ifdef HAVE_GTEST

#include <gtest/gtest.h>

#include <TNL/Meshes/Grid.h>

#include "support.h"

#ifdef __CUDACC__
using Implementations = ::testing::Types<
   TNL::Meshes::Grid<2, double, TNL::Devices::Host, int>,
   TNL::Meshes::Grid<2, float, TNL::Devices::Host, int>,
   TNL::Meshes::Grid<2, double, TNL::Devices::Cuda, int>,
   TNL::Meshes::Grid<2, float, TNL::Devices::Cuda, int>
>;
#else
using Implementations = ::testing::Types<
   TNL::Meshes::Grid<2, double, TNL::Devices::Host, int>,
   TNL::Meshes::Grid<2, float, TNL::Devices::Host, int>
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
   EXPECT_EQ(TypeParam::getMeshDimension(), 2) << "All grids must have dimension 2";
}

TYPED_TEST(GridTestSuite, TestSetWithParameterPack) {
   testDimensionSetByIndex<TypeParam>(this -> grid, 0, 0);
   testDimensionSetByIndex<TypeParam>(this -> grid, 0, 1);
   testDimensionSetByIndex<TypeParam>(this -> grid, 0, 2);
   testDimensionSetByIndex<TypeParam>(this -> grid, 1, 11211);
   testDimensionSetByIndex<TypeParam>(this -> grid, 232121, 21);
   testDimensionSetByIndex<TypeParam>(this -> grid, 54544, 434343);
}

TYPED_TEST(GridTestSuite, TestSetWithCoordinates) {
   testDimensionSetByCoordinate<TypeParam>(this -> grid, { 0, 0 });
   testDimensionSetByCoordinate<TypeParam>(this -> grid, { 0, 1 });
   testDimensionSetByCoordinate<TypeParam>(this -> grid, { 0, 2 });
   testDimensionSetByCoordinate<TypeParam>(this -> grid, { 1, 11211 });
   testDimensionSetByCoordinate<TypeParam>(this -> grid, { 232121, 21 });
   testDimensionSetByCoordinate<TypeParam>(this -> grid, { 54544, 434343 });
}

TYPED_TEST(GridTestSuite, TestEntitiesCount) {
   // GridType, Edges, Vertices | Edges
   testEntitiesCounts(this -> grid, { 0, 0 }, { 0, 0, 0 });

   testEntitiesCounts(this -> grid, { 1, 0 }, { 0, 0, 0 });
   testEntitiesCounts(this -> grid, { 0, 1 }, { 0, 0, 0 });

   testEntitiesCounts(this -> grid, { 1, 1 }, { 4, 4, 1 });

   testEntitiesCounts(this -> grid, { 2, 1 }, { 6, 7, 2 });
   testEntitiesCounts(this -> grid, { 1, 2 }, { 6, 7, 2 });

   testEntitiesCounts(this -> grid, { 2, 2 }, { 9, 12, 4 });

   testEntitiesCounts(this -> grid, { 2, 3 }, { 12, 17, 6 });
   testEntitiesCounts(this -> grid, { 3, 2 }, { 12, 17, 6 });

   testEntitiesCounts(this -> grid, { 50, 50 }, { 51 * 51, 51 * 50 + 50 * 51, 50 * 50 });
   testEntitiesCounts(this -> grid, { 50, 100 }, { 51 * 101, 51 * 100 + 50 * 101, 50 * 100 });
}

TYPED_TEST(GridTestSuite, TestOriginSet) {
   testOriginSetByCoordinate<TypeParam>(this -> grid, { 0.6, 1.2 });
   testOriginSetByCoordinate<TypeParam>(this -> grid, { -1, 23232 });
   testOriginSetByCoordinate<TypeParam>(this -> grid, { 100, -12132 });
   testOriginSetByCoordinate<TypeParam>(this -> grid, { -100000, 32112 });
   testOriginSetByCoordinate<TypeParam>(this -> grid, { 323121, -100312 });

   testOriginSetByIndex<TypeParam>(this -> grid, 0.6, 1.2);
   testOriginSetByIndex<TypeParam>(this -> grid, -1, 23232);
   testOriginSetByIndex<TypeParam>(this -> grid, 100, -12132);
   testOriginSetByIndex<TypeParam>(this -> grid, -100000, 32112);
   testOriginSetByIndex<TypeParam>(this -> grid, 323121, -100312);
}

TYPED_TEST(GridTestSuite, TestSpaceStepsGetter) {
   const int spaceStepsPowersSize = TypeParam::spaceStepsPowersSize;

   testSpaceStepsSetByCoordinate<TypeParam>(this -> grid, spaceStepsPowersSize, { 0.6, 1.2 });
   testSpaceStepsSetByCoordinate<TypeParam>(this -> grid, spaceStepsPowersSize, { 1, 2 });
   testSpaceStepsSetByCoordinate<TypeParam>(this -> grid, spaceStepsPowersSize, { 4, 3 });
   testSpaceStepsSetByCoordinate<TypeParam>(this -> grid, spaceStepsPowersSize, { 0.1, 10 });
   testSpaceStepsSetByCoordinate<TypeParam>(this -> grid, spaceStepsPowersSize, { 4, 5 });

   testSpaceStepsSetByIndex<TypeParam>(this -> grid, spaceStepsPowersSize, 0.6, 1.2);
   testSpaceStepsSetByIndex<TypeParam>(this -> grid, spaceStepsPowersSize, 1, 2);
   testSpaceStepsSetByIndex<TypeParam>(this -> grid, spaceStepsPowersSize, 4, 3);
   testSpaceStepsSetByIndex<TypeParam>(this -> grid, spaceStepsPowersSize, 0.1, 10);
   testSpaceStepsSetByIndex<TypeParam>(this -> grid, spaceStepsPowersSize, 4, 5);
}

TYPED_TEST(GridTestSuite, TestSpaceStepsPowersValues) {
  const int spaceStepsPowersSize = TypeParam::spaceStepsPowersSize;

  testSpaceStepsPowerValues(this -> grid, spaceStepsPowersSize, { 1., 1. });
  testSpaceStepsPowerValues(this -> grid, spaceStepsPowersSize, { 0., 2. });
  testSpaceStepsPowerValues(this -> grid, spaceStepsPowersSize, { 2., 3. });
}

#endif

#include "../../../main.h"
