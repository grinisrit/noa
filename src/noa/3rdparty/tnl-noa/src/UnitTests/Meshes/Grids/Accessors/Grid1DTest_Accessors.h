
#pragma once

#ifdef HAVE_GTEST

#include <gtest/gtest.h>

#include <TNL/Meshes/Grid.h>

#include "support.h"

#ifdef __CUDACC__
using Implementations = ::testing::Types<
   TNL::Meshes::Grid<1, double, TNL::Devices::Host, int>,
   TNL::Meshes::Grid<1, float, TNL::Devices::Host, int>,
   TNL::Meshes::Grid<1, double, TNL::Devices::Cuda, int>,
   TNL::Meshes::Grid<1, float, TNL::Devices::Cuda, int>
>;
#else
using Implementations = ::testing::Types<
   TNL::Meshes::Grid<1, double, TNL::Devices::Host, int>,
   TNL::Meshes::Grid<1, float, TNL::Devices::Host, int>
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
   EXPECT_EQ(TypeParam::getMeshDimension(), 1) << "All grids must have dimension 1";
}

TYPED_TEST(GridTestSuite, TestSetWithParameterPack) {
   testDimensionSetByIndex<TypeParam>(this->grid, 0);
   testDimensionSetByIndex<TypeParam>(this->grid, 1);
   testDimensionSetByIndex<TypeParam>(this->grid, 2);
   testDimensionSetByIndex<TypeParam>(this->grid, 11211);
   testDimensionSetByIndex<TypeParam>(this->grid, 232121);
   testDimensionSetByIndex<TypeParam>(this->grid, 434343);
}

TYPED_TEST(GridTestSuite, TestSetWithCoordinates) {
   testDimensionSetByCoordinate<TypeParam>(this -> grid, { 0 });
   testDimensionSetByCoordinate<TypeParam>(this -> grid, { 1 });
   testDimensionSetByCoordinate<TypeParam>(this -> grid, { 2 });
   testDimensionSetByCoordinate<TypeParam>(this -> grid, { 10232 });
   testDimensionSetByCoordinate<TypeParam>(this -> grid, { 45235423 });
   testDimensionSetByCoordinate<TypeParam>(this -> grid, { 3231312 });
}

TYPED_TEST(GridTestSuite, TestEntitiesCount) {
   // GridType, Edges, Vertices | Edges
   testEntitiesCounts(this -> grid, { 0 }, { 0, 0 });
   testEntitiesCounts(this -> grid, { 1 }, { 2, 1 });
   testEntitiesCounts(this -> grid, { 2 }, { 3, 2 });

   testEntitiesCounts(this -> grid, { 100 }, { 101, 100 });
}

TYPED_TEST(GridTestSuite, TestOriginSet) {
   testOriginSetByCoordinate<TypeParam>(this -> grid, { 0.6 });
   testOriginSetByCoordinate<TypeParam>(this -> grid, { 1. });
   testOriginSetByCoordinate<TypeParam>(this -> grid, { 2. });
   testOriginSetByCoordinate<TypeParam>(this -> grid, { 0.4 });
   testOriginSetByCoordinate<TypeParam>(this -> grid, { 3. });

   testOriginSetByIndex<TypeParam>(this -> grid, 0.6);
   testOriginSetByIndex<TypeParam>(this -> grid, 1.);
   testOriginSetByIndex<TypeParam>(this -> grid, 2.);
   testOriginSetByIndex<TypeParam>(this -> grid, 0.4);
   testOriginSetByIndex<TypeParam>(this -> grid, 3.);
}

TYPED_TEST(GridTestSuite, TestSpaceStepsGetter) {
   const int spaceStepsPowersSize = TypeParam::spaceStepsPowersSize;

   testSpaceStepsSetByCoordinate<TypeParam>(this -> grid, spaceStepsPowersSize, { 0.6 });
   testSpaceStepsSetByCoordinate<TypeParam>(this -> grid, spaceStepsPowersSize, { 1 });
   testSpaceStepsSetByCoordinate<TypeParam>(this -> grid, spaceStepsPowersSize, { 4 });
   testSpaceStepsSetByCoordinate<TypeParam>(this -> grid, spaceStepsPowersSize, { 0.1 });
   testSpaceStepsSetByCoordinate<TypeParam>(this -> grid, spaceStepsPowersSize, { 4 });

   testSpaceStepsSetByIndex<TypeParam>(this -> grid, spaceStepsPowersSize, 0.6);
   testSpaceStepsSetByIndex<TypeParam>(this -> grid, spaceStepsPowersSize, 1);
   testSpaceStepsSetByIndex<TypeParam>(this -> grid, spaceStepsPowersSize, 4);
   testSpaceStepsSetByIndex<TypeParam>(this -> grid, spaceStepsPowersSize, 0.1);
   testSpaceStepsSetByIndex<TypeParam>(this -> grid, spaceStepsPowersSize, 4);
}

TYPED_TEST(GridTestSuite, TestSpaceStepsPowersValues) {
   const int spaceStepsPowersSize = TypeParam::spaceStepsPowersSize;

   testSpaceStepsPowerValues(this -> grid, spaceStepsPowersSize, { 0. });
   testSpaceStepsPowerValues(this -> grid, spaceStepsPowersSize, { 1. });
   testSpaceStepsPowerValues(this -> grid, spaceStepsPowersSize, { 2. });
}

#endif

#include "../../../main.h"
