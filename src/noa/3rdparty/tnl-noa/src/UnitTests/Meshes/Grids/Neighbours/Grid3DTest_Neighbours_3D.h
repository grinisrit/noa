
#pragma once

#ifdef HAVE_GTEST

#include "Grid3DTestSuite.h"
#include <gtest/gtest.h>

TYPED_TEST_SUITE(GridTestSuite, Implementations);

TYPED_TEST(GridTestSuite, TestNeighbour_OF_3D_Entity_TO_0D_By_DynamicGetter) {
   // EntityDimension | NeighbourEntityDimension | Orientation
   for (const auto& dimension : this->dimensions) {
      testDynamicNeighbourEntityGetterForAllStencils<TypeParam, 3, 0>(this -> grid, dimension);
      testDynamicNeighbourEntityGetterForAllStencils<TypeParam, 3, 0, 0>(this -> grid, dimension);
   }
}

TYPED_TEST(GridTestSuite, TestNeighbour_OF_3D_Entity_TO_1D_By_DynamicGetter) {
   // EntityDimension | NeighbourEntityDimension | Orientation
   for (const auto& dimension : this->dimensions) {
      testDynamicNeighbourEntityGetterForAllStencils<TypeParam, 3, 1>(this -> grid, dimension);

      testDynamicNeighbourEntityGetterForAllStencils<TypeParam, 3, 1, 0>(this -> grid, dimension);
      testDynamicNeighbourEntityGetterForAllStencils<TypeParam, 3, 1, 1>(this -> grid, dimension);
      testDynamicNeighbourEntityGetterForAllStencils<TypeParam, 3, 1, 2>(this -> grid, dimension);
   }
}

TYPED_TEST(GridTestSuite, TestNeighbour_OF_3D_Entity_TO_2D_By_DynamicGetter) {
   // EntityDimension | NeighbourEntityDimension | Orientation
   for (const auto& dimension : this->dimensions) {
      testDynamicNeighbourEntityGetterForAllStencils<TypeParam, 3, 2>(this -> grid, dimension);

      testDynamicNeighbourEntityGetterForAllStencils<TypeParam, 3, 2, 0>(this -> grid, dimension);
      testDynamicNeighbourEntityGetterForAllStencils<TypeParam, 3, 2, 1>(this -> grid, dimension);
      testDynamicNeighbourEntityGetterForAllStencils<TypeParam, 3, 2, 2>(this -> grid, dimension);
   }
}

TYPED_TEST(GridTestSuite, TestNeighbour_OF_3D_Entity_TO_3D_By_DynamicGetter) {
   // EntityDimension | NeighbourEntityDimension | Orientation
   for (const auto& dimension : this->dimensions) {
      testDynamicNeighbourEntityGetterForAllStencils<TypeParam, 3, 3>(this -> grid, dimension);
      testDynamicNeighbourEntityGetterForAllStencils<TypeParam, 3, 3, 0>(this -> grid, dimension);
   }
}

#endif

#include "../../../main.h"
