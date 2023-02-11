
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

      std::vector<typename GridType::CoordinatesType > dimensions = {
         { 1, 1, 1 },
         { 2, 1, 1 },
         { 1, 2, 1 },
         { 1, 1, 2 },
         { 2, 2, 2 },
         { 3, 3, 3 },
         { 10, 1, 1 },
         { 1, 10, 1 },
         { 1, 1, 10 }
#if defined(__CUDACC__) || defined(HAVE_OPENMP)
         ,
         { 10, 10, 1 },
         { 1, 10, 10 },
         { 10, 1, 10 },
         { 10, 10, 10 }
#endif
      };

      std::vector<typename GridType::PointType > origins = {
         { 1, 1, 1 },
         { -10, -10, 2 },
         { 0.1, 0.1, 0.1 },
         { 1, 0.2, 1 },
         { 1, -1, 2 },
         { -2, -2, -2 }
      };

      std::vector<typename GridType::PointType > spaceSteps = {
         { 0.1, 0.1, 0.1 },
         { 2, 2.4, 2 },
         { 0.1, 3.1, 1 },
         { 1, 4, 2 },
         { 12, 2, 1 }
      };

#ifndef __CUDACC__
      void SetUp() override {
         if (std::is_same<typename GridType::DeviceType, TNL::Devices::Cuda>::value) {
            GTEST_SKIP() << "No CUDA available on host. Try to compile with CUDA instead";
         }
      }
#endif
};

TYPED_TEST_SUITE(GridTestSuite, Implementations);

TYPED_TEST(GridTestSuite, TestForAllTraverse_0D_Entity) {
   for (const auto& dimension: this->dimensions)
      for (const auto& origin: this->origins)
         for (const auto& spaceStep: this->spaceSteps)
            testForAllTraverse<TypeParam, 0>(this->grid, dimension, origin, spaceStep);
}

TYPED_TEST(GridTestSuite, TestForAllTraverse_1D_Entity) {
   for (const auto& dimension: this->dimensions)
      for (const auto& origin: this->origins)
         for (const auto& spaceStep: this->spaceSteps)
            testForAllTraverse<TypeParam, 1>(this->grid, dimension, origin, spaceStep);
}

TYPED_TEST(GridTestSuite, TestForAllTraverse_2D_Entity) {
   for (const auto& dimension: this->dimensions)
      for (const auto& origin: this->origins)
         for (const auto& spaceStep: this->spaceSteps)
            testForAllTraverse<TypeParam, 2>(this->grid, dimension, origin, spaceStep);
}

TYPED_TEST(GridTestSuite, TestForAllTraverse_3D_Entity) {
   for (const auto& dimension: this->dimensions)
      for (const auto& origin: this->origins)
         for (const auto& spaceStep: this->spaceSteps)
            testForAllTraverse<TypeParam, 3>(this->grid, dimension, origin, spaceStep);
}

TYPED_TEST(GridTestSuite, TestForInteriorTraverse_0D_Entity) {
   for (const auto& dimension : this->dimensions)
      testForInteriorTraverse<TypeParam, 0>(this->grid, dimension);
}

TYPED_TEST(GridTestSuite, TestForInteriorTraverse_1D_Entity) {
   for (const auto& dimension : this->dimensions)
      testForInteriorTraverse<TypeParam, 1>(this->grid, dimension);
}

TYPED_TEST(GridTestSuite, TestForInteriorTraverse_2D_Entity) {
   for (const auto& dimension : this->dimensions)
      testForInteriorTraverse<TypeParam, 2>(this->grid, dimension);
}

TYPED_TEST(GridTestSuite, TestForInteriorTraverse_3D_Entity) {
   for (const auto& dimension : this->dimensions)
      testForInteriorTraverse<TypeParam, 3>(this->grid, dimension);
}

TYPED_TEST(GridTestSuite, TestForBoundaryTraverse_0D_Entity) {
   for (const auto& dimension : this->dimensions)
      testForBoundaryTraverse<TypeParam, 0>(this->grid, dimension);
}

TYPED_TEST(GridTestSuite, TestForBoundaryTraverse_1D_Entity) {
   for (const auto& dimension : this->dimensions)
      testForBoundaryTraverse<TypeParam, 1>(this->grid, dimension);
}

TYPED_TEST(GridTestSuite, TestForBoundaryTraverse_2D_Entity) {
   for (const auto& dimension : this->dimensions)
      testForBoundaryTraverse<TypeParam, 2>(this->grid, dimension);
}

TYPED_TEST(GridTestSuite, TestForBoundaryTraverse_3D_Entity) {
   for (const auto& dimension : this->dimensions)
      testForBoundaryTraverse<TypeParam, 3>(this->grid, dimension);
}

TYPED_TEST(GridTestSuite, TestBoundaryUnionInternalEqualAllProperty_0D_Entity) {
   for (const auto& dimension : this -> dimensions)
      testBoundaryUnionInteriorEqualAllProperty<TypeParam, 0>(this -> grid, dimension);
}

TYPED_TEST(GridTestSuite, TestBoundaryUnionInternalEqualAllProperty_1D_Entity) {
   for (const auto& dimension : this -> dimensions)
      testBoundaryUnionInteriorEqualAllProperty<TypeParam, 1>(this -> grid, dimension);
}

TYPED_TEST(GridTestSuite, TestBoundaryUnionInternalEqualAllProperty_2D_Entity) {
   for (const auto& dimension : this -> dimensions)
      testBoundaryUnionInteriorEqualAllProperty<TypeParam, 2>(this -> grid, dimension);
}

TYPED_TEST(GridTestSuite, TestBoundaryUnionInternalEqualAllProperty_3D_Entity) {
   for (const auto& dimension : this -> dimensions)
      testBoundaryUnionInteriorEqualAllProperty<TypeParam, 3>(this -> grid, dimension);
}

TYPED_TEST(GridTestSuite, TestAllMinusBoundaryEqualInteriorProperty_0D_Entity) {
   for (const auto& dimension : this -> dimensions)
      testAllMinusBoundaryEqualInteriorProperty<TypeParam, 0>(this -> grid, dimension);
}

TYPED_TEST(GridTestSuite, TestAllMinusBoundaryEqualInteriorProperty_1D_Entity) {
   for (const auto& dimension : this -> dimensions)
      testAllMinusBoundaryEqualInteriorProperty<TypeParam, 1>(this -> grid, dimension);
}

TYPED_TEST(GridTestSuite, TestAllMinusBoundaryEqualInteriorProperty_2D_Entity) {
   for (const auto& dimension : this -> dimensions)
      testAllMinusBoundaryEqualInteriorProperty<TypeParam, 2>(this -> grid, dimension);
}

TYPED_TEST(GridTestSuite, TestAllMinusBoundaryEqualInteriorProperty_3D_Entity) {
   for (const auto& dimension : this -> dimensions)
      testAllMinusBoundaryEqualInteriorProperty<TypeParam, 3>(this -> grid, dimension);
}

TYPED_TEST(GridTestSuite, TestAllMinusInteriorEqualBoundaryProperty_0D_Entity) {
   for (const auto& dimension : this -> dimensions)
      testAllMinusInteriorEqualBoundaryProperty<TypeParam, 0>(this -> grid, dimension);
}

TYPED_TEST(GridTestSuite, TestAllMinusInteriorEqualBoundaryProperty_1D_Entity) {
   for (const auto& dimension : this -> dimensions)
      testAllMinusInteriorEqualBoundaryProperty<TypeParam, 1>(this -> grid, dimension);
}

TYPED_TEST(GridTestSuite, TestAllMinusInteriorEqualBoundaryProperty_2D_Entity) {
   for (const auto& dimension : this -> dimensions)
      testAllMinusInteriorEqualBoundaryProperty<TypeParam, 2>(this -> grid, dimension);
}

TYPED_TEST(GridTestSuite, TestAllMinusInteriorEqualBoundaryProperty_3D_Entity) {
   for (const auto& dimension : this -> dimensions)
      testAllMinusInteriorEqualBoundaryProperty<TypeParam, 3>(this -> grid, dimension);
}

#endif

#include "../../../main.h"
