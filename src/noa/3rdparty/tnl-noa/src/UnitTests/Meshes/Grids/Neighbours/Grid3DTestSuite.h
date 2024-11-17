
#pragma once

#ifdef HAVE_GTEST

#include <gtest/gtest.h>

#include <TNL/Meshes/Grid.h>

#include "support.h"

using Implementations = ::testing::Types<
   TNL::Meshes::Grid<3, double, TNL::Devices::Host, int>,
   TNL::Meshes::Grid<3, float, TNL::Devices::Host, int>,
   TNL::Meshes::Grid<3, double, TNL::Devices::Cuda, int>,
   TNL::Meshes::Grid<3, float, TNL::Devices::Cuda, int>
>;

template <class GridType>
class GridTestSuite: public ::testing::Test {
   protected:
      GridType grid;

      std::vector<typename GridType::CoordinatesType> dimensions = {
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

#ifndef __CUDACC__
      void SetUp() override {
         if (std::is_same<typename GridType::DeviceType, TNL::Devices::Cuda>::value) {
            GTEST_SKIP() << "No CUDA available on host. Try to compile with CUDA instead";
         }
      }
#endif
};

template<typename Grid, int EntityDimension, int NeighbourEntityDimension>
void testDynamicNeighbourEntityGetterForAllStencils(Grid& grid, const typename Grid::CoordinatesType& dimension, const int scale = 1) {
   for (int i = -1 * scale; i < 1 * scale; i++)
      for (int j = -1 * scale; j < 1 * scale; j++)
         for (int k = -1 * scale; k < 1 * scale; k++)
            testDynamicNeighbourEntityGetter<Grid, EntityDimension, NeighbourEntityDimension>(grid, dimension, typename Grid::CoordinatesType(i, j, k));
}

template<typename Grid, int EntityDimension, int NeighbourEntityDimension, int NeighbourEntityOrientation>
void testDynamicNeighbourEntityGetterForAllStencils(Grid& grid, const typename Grid::CoordinatesType& dimension, const int scale = 1) {
   for (int i = -1 * scale; i < 1 * scale; i++)
      for (int j = -1 * scale; j < 1 * scale; j++)
         for (int k = -1 * scale; k < 1 * scale; k++)
            testDynamicNeighbourEntityGetter<Grid, EntityDimension, NeighbourEntityDimension, NeighbourEntityOrientation>(grid, dimension, typename Grid::CoordinatesType(i, j, k));
}

#endif
