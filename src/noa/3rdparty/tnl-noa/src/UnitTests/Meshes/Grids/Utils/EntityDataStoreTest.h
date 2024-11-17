#pragma once

#ifdef HAVE_GTEST

#include <gtest/gtest.h>
#include <TNL/Meshes/Grid.h>

#include "../EntityDataStore.h"

using Index = int;
using Real = float;
using Device = TNL::Devices::Host;

template<typename Grid, typename GridEntity, typename DataStore>
void testTraverse(const Grid& grid, DataStore& dataStore, int entitiesCount ) {
  auto view = dataStore.getView();

  auto exec = [ & ]( const auto orientation, const typename Grid::CoordinatesType& normals ) {
    for(typename Grid::IndexType i = 0; i < entitiesCount; i++) {
      typename Grid::CoordinatesType coordinate;

      coordinate = i;

      GridEntity entity(grid, coordinate, normals, orientation);

      view.store(entity, i);

      auto prototype = view.getEntity(i);

      SCOPED_TRACE("Prototype: " + TNL::convertToString(prototype));

      EXPECT_EQ(1, prototype.calls);
      EXPECT_EQ(entity.getCoordinates(), prototype.coordinate);
      EXPECT_EQ(entity.getNormals(), prototype.normals);
      EXPECT_EQ(entity.getIndex(), prototype.index);
      EXPECT_EQ(entity.getOrientation(), prototype.orientation);
      EXPECT_EQ(entity.isBoundary(), prototype.isBoundary);
      EXPECT_EQ(entity.getCenter(), prototype.center);
      EXPECT_EQ(entity.getMeasure(), prototype.measure);

      view.clear(i);

      auto emptyPrototype = view.getEntity(i);

      typename Grid::CoordinatesType zeroCoordinate = 0;
      typename Grid::PointType zeroPoint = 0;

      EXPECT_EQ(0, emptyPrototype.calls);
      EXPECT_EQ(zeroCoordinate, emptyPrototype.coordinate);
      EXPECT_EQ(zeroCoordinate, emptyPrototype.normals);
      EXPECT_EQ(0, emptyPrototype.index);
      EXPECT_EQ(0, emptyPrototype.orientation);
      EXPECT_EQ(false, emptyPrototype.isBoundary);
      EXPECT_EQ(zeroPoint, emptyPrototype.center);
      EXPECT_EQ(0., emptyPrototype.measure);
    }
  };

  TNL::Meshes::Templates::ForEachOrientation<
    Index, GridEntity::getEntityDimension(), Grid::getMeshDimension()
  >::exec( exec );
}

TEST(EntityDataStoreTest, DataStore1DTest) {
  constexpr int Dimension = 1;
  using Grid = TNL::Meshes::Grid<Dimension, Real, Device, Index>;
  using DataStore = EntityDataStore<Index, Real, Device, Dimension>;

  using ZeroDimensionEntity = TNL::Meshes::GridEntity<Grid, 0>;
  using OneDimensionEntity = TNL::Meshes::GridEntity<Grid, 1>;

  Index entitiesCount = 10;

  Grid grid;
  grid.setDimensions(entitiesCount + 1);
  grid.setSpaceSteps(1.);

  DataStore store(entitiesCount);

  testTraverse<Grid, ZeroDimensionEntity, DataStore>(grid, store, entitiesCount);
  testTraverse<Grid, OneDimensionEntity, DataStore>(grid, store, entitiesCount);
}

TEST(EntityDataStoreTest, DataStore2DTest) {
  constexpr int Dimension = 2;

   using Grid = TNL::Meshes::Grid<Dimension, Real, Device, Index>;
  using DataStore = EntityDataStore<Index, Real, Device, Dimension>;

  using ZeroDimensionEntity = TNL::Meshes::GridEntity<Grid, 0>;
  using OneDimensionEntity = TNL::Meshes::GridEntity<Grid, 1>;
  using TwoDimensionEntity = TNL::Meshes::GridEntity<Grid, 2>;

  Index entitiesCount = 10;

  Grid grid;
  grid.setDimensions(entitiesCount + 1, entitiesCount + 1);
  grid.setSpaceSteps(1., 1.);

  DataStore store(entitiesCount);

  testTraverse<Grid, ZeroDimensionEntity, DataStore>(grid, store, entitiesCount);
  testTraverse<Grid, OneDimensionEntity, DataStore>(grid, store, entitiesCount);
  testTraverse<Grid, TwoDimensionEntity, DataStore>(grid, store, entitiesCount);
}

TEST(EntityDataStoreTest, DataStore3DTest) {
  constexpr int Dimension = 3;

  using Grid = TNL::Meshes::Grid<Dimension, Real, Device, Index>;
  using DataStore = EntityDataStore<Index, Real, Device, Dimension>;

  using ZeroDimensionEntity = TNL::Meshes::GridEntity<Grid, 0>;
  using OneDimensionEntity = TNL::Meshes::GridEntity<Grid, 1>;
  using TwoDimensionEntity = TNL::Meshes::GridEntity<Grid, 2>;
  using ThreeDimensionEntity = TNL::Meshes::GridEntity<Grid, 3>;

  Index entitiesCount = 10;

  Grid grid;
  grid.setDimensions(entitiesCount + 1, entitiesCount + 1, entitiesCount + 1);
  grid.setSpaceSteps(1., 1., 1.);

  DataStore store(entitiesCount);

  testTraverse<Grid, ZeroDimensionEntity, DataStore>(grid, store, entitiesCount);
  testTraverse<Grid, OneDimensionEntity, DataStore>(grid, store, entitiesCount);
  testTraverse<Grid, TwoDimensionEntity, DataStore>(grid, store, entitiesCount);
  testTraverse<Grid, ThreeDimensionEntity, DataStore>(grid, store, entitiesCount);
}

#endif
