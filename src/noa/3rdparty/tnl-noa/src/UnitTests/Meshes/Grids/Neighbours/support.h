#pragma once

#ifdef HAVE_GTEST

#include "../EntityDataStore.h"
#include "../CoordinateIterator.h"
#include <gtest/gtest.h>

template<typename Grid, int EntityDimension, int NeighbourEntityDimension>
class NeighbourGridEntityGetterTestCase {
   public:
      using Index = typename Grid::IndexType;
      using Real = typename Grid::RealType;
      using Coordinate = typename Grid::CoordinatesType;
      using DataStore = EntityDataStore<Index, Real, typename Grid::DeviceType, Grid::getMeshDimension()>;
      using HostDataStore = EntityDataStore<Index, Real, TNL::Devices::Host, Grid::getMeshDimension()>;

      void storeByDynamicAccessor(const Grid& grid, DataStore& store, const Coordinate& offset) {
         SCOPED_TRACE("Store using dynamic accessor without orientation");

         constexpr int neighbourOrientationsCount = Grid::getEntityOrientationsCount(NeighbourEntityDimension);

         auto view = store.getView();
         auto update = [=] __cuda_callable__(const typename Grid::template EntityType<EntityDimension>& entity) mutable {
            int neighbourEntityOrientation = TNL::min(entity.getOrientation(), neighbourOrientationsCount - 1);
            Coordinate alignedCoordinate = entity.getCoordinates() + offset;
            Coordinate boundary = grid.getDimensions() + grid.template getNormals<NeighbourEntityDimension>(neighbourEntityOrientation);

            if ((alignedCoordinate >= 0 && alignedCoordinate < boundary)) {
               auto neighbour = entity.template getNeighbourEntity<NeighbourEntityDimension>(offset);

               neighbour.refresh();

               view.store(neighbour, entity.getIndex());
            }
         };

         grid.template forAllEntities<EntityDimension>(update);
      }

      template<int NeighbourEntityOrientation>
      void storeByDynamicAccessorWithOrientation(const Grid& grid, DataStore& store, const Coordinate& offset) {
         SCOPED_TRACE("Store using dynamic accessor with orientation");

         auto view = store.getView();
         auto update = [=] __cuda_callable__(const typename Grid::template EntityType<EntityDimension>& entity) mutable {
            Coordinate alignedCoordinate = entity.getCoordinates() + offset;
            Coordinate boundary = grid.getDimensions() + grid.template getNormals<NeighbourEntityDimension>(NeighbourEntityOrientation);

            if ((alignedCoordinate >= 0 && alignedCoordinate < boundary)) {
               auto neighbour = entity.template getNeighbourEntity<NeighbourEntityDimension, NeighbourEntityOrientation>(offset);

               neighbour.refresh();

               view.store(neighbour, entity.getIndex());
            }
         };

         grid.template forAllEntities<EntityDimension>(update);
      }

      void verify(const Grid& grid, DataStore& store, const Coordinate& offset, const int entityOrientation) {
         auto hostStore = store.template move<TNL::Devices::Host>();
         auto hostStoreView = hostStore.getView();

         constexpr int orientationsCount = Grid::getEntityOrientationsCount(EntityDimension);
         constexpr int neighbourEntityOrientationsCount = Grid::getEntityOrientationsCount(NeighbourEntityDimension);

         auto verify = [&](const auto orientation) {
            GridCoordinateIterator<orientation> iterator(grid.getDimensions());

            if (!iterator.canIterate()) {
               SCOPED_TRACE("Skip iteration");
               EXPECT_EQ(hostStore.getCallsView().getSize(), 0) << "Expect, that we can't iterate, when grid is empty";
               return;
            }

            int neighbourEntityOrientation = 0;

            if (entityOrientation == -1) {
               int tmp = orientation;
               neighbourEntityOrientation = TNL::min(tmp, neighbourEntityOrientationsCount - 1);
            } else {
               neighbourEntityOrientation = entityOrientation;
            }

            const Coordinate neighbourEntityNormals = grid.template getNormals<NeighbourEntityDimension>(neighbourEntityOrientation);

            do {
               Index parentEntityIndex = iterator.getIndex(grid);
               auto neighbourEntity = hostStoreView.getEntity(parentEntityIndex);

               Coordinate alignedCoordinate = iterator.getCoordinate() + offset;

               // Unable to get entity out of bounds
               bool expectCall = alignedCoordinate >= Coordinate(0) && alignedCoordinate < grid.getDimensions() + neighbourEntityNormals;

               EXPECT_EQ(expectCall, neighbourEntity.calls == 1) <<
                         "Expect, that the parent entity was called";
               EXPECT_EQ(expectCall ? alignedCoordinate : Coordinate(0), neighbourEntity.coordinate) <<
                         "Expect, that the coordinate is updated";
               EXPECT_EQ(expectCall ? neighbourEntityNormals : Coordinate(0), neighbourEntity.normals) <<
                         "Expect, that the normals is updated";
               EXPECT_EQ(expectCall ? neighbourEntityOrientation : 0, neighbourEntity.orientation) <<
                         "Expect, that the parent entity was called";
            } while (!iterator.next());
         };

         TNL::Algorithms::staticFor< int, 0, orientationsCount >(verify);
      }
   protected:
      template<int Orientation>
      class GridCoordinateIterator: public CoordinateIterator<typename Grid::IndexType, Grid::getMeshDimension()> {
         public:
            using Base = CoordinateIterator<typename Grid::IndexType, Grid::getMeshDimension()>;
            using EntityNormals = TNL::Meshes::NormalsGetter<Index, EntityDimension, Grid::getMeshDimension()>;

            GridCoordinateIterator(const Coordinate& end): Base(Coordinate(0), end + EntityNormals::template getNormals<Orientation>()) {
               for (Index i = 0; i < this -> current.getSize(); i++) {
                  this -> start[i] = 0;
                  this -> current[i] = 0;
               }
            }

            Coordinate getCoordinate() const {
               return this -> current;
            }

            Index getIndex(const Grid& grid) const {
               Index result = 0;

               for (Index i = 0; i < this->current.getSize(); i++) {
                  if (i == 0) {
                     result += this->current[i];
                  } else {
                     Index offset = 1;

                     for (Index j = 0; j < i; j++) offset *= this->end[j];

                     result += this->current[i] * offset;
                  }
               }

               for (Index i = 0; i < Orientation; i++) result += grid.getOrientedEntitiesCount(EntityDimension, i);

               return result;
            }
      };
};


template<typename Grid,
         int EntityDimension,
         int NeighbourEntityDimension>
void testDynamicNeighbourEntityGetter(Grid& grid, const typename Grid::CoordinatesType& dimensions, const typename Grid::CoordinatesType& offset) {
   SCOPED_TRACE("Grid Dimension: " + TNL::convertToString(Grid::getMeshDimension()));
   SCOPED_TRACE("Entity Dimension: " + TNL::convertToString(EntityDimension));
   SCOPED_TRACE("Neighbour Entity Dimension: " + TNL::convertToString(NeighbourEntityDimension));
   SCOPED_TRACE("Dimension: " + TNL::convertToString(dimensions));

   EXPECT_NO_THROW(grid.setDimensions(dimensions)) << "Verify, that the set of" << dimensions << " doesn't cause assert";

   using Test = NeighbourGridEntityGetterTestCase<Grid, EntityDimension, NeighbourEntityDimension>;

   Test test;
   typename Test::DataStore store(grid.getEntitiesCount(EntityDimension));

   test.storeByDynamicAccessor(grid, store, offset);
   test.verify(grid, store, offset, -1);
}

template<typename Grid,
         int EntityDimension,
         int NeighbourEntityDimension,
         int NeighbourEntityOrientation>
void testDynamicNeighbourEntityGetter(Grid& grid, const typename Grid::CoordinatesType& dimensions, const typename Grid::CoordinatesType& offset) {
   SCOPED_TRACE("Grid Dimension: " + TNL::convertToString(Grid::getMeshDimension()));
   SCOPED_TRACE("Entity Dimension: " + TNL::convertToString(EntityDimension));
   SCOPED_TRACE("Neighbour Entity Dimension: " + TNL::convertToString(NeighbourEntityDimension));
   SCOPED_TRACE("Neighbour Entity Orientation: " + TNL::convertToString(NeighbourEntityOrientation));
   SCOPED_TRACE("Dimension: " + TNL::convertToString(dimensions));

   EXPECT_NO_THROW(grid.setDimensions(dimensions)) << "Verify, that the set of" << dimensions << " doesn't cause assert";

   using Test = NeighbourGridEntityGetterTestCase<Grid, EntityDimension, NeighbourEntityDimension>;

   Test test;
   typename Test::DataStore store(grid.getEntitiesCount(EntityDimension));

   test.template storeByDynamicAccessorWithOrientation<NeighbourEntityOrientation>(grid, store, offset);
   test.verify(grid, store, offset, NeighbourEntityOrientation);
}


#endif
