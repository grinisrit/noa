#pragma once

#ifdef HAVE_GTEST

#include <gtest/gtest.h>
#include <functional>

#include <TNL/Algorithms/staticFor.h>
#include <TNL/Containers/Array.h>
#include <TNL/Meshes/GridDetails/NormalsGetter.h>

#include "../CoordinateIterator.h"
#include "../EntityDataStore.h"

template< typename Grid, int EntityDimension, int Orientation >
class GridCoordinateIterator : public CoordinateIterator< typename Grid::IndexType, Grid::getMeshDimension() >
{
public:
   using Coordinate = typename Grid::CoordinatesType;
   using Point = typename Grid::PointType;
   using Index = typename Grid::IndexType;
   using Real = typename Grid::RealType;
   using Base = CoordinateIterator< Index, Grid::getMeshDimension() >;
   using EntityNormals = TNL::Meshes::NormalsGetter< Index, EntityDimension, Grid::getMeshDimension() >;

   GridCoordinateIterator( const typename Grid::CoordinatesType& end )
   : Base( Coordinate( 0 ), end + EntityNormals::template getNormals< Orientation >() )
   {
      for( Index i = 0; i < this->current.getSize(); i++ ) {
         this->start[ i ] = 0;
         this->current[ i ] = 0;
      }
   }

   bool
   isBoundary( const Grid& grid ) const
   {
      switch( EntityDimension ) {
         case Grid::getMeshDimension():
            for( Index i = 0; i < this->current.getSize(); i++ )
               if( this->current[ i ] == 0 || this->current[ i ] == grid.getDimensions()[ i ] - 1 )
                  return true;

            break;
         default:
            for( Index i = 0; i < this->current.getSize(); i++ )
               if( getNormals()[ i ] && ( this->current[ i ] == 0 || this->current[ i ] == grid.getDimensions()[ i ] ) )
                  return true;
            break;
      }

      return false;
   }

   Coordinate
   getCoordinate() const
   {
      return this->current;
   }

   Index
   getIndex( const Grid& grid ) const
   {
      Index result = 0;

      for( Index i = 0; i < this->current.getSize(); i++ ) {
         if( i == 0 ) {
            result += this->current[ i ];
         }
         else {
            Index offset = 1;

            for( Index j = 0; j < i; j++ )
               offset *= this->end[ j ];

            result += this->current[ i ] * offset;
         }
      }

      for( Index i = 0; i < Orientation; i++ )
         result += grid.getOrientedEntitiesCount( EntityDimension, i );

      return result;
   }

   Coordinate
   getNormals() const
   {
      return EntityNormals::template getNormals< Orientation >();
   }

   Point
   getCenter( const Grid& grid ) const
   {
      Point origin = grid.getOrigin(), center, spaceSteps = grid.getSpaceSteps();

      Coordinate normals = getNormals();

      for( Index i = 0; i < this->current.getSize(); i++ )
         center[ i ] = origin[ i ] + ( this->current[ i ] + (Real) ( 0.5 * ! normals[ i ] ) ) * spaceSteps[ i ];

      return center;
   }

   Real
   getMeasure( const Grid& grid ) const
   {
      if( EntityDimension == 0 ) {
         return 0.0;
      }

      Coordinate normals = getNormals(), powers;

      for( Index i = 0; i < this->current.getSize(); i++ )
         powers[ i ] = ! normals[ i ];

      return grid.getSpaceStepsProducts( powers );
   }
};

template<typename Grid, int EntityDimension>
class GridTraverseTestCase {
   public:
      using Index = typename Grid::IndexType;
      using Real = typename Grid::RealType;
      using Coordinate = typename Grid::CoordinatesType;
      using Point = typename Grid::PointType;
      using DataStore = EntityDataStore<Index, Real, typename Grid::DeviceType, Grid::getMeshDimension()>;
      using HostDataStore = EntityDataStore<Index, Real, TNL::Devices::Host, Grid::getMeshDimension()>;

      template<int Orientation>
      using Iterator = GridCoordinateIterator<Grid, EntityDimension, Orientation>;

      // NVCC is incapable of deducing generic lambda
      using UpdateFunctionType = std::function<void(const typename Grid::template EntityType<EntityDimension>&)>;

      void storeAll(const Grid& grid, DataStore& store) const {
         SCOPED_TRACE("Store all");

         auto view = store.getView();

         auto update = [=] __cuda_callable__ (const typename Grid::template EntityType<EntityDimension>& entity) mutable {
            view.store(entity);
         };

         grid.template forAllEntities<EntityDimension>(update);
      }
      void storeBoundary(const Grid& grid, DataStore& store) const {
         SCOPED_TRACE("Store boundary");

         auto view = store.getView();

         auto update = [=] __cuda_callable__ (const typename Grid::template EntityType<EntityDimension>& entity) mutable {
            view.store(entity);
         };

         grid.template forBoundaryEntities<EntityDimension>(update);
      }
      void storeInterior(const Grid& grid, DataStore& store) const {
         SCOPED_TRACE("Store interior");

         auto view = store.getView();

         auto update = [=] __cuda_callable__ (const typename Grid::template EntityType<EntityDimension>& entity) mutable {
            view.store(entity);
         };

         grid.template forInteriorEntities<EntityDimension>(update);
      }
      void clearAll(const Grid& grid, DataStore& store) const {
         SCOPED_TRACE("Clear all");

         auto view = store.getView();

         auto update = [=] __cuda_callable__ (const typename Grid::template EntityType<EntityDimension>& entity) mutable {
            view.clear(entity);
         };

         grid.template forAllEntities<EntityDimension>(update);
      }
      void clearBoundary(const Grid& grid, DataStore& store) const {
         SCOPED_TRACE("Clear boundary");

         auto view = store.getView();

         auto update = [=] __cuda_callable__ (const typename Grid::template EntityType<EntityDimension>& entity) mutable {
            view.clear(entity);
         };

         grid.template forBoundaryEntities<EntityDimension>(update);
      }
      void clearInterior(const Grid& grid, DataStore& store) const {
         SCOPED_TRACE("Clear interior");

         auto view = store.getView();

         auto update = [=] __cuda_callable__ (const typename Grid::template EntityType<EntityDimension>& entity) mutable {
            view.clear(entity);
         };

         grid.template forInteriorEntities<EntityDimension>(update);
      }

      void verifyAll(const Grid& grid, const DataStore& store) const {
         auto hostStore = store.template move<TNL::Devices::Host>();
         auto hostStoreView = hostStore.getView();

         constexpr int orientationsCount = Grid::getEntityOrientationsCount(EntityDimension);

         SCOPED_TRACE("Verifying forAll");
         SCOPED_TRACE("Orientations Count: " + TNL::convertToString(orientationsCount));

         ASSERT_GT(orientationsCount, 0) << "Every entity must have at least one orientation";

         auto callsView = hostStore.getCallsView();

         for (Index i = 0; i < callsView.getSize(); i++)
            EXPECT_EQ(callsView[i], 1) << "Expect each index to be called only once";

         auto verify = [&](const auto orientation) {
            Iterator<orientation> iterator(grid.getDimensions());

            if (!iterator.canIterate()) {
               SCOPED_TRACE("Skip iteration");
               EXPECT_EQ(callsView.getSize(), 0) << "Expect, that we can't iterate, when grid is empty";
               return;
            }

            do {
               verifyEntity(grid, iterator, hostStoreView, true);
            } while (!iterator.next());
         };

         TNL::Algorithms::staticFor< int, 0, orientationsCount >(verify);
      }
      void verifyBoundary(const Grid& grid, const DataStore& store) const {
         auto hostStore = store.template move<TNL::Devices::Host>();
         auto hostStoreView = hostStore.getView();

         constexpr int orientationsCount = Grid::getEntityOrientationsCount(EntityDimension);

         SCOPED_TRACE("Verifying forBoundary");
         SCOPED_TRACE("Orientations Count: " + TNL::convertToString(orientationsCount));

         ASSERT_GT(orientationsCount, 0) << "Every entity must have at least one orientation";

         auto verify = [&](const auto orientation) {
            Iterator<orientation> iterator(grid.getDimensions());

            if (!iterator.canIterate()) {
               SCOPED_TRACE("Skip iteration");
               EXPECT_EQ(hostStore.getCallsView().getSize(), 0) << "Expect, that we can't iterate, when grid is empty";
               return;
            }

            do {
               verifyEntity(grid, iterator, hostStoreView, iterator.isBoundary(grid));
            } while (!iterator.next());
         };

         TNL::Algorithms::staticFor< int, 0, orientationsCount >(verify);
      }
      void verifyInterior(const Grid& grid, const DataStore& store) const {
         auto hostStore = store.template move<TNL::Devices::Host>();
         auto hostStoreView = hostStore.getView();

         constexpr int orientationsCount = Grid::getEntityOrientationsCount(EntityDimension);

         SCOPED_TRACE("Verifying forInterior");
         SCOPED_TRACE("Orientations Count: " + TNL::convertToString(orientationsCount));

         ASSERT_GT(orientationsCount, 0) << "Every entity must have at least one orientation";

         auto verify = [&](const auto orientation) {
            Iterator<orientation> iterator(grid.getDimensions());

            if (!iterator.canIterate()) {
               SCOPED_TRACE("Skip iteration");
               EXPECT_EQ(hostStore.getCallsView().getSize(), 0) << "Expect, that we can't iterate, when grid is empty";
               return;
            }

            do {
               verifyEntity(grid, iterator, hostStoreView, !iterator.isBoundary(grid));
            } while (!iterator.next());
         };

         TNL::Algorithms::staticFor< int, 0, orientationsCount >(verify);
      }
   private:
      template<int Orientation>
      void verifyEntity(const Grid& grid,
                        const Iterator<Orientation>& iterator,
                        typename HostDataStore::View& dataStore,
                        bool expectCall) const {
         static Real precision = 9e-5;

         auto index = iterator.getIndex(grid);
         auto entity = dataStore.getEntity(index);

         SCOPED_TRACE("Entity: " + TNL::convertToString(entity));

         EXPECT_EQ(entity.calls, expectCall ? 1 : 0) << "Expect the index to be called once";
         EXPECT_EQ(entity.index, expectCall ? index : 0) << "Expect the index was correctly set";
         EXPECT_EQ(entity.isBoundary, expectCall ? iterator.isBoundary(grid) : 0) << "Expect the index was correctly set" ;

         Coordinate coordinate = expectCall ? iterator.getCoordinate() : Coordinate(0);
         Coordinate normals = expectCall ? iterator.getNormals() : Coordinate(0);
         Point center = expectCall ? iterator.getCenter(grid) : Point(0);

         EXPECT_EQ(entity.coordinate, coordinate)
                << "Expect the coordinates are the same on the same index. ";
         EXPECT_EQ(entity.normals, normals)
                << "Expect the normals are the same on the same index. ";

         // CUDA calculates floating points differently.
         EXPECT_NEAR(expectCall ? iterator.getMeasure(grid) : 0.0, entity.measure, precision)
               << "Expect the measure was correctly calculated. ";

         for (Index i = 0; i < Grid::getMeshDimension(); i++)
            EXPECT_NEAR(entity.center[i], center[i], precision)
               << "Expect the centers are the same on the same index. " << entity.center << " " << center;
      }
};

template<typename Grid, int EntityDimension>
void testForAllTraverse(Grid& grid,
                        const typename Grid::CoordinatesType& dimensions,
                        const typename Grid::PointType& origin = typename Grid::PointType(0),
                        const typename Grid::PointType& spaceSteps = typename Grid::PointType(1)) {
   SCOPED_TRACE("Grid Dimension: " + TNL::convertToString(Grid::getMeshDimension()));
   SCOPED_TRACE("Entity Dimension: " + TNL::convertToString(EntityDimension));
   SCOPED_TRACE("Dimension: " + TNL::convertToString(dimensions));
   SCOPED_TRACE("Origin:" + TNL::convertToString(origin));
   SCOPED_TRACE("Space steps:" + TNL::convertToString(spaceSteps));

   EXPECT_NO_THROW(grid.setDimensions(dimensions)) << "Verify, that the set of" << dimensions << " doesn't cause assert";
   EXPECT_NO_THROW(grid.setOrigin(origin)) << "Verify, that the set of" << origin << "doesn't cause assert";
   EXPECT_NO_THROW(grid.setSpaceSteps(spaceSteps)) << "Verify, that the set of" << spaceSteps << "doesn't cause assert";

   using Test = GridTraverseTestCase<Grid, EntityDimension>;

   Test test;
   typename Test::DataStore store(grid.getEntitiesCount(EntityDimension));

   test.storeAll(grid, store);
   test.verifyAll(grid, store);
}

template<typename Grid, int EntityDimension>
void testForInteriorTraverse(Grid& grid,
                             const typename Grid::CoordinatesType& dimensions,
                             const typename Grid::PointType& origin = typename Grid::PointType(0),
                             const typename Grid::PointType& spaceSteps = typename Grid::PointType(1)) {
   SCOPED_TRACE("Grid Dimension: " + TNL::convertToString(Grid::getMeshDimension()));
   SCOPED_TRACE("Entity Dimension: " + TNL::convertToString(EntityDimension));
   SCOPED_TRACE("Dimension: " + TNL::convertToString(dimensions));
   SCOPED_TRACE("Origin:" + TNL::convertToString(origin));
   SCOPED_TRACE("Space steps:" + TNL::convertToString(spaceSteps));

   EXPECT_NO_THROW(grid.setDimensions(dimensions)) << "Verify, that the set of" << dimensions << " doesn't cause assert";
   EXPECT_NO_THROW(grid.setOrigin(origin)) << "Verify, that the set of" << origin << "doesn't cause assert";
   EXPECT_NO_THROW(grid.setSpaceSteps(spaceSteps)) << "Verify, that the set of" << spaceSteps << "doesn't cause assert";

   using Test = GridTraverseTestCase<Grid, EntityDimension>;

   Test test;
   typename Test::DataStore store(grid.getEntitiesCount(EntityDimension));

   test.storeInterior(grid, store);
   test.verifyInterior(grid, store);
}

template<typename Grid, int EntityDimension>
void testForBoundaryTraverse(Grid& grid,
                             const typename Grid::CoordinatesType& dimensions,
                             const typename Grid::PointType& origin = typename Grid::PointType(0),
                             const typename Grid::PointType& spaceSteps = typename Grid::PointType(1)) {
   SCOPED_TRACE("Grid Dimension: " + TNL::convertToString(Grid::getMeshDimension()));
   SCOPED_TRACE("Entity Dimension: " + TNL::convertToString(EntityDimension));
   SCOPED_TRACE("Dimension: " + TNL::convertToString(dimensions));
   SCOPED_TRACE("Origin:" + TNL::convertToString(origin));
   SCOPED_TRACE("Space steps:" + TNL::convertToString(spaceSteps));

   EXPECT_NO_THROW(grid.setDimensions(dimensions)) << "Verify, that the set of" << dimensions << " doesn't cause assert";
   EXPECT_NO_THROW(grid.setOrigin(origin)) << "Verify, that the set of" << origin << "doesn't cause assert";
   EXPECT_NO_THROW(grid.setSpaceSteps(spaceSteps)) << "Verify, that the set of" << spaceSteps << "doesn't cause assert";

   using Test = GridTraverseTestCase<Grid, EntityDimension>;

   Test test;
   typename Test::DataStore store(grid.getEntitiesCount(EntityDimension));

   test.storeBoundary(grid, store);
   test.verifyBoundary(grid, store);
}

template<typename Grid, int EntityDimension>
void testBoundaryUnionInteriorEqualAllProperty(Grid& grid,
                                               const typename Grid::CoordinatesType& dimensions,
                                               const typename Grid::PointType& origin = typename Grid::PointType(0),
                                               const typename Grid::PointType& spaceSteps = typename Grid::PointType(1)) {
   SCOPED_TRACE("Grid Dimension: " + TNL::convertToString(Grid::getMeshDimension()));
   SCOPED_TRACE("Entity Dimension: " + TNL::convertToString(EntityDimension));
   SCOPED_TRACE("Dimension: " + TNL::convertToString(dimensions));
   SCOPED_TRACE("Origin:" + TNL::convertToString(origin));
   SCOPED_TRACE("Space steps:" + TNL::convertToString(spaceSteps));

   EXPECT_NO_THROW(grid.setDimensions(dimensions)) << "Verify, that the set of" << dimensions << " doesn't cause assert";
   EXPECT_NO_THROW(grid.setOrigin(origin)) << "Verify, that the set of" << origin << "doesn't cause assert";
   EXPECT_NO_THROW(grid.setSpaceSteps(spaceSteps)) << "Verify, that the set of" << spaceSteps << "doesn't cause assert";

   using Test = GridTraverseTestCase<Grid, EntityDimension>;

   Test test;
   typename Test::DataStore store(grid.getEntitiesCount(EntityDimension));

   test.storeBoundary(grid, store);
   test.storeInterior(grid, store);
   test.verifyAll(grid, store);
}

template<typename Grid, int EntityDimension>
void testAllMinusBoundaryEqualInteriorProperty(Grid& grid,
                                               const typename Grid::CoordinatesType& dimensions,
                                               const typename Grid::PointType& origin = typename Grid::PointType(0),
                                               const typename Grid::PointType& spaceSteps = typename Grid::PointType(1)) {
   SCOPED_TRACE("Grid Dimension: " + TNL::convertToString(Grid::getMeshDimension()));
   SCOPED_TRACE("Entity Dimension: " + TNL::convertToString(EntityDimension));
   SCOPED_TRACE("Dimension: " + TNL::convertToString(dimensions));
   SCOPED_TRACE("Origin:" + TNL::convertToString(origin));
   SCOPED_TRACE("Space steps:" + TNL::convertToString(spaceSteps));

   EXPECT_NO_THROW(grid.setDimensions(dimensions)) << "Verify, that the set of" << dimensions << " doesn't cause assert";
   EXPECT_NO_THROW(grid.setOrigin(origin)) << "Verify, that the set of" << origin << "doesn't cause assert";
   EXPECT_NO_THROW(grid.setSpaceSteps(spaceSteps)) << "Verify, that the set of" << spaceSteps << "doesn't cause assert";

   using Test = GridTraverseTestCase<Grid, EntityDimension>;

   Test test;
   typename Test::DataStore store(grid.getEntitiesCount(EntityDimension));

   test.storeAll(grid, store);
   test.clearBoundary(grid, store);
   test.verifyInterior(grid, store);
}

template<typename Grid, int EntityDimension>
void testAllMinusInteriorEqualBoundaryProperty(Grid& grid,
                                               const typename Grid::CoordinatesType& dimensions,
                                               const typename Grid::PointType& origin = typename Grid::PointType(0),
                                               const typename Grid::PointType& spaceSteps = typename Grid::PointType(1)) {
   SCOPED_TRACE("Grid Dimension: " + TNL::convertToString(Grid::getMeshDimension()));
   SCOPED_TRACE("Entity Dimension: " + TNL::convertToString(EntityDimension));
   SCOPED_TRACE("Dimension: " + TNL::convertToString(dimensions));
   SCOPED_TRACE("Origin:" + TNL::convertToString(origin));
   SCOPED_TRACE("Space steps:" + TNL::convertToString(spaceSteps));

   EXPECT_NO_THROW(grid.setDimensions(dimensions)) << "Verify, that the set of" << dimensions << " doesn't cause assert";
   EXPECT_NO_THROW(grid.setOrigin(origin)) << "Verify, that the set of" << origin << "doesn't cause assert";
   EXPECT_NO_THROW(grid.setSpaceSteps(spaceSteps)) << "Verify, that the set of" << spaceSteps << "doesn't cause assert";

   using Test = GridTraverseTestCase<Grid, EntityDimension>;

   Test test;
   typename Test::DataStore store(grid.getEntitiesCount(EntityDimension));

   test.storeAll(grid, store);
   test.clearInterior(grid, store);
   test.verifyBoundary(grid, store);
}


#endif
