#pragma once

#ifdef HAVE_GTEST
#include <gtest/gtest.h>
#include <TNL/Containers/Array.h>

#include "../CoordinateIterator.h"

template<typename Device>
class GridAccessorsTestCaseInterface {
   public:
      template<typename Grid>
      void verifyDimensionGetters(const Grid& grid, const typename Grid::CoordinatesType& coordinates) const { FAIL() << "Expect to be specialized"; }

      template<typename Grid>
      void verifyEntitiesCountGetters(const Grid& grid, const TNL::Containers::StaticVector<Grid::getMeshDimension() + 1, typename Grid::IndexType>& entitiesCount) const { FAIL() << "Expect to be specialized"; }

      template<typename Grid>
      void verifyOriginGetters(const Grid& grid, const typename Grid::Point& coordinates) const { FAIL() << "Expect to be specialized"; }

      template<typename Grid>
      void verifySpaceStepsGetter(const Grid& grid, const typename Grid::Point& spaceSteps) const { FAIL() << "Expect to be specialized"; }


      template<typename Grid>
      void verifyDimensionByCoordinateGetter(const Grid& grid, const typename Grid::CoordinatesType& dimensions) const { FAIL() << "Expect to be specialized"; }


      template<typename Grid>
      void verifyEntitiesCountByContainerGetter(const Grid& grid, const TNL::Containers::StaticVector<Grid::getMeshDimension() + 1, typename Grid::IndexType>& entitiesCounts) const { FAIL() << "Expect to be specialized"; }

      template<typename Grid>
      void verifyEntitiesCountByIndexGetter(const Grid& grid, const TNL::Containers::StaticVector<Grid::getMeshDimension() + 1, typename Grid::IndexType>& entitiesCounts) const { FAIL() << "Expect to be specialized"; }

      template<typename Grid>
      void verifyEntitiesCountByIndiciesGetter(const Grid& grid, const TNL::Containers::StaticVector<Grid::getMeshDimension() + 1, typename Grid::IndexType>& entitiesCounts) const { FAIL() << "Expect to be specialized"; }


      template <typename Grid>
      void verifySpaceStepsValues(const Grid& grid, const int spaceStepsSize, const typename Grid::PointType& spaceSteps) {
         using Real = typename Grid::RealType;
         using CoordinatesType = typename Grid::CoordinatesType;

         CoordinatesType start, end;

         for (int i = 0; i < start.getSize(); i++) {
            start[i] = -(spaceStepsSize >> 1);
            end[i] = (spaceStepsSize >> 1);
         }

         CoordinateIterator<int, Grid::getMeshDimension()> iterator(start, end);

         do {
            auto coordinate = iterator.getCoordinate();
            Real product = 1.;

            for (typename Grid::IndexType i = 0; i < start.getSize(); i++) product *= pow(spaceSteps[i], coordinate[i]);

            if (std::is_same<Real, float>::value) {
               EXPECT_FLOAT_EQ(grid.getSpaceStepsProducts(coordinate), product) << "Expect the step size products are the same";
               continue;
            }

             if (std::is_same<Real, double>::value) {
               EXPECT_DOUBLE_EQ(grid.getSpaceStepsProducts(coordinate), product) << "Expect the step size products are the same";
               continue;
            }

            FAIL() << "Unknown type as real was provided for comparison: " << TNL::getType<Real>();
         } while (!iterator.next());
      }
};


template<typename Device>
class GridAccessorsTestCase: public GridAccessorsTestCaseInterface<Device> {};

template<>
class GridAccessorsTestCase<TNL::Devices::Host>: public GridAccessorsTestCaseInterface<TNL::Devices::Host> {
   public:
      template<typename Grid>
      void verifyDimensionGetters(const Grid& grid, const typename Grid::CoordinatesType& dimensions) const {
         this->verifyDimensionByCoordinateGetter<Grid>(grid, dimensions);
      }

      template<typename Grid>
      void verifyEntitiesCountGetters(const Grid& grid, const TNL::Containers::StaticVector<Grid::getMeshDimension() + 1, typename Grid::IndexType>& entitiesCounts) const {
         this->verifyEntitiesCountByContainerGetter<Grid>(grid, entitiesCounts);
         this->verifyEntitiesCountByIndexGetter<Grid>(grid, entitiesCounts);
         this->verifyEntitiesCountByIndiciesGetter<Grid>(grid, entitiesCounts);
      }

      template<typename Grid>
      void verifyOriginGetter(const Grid& grid, const typename Grid::PointType& coordinates) const {
         auto result = grid.getOrigin();

         EXPECT_EQ(coordinates, result) << "Verify, that the origin was correctly set";
      }

      template<typename Grid>
      void verifySpaceStepsGetter(const Grid& grid, const typename Grid::PointType& spaceSteps) const {
         auto result = grid.getSpaceSteps();

         EXPECT_EQ(spaceSteps, result) << "Verify, that space steps were correctly set";
      }

      template<typename Grid>
      void verifyDimensionByCoordinateGetter(const Grid& grid, const typename Grid::CoordinatesType& dimensions) const {
         auto result = grid.getDimensions();

         EXPECT_EQ(dimensions, result) << "Verify, that dimension container accessor returns valid dimension";
      }

      template<typename Grid>
      void verifyEntitiesCountByContainerGetter(const Grid& grid, const TNL::Containers::StaticVector<Grid::getMeshDimension() + 1, typename Grid::IndexType>& entitiesCounts) const {
         auto result = grid.getEntitiesCounts();

         EXPECT_EQ(entitiesCounts, result) << "Verify, that returns expected entities counts";
      }

      template<typename Grid>
      void verifyEntitiesCountByIndexGetter(const Grid& grid, const TNL::Containers::StaticVector<Grid::getMeshDimension() + 1, typename Grid::IndexType>& entitiesCounts) const {
         for (typename Grid::IndexType i = 0; i < entitiesCounts.getSize(); i++)
            EXPECT_EQ(grid.getEntitiesCount(i), entitiesCounts[i]) << "Verify, that index access is correct";
      }

      template<typename Grid>
      void verifyEntitiesCountByIndiciesGetter(const Grid& grid, const TNL::Containers::StaticVector<Grid::getMeshDimension() + 1, typename Grid::IndexType>& entitiesCounts) const {
         for (typename Grid::IndexType i = 0; i < entitiesCounts.getSize(); i++) {
            auto repeated = grid.getEntitiesCounts(i, i, i, i, i, i, i, i, i, i);

            EXPECT_EQ(repeated.getSize(), 10) << "Verify, that all dimension indices are returned";

            for (typename Grid::IndexType j = 0; j < repeated.getSize(); j++)
               EXPECT_EQ(repeated[j], entitiesCounts[i]) << "Verify, that it is possible to request the same dimension multiple times";
         }
      }
};
#ifdef __CUDACC__

template<>
class GridAccessorsTestCase<TNL::Devices::Cuda>: public GridAccessorsTestCaseInterface<TNL::Devices::Cuda> {
   public:
      template<typename Grid>
      void verifyDimensionGetters(const Grid& grid, const typename Grid::CoordinatesType& dimensions) const {
         this->verifyDimensionByCoordinateGetter<Grid>(grid, dimensions);
      }

      template<typename Grid>
      void verifyEntitiesCountGetters(const Grid& grid, const TNL::Containers::StaticVector<Grid::getMeshDimension() + 1, typename Grid::IndexType>& entitiesCount) const {
         this->verifyEntitiesCountByContainerGetter<Grid>(grid, entitiesCount);
         this->verifyEntitiesCountByIndexGetter<Grid>(grid, entitiesCount);
         this->verifyEntitiesCountByIndiciesGetter<Grid>(grid, entitiesCount);
      }

      template<typename Grid>
      void verifyOriginGetter(const Grid& grid, const typename Grid::PointType& coordinates) const {
         auto gridDimension = grid.getMeshDimension();

         auto update = [=] __device__ (const int index, typename Grid::RealType& reference) mutable {
            reference = grid.getOrigin()[index % gridDimension];
         };

         auto verify = [=] __cuda_callable__ (const int index, const typename Grid::RealType& reference) mutable {
            EXPECT_EQ(reference, coordinates[index % gridDimension]);
         };

         this->executeFromDevice<typename Grid::RealType>(update, verify);
      }

      template<typename Grid>
      void verifySpaceStepsGetter(const Grid& grid, const typename Grid::PointType& spaceSteps) const {
         auto gridDimension = grid.getMeshDimension();

         auto update = [=] __device__ (const int index, typename Grid::RealType& reference) mutable {
            reference = grid.getSpaceSteps()[index % gridDimension];
         };

         auto verify = [=] __cuda_callable__ (const int index, const typename Grid::RealType& reference) mutable {
            EXPECT_EQ(reference, spaceSteps[index % gridDimension]);
         };

         this->executeFromDevice<typename Grid::RealType>(update, verify);
      }

      template<typename ContainerElementType, typename Updater, typename Verifier>
      void executeFromDevice(Updater&& updater, Verifier&& verifier) const {
         TNL::Containers::Array<ContainerElementType, TNL::Devices::Cuda> container(100 * 100);

         container.forAllElements(updater);

         TNL::Containers::Array<ContainerElementType, TNL::Devices::Host> result(container);

         result.forAllElements(verifier);
      }


      template<typename Grid>
      void verifyDimensionByCoordinateGetter(const Grid& grid, const typename Grid::CoordinatesType& dimensions) const {
         auto gridDimension = grid.getMeshDimension();

         auto update = [=] __device__ (const int index, typename Grid::IndexType& reference) mutable {
            reference = grid.getDimensions()[index % gridDimension];
         };

         auto verify = [=] __cuda_callable__ (const int index, const typename Grid::IndexType& reference) mutable {
            EXPECT_EQ(reference, dimensions[index % gridDimension]);
         };

         this->executeFromDevice<typename Grid::IndexType>(update, verify);
      }


      template<typename Grid>
      void verifyEntitiesCountByContainerGetter(const Grid& grid, const TNL::Containers::StaticVector<Grid::getMeshDimension() + 1, typename Grid::IndexType>& entitiesCounts) const {
         auto size = entitiesCounts.getSize();

         auto update = [=] __device__ (const int index, typename Grid::IndexType& reference) mutable {
            reference = grid.getEntitiesCounts()[index % size];
         };

         auto verify = [=] __cuda_callable__ (const int index, const typename Grid::IndexType& reference) mutable {
            EXPECT_EQ(reference, entitiesCounts[index % size]);
         };

         this -> executeFromDevice<typename Grid::IndexType>(update, verify);
      }

      template<typename Grid>
      void verifyEntitiesCountByIndexGetter(const Grid& grid, const TNL::Containers::StaticVector<Grid::getMeshDimension() + 1, typename Grid::IndexType>& entitiesCounts) const {
         auto size = entitiesCounts.getSize();

         auto update = [=] __device__ (const int index, typename Grid::IndexType& reference) mutable {
            reference = grid.getEntitiesCount(index % size);
         };

         auto verify = [=] __cuda_callable__ (const int index, const typename Grid::IndexType& reference) mutable {
            EXPECT_EQ(reference, entitiesCounts[index % size]);
         };

         this -> executeFromDevice<typename Grid::IndexType>(update, verify);
      }

      template<typename Grid>
      void verifyEntitiesCountByIndiciesGetter(const Grid& grid, const TNL::Containers::StaticVector<Grid::getMeshDimension() + 1, typename Grid::IndexType>& entitiesCounts) const {
          auto size = entitiesCounts.getSize();

         auto update = [=] __device__ (const int index, typename Grid::IndexType& reference) mutable {
            reference = grid.getEntitiesCounts(index % size)[0];
         };

         auto verify = [=] __cuda_callable__ (const int index, const typename Grid::IndexType& reference) mutable {
            EXPECT_EQ(reference, entitiesCounts[index % size]);
         };

         this -> executeFromDevice<typename Grid::IndexType>(update, verify);
      }
};

#endif

template<typename... Parameters>
std::string makeString(Parameters... parameters) {
   std::ostringstream s;

   for (const auto x: { parameters... })
      s << x << ", ";

   return s.str();
}

template<typename Grid, typename... T>
void testDimensionSetByIndex(Grid& grid, T... dimensions) {
   auto paramString = makeString(dimensions...);

   EXPECT_NO_THROW(grid.setDimensions(dimensions...)) << "Verify, that the set of" << paramString << " doesn't cause assert";

   SCOPED_TRACE("Test dimension set by index");
   SCOPED_TRACE("Dimension: " + TNL::convertToString(typename Grid::CoordinatesType(dimensions...)));
   SCOPED_TRACE("Grid Dimension: " + TNL::convertToString(grid.getMeshDimension()));

   GridAccessorsTestCase<typename Grid::DeviceType> support;

   support.template verifyDimensionGetters<Grid>(grid, typename Grid::CoordinatesType(dimensions...));
}

template<typename Grid, typename... T>
void testDimensionSetByCoordinate(Grid& grid, const typename Grid::CoordinatesType& dimensions) {
   EXPECT_NO_THROW(grid.setDimensions(dimensions)) << "Verify, that the set of" << dimensions << " doesn't cause assert";

   SCOPED_TRACE("Test dimension set by coordinate");
   SCOPED_TRACE("Dimension: " + TNL::convertToString(dimensions));
   SCOPED_TRACE("Grid Dimension: " + TNL::convertToString(grid.getMeshDimension()));

   GridAccessorsTestCase<typename Grid::DeviceType> support;

   support.template verifyDimensionGetters<Grid>(grid, dimensions);
}

template<typename Grid>
void testEntitiesCounts(Grid& grid,
                        const typename Grid::CoordinatesType& dimensions,
                        const TNL::Containers::StaticVector<Grid::getMeshDimension() + 1, typename Grid::IndexType>& entitiesCounts) {
   EXPECT_NO_THROW(grid.setDimensions(dimensions)) << "Verify, that the set of" << dimensions << " doesn't cause assert";

   SCOPED_TRACE("Test entities count");
   SCOPED_TRACE("Dimension: " + TNL::convertToString(dimensions));
   SCOPED_TRACE("Grid Entities Counts: " + TNL::convertToString(grid.getEntitiesCounts()));
   SCOPED_TRACE("Expected Entities Counts:: " + TNL::convertToString(entitiesCounts));

   GridAccessorsTestCase<typename Grid::DeviceType> support;

   support.template verifyEntitiesCountGetters<Grid>(grid, entitiesCounts);
}

template<typename Grid,
         typename... T,
         std::enable_if_t<TNL::Meshes::Templates::conjunction_v<std::is_convertible<typename Grid::RealType, T>...>, bool> = true>
void testOriginSetByIndex(Grid& grid, T... coordinates) {
   auto paramString = makeString(coordinates...);

   EXPECT_NO_THROW(grid.setOrigin(coordinates...)) << "Verify, that the set of" << paramString << " doesn't cause assert";

   SCOPED_TRACE("Test origin set by coordinate");
   SCOPED_TRACE("Coordinates: " + TNL::convertToString(typename Grid::CoordinatesType(coordinates...)));
   SCOPED_TRACE("Grid origin: " + TNL::convertToString(grid.getOrigin()));

   GridAccessorsTestCase<typename Grid::DeviceType> support;

   support.template verifyOriginGetter<Grid>(grid, typename Grid::PointType(coordinates...));
}

template<typename Grid>
void testOriginSetByCoordinate(Grid& grid, const typename Grid::PointType& coordinates) {
   EXPECT_NO_THROW(grid.setOrigin(coordinates)) << "Verify, that the set of " << coordinates << " doesn't cause assert";

   SCOPED_TRACE("Test origin set by index");
   SCOPED_TRACE("Coordinates: " + TNL::convertToString(coordinates));
   SCOPED_TRACE("Grid origin: " + TNL::convertToString(grid.getOrigin()));

   GridAccessorsTestCase<typename Grid::DeviceType> support;

   support.template verifyOriginGetter<Grid>(grid, coordinates);
}

template<typename Grid>
void testSpaceStepsSetByCoordinate(Grid& grid, const int spaceStepsSize, const typename Grid::PointType& spaceSteps) {
   EXPECT_NO_THROW(grid.setSpaceSteps(spaceSteps)) << "Verify, that the set of " << spaceSteps << " doesn't cause assert";

   SCOPED_TRACE("Test space steps set by coordinate");
   SCOPED_TRACE("Space steps: " + TNL::convertToString(spaceSteps));
   SCOPED_TRACE("Grid space steps: " + TNL::convertToString(grid.getSpaceSteps()));

   GridAccessorsTestCase<typename Grid::DeviceType> support;

   support.template verifySpaceStepsGetter<Grid>(grid, spaceSteps);
   support.template verifySpaceStepsValues<Grid>(grid, spaceStepsSize, spaceSteps);
}

template<typename Grid,
         typename... T,
         std::enable_if_t<TNL::Meshes::Templates::conjunction_v<std::is_convertible<typename Grid::RealType, T>...>, bool> = true>
void testSpaceStepsSetByIndex(Grid& grid, const int spaceStepsSize, T... spaceSteps) {
   typename Grid::PointType spaceStepsContainer(spaceSteps...);

   EXPECT_NO_THROW(grid.setSpaceSteps(spaceSteps...)) << "Verify, that the set of " << spaceStepsContainer << " doesn't cause assert";

   SCOPED_TRACE("Test space steps set by index");
   SCOPED_TRACE("Space steps: " + TNL::convertToString(spaceStepsContainer));
   SCOPED_TRACE("Grid space steps: " + TNL::convertToString(grid.getSpaceSteps()));

   GridAccessorsTestCase<typename Grid::DeviceType> support;

   support.template verifySpaceStepsGetter<Grid>(grid, spaceStepsContainer);
   support.template verifySpaceStepsValues<Grid>(grid, spaceStepsSize, spaceStepsContainer);
}

template<typename Grid>
void testSpaceStepsPowerValues(Grid& grid, const int spaceStepsSize, const typename Grid::PointType& spaceSteps) {
   if (spaceStepsSize <= 0)
      GTEST_SKIP() << "Negative space steps sizes are not supported";

   EXPECT_NO_THROW(grid.setSpaceSteps(spaceSteps)) << "Verify, that the set of " << spaceSteps << " doesn't cause assert";

   SCOPED_TRACE("Test space steps set by index");
   SCOPED_TRACE("Space steps: " + TNL::convertToString(spaceSteps));
   SCOPED_TRACE("Grid space steps: " + TNL::convertToString(grid.getSpaceSteps()));

   GridAccessorsTestCase<typename Grid::DeviceType> support;

   support.template verifySpaceStepsValues<Grid>(grid, spaceStepsSize, spaceSteps);
}

#endif
