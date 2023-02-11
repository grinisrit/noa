
#pragma once

#include <TNL/Containers/Array.h>
#include <TNL/Containers/StaticVector.h>

template <typename Index, typename Real, int GridDimension>
struct EntityPrototype {
  public:
   using Coordinate = TNL::Containers::StaticVector<GridDimension, Index>;
   using Point = TNL::Containers::StaticVector<GridDimension, Real>;

   EntityPrototype(const Coordinate& coordinate,
                   const Coordinate& normals,
                   const Index index,
                   const Index calls,
                   const Index orientation,
                   const bool isBoundary,
                   const Point& center,
                   const Real& measure): coordinate(coordinate), normals(normals), index(index), calls(calls), orientation(orientation), isBoundary(isBoundary), center(center), measure(measure) {}

   const Coordinate coordinate;
   const Coordinate normals;
   const Index index;
   const Index calls;
   const Index orientation;
   const bool isBoundary;
   const Point center;
   const Real measure;

   template<typename EntityIndex, typename EntityReal, int EntityGridDimension>
   friend std::ostream & operator << (std::ostream & os, const EntityPrototype<EntityIndex, EntityReal, EntityGridDimension>& entity);
};

template<typename EntityIndex, typename EntityReal, int EntityGridDimension>
std::ostream & operator << (std::ostream & os, const EntityPrototype<EntityIndex, EntityReal, EntityGridDimension>& entity) {
   os << "Coordinate: " << entity.coordinate << std::endl;
   os << "Normals: " << entity.normals << std::endl;
   os << "Index: " << entity.index << std::endl;
   os << "Calls: " << entity.calls << std::endl;
   os << "Orientation: " << entity.orientation << std::endl;
   os << "Is boundary: " << entity.isBoundary << std::endl;
   os << "Center: " << entity.center << std::endl;
   os << "Measure: " << entity.measure << std::endl;

   return os;
}

template<typename Index, typename Real, typename Device, int GridDimension>
struct EntityDataStore {
   public:
      using Coordinate = TNL::Containers::StaticVector<GridDimension, Index>;
      using Point = TNL::Containers::StaticVector<GridDimension, Real>;

      template<typename Value>
      using Container = TNL::Containers::Array<Value, Device, Index>;

      struct View {
         View(typename Container<Index>::ViewType calls,
              typename Container<Index>::ViewType indices,
              typename Container<Index>::ViewType coordinates,
              typename Container<Index>::ViewType normals,
              typename Container<Index>::ViewType orientations,
              typename Container<Index>::ViewType isBoundary,
              typename Container<Real>::ViewType center,
              typename Container<Real>::ViewType measure): calls(calls), indices(indices), coordinates(coordinates), normals(normals), orientations(orientations), isBoundary(isBoundary), center(center), measure(measure) {}

         template <typename Entity>
         __cuda_callable__ void store(const Entity entity) {
            this -> store(entity, entity.getIndex());
         }

         template <typename Entity>
         __cuda_callable__ void store(const Entity entity, const Index index) {
            calls[index] += 1;
            indices[index] = entity.getIndex();
            isBoundary[index] = entity.isBoundary();
            orientations[index] = entity.getOrientation();
            measure[index] = entity.getMeasure();

            auto coordinates = entity.getCoordinates();
            auto normals = entity.getNormals();
            auto center = entity.getCenter();

            for (Index i = 0; i < GridDimension; i++) {
               Index containerIndex = index * GridDimension + i;

               this->coordinates[containerIndex] = coordinates[i];
               this->normals[containerIndex] = normals[i];
               this->center[containerIndex] = center[i];
            }
         }

         template <typename Entity>
         __cuda_callable__ void clear(const Entity entity) {
            auto index = entity.getIndex();

            clear(index);
         }

         __cuda_callable__ void clear(const Index index) {
            calls[index] = 0;
            indices[index] = 0;
            isBoundary[index] = 0;
            orientations[index] = 0;
            measure[index] = 0;

            for (Index i = 0; i < GridDimension; i++) {
               Index containerIndex = index * GridDimension + i;

               coordinates[containerIndex] = 0;
               normals[containerIndex] = 0;
               center[containerIndex] = 0;
            }
         }

         EntityPrototype<Index, Real, GridDimension> getEntity(const Index index) {
            Coordinate coordinates, normals;
            Point center;

            for (Index i = 0; i < GridDimension; i++) {
               Index containerIndex = index * GridDimension + i;

               coordinates[i] = this -> coordinates[containerIndex];
               normals[i] = this -> normals[containerIndex];
               center[i] = this -> center[containerIndex];
            }

            return { coordinates, normals, indices[index], calls[index], orientations[index], isBoundary[index] > 0, center, measure[index] };
         }

         protected:
            typename Container<Index>::ViewType calls, indices, coordinates, normals, orientations, isBoundary;
            typename Container<Real>::ViewType center, measure;
      };

      EntityDataStore(const Index& entitiesCount): entitiesCount(entitiesCount) {
         calls.resize(entitiesCount);
         indices.resize(entitiesCount);
         isBoundary.resize(entitiesCount);
         orientations.resize(entitiesCount);

         measure.resize(entitiesCount);

         coordinates.resize(GridDimension * entitiesCount);
         normals.resize(GridDimension * entitiesCount);
         center.resize(GridDimension * entitiesCount);

         calls = 0;
         indices = 0;
         isBoundary = 0;
         orientations = 0;
         coordinates = 0;
         normals = 0;
         center = 0;
         measure = 0;
      }

      EntityDataStore(const Index& entitiesCount,
                      const Container<Index>& calls,
                      const Container<Index>& indices,
                      const Container<Index>& coordinates,
                      const Container<Index>& normals,
                      const Container<Index>& orientations,
                      const Container<Index>& isBoundary,
                      const Container<Real>& center,
                      const Container<Real>& measure)
          : entitiesCount(entitiesCount),
            calls(calls),
            indices(indices),
            coordinates(coordinates),
            orientations(orientations),
            normals(normals),
            isBoundary(isBoundary),
            center(center),
            measure(measure) {}

      View getView() {
         return { calls.getView(), indices.getView(), coordinates.getView(), normals.getView(), orientations.getView(), isBoundary.getView(), center.getView(), measure.getView() };
      }

      template<typename NewDevice>
      EntityDataStore<Index, Real, NewDevice, GridDimension> move() const {
         using NewIndexContainer = TNL::Containers::Array<Index, NewDevice, Index>;
         using NewRealContainer = TNL::Containers::Array<Real, NewDevice, Index>;

         EntityDataStore<Index, Real, NewDevice, GridDimension> newContainer(this -> entitiesCount,
                                                                             NewIndexContainer(this -> calls),
                                                                             NewIndexContainer(this -> indices),
                                                                             NewIndexContainer(this -> coordinates),
                                                                             NewIndexContainer(this -> normals),
                                                                             NewIndexContainer(this -> orientations),
                                                                             NewIndexContainer(this -> isBoundary),
                                                                             NewRealContainer(this -> center),
                                                                             NewRealContainer(this -> measure));

         return newContainer;
      }

      typename Container<Index>::ViewType getCallsView() { return calls.getView(); }
   private:
      Index entitiesCount;

      Container<Index> calls, indices, coordinates, orientations, normals, isBoundary;
      Container<Real> center, measure;

};
