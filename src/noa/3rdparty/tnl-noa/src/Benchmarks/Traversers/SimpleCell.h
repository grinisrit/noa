// Implemented by: Tomas Oberhuber

#pragma once

#include <TNL/Devices/Cuda.h>
#include <TNL/Meshes/Grid.h>

namespace TNL {
   namespace Benchmarks {
      namespace Traversers {

template< typename Grid >
class SimpleCell{};

template< typename Real,
          typename Device,
          typename Index >
class SimpleCell< Meshes::Grid< 1, Real, Device, Index > >
{
   public:
      using GridType = Meshes::Grid< 1, Real, Device, Index >;
      using RealType = typename GridType::RealType;
      using DeviceType = typename GridType::DeviceType;
      using IndexType = typename GridType::IndexType;
      using CoordinatesType = typename GridType::CoordinatesType;

      constexpr static int getEntityDimension() { return 1; };

      __cuda_callable__
      SimpleCell( const GridType& grid ) :
      grid( grid ){};

      __cuda_callable__
      const GridType& getMesh() const { return this->grid;};

      __cuda_callable__
      CoordinatesType& getCoordinates() { return this->coordinates; };

      __cuda_callable__
      const CoordinatesType& getCoordinates() const { return this->coordinates; };

      __cuda_callable__
      void refresh() {index = this->grid.getEntityIndex( *this );};

      __cuda_callable__
      const IndexType& getIndex() const { return this->index; };

   protected:
      const GridType& grid;
      CoordinatesType coordinates;
      IndexType index;
};

template< typename Real,
          typename Device,
          typename Index >
class SimpleCell< Meshes::Grid< 2, Real, Device, Index > >
{
   public:
      using GridType = Meshes::Grid< 2, Real, Device, Index >;
      using RealType = typename GridType::RealType;
      using DeviceType = typename GridType::DeviceType;
      using IndexType = typename GridType::IndexType;
      using CoordinatesType = typename GridType::CoordinatesType;

      constexpr static int getEntityDimension() { return 2; };

      __cuda_callable__
      SimpleCell( const GridType& grid ) :
      grid( grid ){};

      __cuda_callable__
      const GridType& getMesh() const { return this->grid;};

      __cuda_callable__
      CoordinatesType& getCoordinates() { return this->coordinates; };

      __cuda_callable__
      const CoordinatesType& getCoordinates() const { return this->coordinates; };

      __cuda_callable__
      void refresh() {index = this->grid.getEntityIndex( *this );};

      __cuda_callable__
      const IndexType& getIndex() const { return this->index; };

   protected:
      const GridType& grid;
      CoordinatesType coordinates;
      IndexType index;

};

template< typename Real,
          typename Device,
          typename Index >
class SimpleCell< Meshes::Grid< 3, Real, Device, Index > >
{
   public:
      using GridType = Meshes::Grid< 3, Real, Device, Index >;
      using RealType = typename GridType::RealType;
      using DeviceType = typename GridType::DeviceType;
      using IndexType = typename GridType::IndexType;
      using CoordinatesType = typename GridType::CoordinatesType;

      constexpr static int getEntityDimension() { return 3; };

      __cuda_callable__
      SimpleCell( const GridType& grid ) :
      grid( grid ){};

      __cuda_callable__
      const GridType& getMesh() const { return this->grid;};

      __cuda_callable__
      CoordinatesType& getCoordinates() { return this->coordinates; };

      __cuda_callable__
      const CoordinatesType& getCoordinates() const { return this->coordinates; };

      __cuda_callable__
      void refresh() { index = this->grid.getEntityIndex( *this ); };

      __cuda_callable__
      const IndexType& getIndex() const { return this->index; };

   protected:
      const GridType& grid;
      CoordinatesType coordinates;
      IndexType index;

};

      } // namespace Traversers
   } // namespace Benchmarks
} // namespace TNL
