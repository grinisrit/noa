// Copyright (c) 2004-2023 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <noa/3rdparty/tnl-noa/src/TNL/Meshes/DimensionTag.h>

namespace noa::TNL {
namespace Meshes {
namespace DistributedMeshes {

template< typename Mesh, typename Device, int Dimension >
class GlobalIndexStorage
{
public:
   using GlobalIndexArray = typename Mesh::GlobalIndexArray;

   GlobalIndexStorage() = default;
   GlobalIndexStorage( const GlobalIndexStorage& ) = default;
   GlobalIndexStorage( GlobalIndexStorage&& ) noexcept = default;
   GlobalIndexStorage&
   operator=( const GlobalIndexStorage& ) = default;
   GlobalIndexStorage&
   operator=( GlobalIndexStorage&& ) noexcept = default;

   template< typename Mesh_ >
   GlobalIndexStorage&
   operator=( const Mesh_& mesh )
   {
      globalIndices = mesh.template getGlobalIndices< Dimension >();
      return *this;
   }

   bool
   operator==( const GlobalIndexStorage& other ) const
   {
      return globalIndices == other.getGlobalIndices();
   }

   const GlobalIndexArray&
   getGlobalIndices() const
   {
      return globalIndices;
   }

   GlobalIndexArray&
   getGlobalIndices()
   {
      return globalIndices;
   }

protected:
   GlobalIndexArray globalIndices;
};

template< typename Mesh, typename Device = typename Mesh::DeviceType, typename DimensionTag = Meshes::DimensionTag< 0 > >
class GlobalIndexStorageFamily
// Doxygen complains about recursive classes
//! \cond
: public GlobalIndexStorage< Mesh, Device, DimensionTag::value >,
  public GlobalIndexStorageFamily< Mesh, Device, typename DimensionTag::Increment >
//! \endcond
{
public:
   GlobalIndexStorageFamily() = default;
   GlobalIndexStorageFamily( const GlobalIndexStorageFamily& ) = default;
   GlobalIndexStorageFamily( GlobalIndexStorageFamily&& ) noexcept = default;
   GlobalIndexStorageFamily&
   operator=( const GlobalIndexStorageFamily& ) = default;
   GlobalIndexStorageFamily&
   operator=( GlobalIndexStorageFamily&& ) noexcept = default;

   template< typename Mesh_ >
   GlobalIndexStorageFamily&
   operator=( const Mesh_& mesh )
   {
      GlobalIndexStorage< Mesh, Device, DimensionTag::value >::operator=( mesh );
      GlobalIndexStorageFamily< Mesh, Device, typename DimensionTag::Increment >::operator=( mesh );
      return *this;
   }

   bool
   operator==( const GlobalIndexStorageFamily& other ) const
   {
      return GlobalIndexStorage< Mesh, Device, DimensionTag::value >::operator==( other )
          && GlobalIndexStorageFamily< Mesh, Device, typename DimensionTag::Increment >::operator==( other );
   }
};

template< typename Mesh, typename Device >
class GlobalIndexStorageFamily< Mesh, Device, Meshes::DimensionTag< Mesh::getMeshDimension() + 1 > >
{
public:
   GlobalIndexStorageFamily() = default;
   GlobalIndexStorageFamily( const GlobalIndexStorageFamily& ) = default;
   GlobalIndexStorageFamily( GlobalIndexStorageFamily&& ) noexcept = default;
   GlobalIndexStorageFamily&
   operator=( const GlobalIndexStorageFamily& ) = default;
   GlobalIndexStorageFamily&
   operator=( GlobalIndexStorageFamily&& ) noexcept = default;

   template< typename Mesh_ >
   GlobalIndexStorageFamily&
   operator=( const Mesh_& )
   {
      return *this;
   }

   bool
   operator==( const GlobalIndexStorageFamily& ) const
   {
      return true;
   }
};

}  // namespace DistributedMeshes
}  // namespace Meshes
}  // namespace noa::TNL
