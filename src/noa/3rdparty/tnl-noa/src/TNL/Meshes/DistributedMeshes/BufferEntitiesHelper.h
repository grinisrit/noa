// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <noa/3rdparty/tnl-noa/src/TNL/Algorithms/ParallelFor.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Containers/StaticVector.h>

namespace noa::TNL {
namespace Meshes {
namespace DistributedMeshes {

template< typename MeshFunctionType,
          typename PeriodicBoundariesMaskPointer,
          int dim,
          typename RealType = typename MeshFunctionType::MeshType::RealType,
          typename Device = typename MeshFunctionType::MeshType::DeviceType,
          typename Index = typename MeshFunctionType::MeshType::GlobalIndexType >
class BufferEntitiesHelper;

template< typename MeshFunctionType, typename MaskPointer, typename RealType, typename Device, typename Index >
class BufferEntitiesHelper< MeshFunctionType, MaskPointer, 1, RealType, Device, Index >
{
public:
   static void
   BufferEntities( MeshFunctionType& meshFunction,
                   const MaskPointer& maskPointer,
                   RealType* buffer,
                   bool isBoundary,
                   const Containers::StaticVector< 1, Index >& begin,
                   const Containers::StaticVector< 1, Index >& size,
                   bool tobuffer )
   {
      Index beginx = begin.x();
      Index sizex = size.x();

      auto* mesh = &meshFunction.getMeshPointer().template getData< Device >();
      RealType* meshFunctionData = meshFunction.getData().getData();
      const typename MaskPointer::ObjectType* mask( nullptr );
      if( maskPointer )
         mask = &maskPointer.template getData< Device >();
      auto kernel = [ tobuffer, mesh, buffer, isBoundary, meshFunctionData, mask, beginx ] __cuda_callable__( Index j )
      {
         typename MeshFunctionType::MeshType::Cell entity( *mesh );
         entity.getCoordinates().x() = beginx + j;
         entity.refresh();
         if( ! isBoundary || ! mask || ( *mask )[ entity.getIndex() ] ) {
            if( tobuffer )
               buffer[ j ] = meshFunctionData[ entity.getIndex() ];
            else
               meshFunctionData[ entity.getIndex() ] = buffer[ j ];
         }
      };
      Algorithms::ParallelFor< Device >::exec( 0, sizex, kernel );
   };
};

template< typename MeshFunctionType, typename MaskPointer, typename RealType, typename Device, typename Index >
class BufferEntitiesHelper< MeshFunctionType, MaskPointer, 2, RealType, Device, Index >
{
public:
   static void
   BufferEntities( MeshFunctionType& meshFunction,
                   const MaskPointer& maskPointer,
                   RealType* buffer,
                   bool isBoundary,
                   const Containers::StaticVector< 2, Index >& begin,
                   const Containers::StaticVector< 2, Index >& size,
                   bool tobuffer )
   {
      Index beginx = begin.x();
      Index beginy = begin.y();
      Index sizex = size.x();
      Index sizey = size.y();

      auto* mesh = &meshFunction.getMeshPointer().template getData< Device >();
      RealType* meshFunctionData = meshFunction.getData().getData();
      const typename MaskPointer::ObjectType* mask( nullptr );
      if( maskPointer )
         mask = &maskPointer.template getData< Device >();

      auto kernel = [ tobuffer, mask, mesh, buffer, isBoundary, meshFunctionData, beginx, sizex, beginy ] __cuda_callable__(
                       Index i, Index j )
      {
         typename MeshFunctionType::MeshType::Cell entity( *mesh );
         entity.getCoordinates().x() = beginx + i;
         entity.getCoordinates().y() = beginy + j;
         entity.refresh();
         if( ! isBoundary || ! mask || ( *mask )[ entity.getIndex() ] ) {
            if( tobuffer )
               buffer[ j * sizex + i ] = meshFunctionData[ entity.getIndex() ];
            else
               meshFunctionData[ entity.getIndex() ] = buffer[ j * sizex + i ];
         }
      };
      Algorithms::ParallelFor2D< Device >::exec( 0, 0, sizex, sizey, kernel );
   };
};

template< typename MeshFunctionType, typename MaskPointer, typename RealType, typename Device, typename Index >
class BufferEntitiesHelper< MeshFunctionType, MaskPointer, 3, RealType, Device, Index >
{
public:
   static void
   BufferEntities( MeshFunctionType& meshFunction,
                   const MaskPointer& maskPointer,
                   RealType* buffer,
                   bool isBoundary,
                   const Containers::StaticVector< 3, Index >& begin,
                   const Containers::StaticVector< 3, Index >& size,
                   bool tobuffer )
   {
      Index beginx = begin.x();
      Index beginy = begin.y();
      Index beginz = begin.z();
      Index sizex = size.x();
      Index sizey = size.y();
      Index sizez = size.z();

      auto* mesh = &meshFunction.getMeshPointer().template getData< Device >();
      RealType* meshFunctionData = meshFunction.getData().getData();
      const typename MaskPointer::ObjectType* mask( nullptr );
      if( maskPointer )
         mask = &maskPointer.template getData< Device >();
      auto kernel =
         [ tobuffer, mesh, mask, buffer, isBoundary, meshFunctionData, beginx, sizex, beginy, sizey, beginz ] __cuda_callable__(
            Index i, Index j, Index k )
      {
         typename MeshFunctionType::MeshType::Cell entity( *mesh );
         entity.getCoordinates().x() = beginx + i;
         entity.getCoordinates().y() = beginy + j;
         entity.getCoordinates().z() = beginz + k;
         entity.refresh();
         if( ! isBoundary || ! mask || ( *mask )[ entity.getIndex() ] ) {
            if( tobuffer )
               buffer[ k * sizex * sizey + j * sizex + i ] = meshFunctionData[ entity.getIndex() ];
            else
               meshFunctionData[ entity.getIndex() ] = buffer[ k * sizex * sizey + j * sizex + i ];
         }
      };
      Algorithms::ParallelFor3D< Device >::exec( 0, 0, 0, sizex, sizey, sizez, kernel );
   };
};

}  // namespace DistributedMeshes
}  // namespace Meshes
}  // namespace noa::TNL
