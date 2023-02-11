// Copyright (c) 2004-2023 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <noa/3rdparty/tnl-noa/src/TNL/Cuda/CudaCallable.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Meshes/GridDetails/NormalsGetter.h>

namespace noa::TNL {
namespace Meshes {

template< class, int >
class GridEntity;

template< int GridDimension, int ParentEntityDimension, int NeighbourEntityDimension >
class NeighbourGridEntityGetter
{
public:
   template< class Grid >
   static __cuda_callable__
   inline GridEntity< Grid, NeighbourEntityDimension >
   getEntity( const GridEntity< Grid, ParentEntityDimension >& entity, const typename Grid::CoordinatesType& offset )
   {
      using CoordinatesType = typename Grid::CoordinatesType;

      constexpr int orientationsCount = Templates::combination( NeighbourEntityDimension, GridDimension );

      const CoordinatesType coordinate = entity.getCoordinates() + offset;
      const int orientation = TNL::min( orientationsCount - 1, entity.getOrientation() );
      const CoordinatesType normals =
         orientation == entity.getOrientation() && ParentEntityDimension == NeighbourEntityDimension
            ? entity.getNormals()
            : entity.getMesh().template getNormals< NeighbourEntityDimension >( orientation );

      TNL_ASSERT_GE( coordinate, CoordinatesType( 0 ), "wrong coordinate" );
      TNL_ASSERT_LT( coordinate, entity.getMesh().getDimensions() + normals, "wrong coordinate" );

      return { entity.getMesh(), coordinate, normals, orientation };
   }

   template< class Grid,
             int Orientation,
             std::enable_if_t< Templates::isInLeftClosedRightOpenInterval(
                                  0,
                                  Orientation,
                                  Templates::combination( NeighbourEntityDimension, GridDimension ) ),
                               bool > = true >
   static __cuda_callable__
   inline GridEntity< Grid, NeighbourEntityDimension >
   getEntity( const GridEntity< Grid, ParentEntityDimension >& entity, const typename Grid::CoordinatesType& offset )
   {
      using NormalsGetterType = NormalsGetter< typename Grid::IndexType, NeighbourEntityDimension, GridDimension >;
      using CoordinatesType = typename Grid::CoordinatesType;

      const CoordinatesType coordinate = entity.getCoordinates() + offset;
      const CoordinatesType normals = NormalsGetterType::template getNormals< Orientation >();

      TNL_ASSERT_GE( coordinate, CoordinatesType( 0 ), "wrong coordinate" );
      TNL_ASSERT_LT( coordinate, entity.getMesh().getDimensions() + normals, "wrong coordinate" );

      return { entity.getMesh(), coordinate, normals, Orientation };
   }
};

}  // namespace Meshes
}  // namespace noa::TNL
