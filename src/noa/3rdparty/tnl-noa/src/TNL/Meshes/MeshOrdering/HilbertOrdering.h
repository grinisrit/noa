// Copyright (c) 2004-2023 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#ifdef HAVE_CGAL

   #include <noa/3rdparty/tnl-noa/src/TNL/Meshes/Geometry/getEntityCenter.h>

   // Documentation:
   // - https://doc.cgal.org/latest/Spatial_sorting/index.html
   #include <CGAL/hilbert_sort.h>

namespace noa::TNL {
namespace Meshes {

struct HilbertOrdering
{
   template< typename MeshEntity, typename Mesh, typename PermutationArray >
   static void
   getPermutations( const Mesh& mesh, PermutationArray& perm, PermutationArray& iperm )
   {
      static_assert( std::is_same< typename Mesh::DeviceType, TNL::Devices::Host >::value, "" );
      static_assert( std::is_same< typename PermutationArray::DeviceType, TNL::Devices::Host >::value, "" );
      using GlobalIndexType = typename Mesh::GlobalIndexType;
      using PointType = typename Mesh::PointType;
      using RealType = typename Mesh::RealType;

      // wrappers for CGAL
      using pair = std::pair< PointType, GlobalIndexType >;
      struct Compute_d
      {
         RealType
         operator()( const pair& p, int d ) const
         {
            return p.first[ d ];
         }
      };
      struct Less_d
      {
         bool
         operator()( const pair& p, const pair& q, int d ) const
         {
            return p.first[ d ] < q.first[ d ];
         }
      };
      struct Point_dim_d
      {
         int
         operator()( const pair& p ) const
         {
            return p.first.getSize();
         }
      };
      struct SortingTraits
      {
         using Point_d [[maybe_unused]]  // FIXME: workaround for https://github.com/llvm/llvm-project/issues/59706
         = pair;
         using Compute_coordinate_d [[maybe_unused]]  // FIXME: workaround for https://github.com/llvm/llvm-project/issues/59706
         = Compute_d;
         using Less_coordinate_d [[maybe_unused]]  // FIXME: workaround for https://github.com/llvm/llvm-project/issues/59706
         = Less_d;
         using Point_dimension_d [[maybe_unused]]  // FIXME: workaround for https://github.com/llvm/llvm-project/issues/59706
         = Point_dim_d;
         Compute_coordinate_d
         compute_coordinate_d_object() const
         {
            return {};
         }
         Less_coordinate_d
         less_coordinate_d_object() const
         {
            return {};
         }
         Point_dimension_d
         point_dimension_d_object() const
         {
            return {};
         }
      };

      // create a vector with entity centers and initial indices
      std::vector< pair > points;
      const GlobalIndexType numberOfEntities = mesh.template getEntitiesCount< MeshEntity >();
      for( GlobalIndexType i = 0; i < numberOfEntities; i++ ) {
         const auto& entity = mesh.template getEntity< MeshEntity >( i );
         const auto center = getEntityCenter( mesh, entity );
         points.push_back( std::make_pair( center, i ) );
      }

      // sort the points
      CGAL::hilbert_sort( points.begin(), points.end(), SortingTraits(), CGAL::Hilbert_sort_middle_policy() );

      // build the permutations
      perm.setSize( numberOfEntities );
      iperm.setSize( numberOfEntities );
      for( GlobalIndexType i = 0; i < numberOfEntities; i++ ) {
         perm[ i ] = points[ i ].second;
         iperm[ points[ i ].second ] = i;
      }
   }
};

}  // namespace Meshes
}  // namespace noa::TNL

#endif  // HAVE_CGAL
