// Copyright (c) 2004-2023 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#ifdef HAVE_CGAL

   #include <noa/3rdparty/tnl-noa/src/TNL/Meshes/Geometry/getEntityCenter.h>

   // Documentation:
   // - https://doc.cgal.org/latest/Spatial_searching/index.html
   // - https://doc.cgal.org/latest/Spatial_searching/group__PkgSpatialSearchingDRef.html
   // - https://doc.cgal.org/latest/Spatial_searching/classCGAL_1_1Kd__tree.html
   #include <CGAL/Search_traits.h>
   #include <CGAL/Search_traits_adapter.h>
   #include <CGAL/Kd_tree.h>

namespace noa::TNL {
namespace Meshes {

struct KdTreeOrdering
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

      const GlobalIndexType numberOfEntities = mesh.template getEntitiesCount< MeshEntity >();

      // allocate permutation vectors
      perm.setSize( numberOfEntities );
      iperm.setSize( numberOfEntities );

      // wrapper for CGAL
      struct Construct_coord_iterator
      {
         using result_type = const RealType*;
         result_type
         operator()( const PointType& p ) const
         {
            return static_cast< result_type >( &p[ 0 ] );
         }
         result_type
         operator()( const PointType& p, int ) const
         {
            return static_cast< result_type >( &p[ 0 ] + p.getSize() );
         }
      };

      // CGAL types
      using DimTag = CGAL::Dimension_tag< PointType::getSize() >;
      using Point_and_index = std::tuple< PointType, GlobalIndexType >;
      using Traits_base = CGAL::Search_traits< RealType, PointType, const RealType*, Construct_coord_iterator, DimTag >;
      using TreeTraits =
         CGAL::Search_traits_adapter< Point_and_index, CGAL::Nth_of_tuple_property_map< 0, Point_and_index >, Traits_base >;
      // Note: splitters affect the quality of the ordering
      // available splitters are: https://doc.cgal.org/latest/Spatial_searching/classSplitter.html
      using Splitter = CGAL::Sliding_fair< TreeTraits >;
      using Tree = CGAL::Kd_tree< TreeTraits, Splitter >;

      // build a k-d tree
      Tree tree;
      for( GlobalIndexType i = 0; i < numberOfEntities; i++ ) {
         const auto& entity = mesh.template getEntity< MeshEntity >( i );
         const auto center = getEntityCenter( mesh, entity );
         tree.insert( std::make_tuple( center, i ) );
      }
      tree.build();

      GlobalIndexType permIndex = 0;

      // in-order traversal of the k-d tree
      for( auto iter : tree ) {
         const GlobalIndexType i = std::get< 1 >( iter );
         perm[ permIndex ] = i;
         iperm[ i ] = permIndex;
         permIndex++;
      }
   }
};

}  // namespace Meshes
}  // namespace noa::TNL

#endif  // HAVE_CGAL
