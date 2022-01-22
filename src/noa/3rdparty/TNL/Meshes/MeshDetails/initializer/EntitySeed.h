// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

/***
 * Authors:
 * Oberhuber Tomas, tomas.oberhuber@fjfi.cvut.cz
 * Zabka Vitezslav, zabkav@gmail.com
 */

#pragma once

#include <algorithm>

#include <noa/3rdparty/TNL/Meshes/MeshDetails/traits/MeshTraits.h>
#include <noa/3rdparty/TNL/Meshes/Topologies/Polyhedron.h>

namespace noaTNL {
namespace Meshes {

template< typename EntitySeed >
struct EntitySeedHash;
template< typename EntitySeed >
struct EntitySeedEq;

template< typename MeshConfig,
          typename EntityTopology >
class EntitySeed< MeshConfig, EntityTopology, false >
{
   using MeshTraitsType = MeshTraits< MeshConfig >;
   using SubvertexTraits = typename MeshTraitsType::template SubentityTraits< EntityTopology, 0 >;

   public:
      using GlobalIndexType = typename MeshTraitsType::GlobalIndexType;
      using LocalIndexType  = typename MeshTraitsType::LocalIndexType;
      using IdArrayType     = Containers::StaticArray< SubvertexTraits::count, GlobalIndexType >;
      using HashType        = EntitySeedHash< EntitySeed >;
      using KeyEqual        = EntitySeedEq< EntitySeed >;

      //this function is here only for compatibility with MeshReader
      void setCornersCount( const LocalIndexType& cornersCount ) {}

      static constexpr LocalIndexType getCornersCount()
      {
         return SubvertexTraits::count;
      }

      void setCornerId( const LocalIndexType& cornerIndex, const GlobalIndexType& pointIndex )
      {
         TNL_ASSERT_GE( cornerIndex, 0, "corner index must be non-negative" );
         TNL_ASSERT_LT( cornerIndex, getCornersCount(), "corner index is out of bounds" );
         TNL_ASSERT_GE( pointIndex, 0, "point index must be non-negative" );

         this->cornerIds[ cornerIndex ] = pointIndex;
      }

      IdArrayType& getCornerIds()
      {
         return cornerIds;
      }

      const IdArrayType& getCornerIds() const
      {
         return cornerIds;
      }

   private:
      IdArrayType cornerIds;
};

template< typename MeshConfig >
class EntitySeed< MeshConfig, Topologies::Vertex, false >
{
   using MeshTraitsType = MeshTraits< MeshConfig >;

   public:
      using GlobalIndexType = typename MeshTraitsType::GlobalIndexType;
      using LocalIndexType  = typename MeshTraitsType::LocalIndexType;
      using IdArrayType     = Containers::StaticArray< 1, GlobalIndexType >;
      using HashType        = EntitySeedHash< EntitySeed >;
      using KeyEqual        = EntitySeedEq< EntitySeed >;

      //this function is here only for compatibility with MeshReader
      void setCornersCount( const LocalIndexType& cornersCount ) {}

      static constexpr LocalIndexType getCornersCount()
      {
         return 1;
      }

      void setCornerId( const LocalIndexType& cornerIndex, const GlobalIndexType& pointIndex )
      {
         TNL_ASSERT_EQ( cornerIndex, 0, "corner index must be 0" );
         TNL_ASSERT_GE( pointIndex, 0, "point index must be non-negative" );

         this->cornerIds[ cornerIndex ] = pointIndex;
      }

      IdArrayType& getCornerIds()
      {
         return cornerIds;
      }

      const IdArrayType& getCornerIds() const
      {
         return cornerIds;
      }

   private:
      IdArrayType cornerIds;
};

template< typename MeshConfig,
          typename EntityTopology >
class EntitySeed< MeshConfig, EntityTopology, true >
{
   using MeshTraitsType = MeshTraits< MeshConfig >;

public:
   using GlobalIndexType = typename MeshTraitsType::GlobalIndexType;
   using LocalIndexType  = typename MeshTraitsType::LocalIndexType;
   using IdArrayType     = Containers::Array< GlobalIndexType, Devices::Host, LocalIndexType >;
   using HashType        = EntitySeedHash< EntitySeed >;
   using KeyEqual        = EntitySeedEq< EntitySeed >;

   // this constructor definition is here to avoid default constructor being implicitly declared as __host__ __device__, that causes warning:
   // warning #20011-D: calling a __host__ function("std::allocator<int> ::allocator") 
   // from a __host__ __device__ function("noaTNL::Meshes::EntitySeed< ::MeshTest::TestTwoWedgesMeshConfig,
   // ::noaTNL::Meshes::Topologies::Polygon> ::EntitySeed [subobject]") is not allowed
   EntitySeed()
   {
   }

   void setCornersCount( const LocalIndexType& cornersCount )
   {
      if( std::is_same< EntityTopology, Topologies::Polygon >::value )
         TNL_ASSERT_GE( cornersCount, 3, "polygons must have at least 3 corners" );
      /*else if( std::is_same< EntityTopology, Topologies::Polyhedron >::value )
         TNL_ASSERT_GE( cornersCount, 2, "polyhedron must have at least 2 faces" );*/

      this->cornerIds.setSize( cornersCount );
   }

   LocalIndexType getCornersCount() const
   {
      return this->cornerIds.getSize();
   }

   void setCornerId( const LocalIndexType& cornerIndex, const GlobalIndexType& pointIndex )
   {
      TNL_ASSERT_GE( cornerIndex, 0, "corner index must be non-negative" );
      TNL_ASSERT_LT( cornerIndex, getCornersCount(), "corner index is out of bounds" );
      TNL_ASSERT_GE( pointIndex, 0, "point index must be non-negative" );

      this->cornerIds[ cornerIndex ] = pointIndex;
   }

   IdArrayType& getCornerIds()
   {
      return cornerIds;
   }

   const IdArrayType& getCornerIds() const
   {
      return cornerIds;
   }

private:
   IdArrayType cornerIds;
};

template< typename MeshConfig, typename EntityTopology >
std::ostream& operator<<( std::ostream& str, const EntitySeed< MeshConfig, EntityTopology >& e )
{
   str << e.getCornerIds();
   return str;
};

template< typename EntitySeed >
struct EntitySeedHash
{
   std::size_t operator()( const EntitySeed& seed ) const
   {
      using LocalIndexType = typename EntitySeed::LocalIndexType;
      using GlobalIndexType = typename EntitySeed::GlobalIndexType;

      // Note that we must use an associative function to combine the hashes,
      // because we *want* to ignore the order of the corner IDs.
      std::size_t hash = 0;
      for( LocalIndexType i = 0; i < seed.getCornersCount(); i++ )
//         hash ^= std::hash< GlobalIndexType >{}( seed.getCornerIds()[ i ] );
         hash += std::hash< GlobalIndexType >{}( seed.getCornerIds()[ i ] );
      return hash;
   }
};

template< typename EntitySeed >
struct EntitySeedEq
{
   bool operator()( const EntitySeed& left, const EntitySeed& right ) const
   {
      using IdArrayType = typename EntitySeed::IdArrayType;

      IdArrayType sortedLeft( left.getCornerIds() );
      IdArrayType sortedRight( right.getCornerIds() );

      //use std::sort for now, because polygon EntitySeeds use noaTNL::Containers::Array for cornersIds, that is missing sort function
      std::sort( sortedLeft.getData(), sortedLeft.getData() + sortedLeft.getSize() );
      std::sort( sortedRight.getData(), sortedRight.getData() + sortedRight.getSize() );
      /*sortedLeft.sort();
      sortedRight.sort();*/
      return sortedLeft == sortedRight;
   }
};

template< typename MeshConfig >
struct EntitySeedEq< EntitySeed< MeshConfig, Topologies::Vertex > >
{
   using Seed = EntitySeed< MeshConfig, Topologies::Vertex >;

   bool operator()( const Seed& left, const Seed& right ) const
   {
      return left.getCornerIds() == right.getCornerIds();
   }
};

} // namespace Meshes
} // namespace noaTNL
