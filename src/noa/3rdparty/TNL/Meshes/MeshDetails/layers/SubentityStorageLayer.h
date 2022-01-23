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

#include <noa/3rdparty/TNL/File.h>
#include <noa/3rdparty/TNL/Meshes/DimensionTag.h>
#include <noa/3rdparty/TNL/Meshes/MeshDetails/traits/MeshTraits.h>
#include <noa/3rdparty/TNL/Meshes/MeshDetails/traits/WeakStorageTraits.h>

namespace noa::TNL {
namespace Meshes {

template< typename MeshConfig,
          typename Device,
          typename EntityTopology,
          typename SubdimensionTag,
          bool SubentityStorage = WeakSubentityStorageTrait< MeshConfig, Device, typename MeshTraits< MeshConfig, Device >::template EntityTraits< EntityTopology::dimension >::EntityTopology, SubdimensionTag >::storageEnabled,
          bool IsDynamicTopology = Topologies::IsDynamicTopology< EntityTopology >::value >
class SubentityStorageLayer;

template< typename MeshConfig,
          typename Device,
          typename EntityTopology >
class SubentityStorageLayerFamily
   : public SubentityStorageLayer< MeshConfig, Device, EntityTopology, DimensionTag< 0 > >
{
   using BaseType = SubentityStorageLayer< MeshConfig, Device, EntityTopology, DimensionTag< 0 > >;
   using MeshTraitsType = MeshTraits< MeshConfig, Device >;
   using GlobalIndexType = typename MeshTraitsType::GlobalIndexType;

public:
   // inherit constructors and assignment operators (including templated versions)
   using BaseType::BaseType;
   using BaseType::operator=;

protected:
   template< int Subdimension >
   void
   setSubentitiesCounts( const typename MeshTraitsType::NeighborCountsArray& counts )
   {
      static_assert( EntityTopology::dimension > Subdimension, "Invalid combination of Dimension and Subdimension." );
      BaseType::setSubentitiesCounts( DimensionTag< Subdimension >( ), counts );
   }

   template< int Subdimension >
   void
   setSubentitiesCounts( typename MeshTraitsType::NeighborCountsArray&& counts )
   {
      static_assert( EntityTopology::dimension > Subdimension, "Invalid combination of Dimension and Subdimension." );
      BaseType::setSubentitiesCounts( DimensionTag< Subdimension >( ), std::move( counts ) );
   }

   template< int Subdimension >
   __cuda_callable__
   typename MeshTraitsType::LocalIndexType
   getSubentitiesCount( const GlobalIndexType entityIndex ) const
   {
      static_assert( EntityTopology::dimension > Subdimension, "Invalid combination of Dimension and Subdimension." );
      return BaseType::getSubentitiesCount( DimensionTag< Subdimension >( ), entityIndex );
   }

   template< int Subdimension >
   __cuda_callable__
   typename MeshTraitsType::template SubentityMatrixType< EntityTopology::dimension >&
   getSubentitiesMatrix()
   {
      static_assert( EntityTopology::dimension > Subdimension, "Invalid combination of Dimension and Subdimension." );
      return BaseType::getSubentitiesMatrix( DimensionTag< Subdimension >() );
   }

   template< int Subdimension >
   __cuda_callable__
   const typename MeshTraitsType::template SubentityMatrixType< EntityTopology::dimension >&
   getSubentitiesMatrix() const
   {
      static_assert( EntityTopology::dimension > Subdimension, "Invalid combination of Dimension and Subdimension." );
      return BaseType::getSubentitiesMatrix( DimensionTag< Subdimension >() );
   }
};

/****
 *       Mesh subentity storage layer with specializations
 *
 *  SUBENTITY STORAGE     DYNAMIC TOPOLOGY
 *        TRUE                FALSE
 */
template< typename MeshConfig,
          typename Device,
          typename EntityTopology,
          typename SubdimensionTag >
class SubentityStorageLayer< MeshConfig,
                             Device,
                             EntityTopology,
                             SubdimensionTag,
                             true,
                             false >
   : public SubentityStorageLayer< MeshConfig, Device, EntityTopology, typename SubdimensionTag::Increment >
{
   using BaseType = SubentityStorageLayer< MeshConfig, Device, EntityTopology, typename SubdimensionTag::Increment >;
   using MeshTraitsType      = MeshTraits< MeshConfig, Device >;

protected:
   using GlobalIndexType    = typename MeshTraitsType::GlobalIndexType;
   using LocalIndexType     = typename MeshTraitsType::LocalIndexType;
   using SubentityMatrixType = typename MeshTraitsType::template SubentityMatrixType< EntityTopology::dimension >;

   SubentityStorageLayer() = default;

   explicit SubentityStorageLayer( const SubentityStorageLayer& other ) = default;

   SubentityStorageLayer( SubentityStorageLayer&& other ) = default;

   template< typename Device_ >
   SubentityStorageLayer( const SubentityStorageLayer< MeshConfig, Device_, EntityTopology, SubdimensionTag >& other )
   {
      operator=( other );
   }

   SubentityStorageLayer& operator=( const SubentityStorageLayer& other ) = default;

   SubentityStorageLayer& operator=( SubentityStorageLayer&& other ) = default;

   template< typename Device_ >
   SubentityStorageLayer& operator=( const SubentityStorageLayer< MeshConfig, Device_, EntityTopology, SubdimensionTag >& other )
   {
      BaseType::operator=( other );
      matrix = other.matrix;
      return *this;
   }

   void print( std::ostream& str ) const
   {
      BaseType::print( str );
      str << "Adjacency matrix for subentities with dimension " << SubdimensionTag::value << " of entities with dimension " << EntityTopology::dimension << " is: " << std::endl;
      str << matrix << std::endl;
   }

   bool operator==( const SubentityStorageLayer& layer ) const
   {
      return ( BaseType::operator==( layer ) &&
               matrix == layer.matrix );
   }

protected:
   using BaseType::setSubentitiesCounts;
   void setSubentitiesCounts( SubdimensionTag, const typename MeshTraitsType::NeighborCountsArray& counts )
   {}

   void setSubentitiesCounts( SubdimensionTag, typename MeshTraitsType::NeighborCountsArray&& counts )
   {}

   using BaseType::getSubentitiesCount;
   __cuda_callable__
   LocalIndexType getSubentitiesCount( SubdimensionTag, const GlobalIndexType entityIndex ) const
   {
      using SubentityTraitsType = typename MeshTraitsType::template SubentityTraits< EntityTopology, SubdimensionTag::value >;
      return SubentityTraitsType::count;
   }

   using BaseType::getSubentitiesMatrix;
   __cuda_callable__
   SubentityMatrixType& getSubentitiesMatrix( SubdimensionTag )
   {
      return matrix;
   }

   __cuda_callable__
   const SubentityMatrixType& getSubentitiesMatrix( SubdimensionTag ) const
   {
      return matrix;
   }

private:
   SubentityMatrixType matrix;

   // friend class is needed for templated assignment operators
   template< typename MeshConfig_, typename Device_, typename EntityTopology_, typename SubdimensionTag_, bool Storage_, bool dynamicTopology_ >
   friend class SubentityStorageLayer;
};

/****
 *       Mesh subentity storage layer with specializations
 *
 *  SUBENTITY STORAGE     DYNAMIC TOPOLOGY
 *        TRUE                  TRUE
 */
template< typename MeshConfig,
          typename Device,
          typename EntityTopology,
          typename SubdimensionTag >
class SubentityStorageLayer< MeshConfig,
                             Device,
                             EntityTopology,
                             SubdimensionTag,
                             true,
                             true >
   : public SubentityStorageLayer< MeshConfig, Device, EntityTopology, typename SubdimensionTag::Increment >
{
   using BaseType       = SubentityStorageLayer< MeshConfig, Device, EntityTopology, typename SubdimensionTag::Increment >;
   using MeshTraitsType = MeshTraits< MeshConfig, Device >;

protected:
   using GlobalIndexType     = typename MeshTraitsType::GlobalIndexType;
   using LocalIndexType      = typename MeshTraitsType::LocalIndexType;
   using NeighborCountsArray = typename MeshTraitsType::NeighborCountsArray;
   using SubentityMatrixType = typename MeshTraitsType::template SubentityMatrixType< EntityTopology::dimension >;

   SubentityStorageLayer() = default;

   explicit SubentityStorageLayer( const SubentityStorageLayer& other ) = default;

   SubentityStorageLayer( SubentityStorageLayer&& other ) = default;

   template< typename Device_ >
   SubentityStorageLayer( const SubentityStorageLayer< MeshConfig, Device_, EntityTopology, SubdimensionTag >& other )
   {
      operator=( other );
   }

   SubentityStorageLayer& operator=( const SubentityStorageLayer& other ) = default;

   SubentityStorageLayer& operator=( SubentityStorageLayer&& other ) = default;

   template< typename Device_ >
   SubentityStorageLayer& operator=( const SubentityStorageLayer< MeshConfig, Device_, EntityTopology, SubdimensionTag >& other )
   {
      BaseType::operator=( other );
      subentitiesCounts = other.subentitiesCounts;
      matrix = other.matrix;
      return *this;
   }

   void save( File& file ) const
   {
      BaseType::save( file );
      matrix.save( file );
   }

   void load( File& file )
   {
      BaseType::load( file );
      matrix.load( file );
      matrix.getCompressedRowLengths( subentitiesCounts );
   }

   void print( std::ostream& str ) const
   {
      BaseType::print( str );
      str << "Adjacency matrix for subentities with dimension " << SubdimensionTag::value << " of entities with dimension " << EntityTopology::dimension << " is: " << std::endl;
      str << matrix << std::endl;
   }

   bool operator==( const SubentityStorageLayer& layer ) const
   {
      return ( BaseType::operator==( layer ) &&
               subentitiesCounts == layer.subentitiesCounts &&
               matrix == layer.matrix );
   }

protected:
   using BaseType::setSubentitiesCounts;
   void setSubentitiesCounts( SubdimensionTag, const NeighborCountsArray& counts )
   {
      subentitiesCounts = counts;
   }

   void setSubentitiesCounts( SubdimensionTag, NeighborCountsArray&& counts )
   {
      subentitiesCounts = std::move( counts );
   }

   using BaseType::getSubentitiesCount;
   __cuda_callable__
   LocalIndexType getSubentitiesCount( SubdimensionTag, const GlobalIndexType entityIndex ) const
   {
      return subentitiesCounts[ entityIndex ];
   }

   using BaseType::getSubentitiesMatrix;
   __cuda_callable__
   SubentityMatrixType& getSubentitiesMatrix( SubdimensionTag )
   {
      return matrix;
   }

   __cuda_callable__
   const SubentityMatrixType& getSubentitiesMatrix( SubdimensionTag ) const
   {
      return matrix;
   }

private:
   NeighborCountsArray subentitiesCounts;
   SubentityMatrixType matrix;

   // friend class is needed for templated assignment operators
   template< typename MeshConfig_, typename Device_, typename EntityTopology_, typename SubdimensionTag_, bool Storage_, bool dynamicTopology_ >
   friend class SubentityStorageLayer;
};

/****
 *       Mesh subentity storage layer with specializations
 *
 *  SUBENTITY STORAGE     DYNAMIC TOPOLOGY     TOPOLOGY     Subdimension
 *        TRUE                  TRUE           Polygon           0
 */
template< typename MeshConfig,
          typename Device >
class SubentityStorageLayer< MeshConfig,
                             Device,
                             Topologies::Polygon,
                             DimensionTag< 0 >,
                             true,
                             true >
   : public SubentityStorageLayer< MeshConfig, Device, Topologies::Polygon, typename DimensionTag< 0 >::Increment >
{
   using EntityTopology = Topologies::Polygon;
   using SubdimensionTag = DimensionTag< 0 >;
   using BaseType       = SubentityStorageLayer< MeshConfig, Device, EntityTopology, typename SubdimensionTag::Increment >;
   using MeshTraitsType = MeshTraits< MeshConfig, Device >;

protected:
   using GlobalIndexType     = typename MeshTraitsType::GlobalIndexType;
   using LocalIndexType      = typename MeshTraitsType::LocalIndexType;
   using NeighborCountsArray = typename MeshTraitsType::NeighborCountsArray;
   using SubentityMatrixType = typename MeshTraitsType::template SubentityMatrixType< EntityTopology::dimension >;

   SubentityStorageLayer() = default;

   explicit SubentityStorageLayer( const SubentityStorageLayer& other ) = default;

   SubentityStorageLayer( SubentityStorageLayer&& other ) = default;

   template< typename Device_ >
   SubentityStorageLayer( const SubentityStorageLayer< MeshConfig, Device_, EntityTopology, SubdimensionTag >& other )
   {
      operator=( other );
   }

   SubentityStorageLayer& operator=( const SubentityStorageLayer& other ) = default;

   SubentityStorageLayer& operator=( SubentityStorageLayer&& other ) = default;

   template< typename Device_ >
   SubentityStorageLayer& operator=( const SubentityStorageLayer< MeshConfig, Device_, EntityTopology, SubdimensionTag >& other )
   {
      BaseType::operator=( other );
      subentitiesCounts = other.subentitiesCounts;
      matrix = other.matrix;
      return *this;
   }

   void print( std::ostream& str ) const
   {
      BaseType::print( str );
      str << "Adjacency matrix for subentities with dimension " << SubdimensionTag::value << " of entities with dimension " << EntityTopology::dimension << " is: " << std::endl;
      str << matrix << std::endl;
   }

   bool operator==( const SubentityStorageLayer& layer ) const
   {
      return ( BaseType::operator==( layer ) &&
               subentitiesCounts == layer.subentitiesCounts &&
               matrix == layer.matrix );
   }

protected:
   using BaseType::setSubentitiesCounts;
   void setSubentitiesCounts( SubdimensionTag, const NeighborCountsArray& counts )
   {
      subentitiesCounts = counts;
   }

   void setSubentitiesCounts( SubdimensionTag, NeighborCountsArray&& counts )
   {
      subentitiesCounts = std::move( counts );
   }

   using BaseType::getSubentitiesCount;
   __cuda_callable__
   LocalIndexType getSubentitiesCount( SubdimensionTag, const GlobalIndexType entityIndex ) const
   {
      return subentitiesCounts[ entityIndex ];
   }

   // Subdimension 1 has identical subentitiesCounts as Subdimension 0
   __cuda_callable__
   LocalIndexType getSubentitiesCount( typename SubdimensionTag::Increment, const GlobalIndexType entityIndex ) const
   {
      return subentitiesCounts[ entityIndex ];
   }

   using BaseType::getSubentitiesMatrix;
   __cuda_callable__
   SubentityMatrixType& getSubentitiesMatrix( SubdimensionTag )
   {
      return matrix;
   }

   __cuda_callable__
   const SubentityMatrixType& getSubentitiesMatrix( SubdimensionTag ) const
   {
      return matrix;
   }

private:
   NeighborCountsArray subentitiesCounts;
   SubentityMatrixType matrix;

   // friend class is needed for templated assignment operators
   template< typename MeshConfig_, typename Device_, typename EntityTopology_, typename SubdimensionTag_, bool Storage_, bool dynamicTopology_ >
   friend class SubentityStorageLayer;
};

/****
 *       Mesh subentity storage layer with specializations
 *
 *  SUBENTITY STORAGE     DYNAMIC TOPOLOGY     TOPOLOGY     Subdimension
 *        TRUE                  TRUE           Polygon           1
 */
template< typename MeshConfig,
          typename Device >
class SubentityStorageLayer< MeshConfig,
                             Device,
                             Topologies::Polygon,
                             DimensionTag< 1 >,
                             true,
                             true >
   : public SubentityStorageLayer< MeshConfig, Device, Topologies::Polygon, typename DimensionTag< 1 >::Increment >
{
   using EntityTopology = Topologies::Polygon;
   using SubdimensionTag = DimensionTag< 1 >;
   using BaseType       = SubentityStorageLayer< MeshConfig, Device, EntityTopology, typename SubdimensionTag::Increment >;
   using MeshTraitsType = MeshTraits< MeshConfig, Device >;

protected:
   using GlobalIndexType     = typename MeshTraitsType::GlobalIndexType;
   using LocalIndexType      = typename MeshTraitsType::LocalIndexType;
   using NeighborCountsArray = typename MeshTraitsType::NeighborCountsArray;
   using SubentityMatrixType = typename MeshTraitsType::template SubentityMatrixType< EntityTopology::dimension >;

   SubentityStorageLayer() = default;

   explicit SubentityStorageLayer( const SubentityStorageLayer& other ) = default;

   SubentityStorageLayer( SubentityStorageLayer&& other ) = default;

   template< typename Device_ >
   SubentityStorageLayer( const SubentityStorageLayer< MeshConfig, Device_, EntityTopology, SubdimensionTag >& other )
   {
      operator=( other );
   }

   SubentityStorageLayer& operator=( const SubentityStorageLayer& other ) = default;

   SubentityStorageLayer& operator=( SubentityStorageLayer&& other ) = default;

   template< typename Device_ >
   SubentityStorageLayer& operator=( const SubentityStorageLayer< MeshConfig, Device_, EntityTopology, SubdimensionTag >& other )
   {
      BaseType::operator=( other );
      matrix = other.matrix;
      return *this;
   }

   void print( std::ostream& str ) const
   {
      BaseType::print( str );
      str << "Adjacency matrix for subentities with dimension " << SubdimensionTag::value << " of entities with dimension " << EntityTopology::dimension << " is: " << std::endl;
      str << matrix << std::endl;
   }

   bool operator==( const SubentityStorageLayer& layer ) const
   {
      return ( BaseType::operator==( layer ) &&
               matrix == layer.matrix );
   }

protected:
   using BaseType::setSubentitiesCounts;
   void setSubentitiesCounts( SubdimensionTag, const NeighborCountsArray& counts )
   {}

   void setSubentitiesCounts( SubdimensionTag, NeighborCountsArray&& counts )
   {}

   // getSubentitiesCount for subdimension 1 is defined in the specialization for subdimension 0

   using BaseType::getSubentitiesMatrix;
   __cuda_callable__
   SubentityMatrixType& getSubentitiesMatrix( SubdimensionTag )
   {
      return matrix;
   }

   __cuda_callable__
   const SubentityMatrixType& getSubentitiesMatrix( SubdimensionTag ) const
   {
      return matrix;
   }

private:
   SubentityMatrixType matrix;

   // friend class is needed for templated assignment operators
   template< typename MeshConfig_, typename Device_, typename EntityTopology_, typename SubdimensionTag_, bool Storage_, bool dynamicTopology_ >
   friend class SubentityStorageLayer;
};

/****
 *       Mesh subentity storage layer with specializations
 *
 *  SUBENTITY STORAGE     DYNAMIC TOPOLOGY
 *       FALSE               TRUE/FALSE
 */
template< typename MeshConfig,
          typename Device,
          typename EntityTopology,
          typename SubdimensionTag,
          bool dynamicTopology >
class SubentityStorageLayer< MeshConfig,
                             Device,
                             EntityTopology,
                             SubdimensionTag,
                             false,
                             dynamicTopology >
   : public SubentityStorageLayer< MeshConfig, Device, EntityTopology, typename SubdimensionTag::Increment >
{
   using BaseType = SubentityStorageLayer< MeshConfig, Device, EntityTopology, typename SubdimensionTag::Increment >;
public:
   // inherit constructors and assignment operators (including templated versions)
   using BaseType::BaseType;
   using BaseType::operator=;
};

// termination of recursive inheritance (everything is reduced to EntityStorage == false thanks to the WeakSubentityStorageTrait)
template< typename MeshConfig,
          typename Device,
          typename EntityTopology,
          bool dynamicTopology >
class SubentityStorageLayer< MeshConfig,
                             Device,
                             EntityTopology,
                             DimensionTag< EntityTopology::dimension >,
                             false,
                             dynamicTopology >
{
   using MeshTraitsType = MeshTraits< MeshConfig, Device >;
   using SubdimensionTag = DimensionTag< EntityTopology::dimension >;

protected:
   using GlobalIndexType = typename MeshConfig::GlobalIndexType;

   SubentityStorageLayer() = default;
   explicit SubentityStorageLayer( const SubentityStorageLayer& other ) = default;
   SubentityStorageLayer( SubentityStorageLayer&& other ) = default;
   template< typename Device_ >
   SubentityStorageLayer( const SubentityStorageLayer< MeshConfig, Device_, EntityTopology, SubdimensionTag >& other ) {}
   SubentityStorageLayer& operator=( const SubentityStorageLayer& other ) = default;
   SubentityStorageLayer& operator=( SubentityStorageLayer&& other ) = default;
   template< typename Device_ >
   SubentityStorageLayer& operator=( const SubentityStorageLayer< MeshConfig, Device_, EntityTopology, SubdimensionTag >& other ) { return *this; }

   void print( std::ostream& str ) const {}

   bool operator==( const SubentityStorageLayer& layer ) const
   {
      return true;
   }

   void setSubentitiesCounts( SubdimensionTag, const typename MeshTraitsType::NeighborCountsArray& );
   void setSubentitiesCounts( SubdimensionTag, typename MeshTraitsType::NeighborCountsArray&& );
   void getSubentitiesCount( SubdimensionTag ) {}
   void getSubentitiesMatrix( SubdimensionTag ) {}
};

} // namespace Meshes
} // namespace noa::TNL
