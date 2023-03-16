
// Copyright (c) 2004-2023 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

// Implemented by: Tomáš Oberhuber, Yury Hayeu

#pragma once

#include <noa/3rdparty/tnl-noa/src/TNL/Meshes/Grid.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Meshes/GridDetails/GridEntityGetter.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Meshes/GridDetails/Templates/BooleanOperations.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Meshes/GridDetails/NormalsGetter.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Meshes/GridDetails/Templates/Functions.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Meshes/GridDetails/Templates/ParallelFor.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Meshes/GridDetails/Templates/ForEachOrientation.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Algorithms/staticFor.h>

#ifndef DOXYGEN_ONLY

namespace noa::TNL {
namespace Meshes {

template< int Dimension, typename Real, typename Device, typename Index >
constexpr int
Grid< Dimension, Real, Device, Index >::getMeshDimension()
{
   return Dimension;
}

template< int Dimension, typename Real, typename Device, typename Index >
Grid< Dimension, Real, Device, Index >::Grid()
{
   setDimensions( CoordinatesType( 0 ) );

   this->proportions = 0;
   this->spaceSteps = 0;
   this->origin = 0;
   fillNormals();
   fillEntitiesCount();
}

template< int Dimension, typename Real, typename Device, typename Index >
template< typename... Dimensions,
          std::enable_if_t< Templates::conjunction_v< std::is_convertible< Index, Dimensions >... >, bool >,
          std::enable_if_t< sizeof...( Dimensions ) == Dimension, bool > >
Grid< Dimension, Real, Device, Index >::Grid( Dimensions... dimensions )
{
   setDimensions( dimensions... );

   proportions = 0;
   spaceSteps = 0;
   origin = 0;
}

template< int Dimension, typename Real, typename Device, typename Index >
Grid< Dimension, Real, Device, Index >::Grid( const CoordinatesType& dimensions )
{
   setDimensions( dimensions );

   proportions = 0;
   spaceSteps = 0;
   origin = 0;
}

template< int Dimension, typename Real, typename Device, typename Index >
constexpr Index
Grid< Dimension, Real, Device, Index >::getEntityOrientationsCount( IndexType entityDimension )
{
   return Templates::combination< Index >( entityDimension, Dimension );
}

template< int Dimension, typename Real, typename Device, typename Index >
template< typename... Dimensions,
          std::enable_if_t< Templates::conjunction_v< std::is_convertible< Index, Dimensions >... >, bool >,
          std::enable_if_t< sizeof...( Dimensions ) == Dimension, bool > >
void
Grid< Dimension, Real, Device, Index >::setDimensions( Dimensions... dimensions )
{
   this->dimensions = CoordinatesType( dimensions... );

   TNL_ASSERT_GE( this->dimensions, CoordinatesType( 0 ), "Dimension must be positive" );

   fillNormals();
   fillEntitiesCount();
   fillSpaceSteps();
   this->localBegin = 0;
   this->localEnd = this->getDimensions();
   this->interiorBegin = 1;
   this->interiorEnd = this->getDimensions() - 1;
}

template< int Dimension, typename Real, typename Device, typename Index >
void
Grid< Dimension, Real, Device, Index >::setDimensions(
   const typename Grid< Dimension, Real, Device, Index >::CoordinatesType& dimensions )
{
   this->dimensions = dimensions;

   TNL_ASSERT_GE( this->dimensions, CoordinatesType( 0 ), "Dimension must be positive" );

   fillNormals();
   fillEntitiesCount();
   fillSpaceSteps();
   this->localBegin = 0;
   this->localEnd = this->getDimensions();
   this->interiorBegin = 1;
   this->interiorEnd = this->getDimensions() - 1;
}

template< int Dimension, typename Real, typename Device, typename Index >
__cuda_callable__
const typename Grid< Dimension, Real, Device, Index >::CoordinatesType&
Grid< Dimension, Real, Device, Index >::getDimensions() const noexcept
{
   return this->dimensions;
}

template< int Dimension, typename Real, typename Device, typename Index >
__cuda_callable__
Index
Grid< Dimension, Real, Device, Index >::getEntitiesCount( IndexType dimension ) const
{
   TNL_ASSERT_GE( dimension, 0, "dimension must be greater than or equal to 0" );
   TNL_ASSERT_LE( dimension, Dimension, "dimension must be less than or equal to Dimension" );

   return this->cumulativeEntitiesCountAlongNormals( dimension );
}

template< int Dimension, typename Real, typename Device, typename Index >
template< int EntityDimension, std::enable_if_t< Templates::isInClosedInterval( 0, EntityDimension, Dimension ), bool > >
__cuda_callable__
Index
Grid< Dimension, Real, Device, Index >::getEntitiesCount() const noexcept
{
   return this->cumulativeEntitiesCountAlongNormals( EntityDimension );
}

template< int Dimension, typename Real, typename Device, typename Index >
template< typename EntityType_,
          std::enable_if_t< Templates::isInClosedInterval( 0, EntityType_::getEntityDimension(), Dimension ), bool > >
__cuda_callable__
Index
Grid< Dimension, Real, Device, Index >::getEntitiesCount() const noexcept
{
   return this->template getEntitiesCount< EntityType_::getEntityDimension() >();
}

template< int Dimension, typename Real, typename Device, typename Index >
template< typename... DimensionIndex,
          std::enable_if_t< Templates::conjunction_v< std::is_convertible< Index, DimensionIndex >... >, bool >,
          std::enable_if_t< ( sizeof...( DimensionIndex ) > 0 ), bool > >
__cuda_callable__
Containers::StaticVector< sizeof...( DimensionIndex ), Index >
Grid< Dimension, Real, Device, Index >::getEntitiesCounts( DimensionIndex... indices ) const
{
   Containers::StaticVector< sizeof...( DimensionIndex ), Index > result{ indices... };

   for( std::size_t i = 0; i < sizeof...( DimensionIndex ); i++ )
      result[ i ] = this->cumulativeEntitiesCountAlongNormals( result[ i ] );

   return result;
}

template< int Dimension, typename Real, typename Device, typename Index >
__cuda_callable__
const typename Grid< Dimension, Real, Device, Index >::EntitiesCounts&
Grid< Dimension, Real, Device, Index >::getEntitiesCounts() const noexcept
{
   return this->cumulativeEntitiesCountAlongNormals;
}

template< int Dimension, typename Real, typename Device, typename Index >
void
Grid< Dimension, Real, Device, Index >::setOrigin(
   const typename Grid< Dimension, Real, Device, Index >::PointType& origin ) noexcept
{
   this->origin = origin;
}

template< int Dimension, typename Real, typename Device, typename Index >
template< typename... Coordinates,
          std::enable_if_t< Templates::conjunction_v< std::is_convertible< Real, Coordinates >... >, bool >,
          std::enable_if_t< sizeof...( Coordinates ) == Dimension, bool > >
void
Grid< Dimension, Real, Device, Index >::setOrigin( Coordinates... coordinates ) noexcept
{
   this->origin = PointType( coordinates... );
}

template< int Dimension, typename Real, typename Device, typename Index >
__cuda_callable__
Index
Grid< Dimension, Real, Device, Index >::getOrientedEntitiesCount( IndexType dimension, IndexType orientation ) const
{
   TNL_ASSERT_GE( dimension, 0, "dimension must be greater than or equal to 0" );
   TNL_ASSERT_LE( dimension, Dimension, "dimension must be less than or equal to Dimension" );

   if( dimension == 0 || dimension == Dimension )
      return this->getEntitiesCount( dimension );

   Index index = Templates::firstKCombinationSum( dimension, (Index) Dimension ) + orientation;

   return this->entitiesCountAlongNormals[ index ];
}

template< int Dimension, typename Real, typename Device, typename Index >
template< int EntityDimension >
__cuda_callable__
typename Grid< Dimension, Real, Device, Index >::CoordinatesType
Grid< Dimension, Real, Device, Index >::getNormals( Index orientation ) const noexcept
{
   constexpr Index index = Templates::firstKCombinationSum( EntityDimension, Dimension );

   return this->normals( index + orientation );
}

template< int Dimension, typename Real, typename Device, typename Index >
template< int EntityDimension >
__cuda_callable__
typename Grid< Dimension, Real, Device, Index >::CoordinatesType
Grid< Dimension, Real, Device, Index >::getBasis( Index orientation ) const noexcept
{
   constexpr Index index = Templates::firstKCombinationSum( EntityDimension, Dimension );
   return 1 - this->normals( index + orientation );
}

template< int Dimension, typename Real, typename Device, typename Index >
template< int EntityDimension >
__cuda_callable__
Index
Grid< Dimension, Real, Device, Index >::getOrientation( const CoordinatesType& normals ) const noexcept
{
   constexpr Index index = Templates::firstKCombinationSum( EntityDimension, Dimension );
   const Index count = this->getEntityOrientationsCount( EntityDimension );
   for( IndexType orientation = 0; orientation < count; orientation++ )
      if( this->normals( index + orientation ) == normals )
         return orientation;
   return -1;
}

template< int Dimension, typename Real, typename Device, typename Index >
template< int EntityDimension >
__cuda_callable__
auto
Grid< Dimension, Real, Device, Index >::getEntityCoordinates( IndexType entityIdx,
                                                              CoordinatesType& entityNormals,
                                                              Index& orientation ) const noexcept -> CoordinatesType
{
   orientation = Templates::firstKCombinationSum( EntityDimension, Dimension );
   const Index end = orientation + this->getEntityOrientationsCount( EntityDimension );
   auto entityIdx_( entityIdx );
   while( orientation < end && entityIdx_ >= this->entitiesCountAlongNormals[ orientation ] ) {
      entityIdx_ -= this->entitiesCountAlongNormals[ orientation ];
      orientation++;
   }

   entityNormals = this->normals[ orientation ];
   const CoordinatesType dims = this->getDimensions() + entityNormals;
   CoordinatesType entityCoordinates( 0 );
   int idx = 0;
   while( idx < this->getMeshDimension() - 1 ) {
      entityCoordinates[ idx ] = entityIdx % dims[ idx ];
      entityIdx /= dims[ idx++ ];
   }
   entityCoordinates[ idx ] = entityIdx % dims[ idx ];
   return entityCoordinates;
}

template< int Dimension, typename Real, typename Device, typename Index >
template< int EntityDimension,
          int EntityOrientation,
          std::enable_if_t< Templates::isInClosedInterval( 0, EntityDimension, Dimension ), bool >,
          std::enable_if_t< Templates::isInClosedInterval( 0, EntityOrientation, Dimension ), bool > >
__cuda_callable__
Index
Grid< Dimension, Real, Device, Index >::getOrientedEntitiesCount() const noexcept
{
   if( EntityDimension == 0 || EntityDimension == Dimension )
      return this->getEntitiesCount( EntityDimension );

   constexpr Index index = Templates::firstKCombinationSum( EntityDimension, Dimension ) + EntityOrientation;

   return this->entitiesCountAlongNormals[ index ];
}

template< int Dimension, typename Real, typename Device, typename Index >
__cuda_callable__
const typename Grid< Dimension, Real, Device, Index >::PointType&
Grid< Dimension, Real, Device, Index >::getOrigin() const noexcept
{
   return this->origin;
}

template< int Dimension, typename Real, typename Device, typename Index >
void
Grid< Dimension, Real, Device, Index >::setDomain(
   const typename Grid< Dimension, Real, Device, Index >::PointType& origin,
   const typename Grid< Dimension, Real, Device, Index >::PointType& proportions )
{
   this->origin = origin;
   this->proportions = proportions;

   this->fillSpaceSteps();
}

template< int Dimension, typename Real, typename Device, typename Index >
void
Grid< Dimension, Real, Device, Index >::setSpaceSteps(
   const typename Grid< Dimension, Real, Device, Index >::PointType& spaceSteps ) noexcept
{
   this->spaceSteps = spaceSteps;

   fillSpaceStepsPowers();
   fillProportions();
}

template< int Dimension, typename Real, typename Device, typename Index >
template< typename... Steps,
          std::enable_if_t< Templates::conjunction_v< std::is_convertible< Real, Steps >... >, bool >,
          std::enable_if_t< sizeof...( Steps ) == Dimension, bool > >
void
Grid< Dimension, Real, Device, Index >::setSpaceSteps( Steps... spaceSteps ) noexcept
{
   this->spaceSteps = PointType( spaceSteps... );

   fillSpaceStepsPowers();
   fillProportions();
}

template< int Dimension, typename Real, typename Device, typename Index >
__cuda_callable__
const typename Grid< Dimension, Real, Device, Index >::PointType&
Grid< Dimension, Real, Device, Index >::getSpaceSteps() const noexcept
{
   return this->spaceSteps;
}

template< int Dimension, typename Real, typename Device, typename Index >
__cuda_callable__
const typename Grid< Dimension, Real, Device, Index >::PointType&
Grid< Dimension, Real, Device, Index >::getProportions() const noexcept
{
   return this->proportions;
}

template< int Dimension, typename Real, typename Device, typename Index >
template< typename... Powers,
          std::enable_if_t< Templates::conjunction_v< std::is_convertible< Index, Powers >... >, bool >,
          std::enable_if_t< sizeof...( Powers ) == Dimension, bool > >
__cuda_callable__
Real
Grid< Dimension, Real, Device, Index >::getSpaceStepsProducts( Powers... powers ) const
{
   int index = Templates::makeCollapsedIndex( this->spaceStepsPowersSize, CoordinatesType( powers... ) );

   return this->spaceStepsProducts( index );
}

template< int Dimension, typename Real, typename Device, typename Index >
__cuda_callable__
Real
Grid< Dimension, Real, Device, Index >::getSpaceStepsProducts( const CoordinatesType& powers ) const
{
   int index = Templates::makeCollapsedIndex( this->spaceStepsPowersSize, powers );

   return this->spaceStepsProducts( index );
}

template< int Dimension, typename Real, typename Device, typename Index >
template< Index... Powers, std::enable_if_t< sizeof...( Powers ) == Dimension, bool > >
__cuda_callable__
Real
Grid< Dimension, Real, Device, Index >::getSpaceStepsProducts() const noexcept
{
   constexpr int index = Templates::makeCollapsedIndex< Index, Powers... >( spaceStepsPowersSize );

   return this->spaceStepsProducts( index );
}

template< int Dimension, typename Real, typename Device, typename Index >
__cuda_callable__
Real
Grid< Dimension, Real, Device, Index >::getSmallestSpaceStep() const noexcept
{
   Real minStep = this->spaceSteps[ 0 ];
   Index i = 1;

   while( i != Dimension )
      minStep = min( minStep, this->spaceSteps[ i++ ] );

   return minStep;
}

template< int Dimension, typename Real, typename Device, typename Index >
template< int EntityDimension, typename Func, typename... FuncArgs >
void
Grid< Dimension, Real, Device, Index >::traverseAll( Func func, FuncArgs... args ) const
{
   this->traverseAll< EntityDimension >( CoordinatesType( 0 ), this->getDimensions(), func, args... );
}

template< int Dimension, typename Real, typename Device, typename Index >
template< int EntityDimension, typename Func, typename... FuncArgs >
void
Grid< Dimension, Real, Device, Index >::traverseAll( const CoordinatesType& from,
                                                     const CoordinatesType& to,
                                                     Func func,
                                                     FuncArgs... args ) const
{
   TNL_ASSERT_GE( from, CoordinatesType( 0 ), "Traverse rect must be in the grid dimensions" );
   TNL_ASSERT_LE( to, this->getDimensions(), "Traverse rect be in the grid dimensions" );
   TNL_ASSERT_LE( from, to, "Traverse rect must be defined from leading bottom anchor to trailing top anchor" );

   auto exec = [ & ]( const Index orientation, const CoordinatesType& normals )
   {
      Templates::ParallelFor< Dimension, Device, Index >::exec( from, to + normals, func, normals, orientation, args... );
   };
   Templates::ForEachOrientation< Index, EntityDimension, Dimension >::exec( exec );
}

template< int Dimension, typename Real, typename Device, typename Index >
template< int EntityDimension, typename Func, typename... FuncArgs >
void
Grid< Dimension, Real, Device, Index >::traverseInterior( Func func, FuncArgs... args ) const
{
   this->traverseInterior< EntityDimension >( CoordinatesType( 0 ), this->getDimensions(), func, args... );
}

template< int Dimension, typename Real, typename Device, typename Index >
template< int EntityDimension, typename Func, typename... FuncArgs >
void
Grid< Dimension, Real, Device, Index >::traverseInterior( const CoordinatesType& from,
                                                          const CoordinatesType& to,
                                                          Func func,
                                                          FuncArgs... args ) const
{
   TNL_ASSERT_GE( from, CoordinatesType( 0 ), "Traverse rect must be in the grid dimensions" );
   TNL_ASSERT_LE( to, this->getDimensions(), "Traverse rect be in the grid dimensions" );
   TNL_ASSERT_LE( from, to, "Traverse rect must be defined from leading bottom anchor to trailing top anchor" );

   auto exec = [ & ]( const Index orientation, const CoordinatesType& normals )
   {
      switch( EntityDimension ) {
         case 0:
            {
               Templates::ParallelFor< Dimension, Device, Index >::exec(
                  from + CoordinatesType( 1 ), to, func, normals, orientation, args... );
               break;
            }
         case Dimension:
            {
               Templates::ParallelFor< Dimension, Device, Index >::exec(
                  from + CoordinatesType( 1 ), to - CoordinatesType( 1 ), func, normals, orientation, args... );
               break;
            }
         default:
            {
               Templates::ParallelFor< Dimension, Device, Index >::exec(
                  from + normals, to, func, normals, orientation, args... );
               break;
            }
      }
   };

   Templates::ForEachOrientation< Index, EntityDimension, Dimension >::exec( exec );
}

template< int Dimension, typename Real, typename Device, typename Index >
template< int EntityDimension, typename Func, typename... FuncArgs >
void
Grid< Dimension, Real, Device, Index >::traverseBoundary( Func func, FuncArgs... args ) const
{
   this->traverseBoundary< EntityDimension >( CoordinatesType( 0 ), this->getDimensions(), func, args... );
}

template< int Dimension, typename Real, typename Device, typename Index >
template< int EntityDimension, typename Func, typename... FuncArgs >
void
Grid< Dimension, Real, Device, Index >::traverseBoundary( const CoordinatesType& from,
                                                          const CoordinatesType& to,
                                                          Func func,
                                                          FuncArgs... args ) const
{
   // Boundaries of the grid are formed by the entities of Dimension - 1.
   // We need to traverse each orientation independently.
   constexpr int orientationsCount = getEntityOrientationsCount( Dimension - 1 );
   constexpr bool isDirectedEntity = EntityDimension != 0 && EntityDimension != Dimension;
   constexpr bool isAnyBoundaryIntersects = EntityDimension != Dimension - 1;

   Containers::StaticVector< orientationsCount, Index > isBoundaryTraversed( 0 );

   auto forBoundary = [ & ]( const auto orthogonalOrientation, const auto orientation, const CoordinatesType& normals )
   {
      CoordinatesType start = from;
      CoordinatesType end = to + normals;

      if( isAnyBoundaryIntersects ) {
         for( Index i = 0; i < Dimension; i++ ) {
            start[ i ] = ( ! isDirectedEntity || normals[ i ] ) && isBoundaryTraversed[ i ] ? 1 : 0;
            end[ i ] = end[ i ] - ( ( ! isDirectedEntity || normals[ i ] ) && isBoundaryTraversed[ i ] ? 1 : 0 );
         }
      }

      start[ orthogonalOrientation ] = end[ orthogonalOrientation ] - 1;

      Templates::ParallelFor< Dimension, Device, Index >::exec( start, end, func, normals, orientation, args... );

      // Skip entities defined only once
      if( ! start[ orthogonalOrientation ] && end[ orthogonalOrientation ] )
         return;

      start[ orthogonalOrientation ] = 0;
      end[ orthogonalOrientation ] = 1;

      Templates::ParallelFor< Dimension, Device, Index >::exec( start, end, func, normals, orientation, args... );
   };

   if( ! isAnyBoundaryIntersects ) {
      auto exec = [ & ]( const auto orientation, const CoordinatesType& normals )
      {
         constexpr int orthogonalOrientation = EntityDimension - orientation;

         forBoundary( orthogonalOrientation, orientation, normals );
      };

      Templates::ForEachOrientation< Index, EntityDimension, Dimension >::exec( exec );
      return;
   }

   auto exec = [ & ]( const auto orthogonalOrientation )
   {
      auto exec = [ & ]( const auto orientation, const CoordinatesType& normals )
      {
         forBoundary( orthogonalOrientation, orientation, normals );
      };

      if( EntityDimension == 0 || EntityDimension == Dimension ) {
         Templates::ForEachOrientation< Index, EntityDimension, Dimension >::exec( exec );
      }
      else {
         Templates::ForEachOrientation< Index, EntityDimension, Dimension, orthogonalOrientation >::exec( exec );
      }

      isBoundaryTraversed[ orthogonalOrientation ] = 1;
   };

   Algorithms::staticFor< int, 0, orientationsCount >( exec );
}

template< int Dimension, typename Real, typename Device, typename Index >
template< typename Entity >
__cuda_callable__
Index
Grid< Dimension, Real, Device, Index >::getEntityIndex( const Entity& entity ) const
{
   static_assert( Entity::getEntityDimension() <= Dimension && Entity::getEntityDimension() >= 0,
                  "Wrong grid entity dimensions." );

   return GridEntityGetter< Grid, Entity::getEntityDimension() >::getEntityIndex( *this, entity );
}

template< int Dimension, typename Real, typename Device, typename Index >
template< typename EntityType_ >
__cuda_callable__
EntityType_
Grid< Dimension, Real, Device, Index >::getEntity( const CoordinatesType& coordinates ) const
{
   static_assert( EntityType_::getEntityDimension() <= getMeshDimension(),
                  "Entity dimension must be lower or equal to grid dimension." );
   return EntityType_( *this, coordinates );
}

template< int Dimension, typename Real, typename Device, typename Index >
template< int EntityDimension >
__cuda_callable__
auto
Grid< Dimension, Real, Device, Index >::getEntity( const CoordinatesType& coordinates ) const -> EntityType< EntityDimension >
{
   static_assert( EntityDimension <= getMeshDimension(), "Entity dimension must be lower or equal to grid dimension." );
   return EntityType< EntityDimension >( *this, coordinates );
}

template< int Dimension, typename Real, typename Device, typename Index >
template< typename EntityType_ >
__cuda_callable__
EntityType_
Grid< Dimension, Real, Device, Index >::getEntity( IndexType entityIdx ) const
{
   static_assert( EntityType_::getEntityDimension() <= getMeshDimension(),
                  "Entity dimension must be lower or equal to grid dimension." );
   return EntityType_( *this, entityIdx );
}

template< int Dimension, typename Real, typename Device, typename Index >
template< int EntityDimension >
__cuda_callable__
auto
Grid< Dimension, Real, Device, Index >::getEntity( IndexType entityIdx ) const -> EntityType< EntityDimension >
{
   static_assert( EntityDimension <= getMeshDimension(), "Entity dimension must be lower or equal to grid dimension." );
   return EntityType< EntityDimension >( *this, entityIdx );
}

template< int Dimension, typename Real, typename Device, typename Index >
void
Grid< Dimension, Real, Device, Index >::setLocalSubdomain( const CoordinatesType& begin, const CoordinatesType& end )
{
   this->localBegin = begin;
   this->localEnd = end;
}

template< int Dimension, typename Real, typename Device, typename Index >
void
Grid< Dimension, Real, Device, Index >::setLocalBegin( const CoordinatesType& begin )
{
   this->localBegin = begin;
}

template< int Dimension, typename Real, typename Device, typename Index >
void
Grid< Dimension, Real, Device, Index >::setLocalEnd( const CoordinatesType& end )
{
   this->localEnd = end;
}

template< int Dimension, typename Real, typename Device, typename Index >
auto
Grid< Dimension, Real, Device, Index >::getLocalBegin() const -> const CoordinatesType&
{
   return this->localBegin;
}

template< int Dimension, typename Real, typename Device, typename Index >
auto
Grid< Dimension, Real, Device, Index >::getLocalEnd() const -> const CoordinatesType&
{
   return this->localEnd;
}

template< int Dimension, typename Real, typename Device, typename Index >
void
Grid< Dimension, Real, Device, Index >::setInteriorBegin( const CoordinatesType& begin )
{
   this->interiorBegin = begin;
}

template< int Dimension, typename Real, typename Device, typename Index >
void
Grid< Dimension, Real, Device, Index >::setInteriorEnd( const CoordinatesType& end )
{
   this->interiorBegin = end;
}

template< int Dimension, typename Real, typename Device, typename Index >
auto
Grid< Dimension, Real, Device, Index >::getInteriorBegin() const -> const CoordinatesType&
{
   return this->interiorBegin;
}

template< int Dimension, typename Real, typename Device, typename Index >
auto
Grid< Dimension, Real, Device, Index >::getInteriorEnd() const -> const CoordinatesType&
{
   return this->interiorEnd;
}

template< int Dimension, typename Real, typename Device, typename Index >
void
Grid< Dimension, Real, Device, Index >::writeProlog( TNL::Logger& logger ) const noexcept
{
   logger.writeParameter( "Dimensions:", this->dimensions );

   logger.writeParameter( "Origin:", this->origin );
   logger.writeParameter( "Proportions:", this->proportions );
   logger.writeParameter( "Space steps:", this->spaceSteps );

   TNL::Algorithms::staticFor< IndexType, 0, Dimension + 1 >(
      [ & ]( auto entityDim )
      {
         for( IndexType entityOrientation = 0; entityOrientation < this->getEntityOrientationsCount( entityDim() );
              entityOrientation++ ) {
            auto normals = this->getBasis< entityDim >( entityOrientation );
            TNL::String tmp = TNL::String( "Entities count with basis " ) + TNL::convertToString( normals ) + ":";
            logger.writeParameter( tmp, this->getOrientedEntitiesCount( entityDim, entityOrientation ) );
         }
      } );
}

template< int Dimension, typename Real, typename Device, typename Index >
void
Grid< Dimension, Real, Device, Index >::fillEntitiesCount()
{
   for( Index i = 0; i < Dimension + 1; i++ )
      cumulativeEntitiesCountAlongNormals[ i ] = 0;

   // In case, if some dimension is zero. Clear all counts
   for( Index i = 0; i < Dimension; i++ ) {
      if( dimensions[ i ] == 0 ) {
         for( Index k = 0; k < (Index) entitiesCountAlongNormals.getSize(); k++ )
            entitiesCountAlongNormals[ k ] = 0;

         return;
      }
   }

   for( Index i = 0, j = 0; i <= Dimension; i++ ) {
      for( Index n = 0; n < this->getEntityOrientationsCount( i ); n++, j++ ) {
         int result = 1;
         auto normals = this->normals[ j ];

         for( Index k = 0; k < (Index) normals.getSize(); k++ )
            result *= dimensions[ k ] + normals[ k ];

         entitiesCountAlongNormals[ j ] = result;
         cumulativeEntitiesCountAlongNormals[ i ] += result;
      }
   }
}

template< int Dimension, typename Real, typename Device, typename Index >
void
Grid< Dimension, Real, Device, Index >::fillProportions()
{
   Index i = 0;

   while( i != Dimension ) {
      this->proportions[ i ] = this->spaceSteps[ i ] * this->dimensions[ i ];
      i++;
   }
}

template< int Dimension, typename Real, typename Device, typename Index >
void
Grid< Dimension, Real, Device, Index >::fillSpaceSteps()
{
   bool hasAnyInvalidDimension = false;

   for( Index i = 0; i < Dimension; i++ ) {
      if( this->dimensions[ i ] <= 0 ) {
         hasAnyInvalidDimension = true;
         break;
      }
   }

   if( ! hasAnyInvalidDimension ) {
      for( Index i = 0; i < Dimension; i++ )
         this->spaceSteps[ i ] = this->proportions[ i ] / this->dimensions[ i ];

      fillSpaceStepsPowers();
   }
}

template< int Dimension, typename Real, typename Device, typename Index >
void
Grid< Dimension, Real, Device, Index >::fillSpaceStepsPowers()
{
   Containers::StaticVector< spaceStepsPowersSize * Dimension, Real > powers;

   for( Index i = 0; i < Dimension; i++ ) {
      Index power = -( this->spaceStepsPowersSize >> 1 );

      for( Index j = 0; j < spaceStepsPowersSize; j++ ) {
         powers[ i * spaceStepsPowersSize + j ] = pow( this->spaceSteps[ i ], power );
         power++;
      }
   }

   for( Index i = 0; i < this->spaceStepsProducts.getSize(); i++ ) {
      Real product = 1;
      Index index = i;

      for( Index j = 0; j < Dimension; j++ ) {
         Index residual = index % this->spaceStepsPowersSize;

         index /= this->spaceStepsPowersSize;

         product *= powers[ j * spaceStepsPowersSize + residual ];
      }

      spaceStepsProducts[ i ] = product;
   }
}

template< int Dimension, typename Real, typename Device, typename Index >
void
Grid< Dimension, Real, Device, Index >::fillNormals()
{
   OrientationNormalsContainer container;
   for( int i = 0; i < OrientationNormalsContainer::getSize(); i++ )
      for( int j = 0; j < OrientationNormalsContainer::ValueType::getSize(); j++ )
         container[ i ][ j ] = 0;

   int index = 0;

   auto forEachEntityDimension = [ & ]( const auto entityDimension )
   {
      constexpr Index combinationsCount = getEntityOrientationsCount( entityDimension );

      auto forEachOrientation = [ & ]( const auto orientation, const auto entityDimension )
      {
         container[ index++ ] = NormalsGetter< Index, entityDimension, Dimension >::template getNormals< orientation >();
      };
      Algorithms::staticFor< int, 0, combinationsCount >( forEachOrientation, entityDimension );
   };

   Algorithms::staticFor< int, 0, Dimension + 1 >( forEachEntityDimension );
   this->normals = container;
}

template< int Dimension, typename Real, typename Device, typename Index >
template< int EntityDimension, typename Func, typename... FuncArgs >
void
Grid< Dimension, Real, Device, Index >::forAllEntities( Func func, FuncArgs... args ) const
{
   auto exec = [ = ] __cuda_callable__( const CoordinatesType& coordinate,
                                        const CoordinatesType& normals,
                                        const Index orientation,
                                        const Grid& grid,
                                        FuncArgs... args ) mutable
   {
      EntityType< EntityDimension > entity( grid, coordinate, normals, orientation );

      func( entity, args... );
   };

   this->template traverseAll< EntityDimension >( exec, *this, args... );
}

template< int Dimension, typename Real, typename Device, typename Index >
template< int EntityDimension, typename Func, typename... FuncArgs >
void
Grid< Dimension, Real, Device, Index >::forEntities( const CoordinatesType& begin,
                                                     const CoordinatesType& end,
                                                     Func func,
                                                     FuncArgs... args ) const
{
   auto exec = [ = ] __cuda_callable__( const CoordinatesType& coordinate,
                                        const CoordinatesType& normals,
                                        const Index orientation,
                                        const Grid& grid,
                                        FuncArgs... args ) mutable
   {
      EntityType< EntityDimension > entity( grid, coordinate, normals, orientation );

      func( entity, args... );
   };

   this->template traverseAll< EntityDimension >( begin, end, exec, *this, args... );
}

template< int Dimension, typename Real, typename Device, typename Index >
template< int EntityDimension, typename Func, typename... FuncArgs >
void
Grid< Dimension, Real, Device, Index >::forBoundaryEntities( Func func, FuncArgs... args ) const
{
   auto exec = [ = ] __cuda_callable__( const CoordinatesType& coordinate,
                                        const CoordinatesType& normals,
                                        const Index orientation,
                                        const Grid& grid,
                                        FuncArgs... args ) mutable
   {
      EntityType< EntityDimension > entity( grid, coordinate, normals, orientation );

      func( entity, args... );
   };

   this->template traverseBoundary< EntityDimension >( exec, *this, args... );
}

template< int Dimension, typename Real, typename Device, typename Index >
template< int EntityDimension, typename Func, typename... FuncArgs >
void
Grid< Dimension, Real, Device, Index >::forBoundaryEntities( const CoordinatesType& begin,
                                                             const CoordinatesType& end,
                                                             Func func,
                                                             FuncArgs... args ) const
{
   auto exec = [ = ] __cuda_callable__( const CoordinatesType& coordinate,
                                        const CoordinatesType& normals,
                                        const Index orientation,
                                        const Grid& grid,
                                        FuncArgs... args ) mutable
   {
      EntityType< EntityDimension > entity( grid, coordinate, normals, orientation );

      func( entity, args... );
   };

   this->template traverseBoundary< EntityDimension >( begin, end, exec, *this, args... );
}

template< int Dimension, typename Real, typename Device, typename Index >
template< int EntityDimension, typename Func, typename... FuncArgs >
void
Grid< Dimension, Real, Device, Index >::forInteriorEntities( Func func, FuncArgs... args ) const
{
   auto exec = [ = ] __cuda_callable__( const CoordinatesType& coordinate,
                                        const CoordinatesType& normals,
                                        const Index orientation,
                                        const Grid& grid,
                                        FuncArgs... args ) mutable
   {
      EntityType< EntityDimension > entity( grid, coordinate, normals, orientation );

      func( entity, args... );
   };

   this->template traverseInterior< EntityDimension >( exec, *this, args... );
}

template< int Dimension, typename Real, typename Device, typename Index >
template< int EntityDimension, typename Func, typename... FuncArgs >
void
Grid< Dimension, Real, Device, Index >::forLocalEntities( Func func, FuncArgs... args ) const
{
   auto exec = [ = ] __cuda_callable__( const CoordinatesType& coordinate,
                                        const CoordinatesType& normals,
                                        const Index orientation,
                                        const Grid& grid,
                                        FuncArgs... args ) mutable
   {
      EntityType< EntityDimension > entity( grid, coordinate, normals, orientation );

      func( entity, args... );
   };

   this->template traverseAll< EntityDimension >( this->localBegin, this->localEnd, exec, *this, args... );
}

}  // namespace Meshes
}  // namespace noa::TNL

#endif
