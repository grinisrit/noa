// Copyright (c) 2004-2023 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <noa/3rdparty/tnl-noa/src/TNL/Meshes/Grid.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Exceptions/NotImplementedError.h>

namespace noa::TNL {
namespace Functions {

template< typename Mesh >
class MeshFunctionNormGetter
{};

/***
 * Specialization for grids
 * TODO: implement this even for other devices
 */
template< int Dimension, typename MeshReal, typename MeshIndex >
class MeshFunctionNormGetter< Meshes::Grid< Dimension, MeshReal, Devices::Host, MeshIndex > >
{
public:
   using GridType = Meshes::Grid< Dimension, MeshReal, Devices::Host, MeshIndex >;
   using MeshRealType = MeshReal;
   using DeviceType = Devices::Host;
   using MeshIndexType = MeshIndex;

   template< typename MeshFunctionType >
   static typename MeshFunctionType::RealType
   getNorm( const MeshFunctionType& function, const typename MeshFunctionType::RealType& p )
   {
      using RealType = typename MeshFunctionType::RealType;
      static constexpr int EntityDimension = MeshFunctionType::getEntitiesDimension();
      if( EntityDimension == Dimension ) {
         if( p == 1.0 )
            return function.getMesh().getCellMeasure() * lpNorm( function.getData(), 1.0 );
         if( p == 2.0 )
            return std::sqrt( function.getMesh().getCellMeasure() ) * lpNorm( function.getData(), 2.0 );
         return std::pow( function.getMesh().getCellMeasure(), 1.0 / p ) * lpNorm( function.getData(), p );
      }
      if( EntityDimension > 0 ) {
         using MeshType = typename MeshFunctionType::MeshType;
         using EntityType = typename MeshType::Face;
         if( p == 1.0 ) {
            RealType result( 0.0 );
            for( MeshIndexType i = 0; i < function.getMesh().template getEntitiesCount< EntityType >(); i++ ) {
               EntityType entity = function.getMesh().template getEntity< EntityType >( i );
               result += std::fabs( function[ i ] ) * entity.getMeasure();
            }
            return result;
         }
         if( p == 2.0 ) {
            RealType result( 0.0 );
            for( MeshIndexType i = 0; i < function.getMesh().template getEntitiesCount< EntityType >(); i++ ) {
               EntityType entity = function.getMesh().template getEntity< EntityType >( i );
               result += function[ i ] * function[ i ] * entity.getMeasure();
            }
            return std::sqrt( result );
         }

         RealType result( 0.0 );
         for( MeshIndexType i = 0; i < function.getMesh().template getEntitiesCount< EntityType >(); i++ ) {
            EntityType entity = function.getMesh().template getEntity< EntityType >( i );
            result += std::pow( std::fabs( function[ i ] ), p ) * entity.getMeasure();
         }
         return std::pow( result, 1.0 / p );
      }

      if( p == 1.0 )
         return lpNorm( function.getData(), 1.0 );
      if( p == 2.0 )
         return lpNorm( function.getData(), 2.0 );
      return lpNorm( function.getData(), p );
   }
};

/****
 * Specialization for CUDA devices
 */
template< int Dimension, typename MeshReal, typename MeshIndex >
class MeshFunctionNormGetter< Meshes::Grid< Dimension, MeshReal, Devices::Cuda, MeshIndex > >
{
public:
   using GridType = Meshes::Grid< Dimension, MeshReal, Devices::Cuda, MeshIndex >;
   using MeshRealType = MeshReal;
   using DeviceType = Devices::Cuda;
   using MeshIndexType = MeshIndex;

   template< typename MeshFunctionType >
   static typename MeshFunctionType::RealType
   getNorm( const MeshFunctionType& function, const typename MeshFunctionType::RealType& p )
   {
      static constexpr int EntityDimension = MeshFunctionType::getEntitiesDimension();
      if( EntityDimension == Dimension ) {
         if( p == 1.0 )
            return function.getMesh().getCellMeasure() * function.getData().lpNorm( 1.0 );
         if( p == 2.0 )
            return ::sqrt( function.getMesh().getCellMeasure() ) * function.getData().lpNorm( 2.0 );
         return ::pow( function.getMesh().getCellMeasure(), 1.0 / p ) * function.getData().lpNorm( p );
      }
      if( EntityDimension > 0 ) {
         throw Exceptions::NotImplementedError( "Not implemented yet." );
      }

      if( p == 1.0 )
         return function.getData().lpNorm( 1.0 );
      if( p == 2.0 )
         return function.getData().lpNorm( 2.0 );
      return function.getData().lpNorm( p );
   }
};

}  // namespace Functions
}  // namespace noa::TNL
