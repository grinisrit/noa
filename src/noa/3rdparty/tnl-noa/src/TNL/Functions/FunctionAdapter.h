// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <noa/3rdparty/tnl-noa/src/TNL/Cuda/CudaCallable.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Config/ParameterContainer.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Functions/Domain.h>

namespace noa::TNL {
namespace Functions {

/***
 * MeshType is a type of mesh on which we evaluate the function.
 * DomainType (defined in functions/Domain.h) defines a domain of
 * the function. In TNL, we mostly work with mesh functions. In this case
 * mesh entity and time is passed to the function...
 */
template< typename Mesh, typename Function, int domainType = Function::getDomainType() >
class FunctionAdapter
{
public:
   using FunctionType = Function;
   using MeshType = Mesh;
   using RealType = typename FunctionType::RealType;
   using IndexType = typename MeshType::GlobalIndexType;
   // typedef typename FunctionType::PointType PointType;

   template< typename MeshPointer >
   static bool
   setup( FunctionType& function,
          const MeshPointer& meshPointer,
          const Config::ParameterContainer& parameters,
          const String& prefix = "" )
   {
      return function.setup( meshPointer, parameters, prefix );
   }

   template< typename EntityType >
   __cuda_callable__
   inline static RealType
   getValue( const FunctionType& function, const EntityType& meshEntity, const RealType& time )
   {
      return function( meshEntity, time );
   }
};

/***
 * Specialization for analytic functions. In this case
 * we pass vertex and time to the function ...
 */
template< typename Mesh, typename Function >
class FunctionAdapter< Mesh, Function, SpaceDomain >
{
public:
   using FunctionType = Function;
   using MeshType = Mesh;
   using RealType = typename FunctionType::RealType;
   using IndexType = typename MeshType::GlobalIndexType;
   using PointType = typename FunctionType::PointType;

   template< typename MeshPointer >
   static bool
   setup( FunctionType& function,
          const MeshPointer& meshPointer,
          const Config::ParameterContainer& parameters,
          const String& prefix = "" )
   {
      return function.setup( parameters, prefix );
   }

   template< typename EntityType >
   __cuda_callable__
   inline static RealType
   getValue( const FunctionType& function, const EntityType& meshEntity, const RealType& time )
   {
      return function( meshEntity.getCenter(), time );
   }
};

/***
 * Specialization for analytic space independent functions.
 * Such function does not depend on any space variable and so
 * we pass only time.
 */
template< typename Mesh, typename Function >
class FunctionAdapter< Mesh, Function, NonspaceDomain >
{
public:
   using FunctionType = Function;
   using MeshType = Mesh;
   using RealType = typename FunctionType::RealType;
   using IndexType = typename MeshType::GlobalIndexType;
   using PointType = typename FunctionType::PointType;

   template< typename MeshPointer >
   static bool
   setup( FunctionType& function,
          const MeshPointer& meshPointer,
          const Config::ParameterContainer& parameters,
          const String& prefix = "" )
   {
      return function.setup( parameters, prefix );
   }

   template< typename EntityType >
   __cuda_callable__
   inline static RealType
   getValue( const FunctionType& function, const EntityType& meshEntity, const RealType& time )
   {
      return function.getValue( time );
   }
};

#ifdef UNDEF

/***
 * Specialization for mesh functions
 */
template< typename Mesh, typename Function >
class FunctionAdapter< Mesh, Function, MeshFunction >
{
public:
   typedef Function FunctionType;
   typedef Mesh MeshType;
   typedef typename FunctionType::RealType RealType;
   typedef typename MeshType::GlobalIndexType IndexType;

   template< typename EntityType >
   __cuda_callable__
   inline static RealType
   getValue( const FunctionType& function, const EntityType& meshEntity, const RealType& time )
   {
      return function( meshEntity, time );
   }
};

/***
 * Specialization for analytic functions
 */
template< typename Mesh, typename Function >
class FunctionAdapter< Mesh, Function, SpaceDomain >
{
public:
   typedef Function FunctionType;
   typedef Mesh MeshType;
   typedef typename FunctionType::RealType RealType;
   typedef typename MeshType::GlobalIndexType IndexType;
   typedef typename FunctionType::PointType PointType;

   template< typename EntityType >
   __cuda_callable__
   inline static RealType
   getValue( const FunctionType& function, const EntityType& meshEntity, const RealType& time )
   {
      return function.getValue( meshEntity.getCenter(), time );
   }
};

/***
 * Specialization for constant analytic functions
 */
template< typename Mesh, typename Function >
class FunctionAdapter< Mesh, Function, SpaceDomain >
{
public:
   typedef Function FunctionType;
   typedef Mesh MeshType;
   typedef typename FunctionType::RealType RealType;
   typedef typename MeshType::GlobalIndexType IndexType;
   typedef typename FunctionType::PointType PointType;

   template< typename EntityType >
   __cuda_callable__
   inline static RealType
   getValue( const FunctionType& function, const EntityType& meshEntity, const RealType& time )
   {
      return function.getValue( time );
   }
};
#endif

}  // namespace Functions
}  // namespace noa::TNL
