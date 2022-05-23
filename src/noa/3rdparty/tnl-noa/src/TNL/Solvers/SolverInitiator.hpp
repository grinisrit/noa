// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#include <noa/3rdparty/tnl-noa/src/TNL/Solvers/SolverInitiator.h>

#include <noa/3rdparty/tnl-noa/src/TNL/Devices/Host.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Devices/Cuda.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Config/ParameterContainer.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Meshes/TypeResolver/resolveMeshType.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Solvers/BuildConfigTags.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Solvers/SolverStarter.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Meshes/DummyMesh.h>

namespace noa::TNL {
namespace Solvers {

template<
   template< typename Real, typename Device, typename Index, typename MeshType, typename ConfigTag, typename SolverStarter >
   class ProblemSetter,
   typename Real,
   typename ConfigTag,
   bool enabled = ConfigTagReal< ConfigTag, Real >::enabled >
class SolverInitiatorRealResolver
{};

template<
   template< typename Real, typename Device, typename Index, typename MeshType, typename ConfigTag, typename SolverStarter >
   class ProblemSetter,
   typename Real,
   typename Device,
   typename ConfigTag,
   bool enabled = ConfigTagDevice< ConfigTag, Device >::enabled >
class SolverInitiatorDeviceResolver
{};

template<
   template< typename Real, typename Device, typename Index, typename MeshType, typename ConfigTag, typename SolverStarter >
   class ProblemSetter,
   typename Real,
   typename Device,
   typename Index,
   typename ConfigTag,
   bool enabled = ConfigTagIndex< ConfigTag, Index >::enabled >
class SolverInitiatorIndexResolver
{};

template<
   template< typename Real, typename Device, typename Index, typename MeshType, typename ConfigTag, typename SolverStarter >
   class ProblemSetter,
   typename Real,
   typename Device,
   typename Index,
   typename ConfigTag,
   bool enabled = ConfigTagMeshResolve< ConfigTag >::enabled >
class SolverInitiatorMeshResolver
{};

template<
   template< typename Real, typename Device, typename Index, typename MeshType, typename ConfigTag, typename SolverStarter >
   class ProblemSetter,
   typename ConfigTag >
bool
SolverInitiator< ProblemSetter, ConfigTag >::run( const Config::ParameterContainer& parameters )
{
   const String& realType = parameters.getParameter< String >( "real-type" );
   if( realType == "float" )
      return SolverInitiatorRealResolver< ProblemSetter, float, ConfigTag >::run( parameters );
   if( realType == "double" )
      return SolverInitiatorRealResolver< ProblemSetter, double, ConfigTag >::run( parameters );
   if( realType == "long-double" )
      return SolverInitiatorRealResolver< ProblemSetter, long double, ConfigTag >::run( parameters );
   std::cerr << "The real type '" << realType << "' is not defined. " << std::endl;
   return false;
};

template<
   template< typename Real, typename Device, typename Index, typename MeshType, typename ConfigTag, typename SolverStarter >
   class ProblemSetter,
   typename Real,
   typename ConfigTag >
class SolverInitiatorRealResolver< ProblemSetter, Real, ConfigTag, true >
{
public:
   static bool
   run( const Config::ParameterContainer& parameters )
   {
      const String& device = parameters.getParameter< String >( "device" );
      if( device == "host" )
         return SolverInitiatorDeviceResolver< ProblemSetter, Real, Devices::Host, ConfigTag >::run( parameters );
      if( device == "cuda" )
         return SolverInitiatorDeviceResolver< ProblemSetter, Real, Devices::Cuda, ConfigTag >::run( parameters );
      std::cerr << "The device '" << device << "' is not defined. " << std::endl;
      return false;
   }
};

template<
   template< typename Real, typename Device, typename Index, typename MeshType, typename ConfigTag, typename SolverStarter >
   class ProblemSetter,
   typename Real,
   typename ConfigTag >
class SolverInitiatorRealResolver< ProblemSetter, Real, ConfigTag, false >
{
public:
   static bool
   run( const Config::ParameterContainer& parameters )
   {
      std::cerr << "The real type " << parameters.getParameter< String >( "real-type" ) << " is not supported." << std::endl;
      return false;
   }
};

template<
   template< typename Real, typename Device, typename Index, typename MeshType, typename ConfigTag, typename SolverStarter >
   class ProblemSetter,
   typename Real,
   typename Device,
   typename ConfigTag >
class SolverInitiatorDeviceResolver< ProblemSetter, Real, Device, ConfigTag, true >
{
public:
   static bool
   run( const Config::ParameterContainer& parameters )
   {
      const String& indexType = parameters.getParameter< String >( "index-type" );
      if( indexType == "short-int" )
         return SolverInitiatorIndexResolver< ProblemSetter, Real, Device, short int, ConfigTag >::run( parameters );
      if( indexType == "int" )
         return SolverInitiatorIndexResolver< ProblemSetter, Real, Device, int, ConfigTag >::run( parameters );
      if( indexType == "long int" )
         return SolverInitiatorIndexResolver< ProblemSetter, Real, Device, long int, ConfigTag >::run( parameters );
      std::cerr << "The index type '" << indexType << "' is not defined. " << std::endl;
      return false;
   }
};

template<
   template< typename Real, typename Device, typename Index, typename MeshType, typename ConfigTag, typename SolverStarter >
   class ProblemSetter,
   typename Real,
   typename Device,
   typename ConfigTag >
class SolverInitiatorDeviceResolver< ProblemSetter, Real, Device, ConfigTag, false >
{
public:
   static bool
   run( const Config::ParameterContainer& parameters )
   {
      std::cerr << "The device " << parameters.getParameter< String >( "device" ) << " is not supported." << std::endl;
      return false;
   }
};

template<
   template< typename Real, typename Device, typename Index, typename MeshType, typename ConfigTag, typename SolverStarter >
   class ProblemSetter,
   typename Real,
   typename Device,
   typename Index,
   typename ConfigTag >
class SolverInitiatorIndexResolver< ProblemSetter, Real, Device, Index, ConfigTag, false >
{
public:
   static bool
   run( const Config::ParameterContainer& parameters )
   {
      std::cerr << "The index " << parameters.getParameter< String >( "index-type" ) << " is not supported." << std::endl;
      return false;
   }
};

template<
   template< typename Real, typename Device, typename Index, typename MeshType, typename ConfigTag, typename SolverStarter >
   class ProblemSetter,
   typename Real,
   typename Device,
   typename Index,
   typename ConfigTag >
class SolverInitiatorIndexResolver< ProblemSetter, Real, Device, Index, ConfigTag, true >
{
public:
   static bool
   run( const Config::ParameterContainer& parameters )
   {
      return SolverInitiatorMeshResolver< ProblemSetter, Real, Device, Index, ConfigTag >::run( parameters );
   }
};

template<
   template< typename Real, typename Device, typename Index, typename MeshType, typename ConfigTag, typename SolverStarter >
   class ProblemSetter,
   typename Real,
   typename Device,
   typename Index,
   typename ConfigTag >
class SolverInitiatorMeshResolver< ProblemSetter, Real, Device, Index, ConfigTag, false >
{
public:
   static bool
   run( const Config::ParameterContainer& parameters )
   {
      return ProblemSetter< Real,
                            Device,
                            Index,
                            Meshes::DummyMesh< Real, Device, Index >,
                            ConfigTag,
                            SolverStarter< ConfigTag > >::template run< Real, Device, Index, ConfigTag >( parameters );
   }
};

template<
   template< typename Real, typename Device, typename Index, typename MeshType, typename ConfigTag, typename SolverStarter >
   class ProblemSetter,
   typename Real,
   typename Device,
   typename Index,
   typename ConfigTag >
class SolverInitiatorMeshResolver< ProblemSetter, Real, Device, Index, ConfigTag, true >
{
   // wrapper for MeshTypeResolver
   template< typename MeshType >
   using ProblemSetterWrapper = ProblemSetter< Real, Device, Index, MeshType, ConfigTag, SolverStarter< ConfigTag > >;

public:
   static bool
   run( const Config::ParameterContainer& parameters )
   {
      const String& meshFileName = parameters.getParameter< String >( "mesh" );
      const String& meshFileFormat = parameters.getParameter< String >( "mesh-format" );
      auto wrapper = [ & ]( const auto& reader, auto&& mesh )
      {
         using MeshType = std::decay_t< decltype( mesh ) >;
         return ProblemSetterWrapper< MeshType >::run( parameters );
      };
      return Meshes::resolveMeshType< ConfigTag, Device >( wrapper, meshFileName, meshFileFormat );
   }
};

}  // namespace Solvers
}  // namespace noa::TNL
