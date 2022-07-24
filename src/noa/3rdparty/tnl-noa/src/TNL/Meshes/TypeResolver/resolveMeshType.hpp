// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <filesystem>

#include <noa/3rdparty/tnl-noa/src/TNL/Meshes/TypeResolver/resolveMeshType.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Meshes/TypeResolver/GridTypeResolver.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Meshes/TypeResolver/MeshTypeResolver.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Meshes/Readers/getMeshReader.h>

namespace noa::TNL {
namespace Meshes {

template< typename ConfigTag, typename Device, typename Functor >
bool
resolveMeshType( Functor&& functor,
                 const std::string& fileName,
                 const std::string& fileFormat,
                 const std::string& realType,
                 const std::string& globalIndexType )
{
   std::cout << "Detecting mesh from file " << fileName << " ..." << std::endl;

   std::shared_ptr< Readers::MeshReader > reader = Readers::getMeshReader( fileName, fileFormat );
   if( reader == nullptr )
      return false;

   reader->detectMesh();

   if( realType != "auto" )
      reader->forceRealType( realType );
   if( globalIndexType != "auto" )
      reader->forceGlobalIndexType( globalIndexType );

   if( reader->getMeshType() == "Meshes::Grid" || reader->getMeshType() == "Meshes::DistributedGrid" )
      return GridTypeResolver< ConfigTag, Device >::run( *reader, functor );
   else if( reader->getMeshType() == "Meshes::Mesh" || reader->getMeshType() == "Meshes::DistributedMesh" )
      return MeshTypeResolver< ConfigTag, Device >::run( *reader, functor );
   else {
      std::cerr << "The mesh type " << reader->getMeshType() << " is not supported." << std::endl;
      return false;
   }
}

template< typename ConfigTag, typename Device, typename Functor >
bool
resolveAndLoadMesh( Functor&& functor,
                    const std::string& fileName,
                    const std::string& fileFormat,
                    const std::string& realType,
                    const std::string& globalIndexType )
{
   auto wrapper = [ & ]( auto& reader, auto&& mesh ) -> bool
   {
      using MeshType = std::decay_t< decltype( mesh ) >;
      std::cout << "Loading a mesh from the file " << fileName << " ..." << std::endl;
      try {
         reader.loadMesh( mesh );
      }
      catch( const Meshes::Readers::MeshReaderError& e ) {
         std::cerr << "Failed to load the mesh from the file " << fileName << ". The error is:\n" << e.what() << std::endl;
         return false;
      }
      return functor( reader, std::forward< MeshType >( mesh ) );
   };
   return resolveMeshType< ConfigTag, Device >( wrapper, fileName, fileFormat, realType, globalIndexType );
}

template< typename Mesh >
bool
loadMesh( Mesh& mesh, const std::string& fileName, const std::string& fileFormat )
{
   std::cout << "Loading a mesh from the file " << fileName << " ..." << std::endl;

   std::shared_ptr< Readers::MeshReader > reader = Readers::getMeshReader( fileName, fileFormat );
   if( reader == nullptr )
      return false;

   try {
      reader->loadMesh( mesh );
   }
   catch( const Meshes::Readers::MeshReaderError& e ) {
      std::cerr << "Failed to load the mesh from the file " << fileName << ". The error is:\n" << e.what() << std::endl;
      return false;
   }

   return true;
}

template< typename MeshConfig >
bool
loadMesh( Mesh< MeshConfig, Devices::Cuda >& mesh, const std::string& fileName, const std::string& fileFormat )
{
   Mesh< MeshConfig, Devices::Host > hostMesh;
   if( ! loadMesh( hostMesh, fileName, fileFormat ) )
      return false;
   mesh = hostMesh;
   return true;
}

}  // namespace Meshes
}  // namespace noa::TNL
