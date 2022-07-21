// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <filesystem>

#include <noa/3rdparty/tnl-noa/src/TNL/Meshes/TypeResolver/resolveDistributedMeshType.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Meshes/TypeResolver/resolveMeshType.h>

namespace noa::TNL {
namespace Meshes {

template< typename ConfigTag, typename Device, typename Functor >
bool
resolveDistributedMeshType( Functor&& functor, const std::string& fileName, const std::string& fileFormat )
{
   std::cout << "Detecting distributed mesh from file " << fileName << " ..." << std::endl;

   auto wrapper = [ &functor ]( Readers::MeshReader& reader, auto&& localMesh )
   {
      using LocalMesh = std::decay_t< decltype( localMesh ) >;
      using DistributedMesh = DistributedMeshes::DistributedMesh< LocalMesh >;
      return std::forward< Functor >( functor )( reader, DistributedMesh{ std::move( localMesh ) } );
   };

   return resolveMeshType< ConfigTag, Device >( wrapper, fileName, fileFormat );
}

template< typename ConfigTag, typename Device, typename Functor >
bool
resolveAndLoadDistributedMesh( Functor&& functor,
                               const std::string& fileName,
                               const std::string& fileFormat,
                               const MPI::Comm& communicator )
{
   auto wrapper = [ & ]( Readers::MeshReader& reader, auto&& mesh ) -> bool
   {
      using MeshType = std::decay_t< decltype( mesh ) >;
      std::cout << "Loading a mesh from the file " << fileName << " ..." << std::endl;
      try {
         if( reader.getMeshType() == "Meshes::DistributedMesh" ) {
            auto& pvtu = dynamic_cast< Readers::PVTUReader& >( reader );
            pvtu.setCommunicator( communicator );
            pvtu.loadMesh( mesh );
         }
         else if( reader.getMeshType() == "Meshes::DistributedGrid" ) {
            auto& pvti = dynamic_cast< Readers::PVTIReader& >( reader );
            pvti.setCommunicator( communicator );
            pvti.loadMesh( mesh );
         }
         else
            throw std::runtime_error( "Unknown type of a distributed mesh: " + reader.getMeshType() );
      }
      catch( const Meshes::Readers::MeshReaderError& e ) {
         std::cerr << "Failed to load the mesh from the file " << fileName << ". The error is:\n" << e.what() << std::endl;
         return false;
      }
      return functor( reader, std::forward< MeshType >( mesh ) );
   };
   return resolveDistributedMeshType< ConfigTag, Device >( wrapper, fileName, fileFormat );
}

template< typename Mesh >
bool
loadDistributedMesh( DistributedMeshes::DistributedMesh< Mesh >& distributedMesh,
                     const std::string& fileName,
                     const std::string& fileFormat,
                     const MPI::Comm& communicator )
{
   namespace fs = std::filesystem;
   std::string format = fileFormat;
   if( format == "auto" ) {
      format = fs::path( fileName ).extension();
      if( format.length() > 0 )
         // remove dot from the extension
         format = format.substr( 1 );
   }

   if( format == "pvtu" ) {
      Readers::PVTUReader reader( fileName, communicator );
      reader.loadMesh( distributedMesh );
      return true;
   }
   else if( format == "pvti" ) {
      Readers::PVTIReader reader( fileName, communicator );
      reader.loadMesh( distributedMesh );
      return true;
   }
   else {
      if( fileFormat == "auto" )
         std::cerr << "File '" << fileName << "' has unsupported format (based on the file extension): " << format << ".";
      else
         std::cerr << "Unsupported fileFormat parameter: " << fileFormat << ".";
      std::cerr << " Supported formats are 'pvtu' and 'pvti'." << std::endl;
      return false;
   }
}

}  // namespace Meshes
}  // namespace noa::TNL
