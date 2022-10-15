// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

// Implemented by: Jakub Klinkovský

#pragma once

#include <memory>
#include <filesystem>

#include <noa/3rdparty/tnl-noa/src/TNL/Meshes/Readers/NetgenReader.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Meshes/Readers/VTKReader.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Meshes/Readers/VTUReader.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Meshes/Readers/VTIReader.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Meshes/Readers/PVTUReader.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Meshes/Readers/PVTIReader.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Meshes/Readers/FPMAReader.h>

namespace noa::TNL {
namespace Meshes {
namespace Readers {

inline std::shared_ptr< MeshReader >
getMeshReader( const std::string& fileName, const std::string& fileFormat )
{
   namespace fs = std::filesystem;
   std::string format = fileFormat;
   if( format == "auto" ) {
      format = fs::path( fileName ).extension();
      if( format.length() > 0 )
         // remove dot from the extension
         format = format.substr( 1 );
   }

   if( format == "ng" )
      return std::make_shared< Readers::NetgenReader >( fileName );
   else if( format == "vtk" )
      return std::make_shared< Readers::VTKReader >( fileName );
   else if( format == "vtu" )
      return std::make_shared< Readers::VTUReader >( fileName );
   else if( format == "vti" )
      return std::make_shared< Readers::VTIReader >( fileName );
   else if( format == "pvtu" )
      return std::make_shared< Readers::PVTUReader >( fileName );
   else if( format == "pvti" )
      return std::make_shared< Readers::PVTIReader >( fileName );
   else if( format == "fpma" )
      return std::make_shared< Readers::FPMAReader >( fileName );

   if( fileFormat == "auto" )
      std::cerr << "File '" << fileName << "' has unsupported format (based on the file extension): " << format << ".";
   else
      std::cerr << "Unsupported fileFormat parameter: " << fileFormat << ".";
   std::cerr << " Supported formats are 'ng', 'vtk', 'vtu', 'vti', 'pvtu' and 'pvti'." << std::endl;
   return nullptr;
}

}  // namespace Readers
}  // namespace Meshes
}  // namespace noa::TNL
