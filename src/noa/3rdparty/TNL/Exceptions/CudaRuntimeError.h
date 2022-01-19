// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

// Implemented by: Jakub Klinkovsky

#pragma once

#include "CudaSupportMissing.h"

namespace TNL {
namespace Exceptions {

#ifdef HAVE_CUDA
   using CudaStatusType = cudaError;
#else
   using CudaStatusType = int;
#endif

class CudaRuntimeError
   : public std::runtime_error
{
public:
   CudaRuntimeError( CudaStatusType error_code )
   : std::runtime_error( "CUDA ERROR " + std::to_string( (int) error_code ) + " (" + name( error_code ) + "): "
                         + description( error_code ) + "." ),
     code_( error_code )
   {}

   CudaRuntimeError( CudaStatusType error_code, const std::string& what_arg )
   : std::runtime_error( "CUDA ERROR " + std::to_string( (int) error_code ) + " (" + name( error_code ) + "): "
                         + description( error_code ) + ".\nDetails: " + what_arg ),
     code_(error_code)
   {}

   CudaRuntimeError( CudaStatusType error_code, const char* file_name, int line )
   : std::runtime_error( "CUDA ERROR " + std::to_string( (int) error_code ) + " (" + name( error_code ) + "): "
                         + description( error_code ) + ".\nSource: line " + std::to_string( line )
                         + " in " + file_name + ": " + description( error_code ) ),
     code_(error_code)
   {}

   CudaStatusType code() const
   {
      return code_;
   }

private:
   static std::string name( CudaStatusType error_code )
   {
#ifdef HAVE_CUDA
      return cudaGetErrorName( error_code );
#else
      throw CudaSupportMissing();
#endif
   }

   static std::string description( CudaStatusType error_code )
   {
#ifdef HAVE_CUDA
      return cudaGetErrorString( error_code );
#else
      throw CudaSupportMissing();
#endif
   }

   CudaStatusType code_;
};

} // namespace Exceptions
} // namespace TNL
