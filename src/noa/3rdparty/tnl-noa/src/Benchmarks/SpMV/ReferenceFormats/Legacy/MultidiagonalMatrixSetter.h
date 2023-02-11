#pragma once

#include <TNL/Meshes/Grid.h>
#include "Multidiagonal.h"

namespace TNL {
    namespace Benchmarks {
        namespace SpMV {
            namespace ReferenceFormats {
               namespace Legacy {

template< typename MeshType >
class MultidiagonalMatrixSetter
{
};

template< typename MeshReal,
          typename Device,
          typename MeshIndex >
class MultidiagonalMatrixSetter< Meshes::Grid< 1, MeshReal, Device, MeshIndex > >
{
   public:
      typedef MeshReal MeshRealType;
      typedef Device DeviceType;
      typedef MeshIndex MeshIndexType;
      typedef Meshes::Grid< 1, MeshReal, Device, MeshIndex > MeshType;
      typedef typename MeshType::CoordinatesType CoordinatesType;
      enum { Dimension = 1 };

      template< typename Real, typename Index >
      static bool setupMatrix( const MeshType& mesh,
                               Multidiagonal< Real, Device, Index >& matrix,
                               int stencilSize = 1,
                               bool crossStencil = false );

};

template< typename MeshReal,
          typename Device,
          typename MeshIndex >
class MultidiagonalMatrixSetter< Meshes::Grid< 2, MeshReal, Device, MeshIndex > >
{
   public:
      typedef MeshReal MeshRealType;
      typedef Device DeviceType;
      typedef MeshIndex MeshIndexType;
      typedef Meshes::Grid< 2, MeshReal, Device, MeshIndex > MeshType;
      typedef typename MeshType::CoordinatesType CoordinatesType;
      enum { Dimension = 2 };

      template< typename Real, typename Index >
      static bool setupMatrix( const MeshType& mesh,
                               Multidiagonal< Real, Device, Index >& matrix,
                               int stencilSize = 1,
                               bool crossStencil = false );
};

template< typename MeshReal,
          typename Device,
          typename MeshIndex >
class MultidiagonalMatrixSetter< Meshes::Grid< 3, MeshReal, Device, MeshIndex > >
{
   public:
      typedef MeshReal MeshRealType;
      typedef Device DeviceType;
      typedef MeshIndex MeshIndexType;
      typedef Meshes::Grid< 3, MeshReal, Device, MeshIndex > MeshType;
      typedef typename MeshType::CoordinatesType CoordinatesType;
      enum { Dimension = 3 };

      template< typename Real, typename Index >
      static bool setupMatrix( const MeshType& mesh,
                               Multidiagonal< Real, Device, Index >& matrix,
                               int stencilSize = 1,
                               bool crossStencil = false );
};

               } //namespace Legacy
            } //namespace ReferenceFormats
        } //namespace SpMV
    } //namespace Benchmarks
} // namespace TNL

#include "MultidiagonalMatrixSetter_impl.h"
