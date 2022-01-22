// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <noa/3rdparty/TNL/Meshes/Grid.h>

namespace noaTNL {
namespace Functions {

template< int, typename > class VectorField;
template< typename, int, typename > class MeshFunction;

template< typename VectorField >
class VectorFieldGnuplotWriter
{
public:
   static bool write( const VectorField& function,
                      std::ostream& str );
};

/***
 * 1D grids cells
 */
template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          int VectorFieldSize >
class VectorFieldGnuplotWriter< VectorField< VectorFieldSize, MeshFunction< Meshes::Grid< 1, MeshReal, Device, MeshIndex >, 1, Real > > >
{
public:
   using MeshType = Meshes::Grid< 1, MeshReal, Device, MeshIndex >;
   using RealType = Real;
   using VectorFieldType = Functions::VectorField< VectorFieldSize, MeshFunction< MeshType, 1, RealType > >;

   static bool write( const VectorFieldType& function,
                      std::ostream& str );
};

/***
 * 1D grids vertices
 */
template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          int VectorFieldSize >
class VectorFieldGnuplotWriter< VectorField< VectorFieldSize, MeshFunction< Meshes::Grid< 1, MeshReal, Device, MeshIndex >, 0, Real > > >
{
public:
   using MeshType = Meshes::Grid< 1, MeshReal, Device, MeshIndex >;
   using RealType = Real;
   using VectorFieldType = Functions::VectorField< VectorFieldSize, MeshFunction< MeshType, 0, RealType > >;

   static bool write( const VectorFieldType& function,
                      std::ostream& str );
};


/***
 * 2D grids cells
 */
template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          int VectorFieldSize >
class VectorFieldGnuplotWriter< VectorField< VectorFieldSize, MeshFunction< Meshes::Grid< 2, MeshReal, Device, MeshIndex >, 2, Real > > >
{
public:
   using MeshType = Meshes::Grid< 2, MeshReal, Device, MeshIndex >;
   using RealType = Real;
   using VectorFieldType = Functions::VectorField< VectorFieldSize, MeshFunction< MeshType, 2, RealType > >;

   static bool write( const VectorFieldType& function,
                      std::ostream& str );
};

/***
 * 2D grids faces
 */
template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          int VectorFieldSize >
class VectorFieldGnuplotWriter< VectorField< VectorFieldSize, MeshFunction< Meshes::Grid< 2, MeshReal, Device, MeshIndex >, 1, Real > > >
{
public:
   using MeshType = Meshes::Grid< 2, MeshReal, Device, MeshIndex >;
   using RealType = Real;
   using VectorFieldType = Functions::VectorField< VectorFieldSize, MeshFunction< MeshType, 1, RealType > >;

   static bool write( const VectorFieldType& function,
                      std::ostream& str );
};

/***
 * 2D grids vertices
 */
template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          int VectorFieldSize >
class VectorFieldGnuplotWriter< VectorField< VectorFieldSize, MeshFunction< Meshes::Grid< 2, MeshReal, Device, MeshIndex >, 0, Real > > >
{
public:
   using MeshType = Meshes::Grid< 2, MeshReal, Device, MeshIndex >;
   using RealType = Real;
   using VectorFieldType = Functions::VectorField< VectorFieldSize, MeshFunction< MeshType, 0, RealType > >;

   static bool write( const VectorFieldType& function,
                      std::ostream& str );
};


/***
 * 3D grids cells
 */
template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          int VectorFieldSize >
class VectorFieldGnuplotWriter< VectorField< VectorFieldSize, MeshFunction< Meshes::Grid< 3, MeshReal, Device, MeshIndex >, 3, Real > > >
{
public:
   using MeshType = Meshes::Grid< 3, MeshReal, Device, MeshIndex >;
   using RealType = Real;
   using VectorFieldType = Functions::VectorField< VectorFieldSize, MeshFunction< MeshType, 3, RealType > >;

   static bool write( const VectorFieldType& function,
                      std::ostream& str );
};

/***
 * 3D grids faces
 */
template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          int VectorFieldSize >
class VectorFieldGnuplotWriter< VectorField< VectorFieldSize, MeshFunction< Meshes::Grid< 3, MeshReal, Device, MeshIndex >, 2, Real > > >
{
public:
   using MeshType = Meshes::Grid< 3, MeshReal, Device, MeshIndex >;
   using RealType = Real;
   using VectorFieldType = Functions::VectorField< VectorFieldSize, MeshFunction< MeshType, 2, RealType > >;

   static bool write( const VectorFieldType& function,
                      std::ostream& str );
};

/***
 * 3D grids vertices
 */
template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          int VectorFieldSize >
class VectorFieldGnuplotWriter< VectorField< VectorFieldSize, MeshFunction< Meshes::Grid< 3, MeshReal, Device, MeshIndex >, 0, Real > > >
{
public:
   using MeshType = Meshes::Grid< 3, MeshReal, Device, MeshIndex >;
   using RealType = Real;
   using VectorFieldType = Functions::VectorField< VectorFieldSize, MeshFunction< MeshType, 0, RealType > >;

   static bool write( const VectorFieldType& function,
                      std::ostream& str );
};

} // namespace Functions
} // namespace noaTNL

#include <noa/3rdparty/TNL/Functions/VectorFieldGnuplotWriter_impl.h>
