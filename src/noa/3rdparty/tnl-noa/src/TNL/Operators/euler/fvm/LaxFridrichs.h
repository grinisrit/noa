// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <noa/3rdparty/tnl-noa/src/TNL/Containers/SharedVector.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Meshes/Grid.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Meshes/tnlIdenticalGridGeometry.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Operators/gradient/tnlCentralFDMGradient.h>

namespace noa::TNL {
namespace Operators {

template< typename Mesh, typename PressureGradient = tnlCentralFDMGradient< Mesh > >
class LaxFridrichs
{};

template< typename Real,
          typename Device,
          typename Index,
          typename PressureGradient,
          template< int, typename, typename, typename >
          class GridGeometry >
class LaxFridrichs< Meshes::Grid< 2, Real, Device, Index, GridGeometry >, PressureGradient >
{
public:
   typedef Meshes::Grid< 2, Real, Device, Index, GridGeometry > MeshType;
   typedef Real RealType;
   typedef Device DeviceType;
   typedef Index IndexType;
   typedef typename MeshType ::PointType PointType;
   typedef typename MeshType ::CoordinatesType CoordinatesType;

   LaxFridrichs();

   void
   getExplicitUpdate( const IndexType centralVolume,
                      RealType& rho_t,
                      RealType& rho_u1_t,
                      RealType& rho_u2_t,
                      const RealType& tau ) const;

   void
   getExplicitUpdate( const IndexType centralVolume,
                      RealType& rho_t,
                      RealType& rho_u1_t,
                      RealType& rho_u2_t,
                      RealType& e_t,
                      const RealType& tau ) const;

   void
   setRegularization( const RealType& epsilon );

   void
   setViscosityCoefficient( const RealType& v );

   void
   bindMesh( const MeshType& mesh );

   const MeshType&
   getMesh() const;

   template< typename Vector >
   void
   setRho( Vector& rho );  // TODO: add const

   template< typename Vector >
   void
   setRhoU1( Vector& rho_u1 );  // TODO: add const

   template< typename Vector >
   void
   setRhoU2( Vector& rho_u2 );  // TODO: add const

   template< typename Vector >
   void
   setE( Vector& e );  // TODO: add const

   template< typename Vector >
   void
   setPressureGradient( Vector& grad_p );  // TODO: add const

protected:
   RealType
   regularize( const RealType& r ) const;

   RealType regularizeEps, viscosityCoefficient;

   const MeshType* mesh;

   const PressureGradient* pressureGradient;

   SharedVector< RealType, DeviceType, IndexType > rho, rho_u1, rho_u2, e;
};

template< typename Real, typename Device, typename Index, typename PressureGradient >
class LaxFridrichs< Meshes::Grid< 2, Real, Device, Index, tnlIdenticalGridGeometry >, PressureGradient >
{
public:
   typedef Meshes::Grid< 2, Real, Device, Index, tnlIdenticalGridGeometry > MeshType;
   typedef Real RealType;
   typedef Device DeviceType;
   typedef Index IndexType;
   typedef typename MeshType ::PointType PointType;
   typedef typename MeshType ::CoordinatesType CoordinatesType;

   LaxFridrichs();

   void
   getExplicitUpdate( const IndexType centralVolume,
                      RealType& rho_t,
                      RealType& rho_u1_t,
                      RealType& rho_u2_t,
                      const RealType& tau ) const;

   void
   getExplicitUpdate( const IndexType centralVolume,
                      RealType& rho_t,
                      RealType& rho_u1_t,
                      RealType& rho_u2_t,
                      RealType& e_t,
                      const RealType& tau ) const;

   void
   setRegularization( const RealType& epsilon );

   void
   setViscosityCoefficient( const RealType& v );

   void
   bindMesh( const MeshType& mesh );

   template< typename Vector >
   void
   setRho( Vector& rho );  // TODO: add const

   template< typename Vector >
   void
   setRhoU1( Vector& rho_u1 );  // TODO: add const

   template< typename Vector >
   void
   setRhoU2( Vector& rho_u2 );  // TODO: add const

   template< typename Vector >
   void
   setE( Vector& e );  // TODO: add const

   template< typename Vector >
   void
   setP( Vector& p );  // TODO: add const

   template< typename Vector >
   void
   setPressureGradient( Vector& grad_p );  // TODO: add const

protected:
   RealType
   regularize( const RealType& r ) const;

   RealType regularizeEps, viscosityCoefficient;

   const MeshType* mesh;

   const PressureGradient* pressureGradient;

   SharedVector< RealType, DeviceType, IndexType > rho, rho_u1, rho_u2, energy, p;
};

}  // namespace Operators
}  // namespace noa::TNL

#include <noa/3rdparty/tnl-noa/src/TNL/implementation/operators/euler/fvm/LaxFridrichs_impl.h>
