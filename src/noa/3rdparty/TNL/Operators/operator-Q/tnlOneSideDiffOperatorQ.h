// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <noa/3rdparty/TNL/Meshes/Grid.h>

namespace noa::TNL {
namespace Operators {

template< typename Mesh,
          typename Real = typename Mesh::RealType,
          typename Index = typename Mesh::GlobalIndexType > 
class tnlOneSideDiffOperatorQ
{
};

template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
class tnlOneSideDiffOperatorQ< Meshes::Grid< 1,MeshReal, Device, MeshIndex >, Real, Index >
{
   public: 
   
   typedef Meshes::Grid< 1, MeshReal, Device, MeshIndex > MeshType;
   typedef typename MeshType::CoordinatesType CoordinatesType;
   typedef Real RealType;
   typedef Device DeviceType;
   typedef Index IndexType;

   template< typename MeshFunction, typename MeshEntity >
   __cuda_callable__
   Real operator()( const MeshFunction& u,
                    const MeshEntity& entity,
                    const Real& time = 0.0 ) const;
      
   template< typename MeshFunction, typename MeshEntity >
   __cuda_callable__
   Real getValueStriped( const MeshFunction& u,
                         const MeshEntity& entity,   
                         const Real& time = 0.0 ) const;
          
   void setEps(const Real& eps);
      
   private:
   
   RealType eps, epsSquare;
};


template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
class tnlOneSideDiffOperatorQ< Meshes::Grid< 2,MeshReal, Device, MeshIndex >, Real, Index >
{
   public: 
   
   typedef Meshes::Grid< 2, MeshReal, Device, MeshIndex > MeshType;
   typedef typename MeshType::CoordinatesType CoordinatesType;
   typedef Real RealType;
   typedef Device DeviceType;
   typedef Index IndexType;

   template< typename MeshFunction, typename MeshEntity >
   __cuda_callable__
   Real operator()( const MeshFunction& u,
                    const MeshEntity& entity,
                    const Real& time = 0.0 ) const;

   
   template< typename MeshFunction, typename MeshEntity >
   __cuda_callable__
   Real getValueStriped( const MeshFunction& u,
                         const MeshEntity& entity,          
                         const Real& time = 0.0 ) const;
        
   void setEps( const Real& eps );
   
   private:
   
   RealType eps, epsSquare;
};


template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
class tnlOneSideDiffOperatorQ< Meshes::Grid< 3, MeshReal, Device, MeshIndex >, Real, Index >
{
   public: 
   
   typedef Meshes::Grid< 3, MeshReal, Device, MeshIndex > MeshType;
   typedef typename MeshType::CoordinatesType CoordinatesType;
   typedef Real RealType;
   typedef Device DeviceType;
   typedef Index IndexType;

   template< typename MeshFunction, typename MeshEntity >
   __cuda_callable__
   Real operator()( const MeshFunction& u,
                    const MeshEntity& entity,
                    const Real& time = 0.0 ) const;
   
   template< typename MeshFunction, typename MeshEntity >
   __cuda_callable__
   Real getValueStriped( const MeshFunction& u,
                         const MeshEntity& entity,          
                         const Real& time ) const;
        
   void setEps(const Real& eps);
   
   private:
   
   RealType eps, epsSquare;
};

} // namespace Operators
} // namespace noa::TNL

#include <noa/3rdparty/TNL/Operators/operator-Q/tnlOneSideDiffOperatorQ_impl.h>
