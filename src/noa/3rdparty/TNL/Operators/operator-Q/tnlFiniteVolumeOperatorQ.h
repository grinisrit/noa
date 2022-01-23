// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <noa/3rdparty/TNL/Containers/Vector.h>
#include <noa/3rdparty/TNL/Containers/SharedVector.h>
#include <noa/3rdparty/TNL/Meshes/Grid.h>

namespace noa::TNL {
namespace Operators {

template< typename Mesh,
          typename Real = typename Mesh::RealType,
          typename Index = typename Mesh::GlobalIndexType,
          int Precomputation = 0 > 
class tnlFiniteVolumeOperatorQ
{

};


template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
class tnlFiniteVolumeOperatorQ< Meshes::Grid< 1,MeshReal, Device, MeshIndex >, Real, Index, 0 >
{
   public: 
   
   typedef Meshes::Grid< 1, MeshReal, Device, MeshIndex > MeshType;
   typedef typename MeshType::CoordinatesType CoordinatesType;
   typedef Real RealType;
   typedef Device DeviceType;
   typedef Index IndexType;

   template< typename Vector >
   IndexType bind( Vector& u) 
   { return 0; }

   __cuda_callable__
   void update( const MeshType& mesh, const RealType& time ) 
   {}
   
   template< typename MeshEntity, typename Vector >
   __cuda_callable__
   Real operator()( const MeshType& mesh,
          const MeshEntity& entity,
          const Vector& u,
          const Real& time,
          const IndexType& dx = 0, 
          const IndexType& dy = 0,
          const IndexType& dz = 0 ) const;
          
   bool setEps(const Real& eps);
      
   private:
   
      template< typename MeshEntity, typename Vector, int AxeX = 0, int AxeY = 0, int AxeZ = 0 >
      __cuda_callable__
      Real 
      boundaryDerivative( 
         const MeshEntity& entity,
         const Vector& u,
         const Real& time,
         const IndexType& dx = 0, 
         const IndexType& dy = 0,
         const IndexType& dz = 0 ) const;

      RealType eps;
};


template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
class tnlFiniteVolumeOperatorQ< Meshes::Grid< 2,MeshReal, Device, MeshIndex >, Real, Index, 0 >
{
   public: 
   
   typedef Meshes::Grid< 2, MeshReal, Device, MeshIndex > MeshType;
   typedef typename MeshType::CoordinatesType CoordinatesType;
   typedef Real RealType;
   typedef Device DeviceType;
   typedef Index IndexType;

   template< typename Vector >
   IndexType bind( Vector& u)
   { return 0; }

   __cuda_callable__
   void update( const MeshType& mesh, const RealType& time )
   {}   
   
   template< typename MeshEntity, typename Vector >
   __cuda_callable__
   Real operator()( 
      const MeshEntity& entity,
      const Vector& u,
      const Real& time,
      const IndexType& dx = 0, 
      const IndexType& dy = 0,
      const IndexType& dz = 0 ) const;
        
   bool setEps(const Real& eps);
   
   private:

   template< typename MeshEntity, typename Vector, int AxeX = 0, int AxeY = 0, int AxeZ = 0 >
   __cuda_callable__
   Real boundaryDerivative( const MeshType& mesh,
          const MeshEntity& entity,
          const Vector& u,
          const Real& time,
          const IndexType& dx = 0, 
          const IndexType& dy = 0,
          const IndexType& dz = 0 ) const;
   
   RealType eps;
};


template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
class tnlFiniteVolumeOperatorQ< Meshes::Grid< 3,MeshReal, Device, MeshIndex >, Real, Index, 0 >
{
   public: 
   
   typedef Meshes::Grid< 3, MeshReal, Device, MeshIndex > MeshType;
   typedef typename MeshType::CoordinatesType CoordinatesType;
   typedef Real RealType;
   typedef Device DeviceType;
   typedef Index IndexType;

   template< typename Vector >
   IndexType bind( Vector& u)
   { return 0; }

   __cuda_callable__
   void update( const MeshType& mesh, const RealType& time )
   {}
   
   template< typename MeshEntity, typename Vector >
   __cuda_callable__
   Real operator()(
          const MeshEntity& entity,
          const Vector& u,
          const Real& time,
          const IndexType& dx = 0, 
          const IndexType& dy = 0,
          const IndexType& dz = 0 ) const;
        
   bool setEps(const Real& eps);
   
   private:

   template< typename MeshEntity, typename Vector, int AxeX = 0, int AxeY = 0, int AxeZ = 0 >
   __cuda_callable__
   Real boundaryDerivative( const MeshType& mesh,
          const MeshEntity& entity,
          const Vector& u,
          const Real& time,
          const IndexType& dx = 0, 
          const IndexType& dy = 0,
          const IndexType& dz = 0 ) const;
   
   RealType eps;
};


template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
class tnlFiniteVolumeOperatorQ< Meshes::Grid< 1,MeshReal, Device, MeshIndex >, Real, Index, 1 >
{
   public: 
   
   typedef Meshes::Grid< 1, MeshReal, Device, MeshIndex > MeshType;
   typedef typename MeshType::CoordinatesType CoordinatesType;
   typedef Real RealType;
   typedef Device DeviceType;
   typedef Index IndexType;

   template< typename Vector >
   Index bind( Vector& u);

   __cuda_callable__
   void update( const MeshType& mesh, const RealType& time );
   
   template< typename MeshEntity, typename Vector >
   __cuda_callable__
   Real operator()( const MeshType& mesh,
          const MeshEntity& entity,
          const Vector& u,
          const Real& time,
          const IndexType& dx = 0, 
          const IndexType& dy = 0,
          const IndexType& dz = 0 ) const;
   
   bool setEps(const Real& eps);
   
   private:
   
      template< typename MeshEntity, typename Vector, int AxeX = 0, int AxeY = 0, int AxeZ = 0 >
      __cuda_callable__
      Real boundaryDerivative( const MeshType& mesh,
             const MeshEntity& entity,
             const Vector& u,
             const Real& time,
             const IndexType& dx = 0, 
             const IndexType& dy = 0,
             const IndexType& dz = 0 ) const;    

      SharedVector< RealType, DeviceType, IndexType > u;
      Vector< RealType, DeviceType, IndexType> q;
      RealType eps;
};


template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
class tnlFiniteVolumeOperatorQ< Meshes::Grid< 2,MeshReal, Device, MeshIndex >, Real, Index, 1 >
{
   public: 
   
   typedef Meshes::Grid< 2, MeshReal, Device, MeshIndex > MeshType;
   typedef typename MeshType::CoordinatesType CoordinatesType;
   typedef Real RealType;
   typedef Device DeviceType;
   typedef Index IndexType;
   typedef SharedVector< RealType, DeviceType, IndexType > DofVectorType;

   template< typename Vector >
   Index bind( Vector& u);

   __cuda_callable__
   void update( const MeshType& mesh, const RealType& time ); 
   
   template< typename MeshEntity, typename Vector >
   __cuda_callable__
   Real operator()( const MeshType& mesh,
          const MeshEntity& entity,
          const Vector& u,
          const Real& time,
          const IndexType& dx = 0, 
          const IndexType& dy = 0,
          const IndexType& dz = 0 ) const;
          
   bool setEps(const Real& eps);
   
   private:
   
   template< typename MeshEntity, typename Vector, int AxeX = 0, int AxeY = 0, int AxeZ = 0 >
   __cuda_callable__
   Real boundaryDerivative( const MeshType& mesh,
          const IndexType cellIndex,
          const MeshEntity& entity,
          const Vector& u,
          const Real& time,
          const IndexType& dx = 0, 
          const IndexType& dy = 0,
          const IndexType& dz = 0 ) const;
       
   SharedVector< RealType, DeviceType, IndexType > u;
   Vector< RealType, DeviceType, IndexType> q;
   RealType eps;
};


template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
class tnlFiniteVolumeOperatorQ< Meshes::Grid< 3,MeshReal, Device, MeshIndex >, Real, Index, 1 >
{
   public: 
   
   typedef Meshes::Grid< 3, MeshReal, Device, MeshIndex > MeshType;
   typedef typename MeshType::CoordinatesType CoordinatesType;
   typedef Real RealType;
   typedef Device DeviceType;
   typedef Index IndexType;

   template< typename Vector >
   Index bind( Vector& u);

   __cuda_callable__
   void update( const MeshType& mesh, const RealType& time );
   
   template< typename MeshEntity, typename Vector >
   __cuda_callable__
   Real operator()( const MeshType& mesh,
          const IndexType cellIndex,
          const MeshEntity& entity,
          const Vector& u,
          const Real& time,
          const IndexType& dx = 0, 
          const IndexType& dy = 0,
          const IndexType& dz = 0 ) const;
          
   bool setEps(const Real& eps);
   
   private:
   
      template< typename MeshEntity, typename Vector, int AxeX = 0, int AxeY = 0, int AxeZ = 0 >
      __cuda_callable__
      Real boundaryDerivative( const MeshType& mesh,
             const IndexType cellIndex,
             const MeshEntity& entity,
             const Vector& u,
             const Real& time,
             const IndexType& dx = 0, 
             const IndexType& dy = 0,
             const IndexType& dz = 0 ) const;

      SharedVector< RealType, DeviceType, IndexType > u;
      Vector< RealType, DeviceType, IndexType> q;
      RealType eps;
};

} // namespace Operators
} // namespace noa::TNL

#include <noa/3rdparty/TNL/Operators/operator-Q/tnlFiniteVolumeOperatorQ_impl.h>
