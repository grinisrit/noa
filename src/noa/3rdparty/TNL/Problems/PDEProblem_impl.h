// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <noa/3rdparty/TNL/Problems/PDEProblem.h>
#include <noa/3rdparty/TNL/Meshes/DistributedMeshes/SubdomainOverlapsGetter.h>

namespace noaTNL {
namespace Problems {

template< typename Mesh,
          typename Real,
          typename Device,
          typename Index >
String
PDEProblem< Mesh, Real, Device, Index >::
getPrologHeader() const
{
   return String( "General PDE Problem" );
}

template< typename Mesh,
          typename Real,
          typename Device,
          typename Index >
void
PDEProblem< Mesh, Real, Device, Index >::
writeProlog( Logger& logger, const Config::ParameterContainer& parameters ) const
{
}

template< typename Mesh,
          typename Real,
          typename Device,
          typename Index >
bool
PDEProblem< Mesh, Real, Device, Index >::
writeEpilog( Logger& logger ) const
{
   return true;
}

template< typename Mesh,
          typename Real,
          typename Device,
          typename Index >
typename PDEProblem< Mesh, Real, Device, Index >::IndexType
PDEProblem< Mesh, Real, Device, Index >::
subdomainOverlapSize()
{
   return 1;
}

template< typename Mesh,
          typename Real,
          typename Device,
          typename Index >
void
PDEProblem< Mesh, Real, Device, Index >::
getSubdomainOverlaps( const Config::ParameterContainer& parameters,
                      const String& prefix,
                      const MeshType& mesh,
                      SubdomainOverlapsType& lower,
                      SubdomainOverlapsType& upper )
{
   using namespace Meshes::DistributedMeshes;
   SubdomainOverlapsGetter< MeshType >::getOverlaps( mesh.getDistributedMesh(), lower, upper, this->subdomainOverlapSize() );
}

template< typename Mesh,
          typename Real,
          typename Device,
          typename Index >
void
PDEProblem< Mesh, Real, Device, Index >::
setMesh( MeshPointer& meshPointer)
{
   this->meshPointer = meshPointer;
}

template< typename Mesh,
          typename Real,
          typename Device,
          typename Index >
void
PDEProblem< Mesh, Real, Device, Index >::
setMesh( DistributedMeshPointer& distributedMeshPointer)
{
   this->distributedMeshPointer = distributedMeshPointer;
   // FIXME: DistributedGrid does not have a SharedPointer of the local grid,
   // the interface is fucked up (it should not require us to put SharedPointer everywhere)
   *meshPointer = distributedMeshPointer->getLocalMesh();
}

template< typename Mesh,
          typename Real,
          typename Device,
          typename Index >
const typename PDEProblem< Mesh, Real, Device, Index >::MeshPointer&
PDEProblem< Mesh, Real, Device, Index >::getMesh() const
{
   return this->meshPointer;
}

template< typename Mesh,
          typename Real,
          typename Device,
          typename Index >
typename PDEProblem< Mesh, Real, Device, Index >::MeshPointer&
PDEProblem< Mesh, Real, Device, Index >::getMesh()
{
   return this->meshPointer;
}

template< typename Mesh,
          typename Real,
          typename Device,
          typename Index >
const typename PDEProblem< Mesh, Real, Device, Index >::DistributedMeshPointer&
PDEProblem< Mesh, Real, Device, Index >::getDistributedMesh() const
{
   return this->distributedMeshPointer;
}

template< typename Mesh,
          typename Real,
          typename Device,
          typename Index >
typename PDEProblem< Mesh, Real, Device, Index >::DistributedMeshPointer&
PDEProblem< Mesh, Real, Device, Index >::getDistributedMesh()
{
   return this->distributedMeshPointer;
}

template< typename Mesh,
          typename Real,
          typename Device,
          typename Index >
void
PDEProblem< Mesh, Real, Device, Index >::
setCommonData( CommonDataPointer& commonData )
{
   this->commonDataPointer = commonData;
}

template< typename Mesh,
          typename Real,
          typename Device,
          typename Index >
const typename PDEProblem< Mesh, Real, Device, Index >::CommonDataPointer&
PDEProblem< Mesh, Real, Device, Index >::
getCommonData() const
{
   return this->commonDataPointer;
}

template< typename Mesh,
          typename Real,
          typename Device,
          typename Index >
typename PDEProblem< Mesh, Real, Device, Index >::CommonDataPointer&
PDEProblem< Mesh, Real, Device, Index >::
getCommonData()
{
   return this->commonDataPointer;
}

template< typename Mesh,
          typename Real,
          typename Device,
          typename Index >
bool
PDEProblem< Mesh, Real, Device, Index >::
preIterate( const RealType& time,
            const RealType& tau,
            DofVectorPointer& dofs )
{
   return true;
}

template< typename Mesh,
          typename Real,
          typename Device,
          typename Index >
    template< typename Matrix >
void
PDEProblem< Mesh, Real, Device, Index >::
saveFailedLinearSystem( const Matrix& matrix,
                        const DofVectorType& dofs,
                        const DofVectorType& rhs ) const
{
    matrix.save( "failed-matrix.tnl" );
    File( "failed-dof.vec.tnl", std::ios_base::out ) << dofs;
    File( "failed-rhs.vec.tnl", std::ios_base::out ) << rhs;
    std::cerr << "The linear system has been saved to failed-{matrix,dof.vec,rhs.vec}.tnl" << std::endl;
}

template< typename Mesh,
          typename Real,
          typename Device,
          typename Index >
bool
PDEProblem< Mesh, Real, Device, Index >::
postIterate( const RealType& time,
             const RealType& tau,
             DofVectorPointer& dofs )
{
   return true;
}

template< typename Mesh,
          typename Real,
          typename Device,
          typename Index >
Solvers::SolverMonitor*
PDEProblem< Mesh, Real, Device, Index >::
getSolverMonitor()
{
   return 0;
}

} // namespace Problems
} // namespace noaTNL
