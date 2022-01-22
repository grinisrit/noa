// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <noa/3rdparty/TNL/Meshes/Grid.h>
#include <noa/3rdparty/TNL/Meshes/Mesh.h>

namespace noaTNL {
namespace Solvers {
namespace PDE {   

template< typename Mesh, typename Real >
class MeshDependentTimeSteps
{
};

template< int Dimension,
          typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real >
class MeshDependentTimeSteps< noaTNL::Meshes::Grid< Dimension, MeshReal, Device, MeshIndex >, Real >
{
public:
   using MeshType = noaTNL::Meshes::Grid< Dimension, MeshReal, Device, MeshIndex >;

   bool setTimeStepOrder( const Real& timeStepOrder )
   {
      if( timeStepOrder < 0 ) {
         std::cerr << "The time step order for PDESolver must be zero or positive value." << std::endl;
         return false;
      }
      this->timeStepOrder = timeStepOrder;
      return true;
   }

   const Real& getTimeStepOrder() const
   {
      return timeStepOrder;
   }

   Real getRefinedTimeStep( const MeshType& mesh, const Real& timeStep )
   {
      return timeStep * std::pow( mesh.getSmallestSpaceStep(), this->timeStepOrder );
   }

protected:
   Real timeStepOrder = 0.0;
};

template< typename MeshConfig,
          typename Device,
          typename Real >
class MeshDependentTimeSteps< noaTNL::Meshes::Mesh< MeshConfig, Device >, Real >
{
public:
   using MeshType = noaTNL::Meshes::Mesh< MeshConfig >;

   bool setTimeStepOrder( const Real& timeStepOrder )
   {
      if( timeStepOrder != 0.0 ) {
         std::cerr << "Mesh-dependent time stepping is not available on unstructured meshes, so the time step order must be 0." << std::endl;
         return false;
      }
      this->timeStepOrder = timeStepOrder;
      return true;
   }

   const Real& getTimeStepOrder() const
   {
      return timeStepOrder;
   }

   Real getRefinedTimeStep( const MeshType& mesh, const Real& timeStep )
   {
      return timeStep;
   }

protected:
   Real timeStepOrder = 0.0;
};

} // namespace PDE
} // namespace Solvers
} // namespace noaTNL
