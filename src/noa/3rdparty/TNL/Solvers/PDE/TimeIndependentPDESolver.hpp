// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/

#pragma once

#include <noa/3rdparty/TNL/Solvers/PDE/TimeIndependentPDESolver.h>
#include <noa/3rdparty/TNL/Meshes/TypeResolver/resolveMeshType.h>
#include <noa/3rdparty/TNL/Meshes/TypeResolver/resolveDistributedMeshType.h>
#include <noa/3rdparty/TNL/MPI/Wrappers.h>

namespace noaTNL {
namespace Solvers {
namespace PDE {


template< typename Problem >
TimeIndependentPDESolver< Problem >::
TimeIndependentPDESolver()
: problem( 0 )
{
}

template< typename Problem >
void
TimeIndependentPDESolver< Problem >::
configSetup( Config::ConfigDescription& config,
             const String& prefix )
{
}

template< typename Problem >
bool
TimeIndependentPDESolver< Problem >::
setup( const Config::ParameterContainer& parameters,
       const String& prefix )
{
   /////
   // Load the mesh from the mesh file
   //
   const String& meshFile = parameters.getParameter< String >( "mesh" );
   const String& meshFileFormat = parameters.getParameter< String >( "mesh-format" );
   if( MPI::GetSize() > 1 ) {
      if( ! Meshes::loadDistributedMesh( *distributedMeshPointer, meshFile, meshFileFormat ) )
         return false;
      problem->setMesh( distributedMeshPointer );
   }
   else {
      if( ! Meshes::loadMesh( *meshPointer, meshFile, meshFileFormat ) )
         return false;
      problem->setMesh( meshPointer );
   }

   /****
    * Set-up common data
    */
   if( ! this->commonDataPointer->setup( parameters ) )
   {
      std::cerr << "The problem common data initiation failed!" << std::endl;
      return false;
   }
   problem->setCommonData( this->commonDataPointer );

   /****
    * Setup the problem
    */
   if( ! problem->setup( parameters, prefix ) )
   {
      std::cerr << "The problem initiation failed!" << std::endl;
      return false;
   }

   /****
    * Set DOFs (degrees of freedom)
    */
   TNL_ASSERT_GT( problem->getDofs(), 0, "number of DOFs must be positive" );
   this->dofs->setSize( problem->getDofs() );
   this->dofs->setValue( 0.0 );
   this->problem->bindDofs( this->dofs );


   /***
    * Set-up the initial condition
    */
   std::cout << "Setting up the initial condition ... ";
   if( ! this->problem->setInitialCondition( parameters, this->dofs ) )
      return false;
   std::cout << " [ OK ]" << std::endl;

   return true;
}

template< typename Problem >
bool
TimeIndependentPDESolver< Problem >::
writeProlog( Logger& logger,
             const Config::ParameterContainer& parameters )
{
   logger.writeHeader( problem->getPrologHeader() );
   problem->writeProlog( logger, parameters );
   logger.writeSeparator();
   if( MPI::GetSize() > 1 )
      distributedMeshPointer->writeProlog( logger );
   else
      meshPointer->writeProlog( logger );
   logger.writeSeparator();
   const String& solverName = parameters. getParameter< String >( "discrete-solver" );
   logger.writeParameter< String >( "Discrete solver:", "discrete-solver", parameters );
   if( solverName == "sor" )
      logger.writeParameter< double >( "Omega:", "sor-omega", parameters, 1 );
   if( solverName == "gmres" )
      logger.writeParameter< int >( "Restarting:", "gmres-restarting", parameters, 1 );
   logger.writeParameter< double >( "Convergence residue:", "convergence-residue", parameters );
   logger.writeParameter< double >( "Divergence residue:", "divergence-residue", parameters );
   logger.writeParameter< int >( "Maximal number of iterations:", "max-iterations", parameters );
   logger.writeParameter< int >( "Minimal number of iterations:", "min-iterations", parameters );
   logger.writeSeparator();
   return BaseType::writeProlog( logger, parameters );
}

template< typename Problem >
void
TimeIndependentPDESolver< Problem >::
setProblem( ProblemType& problem )
{
   this->problem = &problem;
}

template< typename Problem >
bool
TimeIndependentPDESolver< Problem >::
solve()
{
   TNL_ASSERT_TRUE( problem, "No problem was set in tnlPDESolver." );

   this->computeTimer->reset();
   this->computeTimer->start();
   if( ! this->problem->solve( this->dofs ) )
   {
      this->computeTimer->stop();
      return false;
   }
   this->computeTimer->stop();
   return true;
}

template< typename Problem >
bool
TimeIndependentPDESolver< Problem >::
writeEpilog( Logger& logger ) const
{
   return this->problem->writeEpilog( logger );
}

} // namespace PDE
} // namespace Solvers
} // namespace noaTNL
