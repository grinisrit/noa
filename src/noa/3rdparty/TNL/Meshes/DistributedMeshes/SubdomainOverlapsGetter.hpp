// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <noa/3rdparty/TNL/MPI/Wrappers.h>
#include <noa/3rdparty/TNL/Assert.h>
#include <noa/3rdparty/TNL/Meshes/Grid.h>

namespace noa::TNL {
namespace Meshes {
namespace DistributedMeshes {

/*
 * TODO: This could work when the MPI directions are rewritten

template< typename Real,
          typename Device,
          typename Index >
void
SubdomainOverlapsGetter< Grid< 1, Real, Device, Index > >::
getOverlaps( const DistributedMeshType* distributedMesh,
             SubdomainOverlapsType& lower,
             SubdomainOverlapsType& upper,
             IndexType subdomainOverlapSize,
             const SubdomainOverlapsType& periodicBoundariesOverlapSize )
{
   // initialize to 0
   lower = upper = 0;

   if( ! MPI::isDistributed() )
      return;
   TNL_ASSERT_TRUE( distributedMesh != NULL, "" );

   const CoordinatesType& subdomainCoordinates = distributedMesh->getSubdomainCoordinates();
   int rank = MPI::GetRank();

   for( int i = 0; i < Dimension; i++ )
   {
      CoordinatesType neighborDirection( 0 );
      neighborDirection[ i ] = -1;
      if( subdomainCoordinates[ i ] > 0 )
         lower[ i ] = subdomainOverlapSize;
      else if( distributedMesh->getPeriodicNeighbors()[ Directions::getDirection( neighborDirection ) ] != rank )
         lower[ i ] = periodicBoundariesOverlapSize[ i ];

      neighborDirection[ i ] = 1;
      if( subdomainCoordinates[ i ] < distributedMesh->getDomainDecomposition()[ i ] - 1 )
         upper[ i ] = subdomainOverlapSize;
      else if( distributedMesh->getPeriodicNeighbors()[ Directions::getDirection( neighborDirection ) ] != rank )
         upper[ i ] = periodicBoundariesOverlapSize[ i ];
   }
}

*/

template< typename Real,
          typename Device,
          typename Index >
void
SubdomainOverlapsGetter< Grid< 1, Real, Device, Index > >::
getOverlaps( const DistributedMeshType* distributedMesh,
             SubdomainOverlapsType& lower,
             SubdomainOverlapsType& upper,
             IndexType subdomainOverlapSize,
             const SubdomainOverlapsType& lowerPeriodicBoundariesOverlapSize,
             const SubdomainOverlapsType& upperPeriodicBoundariesOverlapSize )
{
   // initialize to 0
   lower = upper = 0;

   if( MPI::GetSize() == 1 )
      return;
   TNL_ASSERT_TRUE( distributedMesh != NULL, "" );

   const CoordinatesType& subdomainCoordinates = distributedMesh->getSubdomainCoordinates();
   int rank = MPI::GetRank();

   if( subdomainCoordinates[ 0 ] > 0 )
      lower[ 0 ] = subdomainOverlapSize;
   else if( distributedMesh->getPeriodicNeighbors()[ ZzYzXm ] != rank )
      lower[ 0 ] = lowerPeriodicBoundariesOverlapSize[ 0 ];

   if( subdomainCoordinates[ 0 ] < distributedMesh->getDomainDecomposition()[ 0 ] - 1 )
      upper[ 0 ] = subdomainOverlapSize;
   else if( distributedMesh->getPeriodicNeighbors()[ ZzYzXp ] != rank )
      upper[ 0 ] = upperPeriodicBoundariesOverlapSize[ 0 ];
}


template< typename Real,
          typename Device,
          typename Index >
void
SubdomainOverlapsGetter< Grid< 2, Real, Device, Index > >::
getOverlaps( const DistributedMeshType* distributedMesh,
             SubdomainOverlapsType& lower,
             SubdomainOverlapsType& upper,
             IndexType subdomainOverlapSize,
             const SubdomainOverlapsType& lowerPeriodicBoundariesOverlapSize,
             const SubdomainOverlapsType& upperPeriodicBoundariesOverlapSize )
{
   // initialize to 0
   lower = upper = 0;

   if( MPI::GetSize() == 1 )
      return;
   TNL_ASSERT_TRUE( distributedMesh != NULL, "" );

   const CoordinatesType& subdomainCoordinates = distributedMesh->getSubdomainCoordinates();
   int rank = MPI::GetRank();
   lower = 0;
   upper = 0;

   if( subdomainCoordinates[ 0 ] > 0 )
      lower[ 0 ] = subdomainOverlapSize;
   else if( distributedMesh->getPeriodicNeighbors()[ ZzYzXm ] != rank )
      lower[ 0 ] = lowerPeriodicBoundariesOverlapSize[ 0 ];

   if( subdomainCoordinates[ 0 ] < distributedMesh->getDomainDecomposition()[ 0 ] - 1 )
      upper[ 0 ] = subdomainOverlapSize;
   else if( distributedMesh->getPeriodicNeighbors()[ ZzYzXp ] != rank )
      upper[ 0 ] = upperPeriodicBoundariesOverlapSize[ 0 ];

   if( subdomainCoordinates[ 1 ] > 0 )
      lower[ 1 ] = subdomainOverlapSize;
   else if( distributedMesh->getPeriodicNeighbors()[ ZzYmXz ] != rank )
      lower[ 1 ] = lowerPeriodicBoundariesOverlapSize[ 1 ];

   if( subdomainCoordinates[ 1 ] < distributedMesh->getDomainDecomposition()[ 1 ] - 1 )
      upper[ 1 ] = subdomainOverlapSize;
   else if( distributedMesh->getPeriodicNeighbors()[ ZzYpXz ] != rank )
      upper[ 1 ] = upperPeriodicBoundariesOverlapSize[ 1 ];
}

template< typename Real,
          typename Device,
          typename Index >
void
SubdomainOverlapsGetter< Grid< 3, Real, Device, Index > >::
getOverlaps( const DistributedMeshType* distributedMesh,
             SubdomainOverlapsType& lower,
             SubdomainOverlapsType& upper,
             IndexType subdomainOverlapSize,
             const SubdomainOverlapsType& lowerPeriodicBoundariesOverlapSize,
             const SubdomainOverlapsType& upperPeriodicBoundariesOverlapSize )
{
   // initialize to 0
   lower = upper = 0;

   if( MPI::GetSize() == 1 )
      return;
   TNL_ASSERT_TRUE( distributedMesh != NULL, "" );

   const CoordinatesType& subdomainCoordinates = distributedMesh->getSubdomainCoordinates();
   int rank = MPI::GetRank();

   if( subdomainCoordinates[ 0 ] > 0 )
      lower[ 0 ] = subdomainOverlapSize;
   else if( distributedMesh->getPeriodicNeighbors()[ ZzYzXm ] != rank )
      lower[ 0 ] = lowerPeriodicBoundariesOverlapSize[ 0 ];

   if( subdomainCoordinates[ 0 ] < distributedMesh->getDomainDecomposition()[ 0 ] - 1 )
      upper[ 0 ] = subdomainOverlapSize;
   else if( distributedMesh->getPeriodicNeighbors()[ ZzYzXp ] != rank )
      upper[ 0 ] = upperPeriodicBoundariesOverlapSize[ 0 ];

   if( subdomainCoordinates[ 1 ] > 0 )
      lower[ 1 ] = subdomainOverlapSize;
   else if( distributedMesh->getPeriodicNeighbors()[ ZzYmXz ] != rank )
      lower[ 1 ] = lowerPeriodicBoundariesOverlapSize[ 1 ];

   if( subdomainCoordinates[ 1 ] < distributedMesh->getDomainDecomposition()[ 1 ] - 1 )
      upper[ 1 ] = subdomainOverlapSize;
   else if( distributedMesh->getPeriodicNeighbors()[ ZzYpXz ] != rank )
      upper[ 1 ] = upperPeriodicBoundariesOverlapSize[ 1 ];

   if( subdomainCoordinates[ 2 ] > 0 )
      lower[ 2 ] = subdomainOverlapSize;
   else if( distributedMesh->getPeriodicNeighbors()[ ZmYzXz ] != rank )
      lower[ 2 ] = lowerPeriodicBoundariesOverlapSize[ 2 ];

   if( subdomainCoordinates[ 2 ] < distributedMesh->getDomainDecomposition()[ 2 ] - 1 )
      upper[ 2 ] = subdomainOverlapSize;
   else if( distributedMesh->getPeriodicNeighbors()[ ZpYzXz ] != rank )
      upper[ 2 ] = upperPeriodicBoundariesOverlapSize[ 2 ];
}

} // namespace DistributedMeshes
} // namespace Meshes
} // namespace noa::TNL
