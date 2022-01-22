// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdlib>

#include "DistributedGrid.h"
#include <noa/3rdparty/TNL/Meshes/DistributedMeshes/SubdomainOverlapsGetter.h>
#include <noa/3rdparty/TNL/MPI/Wrappers.h>

namespace noaTNL {
namespace Meshes {
namespace DistributedMeshes {

template< int Dimension, typename Real, typename Device, typename Index >
void
DistributedMesh< Grid< Dimension, Real, Device, Index > >::
setDomainDecomposition( const CoordinatesType& domainDecomposition )
{
   this->domainDecomposition = domainDecomposition;
}

template< int Dimension, typename Real, typename Device, typename Index >
const typename DistributedMesh< Grid< Dimension, Real, Device, Index > >::CoordinatesType&
DistributedMesh< Grid< Dimension, Real, Device, Index > >::
getDomainDecomposition() const
{
   return this->domainDecomposition;
}

template< int Dimension, typename Real, typename Device, typename Index >
void
DistributedMesh< Grid< Dimension, Real, Device, Index > >::
setGlobalGrid( const GridType& globalGrid )
{
   this->globalGrid = globalGrid;
   this->isSet=true;

   for( int i = 0; i < getNeighborsCount(); i++ )
      this->neighbors[ i ] = -1;

   // use MPI only if have more than one process
   this->distributed = MPI::GetSize( communicator ) > 1;

   if( !this->distributed )
   {
      this->subdomainCoordinates.setValue( 0 );
      this->domainDecomposition.setValue( 0 );
      localGrid.setOrigin( globalGrid.getOrigin() );
      localGrid.setDimensions( globalGrid.getDimensions() );
      this->localSize = globalGrid.getDimensions();
      this->globalBegin = 0;
   }
   else
   {
      CoordinatesType numberOfLarger;
      //compute node distribution
      int dims[ Dimension ];
      for( int i = 0; i < Dimension; i++ )
         dims[ i ] = this->domainDecomposition[ i ];
      MPI::Compute_dims( MPI::GetSize( communicator ), Dimension, dims );
      for( int i = 0; i < Dimension; i++ )
         this->domainDecomposition[ i ] = dims[ i ];

      int size = MPI::GetSize( communicator );
      int tmp = MPI::GetRank( communicator );
      for( int i = Dimension - 1; i >= 0; i-- )
      {
         size = size / this->domainDecomposition[ i ];
         this->subdomainCoordinates[ i ] = tmp / size;
         tmp = tmp % size;
      }

      for( int i = 0; i < Dimension; i++ )
      {
         numberOfLarger[ i ] = globalGrid.getDimensions()[ i ] % this->domainDecomposition[ i ];

         this->localSize[ i ] = globalGrid.getDimensions()[ i ] / this->domainDecomposition[ i ];

         if( numberOfLarger[ i ] > this->subdomainCoordinates[ i ] )
            this->localSize[ i ] += 1;

         if( numberOfLarger[ i ] > this->subdomainCoordinates[ i ] )
             this->globalBegin[ i ] = this->subdomainCoordinates[ i ] * this->localSize[ i ];
         else
             this->globalBegin[ i ] = numberOfLarger[ i ] * ( this->localSize[ i ] + 1 ) +
                                     ( this->subdomainCoordinates[ i ] - numberOfLarger[ i ] ) * this->localSize[ i ];
      }

      localGrid.setDimensions( this->localSize );
      this->setupNeighbors();
   }

   // setting space steps computes the grid proportions as a side efect
   localGrid.setSpaceSteps( globalGrid.getSpaceSteps() );
}

template< int Dimension, typename Real, typename Device, typename Index >
void
DistributedMesh< Grid< Dimension, Real, Device, Index > >::
setOverlaps( const SubdomainOverlapsType& lower,
             const SubdomainOverlapsType& upper)
{
   this->lowerOverlap = lower;
   this->upperOverlap = upper;
   localGrid.setOrigin( this->globalGrid.getOrigin() + this->globalGrid.getSpaceSteps() * (this->globalBegin - this->lowerOverlap) );
   localGrid.setDimensions( this->localSize + this->lowerOverlap + this->upperOverlap );
   // setting space steps computes the grid proportions as a side efect
   localGrid.setSpaceSteps( globalGrid.getSpaceSteps() );

   // update local begin and end
   localGrid.setLocalBegin( this->lowerOverlap );
   localGrid.setLocalEnd( localGrid.getDimensions() - this->upperOverlap );

   // update interior begin and end
   CoordinatesType interiorBegin = this->lowerOverlap;
   CoordinatesType interiorEnd = localGrid.getDimensions() - this->upperOverlap;
   const int* neighbors = getNeighbors();
   if( neighbors[ ZzYzXm ] == -1 )
      interiorBegin[0] += 1;
   if( neighbors[ ZzYzXp ] == -1 )
      interiorEnd[0] -= 1;
   if( ZzYmXz < getNeighborsCount() && neighbors[ ZzYmXz ] == -1 )
      interiorBegin[1] += 1;
   if( ZzYpXz < getNeighborsCount() && neighbors[ ZzYpXz ] == -1 )
      interiorEnd[1] -= 1;
   if( ZmYzXz < getNeighborsCount() && neighbors[ ZmYzXz ] == -1 )
      interiorBegin[2] += 1;
   if( ZpYzXz < getNeighborsCount() && neighbors[ ZpYzXz ] == -1 )
      interiorEnd[2] -= 1;
   localGrid.setInteriorBegin( interiorBegin );
   localGrid.setInteriorEnd( interiorEnd );
}

template< int Dimension, typename Real, typename Device, typename Index >
void
DistributedMesh< Grid< Dimension, Real, Device, Index > >::
setGhostLevels( int levels )
{
   SubdomainOverlapsType lowerOverlap, upperOverlap;
   SubdomainOverlapsGetter< GridType >::getOverlaps( this, lowerOverlap, upperOverlap, levels );
   setOverlaps( lowerOverlap, upperOverlap );
}

template< int Dimension, typename Real, typename Device, typename Index >
int
DistributedMesh< Grid< Dimension, Real, Device, Index > >::
getGhostLevels() const
{
   return noaTNL::max( noaTNL::max(lowerOverlap), noaTNL::max(upperOverlap) );
}

template< int Dimension, typename Real, typename Device, typename Index >
const typename DistributedMesh< Grid< Dimension, Real, Device, Index > >::CoordinatesType&
DistributedMesh< Grid< Dimension, Real, Device, Index > >::
getSubdomainCoordinates() const
{
   return this->subdomainCoordinates;
}

template< int Dimension, typename Real, typename Device, typename Index >
bool
DistributedMesh< Grid< Dimension, Real, Device, Index > >::
isDistributed() const
{
   return this->distributed;
};

template< int Dimension, typename Real, typename Device, typename Index >
bool
DistributedMesh< Grid< Dimension, Real, Device, Index > >::
isBoundarySubdomain() const
{
   for( int i = 0; i < getNeighborsCount(); i++ )
      if( this->neighbors[ i ] == -1 )
         return true;
   return false;
}

template< int Dimension, typename Real, typename Device, typename Index >
const typename DistributedMesh< Grid< Dimension, Real, Device, Index > >::CoordinatesType&
DistributedMesh< Grid< Dimension, Real, Device, Index > >::
getLowerOverlap() const
{
   return this->lowerOverlap;
};

template< int Dimension, typename Real, typename Device, typename Index >
const typename DistributedMesh< Grid< Dimension, Real, Device, Index > >::CoordinatesType&
DistributedMesh< Grid< Dimension, Real, Device, Index > >::
getUpperOverlap() const
{
   return this->upperOverlap;
};

template< int Dimension, typename Real, typename Device, typename Index >
const typename DistributedMesh< Grid< Dimension, Real, Device, Index > >::GridType&
DistributedMesh< Grid< Dimension, Real, Device, Index > >::
getLocalMesh() const
{
    return this->localGrid;
}

template< int Dimension, typename Real, typename Device, typename Index >
const typename DistributedMesh< Grid< Dimension, Real, Device, Index > >::CoordinatesType&
DistributedMesh< Grid< Dimension, Real, Device, Index > >::
getLocalSize() const
{
   return this->localSize;
}

template< int Dimension, typename Real, typename Device, typename Index >
const typename DistributedMesh< Grid< Dimension, Real, Device, Index > >::CoordinatesType&
DistributedMesh< Grid< Dimension, Real, Device, Index > >::
getGlobalSize() const
{
   return this->globalGrid.getDimensions();
}

template< int Dimension, typename Real, typename Device, typename Index >
const typename DistributedMesh< Grid< Dimension, Real, Device, Index > >::GridType&
DistributedMesh< Grid< Dimension, Real, Device, Index > >::
getGlobalGrid() const
{
    return this->globalGrid;
}

template< int Dimension, typename Real, typename Device, typename Index >
const typename DistributedMesh< Grid< Dimension, Real, Device, Index > >::CoordinatesType&
DistributedMesh< Grid< Dimension, Real, Device, Index > >::
getGlobalBegin() const
{
   return this->globalBegin;
}

template< int Dimension, typename Real, typename Device, typename Index >
   template< int EntityDimension >
Index
DistributedMesh< Grid< Dimension, Real, Device, Index > >::
getEntitiesCount() const
{
   return this->globalGrid. template getEntitiesCount< EntityDimension >();
}

template< int Dimension, typename Real, typename Device, typename Index >
   template< typename Entity >
Index
DistributedMesh< Grid< Dimension, Real, Device, Index > >::
getEntitiesCount() const
{
   return this->globalGrid. template getEntitiesCount< Entity >();
}

template< int Dimension, typename Real, typename Device, typename Index >
void
DistributedMesh< Grid< Dimension, Real, Device, Index > >::
setCommunicator( MPI_Comm communicator )
{
   this->communicator = communicator;
}

template< int Dimension, typename Real, typename Device, typename Index >
MPI_Comm
DistributedMesh< Grid< Dimension, Real, Device, Index > >::
getCommunicator() const
{
   return this->communicator;
}

template< int Dimension, typename Real, typename Device, typename Index >
int
DistributedMesh< Grid< Dimension, Real, Device, Index > >::
getRankOfProcCoord(const CoordinatesType &nodeCoordinates) const
{
    int DimensionOffset=1;
    int ret=0;
    for(int i=0;i<Dimension;i++)
    {
        ret += DimensionOffset*nodeCoordinates[i];
        DimensionOffset *= this->domainDecomposition[i];
    }
    return ret;
}

template< int Dimension, typename Real, typename Device, typename Index >
bool
DistributedMesh< Grid< Dimension, Real, Device, Index > >::
isThereNeighbor(const CoordinatesType &direction) const
{
    bool res=true;
    for(int i=0;i<Dimension;i++)
    {
        if(direction[i]==-1)
            res&= this->subdomainCoordinates[i]>0;

        if(direction[i]==1)
            res&= this->subdomainCoordinates[i]<this->domainDecomposition[i]-1;
    }
    return res;

}

template< int Dimension, typename Real, typename Device, typename Index >
void
DistributedMesh< Grid< Dimension, Real, Device, Index > >::
setupNeighbors()
{
   for( int i = 0; i < getNeighborsCount(); i++ )
   {
      auto direction = Directions::template getXYZ< Dimension >( i );
      CoordinatesType coordinates = this->subdomainCoordinates+direction;
      if( this->isThereNeighbor( direction ) )
         this->neighbors[ i ] = this->getRankOfProcCoord( coordinates );
      else
         this->neighbors[ i ] = -1;

      // Handling periodic neighbors
      for( int d = 0; d < Dimension; d++ )
      {
         if( coordinates[ d ] == -1 )
            coordinates[ d ] = this->domainDecomposition[ d ] - 1;
         if( coordinates[ d ] == this->domainDecomposition[ d ] )
            coordinates[ d ] = 0;
         this->periodicNeighbors[ i ] = this->getRankOfProcCoord( coordinates );
      }

      //std::cout << "Setting i-th neighbour to " << neighbors[ i ] << " and " << periodicNeighbors[ i ] << std::endl;
   }
}

template< int Dimension, typename Real, typename Device, typename Index >
const int*
DistributedMesh< Grid< Dimension, Real, Device, Index > >::
getNeighbors() const
{
    TNL_ASSERT_TRUE(this->isSet,"DistributedGrid is not set, but used by getNeighbors");
    return this->neighbors;
}

template< int Dimension, typename Real, typename Device, typename Index >
const int*
DistributedMesh< Grid< Dimension, Real, Device, Index > >::
getPeriodicNeighbors() const
{
    TNL_ASSERT_TRUE(this->isSet,"DistributedGrid is not set, but used by getNeighbors");
    return this->periodicNeighbors;
}

template< int Dimension, typename Real, typename Device, typename Index >
    template<typename DistributedGridType >
bool
DistributedMesh< Grid< Dimension, Real, Device, Index > >::
SetupByCut( DistributedGridType &inputDistributedGrid,
            Containers::StaticVector<Dimension, int> savedDimensions,
            Containers::StaticVector<DistributedGridType::getMeshDimension()-Dimension,int> reducedDimensions,
            Containers::StaticVector<DistributedGridType::getMeshDimension()-Dimension,IndexType> fixedIndexs)
{
   bool isInCut=true;
   const int coDimension = DistributedGridType::getMeshDimension()-Dimension;
   for( int i = 0; i < coDimension; i++ )
   {
      auto begin = inputDistributedGrid.getGlobalBegin();
      auto size = inputDistributedGrid.getLocalSize();
      isInCut &= fixedIndexs[i] > begin[reducedDimensions[i]] && fixedIndexs[i] < begin[reducedDimensions[i]] + size[reducedDimensions[i]];
   }

   // create new communicator with used nodes
   const MPI_Comm oldCommunicator = inputDistributedGrid.getCommunicator();
   if(isInCut)
   {
      this->isSet=true;

      auto fromGlobalMesh = inputDistributedGrid.getGlobalGrid();
      // set global grid
      typename GridType::PointType outOrigin;
      typename GridType::PointType outProportions;
      typename GridType::CoordinatesType outDimensions;
      // set local grid
      typename GridType::PointType localOrigin;
      typename GridType::CoordinatesType localBegin, localGridSize;

      for(int i=0; i<Dimension;i++)
      {
         outOrigin[i] = fromGlobalMesh.getOrigin()[savedDimensions[i]];
         outProportions[i] = fromGlobalMesh.getProportions()[savedDimensions[i]];
         outDimensions[i] = fromGlobalMesh.getDimensions()[savedDimensions[i]];

         this->domainDecomposition[i] = inputDistributedGrid.getDomainDecomposition()[savedDimensions[i]];
         this->subdomainCoordinates[i] = inputDistributedGrid.getSubdomainCoordinates()[savedDimensions[i]];

         this->lowerOverlap[i] = inputDistributedGrid.getLowerOverlap()[savedDimensions[i]];
         this->upperOverlap[i] = inputDistributedGrid.getUpperOverlap()[savedDimensions[i]];
         this->localSize[i] = inputDistributedGrid.getLocalSize()[savedDimensions[i]];
         this->globalBegin[i] = inputDistributedGrid.getGlobalBegin()[savedDimensions[i]];
         localGridSize[i] = inputDistributedGrid.getLocalMesh().getDimensions()[savedDimensions[i]];
         localBegin[i] = inputDistributedGrid.getLocalMesh().getLocalBegin()[savedDimensions[i]];
         localOrigin[i] = inputDistributedGrid.getLocalMesh().getOrigin()[savedDimensions[i]];
      }

      this->globalGrid.setDimensions(outDimensions);
      this->globalGrid.setDomain(outOrigin,outProportions);

      // setOverlaps resets the local grid
//      setOverlaps( this->lowerOverlap, this->upperOverlap );

      localGrid.setDimensions( localGridSize );
      localGrid.setOrigin( localOrigin );
      // setting space steps computes the grid proportions as a side efect
      localGrid.setSpaceSteps( globalGrid.getSpaceSteps() );
      localGrid.setLocalBegin( localBegin );
      localGrid.setLocalEnd( localBegin + localSize );
      // TODO: set interiorBegin, interiorEnd

      const int newRank = getRankOfProcCoord(this->subdomainCoordinates);
      this->communicator = MPI::Comm_split( oldCommunicator, 1, newRank );

      setupNeighbors();

      bool isDistributed = false;
      for( int i = 0; i < Dimension; i++ )
      {
         isDistributed |= domainDecomposition[i] > 1;
      }
      this->distributed = isDistributed;

      return true;
   }
   else
   {
      this->communicator = MPI::Comm_split( oldCommunicator, MPI_UNDEFINED, 0 );
      return false;
   }
}

template< int Dimension, typename Real, typename Device, typename Index >
String
DistributedMesh< Grid< Dimension, Real, Device, Index > >::
printProcessCoords() const
{
   String res = convertToString(this->subdomainCoordinates[0]);
   for(int i=1; i<Dimension; i++)
        res=res+String("-")+convertToString(this->subdomainCoordinates[i]);
   return res;
};

template< int Dimension, typename Real, typename Device, typename Index >
String
DistributedMesh< Grid< Dimension, Real, Device, Index > >::
printProcessDistr() const
{
   String res = convertToString(this->domainDecomposition[0]);
   for(int i=1; i<Dimension; i++)
        res=res+String("-")+convertToString(this->domainDecomposition[i]);
   return res;
};

template< int Dimension, typename Real, typename Device, typename Index >
void
DistributedMesh< Grid< Dimension, Real, Device, Index > >::
writeProlog( Logger& logger )
{
   logger.writeParameter( "Domain decomposition:", this->getDomainDecomposition() );
}

template< int Dimension, typename Real, typename Device, typename Index >
bool
DistributedMesh< Grid< Dimension, Real, Device, Index > >::
operator==( const DistributedMesh& other ) const
{
   return globalGrid == other.globalGrid
       && localGrid == other.localGrid
       && localSize == other.localSize
       && globalBegin == other.globalBegin
       && lowerOverlap == other.lowerOverlap
       && upperOverlap == other.upperOverlap
       && domainDecomposition == other.domainDecomposition
       && subdomainCoordinates == other.subdomainCoordinates
       && distributed == other.distributed
       && isSet == other.isSet
       && communicator == other.communicator;
}

template< int Dimension, typename Real, typename Device, typename Index >
bool
DistributedMesh< Grid< Dimension, Real, Device, Index > >::
operator!=( const DistributedMesh& other ) const
{
   return ! operator==( other );
}

template< int Dimension, typename Real, typename Device, typename Index >
std::ostream& operator<<( std::ostream& str, const DistributedMesh< Grid< Dimension, Real, Device, Index > >& grid )
{
   for( int j = 0; j < MPI::GetSize(); j++ )
   {
      if( j == MPI::GetRank() )
      {
         str << "Node : " << MPI::GetRank() << std::endl
             << " globalGrid : " << grid.getGlobalGrid() << std::endl
             << " localGrid : " << grid.getLocalMesh() << std::endl
             << " localSize : " << grid.getLocalSize() << std::endl
             << " globalBegin : " << grid.globalBegin << std::endl
             << " lowerOverlap : " << grid.lowerOverlap << std::endl
             << " upperOverlap : " << grid.upperOverlap << std::endl
             << " domainDecomposition : " << grid.domainDecomposition << std::endl
             << " subdomainCoordinates : " << grid.subdomainCoordinates << std::endl
             << " neighbors : ";
         for( int i = 0; i < grid.getNeighborsCount(); i++ )
            str << " " << grid.getNeighbors()[ i ];
         str << std::endl;
         str << " periodicNeighbours : ";
         for( int i = 0; i < grid.getNeighborsCount(); i++ )
            str << " " << grid.getPeriodicNeighbors()[ i ];
         str << std::endl;
      }
      MPI::Barrier();
   }
   return str;
}

} // namespace DistributedMeshes
} // namespace Meshes
} // namespace noaTNL
