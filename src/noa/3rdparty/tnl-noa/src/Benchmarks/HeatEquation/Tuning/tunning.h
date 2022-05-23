#pragma once

#ifdef HAVE_CUDA
#include<cuda.h>
#endif

#include "SimpleCell.h"

/****
 * Just testing data for measuring performance
 * with different ways of passing data to kernels.
 */
struct Data
{
   double time, tau;
   TNL::Containers::StaticVector< 2, double > c1, c2, c3, c4;
   TNL::Meshes::Grid< 2, double > grid;
};


#ifdef HAVE_CUDA

#define WITH_CELL  // Serves for comparison of performance when using SimpleCell
                   // vs. using only cell index and coordinates

template< typename BoundaryEntitiesProcessor, typename UserData, typename Grid, typename Real, typename Index >
__global__ void _boundaryConditionsKernel( const Grid* grid,
                                           UserData userData )
{
   //Real* u = userData.u;
   //const typename UserData::BoundaryConditionsType* bc = userData.boundaryConditions;
   using Coordinates = typename Grid::CoordinatesType;
   const Index& gridXSize = grid->getDimensions().x();
   const Index& gridYSize = grid->getDimensions().y();
#ifdef WITH_CELL   
   SimpleCell< Grid > cell( *grid,
      Coordinates( ( blockIdx.x ) * blockDim.x + threadIdx.x,
                   ( blockIdx.y ) * blockDim.y + threadIdx.y ) );
   Coordinates& coordinates = cell.getCoordinates();
   cell.refresh();   
#else   
   Coordinates coordinates( ( blockIdx.x ) * blockDim.x + threadIdx.x,
                             ( blockIdx.y ) * blockDim.y + threadIdx.y );
   Index entityIndex = coordinates.y() * gridXSize + coordinates.x();
#endif   
   
   
   if( coordinates.x() == 0 && coordinates.y() < gridYSize )
   {
      //u[ c ] = ( *bc )( *grid, u, c, coordinates, 0 );
#ifdef WITH_CELL
      BoundaryEntitiesProcessor::processEntity( *grid, userData, cell );
#else
      BoundaryEntitiesProcessor::processEntity( *grid, userData, entityIndex, coordinates );
#endif 
   }
   if( coordinates.x() == gridXSize - 1 && coordinates.y() < gridYSize )
   {
      //u[ c ] = ( *bc )( *grid, u, c, coordinates, 0 );

#ifdef WITH_CELL
      BoundaryEntitiesProcessor::processEntity( *grid, userData, cell );
#else
      BoundaryEntitiesProcessor::processEntity( *grid, userData, entityIndex, coordinates );
#endif      
   }
   if( coordinates.y() == 0 && coordinates.x() < gridXSize )
   {
      //u[ c ] = ( *bc )( *grid, u, c, coordinates, 0 );
#ifdef WITH_CELL
      BoundaryEntitiesProcessor::processEntity( *grid, userData, cell );
#else
      BoundaryEntitiesProcessor::processEntity( *grid, userData, entityIndex, coordinates );
#endif      
   }
   if( coordinates.y() == gridYSize -1  && coordinates.x() < gridXSize )
   {
      //u[ c ] = ( *bc )( *grid, u, c, coordinates, 0 );

#ifdef WITH_CELL
      BoundaryEntitiesProcessor::processEntity( *grid, userData, cell );
#else
      BoundaryEntitiesProcessor::processEntity( *grid, userData, entityIndex, coordinates );
#endif      
   }         
}

template< typename InteriorEntitiesProcessor, typename UserData, typename Grid, typename Real, typename Index >
__global__ void _heatEquationKernel( const Grid* grid,
                                     UserData userData )
{
   /*Real* u = userData.u;
   Real* fu = userData.fu;
   const typename UserData::DifferentialOperatorType* op = userData.differentialOperator;*/

   const Index& gridXSize = grid->getDimensions().x();
   const Index& gridYSize = grid->getDimensions().y();
   const Real& hx_inv = grid->template getSpaceStepsProducts< -2,  0 >();
   const Real& hy_inv = grid->template getSpaceStepsProducts<  0, -2 >();
   
   SimpleCell< Grid > cell( *grid );
   cell.getCoordinates().x() = blockIdx.x * blockDim.x + threadIdx.x;
   cell.getCoordinates().y() = blockIdx.y * blockDim.y + threadIdx.y;
   
   /*using Coordinates = typename Grid::CoordinatesType;
   Coordinates coordinates( blockIdx.x * blockDim.x + threadIdx.x, 
                            blockIdx.y * blockDim.y + threadIdx.y );*/

   if( cell.getCoordinates().x() > 0 && cell.getCoordinates().x() < gridXSize - 1 &&
       cell.getCoordinates().y() > 0 && cell.getCoordinates().y() < gridYSize - 1 )
   //if( coordinates.x() > 0 && coordinates.x() < gridXSize - 1 &&
   //    coordinates.y() > 0 && coordinates.y() < gridYSize - 1 )      
   {
#ifdef WITH_CELL      
      cell.refresh();
      InteriorEntitiesProcessor::processEntity( *grid, userData, cell );
#else      
      //const Index entityIndex = cell.getCoordinates().y() * gridXSize + cell.getCoordinates().x();
      const Index entityIndex = coordinates.y() * gridXSize + coordinates.x();
      InteriorEntitiesProcessor::processEntity( *grid, userData, entityIndex, cell.getCoordinates() );
#endif      
      
      
      //fu[ entityIndex ] = ( *op )( *grid, userData.u, entityIndex, coordinates, userData.time ); // + 0.1;
      
      //fu[ entityIndex ] = ( ( u[ entityIndex - 1 ]         - 2.0 * u[ entityIndex ] + u[ entityIndex + 1 ]         ) * hx_inv +
      //                    ( u[ entityIndex - gridXSize ] - 2.0 * u[ entityIndex ] + u[ entityIndex + gridXSize ] ) * hy_inv );


   }  
}


#endif

