#pragma once

#include "ComputeBlockResidue.h"

namespace TNL {
namespace Benchmarks {

#ifdef HAVE_CUDA
template< typename RealType, typename Index >
__global__ void updateUEuler( const Index size,
                              const RealType tau,
                              const RealType* k1,
                              RealType* u,
                              RealType* cudaBlockResidue );
#endif

template< typename Problem, typename SolverMonitor >
Euler< Problem, SolverMonitor >::Euler()
: cflCondition( 0.0 )
{
};

template< typename Problem, typename SolverMonitor >
void Euler< Problem, SolverMonitor >::configSetup( Config::ConfigDescription& config,
                                               const String& prefix )
{
   //ExplicitSolver< Problem >::configSetup( config, prefix );
   config.addEntry< double >( prefix + "euler-cfl", "Coefficient C in the Courant–Friedrichs–Lewy condition.", 0.0 );
};

template< typename Problem, typename SolverMonitor >
bool Euler< Problem, SolverMonitor >::setup( const Config::ParameterContainer& parameters,
                                        const String& prefix )
{
   Solvers::ODE::ExplicitSolver< Problem, SolverMonitor >::setup( parameters, prefix );
   if( parameters.checkParameter( prefix + "euler-cfl" ) )
      this->setCFLCondition( parameters.getParameter< double >( prefix + "euler-cfl" ) );
   return true;
}

template< typename Problem, typename SolverMonitor >
void Euler< Problem, SolverMonitor >::setCFLCondition( const RealType& cfl )
{
   this -> cflCondition = cfl;
}

template< typename Problem, typename SolverMonitor >
const typename Problem :: RealType& Euler< Problem, SolverMonitor >::getCFLCondition() const
{
   return this -> cflCondition;
}

template< typename Problem, typename SolverMonitor >
bool Euler< Problem, SolverMonitor >::solve( DofVectorPointer& u )
{
   /****
    * First setup the supporting meshes k1...k5 and k_tmp.
    */
   //timer.start();
   k1->setLike( *u );
   k1->setValue( 0.0 );


   /****
    * Set necessary parameters
    */
   RealType& time = this->time;
   RealType currentTau = min( this->getTau(), this->getMaxTau() );
   if( time + currentTau > this->getStopTime() ) currentTau = this->getStopTime() - time;
   if( currentTau == 0.0 ) return true;
   this->resetIterations();
   this->setResidue( this->getConvergenceResidue() + 1.0 );

   /****
    * Start the main loop
    */
   while( 1 )
   {
      /****
       * Compute the RHS
       */
      //timer.stop();
      this->problem->getExplicitUpdate( time, currentTau, u, k1 );
      //timer.start();

      RealType lastResidue = this->getResidue();
      RealType maxResidue( 0.0 );
      if( this -> cflCondition != 0.0 )
      {
         maxResidue = VectorOperations::getVectorAbsMax( *k1 );
         if( currentTau * maxResidue > this->cflCondition )
         {
            currentTau *= 0.9;
            continue;
         }
      }
      RealType newResidue( 0.0 );
      computeNewTimeLevel( u, currentTau, newResidue );
      this->setResidue( newResidue );

      /****
       * When time is close to stopTime the new residue
       * may be inaccurate significantly.
       */
      if( currentTau + time == this -> stopTime ) this->setResidue( lastResidue );
      time += currentTau;
      this->problem->applyBoundaryConditions( time, u );

      if( ! this->nextIteration() )
         return this->checkConvergence();

      /****
       * Compute the new time step.
       */
      if( time + currentTau > this -> getStopTime() )
         currentTau = this -> getStopTime() - time; //we don't want to keep such tau
      else this -> tau = currentTau;

      /****
       * Check stop conditions.
       */
      if( time >= this->getStopTime() ||
          ( this -> getConvergenceResidue() != 0.0 && this->getResidue() < this -> getConvergenceResidue() ) )
         return true;

      if( this -> cflCondition != 0.0 )
      {
         currentTau /= 0.95;
         currentTau = min( currentTau, this->getMaxTau() );
      }
   }
};

template< typename Problem, typename SolverMonitor >
void Euler< Problem, SolverMonitor >::computeNewTimeLevel( DofVectorPointer& u,
                                              RealType tau,
                                              RealType& currentResidue )
{
   RealType localResidue = RealType( 0.0 );
   const IndexType size = k1->getSize();
   RealType* _u = u->getData();
   RealType* _k1 = k1->getData();

   if( std::is_same< DeviceType, Devices::Host >::value )
   {
#ifdef HAVE_OPENMP
#pragma omp parallel for reduction(+:localResidue) firstprivate( _u, _k1, tau ) if( Devices::Host::isOMPEnabled() )
#endif
      for( IndexType i = 0; i < size; i ++ )
      {
         const RealType add = tau * _k1[ i ];
         _u[ i ] += add;
         localResidue += std::fabs( add );
      }
   }
   if( std::is_same< DeviceType, Devices::Cuda >::value )
   {
#ifdef HAVE_CUDA
      dim3 cudaBlockSize( 512 );
      const IndexType cudaBlocks = Cuda::getNumberOfBlocks( size, cudaBlockSize.x );
      const IndexType cudaGrids = Cuda::getNumberOfGrids( cudaBlocks );
      this->cudaBlockResidue.setSize( min( cudaBlocks, Cuda::getMaxGridSize() ) );
      const IndexType threadsPerGrid = Cuda::getMaxGridSize() * cudaBlockSize.x;

      localResidue = 0.0;
      for( IndexType gridIdx = 0; gridIdx < cudaGrids; gridIdx ++ )
      {
         const IndexType sharedMemory = cudaBlockSize.x * sizeof( RealType );
         const IndexType gridOffset = gridIdx * threadsPerGrid;
         const IndexType currentSize = min( size - gridOffset, threadsPerGrid );
         const IndexType currentGridSize = Cuda::getNumberOfBlocks( currentSize, cudaBlockSize.x );

         updateUEuler<<< currentGridSize, cudaBlockSize, sharedMemory >>>( currentSize,
                                                                      tau,
                                                                      &_k1[ gridOffset ],
                                                                      &_u[ gridOffset ],
                                                                      this->cudaBlockResidue.getData() );
         localResidue += sum( this->cudaBlockResidue );
         cudaDeviceSynchronize();
         TNL_CHECK_CUDA_DEVICE;
      }
#endif
   }

   localResidue /= tau * ( RealType ) size;
   TNL::MPI::Allreduce( &localResidue, &currentResidue, 1, MPI_SUM, MPI_COMM_WORLD );
   //std::cerr << "Local residue = " << localResidue << " - globalResidue = " << currentResidue << std::endl;
}

#ifdef HAVE_CUDA
template< typename RealType, typename IndexType >
__global__ void updateUEuler( const IndexType size,
                              const RealType tau,
                              const RealType* k1,
                              RealType* u,
                              RealType* cudaBlockResidue )
{
   extern __shared__ void* d_u[];
   RealType* du = ( RealType* ) d_u;
   const IndexType blockOffset = blockIdx. x * blockDim.x;
   const IndexType i = blockOffset  + threadIdx. x;
   if( i < size )
      u[ i ] += du[ threadIdx.x ] = tau * k1[ i ];
   else
      du[ threadIdx.x ] = 0.0;
   du[ threadIdx.x ] = abs( du[ threadIdx.x ] );
   __syncthreads();

   const IndexType rest = size - blockOffset;
   IndexType n =  rest < blockDim.x ? rest : blockDim.x;

   computeBlockResidue( du,
                        cudaBlockResidue,
                        n );
}
#endif

} // namespace Benchmarks
} // namespace TNL
