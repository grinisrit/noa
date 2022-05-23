#pragma once

#include <TNL/Devices/Host.h>
#include <TNL/Devices/Cuda.h>
#include <TNL/Config/ParameterContainer.h>

#include "Merson.h"

namespace TNL {
namespace Benchmarks {

/****
 * In this code we do not use constants and references as we would like to.
 * OpenMP would complain that
 *
 *  error: ‘some-variable’ is predetermined ‘shared’ for ‘firstprivate’
 *
 */

#ifdef HAVE_CUDA

template< typename Real, typename Index >
__global__ void computeK2Arg( const Index size,
                              const Real tau,
                              const Real* u,
                              const Real* k1,
                              Real* k2_arg );

template< typename Real, typename Index >
__global__ void computeK3Arg( const Index size,
                              const Real tau,
                              const Real* u,
                              const Real* k1,
                              const Real* k2,
                              Real* k3_arg );

template< typename Real, typename Index >
__global__ void computeK4Arg( const Index size,
                              const Real tau,
                              const Real* u,
                              const Real* k1,
                              const Real* k3,
                              Real* k4_arg );

template< typename Real, typename Index >
__global__ void computeK5Arg( const Index size,
                              const Real tau,
                              const Real* u,
                              const Real* k1,
                              const Real* k3,
                              const Real* k4,
                              Real* k5_arg );

template< typename Real, typename Index >
__global__ void computeErrorKernel( const Index size,
                                    const Real tau,
                                    const Real* k1,
                                    const Real* k3,
                                    const Real* k4,
                                    const Real* k5,
                                    Real* err );

template< typename Real, typename Index >
__global__ void updateUMerson( const Index size,
                               const Real tau,
                               const Real* k1,
                               const Real* k4,
                               const Real* k5,
                               Real* u,
                               Real* blockResidue );
#endif



template< typename Problem, typename SolverMonitor >
Merson< Problem, SolverMonitor >::Merson()
: adaptivity( 0.00001 )
{
   if( std::is_same< DeviceType, Devices::Host >::value )
   {
      this->openMPErrorEstimateBuffer.setSize( std::max( 1, Devices::Host::getMaxThreadsCount() ) );
   }
};

template< typename Problem, typename SolverMonitor >
void Merson< Problem, SolverMonitor >::configSetup( Config::ConfigDescription& config,
                                                const String& prefix )
{
   //ExplicitSolver< Problem >::configSetup( config, prefix );
   config.addEntry< double >( prefix + "merson-adaptivity", "Time step adaptivity controlling coefficient (the smaller the more precise the computation is, zero means no adaptivity).", 1.0e-4 );
};

template< typename Problem, typename SolverMonitor >
bool Merson< Problem, SolverMonitor >::setup( const Config::ParameterContainer& parameters,
                                         const String& prefix )
{
   Solvers::ODE::ExplicitSolver< Problem, SolverMonitor >::setup( parameters, prefix );
   if( parameters.checkParameter( prefix + "merson-adaptivity" ) )
      this->setAdaptivity( parameters.getParameter< double >( prefix + "merson-adaptivity" ) );
   return true;
}

template< typename Problem, typename SolverMonitor >
void Merson< Problem, SolverMonitor >::setAdaptivity( const RealType& a )
{
   this->adaptivity = a;
};

template< typename Problem, typename SolverMonitor >
bool Merson< Problem, SolverMonitor >::solve( DofVectorPointer& u )
{
   if( ! this->problem )
   {
      std::cerr << "No problem was set for the Merson ODE solver." << std::endl;
      return false;
   }
   if( this->getTau() == 0.0 )
   {
      std::cerr << "The time step for the Merson ODE solver is zero." << std::endl;
      return false;
   }
   /****
    * First setup the supporting meshes k1...k5 and kAux.
    */
   k1->setLike( *u );
   k2->setLike( *u );
   k3->setLike( *u );
   k4->setLike( *u );
   k5->setLike( *u );
   kAux->setLike( *u );
   k1->setValue( 0.0 );
   k2->setValue( 0.0 );
   k3->setValue( 0.0 );
   k4->setValue( 0.0 );
   k5->setValue( 0.0 );
   kAux->setValue( 0.0 );


   /****
    * Set necessary parameters
    */
   RealType& time = this->time;
   RealType currentTau = min( this->getTau(), this->getMaxTau() );
   if( time + currentTau > this->getStopTime() )
      currentTau = this->getStopTime() - time;
   if( currentTau == 0.0 ) return true;
   this->resetIterations();
   this->setResidue( this->getConvergenceResidue() + 1.0 );

   /****
    * Start the main loop
    */
   while( this->checkNextIteration() )
   {
      /****
       * Compute Runge-Kutta coefficients
       */
      computeKFunctions( u, time, currentTau );
      if( this->testingMode )
         writeGrids( u );

      /****
       * Compute an error of the approximation.
       */
      RealType eps( 0.0 );
      if( adaptivity != 0.0 )
         eps = computeError( currentTau );

      if( adaptivity == 0.0 || eps < adaptivity )
      {
         RealType lastResidue = this->getResidue();
         RealType newResidue( 0.0 );
         time += currentTau;
         computeNewTimeLevel( time, currentTau, u, newResidue );
         this->setResidue( newResidue );

         /****
          * When time is close to stopTime the new residue
          * may be inaccurate significantly.
          */
         if( abs( time - this->stopTime ) < 1.0e-7 ) this->setResidue( lastResidue );


         if( ! this->nextIteration() )
            return false;
      }

      /****
       * Compute the new time step.
       */
      if( adaptivity != 0.0 && eps != 0.0 )
      {
         currentTau *= 0.8 * ::pow( adaptivity / eps, 0.2 );
         currentTau = min( currentTau, this->getMaxTau() );
#ifdef USE_MPI
         TNLMPI::Bcast( currentTau, 1, 0 );
#endif
      }
      if( time + currentTau > this->getStopTime() )
         currentTau = this->getStopTime() - time; //we don't want to keep such tau
      else this->tau = currentTau;


      /****
       * Check stop conditions.
       */
      //cerr << "residue = " << residue << std::endl;
      //cerr << "this->getConvergenceResidue() = " << this->getConvergenceResidue() << std::endl;
      if( time >= this->getStopTime() ||
          ( this->getConvergenceResidue() != 0.0 && this->getResidue() < this->getConvergenceResidue() ) )
         return true;
   }
   return this->checkConvergence();

};

template< typename Problem, typename SolverMonitor >
void Merson< Problem, SolverMonitor >::computeKFunctions( DofVectorPointer& u,
                                                          const RealType& time,
                                                          RealType tau )
{
   IndexType size = u->getSize();

   RealType* _k1 = k1->getData();
   RealType* _k2 = k2->getData();
   RealType* _k3 = k3->getData();
   RealType* _k4 = k4->getData();
   //RealType* _k5 = k5->getData();
   RealType* _kAux = kAux->getData();
   RealType* _u = u->getData();

   RealType tau_3 = tau / 3.0;

   if( std::is_same< DeviceType, Devices::Host >::value )
   {
      this->problem->getExplicitUpdate( time, tau, u, k1 );

   #ifdef HAVE_OPENMP
   #pragma omp parallel for firstprivate( size, _kAux, _u, _k1, tau, tau_3 ) if( Devices::Host::isOMPEnabled() )
   #endif
      for( IndexType i = 0; i < size; i ++ )
         _kAux[ i ] = _u[ i ] + tau * ( 1.0 / 3.0 * _k1[ i ] );
      this->problem->applyBoundaryConditions( time + tau_3, kAux );
      this->problem->getExplicitUpdate( time + tau_3, tau, kAux, k2 );

   #ifdef HAVE_OPENMP
   #pragma omp parallel for firstprivate( size, _kAux, _u, _k1, _k2, tau, tau_3 ) if( Devices::Host::isOMPEnabled() )
   #endif
      for( IndexType i = 0; i < size; i ++ )
         _kAux[ i ] = _u[ i ] + tau * 1.0 / 6.0 * ( _k1[ i ] + _k2[ i ] );
      this->problem->applyBoundaryConditions( time + tau_3, kAux );
      this->problem->getExplicitUpdate( time + tau_3, tau, kAux, k3 );

   #ifdef HAVE_OPENMP
   #pragma omp parallel for firstprivate( size, _kAux, _u, _k1, _k3, tau, tau_3 ) if( Devices::Host::isOMPEnabled() )
   #endif
      for( IndexType i = 0; i < size; i ++ )
         _kAux[ i ] = _u[ i ] + tau * ( 0.125 * _k1[ i ] + 0.375 * _k3[ i ] );
      this->problem->applyBoundaryConditions( time + 0.5 * tau, kAux );
      this->problem->getExplicitUpdate( time + 0.5 * tau, tau, kAux, k4 );

   #ifdef HAVE_OPENMP
   #pragma omp parallel for firstprivate( size, _kAux, _u, _k1, _k3, _k4, tau, tau_3 ) if( Devices::Host::isOMPEnabled() )
   #endif
      for( IndexType i = 0; i < size; i ++ )
         _kAux[ i ] = _u[ i ] + tau * ( 0.5 * _k1[ i ] - 1.5 * _k3[ i ] + 2.0 * _k4[ i ] );
      this->problem->applyBoundaryConditions( time + tau, kAux );
      this->problem->getExplicitUpdate( time + tau, tau, kAux, k5 );
   }
   if( std::is_same< DeviceType, Devices::Cuda >::value )
   {
#ifdef HAVE_CUDA
      dim3 cudaBlockSize( 512 );
      const IndexType cudaBlocks = Cuda::getNumberOfBlocks( size, cudaBlockSize.x );
      const IndexType cudaGrids = Cuda::getNumberOfGrids( cudaBlocks );
      this->cudaBlockResidue.setSize( min( cudaBlocks, Cuda::getMaxGridSize() ) );
      const IndexType threadsPerGrid = Cuda::getMaxGridSize() * cudaBlockSize.x;

      this->problem->getExplicitUpdate( time, tau, u, k1 );
      cudaDeviceSynchronize();

      for( IndexType gridIdx = 0; gridIdx < cudaGrids; gridIdx ++ )
      {
         const IndexType gridOffset = gridIdx * threadsPerGrid;
         const IndexType currentSize = min( size - gridOffset, threadsPerGrid );
         computeK2Arg<<< cudaBlocks, cudaBlockSize >>>( currentSize, tau, &_u[ gridOffset ], &_k1[ gridOffset ], &_kAux[ gridOffset ] );
      }
      cudaDeviceSynchronize();
      this->problem->applyBoundaryConditions( time + tau_3, kAux );
      this->problem->getExplicitUpdate( time + tau_3, tau, kAux, k2 );
      cudaDeviceSynchronize();

      for( IndexType gridIdx = 0; gridIdx < cudaGrids; gridIdx ++ )
      {
         const IndexType gridOffset = gridIdx * threadsPerGrid;
         const IndexType currentSize = min( size - gridOffset, threadsPerGrid );
         computeK3Arg<<< cudaBlocks, cudaBlockSize >>>( currentSize, tau, &_u[ gridOffset ], &_k1[ gridOffset ], &_k2[ gridOffset ], &_kAux[ gridOffset ] );
      }
      cudaDeviceSynchronize();
      this->problem->applyBoundaryConditions( time + tau_3, kAux );
      this->problem->getExplicitUpdate( time + tau_3, tau, kAux, k3 );
      cudaDeviceSynchronize();

      for( IndexType gridIdx = 0; gridIdx < cudaGrids; gridIdx ++ )
      {
         const IndexType gridOffset = gridIdx * threadsPerGrid;
         const IndexType currentSize = min( size - gridOffset, threadsPerGrid );
         computeK4Arg<<< cudaBlocks, cudaBlockSize >>>( currentSize, tau, &_u[ gridOffset ], &_k1[ gridOffset ], &_k3[ gridOffset ], &_kAux[ gridOffset ] );
      }
      cudaDeviceSynchronize();
      this->problem->applyBoundaryConditions( time + 0.5 * tau, kAux );
      this->problem->getExplicitUpdate( time + 0.5 * tau, tau, kAux, k4 );
      cudaDeviceSynchronize();

      for( IndexType gridIdx = 0; gridIdx < cudaGrids; gridIdx ++ )
      {
         const IndexType gridOffset = gridIdx * threadsPerGrid;
         const IndexType currentSize = min( size - gridOffset, threadsPerGrid );
         computeK5Arg<<< cudaBlocks, cudaBlockSize >>>( currentSize, tau, &_u[ gridOffset ], &_k1[ gridOffset ], &_k3[ gridOffset ], &_k4[ gridOffset ], &_kAux[ gridOffset ] );
      }
      cudaDeviceSynchronize();
      this->problem->applyBoundaryConditions( time + tau, kAux );
      this->problem->getExplicitUpdate( time + tau, tau, kAux, k5 );
      cudaDeviceSynchronize();
#endif
   }
}

template< typename Problem, typename SolverMonitor >
typename Problem :: RealType Merson< Problem, SolverMonitor >::computeError( const RealType tau )
{
   const IndexType size = k1->getSize();
   const RealType* _k1 = k1->getData();
   const RealType* _k3 = k3->getData();
   const RealType* _k4 = k4->getData();
   const RealType* _k5 = k5->getData();

   RealType eps( 0.0 ), maxEps( 0.0 );
   if( std::is_same< DeviceType, Devices::Host >::value )
   {
      this->openMPErrorEstimateBuffer.setValue( 0.0 );
#ifdef HAVE_OPENMP
#pragma omp parallel if( Devices::Host::isOMPEnabled() )
#endif
      {
         RealType localEps( 0.0 );
#ifdef HAVE_OPENMP
#pragma omp for
#endif
         for( IndexType i = 0; i < size; i ++  )
         {
            RealType err = ( RealType ) ( tau / 3.0 *
                                 abs( 0.2 * _k1[ i ] +
                                     -0.9 * _k3[ i ] +
                                      0.8 * _k4[ i ] +
                                     -0.1 * _k5[ i ] ) );
            localEps = max( localEps, err );
         }
         this->openMPErrorEstimateBuffer[ Devices::Host::getThreadIdx() ] = localEps;
      }
      eps = VectorOperations::getVectorMax( this->openMPErrorEstimateBuffer );
   }
   if( std::is_same< DeviceType, Devices::Cuda >::value )
   {
#ifdef HAVE_CUDA
      RealType* _kAux = kAux->getData();
      dim3 cudaBlockSize( 512 );
      const IndexType cudaBlocks = Cuda::getNumberOfBlocks( size, cudaBlockSize.x );
      const IndexType cudaGrids = Cuda::getNumberOfGrids( cudaBlocks );
      this->cudaBlockResidue.setSize( min( cudaBlocks, Cuda::getMaxGridSize() ) );
      const IndexType threadsPerGrid = Cuda::getMaxGridSize() * cudaBlockSize.x;

      for( IndexType gridIdx = 0; gridIdx < cudaGrids; gridIdx ++ )
      {
         const IndexType gridOffset = gridIdx * threadsPerGrid;
         const IndexType currentSize = min( size - gridOffset, threadsPerGrid );
         computeErrorKernel<<< cudaBlocks, cudaBlockSize >>>( currentSize,
                                                              tau,
                                                              &_k1[ gridOffset ],
                                                              &_k3[ gridOffset ],
                                                              &_k4[ gridOffset ],
                                                              &_k5[ gridOffset ],
                                                              &_kAux[ gridOffset ] );
         cudaDeviceSynchronize();
         eps = std::max( eps, VectorOperations::getVectorMax( *kAux ) );
      }
#endif
   }
   TNL::MPI::Allreduce( &eps, &maxEps, 1, MPI_MAX, MPI_COMM_WORLD );
   return maxEps;
}

template< typename Problem, typename SolverMonitor >
void Merson< Problem, SolverMonitor >::computeNewTimeLevel( const RealType time,
                                             const RealType tau,
                                             DofVectorPointer& u,
                                             RealType& currentResidue )
{
   RealType localResidue = RealType( 0.0 );
   IndexType size = k1->getSize();
   RealType* _u = u->getData();
   RealType* _k1 = k1->getData();
   RealType* _k4 = k4->getData();
   RealType* _k5 = k5->getData();

   if( std::is_same< DeviceType, Devices::Host >::value )
   {
#ifdef HAVE_OPENMP
#pragma omp parallel for reduction(+:localResidue) firstprivate( size, _u, _k1, _k4, _k5, tau ) if( Devices::Host::isOMPEnabled() )
#endif
      for( IndexType i = 0; i < size; i ++ )
      {
         const RealType add = tau / 6.0 * ( _k1[ i ] + 4.0 * _k4[ i ] + _k5[ i ] );
         _u[ i ] += add;
         localResidue += abs( ( RealType ) add );
      }
      this->problem->applyBoundaryConditions( time, u );
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

         updateUMerson<<< cudaBlocks, cudaBlockSize, sharedMemory >>>( currentSize,
                                                                       tau,
                                                                       &_k1[ gridOffset ],
                                                                       &_k4[ gridOffset ],
                                                                       &_k5[ gridOffset ],
                                                                       &_u[ gridOffset ],
                                                                       this->cudaBlockResidue.getData() );
         localResidue += sum( this->cudaBlockResidue );
         cudaDeviceSynchronize();
      }
      this->problem->applyBoundaryConditions( time, u );

#endif
   }

   localResidue /= tau * ( RealType ) size;
   TNL::MPI::Allreduce( &localResidue, &currentResidue, 1, MPI_SUM, MPI_COMM_WORLD );
/*#ifdef USE_MPI
   TNLMPI::Allreduce( localResidue, currentResidue, 1, MPI_SUM);
#else
   currentResidue=localResidue;
#endif*/

}

template< typename Problem, typename SolverMonitor >
void Merson< Problem, SolverMonitor >::writeGrids( const DofVectorPointer& u )
{
   std::cout << "Writing Merson solver grids ...";
   File( "Merson-u.tnl", std::ios_base::out ) << *u;
   File( "Merson-k1.tnl", std::ios_base::out ) << *k1;
   File( "Merson-k2.tnl", std::ios_base::out ) << *k2;
   File( "Merson-k3.tnl", std::ios_base::out ) << *k3;
   File( "Merson-k4.tnl", std::ios_base::out ) << *k4;
   File( "Merson-k5.tnl", std::ios_base::out ) << *k5;
   std::cout << " done. PRESS A KEY." << std::endl;
   getchar();
}

#ifdef HAVE_CUDA

template< typename RealType, typename Index >
__global__ void computeK2Arg( const Index size,
                              const RealType tau,
                              const RealType* u,
                              const RealType* k1,
                              RealType* k2_arg )
{
   int i = blockIdx. x * blockDim. x + threadIdx. x;
   if( i < size )
      k2_arg[ i ] = u[ i ] + tau * ( 1.0 / 3.0 * k1[ i ] );
}

template< typename RealType, typename Index >
__global__ void computeK3Arg( const Index size,
                              const RealType tau,
                              const RealType* u,
                              const RealType* k1,
                              const RealType* k2,
                              RealType* k3_arg )
{
   Index i = blockIdx. x * blockDim. x + threadIdx. x;
   if( i < size )
      k3_arg[ i ] = u[ i ] + tau * 1.0 / 6.0 * ( k1[ i ] + k2[ i ] );
}

template< typename RealType, typename Index >
__global__ void computeK4Arg( const Index size,
                              const RealType tau,
                              const RealType* u,
                              const RealType* k1,
                              const RealType* k3,
                              RealType* k4_arg )
{
   Index i = blockIdx. x * blockDim. x + threadIdx. x;
   if( i < size )
      k4_arg[ i ] = u[ i ] + tau * ( 0.125 * k1[ i ] + 0.375 * k3[ i ] );
}

template< typename RealType, typename Index >
__global__ void computeK5Arg( const Index size,
                              const RealType tau,
                              const RealType* u,
                              const RealType* k1,
                              const RealType* k3,
                              const RealType* k4,
                              RealType* k5_arg )
{
   Index i = blockIdx. x * blockDim. x + threadIdx. x;
   if( i < size )
      k5_arg[ i ] = u[ i ] + tau * ( 0.5 * k1[ i ] - 1.5 * k3[ i ] + 2.0 * k4[ i ] );
}

template< typename RealType, typename Index >
__global__ void computeErrorKernel( const Index size,
                                    const RealType tau,
                                    const RealType* k1,
                                    const RealType* k3,
                                    const RealType* k4,
                                    const RealType* k5,
                                    RealType* err )
{
   Index i = blockIdx. x * blockDim. x + threadIdx. x;
   if( i < size )
      err[ i ] = 1.0 / 3.0 *  tau * abs( 0.2 * k1[ i ] +
                                        -0.9 * k3[ i ] +
                                         0.8 * k4[ i ] +
                                        -0.1 * k5[ i ] );
}

template< typename RealType, typename Index >
__global__ void updateUMerson( const Index size,
                               const RealType tau,
                               const RealType* k1,
                               const RealType* k4,
                               const RealType* k5,
                               RealType* u,
                               RealType* cudaBlockResidue )
{
   extern __shared__ void* d_u[];
   RealType* du = ( RealType* ) d_u;
   const Index blockOffset = blockIdx. x * blockDim. x;
   const Index i = blockOffset  + threadIdx. x;
   if( i < size )
      u[ i ] += du[ threadIdx.x ] = 1.0 / 6.0 * tau * ( k1[ i ] + 4.0 * k4[ i ] + k5[ i ] );
   else
      du[ threadIdx.x ] = 0.0;
   du[ threadIdx.x ] = abs( du[ threadIdx.x ] );
   __syncthreads();

   const Index rest = size - blockOffset;
   Index n =  rest < blockDim.x ? rest : blockDim.x;

   computeBlockResidue( du,
                        cudaBlockResidue,
                        n );
}

#endif

} // namespace Benchmarks
} // namespace TNL
