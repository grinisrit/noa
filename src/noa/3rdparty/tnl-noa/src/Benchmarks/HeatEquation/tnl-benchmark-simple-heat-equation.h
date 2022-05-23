#ifndef TNL_BENCHMARK_SIMPLE_HEAT_EQUATION_H
#define	TNL_BENCHMARK_SIMPLE_HEAT_EQUATION_H

#include <iostream>
#include <cstdio>
#include <TNL/Config/parseCommandLine.h>
#include <TNL/Timer.h>
#include <TNL/Devices/Cuda.h>
#include <TNL/Containers/StaticVector.h>
#include <TNL/Meshes/Grid.h>
#include <TNL/Functions/MeshFunctionView.h>
#include "pure-c-rhs.h"

using namespace std;
using namespace TNL;



#ifdef HAVE_CUDA
template< typename Real, typename Index >
__device__ void computeBlockResidue( Real* du,
                                     Real* blockResidue,
                                     Index n )
{
   if( n == 128 || n ==  64 || n ==  32 || n ==  16 ||
       n ==   8 || n ==   4 || n ==   2 || n == 256 ||
       n == 512 )
    {
       if( blockDim.x >= 512 )
       {
          if( threadIdx.x < 256 )
             du[ threadIdx.x ] += du[ threadIdx.x  + 256 ];
          __syncthreads();
       }
       if( blockDim.x >= 256 )
       {
          if( threadIdx.x < 128 )
             du[ threadIdx.x ] += du[ threadIdx.x  + 128 ];
          __syncthreads();
       }
       if( blockDim.x >= 128 )
       {
          if( threadIdx.x < 64 )
             du[ threadIdx.x ] += du[ threadIdx.x  + 64 ];
          __syncthreads();
       }

       /***
        * This runs in one warp so it is synchronized implicitly.
        */
       if ( threadIdx.x < 32)
       {
          if( blockDim.x >= 64 )
             du[ threadIdx.x ] += du[ threadIdx.x  + 32 ];
          if( blockDim.x >= 32 )
             du[ threadIdx.x ] += du[ threadIdx.x  + 16 ];
          if( blockDim.x >= 16 )
             du[ threadIdx.x ] += du[ threadIdx.x  + 8 ];
          if( blockDim.x >=  8 )
             du[ threadIdx.x ] += du[ threadIdx.x  + 4 ];
          if( blockDim.x >=  4 )
             du[ threadIdx.x ] += du[ threadIdx.x  + 2 ];
          if( blockDim.x >=  2 )
             du[ threadIdx.x ] += du[ threadIdx.x  + 1 ];
       }
    }
    else
    {
       int s;
       if( n >= 512 )
       {
          s = n / 2;
          if( threadIdx.x < s )
             du[ threadIdx.x ] += du[ threadIdx.x + s ];

          __syncthreads();
          if( 2 * s < n  && threadIdx.x == n - 1 )
             du[ 0 ] += du[ threadIdx.x ];
          n = s;
          __syncthreads();
       }
       if( n >= 256 )
       {
          s = n / 2;
          if( threadIdx.x < s )
             du[ threadIdx.x ] += du[ threadIdx.x + s ];

          __syncthreads();
          if( 2 * s < n  && threadIdx.x == n - 1 )
             du[ 0 ] += du[ threadIdx.x ];
          n = s;
          __syncthreads();
       }
       if( n >= 128 )
       {
          s = n / 2;
          if( threadIdx.x < s )
             du[ threadIdx.x ] += du[ threadIdx.x + s ];

          __syncthreads();
          if( 2 * s < n  && threadIdx.x == n - 1 )
             du[ 0 ] += du[ threadIdx.x ];
          n = s;
          __syncthreads();
       }
       if( n >= 64 )
       {
          s = n / 2;
          if( threadIdx.x < s )
             du[ threadIdx.x ] += du[ threadIdx.x + s ];

          __syncthreads();
          if( 2 * s < n  && threadIdx.x == n - 1 )
             du[ 0 ] += du[ threadIdx.x ];
          n = s;
          __syncthreads();

       }
       if( n >= 32 )
       {
          s = n / 2;
          if( threadIdx.x < s )
             du[ threadIdx.x ] += du[ threadIdx.x + s ];

          __syncthreads();
          if( 2 * s < n  && threadIdx.x == n - 1 )
             du[ 0 ] += du[ threadIdx.x ];
          n = s;
          __syncthreads();
       }
       /***
        * This runs in one warp so it is synchronised implicitly.
        */
       if( n >= 16 )
       {
          s = n / 2;
          if( threadIdx.x < s )
             du[ threadIdx.x ] += du[ threadIdx.x + s ];
          if( 2 * s < n  && threadIdx.x == n - 1 )
             du[ 0 ] += du[ threadIdx.x ];
          n = s;
       }
       if( n >= 8 )
       {
          s = n / 2;
          if( threadIdx.x < s )
             du[ threadIdx.x ] += du[ threadIdx.x + s ];
          if( 2 * s < n  && threadIdx.x == n - 1 )
             du[ 0 ] += du[ threadIdx.x ];
          n = s;
       }
       if( n >= 4 )
       {
          s = n / 2;
          if( threadIdx.x < s )
             du[ threadIdx.x ] += du[ threadIdx.x + s ];
          if( 2 * s < n  && threadIdx.x == n - 1 )
             du[ 0 ] += du[ threadIdx.x ];
          n = s;
       }
       if( n >= 2 )
       {
          s = n / 2;
          if( threadIdx.x < s )
             du[ threadIdx.x ] += du[ threadIdx.x + s ];
          if( 2 * s < n  && threadIdx.x == n - 1 )
             du[ 0 ] += du[ threadIdx.x ];
          n = s;
       }
    }

   if( threadIdx.x == 0 )
      blockResidue[ blockIdx.x ] = du[ 0 ];

}


template< typename Real, typename Index >
__global__ void updateKernel( Real* u,
                              Real* aux,
                              Real* cudaBlockResidue,
                              const Index dofs,
                              Real tau )
{
   extern __shared__ Real du[];
   const Index blockOffset = blockIdx.x * blockDim.x;
   Index idx = blockOffset + threadIdx.x;

   if( idx < dofs )
   {
      u[ idx ] += tau * aux[ idx ];
      du[ threadIdx.x ] = fabs( aux[ idx ] );
   }
   else
      du[ threadIdx.x ] = 0.0;

   __syncthreads();

   const Index rest = dofs - blockOffset;
   Index n =  rest < blockDim.x ? rest : blockDim.x;

   computeBlockResidue< Real, Index >( du,
                                       cudaBlockResidue,
                                       n );
}

template< typename Real, typename Index >
bool writeFunction(
   const char* fileName,
   const Real* data,
   const Index xSize,
   const Index ySize,
   const Real& hx,
   const Real& hy,
   const Real& originX,
   const Real& originY )
{
   std::fstream file;
   file.open( fileName, std::ios::out );
   if( ! file )
   {
      std::cerr << "Unable to open file " << fileName << "." << std::endl;
      return false;
   }
   for( Index i = 0; i < xSize; i++ )
   {
      for( Index j = 0; j < ySize; j++ )
         file << i * hx - originX << " " << j * hy - originY << " " << data[ j * xSize + i ] <<std::endl;
      file <<std::endl;
   }
   return true;
}

template< typename Real, typename Index >
bool solveHeatEquationCuda( const Config::ParameterContainer& parameters,
                            Timer& timer,
                            Timer& computationTimer,
                            Timer& updateTimer )
{
   const Real domainXSize = parameters.getParameter< double >( "domain-x-size" );
   const Real domainYSize = parameters.getParameter< double >( "domain-y-size" );
   const Index gridXSize = parameters.getParameter< int >( "grid-x-size" );
   const Index gridYSize = parameters.getParameter< int >( "grid-y-size" );
   const Real sigma = parameters.getParameter< double >( "sigma" );
   Real tau = parameters.getParameter< double >( "time-step" );
   const Real finalTime = parameters.getParameter< double >( "final-time" );
   const bool verbose = parameters.getParameter< bool >( "verbose" );
   const Index dofsCount = gridXSize * gridYSize;
   dim3 cudaUpdateBlocks( dofsCount / 256 + ( dofsCount % 256 != 0 ) );
   dim3 cudaUpdateBlockSize( 256 );

   /****
    * Initiation
    */
   // Workaround for nvcc 10.1.168 - it would modifie the simple expression
   // `new Index[reducedSize]` in the source code to `new (Index[reducedSize])`
   // which is not correct - see e.g. https://stackoverflow.com/a/39671946
   // Thus, the host compiler would spit out some warnings...
   #ifdef __NVCC__
   Real* u = new Real[ static_cast<const Index&>(dofsCount) ];
   Real* aux = new Real[ static_cast<const Index&>(dofsCount) ];
   #else
   Real* u = new Real[ dofsCount ];
   Real* aux = new Real[ dofsCount ];
   #endif
   Real* max_du = new Real[ cudaUpdateBlocks.x ];
   if( ! u || ! aux )
   {
      std::cerr << "I am not able to allocate grid function for grid size " << gridXSize << "x" << gridYSize << "." << std::endl;
      return false;
   }

   const Real hx = domainXSize / ( Real ) gridXSize;
   const Real hy = domainYSize / ( Real ) gridYSize;
   const Real hx_inv = 1.0 / ( hx * hx );
   const Real hy_inv = 1.0 / ( hy * hy );
   if( ! tau )
   {
      tau = hx < hy ? hx * hx : hy * hy;
      if( verbose )
        std::cout << "Setting tau to " << tau << "." << std::endl;
   }

   /****
    * Initial condition
    */
   if( verbose )
     std::cout << "Setting the initial condition ... " << std::endl;
   for( Index j = 0; j < gridYSize; j++ )
      for( Index i = 0; i < gridXSize; i++ )
      {
         const Real x = i * hx - domainXSize / 2.0;
         const Real y = j * hy - domainYSize / 2.0;
         u[ j * gridXSize + i ] = exp( - ( x * x + y * y ) / ( sigma * sigma ) );
      }

   /****
    * Allocate data on the CUDA device
    */
   int cudaErr;
   Real *cuda_u, *cuda_aux, *cuda_max_du;
   cudaMalloc( &cuda_u, dofsCount * sizeof( Real ) );
   cudaMalloc( &cuda_aux, dofsCount * sizeof( Real ) );
   cudaMemcpy( cuda_u, u, dofsCount * sizeof( Real ),  cudaMemcpyHostToDevice );
   cudaMalloc( &cuda_max_du, cudaUpdateBlocks.x * sizeof( Real ) );
   if( ( cudaErr = cudaGetLastError() ) != cudaSuccess )
   {
      std::cerr << "Allocation failed. " << cudaErr << std::endl;
      return false;
   }

   typedef Meshes::Grid< 2, Real, Devices::Cuda, Index > GridType;
   typedef typename GridType::PointType PointType;
   typedef Pointers::SharedPointer<  GridType > GridPointer;
   GridPointer gridPointer;
   gridPointer->setDimensions( gridXSize, gridYSize );
   gridPointer->setDomain( PointType( 0.0, 0.0 ), PointType( domainXSize, domainYSize ) );
   Containers::VectorView< Real, Devices::Cuda, Index > vecU;
   vecU.bind( cuda_u, gridXSize * gridYSize );
   Functions::MeshFunctionView< GridType > meshFunction;
   meshFunction.bind( gridPointer, vecU );
   meshFunction.write( "u", "simple-heat-equation-initial.vti" );

   Containers::VectorView< Real, Devices::Cuda, Index > vecAux;
   vecAux.bind( cuda_aux, gridXSize * gridYSize );
   vecAux.setValue( 0.0 );


   /****
    * Explicit Euler solver
    */
   //const int maxCudaGridSize = tnlCuda::getMaxGridSize();
   dim3 cudaBlockSize( 16, 16 );
   dim3 cudaGridSize( gridXSize / 16 + ( gridXSize % 16 != 0 ),
                      gridYSize / 16 + ( gridYSize % 16 != 0 ) );
   std::cout << "Setting grid size to " << cudaGridSize.x << "," << cudaGridSize.y << "," << cudaGridSize.z << std::endl;
   std::cout << "Setting block size to " << cudaBlockSize.x << "," << cudaBlockSize.y << "," << cudaBlockSize.z << std::endl;

   if( verbose )
     std::cout << "Starting the solver main loop..." <<std::endl;

   timer.reset();
   computationTimer.reset();
   updateTimer.reset();
   timer.start();
   Real time( 0.0 );
   Index iteration( 0 );
   while( time < finalTime )
   {
      computationTimer.start();
      const Real timeLeft = finalTime - time;
      const Real currentTau = tau < timeLeft ? tau : timeLeft;

      if( ! pureCRhsCuda( cudaGridSize, cudaBlockSize, cuda_u, cuda_aux, currentTau, hx_inv, hy_inv, gridXSize, gridYSize) )
         return false;
      computationTimer.stop();

      /*if( iteration % 100 == 0 )
      {
         cudaMemcpy( aux, cuda_aux, dofsCount * sizeof( Real ),  cudaMemcpyDeviceToHost );
         writeFunction( "rhs", aux, gridXSize, gridYSize, hx, hy, domainXSize / 2.0, domainYSize / 2.0 );

         cudaMemcpy( aux, cuda_u, dofsCount * sizeof( Real ),  cudaMemcpyDeviceToHost );
         writeFunction( "u", aux, gridXSize, gridYSize, hx, hy, domainXSize / 2.0, domainYSize / 2.0 );
         getchar();
      }*/

      updateTimer.start();
      /****
       * Update
       */
      //cout << "Update ... " << std::endl;
      updateKernel<<< cudaUpdateBlocks, cudaUpdateBlockSize, cudaUpdateBlockSize.x * sizeof( Real ) >>>( cuda_u, cuda_aux, cuda_max_du, dofsCount, tau );
      if( cudaGetLastError() != cudaSuccess )
      {
         std::cerr << "Update failed." << std::endl;
         return false;
      }

      cudaDeviceSynchronize();
      cudaMemcpy( max_du, cuda_max_du, cudaUpdateBlocks.x * sizeof( Real ), cudaMemcpyDeviceToHost );
      if( ( cudaErr = cudaGetLastError() ) != cudaSuccess )
      {
         std::cerr << "Copying max_du failed. " << cudaErr << std::endl;
         return false;
      }
      for( unsigned int i = 0; i < cudaUpdateBlocks.x; i++ )
         const Real a = fabs( max_du[ i ] );
      updateTimer.stop();

      time += currentTau;
      iteration++;
      if( verbose && iteration % 1000 == 0 )
        std::cout << "Iteration: " << iteration << "\t Time:" << time << "    \r" << flush;
   }
   timer.stop();
   if( verbose )
    std::cout <<std::endl;

   //cudaMemcpy( u, cuda_u, dofsCount * sizeof( Real ), cudaMemcpyDeviceToHost );
   //writeFunction( "final", u, gridXSize, gridYSize, hx, hy, domainXSize / 2.0, domainYSize / 2.0 );

   /****
    * Saving the result
    */
   if( verbose )
     std::cout << "Saving result..." << std::endl;

   meshFunction.write( "u", "simple-heat-equation-result.vti" );

   /***
    * Freeing allocated memory
    */
   if( verbose )
     std::cout << "Freeing allocated memory..." << std::endl;
   delete[] u;
   delete[] aux;
   delete[] max_du;
   cudaFree( cuda_u );
   cudaFree( cuda_aux );
   cudaFree( cuda_max_du );
   return true;
}
#endif

template< typename Real, typename Index >
bool solveHeatEquationHost( const Config::ParameterContainer& parameters,
                            Timer& timer,
                            Timer& computationTimer,
                            Timer& updateTimer )
{
   const Real domainXSize = parameters.getParameter< double >( "domain-x-size" );
   const Real domainYSize = parameters.getParameter< double >( "domain-y-size" );
   const Index gridXSize = parameters.getParameter< int >( "grid-x-size" );
   const Index gridYSize = parameters.getParameter< int >( "grid-y-size" );
   const Real sigma = parameters.getParameter< double >( "sigma" );
   Real tau = parameters.getParameter< double >( "time-step" );
   const Real finalTime = parameters.getParameter< double >( "final-time" );
   const bool verbose = parameters.getParameter< bool >( "verbose" );

   /****
    * Initiation
    */
   Real* __restrict__ u = new Real[ gridXSize * gridYSize ];
   Real* __restrict__ aux = new Real[ gridXSize * gridYSize ];
   if( ! u || ! aux )
   {
      std::cerr << "I am not able to allocate grid function for grid size " << gridXSize << "x" << gridYSize << "." << std::endl;
      return false;
   }
   const Index dofsCount = gridXSize * gridYSize;
   const Real hx = domainXSize / ( Real ) gridXSize;
   const Real hy = domainYSize / ( Real ) gridYSize;
   const Real hx_inv = 1.0 / ( hx * hx );
   const Real hy_inv = 1.0 / ( hy * hy );
   if( ! tau )
   {
      tau = hx < hy ? hx * hx : hy * hy;
      if( verbose )
        std::cout << "Setting tau to " << tau << "." << std::endl;
   }

   /****
    * Initial condition
    */
   if( verbose )
     std::cout << "Setting the initial condition ... " << std::endl;
   for( Index j = 0; j < gridYSize; j++ )
      for( Index i = 0; i < gridXSize; i++ )
      {
         const Real x = i * hx - domainXSize / 2.0;
         const Real y = j * hy - domainYSize / 2.0;
         u[ j * gridXSize + i ] = exp( - sigma * ( x * x + y * y ) );
      }

   /****
    * Explicit Euler solver
    */
   if( verbose )
     std::cout << "Starting the solver main loop..." <<std::endl;

   timer.reset();
   computationTimer.reset();
   updateTimer.reset();
   timer.start();
   Real time( 0.0 );
   Index iteration( 0 );
   while( time < finalTime )
   {
      computationTimer.start();
      const Real timeLeft = finalTime - time;
      const Real currentTau = tau < timeLeft ? tau : timeLeft;

      /****
       * Neumann boundary conditions
       */
      for( Index j = 0; j < gridYSize; j++ )
      {
         aux[ j * gridXSize ] = 0.0; //u[ j * gridXSize + 1 ];
         aux[ j * gridXSize + gridXSize - 1 ] = 0.0; //u[ j * gridXSize + gridXSize - 2 ];
      }
      for( Index i = 0; i < gridXSize; i++ )
      {
         aux[ i ] = 0.0; //u[ gridXSize + i ];
         aux[ ( gridYSize - 1 ) * gridXSize + i ] = 0.0; //u[ ( gridYSize - 2 ) * gridXSize + i ];
      }

      for( Index j = 1; j < gridYSize - 1; j++ )
         for( Index i = 1; i < gridXSize - 1; i++ )
         {
            const Index c = j * gridXSize + i;
            aux[ c ] =  ( ( u[ c - 1 ] - 2.0 * u[ c ] + u[ c + 1 ] ) * hx_inv +
                                     ( u[ c - gridXSize ] - 2.0 * u[ c ] + u[ c + gridXSize ] ) * hy_inv );
         }
      computationTimer.stop();

      updateTimer.start();
      Real residue( 0.0 );
      for( Index i = 0; i < dofsCount; i++ )
      {
         const Real add = currentTau * aux[ i ];
         u[ i ] += add;
         residue += fabs( add );
      }
      updateTimer.stop();

      time += currentTau;
      iteration++;
      if( verbose && iteration % 10000 == 0 )
        std::cout << "Iteration: " << iteration << "\t \t Time:" << time << "    \r" << std::flush;
   }
   timer.stop();
   if( verbose )
    std::cout <<std::endl;


   /****
    * Saving the result
    */
   using GridType = Meshes::Grid< 2, Real, Devices::Host, Index >;
   using PointType = typename GridType::PointType;
   Pointers::SharedPointer<  GridType > gridPointer;
   gridPointer->setDimensions( gridXSize, gridYSize );
   gridPointer->setDomain( PointType( 0.0, 0.0 ), PointType( domainXSize, domainYSize ) );
   Containers::VectorView< Real, Devices::Host, Index > vecU;
   vecU.bind( u, gridXSize * gridYSize );
   Functions::MeshFunctionView< GridType > meshFunction;
   meshFunction.bind( gridPointer, vecU );
   meshFunction.write( "u", "simple-heat-equation-result.vti" );

   /***
    * Freeing allocated memory
    */
   if( verbose )
     std::cout << "Freeing allocated memory..." << std::endl;
   delete[] u;
   delete[] aux;
   return true;
}

int main( int argc, char* argv[] )
{
   Config::ConfigDescription config;
   config.addEntry< String >( "device", "Device the computation will run on.", "host" );
      config.addEntryEnum< String >( "host" );
#ifdef HAVE_CUDA
      config.addEntryEnum< String >( "cuda" );
#endif
   config.addEntry< int >( "grid-x-size", "Grid size along x-axis.", 100 );
   config.addEntry< int >( "grid-y-size", "Grid size along y-axis.", 100 );
   config.addEntry< double >( "domain-x-size", "Domain size along x-axis.", 2.0 );
   config.addEntry< double >( "domain-y-size", "Domain size along y-axis.", 2.0 );
   config.addEntry< double >( "sigma", "Sigma in exponential initial condition.", 2.0 );
   config.addEntry< double >( "time-step", "Time step. By default it is proportional to one over space step square.", 0.0 );
   config.addEntry< double >( "final-time", "Final time of the simulation.", 1.0 );
   config.addEntry< bool >( "verbose", "Verbose mode.", true );

   Config::ParameterContainer parameters;
   if( ! parseCommandLine( argc, argv, config, parameters ) )
      return EXIT_FAILURE;

   Timer timer;
   Timer computationTimer;
   Timer updateTimer;

   String device = parameters.getParameter< String >( "device" );
   if( device == "host" &&
       ! solveHeatEquationHost< double, int >( parameters, timer, computationTimer, updateTimer  ) )
      return EXIT_FAILURE;
#ifdef HAVE_CUDA
   if( device == "cuda" &&
       ! solveHeatEquationCuda< double, int >( parameters, timer, computationTimer, updateTimer ) )
      return EXIT_FAILURE;
#endif

   const bool verbose = parameters.getParameter< bool >( "verbose" );
   if( verbose )
     std::cout <<std::endl << "Finished..." <<std::endl;
   Logger logger( 72, std::cout );
   logger.writeSeparator();
   logger.writeParameter< const char* >( "Compute time:", "" );
   timer.writeLog( logger, 1 );
   logger.writeParameter< const char* >( "Explicit update computation:", "" );
   computationTimer.writeLog( logger, 1 );
   logger.writeParameter< const char* >( "Euler solver update:", "" );
   updateTimer.writeLog( logger, 1 );
   logger.writeSeparator();

   return EXIT_SUCCESS;
}


#endif	/* TNL_BENCHMARK_SIMPLE_HEAT_EQUATION_H */

