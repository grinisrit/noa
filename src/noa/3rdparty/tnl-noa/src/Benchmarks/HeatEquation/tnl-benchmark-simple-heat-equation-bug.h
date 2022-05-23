#pragma once

#include <iostream>
#include <chrono>
#include <stdio.h>
#include <fstream>
#include <iomanip>


class GridEntity
{
   public:
      
      __device__ inline
      GridEntity()
      : //e1( this ), e2( this ), e3( this )
        // ,
        eidx1( -1 ), eidx2( -1 ), eidx3( -1 )        
      {
      }
                  
   protected:
      
      
      int eidx1, eidx2, eidx3;
      
      GridEntity *e1, *e2, *e3;
      
};

template< typename GridEntity >
__global__ void testKernel()
{   
   GridEntity entity;
}

int main( int argc, char* argv[] )
{
   const int gridXSize( 256 );
   const int gridYSize( 256 );        
   dim3 cudaBlockSize( 16, 16 );
   dim3 cudaGridSize( gridXSize / 16 + ( gridXSize % 16 != 0 ),
                      gridYSize / 16 + ( gridYSize % 16 != 0 ) );
         
   int iteration( 0 );
   auto t_start = std::chrono::high_resolution_clock::now();
   while( iteration < 10000 )
   {
      testKernel< GridEntity ><<< cudaGridSize, cudaBlockSize >>>();
      cudaDeviceSynchronize();
      iteration++;
   }
   auto t_stop = std::chrono::high_resolution_clock::now();   
   
   std::cout << "Elapsed time = "
             << std::chrono::duration<double, std::milli>(t_stop-t_start).count() << std::endl;
   
   return EXIT_SUCCESS;   
}
