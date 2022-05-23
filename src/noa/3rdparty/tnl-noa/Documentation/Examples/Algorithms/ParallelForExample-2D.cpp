#include <iostream>
#include <TNL/Containers/Vector.h>
#include <TNL/Algorithms/ParallelFor.h>

using namespace TNL;
using namespace TNL::Containers;
using namespace TNL::Algorithms;

template< typename Device >
void initMeshFunction( const int xSize,
                       const int ySize,
                       Vector< double, Device >& v,
                       const double& c )
{
   auto view = v.getView();
   auto init = [=] __cuda_callable__ ( int i, int j ) mutable
   {
      view[ j * xSize + i ] = c;
   };
   ParallelFor2D< Device >::exec( 0, 0, xSize, ySize, init );
}

int main( int argc, char* argv[] )
{
   /***
    * Define dimensions of 2D mesh function.
    */
   const int xSize( 10 ), ySize( 10 );
   const int size = xSize * ySize;

   /***
    * Firstly, test the mesh function initiation on CPU.
    */
   Vector< double, Devices::Host > host_v;
   initMeshFunction( xSize, ySize, host_v, 1.0 );

   /***
    * And then also on GPU.
    */
#ifdef HAVE_CUDA
   Vector< double, Devices::Cuda > cuda_v( size );
   initMeshFunction( xSize, ySize, cuda_v, 1.0 );
#endif
   return EXIT_SUCCESS;
}
