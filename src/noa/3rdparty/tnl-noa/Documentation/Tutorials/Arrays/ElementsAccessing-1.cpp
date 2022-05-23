#include <iostream>
#include <TNL/Containers/Array.h>
#include <TNL/Containers/ArrayView.h>

using namespace TNL;
using namespace TNL::Containers;

__global__ void initKernel( ArrayView< float, Devices::Cuda > view )
{
   int tid = threadIdx.x;
   if( tid < view.getSize() )
      view[ tid ] = -tid;
}

int main( int argc, char* argv[] )
{
   /****
    * Create new arrays on both host and device
    */
   const int size = 5;
   Array< float, Devices::Host > host_array( size );
   Array< float, Devices::Cuda > device_array( size );

   /****
    * Initiate the host array
    */
   for( int i = 0; i < size; i++ )
      host_array[ i ] = i;

   /****
    * Prepare array view for the device array - we will pass it to a CUDA kernel.
    * NOTE: Better way is to use ArrayView::evaluate or ParallelFor, this is just
    * an example.
    */
   auto device_view = device_array.getView();

   /****
    * Call CUDA kernel to initiate the array on the device
    */
   initKernel<<< 1, size >>>( device_view );

   /****
    * Print the results
    */
   std::cout << " host_array = " << host_array << std::endl;
   std::cout << " device_array = " << device_array << std::endl;
}

