#include <iostream>
#include <functional>
#include <TNL/Containers/Vector.h>
#include <TNL/Algorithms/Segments/CSR.h>
#include <TNL/Devices/Host.h>
#include <TNL/Devices/Cuda.h>

template< typename Device >
void SegmentsExample()
{
   using SegmentsType = typename TNL::Algorithms::Segments::CSR< Device, int >;
   using SegmentViewType = typename SegmentsType::SegmentView;

   /***
    * Create segments with given segments sizes.
    */
   SegmentsType segments{ 1, 2, 3, 4, 5 };
   std::cout << "Segments sizes are: " << segments << std::endl;

   /***
    * Print the elemets mapping using segment view.
    */
   std::cout << "Elements mapping:" << std::endl;
   segments.sequentialForAllSegments( [] __cuda_callable__ ( const SegmentView segment ) {
      printf( "Segment idx. %d: \n", segments.getSegmentIndex() );                 // printf works even in GPU kernels
      for( auto element : segment )
         printf( "%d -> %d  ", element.localIndex(), element.globalIndex() );
   } );

}

int main( int argc, char* argv[] )
{
   std::cout << "Example of CSR segments on host: " << std::endl;
   SegmentsExample< TNL::Devices::Host >();

#ifdef HAVE_CUDA
   std::cout << "Example of CSR segments on CUDA GPU: " << std::endl;
   SegmentsExample< TNL::Devices::Cuda >();
#endif
   return EXIT_SUCCESS;
}
