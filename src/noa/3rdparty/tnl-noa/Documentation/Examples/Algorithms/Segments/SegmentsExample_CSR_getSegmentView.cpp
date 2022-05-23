#include <iostream>
#include <functional>
#include <TNL/Containers/Vector.h>
#include <TNL/Algorithms/Segments/CSR.h>
#include <TNL/Algorithms/SequentialFor.h>
#include <TNL/Devices/Host.h>
#include <TNL/Devices/Cuda.h>

template< typename Device >
void SegmentsExample()
{
   using SegmentsType = typename TNL::Algorithms::Segments::CSR< Device, int >;
   using SegmentView = typename SegmentsType::SegmentViewType;

   /***
    * Create segments with given segments sizes.
    */
   const int size( 5 );
   SegmentsType segments{ 1, 2, 3, 4, 5 };
   auto view = segments.getView();

   /***
    * Print the elemets mapping using segment view.
    */
   std::cout << "Mapping of local indexes to global indexes:" << std::endl;

   auto f = [=] __cuda_callable__ ( int segmentIdx ) {
      printf( "Segment idx. %d: ", segmentIdx );                 // printf works even in GPU kernels
      auto segment = view.getSegmentView( segmentIdx );
      for( auto element : segment )
         printf( "%d -> %d \t", element.localIndex(), element.globalIndex() );
      printf( "\n" );
   };
   TNL::Algorithms::SequentialFor< Device >::exec( 0, size, f );
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
