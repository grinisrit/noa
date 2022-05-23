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

   /***
    * Create segments with given segments sizes.
    */
   TNL::Containers::Vector< int, Device > segmentsSizes{ 1, 2, 3, 4, 5 };
   SegmentsType segments( segmentsSizes );
   std::cout << "Segments sizes are: " << segments << std::endl;

   /***
    * Allocate array for the segments;
    */
   TNL::Containers::Array< double, Device > data( segments.getStorageSize(), 0.0 );

   /***
    * Insert data into particular segments.
    */
   auto data_view = data.getView();
   segments.forAllElements( [=] __cuda_callable__ ( int segmentIdx, int localIdx, int globalIdx ) mutable {
      if( localIdx <= segmentIdx )
         data_view[ globalIdx ] = segmentIdx;
   } );

   /***
    * Print the data managed by the segments.
    */
   auto fetch = [=] __cuda_callable__ ( int globalIdx ) -> double { return data_view[ globalIdx ]; };
   printSegments( segments, fetch, std::cout );
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
