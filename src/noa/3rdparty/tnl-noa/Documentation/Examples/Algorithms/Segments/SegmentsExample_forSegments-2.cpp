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
   SegmentsType segments{ 1, 2, 3, 4, 5 };

   /***
    * Allocate array for the segments;
    */
   TNL::Containers::Array< double, Device > data( segments.getStorageSize(), 0.0 );

   /***
    * Insert data into particular segments.
    */
   auto data_view = data.getView();
   segments.forAllElements( [=] __cuda_callable__ ( int segmentIdx, int localIdx, int globalIdx ) mutable {
      data_view[ globalIdx ] = localIdx + 1;
   } );

   /***
    * Print the data by the segments.
    */
   std::cout << "Values of elements after intial setup: " << std::endl;
   auto fetch = [=] __cuda_callable__ ( int globalIdx ) -> double { return data_view[ globalIdx ]; };
   printSegments( segments, fetch, std::cout );

   /***
    * Divide elements in each segment by a sum of all elements in the segment
    */
   using SegmentViewType = typename SegmentsType::SegmentViewType;
   segments.forAllSegments( [=] __cuda_callable__ ( const SegmentViewType& segment ) mutable {
      // Compute the sum first ...
      double sum = 0.0;
      for( auto element : segment )
         if( element.localIndex() <= element.segmentIndex() )
            sum += data_view[ element.globalIndex() ];
      // ... divide all elements.
      for( auto element : segment )
         if( element.localIndex() <= element.segmentIndex() )
            data_view[ element.globalIndex() ] /= sum;
   } );

   /***
    * Print the data managed by the segments.
    */
   std::cout << "Value of elements after dividing by sum in each segment:" << std::endl;
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
