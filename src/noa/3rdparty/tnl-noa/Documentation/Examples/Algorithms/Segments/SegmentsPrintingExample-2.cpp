#include <iostream>
#include <TNL/Containers/Vector.h>
#include <TNL/Algorithms/Segments/CSR.h>
#include <TNL/Algorithms/Segments/Ellpack.h>
#include <TNL/Algorithms/Segments/ChunkedEllpack.h>
#include <TNL/Algorithms/Segments/BiEllpack.h>
#include <TNL/Devices/Host.h>
#include <TNL/Devices/Cuda.h>

template< typename Segments >
void SegmentsExample()
{
   using Device = typename Segments::DeviceType;

   /***
    * Create segments with given segments sizes.
    */
   TNL::Containers::Vector< int, Device > sizes{ 1, 2, 3, 4, 5 };
   Segments segments( sizes );
   std::cout << "Segments sizes are: " << segments << std::endl;

   /***
    * Allocate array for the segments;
    */
   TNL::Containers::Array< double, Device > data( segments.getStorageSize(), 0.0 );
   data.forAllElements( [=] __cuda_callable__ ( int idx, double& value ) {
      value = idx;
   } );

   /***
    * Print the data managed by the segments.
    */
   auto data_view = data.getView();
   auto fetch = [=] __cuda_callable__ ( int globalIdx ) -> double { return data_view[ globalIdx ]; };
   printSegments( segments, fetch, std::cout ) << std::endl;
}

int main( int argc, char* argv[] )
{
   std::cout << "Example of CSR segments on host: " << std::endl;
   SegmentsExample< TNL::Algorithms::Segments::CSR< TNL::Devices::Host, int > >();

   std::cout << "Example of Ellpack segments on host: " << std::endl;
   SegmentsExample< TNL::Algorithms::Segments::Ellpack< TNL::Devices::Host, int > >();

   std::cout << "Example of ChunkedEllpack segments on host: " << std::endl;
   SegmentsExample< TNL::Algorithms::Segments::ChunkedEllpack< TNL::Devices::Host, int > >();

   std::cout << "Example of BiEllpack segments on host: " << std::endl;
   SegmentsExample< TNL::Algorithms::Segments::BiEllpack< TNL::Devices::Host, int > >();

#ifdef HAVE_CUDA
   std::cout << "Example of CSR segments on host: " << std::endl;
   SegmentsExample< TNL::Algorithms::Segments::CSR< TNL::Devices::Cuda, int > >();

   std::cout << "Example of Ellpack segments on host: " << std::endl;
   SegmentsExample< TNL::Algorithms::Segments::Ellpack< TNL::Devices::Cuda, int > >();

   std::cout << "Example of ChunkedEllpack segments on host: " << std::endl;
   SegmentsExample< TNL::Algorithms::Segments::ChunkedEllpack< TNL::Devices::Cuda, int > >();

   std::cout << "Example of BiEllpack segments on host: " << std::endl;
   SegmentsExample< TNL::Algorithms::Segments::BiEllpack< TNL::Devices::Cuda, int > >();
#endif
   return EXIT_SUCCESS;
}
