#include <iostream>
#include <functional>
#include <TNL/Containers/Vector.h>
#include <TNL/Algorithms/Segments/CSR.h>
#include <TNL/Algorithms/Segments/Ellpack.h>
#include <TNL/Devices/Host.h>
#include <TNL/Devices/Cuda.h>

template< typename Segments >
void SegmentsExample()
{
   using DeviceType = typename Segments::DeviceType;
   using IndexType = typename Segments::IndexType;

   /***
    * Create segments with given segments sizes.
    */
   Segments segments{ 1, 2, 3, 4, 5 };
   std::cout << "Segments sizes are: " << segments << std::endl;

   /***
    * Allocate array for the segments;
    */
   TNL::Containers::Array< double, DeviceType > data( segments.getStorageSize(), 0.0 );

   /***
    * Insert data into particular segments.
    */
   auto data_view = data.getView();
   segments.forAllElements( [=] __cuda_callable__ ( IndexType segmentIdx, IndexType localIdx, IndexType globalIdx ) mutable {
      if( localIdx <= segmentIdx )
         data_view[ globalIdx ] = segmentIdx;
   } );

   /***
    * Print the data managed by the segments.
    */
   auto fetch = [=] __cuda_callable__ ( IndexType globalIdx ) -> double { return data_view[ globalIdx ]; };
   printSegments( segments, fetch, std::cout );

   /***
    * Compute sums of elements in particular segments.
    */
   TNL::Containers::Vector< double, DeviceType, IndexType > sums( segments.getSegmentsCount() );
   auto sums_view = sums.getView();
   auto sum_fetch = [=] __cuda_callable__ ( IndexType segmentIdx, IndexType localIdx, IndexType globalIdx, bool& compute ) -> double {
      return data_view[ globalIdx ];
   };
   auto keep = [=] __cuda_callable__ ( const IndexType& segmentIdx, const double& value ) mutable {
      sums_view[ segmentIdx ] = value;
   };
   segments.reduceAllSegments( sum_fetch, std::plus<>{}, keep, 0.0 );
   std::cout << "The sums are: " << sums << std::endl;
}

int main( int argc, char* argv[] )
{
   using HostCSR = TNL::Algorithms::Segments::CSR< TNL::Devices::Host, int >;
   using HostEllpack = TNL::Algorithms::Segments::Ellpack< TNL::Devices::Host, int >;
   using CudaCSR = TNL::Algorithms::Segments::CSR< TNL::Devices::Cuda, int >;
   using CudaEllpack = TNL::Algorithms::Segments::Ellpack< TNL::Devices::Cuda, int >;


   std::cout << "Example of CSR segments on host: " << std::endl;
   SegmentsExample< HostCSR >();

   std::cout << "Example of Ellpack segments on host: " << std::endl;
   SegmentsExample< HostEllpack >();

#ifdef HAVE_CUDA
   std::cout << "Example of CSR segments on CUDA GPU: " << std::endl;
   SegmentsExample< CudaCSR >();

   std::cout << "Example of Ellpack segments on CUDA GPU: " << std::endl;
   SegmentsExample< CudaEllpack >();
#endif
   return EXIT_SUCCESS;
}
