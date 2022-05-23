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
   const int size( 5 );
   SegmentsType segments{ 1, 2, 3, 4, 5 };

   /***
    * Allocate array for the segments;
    */
   TNL::Containers::Array< double, Device > data( segments.getStorageSize(), 0.0 );

   /***
    * Insert data into particular segments.
    */
   auto data_view = data.getView();
   segments.forElements( 0, size, [=] __cuda_callable__ ( int segmentIdx, int localIdx, int globalIdx ) mutable {
      if( localIdx <= segmentIdx )
         data_view[ globalIdx ] = segmentIdx;
   } );

   /***
    * Compute sums of elements in each segment.
    */
   TNL::Containers::Vector< double, Device > sums( size );
   auto sums_view = sums.getView();
   auto fetch_full = [=] __cuda_callable__ ( int segmentIdx, int localIdx, int globalIdx, bool& compute ) -> double {
      if( localIdx <= segmentIdx )
         return data_view[ globalIdx ];
      else
      {
         compute = false;
         return 0.0;
      }
   };
   auto fetch_brief = [=] __cuda_callable__ ( int globalIdx, bool& compute ) -> double {
      return data_view[ globalIdx ];
   };

   auto keep = [=] __cuda_callable__ ( int globalIdx, const double& value  ) mutable {
      sums_view[ globalIdx ] = value; };
   segments.reduceAllSegments( fetch_full, std::plus<>{}, keep, 0.0 );
   std::cout << "The sums with full fetch form are: " << sums << std::endl;
   segments.reduceAllSegments( fetch_brief, std::plus<>{}, keep, 0.0 );
   std::cout << "The sums with brief fetch form are: " << sums << std::endl;
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
