#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include <TNL/Containers/Array.h>

namespace TNL {

struct ThrustRadixsort
{
   static void sort( Containers::ArrayView< int, Devices::Cuda >& view )
   {
      thrust::sort(thrust::device, view.getData(), view.getData() + view.getSize());
      cudaDeviceSynchronize();
   }
};

} // namespace TNL
