#ifdef HAVE_CUDA_SAMPLES
#include <6_Advanced/sortingNetworks/bitonicSort.cu>
#endif
#include <TNL/Containers/Array.h>

namespace TNL {

struct NvidiaBitonicSort
{
   static void sort( Containers::ArrayView< int, Devices::Cuda >& view )
   {
#ifdef HAVE_CUDA_SAMPLES
      Containers::Array<int, Devices::Cuda> arr;
      arr = view;
      bitonicSort((unsigned *)view.getData(), (unsigned *)arr.getData(),
                  (unsigned *)view.getData(), (unsigned *)arr.getData(),
                  1, arr.getSize(), 1);
      cudaDeviceSynchronize();
#endif
   }
};

} // namespace TNL
