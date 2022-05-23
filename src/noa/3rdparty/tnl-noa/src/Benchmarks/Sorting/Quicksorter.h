#include "../../src/quicksort/quicksort.cuh"

#include "../benchmarker.cpp"
#include "../measure.cu"

template<typename Value>
void sorter(ArrayView<Value, Devices::Cuda> arr)
{
    quicksort(arr);
}