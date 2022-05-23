#include "../../src/bitonicSort/bitonicSort.h"

#include "../benchmarker.cpp"
#include "../measure.cu"

template<typename Value>
void sorter(ArrayView<Value, Devices::Cuda> arr)
{
    bitonicSort(arr);
}