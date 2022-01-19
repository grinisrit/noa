// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once
#include <TNL/Containers/Array.h>
#include <TNL/Algorithms/Sorting/detail/blockBitonicSort.h>
#include <TNL/Algorithms/Sorting/detail/helpers.h>

namespace TNL {
    namespace Algorithms {
        namespace Sorting {

#ifdef HAVE_CUDA

/**
 * this kernel simulates 1 exchange
 * splits input arr that is bitonic into 2 bitonic sequences
 */
template <typename Value, typename CMP>
__global__ void bitonicMergeGlobal(TNL::Containers::ArrayView<Value, TNL::Devices::Cuda> arr,
                                   CMP Cmp,
                                   int monotonicSeqLen, int bitonicLen)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    int part = i / (bitonicLen / 2); //computes which sorting block this thread belongs to

    //the index of 2 elements that should be compared and swapped
    int s = part * bitonicLen + (i & ((bitonicLen / 2) - 1));
    int e = s + bitonicLen / 2;
    if (e >= arr.getSize()) //arr[e] is virtual padding and will not be exchanged with
        return;

    int partsInSeq = monotonicSeqLen / bitonicLen;
    //calculate the direction of swapping
    int monotonicSeqIdx = part / partsInSeq;
    bool ascending = (monotonicSeqIdx & 1) != 0;
    if ((monotonicSeqIdx + 1) * monotonicSeqLen >= arr.getSize()) //special case for part with no "partner" to be merged with in next phase
        ascending = true;

    cmpSwap(arr[s], arr[e], ascending, Cmp);
}

//---------------------------------------------
//---------------------------------------------

/**
 * simulates many layers of merge
 * turns input that is a bitonic sequence into 1 monotonic sequence
 *
 * this version uses shared memory to do the operations
 * */
template <typename Value, typename CMP>
__global__ void bitonicMergeSharedMemory(TNL::Containers::ArrayView<Value, TNL::Devices::Cuda> arr,
                                         CMP Cmp,
                                         int monotonicSeqLen, int bitonicLen)
{
    extern __shared__ int externMem[];
    Value *sharedMem = (Value *)externMem;

    int sharedMemLen = 2 * blockDim.x;

    //1st index and last index of subarray that this threadBlock should merge
    int myBlockStart = blockIdx.x * sharedMemLen;
    int myBlockEnd = TNL::min(arr.getSize(), myBlockStart + sharedMemLen);

    //copy from globalMem into sharedMem
    for (int i = threadIdx.x; myBlockStart + i < myBlockEnd; i += blockDim.x)
        sharedMem[i] = arr[myBlockStart + i];
    __syncthreads();

    //------------------------------------------
    //bitonic activity
    {
        //calculate the direction of swapping
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        int part = i / (bitonicLen / 2);
        int partsInSeq = monotonicSeqLen / bitonicLen;
        int monotonicSeqIdx = part / partsInSeq;

        bool ascending = (monotonicSeqIdx & 1) != 0;
        //special case for parts with no "partner"
        if ((monotonicSeqIdx + 1) * monotonicSeqLen >= arr.getSize())
            ascending = true;
        //------------------------------------------

        //do bitonic merge
        for (; bitonicLen > 1; bitonicLen /= 2)
        {
            //calculates which 2 indexes will be compared and swap
            int part = threadIdx.x / (bitonicLen / 2);
            int s = part * bitonicLen + (threadIdx.x & ((bitonicLen / 2) - 1));
            int e = s + bitonicLen / 2;

            if (e < myBlockEnd - myBlockStart) //not touching virtual padding
                cmpSwap(sharedMem[s], sharedMem[e], ascending, Cmp);
            __syncthreads();
        }
    }

    //------------------------------------------

    //writeback to global memory
    for (int i = threadIdx.x; myBlockStart + i < myBlockEnd; i += blockDim.x)
        arr[myBlockStart + i] = sharedMem[i];
}


/**
 * entrypoint for bitonicSort_Block
 * sorts @param arr in alternating order to create bitonic sequences
 * sharedMem has to be able to store at least blockDim.x*2 elements
 * */
template <typename Value, typename CMP>
__global__ void bitoniSort1stStepSharedMemory(TNL::Containers::ArrayView<Value, TNL::Devices::Cuda> arr, CMP Cmp)
{
    extern __shared__ int externMem[];

    Value * sharedMem = (Value *)externMem;
    int sharedMemLen = 2*blockDim.x;

    int myBlockStart = blockIdx.x * sharedMemLen;
    int myBlockEnd = TNL::min(arr.getSize(), myBlockStart+sharedMemLen);

    //copy from globalMem into sharedMem
    for (int i = threadIdx.x; myBlockStart + i < myBlockEnd; i += blockDim.x)
        sharedMem[i] = arr[myBlockStart + i];
    __syncthreads();

    //------------------------------------------
    //bitonic activity
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        int paddedSize = closestPow2(myBlockEnd - myBlockStart);

        for (int monotonicSeqLen = 2; monotonicSeqLen <= paddedSize; monotonicSeqLen *= 2)
        {
            //calculate the direction of swapping
            int monotonicSeqIdx = i / (monotonicSeqLen/2);
            bool ascending = (monotonicSeqIdx & 1) != 0;
            if ((monotonicSeqIdx + 1) * monotonicSeqLen >= arr.getSize()) //special case for parts with no "partner"
                ascending = true;

            for (int len = monotonicSeqLen; len > 1; len /= 2)
            {
                //calculates which 2 indexes will be compared and swap
                int part = threadIdx.x / (len / 2);
                int s = part * len + (threadIdx.x & ((len / 2) - 1));
                int e = s + len / 2;

                if(e < myBlockEnd - myBlockStart) //touching virtual padding
                    cmpSwap(sharedMem[s], sharedMem[e], ascending, Cmp);
                __syncthreads();
            }
        }
    }

    //writeback to global memory
    for (int i = threadIdx.x; myBlockStart + i < myBlockEnd; i += blockDim.x)
        arr[myBlockStart + i] = sharedMem[i];
}
#endif


template <typename Value, typename CMP>
void bitonicSortWithShared(TNL::Containers::ArrayView<Value, TNL::Devices::Cuda> view, const CMP &Cmp,
                           int gridDim, int blockDim, int sharedMemLen, int sharedMemSize)
{
#ifdef HAVE_CUDA
    int paddedSize = closestPow2(view.getSize());

    bitoniSort1stStepSharedMemory<<<gridDim, blockDim, sharedMemSize>>>(view, Cmp);
    //now alternating monotonic sequences with bitonicLenght of sharedMemLen

    // \/ has bitonicLength of 2 * sharedMemLen
    for (int monotonicSeqLen = 2 * sharedMemLen; monotonicSeqLen <= paddedSize; monotonicSeqLen *= 2)
    {
        for (int bitonicLen = monotonicSeqLen; bitonicLen > 1; bitonicLen /= 2)
        {
            if (bitonicLen > sharedMemLen)
            {
                bitonicMergeGlobal<<<gridDim, blockDim>>>(
                    view, Cmp, monotonicSeqLen, bitonicLen);
            }
            else
            {
                bitonicMergeSharedMemory<<<gridDim, blockDim, sharedMemSize>>>(
                    view, Cmp, monotonicSeqLen, bitonicLen);

                //simulates sorts until bitonicLen == 2 already, no need to continue this loop
                break;
            }
        }
    }
    cudaDeviceSynchronize();
#endif
}

//---------------------------------------------

template <typename Value, typename CMP>
void bitonicSort(TNL::Containers::ArrayView<Value, TNL::Devices::Cuda> view,
                 const CMP &Cmp,
                 int gridDim, int blockDim)

{
#ifdef HAVE_CUDA
    int paddedSize = closestPow2(view.getSize());

    for (int monotonicSeqLen = 2; monotonicSeqLen <= paddedSize; monotonicSeqLen *= 2)
    {
        for (int bitonicLen = monotonicSeqLen; bitonicLen > 1; bitonicLen /= 2)
        {
            bitonicMergeGlobal<<<gridDim, blockDim>>>(view, Cmp, monotonicSeqLen, bitonicLen);
        }
    }
    cudaDeviceSynchronize();
#endif
}

//---------------------------------------------
template <typename Value, typename CMP>
void bitonicSort(TNL::Containers::ArrayView<Value, TNL::Devices::Cuda> src, int begin, int end, const CMP &Cmp)
{
#ifdef HAVE_CUDA
    auto view = src.getView(begin, end);

    int threadsNeeded = view.getSize() / 2 + (view.getSize() % 2 != 0);

    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);

    const int maxThreadsPerBlock = 512;

    int sharedMemLen = maxThreadsPerBlock * 2;
    size_t sharedMemSize = sharedMemLen * sizeof(Value);

    if (sharedMemSize <= deviceProp.sharedMemPerBlock)
    {
        int blockDim = maxThreadsPerBlock;
        int gridDim = threadsNeeded / blockDim + (threadsNeeded % blockDim != 0);
        bitonicSortWithShared(view, Cmp, gridDim, blockDim, sharedMemLen, sharedMemSize);
    }
    else if (sharedMemSize / 2 <= deviceProp.sharedMemPerBlock)
    {
        int blockDim = maxThreadsPerBlock / 2; //256
        int gridDim = threadsNeeded / blockDim + (threadsNeeded % blockDim != 0);
        sharedMemSize /= 2;
        sharedMemLen /= 2;
        bitonicSortWithShared(view, Cmp, gridDim, blockDim, sharedMemLen, sharedMemSize);
    }
    else
    {
        int gridDim = threadsNeeded / maxThreadsPerBlock + (threadsNeeded % maxThreadsPerBlock != 0);
        bitonicSort(view, Cmp, gridDim, maxThreadsPerBlock);
    }
#endif
}

//---------------------------------------------

template <typename Value, typename CMP>
void bitonicSort(TNL::Containers::ArrayView<Value, TNL::Devices::Cuda> arr, int begin, int end)
{
    bitonicSort(arr, begin, end, [] __cuda_callable__(const Value &a, const Value &b) { return a < b; });
}

template <typename Value, typename CMP>
void bitonicSort(TNL::Containers::ArrayView<Value, TNL::Devices::Cuda> arr, const CMP &Cmp)
{
    bitonicSort(arr, 0, arr.getSize(), Cmp);
}

template <typename Value>
void bitonicSort(TNL::Containers::ArrayView<Value, TNL::Devices::Cuda> arr)
{
    bitonicSort(arr, [] __cuda_callable__(const Value &a, const Value &b) { return a < b; });
}

//---------------------------------------------
template <typename Value, typename CMP>
void bitonicSort(std::vector<Value> &vec, int begin, int end, const CMP &Cmp)
{
    TNL::Containers::Array<Value, TNL::Devices::Cuda> Arr(vec);
    auto view = Arr.getView();
    bitonicSort(view, begin, end, Cmp);

    TNL::Algorithms::MultiDeviceMemoryOperations<TNL::Devices::Host, TNL::Devices::Cuda>::
        copy(vec.data(), view.getData(), view.getSize());
}

template <typename Value>
void bitonicSort(std::vector<Value> &vec, int begin, int end)
{
    bitonicSort(vec, begin, end, [] __cuda_callable__(const Value &a, const Value &b) { return a < b; });
}

template <typename Value, typename CMP>
void bitonicSort(std::vector<Value> &vec, const CMP &Cmp)
{
    bitonicSort(vec, 0, vec.size(), Cmp);
}

template <typename Value>
void bitonicSort(std::vector<Value> &vec)
{
    bitonicSort(vec, [] __cuda_callable__(const Value &a, const Value &b) { return a < b; });
}

template <typename Value>
void bitonicSort( TNL::Containers::Array< Value, TNL::Devices::Host > &vec)
{
    bitonicSort(vec, [] __cuda_callable__(const Value &a, const Value &b) { return a < b; });
}


//---------------------------------------------
//---------------------------------------------

#ifdef HAVE_CUDA
template< typename CMP, typename SWAP>
__global__ void bitonicMergeGlobal(int size, CMP Cmp, SWAP Swap,
                                   int monotonicSeqLen, int bitonicLen)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    int part = i / (bitonicLen / 2); //computes which sorting block this thread belongs to

    //the index of 2 elements that should be compared and swapped
    int s = part * bitonicLen + (i & ((bitonicLen / 2) - 1));
    int e = s + bitonicLen / 2;
    if (e >= size) //arr[e] is virtual padding and will not be exchanged with
        return;

    //calculate the direction of swapping
    int partsInSeq = monotonicSeqLen / bitonicLen;
    int monotonicSeqIdx = part / partsInSeq;
    bool ascending = (monotonicSeqIdx & 1) != 0;
    if ((monotonicSeqIdx + 1) * monotonicSeqLen >= size) //special case for part with no "partner" to be merged with in next phase
        ascending = true;

    if( ascending == Cmp(e, s) )
        Swap(s, e);
}

template< typename CMP, typename SWAP >
void bitonicSort(int begin, int end, const CMP &Cmp, SWAP Swap)
{
    int size = end - begin;
    int paddedSize = closestPow2(size);

    int threadsNeeded = size / 2 + (size % 2 != 0);

    const int maxThreadsPerBlock = 512;
    int threadsPerBlock = maxThreadsPerBlock;
    int blocks = threadsNeeded / threadsPerBlock + (threadsNeeded % threadsPerBlock != 0);

    auto compareWithOffset =
        [=] __cuda_callable__(int i, int j) {
            return Cmp(i + begin, j + begin);
        };

    auto swapWithOffset =
        [=] __cuda_callable__(int i, int j) mutable {
            Swap(i + begin, j + begin);
        };

    for (int monotonicSeqLen = 2; monotonicSeqLen <= paddedSize; monotonicSeqLen *= 2)
    {
        for (int bitonicLen = monotonicSeqLen; bitonicLen > 1; bitonicLen /= 2)
        {
            bitonicMergeGlobal<<<blocks, threadsPerBlock>>>(
                size, compareWithOffset, swapWithOffset,
                monotonicSeqLen, bitonicLen);
        }
    }
    cudaDeviceSynchronize();
}
#endif
        } // namespace Sorting
    } // namespace Algorithms
} // namespace TNL
