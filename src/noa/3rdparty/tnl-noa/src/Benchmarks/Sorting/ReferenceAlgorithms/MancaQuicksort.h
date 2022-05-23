//defines the shared memory size
#define SHARED_LIMIT 1024

#define GIGA 1073741824
/*
 * division of the vector to be sorted in buckets
 * the attributes of the object Block are the parameters of each bucket
 */
template <typename Type>
struct Block
{

    unsigned int begin;
    unsigned int end;

    unsigned int nextbegin;
    unsigned int nextend;

    Type pivot;

    //max of the bucket items
    Type maxPiv;
    //min of the bucket items
    Type minPiv;
    //done indicates that a bucket has been analyzed
    short done;
    short select;
};

template <typename Type>
struct Partition
{

    unsigned int ibucket;
    unsigned int from;
    unsigned int end;
    Type pivot;
};

void CUDA_Quicksort(unsigned int *inData, unsigned int *outData, unsigned int dataSize, unsigned int threads, int Device, double *timer);

void CUDA_Quicksort_64(double *inData, double *outData, unsigned int size, unsigned int threads, int Device, double *timer);

typedef unsigned int Type;

void test_bitonicSort(unsigned int *h_InputKey, unsigned int N, double *timer);
void test_MergeSort(unsigned int *h_SrcKey, unsigned int N, double *timer);
void test_thrustSort(Type *h_data, unsigned int N, double *timer);

typedef unsigned int uint;

size_t scanInclusiveShort(
    uint *d_Dst,
    uint *d_Src,
    uint batchSize,
    uint arrayLength);

size_t scanInclusiveLarge(
    uint *d_Dst,
    uint *d_Src,
    uint batchSize,
    uint arrayLength);

template <typename Type>
inline __device__ void warpScanInclusive2(Type &idata, Type &idata2, volatile Type *s_Data, volatile Type *s_Data2, uint size)
{

    //volatile uint* s_Data2;
    //s_Data2 = s_Data + blockDim.x*2;

    uint pos = 2 * threadIdx.x - (threadIdx.x & (size - 1));
    s_Data[pos] = 0;
    s_Data2[pos] = 0;
    pos += size;
    s_Data[pos] = idata;
    s_Data2[pos] = idata2;

    for (uint offset = 1; offset < size; offset <<= 1)
    {
        s_Data[pos] += s_Data[pos - offset];
        s_Data2[pos] += s_Data2[pos - offset];
    }

    idata = s_Data[pos];
    idata2 = s_Data2[pos];
}

template <typename Type>
inline __device__ void warpScanExclusive2(Type &idata, Type &idata2, volatile Type *s_Data, volatile Type *s_Data2, uint size)
{

    //volatile uint* s_Data2;
    //s_Data2 = s_Data + blockDim.x*2;

    uint pos = 2 * threadIdx.x - (threadIdx.x & (size - 1));
    s_Data[pos] = 0;
    s_Data2[pos] = 0;
    pos += size;
    s_Data[pos] = idata;
    s_Data2[pos] = idata2;

    for (uint offset = 1; offset < size; offset <<= 1)
    {
        s_Data[pos] += s_Data[pos - offset];
        s_Data2[pos] += s_Data2[pos - offset];
    }

    idata = s_Data[pos] - idata;
    idata2 = s_Data2[pos] - idata2;
}

#define LOG2_WARP_SIZE 5U
#define WARP_SIZE (1U << LOG2_WARP_SIZE)

template <typename Type>
inline __device__ void scan1Inclusive2(Type &idata, Type &idata2, volatile Type *s_Data, uint size)
{

    volatile Type *s_Data2;
    s_Data2 = s_Data + blockDim.x * 2;

    if (size > WARP_SIZE)
    {

        //Bottom-level inclusive warp scan
        warpScanInclusive2(idata, idata2, s_Data, s_Data2, WARP_SIZE);

        //Save top Types of each warp for exclusive warp scan
        //sync to wait for warp scans to complete (because s_Data is being overwritten)
        __syncthreads();
        if ((threadIdx.x & (WARP_SIZE - 1)) == (WARP_SIZE - 1))
        {
            s_Data[threadIdx.x >> LOG2_WARP_SIZE] = idata;
            s_Data2[threadIdx.x >> LOG2_WARP_SIZE] = idata2;
        }

        //wait for warp scans to complete
        __syncthreads();
        if (threadIdx.x < (blockDim.x / WARP_SIZE))
        {
            //grab top warp Types
            Type val = s_Data[threadIdx.x];
            Type val2 = s_Data2[threadIdx.x];
            //calculate exclsive scan and write back to shared memory
            warpScanExclusive2(val, val2, s_Data, s_Data2, size >> LOG2_WARP_SIZE);
            s_Data[threadIdx.x] = val;
            s_Data2[threadIdx.x] = val2;
        }

        //return updated warp scans with exclusive scan results
        __syncthreads();
        idata += s_Data[threadIdx.x >> LOG2_WARP_SIZE];
        idata2 += s_Data2[threadIdx.x >> LOG2_WARP_SIZE];
    }
    else
        warpScanInclusive2(idata, idata2, s_Data, s_Data2, size);
}

template <typename Type>
inline __device__ void warpCompareInclusive(Type &idata, Type &idata2, volatile Type *s_Data, uint size)
{

    volatile Type *s_Data2;
    s_Data2 = s_Data + blockDim.x * 2;
    uint pos = 2 * threadIdx.x - (threadIdx.x & (size - 1));
    s_Data[pos] = 0;
    s_Data2[pos] = 0;
    pos += size;
    s_Data[pos] = idata;
    s_Data2[pos] = idata2;

    for (uint offset = 1; offset < size; offset <<= 1)
    {
        s_Data[pos] = max(s_Data[pos], s_Data[pos - offset]);
        s_Data2[pos] = min(s_Data2[pos], s_Data2[pos - offset]);
    }

    idata = s_Data[pos];
    idata2 = s_Data2[pos];
}

template <typename Type>
inline __device__ void compareInclusive(Type &idata, Type &idata2, volatile Type *s_Data, uint size)
{

    volatile Type *s_Data2;
    s_Data2 = s_Data + blockDim.x * 2;
    //Bottom-level inclusive warp scan
    warpCompareInclusive(idata, idata2, s_Data, WARP_SIZE);

    //Save top Types of each warp for exclusive warp scan
    //sync to wait for warp scans to complete (because s_Data is being overwritten)
    __syncthreads();
    if ((threadIdx.x & (WARP_SIZE - 1)) == (WARP_SIZE - 1))
    {
        s_Data[threadIdx.x >> LOG2_WARP_SIZE] = idata;
        s_Data2[threadIdx.x >> LOG2_WARP_SIZE] = idata2;
    }

    //wait for warp scans to complete
    __syncthreads();
    if (threadIdx.x < (blockDim.x / WARP_SIZE))
    {
        //grab top warp Types
        Type val = s_Data[threadIdx.x];
        Type val2 = s_Data2[threadIdx.x];
        //calculate exclsive scan and write back to shared memory
        warpCompareInclusive(val, val2, s_Data, size >> LOG2_WARP_SIZE);
        s_Data[threadIdx.x] = val;
        s_Data2[threadIdx.x] = val2;
    }

    //return updated warp scans with exclusive scan results
    __syncthreads();
    idata = max(idata, s_Data[threadIdx.x >> LOG2_WARP_SIZE]);
    idata2 = min(idata2, s_Data2[threadIdx.x >> LOG2_WARP_SIZE]);
}

#include <assert.h>
#include <common/inc/helper_cuda.h>
#include <6_Advanced/scan/scan_common.h>

//All three kernels run 512 threads per workgroup
//Must be a power of two
#define THREADBLOCK_SIZE 256

////////////////////////////////////////////////////////////////////////////////
// Basic ccan codelets
////////////////////////////////////////////////////////////////////////////////
#if (0)
//Naive inclusive scan: O(N * log2(N)) operations
//Allocate 2 * 'size' local memory, initialize the first half
//with 'size' zeros avoiding if(pos >= offset) condition evaluation
//and saving instructions
inline __device__ uint scan1Inclusive(uint idata, volatile uint *s_Data, uint size)
{
    uint pos = 2 * threadIdx.x - (threadIdx.x & (size - 1));
    s_Data[pos] = 0;
    pos += size;
    s_Data[pos] = idata;

    for (uint offset = 1; offset < size; offset <<= 1)
    {
        __syncthreads();
        uint t = s_Data[pos] + s_Data[pos - offset];
        __syncthreads();
        s_Data[pos] = t;
    }

    return s_Data[pos];
}

inline __device__ uint scan1Exclusive(uint idata, volatile uint *s_Data, uint size)
{
    return scan1Inclusive(idata, s_Data, size) - idata;
}

#else
#define LOG2_WARP_SIZE 5U
#define WARP_SIZE (1U << LOG2_WARP_SIZE)

//Almost the same as naive scan1Inclusive, but doesn't need __syncthreads()
//assuming size <= WARP_SIZE
inline __device__ uint warpScanInclusive(uint idata, volatile uint *s_Data, uint size)
{
    uint pos = 2 * threadIdx.x - (threadIdx.x & (size - 1));
    s_Data[pos] = 0;
    pos += size;
    s_Data[pos] = idata;

    for (uint offset = 1; offset < size; offset <<= 1)
        s_Data[pos] += s_Data[pos - offset];

    return s_Data[pos];
}

inline __device__ uint warpScanExclusive(uint idata, volatile uint *s_Data, uint size)
{
    return warpScanInclusive(idata, s_Data, size) - idata;
}

inline __device__ uint scan1Inclusive(uint idata, volatile uint *s_Data, uint size)
{
    if (size > WARP_SIZE)
    {
        //Bottom-level inclusive warp scan
        uint warpResult = warpScanInclusive(idata, s_Data, WARP_SIZE);

        //Save top elements of each warp for exclusive warp scan
        //sync to wait for warp scans to complete (because s_Data is being overwritten)
        __syncthreads();
        if ((threadIdx.x & (WARP_SIZE - 1)) == (WARP_SIZE - 1))
            s_Data[threadIdx.x >> LOG2_WARP_SIZE] = warpResult;

        //wait for warp scans to complete
        __syncthreads();
        if (threadIdx.x < (THREADBLOCK_SIZE / WARP_SIZE))
        {
            //grab top warp elements
            uint val = s_Data[threadIdx.x];
            //calculate exclsive scan and write back to shared memory
            s_Data[threadIdx.x] = warpScanExclusive(val, s_Data, size >> LOG2_WARP_SIZE);
        }

        //return updated warp scans with exclusive scan results
        __syncthreads();
        return warpResult + s_Data[threadIdx.x >> LOG2_WARP_SIZE];
    }
    else
    {
        return warpScanInclusive(idata, s_Data, size);
    }
}

inline __device__ uint scan1Exclusive(uint idata, volatile uint *s_Data, uint size)
{
    return scan1Inclusive(idata, s_Data, size) - idata;
}

#endif

inline __device__ uint4 scan4Inclusive(uint4 idata4, volatile uint *s_Data, uint size)
{
    //Level-0 exclusive scan
    idata4.y += idata4.x;
    idata4.z += idata4.y;
    idata4.w += idata4.z;

    //Level-1 exclusive scan
    uint oval = scan1Exclusive(idata4.w, s_Data, size / 4);

    idata4.x += oval;
    idata4.y += oval;
    idata4.z += oval;
    idata4.w += oval;

    return idata4;
}

//Exclusive vector scan: the array to be scanned is stored
//in local thread memory scope as uint4
inline __device__ uint4 scan4Exclusive(uint4 idata4, volatile uint *s_Data, uint size)
{
    uint4 odata4 = scan4Inclusive(idata4, s_Data, size);
    odata4.x -= idata4.x;
    odata4.y -= idata4.y;
    odata4.z -= idata4.z;
    odata4.w -= idata4.w;
    return odata4;
}

////////////////////////////////////////////////////////////////////////////////
// Scan kernels
////////////////////////////////////////////////////////////////////////////////
__global__ void scanExclusiveShared(
    uint4 *d_Dst,
    uint4 *d_Src,
    uint size)
{
    __shared__ uint s_Data[2 * THREADBLOCK_SIZE];

    uint pos = blockIdx.x * blockDim.x + threadIdx.x;

    //Load data
    uint4 idata4 = d_Src[pos];

    //Calculate exclusive scan
    uint4 odata4 = scan4Exclusive(idata4, s_Data, size);

    //Write back
    d_Dst[pos] = odata4;
}

//Exclusive scan of top elements of bottom-level scans (4 * THREADBLOCK_SIZE)
__global__ void scanExclusiveShared2(
    uint *d_Buf,
    uint *d_Dst,
    uint *d_Src,
    uint N,
    uint arrayLength)
{
    __shared__ uint s_Data[2 * THREADBLOCK_SIZE];

    //Skip loads and stores for inactive threads of last threadblock (pos >= N)
    uint pos = blockIdx.x * blockDim.x + threadIdx.x;

    //Load top elements
    //Convert results of bottom-level scan back to inclusive
    uint idata = 0;
    if (pos < N)
        idata =
            d_Dst[(4 * THREADBLOCK_SIZE) - 1 + (4 * THREADBLOCK_SIZE) * pos] +
            d_Src[(4 * THREADBLOCK_SIZE) - 1 + (4 * THREADBLOCK_SIZE) * pos];

    //Compute
    uint odata = scan1Exclusive(idata, s_Data, arrayLength);

    //Avoid out-of-bound access
    if (pos < N)
        d_Buf[pos] = odata;
}

//Final step of large-array scan: combine basic inclusive scan with exclusive scan of top elements of input arrays
__global__ void uniformUpdate(
    uint4 *d_Data,
    uint *d_Buffer)
{
    __shared__ uint buf;
    uint pos = blockIdx.x * blockDim.x + threadIdx.x;

    if (threadIdx.x == 0)
        buf = d_Buffer[blockIdx.x];
    __syncthreads();

    uint4 data4 = d_Data[pos];
    data4.x += buf;
    data4.y += buf;
    data4.z += buf;
    data4.w += buf;
    d_Data[pos] = data4;
}

////////////////////////////////////////////////////////////////////////////////
// Interface function
////////////////////////////////////////////////////////////////////////////////
//Derived as 32768 (max power-of-two gridDim.x) * 4 * THREADBLOCK_SIZE
//Due to scanExclusiveShared<<<>>>() 1D block addressing
const uint MAX_BATCH_ELEMENTS = 64 * 1048576;
const uint MIN_SHORT_ARRAY_SIZE = 4;
const uint MAX_SHORT_ARRAY_SIZE = 4 * THREADBLOCK_SIZE;
const uint MIN_LARGE_ARRAY_SIZE = 8 * THREADBLOCK_SIZE;
const uint MAX_LARGE_ARRAY_SIZE = 4 * THREADBLOCK_SIZE * THREADBLOCK_SIZE;

//Internal exclusive scan buffer
static uint *d_Buf;

void initScan(void)
{
    checkCudaErrors(cudaMalloc((void **)&d_Buf, (MAX_BATCH_ELEMENTS / (4 * THREADBLOCK_SIZE)) * sizeof(uint)));
}

void closeScan(void)
{
    checkCudaErrors(cudaFree(d_Buf));
}

static uint factorRadix2(uint &log2L, uint L)
{
    if (!L)
    {
        log2L = 0;
        return 0;
    }
    else
    {
        for (log2L = 0; (L & 1) == 0; L >>= 1, log2L++)
            ;
        return L;
    }
}

static uint iDivUp(uint dividend, uint divisor)
{
    return ((dividend % divisor) == 0) ? (dividend / divisor) : (dividend / divisor + 1);
}

size_t scanExclusiveShort(
    uint *d_Dst,
    uint *d_Src,
    uint batchSize,
    uint arrayLength)
{
    //Check power-of-two factorization
    uint log2L;
    uint factorizationRemainder = factorRadix2(log2L, arrayLength);
    assert(factorizationRemainder == 1);

    //Check supported size range
    assert((arrayLength >= MIN_SHORT_ARRAY_SIZE) && (arrayLength <= MAX_SHORT_ARRAY_SIZE));

    //Check total batch size limit
    assert((batchSize * arrayLength) <= MAX_BATCH_ELEMENTS);

    //Check all threadblocks to be fully packed with data
    assert((batchSize * arrayLength) % (4 * THREADBLOCK_SIZE) == 0);

    scanExclusiveShared<<<(batchSize * arrayLength) / (4 * THREADBLOCK_SIZE), THREADBLOCK_SIZE>>>(
        (uint4 *)d_Dst,
        (uint4 *)d_Src,
        arrayLength);
    getLastCudaError("scanExclusiveShared() execution FAILED\n");

    return THREADBLOCK_SIZE;
}

size_t scanExclusiveLarge(
    uint *d_Dst,
    uint *d_Src,
    uint batchSize,
    uint arrayLength)
{
    //Check power-of-two factorization
    uint log2L;
    uint factorizationRemainder = factorRadix2(log2L, arrayLength);
    assert(factorizationRemainder == 1);

    //Check supported size range
    assert((arrayLength >= MIN_LARGE_ARRAY_SIZE) && (arrayLength <= MAX_LARGE_ARRAY_SIZE));

    //Check total batch size limit
    assert((batchSize * arrayLength) <= MAX_BATCH_ELEMENTS);

    scanExclusiveShared<<<(batchSize * arrayLength) / (4 * THREADBLOCK_SIZE), THREADBLOCK_SIZE>>>(
        (uint4 *)d_Dst,
        (uint4 *)d_Src,
        4 * THREADBLOCK_SIZE);
    getLastCudaError("scanExclusiveShared() execution FAILED\n");

    //Not all threadblocks need to be packed with input data:
    //inactive threads of highest threadblock just don't do global reads and writes
    const uint blockCount2 = iDivUp((batchSize * arrayLength) / (4 * THREADBLOCK_SIZE), THREADBLOCK_SIZE);
    scanExclusiveShared2<<<blockCount2, THREADBLOCK_SIZE>>>(
        (uint *)d_Buf,
        (uint *)d_Dst,
        (uint *)d_Src,
        (batchSize * arrayLength) / (4 * THREADBLOCK_SIZE),
        arrayLength / (4 * THREADBLOCK_SIZE));
    getLastCudaError("scanExclusiveShared2() execution FAILED\n");

    uniformUpdate<<<(batchSize * arrayLength) / (4 * THREADBLOCK_SIZE), THREADBLOCK_SIZE>>>(
        (uint4 *)d_Dst,
        (uint *)d_Buf);
    getLastCudaError("uniformUpdate() execution FAILED\n");

    return THREADBLOCK_SIZE;
}

__global__ void scanInclusiveShared2(
    uint *d_Buf,
    uint *d_Dst,
    uint N,
    uint arrayLength)
{
    __shared__ uint s_Data[2 * THREADBLOCK_SIZE];

    //Skip loads and stores for inactive threads of last threadblock (pos >= N)
    uint pos = blockIdx.x * blockDim.x + threadIdx.x;

    //Load top elements
    //Convert results of bottom-level scan back to inclusive
    uint idata = 0;
    if (pos < N)
        idata = d_Dst[(4 * THREADBLOCK_SIZE) - 1 + (4 * THREADBLOCK_SIZE) * pos];

    //Compute
    uint odata = scan1Exclusive(idata, s_Data, arrayLength);

    //Avoid out-of-bound access
    if (pos < N)
        d_Buf[pos] = odata;
}

__global__ void scanInclusiveShared(
    uint4 *d_Dst,
    uint4 *d_Src,
    uint size)
{
    __shared__ uint s_Data[2 * THREADBLOCK_SIZE];

    uint pos = blockIdx.x * blockDim.x + threadIdx.x;

    //  if(pos<warpSize*(size/warpSize+1))
    {
        //Load data
        uint4 idata4 = d_Src[pos];

        //Calculate exclusive scan
        uint4 odata4 = scan4Inclusive(idata4, s_Data, size);

        //Write back
        d_Dst[pos] = odata4;
    }
}

size_t scanInclusiveShort(
    uint *d_Dst,
    uint *d_Src,
    uint batchSize,
    uint arrayLength)
{
    //Check power-of-two factorization
    uint log2L;
    uint factorizationRemainder = factorRadix2(log2L, arrayLength);
    assert(factorizationRemainder == 1);

    //Check supported size range
    //  assert( (arrayLength >= MIN_SHORT_ARRAY_SIZE) && (arrayLength <= MAX_SHORT_ARRAY_SIZE) );

    //Check total batch size limit
    // assert( (batchSize * arrayLength) <= MAX_BATCH_ELEMENTS );

    //Check all threadblocks to be fully packed with data
    //assert( (batchSize * arrayLength) % (4 * THREADBLOCK_SIZE) == 0 );
    int blocks = (batchSize * arrayLength + 4 * THREADBLOCK_SIZE - 1) / (4 * THREADBLOCK_SIZE);
    scanInclusiveShared<<<blocks, THREADBLOCK_SIZE>>>(
        (uint4 *)d_Dst,
        (uint4 *)d_Src,
        arrayLength);
    getLastCudaError("scanExclusiveShared() execution FAILED\n");

    return THREADBLOCK_SIZE;
}

size_t scanInclusiveLarge(
    uint *d_Dst,
    uint *d_Src,
    uint batchSize,
    uint arrayLength)
{
    //Check power-of-two factorization
    uint log2L;
    uint factorizationRemainder = factorRadix2(log2L, arrayLength);
    assert(factorizationRemainder == 1);

    //Check supported size range
    //assert( (arrayLength >= MIN_LARGE_ARRAY_SIZE) && (arrayLength <= MAX_LARGE_ARRAY_SIZE) );

    //Check total batch size limit
    //assert( (batchSize * arrayLength) <= MAX_BATCH_ELEMENTS );

    scanInclusiveShared<<<(batchSize * arrayLength) / (4 * THREADBLOCK_SIZE), THREADBLOCK_SIZE>>>(
        (uint4 *)d_Dst,
        (uint4 *)d_Src,
        4 * THREADBLOCK_SIZE);
    getLastCudaError("scanExclusiveShared() execution FAILED\n");

    //Not all threadblocks need to be packed with input data:
    //inactive threads of highest threadblock just don't do global reads and writes
    const uint blockCount2 = iDivUp((batchSize * arrayLength) / (4 * THREADBLOCK_SIZE), THREADBLOCK_SIZE);
    scanInclusiveShared2<<<blockCount2, THREADBLOCK_SIZE>>>(
        (uint *)d_Buf,
        (uint *)d_Dst,
        (batchSize * arrayLength) / (4 * THREADBLOCK_SIZE),
        arrayLength / (4 * THREADBLOCK_SIZE));
    getLastCudaError("scanExclusiveShared2() execution FAILED\n");

    uniformUpdate<<<(batchSize * arrayLength) / (4 * THREADBLOCK_SIZE), THREADBLOCK_SIZE>>>(
        (uint4 *)d_Dst,
        (uint *)d_Buf);
    getLastCudaError("uniformUpdate() execution FAILED\n");

    return THREADBLOCK_SIZE;
}


#include <thrust/scan.h>
#include <common/inc/helper_cuda.h>
#include <common/inc/helper_timer.h>
#include <6_Advanced/scan/scan_common.h>

extern __shared__ uint sMemory[];

__device__ inline double atomicMax(double *address, double val)
{
    unsigned long long int *address_as_ull = (unsigned long long int *)address;
    unsigned long long int assumed;
    unsigned long long int old = *address_as_ull;

    assumed = old;
    old = atomicCAS(address_as_ull,
                    assumed,
                    __double_as_longlong(max(val, __longlong_as_double(assumed))));

    while (assumed != old)
    {
        assumed = old;
        old = atomicCAS(address_as_ull,
                        assumed,
                        __double_as_longlong(max(val, __longlong_as_double(assumed))));
    }
    return __longlong_as_double(old);
}

__device__ inline double atomicMin(double *address, double val)
{
    unsigned long long int *address_as_ull = (unsigned long long int *)address;
    unsigned long long int old = *address_as_ull, assumed;

    assumed = old;
    old = atomicCAS(address_as_ull,
                    assumed,
                    __double_as_longlong(min(val, __longlong_as_double(assumed))));
    while (assumed != old)
    {
        assumed = old;
        old = atomicCAS(address_as_ull,
                        assumed,
                        __double_as_longlong(min(val, __longlong_as_double(assumed))));
    }
    return __longlong_as_double(old);
}

template <typename Type>
__device__ inline void Comparator(

    Type &valA,
    Type &valB,
    uint dir)
{
    Type t;
    if ((valA > valB) == dir)
    {
        t = valA;
        valA = valB;
        valB = t;
    }
}

static __device__ __forceinline__ unsigned int __qsflo(unsigned int word)
{
    unsigned int ret;
    asm volatile("bfind.u32 %0, %1;"
                 : "=r"(ret)
                 : "r"(word));
    return ret;
}

template <typename Type>
__global__ void globalBitonicSort(Type *indata, Type *outdata, Block<Type> *bucket, bool inputSelect)
{
    __shared__ uint shared[1024];

    Type *data;

    Block<Type> cord = bucket[blockIdx.x];

    uint size = cord.end - cord.begin;
    bool select = !(cord.select);

    if (cord.end - cord.begin > 1024 || cord.end - cord.begin == 0)
        return;

    unsigned int bitonicSize = 1 << (__qsflo(size - 1U) + 1);

    if (select)
        data = indata;
    else
        data = outdata;

    //__syncthreads();

    for (int i = threadIdx.x; i < size; i += blockDim.x)
        shared[i] = data[i + cord.begin];

    for (int i = threadIdx.x + size; i < bitonicSize; i += blockDim.x)
        shared[i] = 0xffffffff;

    __syncthreads();

    for (uint size = 2; size < bitonicSize; size <<= 1)
    {
        //Bitonic merge
        uint ddd = 1 ^ ((threadIdx.x & (size / 2)) != 0);
        for (uint stride = size / 2; stride > 0; stride >>= 1)
        {
            __syncthreads();
            uint pos = 2 * threadIdx.x - (threadIdx.x & (stride - 1));
            //if(pos <bitonicSize){
            Comparator(
                shared[pos + 0],
                shared[pos + stride],
                ddd);
            // }
        }
    }

    //ddd == dir for the last bitonic merge step

    for (uint stride = bitonicSize / 2; stride > 0; stride >>= 1)
    {
        __syncthreads();
        uint pos = 2 * threadIdx.x - (threadIdx.x & (stride - 1));
        // if(pos <bitonicSize){
        Comparator(
            shared[pos + 0],
            shared[pos + stride],
            1);
        // }
    }

    __syncthreads();

    // Write back the sorted data to its correct position
    for (int i = threadIdx.x; i < size; i += blockDim.x)
        indata[i + cord.begin] = shared[i];
}

template <typename Type>
__global__ void quick(Type *indata, Type *buffer, Partition<Type> *partition, Block<Type> *bucket)
{

    __shared__ Type sh_out[1024];

    __shared__ uint start1, end1;
    __shared__ uint left, right;

    int tix = threadIdx.x;

    uint start = partition[blockIdx.x].from;
    uint end = partition[blockIdx.x].end;
    Type pivot = partition[blockIdx.x].pivot;
    uint nseq = partition[blockIdx.x].ibucket;

    uint lo = 0;
    uint hi = 0;

    Type lmin = 0xffffffff;
    Type rmax = 0;

    Type d;

    // start read on 1° tile and store the coordinates of the items that must
    // be moved on the left or on the right of the pivot

    if (tix + start < end)
    {
        d = indata[tix + start];

        //count items smaller or bigger than the pivot
        // if d<pivot then ll++ else ll
        lo = (d < pivot) * (lo + 1) + (d >= pivot) * lo;
        // if d>pivot then lr++ else lr
        hi = (d <= pivot) * (hi) + (d > pivot) * (hi + 1);

        lmin = d;
        rmax = d;
    }

    //read and store the coordinates on next tiles for each block
    for (uint i = tix + start + blockDim.x; i < end; i += blockDim.x)
    {
        Type d = indata[i];

        //count items smaller or bigger than the pivot
        lo = (d < pivot) * (lo + 1) + (d >= pivot) * lo;
        hi = (d <= pivot) * (hi) + (d > pivot) * (hi + 1);

        //compute max and min of tile items
        lmin = min(lmin, d);
        rmax = max(rmax, d);
    }

    //compute max and min of every partition

    compareInclusive(rmax, lmin, (Type *)sh_out, blockDim.x);

    __syncthreads();

    if (tix == blockDim.x - 1)
    {
        //compute absolute max and min for the bucket
        atomicMax(&bucket[nseq].maxPiv, rmax);
        atomicMin(&bucket[nseq].minPiv, lmin);
    }

    __syncthreads();

    /*
	 * calculate the coordinates of its assigned item to each thread,
	 * which are necessary to known in which subsequences the item must be copied
	 *
	 */
    scan1Inclusive2(lo, hi, (uint *)sh_out, blockDim.x);
    lo = lo - 1;
    hi = SHARED_LIMIT - hi;

    if (tix == blockDim.x - 1)
    {
        left = lo + 1;
        right = SHARED_LIMIT - hi;

        start1 = atomicAdd(&bucket[nseq].nextbegin, left);
        end1 = atomicSub(&bucket[nseq].nextend, right);
    }

    __syncthreads();

    //thread blocks write on the shared memory the items smaller and bigger than the first tile's pivot
    if (tix + start < end)
    {
        //items smaller than pivot
        if (d < pivot)
        {
            sh_out[lo] = d;
            lo--;
        }

        //items bigger than pivot
        if (d > pivot)
        {
            sh_out[hi] = d;
            hi++;
        }
    }

    //thread blocks write on the shared memory the items smaller and bigger than next tiles' pivot
    for (uint i = start + tix + blockDim.x; i < end; i += blockDim.x)
    {

        Type d = indata[i];
        //items smaller than the pivot
        if (d < pivot)
        {
            sh_out[lo] = d;
            lo--;
        }

        //items bigger than the pivot
        if (d > pivot)
        {
            sh_out[hi] = d;
            hi++;
        }
    }

    __syncthreads();

    //items smaller and bigger than the pivot already sorted in the shared memory are coalesced written on the global memory
    //partial results of each thread block stored on the shared memory are merged together in two subsequences within the global memory
    //coalesced writing of next tiles on the global memory
    for (uint i = tix; i < SHARED_LIMIT; i += blockDim.x)
    {
        if (i < left)
            buffer[start1 + i] = sh_out[i];

        if (i >= SHARED_LIMIT - right)
            buffer[end1 + i - SHARED_LIMIT] = sh_out[i];
    }
}

//this function assigns the attributes to each partition of each bucket
//a thread block is assigned to a specific partition
template <typename Type>
__global__ void partitionAssign(struct Block<Type> *bucket, uint *npartitions, struct Partition<Type> *partition)
{
    int tx = threadIdx.x;
    int bx = blockIdx.x;

    uint beg = bucket[bx].nextbegin;
    uint end = bucket[bx].nextend;
    Type pivot = bucket[bx].pivot;

    uint from;
    uint to;

    if (bx > 0)
    {
        from = npartitions[bx - 1];
        to = npartitions[bx];
    }
    else
    {
        from = 0;
        to = npartitions[bx];
    }

    uint i = tx + from;

    if (i < to)
    {
        uint begin = beg + SHARED_LIMIT * tx;
        partition[i].from = begin;
        partition[i].end = begin + SHARED_LIMIT;
        partition[i].pivot = pivot;
        partition[i].ibucket = bx;
    }

    for (uint i = tx + from + blockDim.x; i < to; i += blockDim.x)
    {
        uint begin = beg + SHARED_LIMIT * (i - from);
        partition[i].from = begin;
        partition[i].end = begin + SHARED_LIMIT;
        partition[i].pivot = pivot;
        partition[i].ibucket = bx;
    }
    __syncthreads();
    if (tx == 0 && to - from > 0)
        partition[to - 1].end = end;
}

//this function enters the pivot value in the central bucket's items
template <typename Type>
__global__ void insertPivot(Type *data, struct Block<Type> *bucket, int nbucket)
{

    Type pivot = bucket[blockIdx.x].pivot;
    uint start = bucket[blockIdx.x].nextbegin;
    uint end = bucket[blockIdx.x].nextend;
    bool is_altered = bucket[blockIdx.x].done;

    if (is_altered && blockIdx.x < nbucket)
        for (uint j = start + threadIdx.x; j < end; j += blockDim.x)
            data[j] = pivot;
}

//this function assigns the new attributes of each bucket
template <typename Type>
__global__ void bucketAssign(Block<Type> *bucket, uint *npartitions, int nbucket, int select)
{

    uint i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < nbucket)
    {
        bool is_altered = bucket[i].done;
        if (is_altered)
        {
            //read on i node
            uint orgbeg = bucket[i].begin;
            uint from = bucket[i].nextbegin;
            uint orgend = bucket[i].end;
            uint end = bucket[i].nextend;
            Type pivot = bucket[i].pivot;
            Type minPiv = bucket[i].minPiv;
            Type maxPiv = bucket[i].maxPiv;

            //compare each bucket's max and min to the pivot
            Type lmaxpiv = min(pivot, maxPiv);
            Type rminpiv = max(pivot, minPiv);

            //write on i+nbucket node
            bucket[i + nbucket].begin = orgbeg;
            bucket[i + nbucket].nextbegin = orgbeg;
            bucket[i + nbucket].nextend = from;
            bucket[i + nbucket].end = from;
            bucket[i + nbucket].pivot = (minPiv + lmaxpiv) / 2;

            //if(select)
            //	bucket[i+nbucket].done   = (from-orgbeg)>1024;// && (minPiv!=maxPiv);
            //else
            bucket[i + nbucket].done = (from - orgbeg) > 1024 && (minPiv != maxPiv);
            bucket[i + nbucket].select = select;
            bucket[i + nbucket].minPiv = ( Type ) 0xffffffffffffffff;
            bucket[i + nbucket].maxPiv = 0;
            //bucket[i+nbucket].finish=false;

            //calculate the number of partitions (npartitions) necessary to the i+nbucket bucket
            if (!bucket[i + nbucket].done)
                npartitions[i + nbucket] = 0;
            else
                npartitions[i + nbucket] = (from - orgbeg + SHARED_LIMIT - 1) / SHARED_LIMIT;

            //write on i node
            bucket[i].begin = end;
            bucket[i].nextbegin = end;
            bucket[i].nextend = orgend;
            bucket[i].pivot = (rminpiv + maxPiv) / 2 + 1;

            //if(select)
            //bucket[i].done   = (orgend-end)>1024;// && (minPiv!=maxPiv);
            //	else
            bucket[i].done = (orgend - end) > 1024 && (minPiv != maxPiv);
            bucket[i].select = select;
            bucket[i].minPiv = ( Type ) 0xffffffffffffffff;
            bucket[i].maxPiv = 0;
            //bucket[i].finish=false;

            //calculate the number of partitions (npartitions) necessary to the i-bucket bucket
            if (!bucket[i].done)
                npartitions[i] = 0;
            else
                npartitions[i] = (orgend - end + SHARED_LIMIT - 1) / SHARED_LIMIT;
        }
    }
}

template <typename Type>
__global__ void init(Type *data, Block<Type> *bucket, uint *npartitions, int size, int nblocks)
{
    uint i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < nblocks)
    {
        bucket[i].nextbegin = 0;
        bucket[i].begin = 0;

        bucket[i].nextend = 0 + size * (i == 0);
        bucket[i].end = 0 + size * (i == 0);
        npartitions[i] = 0;
        bucket[i].done = false + i == 0;
        bucket[i].select = false;
        bucket[i].maxPiv = 0x0;
        bucket[i].minPiv = ( Type ) 0xffffffffffffffff;
        //bucket[i].pivot  = 0+ (i==0)*((min(min(data[0],data[size/2]),data[size-1]) + max(max(data[0],data[size/2]),data[size-1]))/2);
        bucket[i].pivot = data[size / 2];
    }
}

template <typename Type>
void sort(Type *ddata, Type *outputData, uint size, uint threadCount, int device, double *wallClock)
{

    cudaSetDevice(device);

    cudaGetLastError();
    //cudaDeviceReset();

    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, device);

    StopWatchInterface *htimer = NULL;
    Type *dbuffer;

    Block<Type> *dbucket;
    struct Partition<Type> *partition;
    uint *npartitions1, *npartitions2;

    uint *cudaBlocks = (uint *)malloc(4);

    uint blocks = (size + SHARED_LIMIT - 1) / SHARED_LIMIT;
    uint nblock = 10 * blocks;
    int partition_max = 262144;

    //unsigned long long int total = partition_max * sizeof(Block<Type>) + nblock * sizeof(Partition<Type>) + 2 * partition_max * sizeof(uint) + 3 * (size) * sizeof(Type);

    //Allocating and initializing CUDA arrays
    sdkCreateTimer(&htimer);
    checkCudaErrors(cudaMalloc((void **)&dbucket, partition_max * sizeof(Block<Type>)));
    checkCudaErrors(cudaMalloc((void **)&partition, nblock * sizeof(Partition<Type>))); //nblock

    checkCudaErrors(cudaMalloc((void **)&npartitions1, partition_max * sizeof(uint)));
    checkCudaErrors(cudaMalloc((void **)&npartitions2, partition_max * sizeof(uint)));

    checkCudaErrors(cudaMalloc((void **)&dbuffer, (size) * sizeof(Type)));

    initScan();

    //setting GPU Cache
    cudaFuncSetCacheConfig(init<Type>, cudaFuncCachePreferL1);
    cudaFuncSetCacheConfig(insertPivot<Type>, cudaFuncCachePreferL1);
    cudaFuncSetCacheConfig(bucketAssign<Type>, cudaFuncCachePreferL1);
    cudaFuncSetCacheConfig(partitionAssign<Type>, cudaFuncCachePreferL1);
    cudaFuncSetCacheConfig(quick<Type>, cudaFuncCachePreferShared);
    cudaFuncSetCacheConfig(globalBitonicSort<Type>, cudaFuncCachePreferShared);

    checkCudaErrors(cudaDeviceSynchronize());
    sdkResetTimer(&htimer);
    sdkStartTimer(&htimer);

    //initializing bucket array: initial attributes for each bucket
    init<Type><<<(nblock + 255) / 256, 256>>>(ddata, dbucket, npartitions1, size, partition_max);

    uint nbucket = 1;
    uint numIterations = 0;
    bool inputSelect = true;

    *cudaBlocks = blocks;
    checkCudaErrors(cudaDeviceSynchronize());
    getLastCudaError("init() execution FAILED\n");
    checkCudaErrors(cudaMemcpy(&npartitions2[0], cudaBlocks, sizeof(uint), cudaMemcpyHostToDevice));

    // beginning of the first phase
    // this phase goes on until the size of the buckets is comparable to the SHARED_LIMIT size
    while (1)
    {

        /*
		 *       	---------------------    Pre-processing: Partitioning    ---------------------
		 *
		 * buckets are further divided in partitions based on their size
		 * the number of partitions needed for each subsequence is determined by the number of elements which can be
		 * processed by each thread block.
		 *
		 * the number of partitions (npartitions) for each block will depend on the shared memory size (SHARED_LIMIT)
		 *
		 */

        if (numIterations > 0)
        { //1024 is the shared memory limit of scanInclusiveShort()
            if (nbucket <= 1024)
                scanInclusiveShort(npartitions2, npartitions1, 1, nbucket);
            else
                scanInclusiveLarge(npartitions2, npartitions1, 1, nbucket);

            checkCudaErrors(cudaMemcpy(cudaBlocks, &npartitions2[nbucket - 1], sizeof(uint), cudaMemcpyDeviceToHost));
        }

        if (*cudaBlocks == 0)
            break;

        /*
		 *  ---------------------     step 1    ---------------------
		 *
		 * 	A thread block is assigned to each different partition
		 * 	each partition is assigned coordinates, pivot and ....
		 */

        partitionAssign<Type><<<nbucket, 1024>>>(dbucket, npartitions2, partition);
        cudaDeviceSynchronize();
        getLastCudaError("partitionAssign() execution FAILED\n");

        /*
		 	 ---------------------    step 2a    ---------------------

		 	 in this function each thread block creates two subsequences
		 	 to divide the items in the partition whose value is lower than
			 the pivot value, from the items whose value is higher than the pivot value
		 */

        if (inputSelect)
            quick<Type><<<*cudaBlocks, threadCount>>>(ddata, dbuffer, partition, dbucket);
        else
            quick<Type><<<*cudaBlocks, threadCount>>>(dbuffer, ddata, partition, dbucket);
        cudaDeviceSynchronize();
        getLastCudaError("quick() execution FAILED\n");

        //step 2b: this function enters the pivot value in the central bucket's items
        insertPivot<Type><<<nbucket, 512>>>(ddata, dbucket, nbucket);

        //step 3: parameters are assigned, linked to the two new buckets created in step 2
        bucketAssign<Type><<<(nbucket + 255) / 256, 256>>>(dbucket, npartitions1, nbucket, inputSelect);
        cudaDeviceSynchronize();
        getLastCudaError("insertPivot() or bucketAssign() execution FAILED\n");

        nbucket *= 2;

        inputSelect = !inputSelect;
        numIterations++;
        if (nbucket > deviceProp.maxGridSize[0])
            break;
        //if(numIterations==18) break;
    }

    /*
	 * start second phase:
	 * now the size of the buckets is such that they can be entirely processed by a thread block
	 *
	 */

    if (nbucket > deviceProp.maxGridSize[0])
        fprintf(stderr, "ERROR: CUDA-Quicksort can't terminate sorting as the block threads needed to finish it are more than the Maximum x-dimension of FERMI GPU thread blocks. Please use Kepler GPUs as the Maximum x-dimension of their thread blocks is much higher\n");
    else
        globalBitonicSort<Type><<<nbucket, 512, 0>>>(ddata, dbuffer, dbucket, inputSelect);

    checkCudaErrors(cudaDeviceSynchronize());
    getLastCudaError("globalBitonicSort() execution FAILED\n");

    sdkStopTimer(&htimer);
    *wallClock = sdkGetTimerValue(&htimer);

    // release resources
    checkCudaErrors(cudaFree(dbuffer));
    checkCudaErrors(cudaFree(dbucket));
    checkCudaErrors(cudaFree(npartitions2));
    checkCudaErrors(cudaFree(npartitions1));
    free(cudaBlocks);

    closeScan();
    return;
}

void CUDA_Quicksort(uint* inputData, uint* outputData, uint dataSize, uint threadCount, int Device, double* wallClock)
{

	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, Device);

	if(deviceProp.major<2)
	{
		fprintf(stderr, "Error: the GPU device %d has a Compute Capability of %d.%d, while a Compute Capability of 2.x is required to run the code\n",
				Device, deviceProp.major, deviceProp.minor);

		int deviceCount;
		cudaGetDeviceCount(&deviceCount);

		fprintf(stderr, "       the Host system has the following GPU devices:\n");

		for (int device = 0; device < deviceCount; device++) {

				fprintf(stderr, "\t  the GPU device %d is a %s, with Compute Capability %d.%d\n",
						device, deviceProp.name, deviceProp.major, deviceProp.minor);
		}

		return;
	}

	sort<uint>(inputData,outputData, dataSize,threadCount,Device, wallClock);
}

void CUDA_Quicksort_64(double* inputData,double* outputData, uint dataSize, uint threadCount, int Device, double* wallClock)
{

	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, Device);

	if(deviceProp.major<2)
	{
		fprintf(stderr, "Error: the GPU device %d has a Compute Capability of %d.%d, while a Compute Capability of 2.x is required to run the code\n",
				Device, deviceProp.major, deviceProp.minor);

		int deviceCount;
		cudaGetDeviceCount(&deviceCount);

		fprintf(stderr, "       the Host system has the following GPU devices:\n");

		for (int device = 0; device < deviceCount; device++) {

				fprintf(stderr, "\t  the GPU device %d is a %s, with Compute Capability %d.%d\n",
						device, deviceProp.name, deviceProp.major, deviceProp.minor);
		}

		return;
	}

	sort<double>(inputData,outputData, dataSize,threadCount,Device,wallClock);

}

struct MancaQuicksort
{
   static void sort( TNL::Containers::ArrayView< int, TNL::Devices::Cuda >& array )
   {
      double timer;
      CUDA_Quicksort( ( unsigned * ) array.getData(),  (unsigned * ) array.getData(), array.getSize(), 256, 0, &timer );
      //return;
   }
};
