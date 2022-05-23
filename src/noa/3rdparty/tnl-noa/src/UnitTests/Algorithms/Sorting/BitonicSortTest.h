#include <vector>
#include <algorithm>
#include <numeric>
#include <random>
#include <iostream>

#include <TNL/Containers/Array.h>
#include <TNL/Algorithms/MemoryOperations.h>
#include <TNL/Algorithms/Sorting/BitonicSort.h>
#include <TNL/Algorithms/sort.h>

#if defined HAVE_GTEST && defined HAVE_CUDA
#include <gtest/gtest.h>


using namespace TNL;
using namespace TNL::Algorithms;
using namespace TNL::Algorithms::Sorting;

TEST(permutations, allPermutationSize_2_to_7)
{
    for(int i = 2; i<=7; i++ )
    {
        int size = i;
        std::vector<int> orig(size);
        std::iota(orig.begin(), orig.end(), 0);

        do
        {
            TNL::Containers::Array<int, TNL::Devices::Cuda> cudaArr(orig);
            auto view = cudaArr.getView();

            BitonicSort::sort(view);

            EXPECT_TRUE( Algorithms::isAscending( view ) ) << "failed " << i << std::endl;
        }
        while (std::next_permutation(orig.begin(), orig.end()));
    }
}

TEST(permutations, allPermutationSize_8)
{
    int size = 9;
    const int stride = 151;
    int i = 0;

    std::vector<int> orig(size);
    std::iota(orig.begin(), orig.end(), 0);

    do
    {
        if ((i++) % stride != 0)
            continue;

        TNL::Containers::Array<int, TNL::Devices::Cuda> cudaArr(orig);
        auto view = cudaArr.getView();

        BitonicSort::sort(view);

        EXPECT_TRUE(Algorithms::isAscending(view)) << "result " << view << std::endl;
    }
    while (std::next_permutation(orig.begin(), orig.end()));
}

TEST(permutations, somePermutationSize9)
{
    int size = 9;
    const int stride = 227;
    int i = 0;

    std::vector<int> orig(size);
    std::iota(orig.begin(), orig.end(), 0);

    do
    {
        if ((i++) % stride != 0)
            continue;

        TNL::Containers::Array<int, TNL::Devices::Cuda> cudaArr(orig);
        auto view = cudaArr.getView();

        BitonicSort::sort(view);

        EXPECT_TRUE(Algorithms::isAscending(view)) << "result " << view << std::endl;
    }
    while (std::next_permutation(orig.begin(), orig.end()));
}

TEST(selectedSize, size15)
{
    TNL::Containers::Array<int, TNL::Devices::Cuda> cudaArr{5, 9, 4, 8, 6, 1, 2, 3, 4, 8, 1, 6, 9, 4, 9};
    auto view = cudaArr.getView();
    EXPECT_EQ(15, view.getSize()) << "size not 15" << std::endl;
    BitonicSort::sort(view);
    EXPECT_TRUE(Algorithms::isAscending(view)) << "result " << view << std::endl;
}

TEST(multiblock, 32768_decreasingNegative)
{
    std::vector<int> arr(1<<15);
    for (size_t i = 0; i < arr.size(); i++)
        arr[i] = -i;

    TNL::Containers::Array<int, TNL::Devices::Cuda> cudaArr(arr);
    auto view = cudaArr.getView();

    BitonicSort::sort(view);
    EXPECT_TRUE(Algorithms::isAscending(view)) << "result " << view << std::endl;
}

TEST(randomGenerated, smallArray_randomVal)
{
    std::srand(2006);
    for(int i = 0; i < 100; i++)
    {
        std::vector<int> arr(std::rand()%(1<<10));
        for(auto & x : arr)
            x = std::rand();

        TNL::Containers::Array<int, TNL::Devices::Cuda> cudaArr(arr);

        auto view = cudaArr.getView();
        BitonicSort::sort(view);
        EXPECT_TRUE(Algorithms::isAscending(view));
    }
}

TEST(randomGenerated, bigArray_all0)
{
    std::srand(304);
    for(int i = 0; i < 50; i++)
    {
        int size = (1<<20) + (std::rand()% (1<<19));

        TNL::Containers::Array<int, TNL::Devices::Cuda> cudaArr(size);

        auto view = cudaArr.getView();
        BitonicSort::sort(view);
        EXPECT_TRUE(Algorithms::isAscending(view));
    }
}

TEST(nonIntegerType, float_notPow2)
{
    TNL::Containers::Array<float, TNL::Devices::Cuda> cudaArr{5.0, 9.4, 4.6, 8.9, 6.2, 1.15184, 2.23};
    auto view = cudaArr.getView();
    BitonicSort::sort(view);
    EXPECT_TRUE(Algorithms::isAscending(view)) << "result " << view << std::endl;
}

TEST(nonIntegerType, double_notPow2)
{
    TNL::Containers::Array<double, TNL::Devices::Cuda> cudaArr{5.0, 9.4, 4.6, 8.9, 6.2, 1.15184, 2.23};
    auto view = cudaArr.getView();
    BitonicSort::sort(view);
    EXPECT_TRUE(Algorithms::isAscending(view)) << "result " << view << std::endl;
}


struct TMPSTRUCT{
    uint8_t m_data[6];
    __cuda_callable__ TMPSTRUCT(){m_data[0] = 0;}
    __cuda_callable__ TMPSTRUCT(int first){m_data[0] = first;};
    __cuda_callable__ bool operator <(const TMPSTRUCT& other) const { return m_data[0] < other.m_data[0];}
    __cuda_callable__ TMPSTRUCT& operator =(const TMPSTRUCT& other) {m_data[0] = other.m_data[0]; return *this;}

};

TEST(nonIntegerType, struct)
{
    TNL::Containers::Array<TMPSTRUCT, TNL::Devices::Cuda> cudaArr{TMPSTRUCT(5), TMPSTRUCT(6), TMPSTRUCT(9), TMPSTRUCT(1)};
    auto view = cudaArr.getView();
    BitonicSort::sort(view);
    EXPECT_TRUE(Algorithms::isAscending(view));
}

struct TMPSTRUCT_64b{
    uint8_t m_data[64];
    __cuda_callable__ TMPSTRUCT_64b(){m_data[0] = 0;}
    __cuda_callable__ TMPSTRUCT_64b(int first){m_data[0] = first;};
    __cuda_callable__ bool operator <(const TMPSTRUCT_64b& other) const { return m_data[0] < other.m_data[0];}
    __cuda_callable__ TMPSTRUCT_64b& operator =(const TMPSTRUCT_64b& other) {m_data[0] = other.m_data[0]; return *this;}
};

TEST(nonIntegerType, struct_64b)
{
    std::srand(61513);
    int size = std::rand() % (1<<15);
    std::vector<TMPSTRUCT_64b> vec(size);
    for(auto & x : vec)
        x = TMPSTRUCT_64b(std::rand());

    TNL::Containers::Array<TMPSTRUCT_64b, TNL::Devices::Cuda> cudaArr(vec);
    auto view = cudaArr.getView();
    BitonicSort::sort(view);
    EXPECT_TRUE(Algorithms::isAscending(view));
}

struct TMPSTRUCT_128b{
    uint8_t m_data[128];
    __cuda_callable__ TMPSTRUCT_128b(){m_data[0] = 0;}
    __cuda_callable__ TMPSTRUCT_128b(int first){m_data[0] = first;};
    __cuda_callable__ bool operator <(const TMPSTRUCT_128b& other) const { return m_data[0] < other.m_data[0];}
    __cuda_callable__ TMPSTRUCT_128b& operator =(const TMPSTRUCT_128b& other) {m_data[0] = other.m_data[0]; return *this;}
};

TEST(nonIntegerType, struct_128b)
{
    std::srand(98451);
    int size = std::rand() % (1<<14);
    std::vector<TMPSTRUCT_128b> vec(size);
    for(auto & x : vec)
        x = TMPSTRUCT_128b(std::rand());

    TNL::Containers::Array<TMPSTRUCT_128b, TNL::Devices::Cuda> cudaArr(vec);
    auto view = cudaArr.getView();
    BitonicSort::sort(view);
    EXPECT_TRUE(Algorithms::isAscending(view));
}

//error bypassing
//https://mmg-gitlab.fjfi.cvut.cz/gitlab/tnl/tnl-dev/blob/fbc34f6a97c13ec865ef7969b9704533222ed408/src/UnitTests/Containers/VectorTest-8.h
void descendingSort(TNL::Containers::ArrayView<int, TNL::Devices::Cuda> view)
{
    auto cmpDescending = [] __cuda_callable__ (int a, int b) {return a > b;};
    bitonicSort(view, cmpDescending);
}

TEST(sortWithFunction, descending)
{
    TNL::Containers::Array<int, TNL::Devices::Cuda> cudaArr{6, 9, 4, 2, 3};
    auto view = cudaArr.getView();
    descendingSort(view);

    EXPECT_FALSE(Algorithms::isAscending(view)) << "result " << view << std::endl;

    EXPECT_TRUE(view.getElement(0) == 9);
    EXPECT_TRUE(view.getElement(1) == 6);
    EXPECT_TRUE(view.getElement(2) == 4);
    EXPECT_TRUE(view.getElement(3) == 3);
    EXPECT_TRUE(view.getElement(4) == 2);
}

/*TEST(sortHostArray, hostArray)
{
    TNL::Containers::Array< int > arr( 84561 );
    for( size_t i = 0; i < arr.getSize(); i++ )
        arr[i] = -i;

    bitonicSort(arr);

    EXPECT_TRUE( TNL::Algorithms::isAscending(arr) );
}*/

/*TEST(sortRange, secondHalf)
{
    std::vector<int> arr(19);
    int s = 19/2;
    for(size_t i = 0; i < s; i++) arr[i] = -1;
    for(size_t i = s; i < 19; i++) arr[i] = -i;

    bitonicSort(arr, s, 19);

    EXPECT_TRUE(TNL::Algorithms::isAscending(arr.begin() + s, arr.end()));
    EXPECT_TRUE(arr[0] == -1);
    EXPECT_TRUE(arr[s-1] == -1);
}

TEST(sortRange, middle)
{
    std::srand(8705);

    std::vector<int> arr(20);
    int s = 5, e = 15;
    for(size_t i = 0; i < s; i++) arr[i] = -1;
    for(size_t i = e; i < 20; i++) arr[i] = -1;

    for(size_t i = s; i < e; i++) arr[i] = std::rand();

    bitonicSort(arr, s, e);

    EXPECT_TRUE(TNL::Algorithms::isAscending(arr.begin() + s, arr.begin() + e));
    EXPECT_TRUE(arr[0] == -1);
    EXPECT_TRUE(arr.back() == -1);
    EXPECT_TRUE(arr[s-1] == -1);
    EXPECT_TRUE(arr[e] == -1);
}

TEST(sortRange, middleMultiBlock)
{
    std::srand(4513);
    int size = 1<<20;
    int s = 2000, e = size - 1512;

    std::vector<int> arr(size);
    for(size_t i = 0; i < s; i++) arr[i] = -1;
    for(size_t i = e; i < size; i++) arr[i] = -1;

    for(size_t i = s; i < e; i++) arr[i] = std::rand();

    bitonicSort(arr, s, e);

    EXPECT_TRUE(TNL::Algorithms::isAscending(arr.begin() + s, arr.begin() + e));

    EXPECT_TRUE(arr[0] == -1);
    EXPECT_TRUE(arr[std::rand() % s] == -1);
    EXPECT_TRUE(arr[s-1] == -1);

    EXPECT_TRUE(arr[e] == -1);
    EXPECT_TRUE(arr[e + (std::rand() % (size - e))] == -1);
    EXPECT_TRUE(arr.back() == -1);
}*/

template<typename TYPE>
void fetchAndSwapSorter(TNL::Containers::ArrayView<TYPE, TNL::Devices::Cuda> view)
{
    //auto Fetch = [=]__cuda_callable__(int i){return view[i];};
    auto Cmp = [=]__cuda_callable__(const int i, const int j ){return view[ i ]  < view[ j ];};
    auto Swap = [=] __cuda_callable__ (int i, int j) mutable {TNL::swap(view[i], view[j]);};
    bitonicSort(0, view.getSize(), Cmp, Swap);
}

TEST(fetchAndSwap, oneBlockSort)
{
    int size = 9;
    const int stride = 227;
    int i = 0;

    std::vector<int> orig(size);
    std::iota(orig.begin(), orig.end(), 0);

    do
    {
        if ((i++) % stride != 0)
            continue;

        TNL::Containers::Array<int, TNL::Devices::Cuda> cudaArr(orig);
        auto view = cudaArr.getView();
        fetchAndSwapSorter(view);
        EXPECT_TRUE(Algorithms::isAscending(view)) << "result " << view << std::endl;
    }
    while (std::next_permutation(orig.begin(), orig.end()));
}

TEST(fetchAndSwap, typeDouble)
{
    int size = 5;
    std::vector<double> orig(size);
    std::iota(orig.begin(), orig.end(), 0);

    do
    {
        TNL::Containers::Array<double, TNL::Devices::Cuda> cudaArr(orig);
        auto view = cudaArr.getView();
        fetchAndSwapSorter(view);
        EXPECT_TRUE(Algorithms::isAscending(view)) << "result " << view << std::endl;
    }
    while (std::next_permutation(orig.begin(), orig.end()));
}

void fetchAndSwap_sortMiddle(TNL::Containers::ArrayView<int, TNL::Devices::Cuda> view, int from, int to)
{
    //auto Fetch = [=]__cuda_callable__(int i){return view[i];};
    auto Cmp = [=]__cuda_callable__(const int i, const int j ){ return view[ i ] < view[ j ]; };
    auto Swap = [=] __cuda_callable__ (int i, int j) mutable { TNL::swap(view[i], view[j]); };
    bitonicSort(from, to, Cmp, Swap);
}

TEST(fetchAndSwap, sortMiddle)
{
    std::vector<int> orig{5, 9, 4, 54, 21, 6, 7, 9, 0, 9, 42, 4};
    TNL::Containers::Array<int, TNL::Devices::Cuda> cudaArr(orig);
    auto view = cudaArr.getView();
    size_t from = 3, to = 8;

    fetchAndSwap_sortMiddle(view, from, to);
    EXPECT_TRUE(Algorithms::isAscending(view.getView(3, 8))) << "result " << view << std::endl;

    for(size_t i = 0; i < orig.size(); i++)
    {
        if(i < from || i >= to)
            EXPECT_TRUE(view.getElement(i) == orig[i]);
    }
}

#endif

#include "../../main.h"
