#include <vector>
#include <algorithm>
#include <numeric>
#include <random>
#include <iostream>

#include <TNL/Containers/Array.h>
#include <TNL/Algorithms/MemoryOperations.h>
#include <TNL/Algorithms/Sorting/BubbleSort.h>
#include <TNL/Algorithms/sort.h>

#if defined HAVE_GTEST
#include <gtest/gtest.h>


using namespace TNL;
using namespace TNL::Algorithms;
using namespace TNL::Algorithms::Sorting;

template<typename TYPE>
void fetchAndSwapSorter( TNL::Containers::ArrayView< TYPE, TNL::Devices::Host > view)
{
    auto Cmp = [=]__cuda_callable__(const int i, const int j ){return view[ i ]  < view[ j ];};
    auto Swap = [=] __cuda_callable__ (int i, int j) mutable {TNL::swap(view[i], view[j]);};
    BubbleSort::inplaceSort< TNL::Devices::Host >(0, view.getSize(), Cmp, Swap);
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

        TNL::Containers::Array<int, TNL::Devices::Host> arr(orig);
        auto view = arr.getView();
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
        TNL::Containers::Array<double, TNL::Devices::Host> arr(orig);
        auto view = arr.getView();
        fetchAndSwapSorter(view);
        EXPECT_TRUE(Algorithms::isAscending(view)) << "result " << view << std::endl;
    }
    while (std::next_permutation(orig.begin(), orig.end()));
}

void fetchAndSwap_sortMiddle(TNL::Containers::ArrayView<int, TNL::Devices::Host> view, int from, int to)
{
    //auto Fetch = [=]__cuda_callable__(int i){return view[i];};
    auto Cmp = [=]__cuda_callable__(const int i, const int j ){ return view[ i ] < view[ j ]; };
    auto Swap = [=] __cuda_callable__ (int i, int j) mutable { TNL::swap(view[i], view[j]); };
    BubbleSort::inplaceSort< TNL::Devices::Host >(from, to, Cmp, Swap);
}

TEST(fetchAndSwap, sortMiddle)
{
    std::vector<int> orig{5, 9, 4, 54, 21, 6, 7, 9, 0, 9, 42, 4};
    TNL::Containers::Array<int, TNL::Devices::Host> arr(orig);
    auto view = arr.getView();
    size_t from = 3, to = 8;

    fetchAndSwap_sortMiddle(view, from, to);
    EXPECT_TRUE(Algorithms::isAscending(view.getView(3, 8))) << "result " << view << std::endl;

    for(size_t i = 0; i < orig.size(); i++)
    {
        if( i < from || i >= to )
        {
            EXPECT_TRUE(view.getElement(i) == orig[i]);
        }
    }
}

#endif

#include "../../main.h"
