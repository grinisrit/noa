#include <vector>
#include <algorithm>
#include <numeric>
#include <random>


#include <TNL/Containers/Array.h>
#include <TNL/Algorithms/MemoryOperations.h>
#include <TNL/Algorithms/sort.h>
#include <TNL/Algorithms/Sorting/Quicksort.h>

#if defined HAVE_CUDA && defined HAVE_GTEST
#include <thrust/sort.h>
#include <thrust/execution_policy.h>

#include <gtest/gtest.h>

using namespace TNL;
using namespace TNL::Algorithms;
using namespace TNL::Algorithms::Sorting;

TEST(selectedSize, size15)
{
   TNL::Containers::Array<int, TNL::Devices::Cuda> cudaArr{5, 9, 4, 8, 6, 1, 2, 3, 4, 8, 1, 6, 9, 4, 9};
   auto view = cudaArr.getView();
   EXPECT_EQ(15, view.getSize()) << "size not 15" << std::endl;
   Quicksort::sort( view );
   EXPECT_TRUE(Algorithms::isAscending(view)) << "result " << view << std::endl;
}

TEST(multiblock, 32768_decreasingNegative)
{
   std::vector<int> arr(1<<15);
   for (size_t i = 0; i < arr.size(); i++)
      arr[i] = -i;

   TNL::Containers::Array<int, TNL::Devices::Cuda> cudaArr(arr);
   auto view = cudaArr.getView();
   Quicksort::sort( view );

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
      Quicksort::sort( view );

      EXPECT_TRUE(Algorithms::isAscending(view));
    }
}

TEST(randomGenerated, bigArray_randomVal)
{
   std::srand(304);
   for(int i = 0; i < 50; i++)
   {
      int size = (1<<20) + (std::rand()% (1<<19));
      std::vector<int> arr(size);
      for(auto & x : arr) x = std::rand();
      TNL::Containers::Array<int, TNL::Devices::Cuda> cudaArr(arr);

      auto view = cudaArr.getView();
      Quicksort::sort( view );
      EXPECT_TRUE(Algorithms::isAscending(view));
    }
}

TEST(noLostElement, smallArray)
{
   std::srand(9151);

   int size = (1<<7);
   std::vector<int> arr(size);
   for(auto & x : arr) x = std::rand();

   TNL::Containers::Array<int, TNL::Devices::Cuda> cudaArr(arr);
   auto view = cudaArr.getView();
   Quicksort::sort( view );

   std::sort(arr.begin(), arr.end());
   TNL::Containers::Array<int, TNL::Devices::Cuda> cudaArr2(arr);
   EXPECT_TRUE(view == cudaArr2.getView());
}

TEST(noLostElement, midSizedArray)
{
   std::srand(91503);

   int size = (1<<15);
   std::vector<int> arr(size);
   for(auto & x : arr) x = std::rand();

   TNL::Containers::Array<int, TNL::Devices::Cuda> cudaArr(arr);
   auto view = cudaArr.getView();
   Quicksort::sort( view );

   std::sort(arr.begin(), arr.end());
   TNL::Containers::Array<int, TNL::Devices::Cuda> cudaArr2(arr);
   EXPECT_TRUE(view == cudaArr2.getView());
}

TEST(noLostElement, bigSizedArray)
{
   std::srand(15611);

   int size = (1<<22);
   std::vector<int> arr(size);
   for(auto & x : arr) x = std::rand();
   for(int i = 0; i < 10000; i++)
      arr[std::rand() % arr.size()] = (1<<10);

   TNL::Containers::Array<int, TNL::Devices::Cuda> cudaArr(arr);
   auto view = cudaArr.getView();
   Quicksort::sort( view );

   TNL::Containers::Array<int, TNL::Devices::Cuda> cudaArr2(arr);
   thrust::sort(thrust::device, cudaArr2.getData(), cudaArr2.getData() + cudaArr2.getSize());
   EXPECT_TRUE(view == cudaArr2.getView());
}

TEST(types, type_double)
{
   std::srand(8451);

   int size = (1<<16);
   std::vector<double> arr(size);
   for(auto & x : arr) x = std::rand();
   for(int i = 0; i < 10000; i++)
      arr[std::rand() % arr.size()] = (1<<10);

   TNL::Containers::Array<double, TNL::Devices::Cuda> cudaArr(arr);
   auto view = cudaArr.getView();
   Quicksort::sort( view );

   TNL::Containers::Array<double, TNL::Devices::Cuda> cudaArr2(arr);
   thrust::sort(thrust::device, cudaArr2.getData(), cudaArr2.getData() + cudaArr2.getSize());
   EXPECT_TRUE(view == cudaArr2.getView());
}

struct TMPSTRUCT_xyz{
   double x, y, z;
   __cuda_callable__ TMPSTRUCT_xyz(): x(0){}
   __cuda_callable__ TMPSTRUCT_xyz(int first){x = first;};
   __cuda_callable__ bool operator <(const TMPSTRUCT_xyz& other) const { return x< other.x;}
   __cuda_callable__ TMPSTRUCT_xyz& operator =(const TMPSTRUCT_xyz& other) {x = other.x; return *this;}
};
std::ostream & operator<<(std::ostream & out, const TMPSTRUCT_xyz & data){return out << data.x;}

TEST(types, struct_3D_points)
{
   std::srand(46151);

   int size = (1<<18);
   std::vector<TMPSTRUCT_xyz> arr(size);
   for(auto & x : arr) x = TMPSTRUCT_xyz(std::rand());

   TNL::Containers::Array<TMPSTRUCT_xyz, TNL::Devices::Cuda> cudaArr(arr);
   auto view = cudaArr.getView();
   //thrust::sort(thrust::device, cudaArr.getData(), cudaArr.getData() + cudaArr.getSize());
   //std::cout << view << std::endl;
   Quicksort::sort( view );

   EXPECT_TRUE(Algorithms::isAscending(view));
}

struct TMPSTRUCT_64b{
   uint8_t m_Data[64];
   __cuda_callable__ TMPSTRUCT_64b() {m_Data[0] = 0;}
   __cuda_callable__ TMPSTRUCT_64b(int first){m_Data[0] = first;};
   __cuda_callable__ bool operator <(const TMPSTRUCT_64b& other) const { return m_Data[0]< other.m_Data[0];}
   __cuda_callable__ TMPSTRUCT_64b& operator =(const TMPSTRUCT_64b& other) {m_Data[0] = other.m_Data[0]; return *this;}
};
std::ostream & operator<<(std::ostream & out, const TMPSTRUCT_64b & data){return out << (unsigned) data.m_Data[0];}

TEST(types, struct_64b)
{
   std::srand(96);

   int size = (1<<18);
   std::vector<TMPSTRUCT_64b> arr(size);
   for(auto & x : arr) x = TMPSTRUCT_64b(std::rand() % 512);

   TNL::Containers::Array<TMPSTRUCT_64b, TNL::Devices::Cuda> cudaArr(arr);
   auto view = cudaArr.getView();
   //thrust::sort(thrust::device, cudaArr.getData(), cudaArr.getData() + cudaArr.getSize());
   //std::cout << view << std::endl;
   Quicksort::sort( view );

   EXPECT_TRUE(Algorithms::isAscending(view));
}

#endif

#include "../../main.h"
