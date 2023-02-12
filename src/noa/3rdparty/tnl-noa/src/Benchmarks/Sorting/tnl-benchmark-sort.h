#include <iostream>
#include <fstream>
#include <iomanip>
#include <vector>
#include <TNL/Algorithms/sort.h>
using namespace std;

#include "generators.h"
#include "Measurer.h"

#ifndef LOW_POW
    #define LOW_POW 10
#endif

#ifndef HIGH_POW
    #define HIGH_POW 25
#endif

#ifndef TRIES
    #define TRIES 20
#endif

using namespace TNL;
using namespace TNL::Algorithms;
using namespace TNL::Algorithms::Sorting;

template< typename Sorter >
void start(ostream & out, string delim)
{
    out << "size" << delim;
    out << "random" << delim;
    out << "shuffle" << delim;
    out << "sorted" << delim;
    out << "almost" << delim;
    out << "decreas" << delim;
    out << "gauss" << delim;
    out << "bucket" << delim;
    out << "stagger" << delim;
    out << "zero_entropy";
    out << endl;

    int wrongAnsCnt = 0;

    for(int pow = LOW_POW; pow <= HIGH_POW; pow++)
    {
        int size =(1<< pow);
        vector<int> vec(size);

        out << "2^" << pow << delim << flush;
        out << fixed << setprecision(3);

        out << Measurer< Sorter >::measure( generateRandom(size), TRIES, wrongAnsCnt);
        out << delim << flush;

        out << Measurer< Sorter >::measure( generateShuffle(size), TRIES, wrongAnsCnt);
        out << delim << flush;

        out << Measurer< Sorter >::measure( generateSorted(size), TRIES, wrongAnsCnt);
        out << delim << flush;

        out << Measurer< Sorter >::measure( generateAlmostSorted(size), TRIES, wrongAnsCnt);
        out << delim << flush;

        out << Measurer< Sorter >::measure( generateDecreasing(size), TRIES, wrongAnsCnt);
        out << delim << flush;

        out << Measurer< Sorter >::measure( generateGaussian(size), TRIES, wrongAnsCnt) ;
        out << delim << flush;

        out << Measurer< Sorter >::measure( generateBucket(size), TRIES, wrongAnsCnt);
        out << delim << flush;

        out << Measurer< Sorter >::measure( generateStaggered(size), TRIES, wrongAnsCnt);
        out << delim << flush;

        out << Measurer< Sorter >::measure( generateZero_entropy(size), TRIES, wrongAnsCnt);
        out << endl;
    }

    if(wrongAnsCnt > 0)
        std::cerr << wrongAnsCnt << "tries were sorted incorrectly" << std::endl;
}

int main(int argc, char *argv[])
{
   if(argc == 1)
   {
#ifdef __CUDACC__
      std::cout << "Quicksort on GPU ... " << std::endl;
      start< Quicksort >(cout, "\t");
      std::cout << "Bitonic sort on GPU ... " << std::endl;
      start< BitonicSort >( cout, "\t" );

// FIXME: clang 14 fails to compile the reference algorithms (e.g. due to compile errors in thrust or cub)
#if defined(__CUDA__) && !defined(__clang__)
#ifdef HAVE_CUDA_SAMPLES
      std::cout << "Manca quicksort on GPU ... " << std::endl;
      start< MancaQuicksort >( cout, "\t" );
      std::cout << "Nvidia bitonic sort on GPU ... " << std::endl;
      start< NvidiaBitonicSort >( cout, "\t" );
#endif
      std::cout << "Cederman quicksort on GPU ... " << std::endl;
      start< CedermanQuicksort >( cout, "\t" );
      std::cout << "Thrust radixsort on GPU ... " << std::endl;
      start< ThrustRadixsort >( cout, "\t" );
#endif
#endif

      std::cout << "STL sort on CPU ... " << std::endl;
      start< STLSort >( cout, "\t" );
   }
   else
   {
      std::ofstream out(argv[1]);
#ifdef __CUDACC__
      std::cout << "Quicksort on GPU ... " << std::endl;
      start< Quicksort >(out, ",");
      std::cout << "Bitonic sort on GPU ... " << std::endl;
      start< BitonicSort >(out, ",");

// FIXME: clang 14 fails to compile the reference algorithms (e.g. due to compile errors in thrust or cub)
#if defined(__CUDA__) && !defined(__clang__)
#ifdef HAVE_CUDA_SAMPLES
      std::cout << "Manca quicksort on GPU ... " << std::endl;
      start< MancaQuicksort >( out, "," );
      std::cout << "Nvidia bitonic sort on GPU ... " << std::endl;
      start< NvidiaBitonicSort >( out, "," );
#endif
      std::cout << "Cederman quicksort on GPU ... " << std::endl;
      start< CedermanQuicksort >( out, "," );
      std::cout << "Thrust radixsort on GPU ... " << std::endl;
      start< ThrustRadixsort >( out, "," );
#endif
#endif

      std::cout << "STL sort on CPU ... " << std::endl;
      start< STLSort >( out, "," );
    }
    return 0;
}
