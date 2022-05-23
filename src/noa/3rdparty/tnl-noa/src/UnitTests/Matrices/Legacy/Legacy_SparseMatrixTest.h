#include <Benchmarks/SpMV/ReferenceFormats/Legacy/CSR.h>

#include "Legacy_SparseMatrixTest.hpp"
#include <iostream>

#ifdef HAVE_GTEST 
#include <gtest/gtest.h>

using CSR_host_float = TNL::Benchmarks::SpMV::ReferenceFormats::Legacy::CSR< float, TNL::Devices::Host, int >;
using CSR_host_int = TNL::Benchmarks::SpMV::ReferenceFormats::Legacy::CSR< int, TNL::Devices::Host, int >;

using CSR_cuda_float = TNL::Benchmarks::SpMV::ReferenceFormats::Legacy::CSR< float, TNL::Devices::Cuda, int >;
using CSR_cuda_int = TNL::Benchmarks::SpMV::ReferenceFormats::Legacy::CSR< int, TNL::Devices::Cuda, int >;

TEST( SparseMatrixTest, CSR_perforSORIterationTest_Host )
{
    test_PerformSORIteration< CSR_host_float >();
}

#ifdef HAVE_CUDA
TEST( SparseMatrixTest, CSR_perforSORIterationTest_Cuda )
{
   //    test_PerformSORIteration< CSR_cuda_float >();
}
#endif

#endif

#include "../../main.h"
