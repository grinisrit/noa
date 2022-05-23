#include <iostream>
#include <TNL/Matrices/Sandbox/SparseSandboxMatrix.h>

#ifdef HAVE_GTEST
#include <gtest/gtest.h>

const char* saveAndLoadFileName = "test_SandboxMatrixTest_SandboxMatrix";

// types for which MatrixTest is instantiated
using MatrixTypes = ::testing::Types
<
    TNL::Matrices::Sandbox::SparseSandboxMatrix< int,     TNL::Devices::Host, int,   TNL::Matrices::GeneralMatrix >,
    TNL::Matrices::Sandbox::SparseSandboxMatrix< long,    TNL::Devices::Host, int,   TNL::Matrices::GeneralMatrix >,
    TNL::Matrices::Sandbox::SparseSandboxMatrix< float,   TNL::Devices::Host, int,   TNL::Matrices::GeneralMatrix >,
    TNL::Matrices::Sandbox::SparseSandboxMatrix< double,  TNL::Devices::Host, int,   TNL::Matrices::GeneralMatrix >,
    TNL::Matrices::Sandbox::SparseSandboxMatrix< int,     TNL::Devices::Host, long,  TNL::Matrices::GeneralMatrix >,
    TNL::Matrices::Sandbox::SparseSandboxMatrix< long,    TNL::Devices::Host, long,  TNL::Matrices::GeneralMatrix >,
    TNL::Matrices::Sandbox::SparseSandboxMatrix< float,   TNL::Devices::Host, long,  TNL::Matrices::GeneralMatrix >,
    TNL::Matrices::Sandbox::SparseSandboxMatrix< double,  TNL::Devices::Host, long,  TNL::Matrices::GeneralMatrix >
#ifdef HAVE_CUDA
   ,TNL::Matrices::Sandbox::SparseSandboxMatrix< int,     TNL::Devices::Cuda, int,   TNL::Matrices::GeneralMatrix >,
    TNL::Matrices::Sandbox::SparseSandboxMatrix< long,    TNL::Devices::Cuda, int,   TNL::Matrices::GeneralMatrix >,
    TNL::Matrices::Sandbox::SparseSandboxMatrix< float,   TNL::Devices::Cuda, int,   TNL::Matrices::GeneralMatrix >,
    TNL::Matrices::Sandbox::SparseSandboxMatrix< double,  TNL::Devices::Cuda, int,   TNL::Matrices::GeneralMatrix >,
    TNL::Matrices::Sandbox::SparseSandboxMatrix< int,     TNL::Devices::Cuda, long,  TNL::Matrices::GeneralMatrix >,
    TNL::Matrices::Sandbox::SparseSandboxMatrix< long,    TNL::Devices::Cuda, long,  TNL::Matrices::GeneralMatrix >,
    TNL::Matrices::Sandbox::SparseSandboxMatrix< float,   TNL::Devices::Cuda, long,  TNL::Matrices::GeneralMatrix >,
    TNL::Matrices::Sandbox::SparseSandboxMatrix< double,  TNL::Devices::Cuda, long,  TNL::Matrices::GeneralMatrix >
#endif
>;

#endif

#include "SparseMatrixTest.h"
#include "../main.h"
