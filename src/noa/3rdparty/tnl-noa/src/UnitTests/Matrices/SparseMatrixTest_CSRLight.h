#include <iostream>
#include <TNL/Algorithms/Segments/CSR.h>
#include <TNL/Matrices/SparseMatrix.h>

#ifdef HAVE_GTEST
#include <gtest/gtest.h>

const char* saveAndLoadFileName = "test_SparseMatrixTest_CSRLight_segments";

// types for which MatrixTest is instantiated
using MatrixTypes = ::testing::Types
<
    TNL::Matrices::SparseMatrix< int,     TNL::Devices::Host, int,   TNL::Matrices::GeneralMatrix, TNL::Algorithms::Segments::CSRLight >,
    TNL::Matrices::SparseMatrix< long,    TNL::Devices::Host, int,   TNL::Matrices::GeneralMatrix, TNL::Algorithms::Segments::CSRLight >,
    TNL::Matrices::SparseMatrix< float,   TNL::Devices::Host, int,   TNL::Matrices::GeneralMatrix, TNL::Algorithms::Segments::CSRLight >,
    TNL::Matrices::SparseMatrix< double,  TNL::Devices::Host, int,   TNL::Matrices::GeneralMatrix, TNL::Algorithms::Segments::CSRLight >,
    TNL::Matrices::SparseMatrix< int,     TNL::Devices::Host, long,  TNL::Matrices::GeneralMatrix, TNL::Algorithms::Segments::CSRLight >,
    TNL::Matrices::SparseMatrix< long,    TNL::Devices::Host, long,  TNL::Matrices::GeneralMatrix, TNL::Algorithms::Segments::CSRLight >,
    TNL::Matrices::SparseMatrix< float,   TNL::Devices::Host, long,  TNL::Matrices::GeneralMatrix, TNL::Algorithms::Segments::CSRLight >,
    TNL::Matrices::SparseMatrix< double,  TNL::Devices::Host, long,  TNL::Matrices::GeneralMatrix, TNL::Algorithms::Segments::CSRLight >
#ifdef __CUDACC__
   ,TNL::Matrices::SparseMatrix< int,     TNL::Devices::Cuda, int,   TNL::Matrices::GeneralMatrix, TNL::Algorithms::Segments::CSRLight >,
    TNL::Matrices::SparseMatrix< long,    TNL::Devices::Cuda, int,   TNL::Matrices::GeneralMatrix, TNL::Algorithms::Segments::CSRLight >,
    TNL::Matrices::SparseMatrix< float,   TNL::Devices::Cuda, int,   TNL::Matrices::GeneralMatrix, TNL::Algorithms::Segments::CSRLight >,
    TNL::Matrices::SparseMatrix< double,  TNL::Devices::Cuda, int,   TNL::Matrices::GeneralMatrix, TNL::Algorithms::Segments::CSRLight >,
    TNL::Matrices::SparseMatrix< int,     TNL::Devices::Cuda, long,  TNL::Matrices::GeneralMatrix, TNL::Algorithms::Segments::CSRLight >,
    TNL::Matrices::SparseMatrix< long,    TNL::Devices::Cuda, long,  TNL::Matrices::GeneralMatrix, TNL::Algorithms::Segments::CSRLight >,
    TNL::Matrices::SparseMatrix< float,   TNL::Devices::Cuda, long,  TNL::Matrices::GeneralMatrix, TNL::Algorithms::Segments::CSRLight >,
    TNL::Matrices::SparseMatrix< double,  TNL::Devices::Cuda, long,  TNL::Matrices::GeneralMatrix, TNL::Algorithms::Segments::CSRLight >
#endif
>;

#endif

#include "SparseMatrixTest.h"
#include "../main.h"
