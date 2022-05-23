#ifdef HAVE_GTEST
#include <gtest/gtest.h>
#include <TNL/Algorithms/Segments/CSR.h>
#include <TNL/Matrices/SparseMatrix.h>

// test fixture for typed tests
//template< typename Matrix >
//class MatrixTest : public ::testing::Test
//{
//protected:
//   using MatrixType = Matrix;
//};

// types for which MatrixTest is instantiated
using MatrixTypes = ::testing::Types
<
    TNL::Matrices::SparseMatrix< int,     TNL::Devices::Host, short, TNL::Matrices::SymmetricMatrix, TNL::Algorithms::Segments::CSRDefault >,
    TNL::Matrices::SparseMatrix< long,    TNL::Devices::Host, short, TNL::Matrices::SymmetricMatrix, TNL::Algorithms::Segments::CSRDefault >,
    TNL::Matrices::SparseMatrix< float,   TNL::Devices::Host, short, TNL::Matrices::SymmetricMatrix, TNL::Algorithms::Segments::CSRDefault >,
    TNL::Matrices::SparseMatrix< double,  TNL::Devices::Host, short, TNL::Matrices::SymmetricMatrix, TNL::Algorithms::Segments::CSRDefault >,
    TNL::Matrices::SparseMatrix< int,     TNL::Devices::Host, int,   TNL::Matrices::SymmetricMatrix, TNL::Algorithms::Segments::CSRDefault >,
    TNL::Matrices::SparseMatrix< long,    TNL::Devices::Host, int,   TNL::Matrices::SymmetricMatrix, TNL::Algorithms::Segments::CSRDefault >,
    TNL::Matrices::SparseMatrix< float,   TNL::Devices::Host, int,   TNL::Matrices::SymmetricMatrix, TNL::Algorithms::Segments::CSRDefault >,
    TNL::Matrices::SparseMatrix< double,  TNL::Devices::Host, int,   TNL::Matrices::SymmetricMatrix, TNL::Algorithms::Segments::CSRDefault >,
    TNL::Matrices::SparseMatrix< int,     TNL::Devices::Host, long,  TNL::Matrices::SymmetricMatrix, TNL::Algorithms::Segments::CSRDefault >,
    TNL::Matrices::SparseMatrix< long,    TNL::Devices::Host, long,  TNL::Matrices::SymmetricMatrix, TNL::Algorithms::Segments::CSRDefault >,
    TNL::Matrices::SparseMatrix< float,   TNL::Devices::Host, long,  TNL::Matrices::SymmetricMatrix, TNL::Algorithms::Segments::CSRDefault >,
    TNL::Matrices::SparseMatrix< double,  TNL::Devices::Host, long,  TNL::Matrices::SymmetricMatrix, TNL::Algorithms::Segments::CSRDefault >
#ifdef HAVE_CUDA // Commented types are not supported by atomic operations on GPU.
    //,TNL::Matrices::SparseMatrix< int,     TNL::Devices::Cuda, short, TNL::Matrices::SymmetricMatrix, TNL::Algorithms::Segments::CSRDefault >
    //,TNL::Matrices::SparseMatrix< long,    TNL::Devices::Cuda, short, TNL::Matrices::SymmetricMatrix, TNL::Algorithms::Segments::CSRDefault >
    //,TNL::Matrices::SparseMatrix< float,   TNL::Devices::Cuda, short, TNL::Matrices::SymmetricMatrix, TNL::Algorithms::Segments::CSRDefault >
    //,TNL::Matrices::SparseMatrix< double,  TNL::Devices::Cuda, short, TNL::Matrices::SymmetricMatrix, TNL::Algorithms::Segments::CSRDefault >
    ,TNL::Matrices::SparseMatrix< int,     TNL::Devices::Cuda, int,   TNL::Matrices::SymmetricMatrix, TNL::Algorithms::Segments::CSRDefault >
    //,TNL::Matrices::SparseMatrix< long,    TNL::Devices::Cuda, int,   TNL::Matrices::SymmetricMatrix, TNL::Algorithms::Segments::CSRDefault >
    ,TNL::Matrices::SparseMatrix< float,   TNL::Devices::Cuda, int,   TNL::Matrices::SymmetricMatrix, TNL::Algorithms::Segments::CSRDefault >
    ,TNL::Matrices::SparseMatrix< double,  TNL::Devices::Cuda, int,   TNL::Matrices::SymmetricMatrix, TNL::Algorithms::Segments::CSRDefault >
    //,TNL::Matrices::SparseMatrix< int,     TNL::Devices::Cuda, long,  TNL::Matrices::SymmetricMatrix, TNL::Algorithms::Segments::CSRDefault >
    //,TNL::Matrices::SparseMatrix< long,    TNL::Devices::Cuda, long,  TNL::Matrices::SymmetricMatrix, TNL::Algorithms::Segments::CSRDefault >
    //,TNL::Matrices::SparseMatrix< float,   TNL::Devices::Cuda, long,  TNL::Matrices::SymmetricMatrix, TNL::Algorithms::Segments::CSRDefault >
    //,TNL::Matrices::SparseMatrix< double,  TNL::Devices::Cuda, long,  TNL::Matrices::SymmetricMatrix, TNL::Algorithms::Segments::CSRDefault >
#endif // HAVE_CUDA
>;

const char* saveAndLoadTestFileName = "test_SymmetricSparseMatrixTest_CSR_segments";

#include "SymmetricSparseMatrixTest.h"

#endif // HAVE_GTEST

#include "../main.h"
