// conversions have to be registered for each object file
#include "../tnl_conversions.h"

#include "SparseMatrix.h"
#include "../typedefs.h"

#include <TNL/Matrices/SparseMatrix.h>
#include <TNL/Matrices/SparseOperations.h>
#include <TNL/Algorithms/Segments/CSR.h>
#include <TNL/Algorithms/Segments/Ellpack.h>
#include <TNL/Algorithms/Segments/SlicedEllpack.h>

template< typename Device, typename Index, typename IndexAllocator >
using CSR = TNL::Algorithms::Segments::CSRDefault< Device, Index, IndexAllocator >;
template< typename Device, typename Index, typename IndexAllocator >
using Ellpack = TNL::Algorithms::Segments::Ellpack< Device, Index, IndexAllocator >;
template< typename Device, typename Index, typename IndexAllocator >
using SlicedEllpack = TNL::Algorithms::Segments::SlicedEllpack< Device, Index, IndexAllocator >;

using CSR_host = TNL::Matrices::SparseMatrix< RealType, TNL::Devices::Host, IndexType, TNL::Matrices::GeneralMatrix, CSR >;
using CSR_cuda = TNL::Matrices::SparseMatrix< RealType, TNL::Devices::Cuda, IndexType, TNL::Matrices::GeneralMatrix, CSR >;
using E_host   = TNL::Matrices::SparseMatrix< RealType, TNL::Devices::Host, IndexType, TNL::Matrices::GeneralMatrix, Ellpack >;
using E_cuda   = TNL::Matrices::SparseMatrix< RealType, TNL::Devices::Cuda, IndexType, TNL::Matrices::GeneralMatrix, Ellpack >;
using SE_host  = TNL::Matrices::SparseMatrix< RealType, TNL::Devices::Host, IndexType, TNL::Matrices::GeneralMatrix, SlicedEllpack >;
using SE_cuda  = TNL::Matrices::SparseMatrix< RealType, TNL::Devices::Cuda, IndexType, TNL::Matrices::GeneralMatrix, SlicedEllpack >;

void export_SparseMatrices( py::module & m )
{
   export_Matrix< CSR_host >( m, "CSR" );
   export_Matrix< E_host   >( m, "Ellpack" );
   export_Matrix< SE_host  >( m, "SlicedEllpack" );

   // NOTE: all exported formats (CSR, Ellpack, SlicedEllpack) use the same SegmentView,
   // so the RowView and ConstRowView are also the same types in all three formats
   export_RowView< typename CSR_host::RowView >( m, "SparseMatrixRowView" );
   export_RowView< typename CSR_host::ConstRowView >( m, "SparseMatrixConstRowView" );

   m.def("copySparseMatrix", &TNL::Matrices::copySparseMatrix< CSR_host, E_host >);
   m.def("copySparseMatrix", &TNL::Matrices::copySparseMatrix< E_host, CSR_host >);
   m.def("copySparseMatrix", &TNL::Matrices::copySparseMatrix< CSR_host, SE_host >);
   m.def("copySparseMatrix", &TNL::Matrices::copySparseMatrix< SE_host, CSR_host >);
   m.def("copySparseMatrix", &TNL::Matrices::copySparseMatrix< E_host, SE_host >);
   m.def("copySparseMatrix", &TNL::Matrices::copySparseMatrix< SE_host, E_host >);
}
