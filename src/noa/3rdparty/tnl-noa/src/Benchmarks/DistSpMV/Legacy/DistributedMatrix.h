// Implemented by: Jakub Klinkovský

#pragma once

#include <type_traits>

#include <TNL/Containers/Subrange.h>
#include <TNL/Containers/DistributedVector.h>
#include <TNL/Containers/DistributedVectorView.h>
#include "DistributedSpMV.h"

namespace TNL {
namespace Matrices {
namespace Legacy {

// TODO: 2D distribution for dense matrices (maybe it should be in different template,
//       because e.g. setRowFast doesn't make sense for dense matrices)
template< typename Matrix >
class DistributedMatrix
{
public:
   using MatrixType = Matrix;
   using RealType = typename Matrix::RealType;
   using DeviceType = typename Matrix::DeviceType;
   using IndexType = typename Matrix::IndexType;
   using LocalRangeType = Containers::Subrange< typename Matrix::IndexType >;

   using RowsCapacitiesType = Containers::DistributedVector< IndexType, DeviceType, IndexType >;

   using MatrixRow = typename Matrix::RowView;
   using ConstMatrixRow = typename Matrix::ConstRowView;

   template< typename _Real = RealType,
             typename _Device = DeviceType,
             typename _Index = IndexType >
   using Self = DistributedMatrix< typename MatrixType::template Self< _Real, _Device, _Index > >;

   DistributedMatrix() = default;

   DistributedMatrix( DistributedMatrix& ) = default;

   DistributedMatrix( LocalRangeType localRowRange, IndexType rows, IndexType columns, MPI_Comm communicator );

   void setDistribution( LocalRangeType localRowRange, IndexType rows, IndexType columns, MPI_Comm communicator );

   const LocalRangeType& getLocalRowRange() const;

   MPI_Comm getCommunicator() const;

   const Matrix& getLocalMatrix() const;

   Matrix& getLocalMatrix();


   /*
    * Some common Matrix methods follow below.
    */

   DistributedMatrix& operator=( const DistributedMatrix& matrix );

   template< typename MatrixT >
   DistributedMatrix& operator=( const MatrixT& matrix );

   template< typename MatrixT >
   void setLike( const MatrixT& matrix );

   void reset();

   IndexType getRows() const;

   IndexType getColumns() const;

   template< typename RowCapacitiesVector >
   void setRowCapacities( const RowCapacitiesVector& rowCapacities );

   template< typename Vector >
   void getCompressedRowLengths( Vector& rowLengths ) const;

   IndexType getRowCapacity( IndexType row ) const;

   void setElement( IndexType row,
                    IndexType column,
                    RealType value );

   RealType getElement( IndexType row,
                        IndexType column ) const;

   RealType getElementFast( IndexType row,
                            IndexType column ) const;

   MatrixRow getRow( IndexType row );

   ConstMatrixRow getRow( IndexType row ) const;

   // multiplication with a global vector
   template< typename InVector,
             typename OutVector >
   typename std::enable_if< ! HasGetCommunicatorMethod< InVector >::value >::type
   vectorProduct( const InVector& inVector,
                  OutVector& outVector ) const;

   // Optimization for distributed matrix-vector multiplication
   void updateVectorProductCommunicationPattern();

   // multiplication with a distributed vector
   // (not const because it modifies internal bufers)
   template< typename InVector,
             typename OutVector >
   typename std::enable_if< HasGetCommunicatorMethod< InVector >::value >::type
   vectorProduct( const InVector& inVector,
                  OutVector& outVector ) const;

protected:
   LocalRangeType localRowRange;
   IndexType rows = 0;  // global rows count
   MPI_Comm communicator = MPI_COMM_NULL;
   Matrix localMatrix;

   DistributedSpMV< Matrix > spmv;
};

} // namespace Legacy
} // namespace Matrices
} // namespace TNL

#include "DistributedMatrix_impl.h"
