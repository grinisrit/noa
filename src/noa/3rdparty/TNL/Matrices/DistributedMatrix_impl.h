// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

// Implemented by: Jakub Klinkovský

#pragma once

#include "DistributedMatrix.h"

namespace noa::TNL {
namespace Matrices {

template< typename Matrix >
DistributedMatrix< Matrix >::
DistributedMatrix( LocalRangeType localRowRange, IndexType rows, IndexType columns, MPI_Comm communicator )
{
   setDistribution( localRowRange, rows, columns, communicator );
}

template< typename Matrix >
void
DistributedMatrix< Matrix >::
setDistribution( LocalRangeType localRowRange, IndexType rows, IndexType columns, MPI_Comm communicator )
{
   this->localRowRange = localRowRange;
   this->rows = rows;
   this->communicator = communicator;
   if( communicator != MPI_COMM_NULL )
      localMatrix.setDimensions( localRowRange.getSize(), columns );
}

template< typename Matrix >
const Containers::Subrange< typename Matrix::IndexType >&
DistributedMatrix< Matrix >::
getLocalRowRange() const
{
   return localRowRange;
}

template< typename Matrix >
MPI_Comm
DistributedMatrix< Matrix >::
getCommunicator() const
{
   return communicator;
}

template< typename Matrix >
const Matrix&
DistributedMatrix< Matrix >::
getLocalMatrix() const
{
   return localMatrix;
}

template< typename Matrix >
Matrix&
DistributedMatrix< Matrix >::
getLocalMatrix()
{
   return localMatrix;
}


/*
 * Some common Matrix methods follow below.
 */

template< typename Matrix >
DistributedMatrix< Matrix >&
DistributedMatrix< Matrix >::
operator=( const DistributedMatrix& matrix )
{
   setLike( matrix );
   localMatrix = matrix.getLocalMatrix();
   return *this;
}

template< typename Matrix >
   template< typename MatrixT >
DistributedMatrix< Matrix >&
DistributedMatrix< Matrix >::
operator=( const MatrixT& matrix )
{
   setLike( matrix );
   localMatrix = matrix.getLocalMatrix();
   return *this;
}

template< typename Matrix >
   template< typename MatrixT >
void
DistributedMatrix< Matrix >::
setLike( const MatrixT& matrix )
{
   localRowRange = matrix.getLocalRowRange();
   rows = matrix.getRows();
   communicator = matrix.getCommunicator();
   localMatrix.setLike( matrix.getLocalMatrix() );
}

template< typename Matrix >
void
DistributedMatrix< Matrix >::
reset()
{
   localRowRange.reset();
   rows = 0;
   communicator = MPI_COMM_NULL;
   localMatrix.reset();
}

template< typename Matrix >
typename Matrix::IndexType
DistributedMatrix< Matrix >::
getRows() const
{
   return rows;
}

template< typename Matrix >
typename Matrix::IndexType
DistributedMatrix< Matrix >::
getColumns() const
{
   return localMatrix.getColumns();
}

template< typename Matrix >
   template< typename RowCapacitiesVector >
void
DistributedMatrix< Matrix >::
setRowCapacities( const RowCapacitiesVector& rowCapacities )
{
   TNL_ASSERT_EQ( rowCapacities.getSize(), getRows(), "row lengths vector has wrong size" );
   TNL_ASSERT_EQ( rowCapacities.getLocalRange(), getLocalRowRange(), "row lengths vector has wrong distribution" );
   TNL_ASSERT_EQ( rowCapacities.getCommunicator(), getCommunicator(), "row lengths vector has wrong communicator" );

   if( getCommunicator() != MPI_COMM_NULL ) {
      localMatrix.setRowCapacities( rowCapacities.getConstLocalView() );
   }
}

template< typename Matrix >
   template< typename Vector >
void
DistributedMatrix< Matrix >::
getCompressedRowLengths( Vector& rowLengths ) const
{
   if( getCommunicator() != MPI_COMM_NULL ) {
      rowLengths.setDistribution( getLocalRowRange(), 0, getRows(), getCommunicator() );
      auto localRowLengths = rowLengths.getLocalView();
      localMatrix.getCompressedRowLengths( localRowLengths );
   }
}

template< typename Matrix >
typename Matrix::IndexType
DistributedMatrix< Matrix >::
getRowCapacity( IndexType row ) const
{
   const IndexType localRow = localRowRange.getLocalIndex( row );
   return localMatrix.getRowCapacity( localRow );
}

template< typename Matrix >
void
DistributedMatrix< Matrix >::
setElement( IndexType row,
            IndexType column,
            RealType value )
{
   const IndexType localRow = localRowRange.getLocalIndex( row );
   localMatrix.setElement( localRow, column, value );
}

template< typename Matrix >
typename Matrix::RealType
DistributedMatrix< Matrix >::
getElement( IndexType row,
            IndexType column ) const
{
   const IndexType localRow = localRowRange.getLocalIndex( row );
   return localMatrix.getElement( localRow, column );
}

template< typename Matrix >
typename Matrix::RealType
DistributedMatrix< Matrix >::
getElementFast( IndexType row,
                IndexType column ) const
{
   const IndexType localRow = localRowRange.getLocalIndex( row );
   return localMatrix.getElementFast( localRow, column );
}

template< typename Matrix >
typename DistributedMatrix< Matrix >::MatrixRow
DistributedMatrix< Matrix >::
getRow( IndexType row )
{
   const IndexType localRow = localRowRange.getLocalIndex( row );
   return localMatrix.getRow( localRow );
}

template< typename Matrix >
typename DistributedMatrix< Matrix >::ConstMatrixRow
DistributedMatrix< Matrix >::
getRow( IndexType row ) const
{
   const IndexType localRow = localRowRange.getLocalIndex( row );
   return localMatrix.getRow( localRow );
}

template< typename Matrix >
   template< typename InVector,
             typename OutVector >
typename std::enable_if< ! HasGetCommunicatorMethod< InVector >::value >::type
DistributedMatrix< Matrix >::
vectorProduct( const InVector& inVector,
               OutVector& outVector ) const
{
   TNL_ASSERT_EQ( inVector.getSize(), getColumns(), "input vector has wrong size" );
   TNL_ASSERT_EQ( outVector.getSize(), getRows(), "output vector has wrong size" );
   TNL_ASSERT_EQ( outVector.getLocalRange(), getLocalRowRange(), "output vector has wrong distribution" );
   TNL_ASSERT_EQ( outVector.getCommunicator(), getCommunicator(), "output vector has wrong communicator" );

   auto outView = outVector.getLocalView();
   localMatrix.vectorProduct( inVector, outView );
}

template< typename Matrix >
   template< typename InVector,
             typename OutVector >
typename std::enable_if< HasGetCommunicatorMethod< InVector >::value >::type
DistributedMatrix< Matrix >::
vectorProduct( const InVector& inVector,
               OutVector& outVector ) const
{
   TNL_ASSERT_EQ( inVector.getLocalRange(), getLocalRowRange(), "input vector has wrong distribution" );
   TNL_ASSERT_EQ( inVector.getCommunicator(), getCommunicator(), "input vector has wrong communicator" );
   TNL_ASSERT_EQ( outVector.getSize(), getRows(), "output vector has wrong size" );
   TNL_ASSERT_EQ( outVector.getLocalRange(), getLocalRowRange(), "output vector has wrong distribution" );
   TNL_ASSERT_EQ( outVector.getCommunicator(), getCommunicator(), "output vector has wrong communicator" );

   if( getCommunicator() == MPI_COMM_NULL )
      return;

   TNL_ASSERT_EQ( inVector.getConstLocalViewWithGhosts().getSize(), localMatrix.getColumns(), "the matrix uses non-local and non-ghost column indices" );
   TNL_ASSERT_EQ( inVector.getGhosts(), localMatrix.getColumns() - localMatrix.getRows(), "input vector has wrong ghosts size" );
   TNL_ASSERT_EQ( outVector.getGhosts(), localMatrix.getColumns() - localMatrix.getRows(), "output vector has wrong ghosts size" );
   TNL_ASSERT_EQ( outVector.getConstLocalView().getSize(), localMatrix.getRows(), "number of local matrix rows does not match the output vector local size" );

   inVector.waitForSynchronization();
   const auto inView = inVector.getConstLocalViewWithGhosts();
   auto outView = outVector.getLocalView();
   localMatrix.vectorProduct( inView, outView );
   // TODO: synchronization is not always necessary, e.g. when a preconditioning step follows
//   outVector.startSynchronization();
}

} // namespace Matrices
} // namespace noa::TNL
