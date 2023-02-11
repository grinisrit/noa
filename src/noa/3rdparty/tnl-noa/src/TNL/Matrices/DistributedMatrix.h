// Copyright (c) 2004-2023 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

// Implemented by: Jakub Klinkovský

#pragma once

#include <type_traits>

#include <noa/3rdparty/tnl-noa/src/TNL/Containers/Subrange.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Containers/DistributedVector.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Containers/DistributedVectorView.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Exceptions/NotImplementedError.h>

namespace noa::TNL {
namespace Matrices {

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

   template< typename _Real = RealType, typename _Device = DeviceType, typename _Index = IndexType >
   using Self = DistributedMatrix< typename MatrixType::template Self< _Real, _Device, _Index > >;

   DistributedMatrix() = default;

   DistributedMatrix( DistributedMatrix& ) = default;

   DistributedMatrix( LocalRangeType localRowRange, IndexType rows, IndexType columns, const MPI::Comm& communicator );

   void
   setDistribution( LocalRangeType localRowRange, IndexType rows, IndexType columns, const MPI::Comm& communicator );

   const LocalRangeType&
   getLocalRowRange() const;

   const MPI::Comm&
   getCommunicator() const;

   const Matrix&
   getLocalMatrix() const;

   Matrix&
   getLocalMatrix();

   /*
    * Some common Matrix methods follow below.
    */

   DistributedMatrix&
   operator=( const DistributedMatrix& matrix );

   template< typename MatrixT >
   DistributedMatrix&
   operator=( const MatrixT& matrix );

   template< typename MatrixT >
   void
   setLike( const MatrixT& matrix );

   void
   reset();

   IndexType
   getRows() const;

   IndexType
   getColumns() const;

   template< typename RowCapacitiesVector >
   void
   setRowCapacities( const RowCapacitiesVector& rowCapacities );

   template< typename Vector >
   void
   getCompressedRowLengths( Vector& rowLengths ) const;

   IndexType
   getRowCapacity( IndexType row ) const;

   void
   setElement( IndexType row, IndexType column, RealType value );

   RealType
   getElement( IndexType row, IndexType column ) const;

   RealType
   getElementFast( IndexType row, IndexType column ) const;

   MatrixRow
   getRow( IndexType row );

   ConstMatrixRow
   getRow( IndexType row ) const;

   // multiplication with a global vector
   template< typename InVector, typename OutVector >
   typename std::enable_if< ! HasGetCommunicatorMethod< InVector >::value >::type
   vectorProduct( const InVector& inVector, OutVector& outVector ) const;

   // Optimization for distributed matrix-vector multiplication
   void
   updateVectorProductCommunicationPattern();

   // multiplication with a distributed vector
   // (not const because it modifies internal bufers)
   template< typename InVector, typename OutVector >
   typename std::enable_if< HasGetCommunicatorMethod< InVector >::value >::type
   vectorProduct( const InVector& inVector, OutVector& outVector ) const;

   // TODO
   template< typename Fetch, typename Reduce, typename Keep, typename FetchReal >
   void
   reduceRows( IndexType begin, IndexType end, Fetch& fetch, const Reduce& reduce, Keep& keep, const FetchReal& identity )
   {
      throw Exceptions::NotImplementedError( "reduceRows is not implemented in DistributedMatrix" );
   }

   template< typename Fetch, typename Reduce, typename Keep, typename FetchReal >
   void
   reduceRows( IndexType begin, IndexType end, Fetch& fetch, const Reduce& reduce, Keep& keep, const FetchReal& identity ) const
   {
      throw Exceptions::NotImplementedError( "reduceRows is not implemented in DistributedMatrix" );
   }

   template< typename Fetch, typename Reduce, typename Keep, typename FetchReal >
   void
   reduceAllRows( Fetch& fetch, const Reduce& reduce, Keep& keep, const FetchReal& identity )
   {
      throw Exceptions::NotImplementedError( "reduceAllRows is not implemented in DistributedMatrix" );
   }

   template< typename Fetch, typename Reduce, typename Keep, typename FetchReal >
   void
   reduceAllRows( Fetch& fetch, const Reduce& reduce, Keep& keep, const FetchReal& identity ) const
   {
      throw Exceptions::NotImplementedError( "reduceAllRows is not implemented in DistributedMatrix" );
   }

   template< typename Function >
   void
   forElements( IndexType begin, IndexType end, Function&& function ) const
   {
      throw Exceptions::NotImplementedError( "forElements is not implemented in DistributedMatrix" );
   }

   template< typename Function >
   void
   forElements( IndexType begin, IndexType end, Function&& function )
   {
      throw Exceptions::NotImplementedError( "forElements is not implemented in DistributedMatrix" );
   }

   template< typename Function >
   void
   forAllElements( Function&& function ) const
   {
      throw Exceptions::NotImplementedError( "forAllElements is not implemented in DistributedMatrix" );
   }

   template< typename Function >
   void
   forAllElements( Function&& function )
   {
      throw Exceptions::NotImplementedError( "forAllElements is not implemented in DistributedMatrix" );
   }

   template< typename Function >
   void
   forRows( IndexType begin, IndexType end, Function&& function )
   {
      throw Exceptions::NotImplementedError( "forRows is not implemented in DistributedMatrix" );
   }

   template< typename Function >
   void
   forRows( IndexType begin, IndexType end, Function&& function ) const
   {
      throw Exceptions::NotImplementedError( "forRows is not implemented in DistributedMatrix" );
   }

   template< typename Function >
   void
   forAllRows( Function&& function )
   {
      throw Exceptions::NotImplementedError( "forAllRows is not implemented in DistributedMatrix" );
   }

   template< typename Function >
   void
   forAllRows( Function&& function ) const
   {
      throw Exceptions::NotImplementedError( "forAllRows is not implemented in DistributedMatrix" );
   }

protected:
   LocalRangeType localRowRange;
   IndexType rows = 0;  // global rows count
   MPI::Comm communicator = MPI_COMM_NULL;
   Matrix localMatrix;
};

}  // namespace Matrices
}  // namespace noa::TNL

#include "DistributedMatrix_impl.h"
