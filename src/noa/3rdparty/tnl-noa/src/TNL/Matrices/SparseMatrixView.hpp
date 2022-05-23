// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <iomanip>
#include <functional>
#include <noa/3rdparty/tnl-noa/src/TNL/Matrices/SparseMatrixView.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Algorithms/reduce.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Algorithms/AtomicOperations.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Matrices/details/SparseMatrix.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Exceptions/NotImplementedError.h>

namespace noa::TNL {
namespace Matrices {

template< typename Real,
          typename Device,
          typename Index,
          typename MatrixType,
          template< typename, typename >
          class SegmentsView,
          typename ComputeReal >
__cuda_callable__
SparseMatrixView< Real, Device, Index, MatrixType, SegmentsView, ComputeReal >::SparseMatrixView() = default;

template< typename Real,
          typename Device,
          typename Index,
          typename MatrixType,
          template< typename, typename >
          class SegmentsView,
          typename ComputeReal >
__cuda_callable__
SparseMatrixView< Real, Device, Index, MatrixType, SegmentsView, ComputeReal >::SparseMatrixView(
   const IndexType rows,
   const IndexType columns,
   const ValuesViewType& values,
   const ColumnsIndexesViewType& columnIndexes,
   const SegmentsViewType& segments )
: MatrixView< Real, Device, Index >( rows, columns, values ), columnIndexes( columnIndexes ), segments( segments )
{}

template< typename Real,
          typename Device,
          typename Index,
          typename MatrixType,
          template< typename, typename >
          class SegmentsView,
          typename ComputeReal >
__cuda_callable__
auto
SparseMatrixView< Real, Device, Index, MatrixType, SegmentsView, ComputeReal >::getView() -> ViewType
{
   return ViewType( this->getRows(),
                    this->getColumns(),
                    this->getValues().getView(),
                    this->columnIndexes.getView(),
                    this->segments.getView() );
}

template< typename Real,
          typename Device,
          typename Index,
          typename MatrixType,
          template< typename, typename >
          class SegmentsView,
          typename ComputeReal >
__cuda_callable__
auto
SparseMatrixView< Real, Device, Index, MatrixType, SegmentsView, ComputeReal >::getConstView() const -> ConstViewType
{
   return ConstViewType( this->getRows(),
                         this->getColumns(),
                         this->getValues().getConstView(),
                         this->getColumnIndexes().getConstView(),
                         const_cast< SparseMatrixView* >( this )->segments.getView() );
}

template< typename Real,
          typename Device,
          typename Index,
          typename MatrixType,
          template< typename, typename >
          class SegmentsView,
          typename ComputeReal >
std::string
SparseMatrixView< Real, Device, Index, MatrixType, SegmentsView, ComputeReal >::getSerializationType()
{
   return "Matrices::SparseMatrix< " + TNL::getSerializationType< RealType >() + ", "
        + TNL::getSerializationType< SegmentsViewType >() + ", [any_device], " + TNL::getSerializationType< IndexType >() + ", "
        + MatrixType::getSerializationType() + ", [any_allocator], [any_allocator] >";
}

template< typename Real,
          typename Device,
          typename Index,
          typename MatrixType,
          template< typename, typename >
          class SegmentsView,
          typename ComputeReal >
std::string
SparseMatrixView< Real, Device, Index, MatrixType, SegmentsView, ComputeReal >::getSerializationTypeVirtual() const
{
   return this->getSerializationType();
}

template< typename Real,
          typename Device,
          typename Index,
          typename MatrixType,
          template< typename, typename >
          class SegmentsView,
          typename ComputeReal >
template< typename Vector >
void
SparseMatrixView< Real, Device, Index, MatrixType, SegmentsView, ComputeReal >::getCompressedRowLengths(
   Vector& rowLengths ) const
{
   details::set_size_if_resizable( rowLengths, this->getRows() );
   rowLengths = 0;
   auto rowLengths_view = rowLengths.getView();
   auto fetch = [] __cuda_callable__( IndexType row, IndexType column, const RealType& value ) -> IndexType
   {
      return ( value != 0.0 );
   };
   auto keep = [ = ] __cuda_callable__( const IndexType rowIdx, const IndexType value ) mutable
   {
      rowLengths_view[ rowIdx ] = value;
   };
   this->reduceAllRows( fetch, std::plus<>{}, keep, 0 );
}

template< typename Real,
          typename Device,
          typename Index,
          typename MatrixType,
          template< typename, typename >
          class SegmentsView,
          typename ComputeReal >
template< typename Vector >
void
SparseMatrixView< Real, Device, Index, MatrixType, SegmentsView, ComputeReal >::getRowCapacities( Vector& rowLengths ) const
{
   details::set_size_if_resizable( rowLengths, this->getRows() );
   rowLengths = 0;
   auto rowLengths_view = rowLengths.getView();
   auto fetch = [] __cuda_callable__( IndexType row, IndexType column, const RealType& value ) -> IndexType
   {
      return 1;
   };
   auto keep = [ = ] __cuda_callable__( const IndexType rowIdx, const IndexType value ) mutable
   {
      rowLengths_view[ rowIdx ] = value;
   };
   this->reduceAllRows( fetch, std::plus<>{}, keep, 0 );
}

template< typename Real,
          typename Device,
          typename Index,
          typename MatrixType,
          template< typename, typename >
          class SegmentsView,
          typename ComputeReal >
__cuda_callable__
Index
SparseMatrixView< Real, Device, Index, MatrixType, SegmentsView, ComputeReal >::getRowCapacity( const IndexType row ) const
{
   return this->segments.getSegmentSize( row );
}

template< typename Real,
          typename Device,
          typename Index,
          typename MatrixType,
          template< typename, typename >
          class SegmentsView,
          typename ComputeReal >
Index
SparseMatrixView< Real, Device, Index, MatrixType, SegmentsView, ComputeReal >::getNonzeroElementsCount() const
{
   const auto columns_view = this->columnIndexes.getConstView();
   const IndexType paddingIndex = this->getPaddingIndex();
   if( ! isSymmetric() ) {
      auto fetch = [ = ] __cuda_callable__( const IndexType i ) -> IndexType
      {
         return ( columns_view[ i ] != paddingIndex );
      };
      return Algorithms::reduce< DeviceType >( (IndexType) 0, this->columnIndexes.getSize(), fetch, std::plus<>{}, 0 );
   }
   else {
      const auto rows = this->getRows();
      const auto columns = this->getColumns();
      Containers::Vector< IndexType, DeviceType, IndexType > row_sums( this->getRows(), 0 );
      auto row_sums_view = row_sums.getView();
      const auto columnIndexesView = this->columnIndexes.getConstView();
      auto fetch = [ = ] __cuda_callable__( IndexType row, IndexType localIdx, IndexType globalIdx, bool& compute ) -> IndexType
      {
         const IndexType column = columnIndexesView[ globalIdx ];
         compute = ( column != paddingIndex );
         if( ! compute )
            return 0.0;
         return 1 + ( column != row && column < rows && row < columns );  // the addition is for non-diagonal elements
      };
      auto keeper = [ = ] __cuda_callable__( IndexType row, const IndexType& value ) mutable
      {
         row_sums_view[ row ] = value;
      };
      this->segments.reduceSegments( 0, this->getRows(), fetch, std::plus<>{}, keeper, (IndexType) 0 );
      return sum( row_sums );
   }
}

template< typename Real,
          typename Device,
          typename Index,
          typename MatrixType,
          template< typename, typename >
          class SegmentsView,
          typename ComputeReal >
__cuda_callable__
auto
SparseMatrixView< Real, Device, Index, MatrixType, SegmentsView, ComputeReal >::getRow( const IndexType& rowIdx ) const
   -> ConstRowView
{
   TNL_ASSERT_LT( rowIdx, this->getRows(), "Row index is larger than number of matrix rows." );
   return ConstRowView( this->segments.getSegmentView( rowIdx ), this->values, this->columnIndexes );
}

template< typename Real,
          typename Device,
          typename Index,
          typename MatrixType,
          template< typename, typename >
          class SegmentsView,
          typename ComputeReal >
__cuda_callable__
auto
SparseMatrixView< Real, Device, Index, MatrixType, SegmentsView, ComputeReal >::getRow( const IndexType& rowIdx ) -> RowView
{
   TNL_ASSERT_LT( rowIdx, this->getRows(), "Row index is larger than number of matrix rows." );
   return RowView( this->segments.getSegmentView( rowIdx ), this->values, this->columnIndexes );
}

template< typename Real,
          typename Device,
          typename Index,
          typename MatrixType,
          template< typename, typename >
          class SegmentsView,
          typename ComputeReal >
__cuda_callable__
void
SparseMatrixView< Real, Device, Index, MatrixType, SegmentsView, ComputeReal >::setElement( const IndexType row,
                                                                                            const IndexType column,
                                                                                            const RealType& value )
{
   this->addElement( row, column, value, 0.0 );
}

template< typename Real,
          typename Device,
          typename Index,
          typename MatrixType,
          template< typename, typename >
          class SegmentsView,
          typename ComputeReal >
__cuda_callable__
void
SparseMatrixView< Real, Device, Index, MatrixType, SegmentsView, ComputeReal >::addElement(
   IndexType row,
   IndexType column,
   const RealType& value,
   const RealType& thisElementMultiplicator )
{
   TNL_ASSERT_GE( row, 0, "Sparse matrix row index cannot be negative." );
   TNL_ASSERT_LT( row, this->getRows(), "Sparse matrix row index is larger than number of matrix rows." );
   TNL_ASSERT_GE( column, 0, "Sparse matrix column index cannot be negative." );
   TNL_ASSERT_LT( column, this->getColumns(), "Sparse matrix column index is larger than number of matrix columns." );

   if( isSymmetric() && row < column ) {
      swap( row, column );
      TNL_ASSERT_LT( row, this->getRows(), "Column index is out of the symmetric part of the matrix after transposition." );
      TNL_ASSERT_LT( column, this->getColumns(), "Row index is out of the symmetric part of the matrix after transposition." );
   }

   const IndexType rowSize = this->segments.getSegmentSize( row );
   IndexType col( this->getPaddingIndex() );
   IndexType i;
   IndexType globalIdx;
   for( i = 0; i < rowSize; i++ ) {
      globalIdx = this->segments.getGlobalIndex( row, i );
      TNL_ASSERT_LT( globalIdx, this->columnIndexes.getSize(), "" );
      col = this->columnIndexes.getElement( globalIdx );
      if( col == column ) {
         if( ! isBinary() )
            this->values.setElement( globalIdx, thisElementMultiplicator * this->values.getElement( globalIdx ) + value );
         return;
      }
      if( col == this->getPaddingIndex() || col > column )
         break;
   }
   if( i == rowSize ) {
#ifndef __CUDA_ARCH__
      std::stringstream msg;
      msg << "The capacity of the sparse matrix row number " << row << " was exceeded.";
      throw std::logic_error( msg.str() );
#else
      TNL_ASSERT_TRUE( false, "" );
      return;
#endif
   }
   if( col == this->getPaddingIndex() ) {
      this->columnIndexes.setElement( globalIdx, column );
      if( ! isBinary() )
         this->values.setElement( globalIdx, value );
      return;
   }
   else {
      IndexType j = rowSize - 1;
      while( j > i ) {
         const IndexType globalIdx1 = this->segments.getGlobalIndex( row, j );
         const IndexType globalIdx2 = this->segments.getGlobalIndex( row, j - 1 );
         TNL_ASSERT_LT( globalIdx1, this->columnIndexes.getSize(), "" );
         TNL_ASSERT_LT( globalIdx2, this->columnIndexes.getSize(), "" );
         this->columnIndexes.setElement( globalIdx1, this->columnIndexes.getElement( globalIdx2 ) );
         if( ! isBinary() )
            this->values.setElement( globalIdx1, this->values.getElement( globalIdx2 ) );
         j--;
      }

      this->columnIndexes.setElement( globalIdx, column );
      if( ! isBinary() )
         this->values.setElement( globalIdx, value );
      return;
   }
}

template< typename Real,
          typename Device,
          typename Index,
          typename MatrixType,
          template< typename, typename >
          class SegmentsView,
          typename ComputeReal >
__cuda_callable__
auto
SparseMatrixView< Real, Device, Index, MatrixType, SegmentsView, ComputeReal >::getElement( IndexType row,
                                                                                            IndexType column ) const -> RealType
{
   TNL_ASSERT_GE( row, 0, "Sparse matrix row index cannot be negative." );
   TNL_ASSERT_LT( row, this->getRows(), "Sparse matrix row index is larger than number of matrix rows." );
   TNL_ASSERT_GE( column, 0, "Sparse matrix column index cannot be negative." );
   TNL_ASSERT_LT( column, this->getColumns(), "Sparse matrix column index is larger than number of matrix columns." );

   if( isSymmetric() && row < column ) {
      swap( row, column );
      if( row >= this->getRows() || column >= this->getColumns() )
         return 0.0;
   }

   const IndexType rowSize = this->segments.getSegmentSize( row );
   for( IndexType i = 0; i < rowSize; i++ ) {
      const IndexType globalIdx = this->segments.getGlobalIndex( row, i );
      TNL_ASSERT_LT( globalIdx, this->columnIndexes.getSize(), "" );
      const IndexType col = this->columnIndexes.getElement( globalIdx );
      if( col == column ) {
         if( isBinary() )
            return 1;
         else
            return this->values.getElement( globalIdx );
      }
   }
   return 0.0;
}

template< typename Real,
          typename Device,
          typename Index,
          typename MatrixType,
          template< typename, typename >
          class SegmentsView,
          typename ComputeReal >
template< typename InVector, typename OutVector >
void
SparseMatrixView< Real, Device, Index, MatrixType, SegmentsView, ComputeReal >::vectorProduct(
   const InVector& inVector,
   OutVector& outVector,
   ComputeRealType matrixMultiplicator,
   ComputeRealType outVectorMultiplicator,
   IndexType firstRow,
   IndexType lastRow ) const
{
   TNL_ASSERT_EQ( this->getColumns(), inVector.getSize(), "Matrix columns do not fit with input vector." );
   TNL_ASSERT_EQ( this->getRows(), outVector.getSize(), "Matrix rows do not fit with output vector." );

   using OutVectorReal = typename OutVector::RealType;
   static_assert(
      ! MatrixType::isSymmetric() || ! std::is_same< Device, Devices::Cuda >::value
         || ( std::is_same< OutVectorReal, float >::value || std::is_same< OutVectorReal, double >::value
              || std::is_same< OutVectorReal, int >::value || std::is_same< OutVectorReal, long long int >::value ),
      "Given Real type is not supported by atomic operations on GPU which are necessary for symmetric operations." );

   const auto inVectorView = inVector.getConstView();
   auto outVectorView = outVector.getView();
   const auto valuesView = this->values.getConstView();
   const auto columnIndexesView = this->columnIndexes.getConstView();
   const IndexType paddingIndex = this->getPaddingIndex();
   if( isSymmetric() && outVectorMultiplicator != 1.0 )
      outVector *= outVectorMultiplicator;
   auto symmetricFetch = [ = ] __cuda_callable__(
                            IndexType row, IndexType localIdx, IndexType globalIdx, bool& compute ) mutable -> ComputeRealType
   {
      const IndexType column = columnIndexesView[ globalIdx ];
      compute = ( column != paddingIndex );
      if( ! compute )
         return 0.0;
      if( isSymmetric() && column < row ) {
         if( isBinary() )
            Algorithms::AtomicOperations< DeviceType >::add( outVectorView[ column ],
                                                             (OutVectorReal) matrixMultiplicator * inVectorView[ row ] );
         else
            Algorithms::AtomicOperations< DeviceType >::add(
               outVectorView[ column ], (OutVectorReal) matrixMultiplicator * valuesView[ globalIdx ] * inVectorView[ row ] );
      }
      if( isBinary() )
         return inVectorView[ column ];
      return valuesView[ globalIdx ] * inVectorView[ column ];
   };
   auto fetch = [ = ] __cuda_callable__( IndexType globalIdx, bool& compute ) mutable -> ComputeRealType
   {
      const IndexType column = columnIndexesView[ globalIdx ];
      if( SegmentsViewType::havePadding() ) {
         compute = ( column != paddingIndex );
         if( ! compute )
            return 0.0;
      }
      if( isBinary() )
         return inVectorView[ column ];
      return valuesView[ globalIdx ] * inVectorView[ column ];
   };

   auto keeperGeneral = [ = ] __cuda_callable__( IndexType row, const ComputeRealType& value ) mutable
   {
      if( isSymmetric() ) {
         typename OutVector::RealType aux = matrixMultiplicator * value;
         Algorithms::AtomicOperations< DeviceType >::add( outVectorView[ row ], aux );
      }
      else {
         if( outVectorMultiplicator == 0.0 )
            outVectorView[ row ] = matrixMultiplicator * value;
         else
            outVectorView[ row ] = outVectorMultiplicator * outVectorView[ row ] + matrixMultiplicator * value;
      }
   };
   auto keeperDirect = [ = ] __cuda_callable__( IndexType row, const ComputeRealType& value ) mutable
   {
      outVectorView[ row ] = value;
   };
   auto keeperMatrixMult = [ = ] __cuda_callable__( IndexType row, const ComputeRealType& value ) mutable
   {
      outVectorView[ row ] = matrixMultiplicator * value;
   };
   auto keeperVectorMult = [ = ] __cuda_callable__( IndexType row, const ComputeRealType& value ) mutable
   {
      outVectorView[ row ] = outVectorMultiplicator * outVectorView[ row ] + value;
   };

   if( lastRow == 0 )
      lastRow = this->getRows();
   if( isSymmetric() )
      this->segments.reduceSegments( firstRow, lastRow, symmetricFetch, std::plus<>{}, keeperGeneral, (ComputeRealType) 0.0 );
   else {
      if( outVectorMultiplicator == 0.0 ) {
         if( matrixMultiplicator == 1.0 )
            this->segments.reduceSegments( firstRow, lastRow, fetch, std::plus<>{}, keeperDirect, (ComputeRealType) 0.0 );
         else
            this->segments.reduceSegments( firstRow, lastRow, fetch, std::plus<>{}, keeperMatrixMult, (ComputeRealType) 0.0 );
      }
      else {
         if( matrixMultiplicator == 1.0 )
            this->segments.reduceSegments( firstRow, lastRow, fetch, std::plus<>{}, keeperVectorMult, (ComputeRealType) 0.0 );
         else
            this->segments.reduceSegments( firstRow, lastRow, fetch, std::plus<>{}, keeperGeneral, (ComputeRealType) 0.0 );
      }
   }
}

template< typename Real,
          typename Device,
          typename Index,
          typename MatrixType,
          template< typename, typename >
          class SegmentsView,
          typename ComputeReal >
template< typename Fetch, typename Reduce, typename Keep, typename FetchValue >
void
SparseMatrixView< Real, Device, Index, MatrixType, SegmentsView, ComputeReal >::reduceRows( IndexType begin,
                                                                                            IndexType end,
                                                                                            Fetch& fetch,
                                                                                            const Reduce& reduce,
                                                                                            Keep& keep,
                                                                                            const FetchValue& identity )
{
   auto columns_view = this->columnIndexes.getView();
   auto values_view = this->values.getView();
   const IndexType paddingIndex_ = this->getPaddingIndex();
   auto fetch_ = [ = ] __cuda_callable__( IndexType rowIdx, IndexType localIdx, IndexType globalIdx, bool& compute ) mutable
      -> decltype( fetch( IndexType(), IndexType(), RealType() ) )
   {
      IndexType& columnIdx = columns_view[ globalIdx ];
      if( columnIdx != paddingIndex_ ) {
         if( isBinary() )
            return fetch( rowIdx, columnIdx, 1 );
         else
            return fetch( rowIdx, columnIdx, values_view[ globalIdx ] );
      }
      return identity;
   };
   this->segments.reduceSegments( begin, end, fetch_, reduce, keep, identity );
}

template< typename Real,
          typename Device,
          typename Index,
          typename MatrixType,
          template< typename, typename >
          class SegmentsView,
          typename ComputeReal >
template< typename Fetch, typename Reduce, typename Keep, typename FetchValue >
void
SparseMatrixView< Real, Device, Index, MatrixType, SegmentsView, ComputeReal >::reduceRows( IndexType begin,
                                                                                            IndexType end,
                                                                                            Fetch& fetch,
                                                                                            const Reduce& reduce,
                                                                                            Keep& keep,
                                                                                            const FetchValue& identity ) const
{
   const auto columns_view = this->columnIndexes.getConstView();
   const auto values_view = this->values.getConstView();
   const IndexType paddingIndex_ = this->getPaddingIndex();
   auto fetch_ = [ = ] __cuda_callable__( IndexType rowIdx, IndexType localIdx, IndexType globalIdx, bool& compute ) mutable
      -> decltype( fetch( IndexType(), IndexType(), RealType() ) )
   {
      TNL_ASSERT_LT( globalIdx, (IndexType) columns_view.getSize(), "" );
      IndexType columnIdx = columns_view[ globalIdx ];
      if( columnIdx != paddingIndex_ ) {
         if( isBinary() )
            return fetch( rowIdx, columnIdx, 1 );
         else
            return fetch( rowIdx, columnIdx, values_view[ globalIdx ] );
      }
      return identity;
   };
   this->segments.reduceSegments( begin, end, fetch_, reduce, keep, identity );
}

template< typename Real,
          typename Device,
          typename Index,
          typename MatrixType,
          template< typename, typename >
          class SegmentsView,
          typename ComputeReal >
template< typename Fetch, typename Reduce, typename Keep, typename FetchReal >
void
SparseMatrixView< Real, Device, Index, MatrixType, SegmentsView, ComputeReal >::reduceAllRows( Fetch& fetch,
                                                                                               const Reduce& reduce,
                                                                                               Keep& keep,
                                                                                               const FetchReal& identity )
{
   this->reduceRows( (IndexType) 0, this->getRows(), fetch, reduce, keep, identity );
}

template< typename Real,
          typename Device,
          typename Index,
          typename MatrixType,
          template< typename, typename >
          class SegmentsView,
          typename ComputeReal >
template< typename Fetch, typename Reduce, typename Keep, typename FetchReal >
void
SparseMatrixView< Real, Device, Index, MatrixType, SegmentsView, ComputeReal >::reduceAllRows( Fetch& fetch,
                                                                                               const Reduce& reduce,
                                                                                               Keep& keep,
                                                                                               const FetchReal& identity ) const
{
   this->reduceRows( (IndexType) 0, this->getRows(), fetch, reduce, keep, identity );
}

template< typename Real,
          typename Device,
          typename Index,
          typename MatrixType,
          template< typename, typename >
          class SegmentsView,
          typename ComputeReal >
template< typename Function >
void
SparseMatrixView< Real, Device, Index, MatrixType, SegmentsView, ComputeReal >::forElements( IndexType begin,
                                                                                             IndexType end,
                                                                                             Function& function ) const
{
   const auto columns_view = this->columnIndexes.getConstView();
   const auto values_view = this->values.getConstView();
   // const IndexType paddingIndex_ = this->getPaddingIndex();
   auto columns = this->getColumns();
   auto f = [ = ] __cuda_callable__( IndexType rowIdx, IndexType localIdx, IndexType globalIdx ) mutable
   {
      if( localIdx < columns ) {
         if( isBinary() )
            function( rowIdx, localIdx, columns_view[ globalIdx ], (RealType) 1.0 );
         else
            function( rowIdx, localIdx, columns_view[ globalIdx ], values_view[ globalIdx ] );
      }
      // return true;
   };
   this->segments.forElements( begin, end, f );
}

template< typename Real,
          typename Device,
          typename Index,
          typename MatrixType,
          template< typename, typename >
          class SegmentsView,
          typename ComputeReal >
template< typename Function >
void
SparseMatrixView< Real, Device, Index, MatrixType, SegmentsView, ComputeReal >::forElements( IndexType begin,
                                                                                             IndexType end,
                                                                                             Function& function )
{
   auto columns_view = this->columnIndexes.getView();
   auto values_view = this->values.getView();
   const IndexType paddingIndex_ = this->getPaddingIndex();
   auto columns = this->getColumns();
   auto f = [ = ] __cuda_callable__( IndexType rowIdx, IndexType localIdx, IndexType globalIdx ) mutable
   {
      if( localIdx < columns ) {
         if( isBinary() ) {
            RealType one( columns_view[ globalIdx ] != paddingIndex_ );
            function( rowIdx, localIdx, columns_view[ globalIdx ], one );
         }
         else
            function( rowIdx, localIdx, columns_view[ globalIdx ], values_view[ globalIdx ] );
      }
   };
   this->segments.forElements( begin, end, f );
}

template< typename Real,
          typename Device,
          typename Index,
          typename MatrixType,
          template< typename, typename >
          class SegmentsView,
          typename ComputeReal >
template< typename Function >
void
SparseMatrixView< Real, Device, Index, MatrixType, SegmentsView, ComputeReal >::forAllElements( Function& function ) const
{
   this->forElements( (IndexType) 0, this->getRows(), function );
}

template< typename Real,
          typename Device,
          typename Index,
          typename MatrixType,
          template< typename, typename >
          class SegmentsView,
          typename ComputeReal >
template< typename Function >
void
SparseMatrixView< Real, Device, Index, MatrixType, SegmentsView, ComputeReal >::forAllElements( Function& function )
{
   this->forElements( (IndexType) 0, this->getRows(), function );
}

template< typename Real,
          typename Device,
          typename Index,
          typename MatrixType,
          template< typename, typename >
          class SegmentsView,
          typename ComputeReal >
template< typename Function >
void
SparseMatrixView< Real, Device, Index, MatrixType, SegmentsView, ComputeReal >::forRows( IndexType begin,
                                                                                         IndexType end,
                                                                                         Function&& function )
{
   auto columns_view = this->columnIndexes.getView();
   auto values_view = this->values.getView();
   using SegmentViewType = typename SegmentsViewType::SegmentViewType;
   auto f = [ = ] __cuda_callable__( SegmentViewType & segmentView ) mutable
   {
      auto rowView = RowView( segmentView, values_view, columns_view );
      function( rowView );
   };
   this->segments.forSegments( begin, end, f );
}

template< typename Real,
          typename Device,
          typename Index,
          typename MatrixType,
          template< typename, typename >
          class SegmentsView,
          typename ComputeReal >
template< typename Function >
void
SparseMatrixView< Real, Device, Index, MatrixType, SegmentsView, ComputeReal >::forRows( IndexType begin,
                                                                                         IndexType end,
                                                                                         Function&& function ) const
{
   const auto columns_view = this->columnIndexes.getConstView();
   const auto values_view = this->values.getConstView();
   using SegmentViewType = typename SegmentsViewType::SegmentViewType;
   auto f = [ = ] __cuda_callable__( const SegmentViewType& segmentView ) mutable
   {
      const auto rowView = ConstRowView( segmentView, values_view, columns_view );
      function( rowView );
   };
   this->segments.forSegments( begin, end, f );
}

template< typename Real,
          typename Device,
          typename Index,
          typename MatrixType,
          template< typename, typename >
          class SegmentsView,
          typename ComputeReal >
template< typename Function >
void
SparseMatrixView< Real, Device, Index, MatrixType, SegmentsView, ComputeReal >::forAllRows( Function&& function )
{
   this->forRows( (IndexType) 0, this->getRows(), function );
}

template< typename Real,
          typename Device,
          typename Index,
          typename MatrixType,
          template< typename, typename >
          class SegmentsView,
          typename ComputeReal >
template< typename Function >
void
SparseMatrixView< Real, Device, Index, MatrixType, SegmentsView, ComputeReal >::forAllRows( Function&& function ) const
{
   this->forRows( (IndexType) 0, this->getRows(), function );
}

template< typename Real,
          typename Device,
          typename Index,
          typename MatrixType,
          template< typename, typename >
          class SegmentsView,
          typename ComputeReal >
template< typename Function >
void
SparseMatrixView< Real, Device, Index, MatrixType, SegmentsView, ComputeReal >::sequentialForRows( IndexType begin,
                                                                                                   IndexType end,
                                                                                                   Function& function ) const
{
   for( IndexType row = begin; row < end; row++ )
      this->forRows( row, row + 1, function );
}

template< typename Real,
          typename Device,
          typename Index,
          typename MatrixType,
          template< typename, typename >
          class SegmentsView,
          typename ComputeReal >
template< typename Function >
void
SparseMatrixView< Real, Device, Index, MatrixType, SegmentsView, ComputeReal >::sequentialForRows( IndexType begin,
                                                                                                   IndexType end,
                                                                                                   Function& function )
{
   for( IndexType row = begin; row < end; row++ )
      this->forRows( row, row + 1, function );
}

template< typename Real,
          typename Device,
          typename Index,
          typename MatrixType,
          template< typename, typename >
          class SegmentsView,
          typename ComputeReal >
template< typename Function >
void
SparseMatrixView< Real, Device, Index, MatrixType, SegmentsView, ComputeReal >::sequentialForAllRows( Function& function ) const
{
   this->sequentialForRows( (IndexType) 0, this->getRows(), function );
}

template< typename Real,
          typename Device,
          typename Index,
          typename MatrixType,
          template< typename, typename >
          class SegmentsView,
          typename ComputeReal >
template< typename Function >
void
SparseMatrixView< Real, Device, Index, MatrixType, SegmentsView, ComputeReal >::sequentialForAllRows( Function& function )
{
   this->sequentialForRows( (IndexType) 0, this->getRows(), function );
}

/*template< typename Real,
          template< typename, typename > class SegmentsView,
          typename Device,
          typename Index,
          typename RealAllocator,
          typename IndexAllocator >
template< typename Real2, template< typename, typename > class Segments2, typename Index2, typename RealAllocator2, typename
IndexAllocator2 > void SparseMatrixView< Real, Device, Index, MatrixType, SegmentsView, ComputeReal >:: addMatrix( const
SparseMatrixView< Real2, Segments2, Device, Index2, RealAllocator2, IndexAllocator2 >& matrix, const RealType&
matrixMultiplicator, const RealType& thisMatrixMultiplicator )
{

}

template< typename Real,
          template< typename, typename > class SegmentsView,
          typename Device,
          typename Index,
          typename RealAllocator,
          typename IndexAllocator >
template< typename Real2, typename Index2 >
void
SparseMatrixView< Real, Device, Index, MatrixType, SegmentsView, ComputeReal >::
getTransposition( const SparseMatrixView< Real2, Device, Index2 >& matrix,
                  const RealType& matrixMultiplicator )
{

}*/

template< typename Real,
          typename Device,
          typename Index,
          typename MatrixType,
          template< typename, typename >
          class SegmentsView,
          typename ComputeReal >
SparseMatrixView< Real, Device, Index, MatrixType, SegmentsView, ComputeReal >&
SparseMatrixView< Real, Device, Index, MatrixType, SegmentsView, ComputeReal >::operator=(
   const SparseMatrixView< Real, Device, Index, MatrixType, SegmentsView, ComputeReal >& matrix )
{
   MatrixView< Real, Device, Index >::operator=( matrix );
   this->columnIndexes.bind( matrix.columnIndexes );
   this->segments = matrix.segments;
   return *this;
}

template< typename Real,
          typename Device,
          typename Index,
          typename MatrixType,
          template< typename, typename >
          class SegmentsView,
          typename ComputeReal >
template< typename Matrix >
bool
SparseMatrixView< Real, Device, Index, MatrixType, SegmentsView, ComputeReal >::operator==( const Matrix& m ) const
{
   const auto& view1 = *this;
   const auto view2 = m.getConstView();
   auto fetch = [ = ] __cuda_callable__( const IndexType i ) -> bool
   {
      return view1.getRow( i ) == view2.getRow( i );
   };
   return Algorithms::reduce< DeviceType >( (IndexType) 0, this->getRows(), fetch, std::logical_and<>{}, true );
}

template< typename Real,
          typename Device,
          typename Index,
          typename MatrixType,
          template< typename, typename >
          class SegmentsView,
          typename ComputeReal >
template< typename Matrix >
bool
SparseMatrixView< Real, Device, Index, MatrixType, SegmentsView, ComputeReal >::operator!=( const Matrix& m ) const
{
   return ! operator==( m );
}

template< typename Real,
          typename Device,
          typename Index,
          typename MatrixType,
          template< typename, typename >
          class SegmentsView,
          typename ComputeReal >
void
SparseMatrixView< Real, Device, Index, MatrixType, SegmentsView, ComputeReal >::save( File& file ) const
{
   MatrixView< Real, Device, Index >::save( file );
   file << this->columnIndexes;
   this->segments.save( file );
}

template< typename Real,
          typename Device,
          typename Index,
          typename MatrixType,
          template< typename, typename >
          class SegmentsView,
          typename ComputeReal >
void
SparseMatrixView< Real, Device, Index, MatrixType, SegmentsView, ComputeReal >::save( const String& fileName ) const
{
   Object::save( fileName );
}

template< typename Real,
          typename Device,
          typename Index,
          typename MatrixType,
          template< typename, typename >
          class SegmentsView,
          typename ComputeReal >
void
SparseMatrixView< Real, Device, Index, MatrixType, SegmentsView, ComputeReal >::print( std::ostream& str ) const
{
   if( isSymmetric() ) {
      for( IndexType row = 0; row < this->getRows(); row++ ) {
         str << "Row: " << row << " -> ";
         for( IndexType column = 0; column < this->getColumns(); column++ ) {
            auto value = this->getElement( row, column );
            if( value != (RealType) 0 )
               str << " Col:" << column << "->" << value << "\t";
         }
         str << std::endl;
      }
   }
   else
      for( IndexType row = 0; row < this->getRows(); row++ ) {
         str << "Row: " << row << " -> ";
         const auto rowLength = this->segments.getSegmentSize( row );
         for( IndexType i = 0; i < rowLength; i++ ) {
            const IndexType globalIdx = this->segments.getGlobalIndex( row, i );
            const IndexType column = this->columnIndexes.getElement( globalIdx );
            if( column == this->getPaddingIndex() )
               break;
            RealType value;
            if( isBinary() )
               value = (RealType) 1.0;
            else
               value = this->values.getElement( globalIdx );
            if( value ) {
               std::stringstream str_;
               str_ << std::setw( 4 ) << std::right << column << ":" << std::setw( 4 ) << std::left << value;
               str << std::setw( 10 ) << str_.str();
            }
         }
         str << std::endl;
      }
}

template< typename Real,
          typename Device,
          typename Index,
          typename MatrixType,
          template< typename, typename >
          class SegmentsView,
          typename ComputeReal >
__cuda_callable__
Index
SparseMatrixView< Real, Device, Index, MatrixType, SegmentsView, ComputeReal >::getPaddingIndex() const
{
   return -1;
}

template< typename Real,
          typename Device,
          typename Index,
          typename MatrixType,
          template< typename, typename >
          class SegmentsView,
          typename ComputeReal >
auto
SparseMatrixView< Real, Device, Index, MatrixType, SegmentsView, ComputeReal >::getSegments() const -> const SegmentsViewType&
{
   return this->segments;
}

template< typename Real,
          typename Device,
          typename Index,
          typename MatrixType,
          template< typename, typename >
          class SegmentsView,
          typename ComputeReal >
auto
SparseMatrixView< Real, Device, Index, MatrixType, SegmentsView, ComputeReal >::getSegments() -> SegmentsViewType&
{
   return this->segments;
}

template< typename Real,
          typename Device,
          typename Index,
          typename MatrixType,
          template< typename, typename >
          class SegmentsView,
          typename ComputeReal >
auto
SparseMatrixView< Real, Device, Index, MatrixType, SegmentsView, ComputeReal >::getColumnIndexes() const
   -> const ColumnsIndexesViewType&
{
   return this->columnIndexes;
}

template< typename Real,
          typename Device,
          typename Index,
          typename MatrixType,
          template< typename, typename >
          class SegmentsView,
          typename ComputeReal >
auto
SparseMatrixView< Real, Device, Index, MatrixType, SegmentsView, ComputeReal >::getColumnIndexes() -> ColumnsIndexesViewType&
{
   return this->columnIndexes;
}

}  // namespace Matrices
}  // namespace  TNL
