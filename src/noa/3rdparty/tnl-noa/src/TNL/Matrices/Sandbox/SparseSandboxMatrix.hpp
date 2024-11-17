// Copyright (c) 2004-2023 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <functional>
#include <sstream>
#include <noa/3rdparty/tnl-noa/src/TNL/Algorithms/reduce.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Matrices/Sandbox/SparseSandboxMatrix.h>

namespace noa::TNL {
namespace Matrices {
namespace Sandbox {

template< typename Real, typename Device, typename Index, typename MatrixType, typename RealAllocator, typename IndexAllocator >
SparseSandboxMatrix< Real, Device, Index, MatrixType, RealAllocator, IndexAllocator >::SparseSandboxMatrix(
   const RealAllocatorType& realAllocator,
   const IndexAllocatorType& indexAllocator )
: BaseType( realAllocator ), columnIndexes( indexAllocator ), rowPointers( (IndexType) 1, (IndexType) 0, indexAllocator )
{
   this->view = this->getView();
}

template< typename Real, typename Device, typename Index, typename MatrixType, typename RealAllocator, typename IndexAllocator >
template< typename Index_t, std::enable_if_t< std::is_integral< Index_t >::value, int > >
SparseSandboxMatrix< Real, Device, Index, MatrixType, RealAllocator, IndexAllocator >::SparseSandboxMatrix(
   const Index_t rows,
   const Index_t columns,
   const RealAllocatorType& realAllocator,
   const IndexAllocatorType& indexAllocator )
: BaseType( rows, columns, realAllocator ), columnIndexes( indexAllocator ),
  rowPointers( rows + 1, (IndexType) 0, indexAllocator )
{
   this->view = this->getView();
}

template< typename Real, typename Device, typename Index, typename MatrixType, typename RealAllocator, typename IndexAllocator >
template< typename ListIndex >
SparseSandboxMatrix< Real, Device, Index, MatrixType, RealAllocator, IndexAllocator >::SparseSandboxMatrix(
   const std::initializer_list< ListIndex >& rowCapacities,
   const IndexType columns,
   const RealAllocatorType& realAllocator,
   const IndexAllocatorType& indexAllocator )
: BaseType( rowCapacities.size(), columns, realAllocator ), columnIndexes( indexAllocator ),
  rowPointers( rowCapacities.size() + 1, (IndexType) 0, indexAllocator )
{
   this->setRowCapacities( RowsCapacitiesType( rowCapacities ) );
}

template< typename Real, typename Device, typename Index, typename MatrixType, typename RealAllocator, typename IndexAllocator >
template< typename RowCapacitiesVector, std::enable_if_t< TNL::IsArrayType< RowCapacitiesVector >::value, int > >
SparseSandboxMatrix< Real, Device, Index, MatrixType, RealAllocator, IndexAllocator >::SparseSandboxMatrix(
   const RowCapacitiesVector& rowCapacities,
   const IndexType columns,
   const RealAllocatorType& realAllocator,
   const IndexAllocatorType& indexAllocator )
: BaseType( rowCapacities.getSize(), columns, realAllocator ), columnIndexes( indexAllocator ),
  rowPointers( rowCapacities.getSize() + 1, (IndexType) 0, indexAllocator )
{
   this->setRowCapacities( rowCapacities );
}

template< typename Real, typename Device, typename Index, typename MatrixType, typename RealAllocator, typename IndexAllocator >
SparseSandboxMatrix< Real, Device, Index, MatrixType, RealAllocator, IndexAllocator >::SparseSandboxMatrix(
   const IndexType rows,
   const IndexType columns,
   const std::initializer_list< std::tuple< IndexType, IndexType, RealType > >& data,
   const RealAllocatorType& realAllocator,
   const IndexAllocatorType& indexAllocator )
: BaseType( rows, columns, realAllocator ), columnIndexes( indexAllocator ),
  rowPointers( rows + 1, (IndexType) 0, indexAllocator )
{
   this->setElements( data );
   this->view = this->getView();
}

template< typename Real, typename Device, typename Index, typename MatrixType, typename RealAllocator, typename IndexAllocator >
template< typename MapIndex, typename MapValue >
SparseSandboxMatrix< Real, Device, Index, MatrixType, RealAllocator, IndexAllocator >::SparseSandboxMatrix(
   const IndexType rows,
   const IndexType columns,
   const std::map< std::pair< MapIndex, MapIndex >, MapValue >& map,
   const RealAllocatorType& realAllocator,
   const IndexAllocatorType& indexAllocator )
: BaseType( rows, columns, realAllocator ), columnIndexes( indexAllocator ),
  rowPointers( rows + 1, (IndexType) 0, indexAllocator )
{
   this->setDimensions( rows, columns );
   this->setElements( map );
   this->view = this->getView();
}

template< typename Real, typename Device, typename Index, typename MatrixType, typename RealAllocator, typename IndexAllocator >
auto
SparseSandboxMatrix< Real, Device, Index, MatrixType, RealAllocator, IndexAllocator >::getView() -> ViewType
{
   return { this->getRows(), this->getColumns(), this->getValues().getView(), columnIndexes.getView(), rowPointers.getView() };
}

template< typename Real, typename Device, typename Index, typename MatrixType, typename RealAllocator, typename IndexAllocator >
auto
SparseSandboxMatrix< Real, Device, Index, MatrixType, RealAllocator, IndexAllocator >::getConstView() const -> ConstViewType
{
   return { this->getRows(),
            this->getColumns(),
            this->getValues().getConstView(),
            columnIndexes.getConstView(),
            rowPointers.getConstView() };
}

template< typename Real, typename Device, typename Index, typename MatrixType, typename RealAllocator, typename IndexAllocator >
std::string
SparseSandboxMatrix< Real, Device, Index, MatrixType, RealAllocator, IndexAllocator >::getSerializationType()
{
   return ViewType::getSerializationType();
}

template< typename Real, typename Device, typename Index, typename MatrixType, typename RealAllocator, typename IndexAllocator >
std::string
SparseSandboxMatrix< Real, Device, Index, MatrixType, RealAllocator, IndexAllocator >::getSerializationTypeVirtual() const
{
   return this->getSerializationType();
}

template< typename Real, typename Device, typename Index, typename MatrixType, typename RealAllocator, typename IndexAllocator >
void
SparseSandboxMatrix< Real, Device, Index, MatrixType, RealAllocator, IndexAllocator >::setDimensions( const IndexType rows,
                                                                                                      const IndexType columns )
{
   BaseType::setDimensions( rows, columns );
   this->view = this->getView();
}

template< typename Real, typename Device, typename Index, typename MatrixType, typename RealAllocator, typename IndexAllocator >
template< typename Matrix_ >
void
SparseSandboxMatrix< Real, Device, Index, MatrixType, RealAllocator, IndexAllocator >::setLike( const Matrix_& matrix )
{
   BaseType::setLike( matrix );
   // SANDBOX_TODO: Replace the following line with assignment of metadata required by your format.
   //               Do not assign matrix elements here.
   this->rowPointers = matrix.rowPointers;
   this->view = this->getView();
}

template< typename Real, typename Device, typename Index, typename MatrixType, typename RealAllocator, typename IndexAllocator >
template< typename RowsCapacitiesVector >
void
SparseSandboxMatrix< Real, Device, Index, MatrixType, RealAllocator, IndexAllocator >::setRowCapacities(
   const RowsCapacitiesVector& rowsCapacities )
{
   TNL_ASSERT_EQ(
      rowsCapacities.getSize(), this->getRows(), "Number of matrix rows does not fit with rowCapacities vector size." );
   using RowsCapacitiesVectorDevice = typename RowsCapacitiesVector::DeviceType;

   // SANDBOX_TODO: Replace the following lines with the setup of your sparse matrix format based on
   //               `rowsCapacities`. This container has the same number of elements as is the number of
   //               rows of this matrix. Each element says how many nonzero elements the user needs to have
   //               in each row. This number can be increased if the sparse matrix format uses padding zeros.
   this->rowPointers.setSize( this->getRows() + 1 );
   if( std::is_same< DeviceType, RowsCapacitiesVectorDevice >::value ) {
      // GOTCHA: when this->getRows() == 0, getView returns a full view with size == 1
      if( this->getRows() > 0 ) {
         auto view = this->rowPointers.getView( 0, this->getRows() );
         view = rowsCapacities;
      }
   }
   else {
      RowsCapacitiesType thisRowsCapacities;
      thisRowsCapacities = rowsCapacities;
      if( this->getRows() > 0 ) {
         auto view = this->rowPointers.getView( 0, this->getRows() );
         view = thisRowsCapacities;
      }
   }
   this->rowPointers.setElement( this->getRows(), 0 );
   Algorithms::inplaceExclusiveScan( this->rowPointers );
   // this->rowPointers.template scan< Algorithms::ScanType::Exclusive >();
   //  End of sparse matrix format initiation.

   // SANDBOX_TODO: Compute number of all elements that need to be allocated by your format.
   const auto storageSize = rowPointers.getElement( this->getRows() );

   // The rest of this methods needs no changes.
   if( ! isBinary() ) {
      this->values.setSize( storageSize );
      this->values = (RealType) 0;
   }
   this->columnIndexes.setSize( storageSize );
   this->columnIndexes = this->getPaddingIndex();
   this->view = this->getView();
}

template< typename Real, typename Device, typename Index, typename MatrixType, typename RealAllocator, typename IndexAllocator >
template< typename Vector >
void
SparseSandboxMatrix< Real, Device, Index, MatrixType, RealAllocator, IndexAllocator >::getRowCapacities(
   Vector& rowCapacities ) const
{
   this->view.getRowCapacities( rowCapacities );
}

template< typename Real, typename Device, typename Index, typename MatrixType, typename RealAllocator, typename IndexAllocator >
void
SparseSandboxMatrix< Real, Device, Index, MatrixType, RealAllocator, IndexAllocator >::setElements(
   const std::initializer_list< std::tuple< IndexType, IndexType, RealType > >& data )
{
   const auto& rows = this->getRows();
   const auto& columns = this->getColumns();
   Containers::Vector< IndexType, Devices::Host, IndexType > rowCapacities( rows, 0 );
   for( const auto& i : data ) {
      if( std::get< 0 >( i ) >= rows ) {
         std::stringstream s;
         s << "Wrong row index " << std::get< 0 >( i ) << " in an initializer list";
         throw std::logic_error( s.str() );
      }
      rowCapacities[ std::get< 0 >( i ) ]++;
   }
   SparseSandboxMatrix< Real, Devices::Host, Index, MatrixType > hostMatrix( rows, columns );
   hostMatrix.setRowCapacities( rowCapacities );
   for( const auto& i : data ) {
      if( std::get< 1 >( i ) >= columns ) {
         std::stringstream s;
         s << "Wrong column index " << std::get< 1 >( i ) << " in an initializer list";
         throw std::logic_error( s.str() );
      }
      hostMatrix.setElement( std::get< 0 >( i ), std::get< 1 >( i ), std::get< 2 >( i ) );
   }
   ( *this ) = hostMatrix;
}

template< typename Real, typename Device, typename Index, typename MatrixType, typename RealAllocator, typename IndexAllocator >
template< typename MapIndex, typename MapValue >
void
SparseSandboxMatrix< Real, Device, Index, MatrixType, RealAllocator, IndexAllocator >::setElements(
   const std::map< std::pair< MapIndex, MapIndex >, MapValue >& map )
{
   Containers::Vector< IndexType, Devices::Host, IndexType > rowsCapacities( this->getRows(), 0 );
   for( auto element : map )
      rowsCapacities[ element.first.first ]++;
   if( ! std::is_same< DeviceType, Devices::Host >::value ) {
      SparseSandboxMatrix< Real, Devices::Host, Index, MatrixType > hostMatrix( this->getRows(), this->getColumns() );
      hostMatrix.setRowCapacities( rowsCapacities );
      for( auto element : map )
         hostMatrix.setElement( element.first.first, element.first.second, element.second );
      *this = hostMatrix;
   }
   else {
      this->setRowCapacities( rowsCapacities );
      for( auto element : map )
         this->setElement( element.first.first, element.first.second, element.second );
   }
}

template< typename Real, typename Device, typename Index, typename MatrixType, typename RealAllocator, typename IndexAllocator >
template< typename Vector >
void
SparseSandboxMatrix< Real, Device, Index, MatrixType, RealAllocator, IndexAllocator >::getCompressedRowLengths(
   Vector& rowLengths ) const
{
   this->view.getCompressedRowLengths( rowLengths );
}

template< typename Real, typename Device, typename Index, typename MatrixType, typename RealAllocator, typename IndexAllocator >
__cuda_callable__
Index
SparseSandboxMatrix< Real, Device, Index, MatrixType, RealAllocator, IndexAllocator >::getRowCapacity(
   const IndexType row ) const
{
   return this->view.getRowCapacity( row );
}

template< typename Real, typename Device, typename Index, typename MatrixType, typename RealAllocator, typename IndexAllocator >
Index
SparseSandboxMatrix< Real, Device, Index, MatrixType, RealAllocator, IndexAllocator >::getNonzeroElementsCount() const
{
   return this->view.getNonzeroElementsCount();
}

template< typename Real, typename Device, typename Index, typename MatrixType, typename RealAllocator, typename IndexAllocator >
void
SparseSandboxMatrix< Real, Device, Index, MatrixType, RealAllocator, IndexAllocator >::reset()
{
   BaseType::reset();
   this->columnIndexes.reset();
   // SANDBOX_TODO: Reset the metadata required by your format here.
   this->rowPointers.reset();
   this->view = this->getView();
}

template< typename Real, typename Device, typename Index, typename MatrixType, typename RealAllocator, typename IndexAllocator >
__cuda_callable__
auto
SparseSandboxMatrix< Real, Device, Index, MatrixType, RealAllocator, IndexAllocator >::getRow( const IndexType& rowIdx ) const
   -> ConstRowView
{
   return this->view.getRow( rowIdx );
}

template< typename Real, typename Device, typename Index, typename MatrixType, typename RealAllocator, typename IndexAllocator >
__cuda_callable__
auto
SparseSandboxMatrix< Real, Device, Index, MatrixType, RealAllocator, IndexAllocator >::getRow( const IndexType& rowIdx )
   -> RowView
{
   return this->view.getRow( rowIdx );
}

template< typename Real, typename Device, typename Index, typename MatrixType, typename RealAllocator, typename IndexAllocator >
__cuda_callable__
void
SparseSandboxMatrix< Real, Device, Index, MatrixType, RealAllocator, IndexAllocator >::setElement( const IndexType row,
                                                                                                   const IndexType column,
                                                                                                   const RealType& value )
{
   this->view.setElement( row, column, value );
}

template< typename Real, typename Device, typename Index, typename MatrixType, typename RealAllocator, typename IndexAllocator >
__cuda_callable__
void
SparseSandboxMatrix< Real, Device, Index, MatrixType, RealAllocator, IndexAllocator >::addElement(
   const IndexType row,
   const IndexType column,
   const RealType& value,
   const RealType& thisElementMultiplicator )
{
   this->view.addElement( row, column, value, thisElementMultiplicator );
}

template< typename Real, typename Device, typename Index, typename MatrixType, typename RealAllocator, typename IndexAllocator >
__cuda_callable__
auto
SparseSandboxMatrix< Real, Device, Index, MatrixType, RealAllocator, IndexAllocator >::getElement(
   const IndexType row,
   const IndexType column ) const -> RealType
{
   return this->view.getElement( row, column );
}

template< typename Real, typename Device, typename Index, typename MatrixType, typename RealAllocator, typename IndexAllocator >
template< typename InVector, typename OutVector >
void
SparseSandboxMatrix< Real, Device, Index, MatrixType, RealAllocator, IndexAllocator >::vectorProduct(
   const InVector& inVector,
   OutVector& outVector,
   RealType matrixMultiplicator,
   RealType outVectorMultiplicator,
   IndexType firstRow,
   IndexType lastRow ) const
{
   this->view.vectorProduct( inVector, outVector, matrixMultiplicator, outVectorMultiplicator, firstRow, lastRow );
}

template< typename Real, typename Device, typename Index, typename MatrixType, typename RealAllocator, typename IndexAllocator >
template< typename Fetch, typename Reduce, typename Keep, typename FetchValue >
void
SparseSandboxMatrix< Real, Device, Index, MatrixType, RealAllocator, IndexAllocator >::reduceRows( IndexType begin,
                                                                                                   IndexType end,
                                                                                                   Fetch& fetch,
                                                                                                   const Reduce& reduce,
                                                                                                   Keep& keep,
                                                                                                   const FetchValue& zero )
{
   this->view.reduceRows( begin, end, fetch, reduce, keep, zero );
}

template< typename Real, typename Device, typename Index, typename MatrixType, typename RealAllocator, typename IndexAllocator >
template< typename Fetch, typename Reduce, typename Keep, typename FetchValue >
void
SparseSandboxMatrix< Real, Device, Index, MatrixType, RealAllocator, IndexAllocator >::reduceRows(
   IndexType begin,
   IndexType end,
   Fetch& fetch,
   const Reduce& reduce,
   Keep& keep,
   const FetchValue& zero ) const
{
   this->view.reduceRows( begin, end, fetch, reduce, keep, zero );
}

template< typename Real, typename Device, typename Index, typename MatrixType, typename RealAllocator, typename IndexAllocator >
template< typename Fetch, typename Reduce, typename Keep, typename FetchReal >
void
SparseSandboxMatrix< Real, Device, Index, MatrixType, RealAllocator, IndexAllocator >::reduceAllRows( Fetch&& fetch,
                                                                                                      const Reduce&& reduce,
                                                                                                      Keep&& keep,
                                                                                                      const FetchReal& zero )
{
   this->reduceRows( 0, this->getRows(), fetch, reduce, keep, zero );
}

template< typename Real, typename Device, typename Index, typename MatrixType, typename RealAllocator, typename IndexAllocator >
template< typename Fetch, typename Reduce, typename Keep, typename FetchReal >
void
SparseSandboxMatrix< Real, Device, Index, MatrixType, RealAllocator, IndexAllocator >::reduceAllRows(
   Fetch& fetch,
   const Reduce& reduce,
   Keep& keep,
   const FetchReal& zero ) const
{
   this->reduceRows( 0, this->getRows(), fetch, reduce, keep, zero );
}

template< typename Real, typename Device, typename Index, typename MatrixType, typename RealAllocator, typename IndexAllocator >
template< typename Function >
void
SparseSandboxMatrix< Real, Device, Index, MatrixType, RealAllocator, IndexAllocator >::forElements( IndexType begin,
                                                                                                    IndexType end,
                                                                                                    Function&& function ) const
{
   this->view.forElements( begin, end, function );
}

template< typename Real, typename Device, typename Index, typename MatrixType, typename RealAllocator, typename IndexAllocator >
template< typename Function >
void
SparseSandboxMatrix< Real, Device, Index, MatrixType, RealAllocator, IndexAllocator >::forElements( IndexType begin,
                                                                                                    IndexType end,
                                                                                                    Function&& function )
{
   this->view.forElements( begin, end, function );
}

template< typename Real, typename Device, typename Index, typename MatrixType, typename RealAllocator, typename IndexAllocator >
template< typename Function >
void
SparseSandboxMatrix< Real, Device, Index, MatrixType, RealAllocator, IndexAllocator >::forAllElements(
   Function&& function ) const
{
   this->forElements( 0, this->getRows(), function );
}

template< typename Real, typename Device, typename Index, typename MatrixType, typename RealAllocator, typename IndexAllocator >
template< typename Function >
void
SparseSandboxMatrix< Real, Device, Index, MatrixType, RealAllocator, IndexAllocator >::forAllElements( Function&& function )
{
   this->forElements( 0, this->getRows(), function );
}

template< typename Real, typename Device, typename Index, typename MatrixType, typename RealAllocator, typename IndexAllocator >
template< typename Function >
void
SparseSandboxMatrix< Real, Device, Index, MatrixType, RealAllocator, IndexAllocator >::forRows( IndexType begin,
                                                                                                IndexType end,
                                                                                                Function&& function )
{
   this->getView().forRows( begin, end, function );
}

template< typename Real, typename Device, typename Index, typename MatrixType, typename RealAllocator, typename IndexAllocator >
template< typename Function >
void
SparseSandboxMatrix< Real, Device, Index, MatrixType, RealAllocator, IndexAllocator >::forRows( IndexType begin,
                                                                                                IndexType end,
                                                                                                Function&& function ) const
{
   this->getConstView().forRows( begin, end, function );
}

template< typename Real, typename Device, typename Index, typename MatrixType, typename RealAllocator, typename IndexAllocator >
template< typename Function >
void
SparseSandboxMatrix< Real, Device, Index, MatrixType, RealAllocator, IndexAllocator >::forAllRows( Function&& function )
{
   this->getView().forAllRows( function );
}

template< typename Real, typename Device, typename Index, typename MatrixType, typename RealAllocator, typename IndexAllocator >
template< typename Function >
void
SparseSandboxMatrix< Real, Device, Index, MatrixType, RealAllocator, IndexAllocator >::forAllRows( Function&& function ) const
{
   this->getConstView().forAllRows( function );
}

template< typename Real, typename Device, typename Index, typename MatrixType, typename RealAllocator, typename IndexAllocator >
template< typename Function >
void
SparseSandboxMatrix< Real, Device, Index, MatrixType, RealAllocator, IndexAllocator >::sequentialForRows(
   IndexType begin,
   IndexType end,
   Function& function ) const
{
   this->view.sequentialForRows( begin, end, function );
}

template< typename Real, typename Device, typename Index, typename MatrixType, typename RealAllocator, typename IndexAllocator >
template< typename Function >
void
SparseSandboxMatrix< Real, Device, Index, MatrixType, RealAllocator, IndexAllocator >::sequentialForRows( IndexType first,
                                                                                                          IndexType last,
                                                                                                          Function& function )
{
   this->view.sequentialForRows( first, last, function );
}

template< typename Real, typename Device, typename Index, typename MatrixType, typename RealAllocator, typename IndexAllocator >
template< typename Function >
void
SparseSandboxMatrix< Real, Device, Index, MatrixType, RealAllocator, IndexAllocator >::sequentialForAllRows(
   Function& function ) const
{
   this->sequentialForRows( 0, this->getRows(), function );
}

template< typename Real, typename Device, typename Index, typename MatrixType, typename RealAllocator, typename IndexAllocator >
template< typename Function >
void
SparseSandboxMatrix< Real, Device, Index, MatrixType, RealAllocator, IndexAllocator >::sequentialForAllRows(
   Function& function )
{
   this->sequentialForRows( 0, this->getRows(), function );
}

/*
template< typename Real,
          template< typename, typename, typename > class Segments,
          typename Device,
          typename Index,
          typename RealAllocator,
          typename IndexAllocator >
template< typename Real2, template< typename, typename > class Segments2, typename Index2, typename RealAllocator2, typename
IndexAllocator2 > void SparseSandboxMatrix< Real, Device, Index, MatrixType, RealAllocator, IndexAllocator >:: addMatrix( const
SparseSandboxMatrix< Real2, Segments2, Device, Index2, RealAllocator2, IndexAllocator2 >& matrix, const RealType&
matrixMultiplicator, const RealType& thisMatrixMultiplicator )
{
}
*/

template< typename Real, typename Device, typename Index, typename MatrixType, typename RealAllocator, typename IndexAllocator >
template< typename Real2, typename Index2 >
void
SparseSandboxMatrix< Real, Device, Index, MatrixType, RealAllocator, IndexAllocator >::getTransposition(
   const SparseSandboxMatrix< Real2, Device, Index2, MatrixType >& matrix,
   const RealType& matrixMultiplicator )
{
   // set transposed dimensions
   setDimensions( matrix.getColumns(), matrix.getRows() );

   // stage 1: compute row capacities for the transposition
   RowsCapacitiesType capacities;
   capacities.resize( this->getRows(), 0 );
   auto capacities_view = capacities.getView();
   using MatrixRowView = typename SparseSandboxMatrix< Real2, Device, Index2, MatrixType >::ConstRowView;
   matrix.forAllRows(
      [ = ] __cuda_callable__( const MatrixRowView& row ) mutable
      {
         for( IndexType c = 0; c < row.getSize(); c++ ) {
            // row index of the transpose = column index of the input
            const IndexType& transRowIdx = row.getColumnIndex( c );
            if( transRowIdx < 0 )
               continue;
            // increment the capacity for the row in the transpose
            Algorithms::AtomicOperations< DeviceType >::add( capacities_view[ row.getColumnIndex( c ) ], IndexType( 1 ) );
         }
      } );

   // set the row capacities
   setRowCapacities( capacities );
   capacities.reset();

   // index of the first unwritten element per row
   RowsCapacitiesType offsets;
   offsets.resize( this->getRows(), 0 );
   auto offsets_view = offsets.getView();

   // stage 2: copy and transpose the data
   auto trans_view = getView();
   matrix.forAllRows(
      [ = ] __cuda_callable__( const MatrixRowView& row ) mutable
      {
         // row index of the input = column index of the transpose
         const IndexType& rowIdx = row.getRowIndex();
         for( IndexType c = 0; c < row.getSize(); c++ ) {
            // row index of the transpose = column index of the input
            const IndexType& transRowIdx = row.getColumnIndex( c );
            if( transRowIdx < 0 )
               continue;
            // local index in the row of the transpose
            const IndexType transLocalIdx =
               Algorithms::AtomicOperations< DeviceType >::add( offsets_view[ transRowIdx ], IndexType( 1 ) );
            // get the row in the transposed matrix and set the value
            auto transRow = trans_view.getRow( transRowIdx );
            transRow.setElement( transLocalIdx, rowIdx, row.getValue( c ) * matrixMultiplicator );
         }
      } );
}

// copy assignment
template< typename Real, typename Device, typename Index, typename MatrixType, typename RealAllocator, typename IndexAllocator >
SparseSandboxMatrix< Real, Device, Index, MatrixType, RealAllocator, IndexAllocator >&
SparseSandboxMatrix< Real, Device, Index, MatrixType, RealAllocator, IndexAllocator >::operator=(
   const SparseSandboxMatrix& matrix )
{
   Matrix< Real, Device, Index >::operator=( matrix );
   this->columnIndexes = matrix.columnIndexes;
   // SANDBOX_TODO: Replace the following line with an assignment of metadata required by you sparse matrix format.
   this->rowPointers = matrix.rowPointers;
   this->view = this->getView();
   return *this;
}

template< typename Real, typename Device, typename Index, typename MatrixType, typename RealAllocator, typename IndexAllocator >
template< typename Device_ >
SparseSandboxMatrix< Real, Device, Index, MatrixType, RealAllocator, IndexAllocator >&
SparseSandboxMatrix< Real, Device, Index, MatrixType, RealAllocator, IndexAllocator >::operator=(
   const SparseSandboxMatrix< RealType, Device_, IndexType, MatrixType, RealAllocator, IndexAllocator >& matrix )
{
   Matrix< Real, Device, Index >::operator=( matrix );
   this->columnIndexes = matrix.columnIndexes;
   // SANDBOX_TODO: Replace the following line with an assignment of metadata required by you sparse matrix format.
   this->rowPointers = matrix.rowPointers;
   this->view = this->getView();
   return *this;
}

template< typename Real, typename Device, typename Index, typename MatrixType, typename RealAllocator, typename IndexAllocator >
template< typename Real_, typename Device_, typename Index_, ElementsOrganization Organization, typename RealAllocator_ >
SparseSandboxMatrix< Real, Device, Index, MatrixType, RealAllocator, IndexAllocator >&
SparseSandboxMatrix< Real, Device, Index, MatrixType, RealAllocator, IndexAllocator >::operator=(
   const DenseMatrix< Real_, Device_, Index_, Organization, RealAllocator_ >& matrix )
{
   using RHSMatrix = DenseMatrix< Real_, Device_, Index_, Organization, RealAllocator_ >;
   using RHSIndexType = typename RHSMatrix::IndexType;
   using RHSRealType = typename RHSMatrix::RealType;
   using RHSDeviceType = typename RHSMatrix::DeviceType;
   using RHSRealAllocatorType = typename RHSMatrix::RealAllocatorType;

   Containers::Vector< RHSIndexType, RHSDeviceType, RHSIndexType > rowLengths;
   matrix.getCompressedRowLengths( rowLengths );
   this->setLike( matrix );
   this->setRowCapacities( rowLengths );
   Containers::Vector< IndexType, DeviceType, IndexType > rowLocalIndexes( matrix.getRows() );
   rowLocalIndexes = 0;

   const IndexType paddingIndex = this->getPaddingIndex();
   auto columns_view = this->columnIndexes.getView();
   auto values_view = this->values.getView();
   auto rowLocalIndexes_view = rowLocalIndexes.getView();
   columns_view = paddingIndex;

   if( std::is_same< DeviceType, RHSDeviceType >::value ) {
      const auto segments_view = this->segments.getView();
      auto f = [ = ] __cuda_callable__(
                  RHSIndexType rowIdx, RHSIndexType localIdx, RHSIndexType columnIdx, const RHSRealType& value ) mutable
      {
         if( value != 0.0 ) {
            IndexType thisGlobalIdx = segments_view.getGlobalIndex( rowIdx, rowLocalIndexes_view[ rowIdx ]++ );
            columns_view[ thisGlobalIdx ] = columnIdx;
            if( ! isBinary() )
               values_view[ thisGlobalIdx ] = value;
         }
      };
      matrix.forAllElements( f );
   }
   else {
      const IndexType maxRowLength = matrix.getColumns();
      const IndexType bufferRowsCount( 128 );
      const size_t bufferSize = bufferRowsCount * maxRowLength;
      Containers::Vector< RHSRealType, RHSDeviceType, RHSIndexType, RHSRealAllocatorType > matrixValuesBuffer( bufferSize );
      Containers::Vector< RealType, DeviceType, IndexType, RealAllocatorType > thisValuesBuffer( bufferSize );
      Containers::Vector< IndexType, DeviceType, IndexType, IndexAllocatorType > thisColumnsBuffer( bufferSize );
      auto matrixValuesBuffer_view = matrixValuesBuffer.getView();
      auto thisValuesBuffer_view = thisValuesBuffer.getView();

      IndexType baseRow( 0 );
      const IndexType rowsCount = this->getRows();
      while( baseRow < rowsCount ) {
         const IndexType lastRow = min( baseRow + bufferRowsCount, rowsCount );
         thisColumnsBuffer = paddingIndex;

         ////
         // Copy matrix elements into buffer
         auto f1 = [ = ] __cuda_callable__(
                      RHSIndexType rowIdx, RHSIndexType localIdx, RHSIndexType columnIndex, const RHSRealType& value ) mutable
         {
            const IndexType bufferIdx = ( rowIdx - baseRow ) * maxRowLength + localIdx;
            matrixValuesBuffer_view[ bufferIdx ] = value;
         };
         matrix.forElements( baseRow, lastRow, f1 );

         ////
         // Copy the source matrix buffer to this matrix buffer
         thisValuesBuffer_view = matrixValuesBuffer_view;

         ////
         // Copy matrix elements from the buffer to the matrix and ignoring
         // zero matrix elements.
         const IndexType matrix_columns = this->getColumns();
         auto f2 =
            [ = ] __cuda_callable__( IndexType rowIdx, IndexType localIdx, IndexType & columnIndex, RealType & value ) mutable
         {
            RealType inValue( 0.0 );
            IndexType bufferIdx, column( rowLocalIndexes_view[ rowIdx ] );
            while( inValue == 0.0 && column < matrix_columns ) {
               bufferIdx = ( rowIdx - baseRow ) * maxRowLength + column++;
               inValue = thisValuesBuffer_view[ bufferIdx ];
            }
            rowLocalIndexes_view[ rowIdx ] = column;
            if( inValue == 0.0 ) {
               columnIndex = paddingIndex;
               value = 0.0;
            }
            else {
               columnIndex = column - 1;
               value = inValue;
            }
         };
         this->forElements( baseRow, lastRow, f2 );
         baseRow += bufferRowsCount;
      }
      // std::cerr << "This matrix = " << std::endl << *this << std::endl;
   }
   this->view = this->getView();
   return *this;
}

template< typename Real, typename Device, typename Index, typename MatrixType, typename RealAllocator, typename IndexAllocator >
template< typename RHSMatrix >
SparseSandboxMatrix< Real, Device, Index, MatrixType, RealAllocator, IndexAllocator >&
SparseSandboxMatrix< Real, Device, Index, MatrixType, RealAllocator, IndexAllocator >::operator=( const RHSMatrix& matrix )
{
   using RHSIndexType = typename RHSMatrix::IndexType;
   using RHSRealType = typename RHSMatrix::RealType;
   using RHSDeviceType = typename RHSMatrix::DeviceType;
   using RHSRealAllocatorType = typename RHSMatrix::RealAllocatorType;

   Containers::Vector< RHSIndexType, RHSDeviceType, RHSIndexType > rowCapacities;
   matrix.getRowCapacities( rowCapacities );
   this->setDimensions( matrix.getRows(), matrix.getColumns() );
   this->setRowCapacities( rowCapacities );
   Containers::Vector< IndexType, DeviceType, IndexType > rowLocalIndexes( matrix.getRows() );
   rowLocalIndexes = 0;

   const IndexType paddingIndex = this->getPaddingIndex();
   auto columns_view = this->columnIndexes.getView();
   auto values_view = this->values.getView();
   auto rowLocalIndexes_view = rowLocalIndexes.getView();
   columns_view = paddingIndex;

   // SANDBOX_TODO: Modify the follwoing accoring to your format
   auto row_pointers_view = this->rowPointers.getView();
   if( std::is_same< DeviceType, RHSDeviceType >::value ) {
      auto f = [ = ] __cuda_callable__(
                  RHSIndexType rowIdx, RHSIndexType localIdx_, RHSIndexType columnIndex, const RHSRealType& value ) mutable
      {
         IndexType localIdx( rowLocalIndexes_view[ rowIdx ] );
         IndexType thisRowBegin = row_pointers_view[ rowIdx ];
         if( value != 0.0 && columnIndex != paddingIndex ) {
            IndexType thisGlobalIdx = thisRowBegin + localIdx++;
            columns_view[ thisGlobalIdx ] = columnIndex;
            if( ! isBinary() )
               values_view[ thisGlobalIdx ] = value;
            rowLocalIndexes_view[ rowIdx ] = localIdx;
         }
      };
      matrix.forAllElements( f );
   }
   else {
      const IndexType maxRowLength = max( rowCapacities );
      const IndexType bufferRowsCount( 128 );
      const size_t bufferSize = bufferRowsCount * maxRowLength;
      Containers::Vector< RHSRealType, RHSDeviceType, RHSIndexType, RHSRealAllocatorType > matrixValuesBuffer( bufferSize );
      Containers::Vector< RHSIndexType, RHSDeviceType, RHSIndexType > matrixColumnsBuffer( bufferSize );
      Containers::Vector< RealType, DeviceType, IndexType, RealAllocatorType > thisValuesBuffer( bufferSize );
      Containers::Vector< IndexType, DeviceType, IndexType > thisColumnsBuffer( bufferSize );
      Containers::Vector< IndexType, DeviceType, IndexType > thisRowLengths;
      Containers::Vector< RHSIndexType, RHSDeviceType, RHSIndexType > rhsRowLengths;
      matrix.getCompressedRowLengths( rhsRowLengths );
      thisRowLengths = rhsRowLengths;
      auto matrixValuesBuffer_view = matrixValuesBuffer.getView();
      auto matrixColumnsBuffer_view = matrixColumnsBuffer.getView();
      auto thisValuesBuffer_view = thisValuesBuffer.getView();
      auto thisColumnsBuffer_view = thisColumnsBuffer.getView();
      matrixValuesBuffer_view = 0.0;

      IndexType baseRow( 0 );
      const IndexType rowsCount = this->getRows();
      while( baseRow < rowsCount ) {
         const IndexType lastRow = min( baseRow + bufferRowsCount, rowsCount );
         thisColumnsBuffer = paddingIndex;
         matrixColumnsBuffer_view = paddingIndex;

         ////
         // Copy matrix elements into buffer
         auto f1 = [ = ] __cuda_callable__(
                      RHSIndexType rowIdx, RHSIndexType localIdx, RHSIndexType columnIndex, const RHSRealType& value ) mutable
         {
            if( columnIndex != paddingIndex ) {
               TNL_ASSERT_LT( rowIdx - baseRow, bufferRowsCount, "" );
               TNL_ASSERT_LT( localIdx, maxRowLength, "" );
               const IndexType bufferIdx = ( rowIdx - baseRow ) * maxRowLength + localIdx;
               TNL_ASSERT_LT( bufferIdx, (IndexType) bufferSize, "" );
               matrixColumnsBuffer_view[ bufferIdx ] = columnIndex;
               matrixValuesBuffer_view[ bufferIdx ] = value;
            }
         };
         matrix.forElements( baseRow, lastRow, f1 );

         ////
         // Copy the source matrix buffer to this matrix buffer
         thisValuesBuffer_view = matrixValuesBuffer_view;
         thisColumnsBuffer_view = matrixColumnsBuffer_view;

         ////
         // Copy matrix elements from the buffer to the matrix and ignoring
         // zero matrix elements
         // const IndexType matrix_columns = this->getColumns();
         const auto thisRowLengths_view = thisRowLengths.getConstView();
         auto f2 =
            [ = ] __cuda_callable__( IndexType rowIdx, IndexType localIdx, IndexType & columnIndex, RealType & value ) mutable
         {
            RealType inValue( 0.0 );
            size_t bufferIdx;
            IndexType bufferLocalIdx( rowLocalIndexes_view[ rowIdx ] );
            while( inValue == 0.0 && localIdx < thisRowLengths_view[ rowIdx ] ) {
               bufferIdx = ( rowIdx - baseRow ) * maxRowLength + bufferLocalIdx++;
               TNL_ASSERT_LT( bufferIdx, bufferSize, "" );
               inValue = thisValuesBuffer_view[ bufferIdx ];
            }
            rowLocalIndexes_view[ rowIdx ] = bufferLocalIdx;
            if( inValue == 0.0 ) {
               columnIndex = paddingIndex;
               value = 0.0;
            }
            else {
               columnIndex = thisColumnsBuffer_view[ bufferIdx ];  // column - 1;
               value = inValue;
            }
         };
         this->forElements( baseRow, lastRow, f2 );
         baseRow += bufferRowsCount;
      }
   }
   this->view = this->getView();
   return *this;
}

template< typename Real, typename Device, typename Index, typename MatrixType, typename RealAllocator, typename IndexAllocator >
template< typename Matrix >
bool
SparseSandboxMatrix< Real, Device, Index, MatrixType, RealAllocator, IndexAllocator >::operator==( const Matrix& m ) const
{
   return view == m;
}

template< typename Real, typename Device, typename Index, typename MatrixType, typename RealAllocator, typename IndexAllocator >
template< typename Matrix >
bool
SparseSandboxMatrix< Real, Device, Index, MatrixType, RealAllocator, IndexAllocator >::operator!=( const Matrix& m ) const
{
   return view != m;
}

template< typename Real, typename Device, typename Index, typename MatrixType, typename RealAllocator, typename IndexAllocator >
void
SparseSandboxMatrix< Real, Device, Index, MatrixType, RealAllocator, IndexAllocator >::save( File& file ) const
{
   this->view.save( file );
}

template< typename Real, typename Device, typename Index, typename MatrixType, typename RealAllocator, typename IndexAllocator >
void
SparseSandboxMatrix< Real, Device, Index, MatrixType, RealAllocator, IndexAllocator >::load( File& file )
{
   Matrix< RealType, DeviceType, IndexType >::load( file );
   file >> this->columnIndexes;
   // SANDBOX_TODO: Replace the following line with loading of metadata required by your sparse matrix format.
   file >> rowPointers;
   this->view = this->getView();
}

template< typename Real, typename Device, typename Index, typename MatrixType, typename RealAllocator, typename IndexAllocator >
void
SparseSandboxMatrix< Real, Device, Index, MatrixType, RealAllocator, IndexAllocator >::save( const String& fileName ) const
{
   Object::save( fileName );
}

template< typename Real, typename Device, typename Index, typename MatrixType, typename RealAllocator, typename IndexAllocator >
void
SparseSandboxMatrix< Real, Device, Index, MatrixType, RealAllocator, IndexAllocator >::load( const String& fileName )
{
   Object::load( fileName );
}

template< typename Real, typename Device, typename Index, typename MatrixType, typename RealAllocator, typename IndexAllocator >
void
SparseSandboxMatrix< Real, Device, Index, MatrixType, RealAllocator, IndexAllocator >::print( std::ostream& str ) const
{
   this->view.print( str );
}

template< typename Real, typename Device, typename Index, typename MatrixType, typename RealAllocator, typename IndexAllocator >
__cuda_callable__
Index
SparseSandboxMatrix< Real, Device, Index, MatrixType, RealAllocator, IndexAllocator >::getPaddingIndex() const
{
   return -1;
}

template< typename Real, typename Device, typename Index, typename MatrixType, typename RealAllocator, typename IndexAllocator >
auto
SparseSandboxMatrix< Real, Device, Index, MatrixType, RealAllocator, IndexAllocator >::getColumnIndexes() const
   -> const ColumnsIndexesVectorType&
{
   return this->columnIndexes;
}

template< typename Real, typename Device, typename Index, typename MatrixType, typename RealAllocator, typename IndexAllocator >
auto
SparseSandboxMatrix< Real, Device, Index, MatrixType, RealAllocator, IndexAllocator >::getColumnIndexes()
   -> ColumnsIndexesVectorType&
{
   return this->columnIndexes;
}

}  // namespace Sandbox
}  // namespace Matrices
}  // namespace noa::TNL
