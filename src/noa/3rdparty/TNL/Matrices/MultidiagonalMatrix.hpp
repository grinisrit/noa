// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <sstream>
#include <noa/3rdparty/TNL/Assert.h>
#include <noa/3rdparty/TNL/Matrices/MultidiagonalMatrix.h>
#include <noa/3rdparty/TNL/Exceptions/NotImplementedError.h>

namespace noa::TNL {
namespace Matrices {

template< typename Real,
          typename Device,
          typename Index,
          ElementsOrganization Organization,
          typename RealAllocator,
          typename IndexAllocator >
MultidiagonalMatrix< Real, Device, Index, Organization, RealAllocator, IndexAllocator >::
MultidiagonalMatrix()
{
}

template< typename Real,
          typename Device,
          typename Index,
          ElementsOrganization Organization,
          typename RealAllocator,
          typename IndexAllocator >
   template< typename Vector >
MultidiagonalMatrix< Real, Device, Index, Organization, RealAllocator, IndexAllocator >::
MultidiagonalMatrix( const IndexType rows,
               const IndexType columns,
               const Vector& diagonalsOffsets )
{
   TNL_ASSERT_GT( diagonalsOffsets.getSize(), 0, "Cannot construct multidiagonal matrix with no diagonals offsets." );
   this->setDimensions( rows, columns, diagonalsOffsets );
}

template< typename Real,
          typename Device,
          typename Index,
          ElementsOrganization Organization,
          typename RealAllocator,
          typename IndexAllocator >
      template< typename ListIndex >
MultidiagonalMatrix< Real, Device, Index, Organization, RealAllocator, IndexAllocator >::
MultidiagonalMatrix( const IndexType rows,
                     const IndexType columns,
                     const std::initializer_list< ListIndex > diagonalsOffsets )
{
   Containers::Vector< IndexType, DeviceType, IndexType > offsets( diagonalsOffsets );
   TNL_ASSERT_GT( offsets.getSize(), 0, "Cannot construct multidiagonal matrix with no diagonals offsets." );
   this->setDimensions( rows, columns, offsets );
}

template< typename Real,
          typename Device,
          typename Index,
          ElementsOrganization Organization,
          typename RealAllocator,
          typename IndexAllocator >
   template< typename ListIndex, typename ListReal >
MultidiagonalMatrix< Real, Device, Index, Organization, RealAllocator, IndexAllocator >::
MultidiagonalMatrix( const IndexType columns,
                     const std::initializer_list< ListIndex > diagonalsOffsets,
                     const std::initializer_list< std::initializer_list< ListReal > >& data )
{
   Containers::Vector< IndexType, DeviceType, IndexType > offsets( diagonalsOffsets );
   TNL_ASSERT_GT( offsets.getSize(), 0, "Cannot construct multidiagonal matrix with no diagonals offsets." );
   this->setDimensions( data.size(), columns, offsets );
   this->setElements( data );
}

template< typename Real,
          typename Device,
          typename Index,
          ElementsOrganization Organization,
          typename RealAllocator,
          typename IndexAllocator >
auto
MultidiagonalMatrix< Real, Device, Index, Organization, RealAllocator, IndexAllocator >::
getView() const -> ViewType
{
   // TODO: fix when getConstView works
   return ViewType( const_cast< MultidiagonalMatrix* >( this )->values.getView(),
                    const_cast< MultidiagonalMatrix* >( this )->diagonalsOffsets.getView(),
                    const_cast< MultidiagonalMatrix* >( this )->hostDiagonalsOffsets.getView(),
                    indexer );
}

template< typename Real,
          typename Device,
          typename Index,
          ElementsOrganization Organization,
          typename RealAllocator,
          typename IndexAllocator >
String
MultidiagonalMatrix< Real, Device, Index, Organization, RealAllocator, IndexAllocator >::
getSerializationType()
{
   return ViewType::getSerializationType();
}

template< typename Real,
          typename Device,
          typename Index,
          ElementsOrganization Organization,
          typename RealAllocator,
          typename IndexAllocator >
String
MultidiagonalMatrix< Real, Device, Index, Organization, RealAllocator, IndexAllocator >::
getSerializationTypeVirtual() const
{
   return this->getSerializationType();
}

template< typename Real,
          typename Device,
          typename Index,
          ElementsOrganization Organization,
          typename RealAllocator,
          typename IndexAllocator >
   template< typename Vector >
void
MultidiagonalMatrix< Real, Device, Index, Organization, RealAllocator, IndexAllocator >::
setDimensions( const IndexType rows,
               const IndexType columns,
               const Vector& diagonalsOffsets )
{
   Matrix< Real, Device, Index >::setDimensions( rows, columns );
   this->diagonalsOffsets = diagonalsOffsets;
   this->hostDiagonalsOffsets = diagonalsOffsets;
   const IndexType minOffset = min( diagonalsOffsets );
   IndexType nonemptyRows = min( rows, columns );
   if( rows > columns && minOffset < 0 )
      nonemptyRows = min( rows, nonemptyRows - minOffset );
   this->indexer.set( rows, columns, diagonalsOffsets.getSize(), nonemptyRows );
   this->values.setSize( this->indexer.getStorageSize() );
   this->values = 0.0;
   this->view = this->getView();
}

template< typename Real,
          typename Device,
          typename Index,
          ElementsOrganization Organization,
          typename RealAllocator,
          typename IndexAllocator >
   template< typename Vector >
void
MultidiagonalMatrix< Real, Device, Index, Organization, RealAllocator, IndexAllocator >::
setDiagonalsOffsets( const Vector& diagonalsOffsets )
{
   TNL_ASSERT_GT( diagonalsOffsets.getSize(), 0, "Cannot construct multidiagonal matrix with no diagonals offsets." );
   this->setDimensions( this->getRows(), this->getColumns(), diagonalsOffsets );
}

template< typename Real,
          typename Device,
          typename Index,
          ElementsOrganization Organization,
          typename RealAllocator,
          typename IndexAllocator >
   template< typename ListIndex >
void
MultidiagonalMatrix< Real, Device, Index, Organization, RealAllocator, IndexAllocator >::
setDiagonalsOffsets( const std::initializer_list< ListIndex > diagonalsOffsets )
{
   Containers::Vector< IndexType, DeviceType, IndexType > offsets( diagonalsOffsets );
   TNL_ASSERT_GT( offsets.getSize(), 0, "Cannot construct multidiagonal matrix with no diagonals offsets." );
   this->setDimensions( this->getRows(), this->getColumns(), offsets );
}

template< typename Real,
          typename Device,
          typename Index,
          ElementsOrganization Organization,
          typename RealAllocator,
          typename IndexAllocator >
  template< typename RowCapacitiesVector >
void
MultidiagonalMatrix< Real, Device, Index, Organization, RealAllocator, IndexAllocator >::
setRowCapacities( const RowCapacitiesVector& rowLengths )
{
   if( max( rowLengths ) > 3 )
      throw std::logic_error( "Too many non-zero elements per row in a tri-diagonal matrix." );
   if( rowLengths.getElement( 0 ) > 2 )
      throw std::logic_error( "Too many non-zero elements per row in a tri-diagonal matrix." );
   const IndexType diagonalLength = min( this->getRows(), this->getColumns() );
   if( this->getRows() > this->getColumns() )
      if( rowLengths.getElement( this->getRows()-1 ) > 1 )
         throw std::logic_error( "Too many non-zero elements per row in a tri-diagonal matrix." );
   if( this->getRows() == this->getColumns() )
      if( rowLengths.getElement( this->getRows()-1 ) > 2 )
         throw std::logic_error( "Too many non-zero elements per row in a tri-diagonal matrix." );
   if( this->getRows() < this->getColumns() )
      if( rowLengths.getElement( this->getRows()-1 ) > 3 )
         throw std::logic_error( "Too many non-zero elements per row in a tri-diagonal matrix." );
}

template< typename Real,
          typename Device,
          typename Index,
          ElementsOrganization Organization,
          typename RealAllocator,
          typename IndexAllocator >
   template< typename Vector >
void
MultidiagonalMatrix< Real, Device, Index, Organization, RealAllocator, IndexAllocator >::
getRowCapacities( Vector& rowCapacities ) const
{
   return this->view.getRowCapacities( rowCapacities );
}

template< typename Real,
          typename Device,
          typename Index,
          ElementsOrganization Organization,
          typename RealAllocator,
          typename IndexAllocator >
   template< typename ListReal >
void
MultidiagonalMatrix< Real, Device, Index, Organization, RealAllocator, IndexAllocator >::
setElements( const std::initializer_list< std::initializer_list< ListReal > >& data )
{
   if( std::is_same< DeviceType, Devices::Host >::value )
   {
      this->getValues() = 0.0;
      auto row_it = data.begin();
      for( size_t rowIdx = 0; rowIdx < data.size(); rowIdx++ )
      {
         auto data_it = row_it->begin();
         IndexType i = 0;
         while( data_it != row_it->end() )
            this->getRow( rowIdx ).setElement( i++, *data_it++ );
         row_it ++;
      }
   }
   else
   {
      MultidiagonalMatrix< Real, Devices::Host, Index, Organization > hostMatrix(
         this->getRows(),
         this->getColumns(),
         this->getDiagonalsOffsets() );
      hostMatrix.setElements( data );
      *this = hostMatrix;
   }
}

template< typename Real,
          typename Device,
          typename Index,
          ElementsOrganization Organization,
          typename RealAllocator,
          typename IndexAllocator >
const Index
MultidiagonalMatrix< Real, Device, Index, Organization, RealAllocator, IndexAllocator >::
getDiagonalsCount() const
{
   return this->view.getDiagonalsCount();
}

template< typename Real,
          typename Device,
          typename Index,
          ElementsOrganization Organization,
          typename RealAllocator,
          typename IndexAllocator >
auto
MultidiagonalMatrix< Real, Device, Index, Organization, RealAllocator, IndexAllocator >::
getDiagonalsOffsets() const -> const DiagonalsOffsetsType&
{
   return this->diagonalsOffsets;
}

template< typename Real,
          typename Device,
          typename Index,
          ElementsOrganization Organization,
          typename RealAllocator,
          typename IndexAllocator >
   template< typename Vector >
void
MultidiagonalMatrix< Real, Device, Index, Organization, RealAllocator, IndexAllocator >::
getCompressedRowLengths( Vector& rowLengths ) const
{
   return this->view.getCompressedRowLengths( rowLengths );
}

template< typename Real,
          typename Device,
          typename Index,
          ElementsOrganization Organization,
          typename RealAllocator,
          typename IndexAllocator >
Index
MultidiagonalMatrix< Real, Device, Index, Organization, RealAllocator, IndexAllocator >::
getRowLength( const IndexType row ) const
{
   return this->view.getRowLength( row );
}

template< typename Real,
          typename Device,
          typename Index,
          ElementsOrganization Organization,
          typename RealAllocator,
          typename IndexAllocator >
   template< typename Real_, typename Device_, typename Index_, ElementsOrganization Organization_, typename RealAllocator_, typename IndexAllocator_ >
void
MultidiagonalMatrix< Real, Device, Index, Organization, RealAllocator, IndexAllocator >::
setLike( const MultidiagonalMatrix< Real_, Device_, Index_, Organization_, RealAllocator_, IndexAllocator_ >& matrix )
{
   this->setDimensions( matrix.getRows(), matrix.getColumns(), matrix.getDiagonalsOffsets() );
}

template< typename Real,
          typename Device,
          typename Index,
          ElementsOrganization Organization,
          typename RealAllocator,
          typename IndexAllocator >
Index
MultidiagonalMatrix< Real, Device, Index, Organization, RealAllocator, IndexAllocator >::
getNonzeroElementsCount() const
{
   return this->view.getNonzeroElementsCount();
}

template< typename Real,
          typename Device,
          typename Index,
          ElementsOrganization Organization,
          typename RealAllocator,
          typename IndexAllocator >
void
MultidiagonalMatrix< Real, Device, Index, Organization, RealAllocator, IndexAllocator >::
reset()
{
   Matrix< Real, Device, Index >::reset();
}

template< typename Real,
          typename Device,
          typename Index,
          ElementsOrganization Organization,
          typename RealAllocator,
          typename IndexAllocator >
   template< typename Real_, typename Device_, typename Index_, ElementsOrganization Organization_, typename RealAllocator_, typename IndexAllocator_ >
bool
MultidiagonalMatrix< Real, Device, Index, Organization, RealAllocator, IndexAllocator >::
operator == ( const MultidiagonalMatrix< Real_, Device_, Index_, Organization_, RealAllocator_, IndexAllocator_ >& matrix ) const
{
   if( Organization == Organization_ )
      return this->values == matrix.values;
   else
   {
      TNL_ASSERT_TRUE( false, "TODO" );
   }
}

template< typename Real,
          typename Device,
          typename Index,
          ElementsOrganization Organization,
          typename RealAllocator,
          typename IndexAllocator >
   template< typename Real_, typename Device_, typename Index_, ElementsOrganization Organization_, typename RealAllocator_, typename IndexAllocator_ >
bool
MultidiagonalMatrix< Real, Device, Index, Organization, RealAllocator, IndexAllocator >::
operator != ( const MultidiagonalMatrix< Real_, Device_, Index_, Organization_, RealAllocator_, IndexAllocator_ >& matrix ) const
{
   return ! this->operator==( matrix );
}

template< typename Real,
          typename Device,
          typename Index,
          ElementsOrganization Organization,
          typename RealAllocator,
          typename IndexAllocator >
void
MultidiagonalMatrix< Real, Device, Index, Organization, RealAllocator, IndexAllocator >::
setValue( const RealType& v )
{
   this->view.setValue( v );
}

template< typename Real,
          typename Device,
          typename Index,
          ElementsOrganization Organization,
          typename RealAllocator,
          typename IndexAllocator >
__cuda_callable__
auto
MultidiagonalMatrix< Real, Device, Index, Organization, RealAllocator, IndexAllocator >::
getRow( const IndexType& rowIdx ) const -> const ConstRowView
{
   return this->view.getRow( rowIdx );
}

template< typename Real,
          typename Device,
          typename Index,
          ElementsOrganization Organization,
          typename RealAllocator,
          typename IndexAllocator >
__cuda_callable__
auto
MultidiagonalMatrix< Real, Device, Index, Organization, RealAllocator, IndexAllocator >::
getRow( const IndexType& rowIdx ) -> RowView
{
   return this->view.getRow( rowIdx );
}

template< typename Real,
          typename Device,
          typename Index,
          ElementsOrganization Organization,
          typename RealAllocator,
          typename IndexAllocator >
__cuda_callable__
void
MultidiagonalMatrix< Real, Device, Index, Organization, RealAllocator, IndexAllocator >::
setElement( const IndexType row, const IndexType column, const RealType& value )
{
   this->view.setElement( row, column, value );
}

template< typename Real,
          typename Device,
          typename Index,
          ElementsOrganization Organization,
          typename RealAllocator,
          typename IndexAllocator >
__cuda_callable__
void
MultidiagonalMatrix< Real, Device, Index, Organization, RealAllocator, IndexAllocator >::
addElement( const IndexType row,
            const IndexType column,
            const RealType& value,
            const RealType& thisElementMultiplicator )
{
   this->view.addElement( row, column, value, thisElementMultiplicator );
}

template< typename Real,
          typename Device,
          typename Index,
          ElementsOrganization Organization,
          typename RealAllocator,
          typename IndexAllocator >
__cuda_callable__
Real
MultidiagonalMatrix< Real, Device, Index, Organization, RealAllocator, IndexAllocator >::
getElement( const IndexType row, const IndexType column ) const
{
   return this->view.getElement( row, column );
}

template< typename Real,
          typename Device,
          typename Index,
          ElementsOrganization Organization,
          typename RealAllocator,
          typename IndexAllocator >
   template< typename Fetch, typename Reduce, typename Keep, typename FetchReal >
void
MultidiagonalMatrix< Real, Device, Index, Organization, RealAllocator, IndexAllocator >::
reduceRows( IndexType first, IndexType last, Fetch& fetch, Reduce& reduce, Keep& keep, const FetchReal& identity ) const
{
   this->view.reduceRows( first, last, fetch, reduce, keep, identity );
}

template< typename Real,
          typename Device,
          typename Index,
          ElementsOrganization Organization,
          typename RealAllocator,
          typename IndexAllocator >
   template< typename Fetch, typename Reduce, typename Keep, typename FetchReal >
void
MultidiagonalMatrix< Real, Device, Index, Organization, RealAllocator, IndexAllocator >::
reduceRows( IndexType first, IndexType last, Fetch& fetch, Reduce& reduce, Keep& keep, const FetchReal& identity )
{
   this->view.reduceRows( first, last, fetch, reduce, keep, identity );
}

template< typename Real,
          typename Device,
          typename Index,
          ElementsOrganization Organization,
          typename RealAllocator,
          typename IndexAllocator >
   template< typename Fetch, typename Reduce, typename Keep, typename FetchReal >
void
MultidiagonalMatrix< Real, Device, Index, Organization, RealAllocator, IndexAllocator >::
reduceAllRows( Fetch& fetch, Reduce& reduce, Keep& keep, const FetchReal& identity ) const
{
   this->view.reduceRows( (IndexType) 0, this->getRows(), fetch, reduce, keep, identity );
}

template< typename Real,
          typename Device,
          typename Index,
          ElementsOrganization Organization,
          typename RealAllocator,
          typename IndexAllocator >
   template< typename Fetch, typename Reduce, typename Keep, typename FetchReal >
void
MultidiagonalMatrix< Real, Device, Index, Organization, RealAllocator, IndexAllocator >::
reduceAllRows( Fetch& fetch, Reduce& reduce, Keep& keep, const FetchReal& identity )
{
   this->view.reduceRows( (IndexType) 0, this->getRows(), fetch, reduce, keep, identity );
}

template< typename Real,
          typename Device,
          typename Index,
          ElementsOrganization Organization,
          typename RealAllocator,
          typename IndexAllocator >
   template< typename Function >
void
MultidiagonalMatrix< Real, Device, Index, Organization, RealAllocator, IndexAllocator >::
forElements( IndexType first, IndexType last, Function& function ) const
{
   this->view.forElements( first, last, function );
}

template< typename Real,
          typename Device,
          typename Index,
          ElementsOrganization Organization,
          typename RealAllocator,
          typename IndexAllocator >
  template< typename Function >
void
MultidiagonalMatrix< Real, Device, Index, Organization, RealAllocator, IndexAllocator >::
forElements( IndexType first, IndexType last, Function& function )
{
   this->view.forElements( first, last, function );
}

template< typename Real,
          typename Device,
          typename Index,
          ElementsOrganization Organization,
          typename RealAllocator,
          typename IndexAllocator >
   template< typename Function >
void
MultidiagonalMatrix< Real, Device, Index, Organization, RealAllocator, IndexAllocator >::
forAllElements( Function& function ) const
{
   this->view.forElements( (IndexType) 0, this->getRows(), function );
}

template< typename Real,
          typename Device,
          typename Index,
          ElementsOrganization Organization,
          typename RealAllocator,
          typename IndexAllocator >
   template< typename Function >
void
MultidiagonalMatrix< Real, Device, Index, Organization, RealAllocator, IndexAllocator >::
forAllElements( Function& function )
{
   this->view.forElements( (IndexType) 0, this->getRows(), function );
}

template< typename Real,
          typename Device,
          typename Index,
          ElementsOrganization Organization,
          typename RealAllocator,
          typename IndexAllocator >
   template< typename Function >
void
MultidiagonalMatrix< Real, Device, Index, Organization, RealAllocator, IndexAllocator >::
forRows( IndexType begin, IndexType end, Function&& function )
{
   this->getView().forRows( begin, end, function );
}

template< typename Real,
          typename Device,
          typename Index,
          ElementsOrganization Organization,
          typename RealAllocator,
          typename IndexAllocator >
   template< typename Function >
void
MultidiagonalMatrix< Real, Device, Index, Organization, RealAllocator, IndexAllocator >::
forRows( IndexType begin, IndexType end, Function&& function ) const
{
   this->getConstView().forRows( begin, end, function );
}

template< typename Real,
          typename Device,
          typename Index,
          ElementsOrganization Organization,
          typename RealAllocator,
          typename IndexAllocator >
   template< typename Function >
void
MultidiagonalMatrix< Real, Device, Index, Organization, RealAllocator, IndexAllocator >::
forAllRows( Function&& function )
{
   this->getView().forAllRows( function );
}

template< typename Real,
          typename Device,
          typename Index,
          ElementsOrganization Organization,
          typename RealAllocator,
          typename IndexAllocator >
   template< typename Function >
void
MultidiagonalMatrix< Real, Device, Index, Organization, RealAllocator, IndexAllocator >::
forAllRows( Function&& function ) const
{
   this->getConsView().forAllRows( function );
}

template< typename Real,
          typename Device,
          typename Index,
          ElementsOrganization Organization,
          typename RealAllocator,
          typename IndexAllocator >
   template< typename Function >
void
MultidiagonalMatrix< Real, Device, Index, Organization, RealAllocator, IndexAllocator >::
sequentialForRows( IndexType begin, IndexType end, Function& function ) const
{
   this->view.sequentialForRows( begin, end, function );
}

template< typename Real,
          typename Device,
          typename Index,
          ElementsOrganization Organization,
          typename RealAllocator,
          typename IndexAllocator >
   template< typename Function >
void
MultidiagonalMatrix< Real, Device, Index, Organization, RealAllocator, IndexAllocator >::
sequentialForRows( IndexType first, IndexType last, Function& function )
{
   this->view.sequentialForRows( first, last, function );
}

template< typename Real,
          typename Device,
          typename Index,
          ElementsOrganization Organization,
          typename RealAllocator,
          typename IndexAllocator >
   template< typename Function >
void
MultidiagonalMatrix< Real, Device, Index, Organization, RealAllocator, IndexAllocator >::
sequentialForAllRows( Function& function ) const
{
   this->sequentialForRows( (IndexType) 0, this->getRows(), function );
}

template< typename Real,
          typename Device,
          typename Index,
          ElementsOrganization Organization,
          typename RealAllocator,
          typename IndexAllocator >
   template< typename Function >
void
MultidiagonalMatrix< Real, Device, Index, Organization, RealAllocator, IndexAllocator >::
sequentialForAllRows( Function& function )
{
   this->sequentialForRows( (IndexType) 0, this->getRows(), function );
}

template< typename Real,
          typename Device,
          typename Index,
          ElementsOrganization Organization,
          typename RealAllocator,
          typename IndexAllocator >
   template< typename InVector,
             typename OutVector >
void
MultidiagonalMatrix< Real, Device, Index, Organization, RealAllocator, IndexAllocator >::
vectorProduct( const InVector& inVector,
               OutVector& outVector,
               const RealType matrixMultiplicator,
               const RealType outVectorMultiplicator,
               const IndexType firstRow,
               IndexType lastRow ) const
{
   this->view.vectorProduct( inVector, outVector, matrixMultiplicator,
                              outVectorMultiplicator, firstRow, lastRow );
}

template< typename Real,
          typename Device,
          typename Index,
          ElementsOrganization Organization,
          typename RealAllocator,
          typename IndexAllocator >
   template< typename Real_, typename Device_, typename Index_, ElementsOrganization Organization_, typename RealAllocator_ >
void
MultidiagonalMatrix< Real, Device, Index, Organization, RealAllocator, IndexAllocator >::
addMatrix( const MultidiagonalMatrix< Real_, Device_, Index_, Organization_, RealAllocator_ >& matrix,
           const RealType& matrixMultiplicator,
           const RealType& thisMatrixMultiplicator )
{
   this->view.addMatrix( matrix.getView(), matrixMultiplicator, thisMatrixMultiplicator );
}

#ifdef HAVE_CUDA
template< typename Real,
          typename Real2,
          typename Index,
          typename Index2 >
__global__ void MultidiagonalMatrixTranspositionCudaKernel( const MultidiagonalMatrix< Real2, Devices::Cuda, Index2 >* inMatrix,
                                                             MultidiagonalMatrix< Real, Devices::Cuda, Index >* outMatrix,
                                                             const Real matrixMultiplicator,
                                                             const Index gridIdx )
{
   const Index rowIdx = ( gridIdx * Cuda::getMaxGridSize() + blockIdx.x ) * blockDim.x + threadIdx.x;
   if( rowIdx < inMatrix->getRows() )
   {
      if( rowIdx > 0 )
        outMatrix->setElementFast( rowIdx-1,
                                   rowIdx,
                                   matrixMultiplicator * inMatrix->getElementFast( rowIdx, rowIdx-1 ) );
      outMatrix->setElementFast( rowIdx,
                                 rowIdx,
                                 matrixMultiplicator * inMatrix->getElementFast( rowIdx, rowIdx ) );
      if( rowIdx < inMatrix->getRows()-1 )
         outMatrix->setElementFast( rowIdx+1,
                                    rowIdx,
                                    matrixMultiplicator * inMatrix->getElementFast( rowIdx, rowIdx+1 ) );
   }
}
#endif

template< typename Real,
          typename Device,
          typename Index,
          ElementsOrganization Organization,
          typename RealAllocator,
          typename IndexAllocator >
   template< typename Real2, typename Index2 >
void MultidiagonalMatrix< Real, Device, Index, Organization, RealAllocator, IndexAllocator >::
getTransposition( const MultidiagonalMatrix< Real2, Device, Index2 >& matrix,
                  const RealType& matrixMultiplicator )
{
   TNL_ASSERT( this->getRows() == matrix.getRows(),
               std::cerr << "This matrix rows: " << this->getRows() << std::endl
                    << "That matrix rows: " << matrix.getRows() << std::endl );
   if( std::is_same< Device, Devices::Host >::value )
   {
      const IndexType& rows = matrix.getRows();
      for( IndexType i = 1; i < rows; i++ )
      {
         RealType aux = matrix. getElement( i, i - 1 );
         this->setElement( i, i - 1, matrix.getElement( i - 1, i ) );
         this->setElement( i, i, matrix.getElement( i, i ) );
         this->setElement( i - 1, i, aux );
      }
   }
   if( std::is_same< Device, Devices::Cuda >::value )
   {
#ifdef HAVE_CUDA
      MultidiagonalMatrix* kernel_this = Cuda::passToDevice( *this );
      typedef  MultidiagonalMatrix< Real2, Device, Index2 > InMatrixType;
      InMatrixType* kernel_inMatrix = Cuda::passToDevice( matrix );
      dim3 cudaBlockSize( 256 ), cudaGridSize( Cuda::getMaxGridSize() );
      const IndexType cudaBlocks = roundUpDivision( matrix.getRows(), cudaBlockSize.x );
      const IndexType cudaGrids = roundUpDivision( cudaBlocks, Cuda::getMaxGridSize() );
      for( IndexType gridIdx = 0; gridIdx < cudaGrids; gridIdx++ )
      {
         if( gridIdx == cudaGrids - 1 )
            cudaGridSize.x = cudaBlocks % Cuda::getMaxGridSize();
         MultidiagonalMatrixTranspositionCudaKernel<<< cudaGridSize, cudaBlockSize >>>
                                                    ( kernel_inMatrix,
                                                      kernel_this,
                                                      matrixMultiplicator,
                                                      gridIdx );
      }
      Cuda::freeFromDevice( kernel_this );
      Cuda::freeFromDevice( kernel_inMatrix );
      TNL_CHECK_CUDA_DEVICE;
#endif
   }
}

// copy assignment
template< typename Real,
          typename Device,
          typename Index,
          ElementsOrganization Organization,
          typename RealAllocator,
          typename IndexAllocator >
MultidiagonalMatrix< Real, Device, Index, Organization, RealAllocator, IndexAllocator >&
MultidiagonalMatrix< Real, Device, Index, Organization, RealAllocator, IndexAllocator >::operator=( const MultidiagonalMatrix& matrix )
{
   this->setLike( matrix );
   this->values = matrix.values;
   return *this;
}

// cross-device copy assignment
template< typename Real,
          typename Device,
          typename Index,
          ElementsOrganization Organization,
          typename RealAllocator,
          typename IndexAllocator >
   template< typename Real_, typename Device_, typename Index_, ElementsOrganization Organization_, typename RealAllocator_, typename IndexAllocator_ >
MultidiagonalMatrix< Real, Device, Index, Organization, RealAllocator, IndexAllocator >&
MultidiagonalMatrix< Real, Device, Index, Organization, RealAllocator, IndexAllocator >::
operator=( const MultidiagonalMatrix< Real_, Device_, Index_, Organization_, RealAllocator_, IndexAllocator_ >& matrix )
{
   using RHSMatrix = MultidiagonalMatrix< Real_, Device_, Index_, Organization_, RealAllocator_, IndexAllocator_ >;
   using RHSIndexType = typename RHSMatrix::IndexType;
   using RHSRealType = typename RHSMatrix::RealType;
   using RHSDeviceType = typename RHSMatrix::DeviceType;
   using RHSRealAllocatorType = typename RHSMatrix::RealAllocatorType;
   using RHSIndexAllocatorType = typename RHSMatrix::IndexAllocatorType;

   this->setLike( matrix );
   if( Organization == Organization_ )
      this->values = matrix.getValues();
   else
   {
      if( std::is_same< Device, Device_ >::value )
      {
         const auto matrix_view = matrix.getView();
         auto f = [=] __cuda_callable__ ( const IndexType& rowIdx, const IndexType& localIdx, const IndexType& column, Real& value ) mutable {
            value = matrix_view.getValues()[ matrix_view.getIndexer().getGlobalIndex( rowIdx, localIdx ) ];
         };
         this->forAllElements( f );
      }
      else
      {
         const IndexType maxRowLength = this->diagonalsOffsets.getSize();
         const IndexType bufferRowsCount( 128 );
         const size_t bufferSize = bufferRowsCount * maxRowLength;
         Containers::Vector< RHSRealType, RHSDeviceType, RHSIndexType, RHSRealAllocatorType > matrixValuesBuffer( bufferSize );
         Containers::Vector< RHSIndexType, RHSDeviceType, RHSIndexType, RHSIndexAllocatorType > matrixColumnsBuffer( bufferSize );
         Containers::Vector< RealType, DeviceType, IndexType, RealAllocatorType > thisValuesBuffer( bufferSize );
         Containers::Vector< IndexType, DeviceType, IndexType, IndexAllocatorType > thisColumnsBuffer( bufferSize );
         auto matrixValuesBuffer_view = matrixValuesBuffer.getView();
         auto thisValuesBuffer_view = thisValuesBuffer.getView();

         IndexType baseRow( 0 );
         const IndexType rowsCount = this->getRows();
         while( baseRow < rowsCount )
         {
            const IndexType lastRow = min( baseRow + bufferRowsCount, rowsCount );

            ////
            // Copy matrix elements into buffer
            auto f1 = [=] __cuda_callable__ ( RHSIndexType rowIdx, RHSIndexType localIdx, RHSIndexType columnIndex, const RHSRealType& value ) mutable {
                  const IndexType bufferIdx = ( rowIdx - baseRow ) * maxRowLength + localIdx;
                  matrixValuesBuffer_view[ bufferIdx ] = value;
            };
            matrix.forElements( baseRow, lastRow, f1 );

            ////
            // Copy the source matrix buffer to this matrix buffer
            thisValuesBuffer_view = matrixValuesBuffer_view;

            ////
            // Copy matrix elements from the buffer to the matrix
            auto f2 = [=] __cuda_callable__ ( const IndexType rowIdx, const IndexType localIdx, const IndexType columnIndex, RealType& value ) mutable {
               const IndexType bufferIdx = ( rowIdx - baseRow ) * maxRowLength + localIdx;
                  value = thisValuesBuffer_view[ bufferIdx ];
            };
            this->forElements( baseRow, lastRow, f2 );
            baseRow += bufferRowsCount;
         }
      }
   }
   return *this;
}

template< typename Real,
          typename Device,
          typename Index,
          ElementsOrganization Organization,
          typename RealAllocator,
          typename IndexAllocator >
void MultidiagonalMatrix< Real, Device, Index, Organization, RealAllocator, IndexAllocator >::save( File& file ) const
{
   Matrix< Real, Device, Index >::save( file );
   file << diagonalsOffsets;
}

template< typename Real,
          typename Device,
          typename Index,
          ElementsOrganization Organization,
          typename RealAllocator,
          typename IndexAllocator >
void MultidiagonalMatrix< Real, Device, Index, Organization, RealAllocator, IndexAllocator >::load( File& file )
{
   Matrix< Real, Device, Index >::load( file );
   file >> this->diagonalsOffsets;
   this->hostDiagonalsOffsets = this->diagonalsOffsets;
   const IndexType minOffset = min( diagonalsOffsets );
   IndexType nonemptyRows = min( this->getRows(), this->getColumns() );
   if( this->getRows() > this->getColumns() && minOffset < 0 )
      nonemptyRows = min( this->getRows(), nonemptyRows - minOffset );
   this->indexer.set( this->getRows(), this->getColumns(), diagonalsOffsets.getSize(), nonemptyRows );
   this->view = this->getView();
}

template< typename Real,
          typename Device,
          typename Index,
          ElementsOrganization Organization,
          typename RealAllocator,
          typename IndexAllocator >
void MultidiagonalMatrix< Real, Device, Index, Organization, RealAllocator, IndexAllocator >::save( const String& fileName ) const
{
   Object::save( fileName );
}

template< typename Real,
          typename Device,
          typename Index,
          ElementsOrganization Organization,
          typename RealAllocator,
          typename IndexAllocator >
void MultidiagonalMatrix< Real, Device, Index, Organization, RealAllocator, IndexAllocator >::load( const String& fileName )
{
   Object::load( fileName );
}

template< typename Real,
          typename Device,
          typename Index,
          ElementsOrganization Organization,
          typename RealAllocator,
          typename IndexAllocator >
void
MultidiagonalMatrix< Real, Device, Index, Organization, RealAllocator, IndexAllocator >::
print( std::ostream& str ) const
{
   this->view.print( str );
}

template< typename Real,
          typename Device,
          typename Index,
          ElementsOrganization Organization,
          typename RealAllocator,
          typename IndexAllocator >
auto
MultidiagonalMatrix< Real, Device, Index, Organization, RealAllocator, IndexAllocator >::
getIndexer() const -> const IndexerType&
{
   return this->indexer;
}

template< typename Real,
          typename Device,
          typename Index,
          ElementsOrganization Organization,
          typename RealAllocator,
          typename IndexAllocator >
auto
MultidiagonalMatrix< Real, Device, Index, Organization, RealAllocator, IndexAllocator >::
getIndexer() -> IndexerType&
{
   return this->indexer;
}

template< typename Real,
          typename Device,
          typename Index,
          ElementsOrganization Organization,
          typename RealAllocator,
          typename IndexAllocator >
__cuda_callable__
Index
MultidiagonalMatrix< Real, Device, Index, Organization, RealAllocator, IndexAllocator >::
getPaddingIndex() const
{
   return this->view.getPaddingIndex();
}

} // namespace Matrices
} // namespace noa::TNL
