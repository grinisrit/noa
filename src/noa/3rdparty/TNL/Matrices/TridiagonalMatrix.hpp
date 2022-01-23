// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <sstream>
#include <noa/3rdparty/TNL/Assert.h>
#include <noa/3rdparty/TNL/Matrices/TridiagonalMatrix.h>
#include <noa/3rdparty/TNL/Exceptions/NotImplementedError.h>

namespace noa::TNL {
namespace Matrices {

template< typename Real,
          typename Device,
          typename Index,
          ElementsOrganization Organization,
          typename RealAllocator >
TridiagonalMatrix< Real, Device, Index, Organization, RealAllocator >::
TridiagonalMatrix()
{
}

template< typename Real,
          typename Device,
          typename Index,
          ElementsOrganization Organization,
          typename RealAllocator >
TridiagonalMatrix< Real, Device, Index, Organization, RealAllocator >::
TridiagonalMatrix( const IndexType rows, const IndexType columns )
{
   this->setDimensions( rows, columns );
}

template< typename Real,
          typename Device,
          typename Index,
          ElementsOrganization Organization,
          typename RealAllocator >
   template< typename ListReal >
TridiagonalMatrix< Real, Device, Index, Organization, RealAllocator >::
TridiagonalMatrix( const IndexType columns,
                   const std::initializer_list< std::initializer_list< ListReal > >& data )
{
   this->setDimensions( data.size(), columns );
   this->setElements( data );
}

template< typename Real,
          typename Device,
          typename Index,
          ElementsOrganization Organization,
          typename RealAllocator >
auto
TridiagonalMatrix< Real, Device, Index, Organization, RealAllocator >::
getView() const -> ViewType
{
   // TODO: fix when getConstView works
   return ViewType( const_cast< TridiagonalMatrix* >( this )->values.getView(), indexer );
}

template< typename Real,
          typename Device,
          typename Index,
          ElementsOrganization Organization,
          typename RealAllocator >
String
TridiagonalMatrix< Real, Device, Index, Organization, RealAllocator >::
getSerializationType()
{
   return String( "Matrices::TridiagonalMatrix< " ) +
          noa::TNL::getSerializationType< RealType >() + ", [any_device], " +
          noa::TNL::getSerializationType< IndexType >() + ", " +
          ( Organization ? "true" : "false" ) + ", [any_allocator] >";
}

template< typename Real,
          typename Device,
          typename Index,
          ElementsOrganization Organization,
          typename RealAllocator >
String
TridiagonalMatrix< Real, Device, Index, Organization, RealAllocator >::
getSerializationTypeVirtual() const
{
   return this->getSerializationType();
}

template< typename Real,
          typename Device,
          typename Index,
          ElementsOrganization Organization,
          typename RealAllocator >
void
TridiagonalMatrix< Real, Device, Index, Organization, RealAllocator >::
setDimensions( const IndexType rows, const IndexType columns )
{
   Matrix< Real, Device, Index >::setDimensions( rows, columns );
   this->indexer.setDimensions( rows, columns );
   this->values.setSize( this->indexer.getStorageSize() );
   this->values = 0.0;
   this->view = this->getView();
}

template< typename Real,
          typename Device,
          typename Index,
          ElementsOrganization Organization,
          typename RealAllocator >
   template< typename RowCapacitiesVector >
void
TridiagonalMatrix< Real, Device, Index, Organization, RealAllocator >::
setRowCapacities( const RowCapacitiesVector& rowCapacities )
{
   if( max( rowCapacities ) > 3 )
      throw std::logic_error( "Too many non-zero elements per row in a tri-diagonal matrix." );
   if( rowCapacities.getElement( 0 ) > 2 )
      throw std::logic_error( "Too many non-zero elements per row in a tri-diagonal matrix." );
   const IndexType diagonalLength = min( this->getRows(), this->getColumns() );
   if( this->getRows() > this->getColumns() )
      if( rowCapacities.getElement( this->getRows()-1 ) > 1 )
         throw std::logic_error( "Too many non-zero elements per row in a tri-diagonal matrix." );
   if( this->getRows() == this->getColumns() )
      if( rowCapacities.getElement( this->getRows()-1 ) > 2 )
         throw std::logic_error( "Too many non-zero elements per row in a tri-diagonal matrix." );
   if( this->getRows() < this->getColumns() )
      if( rowCapacities.getElement( this->getRows()-1 ) > 3 )
         throw std::logic_error( "Too many non-zero elements per row in a tri-diagonal matrix." );
}

template< typename Real,
          typename Device,
          typename Index,
          ElementsOrganization Organization,
          typename RealAllocator >
   template< typename Vector >
void
TridiagonalMatrix< Real, Device, Index, Organization, RealAllocator >::
getRowCapacities( Vector& rowCapacities ) const
{
   return this->view.getRowCapacities( rowCapacities );
}

template< typename Real,
          typename Device,
          typename Index,
          ElementsOrganization Organization,
          typename RealAllocator >
   template< typename ListReal >
void
TridiagonalMatrix< Real, Device, Index, Organization, RealAllocator >::
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
      TridiagonalMatrix< Real, Devices::Host, Index, Organization > hostMatrix(
         this->getRows(),
         this->getColumns() );
      hostMatrix.setElements( data );
      *this = hostMatrix;
   }
}


template< typename Real,
          typename Device,
          typename Index,
          ElementsOrganization Organization,
          typename RealAllocator >
   template< typename Vector >
void
TridiagonalMatrix< Real, Device, Index, Organization, RealAllocator >::
getCompressedRowLengths( Vector& rowLengths ) const
{
   return this->view.getCompressedRowLengths( rowLengths );
}

template< typename Real,
          typename Device,
          typename Index,
          ElementsOrganization Organization,
          typename RealAllocator >
   template< typename Real_, typename Device_, typename Index_, ElementsOrganization Organization_, typename RealAllocator_ >
void
TridiagonalMatrix< Real, Device, Index, Organization, RealAllocator >::
setLike( const TridiagonalMatrix< Real_, Device_, Index_, Organization_, RealAllocator_ >& m )
{
   this->setDimensions( m.getRows(), m.getColumns() );
}

template< typename Real,
          typename Device,
          typename Index,
          ElementsOrganization Organization,
          typename RealAllocator >
Index
TridiagonalMatrix< Real, Device, Index, Organization, RealAllocator >::
getNonzeroElementsCount() const
{
   return this->view.getNonzeroElementsCount();
}

template< typename Real,
          typename Device,
          typename Index,
          ElementsOrganization Organization,
          typename RealAllocator >
void
TridiagonalMatrix< Real, Device, Index, Organization, RealAllocator >::
reset()
{
   Matrix< Real, Device, Index >::reset();
}

template< typename Real,
          typename Device,
          typename Index,
          ElementsOrganization Organization,
          typename RealAllocator >
   template< typename Real_, typename Device_, typename Index_, ElementsOrganization Organization_, typename RealAllocator_ >
bool
TridiagonalMatrix< Real, Device, Index, Organization, RealAllocator >::
operator == ( const TridiagonalMatrix< Real_, Device_, Index_, Organization_, RealAllocator_ >& matrix ) const
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
          typename RealAllocator >
   template< typename Real_, typename Device_, typename Index_, ElementsOrganization Organization_, typename RealAllocator_ >
bool
TridiagonalMatrix< Real, Device, Index, Organization, RealAllocator >::
operator != ( const TridiagonalMatrix< Real_, Device_, Index_, Organization_, RealAllocator_ >& matrix ) const
{
   return ! this->operator==( matrix );
}

template< typename Real,
          typename Device,
          typename Index,
          ElementsOrganization Organization,
          typename RealAllocator >
void
TridiagonalMatrix< Real, Device, Index, Organization, RealAllocator >::
setValue( const RealType& v )
{
   this->view.setValue( v );
}

template< typename Real,
          typename Device,
          typename Index,
          ElementsOrganization Organization,
          typename RealAllocator >
__cuda_callable__
auto
TridiagonalMatrix< Real, Device, Index, Organization, RealAllocator >::
getRow( const IndexType& rowIdx ) const -> const ConstRowView
{
   return this->view.getRow( rowIdx );
}

template< typename Real,
          typename Device,
          typename Index,
          ElementsOrganization Organization,
          typename RealAllocator >
__cuda_callable__
auto
TridiagonalMatrix< Real, Device, Index, Organization, RealAllocator >::
getRow( const IndexType& rowIdx ) -> RowView
{
   return this->view.getRow( rowIdx );
}

template< typename Real,
          typename Device,
          typename Index,
          ElementsOrganization Organization,
          typename RealAllocator >
void
TridiagonalMatrix< Real, Device, Index, Organization, RealAllocator >::
setElement( const IndexType row, const IndexType column, const RealType& value )
{
   this->view.setElement( row, column, value );
}

template< typename Real,
          typename Device,
          typename Index,
          ElementsOrganization Organization,
          typename RealAllocator >
void
TridiagonalMatrix< Real, Device, Index, Organization, RealAllocator >::
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
          typename RealAllocator >
Real
TridiagonalMatrix< Real, Device, Index, Organization, RealAllocator >::
getElement( const IndexType row, const IndexType column ) const
{
   return this->view.getElement( row, column );
}

template< typename Real,
          typename Device,
          typename Index,
          ElementsOrganization Organization,
          typename RealAllocator >
   template< typename Fetch, typename Reduce, typename Keep, typename FetchReal >
void
TridiagonalMatrix< Real, Device, Index, Organization, RealAllocator >::
reduceRows( IndexType first, IndexType last, Fetch& fetch, Reduce& reduce, Keep& keep, const FetchReal& identity ) const
{
   this->view.reduceRows( first, last, fetch, reduce, keep, identity );
}

template< typename Real,
          typename Device,
          typename Index,
          ElementsOrganization Organization,
          typename RealAllocator >
   template< typename Fetch, typename Reduce, typename Keep, typename FetchReal >
void
TridiagonalMatrix< Real, Device, Index, Organization, RealAllocator >::
reduceRows( IndexType first, IndexType last, Fetch& fetch, Reduce& reduce, Keep& keep, const FetchReal& identity )
{
   this->view.reduceRows( first, last, fetch, reduce, keep, identity );
}

template< typename Real,
          typename Device,
          typename Index,
          ElementsOrganization Organization,
          typename RealAllocator >
   template< typename Fetch, typename Reduce, typename Keep, typename FetchReal >
void
TridiagonalMatrix< Real, Device, Index, Organization, RealAllocator >::
reduceAllRows( Fetch& fetch, Reduce& reduce, Keep& keep, const FetchReal& identity ) const
{
   this->view.reduceRows( (IndexType) 0, this->getRows(), fetch, reduce, keep, identity );
}

template< typename Real,
          typename Device,
          typename Index,
          ElementsOrganization Organization,
          typename RealAllocator >
   template< typename Fetch, typename Reduce, typename Keep, typename FetchReal >
void
TridiagonalMatrix< Real, Device, Index, Organization, RealAllocator >::
reduceAllRows( Fetch& fetch, Reduce& reduce, Keep& keep, const FetchReal& identity )
{
   this->view.reduceRows( (IndexType) 0, this->getRows(), fetch, reduce, keep, identity );
}

template< typename Real,
          typename Device,
          typename Index,
          ElementsOrganization Organization,
          typename RealAllocator >
   template< typename Function >
void
TridiagonalMatrix< Real, Device, Index, Organization, RealAllocator >::
forElements( IndexType first, IndexType last, Function& function ) const
{
   this->view.forElements( first, last, function );
}

template< typename Real,
          typename Device,
          typename Index,
          ElementsOrganization Organization,
          typename RealAllocator >
  template< typename Function >
void
TridiagonalMatrix< Real, Device, Index, Organization, RealAllocator >::
forElements( IndexType first, IndexType last, Function& function )
{
   this->view.forElements( first, last, function );
}

template< typename Real,
          typename Device,
          typename Index,
          ElementsOrganization Organization,
          typename RealAllocator >
   template< typename Function >
void
TridiagonalMatrix< Real, Device, Index, Organization, RealAllocator >::
forAllElements( Function& function ) const
{
   this->view.forElements( (IndexType) 0, this->getRows(), function );
}

template< typename Real,
          typename Device,
          typename Index,
          ElementsOrganization Organization,
          typename RealAllocator >
   template< typename Function >
void
TridiagonalMatrix< Real, Device, Index, Organization, RealAllocator >::
forAllElements( Function& function )
{
   this->view.forElements( (IndexType) 0, this->getRows(), function );
}

template< typename Real,
          typename Device,
          typename Index,
          ElementsOrganization Organization,
          typename RealAllocator >
   template< typename Function >
void
TridiagonalMatrix< Real, Device, Index, Organization, RealAllocator >::
forRows( IndexType begin, IndexType end, Function&& function )
{
   this->getView().forRows( begin, end, function );
}

template< typename Real,
          typename Device,
          typename Index,
          ElementsOrganization Organization,
          typename RealAllocator >
   template< typename Function >
void
TridiagonalMatrix< Real, Device, Index, Organization, RealAllocator >::
forRows( IndexType begin, IndexType end, Function&& function ) const
{
   this->getConstView().forRows( begin, end, function );
}

template< typename Real,
          typename Device,
          typename Index,
          ElementsOrganization Organization,
          typename RealAllocator >
   template< typename Function >
void
TridiagonalMatrix< Real, Device, Index, Organization, RealAllocator >::
forAllRows( Function&& function )
{
   this->getView().forAllRows( function );
}

template< typename Real,
          typename Device,
          typename Index,
          ElementsOrganization Organization,
          typename RealAllocator >
   template< typename Function >
void
TridiagonalMatrix< Real, Device, Index, Organization, RealAllocator >::
forAllRows( Function&& function ) const
{
   this->getConsView().forAllRows( function );
}

template< typename Real,
          typename Device,
          typename Index,
          ElementsOrganization Organization,
          typename RealAllocator >
   template< typename Function >
void
TridiagonalMatrix< Real, Device, Index, Organization, RealAllocator >::
sequentialForRows( IndexType begin, IndexType end, Function& function ) const
{
   this->view.sequentialForRows( begin, end, function );
}

template< typename Real,
          typename Device,
          typename Index,
          ElementsOrganization Organization,
          typename RealAllocator >
   template< typename Function >
void
TridiagonalMatrix< Real, Device, Index, Organization, RealAllocator >::
sequentialForRows( IndexType first, IndexType last, Function& function )
{
   this->view.sequentialForRows( first, last, function );
}

template< typename Real,
          typename Device,
          typename Index,
          ElementsOrganization Organization,
          typename RealAllocator >
   template< typename Function >
void
TridiagonalMatrix< Real, Device, Index, Organization, RealAllocator >::
sequentialForAllRows( Function& function ) const
{
   this->sequentialForRows( (IndexType) 0, this->getRows(), function );
}

template< typename Real,
          typename Device,
          typename Index,
          ElementsOrganization Organization,
          typename RealAllocator >
   template< typename Function >
void
TridiagonalMatrix< Real, Device, Index, Organization, RealAllocator >::
sequentialForAllRows( Function& function )
{
   this->sequentialForRows( (IndexType) 0, this->getRows(), function );
}

template< typename Real,
          typename Device,
          typename Index,
          ElementsOrganization Organization,
          typename RealAllocator >
   template< typename InVector, typename OutVector >
void
TridiagonalMatrix< Real, Device, Index, Organization, RealAllocator >::
vectorProduct( const InVector& inVector,
               OutVector& outVector,
               const RealType matrixMultiplicator,
               const RealType outVectorMultiplicator,
               const IndexType begin,
               IndexType end ) const
{
   this->view.vectorProduct( inVector, outVector, matrixMultiplicator, outVectorMultiplicator, begin, end );
}

template< typename Real,
          typename Device,
          typename Index,
          ElementsOrganization Organization,
          typename RealAllocator >
   template< typename Real_, typename Device_, typename Index_, ElementsOrganization Organization_, typename RealAllocator_ >
void
TridiagonalMatrix< Real, Device, Index, Organization, RealAllocator >::
addMatrix( const TridiagonalMatrix< Real_, Device_, Index_, Organization_, RealAllocator_ >& matrix,
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
__global__ void TridiagonalMatrixTranspositionCudaKernel( const TridiagonalMatrix< Real2, Devices::Cuda, Index2 >* inMatrix,
                                                             TridiagonalMatrix< Real, Devices::Cuda, Index >* outMatrix,
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
          typename RealAllocator >
   template< typename Real2, typename Index2 >
void TridiagonalMatrix< Real, Device, Index, Organization, RealAllocator >::getTransposition( const TridiagonalMatrix< Real2, Device, Index2 >& matrix,
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
      TridiagonalMatrix* kernel_this = Cuda::passToDevice( *this );
      typedef  TridiagonalMatrix< Real2, Device, Index2 > InMatrixType;
      InMatrixType* kernel_inMatrix = Cuda::passToDevice( matrix );
      dim3 cudaBlockSize( 256 ), cudaGridSize( Cuda::getMaxGridSize() );
      const IndexType cudaBlocks = roundUpDivision( matrix.getRows(), cudaBlockSize.x );
      const IndexType cudaGrids = roundUpDivision( cudaBlocks, Cuda::getMaxGridSize() );
      for( IndexType gridIdx = 0; gridIdx < cudaGrids; gridIdx++ )
      {
         if( gridIdx == cudaGrids - 1 )
            cudaGridSize.x = cudaBlocks % Cuda::getMaxGridSize();
         TridiagonalMatrixTranspositionCudaKernel<<< cudaGridSize, cudaBlockSize >>>
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
          typename RealAllocator >
TridiagonalMatrix< Real, Device, Index, Organization, RealAllocator >&
TridiagonalMatrix< Real, Device, Index, Organization, RealAllocator >::operator=( const TridiagonalMatrix& matrix )
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
          typename RealAllocator >
   template< typename Real_, typename Device_, typename Index_, ElementsOrganization Organization_, typename RealAllocator_ >
TridiagonalMatrix< Real, Device, Index, Organization, RealAllocator >&
TridiagonalMatrix< Real, Device, Index, Organization, RealAllocator >::
operator=( const TridiagonalMatrix< Real_, Device_, Index_, Organization_, RealAllocator_ >& matrix )
{
   static_assert( std::is_same< Device, Devices::Host >::value || std::is_same< Device, Devices::Cuda >::value,
                  "unknown device" );
   static_assert( std::is_same< Device_, Devices::Host >::value || std::is_same< Device_, Devices::Cuda >::value,
                  "unknown device" );

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
         TridiagonalMatrix< Real, Device, Index, Organization_ > auxMatrix;
         auxMatrix = matrix;
         const auto matrix_view = auxMatrix.getView();
         auto f = [=] __cuda_callable__ ( const IndexType& rowIdx, const IndexType& localIdx, const IndexType& column, Real& value ) mutable {
            value = matrix_view.getValues()[ matrix_view.getIndexer().getGlobalIndex( rowIdx, localIdx ) ];
         };
         this->forAllElements( f );
      }
   }
   return *this;
}

template< typename Real,
          typename Device,
          typename Index,
          ElementsOrganization Organization,
          typename RealAllocator >
void TridiagonalMatrix< Real, Device, Index, Organization, RealAllocator >::save( File& file ) const
{
   Matrix< Real, Device, Index >::save( file );
}

template< typename Real,
          typename Device,
          typename Index,
          ElementsOrganization Organization,
          typename RealAllocator >
void TridiagonalMatrix< Real, Device, Index, Organization, RealAllocator >::load( File& file )
{
   Matrix< Real, Device, Index >::load( file );
   this->indexer.setDimensions( this->getRows(), this->getColumns() );
   this->view = this->getView();
}

template< typename Real,
          typename Device,
          typename Index,
          ElementsOrganization Organization,
          typename RealAllocator >
void TridiagonalMatrix< Real, Device, Index, Organization, RealAllocator >::save( const String& fileName ) const
{
   Object::save( fileName );
}

template< typename Real,
          typename Device,
          typename Index,
          ElementsOrganization Organization,
          typename RealAllocator >
void TridiagonalMatrix< Real, Device, Index, Organization, RealAllocator >::load( const String& fileName )
{
   Object::load( fileName );
}

template< typename Real,
          typename Device,
          typename Index,
          ElementsOrganization Organization,
          typename RealAllocator >
void
TridiagonalMatrix< Real, Device, Index, Organization, RealAllocator >::
print( std::ostream& str ) const
{
   this->view.print( str );
}

template< typename Real,
          typename Device,
          typename Index,
          ElementsOrganization Organization,
          typename RealAllocator >
auto
TridiagonalMatrix< Real, Device, Index, Organization, RealAllocator >::
getIndexer() const -> const IndexerType&
{
   return this->indexer;
}

template< typename Real,
          typename Device,
          typename Index,
          ElementsOrganization Organization,
          typename RealAllocator >
auto
TridiagonalMatrix< Real, Device, Index, Organization, RealAllocator >::
getIndexer() -> IndexerType&
{
   return this->indexer;
}

template< typename Real,
          typename Device,
          typename Index,
          ElementsOrganization Organization,
          typename RealAllocator >
__cuda_callable__
Index
TridiagonalMatrix< Real, Device, Index, Organization, RealAllocator >::
getElementIndex( const IndexType row, const IndexType column ) const
{
   IndexType localIdx = column - row + 1;

   TNL_ASSERT_GE( localIdx, 0, "" );
   TNL_ASSERT_LT( localIdx, 3, "" );

   return this->indexer.getGlobalIndex( row, localIdx );
}

template< typename Real,
          typename Device,
          typename Index,
          ElementsOrganization Organization,
          typename RealAllocator >
__cuda_callable__
Index
TridiagonalMatrix< Real, Device, Index, Organization, RealAllocator >::
getPaddingIndex() const
{
   return this->view.getPaddingIndex();
}

} // namespace Matrices
} // namespace noa::TNL
