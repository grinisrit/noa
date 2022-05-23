// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <iomanip>
#include <noa/3rdparty/tnl-noa/src/TNL/Assert.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Matrices/TridiagonalMatrixView.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Exceptions/NotImplementedError.h>

namespace noa::TNL {
namespace Matrices {

template< typename Real, typename Device, typename Index, ElementsOrganization Organization >
TridiagonalMatrixView< Real, Device, Index, Organization >::TridiagonalMatrixView() = default;

template< typename Real, typename Device, typename Index, ElementsOrganization Organization >
TridiagonalMatrixView< Real, Device, Index, Organization >::TridiagonalMatrixView( const ValuesViewType& values,
                                                                                   const IndexerType& indexer )
: MatrixView< Real, Device, Index >( indexer.getRows(), indexer.getColumns(), values ), indexer( indexer )
{}

template< typename Real, typename Device, typename Index, ElementsOrganization Organization >
auto
TridiagonalMatrixView< Real, Device, Index, Organization >::getView() -> ViewType
{
   return ViewType( this->values.getView(), indexer );
}

template< typename Real, typename Device, typename Index, ElementsOrganization Organization >
auto
TridiagonalMatrixView< Real, Device, Index, Organization >::getConstView() const -> ConstViewType
{
   return ConstViewType( this->values.getConstView(), indexer );
}

template< typename Real, typename Device, typename Index, ElementsOrganization Organization >
std::string
TridiagonalMatrixView< Real, Device, Index, Organization >::getSerializationType()
{
   return "Matrices::TridiagonalMatrix< " + TNL::getSerializationType< RealType >() + ", [any_device], "
        + TNL::getSerializationType< IndexType >() + ", " + TNL::getSerializationType( Organization ) + ", [any_allocator] >";
}

template< typename Real, typename Device, typename Index, ElementsOrganization Organization >
std::string
TridiagonalMatrixView< Real, Device, Index, Organization >::getSerializationTypeVirtual() const
{
   return this->getSerializationType();
}

template< typename Real, typename Device, typename Index, ElementsOrganization Organization >
template< typename Vector >
void
TridiagonalMatrixView< Real, Device, Index, Organization >::getRowCapacities( Vector& rowCapacities ) const
{
   rowCapacities.setSize( this->getRows() );
   rowCapacities = 3;
}

template< typename Real, typename Device, typename Index, ElementsOrganization Organization >
template< typename Vector >
void
TridiagonalMatrixView< Real, Device, Index, Organization >::getCompressedRowLengths( Vector& rowLengths ) const
{
   rowLengths.setSize( this->getRows() );
   rowLengths = 0;
   auto rowLengths_view = rowLengths.getView();
   auto fetch = [] __cuda_callable__( IndexType row, IndexType column, const RealType& value ) -> IndexType
   {
      return ( value != 0.0 );
   };
   auto reduce = [] __cuda_callable__( const IndexType& aux, const IndexType a ) -> IndexType
   {
      return aux + a;
   };
   auto keep = [ = ] __cuda_callable__( const IndexType rowIdx, const IndexType value ) mutable
   {
      rowLengths_view[ rowIdx ] = value;
   };
   this->reduceAllRows( fetch, reduce, keep, 0 );
}

template< typename Real, typename Device, typename Index, ElementsOrganization Organization >
Index
TridiagonalMatrixView< Real, Device, Index, Organization >::getNonzeroElementsCount() const
{
   const auto values_view = this->values.getConstView();
   auto fetch = [ = ] __cuda_callable__( const IndexType i ) -> IndexType
   {
      return ( values_view[ i ] != 0.0 );
   };
   return Algorithms::reduce< DeviceType >( (IndexType) 0, this->values.getSize(), fetch, std::plus<>{}, 0 );
}

template< typename Real, typename Device, typename Index, ElementsOrganization Organization >
template< typename Real_, typename Device_, typename Index_, ElementsOrganization Organization_ >
bool
TridiagonalMatrixView< Real, Device, Index, Organization >::operator==(
   const TridiagonalMatrixView< Real_, Device_, Index_, Organization_ >& matrix ) const
{
   if( Organization == Organization_ )
      return this->values == matrix.values;
   else {
      TNL_ASSERT_TRUE( false, "TODO" );
   }
}

template< typename Real, typename Device, typename Index, ElementsOrganization Organization >
template< typename Real_, typename Device_, typename Index_, ElementsOrganization Organization_ >
bool
TridiagonalMatrixView< Real, Device, Index, Organization >::operator!=(
   const TridiagonalMatrixView< Real_, Device_, Index_, Organization_ >& matrix ) const
{
   return ! this->operator==( matrix );
}

template< typename Real, typename Device, typename Index, ElementsOrganization Organization >
void
TridiagonalMatrixView< Real, Device, Index, Organization >::setValue( const RealType& v )
{
   this->values = v;
}

template< typename Real, typename Device, typename Index, ElementsOrganization Organization >
__cuda_callable__
auto
TridiagonalMatrixView< Real, Device, Index, Organization >::getRow( const IndexType& rowIdx ) const -> const ConstRowView
{
   return ConstRowView( rowIdx, this->values.getView(), this->indexer );
}

template< typename Real, typename Device, typename Index, ElementsOrganization Organization >
__cuda_callable__
auto
TridiagonalMatrixView< Real, Device, Index, Organization >::getRow( const IndexType& rowIdx ) -> RowView
{
   return RowView( rowIdx, this->values.getView(), this->indexer );
}

template< typename Real, typename Device, typename Index, ElementsOrganization Organization >
__cuda_callable__
void
TridiagonalMatrixView< Real, Device, Index, Organization >::setElement( const IndexType row,
                                                                        const IndexType column,
                                                                        const RealType& value )
{
   TNL_ASSERT_GE( row, 0, "" );
   TNL_ASSERT_LT( row, this->getRows(), "" );
   TNL_ASSERT_GE( column, 0, "" );
   TNL_ASSERT_LT( column, this->getColumns(), "" );
   if( abs( row - column ) > 1 ) {
#ifdef __CUDA_ARCH__
      TNL_ASSERT_TRUE( false, "Wrong matrix element coordinates tridiagonal matrix." );
#else
      std::stringstream msg;
      msg << "Wrong matrix element coordinates ( " << row << ", " << column << " ) in tridiagonal matrix.";
      throw std::logic_error( msg.str() );
#endif
   }
   this->values.setElement( this->getElementIndex( row, column ), value );
}

template< typename Real, typename Device, typename Index, ElementsOrganization Organization >
__cuda_callable__
void
TridiagonalMatrixView< Real, Device, Index, Organization >::addElement( const IndexType row,
                                                                        const IndexType column,
                                                                        const RealType& value,
                                                                        const RealType& thisElementMultiplicator )
{
   TNL_ASSERT_GE( row, 0, "" );
   TNL_ASSERT_LT( row, this->getRows(), "" );
   TNL_ASSERT_GE( column, 0, "" );
   TNL_ASSERT_LT( column, this->getColumns(), "" );
   if( abs( row - column ) > 1 ) {
#ifdef __CUDA_ARCH__
      TNL_ASSERT_TRUE( false, "Wrong matrix element coordinates tridiagonal matrix." );
#else
      std::stringstream msg;
      msg << "Wrong matrix element coordinates ( " << row << ", " << column << " ) in tridiagonal matrix.";
      throw std::logic_error( msg.str() );
#endif
   }
   const Index i = this->getElementIndex( row, column );
   this->values.setElement( i, thisElementMultiplicator * this->values.getElement( i ) + value );
}

template< typename Real, typename Device, typename Index, ElementsOrganization Organization >
__cuda_callable__
Real
TridiagonalMatrixView< Real, Device, Index, Organization >::getElement( const IndexType row, const IndexType column ) const
{
   TNL_ASSERT_GE( row, 0, "" );
   TNL_ASSERT_LT( row, this->getRows(), "" );
   TNL_ASSERT_GE( column, 0, "" );
   TNL_ASSERT_LT( column, this->getColumns(), "" );

   if( abs( column - row ) > 1 )
      return 0.0;
   return this->values.getElement( this->getElementIndex( row, column ) );
}

template< typename Real, typename Device, typename Index, ElementsOrganization Organization >
template< typename Fetch, typename Reduce, typename Keep, typename FetchReal >
void
TridiagonalMatrixView< Real, Device, Index, Organization >::reduceRows( IndexType first,
                                                                        IndexType last,
                                                                        Fetch& fetch,
                                                                        Reduce& reduce,
                                                                        Keep& keep,
                                                                        const FetchReal& identity ) const
{
   using Real_ = decltype( fetch( IndexType(), IndexType(), RealType() ) );
   const auto values_view = this->values.getConstView();
   const auto indexer = this->indexer;
   auto f = [ = ] __cuda_callable__( IndexType rowIdx ) mutable
   {
      Real_ sum = identity;
      if( rowIdx == 0 ) {
         sum = reduce( sum, fetch( 0, 1, values_view[ indexer.getGlobalIndex( 0, 1 ) ] ) );
         sum = reduce( sum, fetch( 0, 2, values_view[ indexer.getGlobalIndex( 0, 2 ) ] ) );
         keep( 0, sum );
         return;
      }
      if( rowIdx + 1 < indexer.getColumns() ) {
         sum = reduce( sum, fetch( rowIdx, rowIdx - 1, values_view[ indexer.getGlobalIndex( rowIdx, 0 ) ] ) );
         sum = reduce( sum, fetch( rowIdx, rowIdx, values_view[ indexer.getGlobalIndex( rowIdx, 1 ) ] ) );
         sum = reduce( sum, fetch( rowIdx, rowIdx + 1, values_view[ indexer.getGlobalIndex( rowIdx, 2 ) ] ) );
         keep( rowIdx, sum );
         return;
      }
      if( rowIdx < indexer.getColumns() ) {
         sum = reduce( sum, fetch( rowIdx, rowIdx - 1, values_view[ indexer.getGlobalIndex( rowIdx, 0 ) ] ) );
         sum = reduce( sum, fetch( rowIdx, rowIdx, values_view[ indexer.getGlobalIndex( rowIdx, 1 ) ] ) );
         keep( rowIdx, sum );
      }
      else {
         keep( rowIdx, fetch( rowIdx, rowIdx - 1, values_view[ indexer.getGlobalIndex( rowIdx, 0 ) ] ) );
      }
   };
   Algorithms::ParallelFor< DeviceType >::exec( first, last, f );
}

template< typename Real, typename Device, typename Index, ElementsOrganization Organization >
template< typename Fetch, typename Reduce, typename Keep, typename FetchReal >
void
TridiagonalMatrixView< Real, Device, Index, Organization >::reduceRows( IndexType first,
                                                                        IndexType last,
                                                                        Fetch& fetch,
                                                                        Reduce& reduce,
                                                                        Keep& keep,
                                                                        const FetchReal& identity )
{
   using Real_ = decltype( fetch( IndexType(), IndexType(), RealType() ) );
   auto values_view = this->values.getConstView();
   const auto indexer = this->indexer;
   auto f = [ = ] __cuda_callable__( IndexType rowIdx ) mutable
   {
      Real_ sum = identity;
      if( rowIdx == 0 ) {
         sum = reduce( sum, fetch( 0, 1, values_view[ indexer.getGlobalIndex( 0, 1 ) ] ) );
         sum = reduce( sum, fetch( 0, 2, values_view[ indexer.getGlobalIndex( 0, 2 ) ] ) );
         keep( 0, sum );
         return;
      }
      if( rowIdx + 1 < indexer.getColumns() ) {
         sum = reduce( sum, fetch( rowIdx, rowIdx - 1, values_view[ indexer.getGlobalIndex( rowIdx, 0 ) ] ) );
         sum = reduce( sum, fetch( rowIdx, rowIdx, values_view[ indexer.getGlobalIndex( rowIdx, 1 ) ] ) );
         sum = reduce( sum, fetch( rowIdx, rowIdx + 1, values_view[ indexer.getGlobalIndex( rowIdx, 2 ) ] ) );
         keep( rowIdx, sum );
         return;
      }
      if( rowIdx < indexer.getColumns() ) {
         sum = reduce( sum, fetch( rowIdx, rowIdx - 1, values_view[ indexer.getGlobalIndex( rowIdx, 0 ) ] ) );
         sum = reduce( sum, fetch( rowIdx, rowIdx, values_view[ indexer.getGlobalIndex( rowIdx, 1 ) ] ) );
         keep( rowIdx, sum );
      }
      else {
         keep( rowIdx, fetch( rowIdx, rowIdx - 1, values_view[ indexer.getGlobalIndex( rowIdx, 0 ) ] ) );
      }
   };
   Algorithms::ParallelFor< DeviceType >::exec( first, last, f );
}

template< typename Real, typename Device, typename Index, ElementsOrganization Organization >
template< typename Fetch, typename Reduce, typename Keep, typename FetchReal >
void
TridiagonalMatrixView< Real, Device, Index, Organization >::reduceAllRows( Fetch& fetch,
                                                                           Reduce& reduce,
                                                                           Keep& keep,
                                                                           const FetchReal& identity ) const
{
   this->reduceRows( (IndexType) 0, this->indexer.getNonemptyRowsCount(), fetch, reduce, keep, identity );
}

template< typename Real, typename Device, typename Index, ElementsOrganization Organization >
template< typename Fetch, typename Reduce, typename Keep, typename FetchReal >
void
TridiagonalMatrixView< Real, Device, Index, Organization >::reduceAllRows( Fetch& fetch,
                                                                           Reduce& reduce,
                                                                           Keep& keep,
                                                                           const FetchReal& identity )
{
   this->reduceRows( (IndexType) 0, this->indexer.getNonemptyRowsCount(), fetch, reduce, keep, identity );
}

template< typename Real, typename Device, typename Index, ElementsOrganization Organization >
template< typename Function >
void
TridiagonalMatrixView< Real, Device, Index, Organization >::forElements( IndexType first,
                                                                         IndexType last,
                                                                         Function& function ) const
{
   const auto values_view = this->values.getConstView();
   const auto indexer = this->indexer;
   auto f = [ = ] __cuda_callable__( IndexType rowIdx ) mutable
   {
      if( rowIdx == 0 ) {
         function( 0, 1, 0, values_view[ indexer.getGlobalIndex( 0, 1 ) ] );
         function( 0, 2, 1, values_view[ indexer.getGlobalIndex( 0, 2 ) ] );
      }
      else if( rowIdx + 1 < indexer.getColumns() ) {
         function( rowIdx, 0, rowIdx - 1, values_view[ indexer.getGlobalIndex( rowIdx, 0 ) ] );
         function( rowIdx, 1, rowIdx, values_view[ indexer.getGlobalIndex( rowIdx, 1 ) ] );
         function( rowIdx, 2, rowIdx + 1, values_view[ indexer.getGlobalIndex( rowIdx, 2 ) ] );
      }
      else if( rowIdx < indexer.getColumns() ) {
         function( rowIdx, 0, rowIdx - 1, values_view[ indexer.getGlobalIndex( rowIdx, 0 ) ] );
         function( rowIdx, 1, rowIdx, values_view[ indexer.getGlobalIndex( rowIdx, 1 ) ] );
      }
      else
         function( rowIdx, 0, rowIdx, values_view[ indexer.getGlobalIndex( rowIdx, 0 ) ] );
   };
   Algorithms::ParallelFor< DeviceType >::exec( first, last, f );
}

template< typename Real, typename Device, typename Index, ElementsOrganization Organization >
template< typename Function >
void
TridiagonalMatrixView< Real, Device, Index, Organization >::forElements( IndexType first, IndexType last, Function& function )
{
   auto values_view = this->values.getView();
   const auto indexer = this->indexer;
   auto f = [ = ] __cuda_callable__( IndexType rowIdx ) mutable
   {
      if( rowIdx == 0 ) {
         function( 0, 1, 0, values_view[ indexer.getGlobalIndex( 0, 1 ) ] );
         function( 0, 2, 1, values_view[ indexer.getGlobalIndex( 0, 2 ) ] );
      }
      else if( rowIdx + 1 < indexer.getColumns() ) {
         function( rowIdx, 0, rowIdx - 1, values_view[ indexer.getGlobalIndex( rowIdx, 0 ) ] );
         function( rowIdx, 1, rowIdx, values_view[ indexer.getGlobalIndex( rowIdx, 1 ) ] );
         function( rowIdx, 2, rowIdx + 1, values_view[ indexer.getGlobalIndex( rowIdx, 2 ) ] );
      }
      else if( rowIdx < indexer.getColumns() ) {
         function( rowIdx, 0, rowIdx - 1, values_view[ indexer.getGlobalIndex( rowIdx, 0 ) ] );
         function( rowIdx, 1, rowIdx, values_view[ indexer.getGlobalIndex( rowIdx, 1 ) ] );
      }
      else
         function( rowIdx, 0, rowIdx, values_view[ indexer.getGlobalIndex( rowIdx, 0 ) ] );
   };
   Algorithms::ParallelFor< DeviceType >::exec( first, last, f );
}

template< typename Real, typename Device, typename Index, ElementsOrganization Organization >
template< typename Function >
void
TridiagonalMatrixView< Real, Device, Index, Organization >::forAllElements( Function& function ) const
{
   this->forElements( (IndexType) 0, this->indxer.getNonEmptyRowsCount(), function );
}

template< typename Real, typename Device, typename Index, ElementsOrganization Organization >
template< typename Function >
void
TridiagonalMatrixView< Real, Device, Index, Organization >::forAllElements( Function& function )
{
   this->forElements( (IndexType) 0, this->indexer.getNonemptyRowsCount(), function );
}

template< typename Real, typename Device, typename Index, ElementsOrganization Organization >
template< typename Function >
void
TridiagonalMatrixView< Real, Device, Index, Organization >::forRows( IndexType begin, IndexType end, Function&& function )
{
   auto view = *this;
   auto f = [ = ] __cuda_callable__( const IndexType rowIdx ) mutable
   {
      auto rowView = view.getRow( rowIdx );
      function( rowView );
   };
   TNL::Algorithms::ParallelFor< DeviceType >::exec( begin, end, f );
}

template< typename Real, typename Device, typename Index, ElementsOrganization Organization >
template< typename Function >
void
TridiagonalMatrixView< Real, Device, Index, Organization >::forRows( IndexType begin, IndexType end, Function&& function ) const
{
   auto view = *this;
   auto f = [ = ] __cuda_callable__( const IndexType rowIdx ) mutable
   {
      auto rowView = view.getRow( rowIdx );
      function( rowView );
   };
   TNL::Algorithms::ParallelFor< DeviceType >::exec( begin, end, f );
}

template< typename Real, typename Device, typename Index, ElementsOrganization Organization >
template< typename Function >
void
TridiagonalMatrixView< Real, Device, Index, Organization >::forAllRows( Function&& function )
{
   this->forRows( (IndexType) 0, this->getRows(), function );
}

template< typename Real, typename Device, typename Index, ElementsOrganization Organization >
template< typename Function >
void
TridiagonalMatrixView< Real, Device, Index, Organization >::forAllRows( Function&& function ) const
{
   this->forRows( (IndexType) 0, this->getRows(), function );
}

template< typename Real, typename Device, typename Index, ElementsOrganization Organization >
template< typename Function >
void
TridiagonalMatrixView< Real, Device, Index, Organization >::sequentialForRows( IndexType begin,
                                                                               IndexType end,
                                                                               Function& function ) const
{
   for( IndexType row = begin; row < end; row++ )
      this->forElements( row, row + 1, function );
}

template< typename Real, typename Device, typename Index, ElementsOrganization Organization >
template< typename Function >
void
TridiagonalMatrixView< Real, Device, Index, Organization >::sequentialForRows( IndexType begin,
                                                                               IndexType end,
                                                                               Function& function )
{
   for( IndexType row = begin; row < end; row++ )
      this->forElements( row, row + 1, function );
}

template< typename Real, typename Device, typename Index, ElementsOrganization Organization >
template< typename Function >
void
TridiagonalMatrixView< Real, Device, Index, Organization >::sequentialForAllRows( Function& function ) const
{
   this->sequentialForRows( (IndexType) 0, this->getRows(), function );
}

template< typename Real, typename Device, typename Index, ElementsOrganization Organization >
template< typename Function >
void
TridiagonalMatrixView< Real, Device, Index, Organization >::sequentialForAllRows( Function& function )
{
   this->sequentialForRows( (IndexType) 0, this->getRows(), function );
}

template< typename Real, typename Device, typename Index, ElementsOrganization Organization >
template< typename InVector, typename OutVector >
void
TridiagonalMatrixView< Real, Device, Index, Organization >::vectorProduct( const InVector& inVector,
                                                                           OutVector& outVector,
                                                                           RealType matrixMultiplicator,
                                                                           RealType outVectorMultiplicator,
                                                                           IndexType begin,
                                                                           IndexType end ) const
{
   TNL_ASSERT_EQ( this->getColumns(), inVector.getSize(), "Matrix columns do not fit with input vector." );
   TNL_ASSERT_EQ( this->getRows(), outVector.getSize(), "Matrix rows do not fit with output vector." );

   const auto inVectorView = inVector.getConstView();
   auto outVectorView = outVector.getView();
   auto fetch = [ = ] __cuda_callable__( const IndexType& row, const IndexType& column, const RealType& value ) -> RealType
   {
      return value * inVectorView[ column ];
   };
   auto reduction = [] __cuda_callable__( const RealType& sum, const RealType& value ) -> RealType
   {
      return sum + value;
   };
   auto keeper1 = [ = ] __cuda_callable__( IndexType row, const RealType& value ) mutable
   {
      outVectorView[ row ] = value;
   };
   auto keeper2 = [ = ] __cuda_callable__( IndexType row, const RealType& value ) mutable
   {
      outVectorView[ row ] = outVectorMultiplicator * outVectorView[ row ] + matrixMultiplicator * value;
   };
   if( end == 0 )
      end = this->getRows();
   if( matrixMultiplicator == 1.0 && outVectorMultiplicator == 0.0 )
      this->reduceRows( begin, end, fetch, reduction, keeper1, (RealType) 0.0 );
   else
      this->reduceRows( begin, end, fetch, reduction, keeper2, (RealType) 0.0 );
}

template< typename Real, typename Device, typename Index, ElementsOrganization Organization >
TridiagonalMatrixView< Real, Device, Index, Organization >&
TridiagonalMatrixView< Real, Device, Index, Organization >::operator=( const TridiagonalMatrixView& view )
{
   MatrixView< Real, Device, Index >::operator=( view );
   this->indexer = view.indexer;
   return *this;
}

template< typename Real, typename Device, typename Index, ElementsOrganization Organization >
template< typename Real_, typename Device_, typename Index_, ElementsOrganization Organization_ >
void
TridiagonalMatrixView< Real, Device, Index, Organization >::addMatrix(
   const TridiagonalMatrixView< Real_, Device_, Index_, Organization_ >& matrix,
   const RealType& matrixMultiplicator,
   const RealType& thisMatrixMultiplicator )
{
   TNL_ASSERT_EQ( this->getRows(), matrix.getRows(), "Matrices rows are not equal." );
   TNL_ASSERT_EQ( this->getColumns(), matrix.getColumns(), "Matrices columns are not equal." );

   if( Organization == Organization_ ) {
      if( thisMatrixMultiplicator == 1.0 )
         this->values += matrixMultiplicator * matrix.getValues();
      else
         this->values = thisMatrixMultiplicator * this->values + matrixMultiplicator * matrix.getValues();
   }
   else {
      const auto matrix_view = matrix;
      const auto matrixMult = matrixMultiplicator;
      const auto thisMult = thisMatrixMultiplicator;
      auto add0 = [ = ] __cuda_callable__(
                     const IndexType& rowIdx, const IndexType& localIdx, const IndexType& column, Real& value ) mutable
      {
         value = matrixMult * matrix.getValues()[ matrix.getIndexer().getGlobalIndex( rowIdx, localIdx ) ];
      };
      auto add1 = [ = ] __cuda_callable__(
                     const IndexType& rowIdx, const IndexType& localIdx, const IndexType& column, Real& value ) mutable
      {
         value += matrixMult * matrix.getValues()[ matrix.getIndexer().getGlobalIndex( rowIdx, localIdx ) ];
      };
      auto addGen = [ = ] __cuda_callable__(
                       const IndexType& rowIdx, const IndexType& localIdx, const IndexType& column, Real& value ) mutable
      {
         value = thisMult * value + matrixMult * matrix.getValues()[ matrix.getIndexer().getGlobalIndex( rowIdx, localIdx ) ];
      };
      if( thisMult == 0.0 )
         this->forAllElements( add0 );
      else if( thisMult == 1.0 )
         this->forAllElements( add1 );
      else
         this->forAllElements( addGen );
   }
}

#ifdef HAVE_CUDA
/*template< typename Real,
          typename Real2,
          typename Index,
          typename Index2 >
__global__ void TridiagonalTranspositionCudaKernel( const Tridiagonal< Real2, Devices::Cuda, Index2 >* inMatrix,
                                                             Tridiagonal< Real, Devices::Cuda, Index >* outMatrix,
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
}*/
#endif

template< typename Real, typename Device, typename Index, ElementsOrganization Organization >
template< typename Real2, typename Index2 >
void
TridiagonalMatrixView< Real, Device, Index, Organization >::getTransposition(
   const TridiagonalMatrixView< Real2, Device, Index2 >& matrix,
   const RealType& matrixMultiplicator )
{
   TNL_ASSERT( this->getRows() == matrix.getRows(),
               std::cerr << "This matrix rows: " << this->getRows() << std::endl
                         << "That matrix rows: " << matrix.getRows() << std::endl );
   if( std::is_same< Device, Devices::Host >::value ) {
      const IndexType& rows = matrix.getRows();
      for( IndexType i = 1; i < rows; i++ ) {
         RealType aux = matrix.getElement( i, i - 1 );
         this->setElement( i, i - 1, matrix.getElement( i - 1, i ) );
         this->setElement( i, i, matrix.getElement( i, i ) );
         this->setElement( i - 1, i, aux );
      }
   }
   if( std::is_same< Device, Devices::Cuda >::value ) {
#ifdef HAVE_CUDA
      /*Tridiagonal* kernel_this = Cuda::passToDevice( *this );
      typedef  Tridiagonal< Real2, Device, Index2 > InMatrixType;
      InMatrixType* kernel_inMatrix = Cuda::passToDevice( matrix );
      dim3 cudaBlockSize( 256 ), cudaGridSize( Cuda::getMaxGridSize() );
      const IndexType cudaBlocks = roundUpDivision( matrix.getRows(), cudaBlockSize.x );
      const IndexType cudaGrids = roundUpDivision( cudaBlocks, Cuda::getMaxGridSize() );
      for( IndexType gridIdx = 0; gridIdx < cudaGrids; gridIdx++ )
      {
         if( gridIdx == cudaGrids - 1 )
            cudaGridSize.x = cudaBlocks % Cuda::getMaxGridSize();
         TridiagonalTranspositionCudaKernel<<< cudaGridSize, cudaBlockSize >>>
                                                    ( kernel_inMatrix,
                                                      kernel_this,
                                                      matrixMultiplicator,
                                                      gridIdx );
      }
      Cuda::freeFromDevice( kernel_this );
      Cuda::freeFromDevice( kernel_inMatrix );
      TNL_CHECK_CUDA_DEVICE;*/
#endif
   }
}

template< typename Real, typename Device, typename Index, ElementsOrganization Organization >
void
TridiagonalMatrixView< Real, Device, Index, Organization >::save( File& file ) const
{
   MatrixView< Real, Device, Index >::save( file );
}

template< typename Real, typename Device, typename Index, ElementsOrganization Organization >
void
TridiagonalMatrixView< Real, Device, Index, Organization >::save( const String& fileName ) const
{
   Object::save( fileName );
}

template< typename Real, typename Device, typename Index, ElementsOrganization Organization >
void
TridiagonalMatrixView< Real, Device, Index, Organization >::print( std::ostream& str ) const
{
   for( IndexType row = 0; row < this->getRows(); row++ ) {
      str << "Row: " << row << " -> ";
      for( IndexType column = row - 1; column < row + 2; column++ )
         if( column >= 0 && column < this->columns ) {
            auto value = this->getElement( row, column );
            if( value ) {
               std::stringstream str_;
               str_ << std::setw( 4 ) << std::right << column << ":" << std::setw( 4 ) << std::left << value;
               str << std::setw( 10 ) << str_.str();
            }
         }
      str << std::endl;
   }
}

template< typename Real, typename Device, typename Index, ElementsOrganization Organization >
__cuda_callable__
auto
TridiagonalMatrixView< Real, Device, Index, Organization >::getIndexer() const -> const IndexerType&
{
   return this->indexer;
}

template< typename Real, typename Device, typename Index, ElementsOrganization Organization >
__cuda_callable__
auto
TridiagonalMatrixView< Real, Device, Index, Organization >::getIndexer() -> IndexerType&
{
   return this->indexer;
}

template< typename Real, typename Device, typename Index, ElementsOrganization Organization >
__cuda_callable__
Index
TridiagonalMatrixView< Real, Device, Index, Organization >::getElementIndex( const IndexType row, const IndexType column ) const
{
   IndexType localIdx = column - row + 1;

   TNL_ASSERT_GE( localIdx, 0, "" );
   TNL_ASSERT_LT( localIdx, 3, "" );

   return this->indexer.getGlobalIndex( row, localIdx );
}

template< typename Real, typename Device, typename Index, ElementsOrganization Organization >
__cuda_callable__
Index
TridiagonalMatrixView< Real, Device, Index, Organization >::getPaddingIndex() const
{
   return -1;
}

}  // namespace Matrices
}  // namespace noa::TNL
