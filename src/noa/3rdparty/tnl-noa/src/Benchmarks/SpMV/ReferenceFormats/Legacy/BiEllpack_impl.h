#pragma once

#include <cstdio>

#include "BiEllpack.h"
#include <TNL/Containers/Vector.h>
#include <TNL/Algorithms/scan.h>
#include <TNL/Math.h>
#include "MemoryHelpers.h"

namespace TNL {
    namespace Benchmarks {
        namespace SpMV {
            namespace ReferenceFormats {
      			namespace Legacy {


template< typename Real,
          typename Device,
          typename Index>
__cuda_callable__
Index BiEllpack< Real, Device, Index >::power( const IndexType number,
							  const IndexType exponent ) const
{
    if( exponent >= 0 )
    {
        IndexType result = 1;
        for( IndexType i = 0; i < exponent; i++ )
            result *= number;
        return result;
    }
    return 0;
}

template< typename Real,
	  typename Device,
	  typename Index >
BiEllpack< Real, Device, Index >::BiEllpack()
: warpSize( 32 ),
  logWarpSize( 5 )
{}

template< typename Real,
	  typename Device,
	  typename Index >
void
BiEllpack< Real, Device, Index >::
setDimensions( const IndexType rows, const IndexType columns )
{
   TNL_ASSERT( rows >= 0 && columns >= 0, std::cerr << "rows = " << rows << "columns = " << columns << std::endl );

   if( this->getRows() % this->warpSize != 0 )
      this->setVirtualRows( this->getRows() + this->warpSize - ( this->getRows() % this->warpSize ) );
   else
      this->setVirtualRows( this->getRows() );
   IndexType strips = this->virtualRows / this->warpSize;

   Sparse< Real, Device, Index >::setDimensions( rows, columns );
   this->rowPermArray.setSize( this->rows );
   this->groupPointers.setSize( strips * ( this->logWarpSize + 1 ) + 1 );

   for( IndexType row = 0; row < this->getRows(); row++ )
      this->rowPermArray.setElement(row, row);
}

template< typename Real,
	  typename Device,
	  typename Index >
void
BiEllpack< Real, Device, Index >::
setCompressedRowLengths( ConstRowsCapacitiesTypeView constRowLengths )
{
    RowsCapacitiesType rowLengths;
    rowLengths.reset();
    rowLengths.setLike( constRowLengths );

    rowLengths = constRowLengths;

    if( this->getRows() % this->warpSize != 0 )
            this->setVirtualRows( this->getRows() + this->warpSize - ( this->getRows() % this->warpSize ) );
    else
            this->setVirtualRows( this->getRows() );
    IndexType strips = this->virtualRows / this->warpSize;
    this->rowPermArray.setSize( this->rows );
    this->groupPointers.setSize( strips * ( this->logWarpSize + 1 ) + 1 );

    this->groupPointers.setValue( 0 );

    DeviceDependentCode::performRowBubbleSort( *this, rowLengths );
    DeviceDependentCode::computeColumnSizes( *this, rowLengths );

    Algorithms::inplaceExclusiveScan( this->groupPointers );

    DeviceDependentCode::verifyRowPerm( *this, rowLengths );
    DeviceDependentCode::verifyRowLengths( *this, rowLengths );

    return this->allocateMatrixElements( this->warpSize * this->groupPointers.getElement( strips * ( this->logWarpSize + 1 ) ) );
}

template< typename Real,
	  typename Device,
	  typename Index >
void
BiEllpack< Real, Device, Index >::
setRowCapacities( ConstRowsCapacitiesTypeView constRowLengths )
{
   setCompressedRowLengths( constRowLengths );
}

template< typename Real,
          typename Device,
          typename Index >
void BiEllpack< Real, Device, Index >::getCompressedRowLengths( RowsCapacitiesTypeView rowLengths ) const
{
   TNL_ASSERT_EQ( rowLengths.getSize(), this->getRows(), "invalid size of the rowLengths vector" );
   for( IndexType row = 0; row < this->getRows(); row++ )
      rowLengths.setElement( row, this->getRowLength( row ) );
}

template< typename Real,
          typename Device,
          typename Index >
__cuda_callable__
Index BiEllpack< Real, Device, Index >::getStripLength( const IndexType strip ) const
{
	TNL_ASSERT( strip >= 0, std::cerr << "strip = " << strip );

    return this->groupPointers.getElement( ( strip + 1 ) * ( this->logWarpSize + 1 ) )
           - this->groupPointers.getElement( strip * ( this->logWarpSize + 1 ) );
}

template< typename Real,
          typename Device,
          typename Index >
__cuda_callable__
Index BiEllpack< Real, Device, Index >::getNumberOfGroups( const IndexType row ) const
{
	TNL_ASSERT( row >= 0 && row < this->getRows(),
	            std::cerr <<  "row = " << row
                              << " this->getRows() = " << this->getRows() );

	IndexType strip = row / this->warpSize;
	IndexType rowStripPermutation = this->rowPermArray.getElement( row ) - this->warpSize * strip;
	IndexType numberOfGroups = this->logWarpSize + 1;
	IndexType bisection = 1;
	for( IndexType i = 0; i < this->logWarpSize + 1; i++ )
	{
		if( rowStripPermutation < bisection )
			return ( numberOfGroups - i );
		bisection *= 2;
	}
    // FIXME: non-void function always has to return something sensible
#ifndef __CUDA_ARCH__
    throw "bug - row was not found";
#else
    TNL_ASSERT_TRUE( false, "bug - row was not found" );
    return 0;
#endif
}

template< typename Real,
		  typename Device,
		  typename Index >
Index BiEllpack< Real, Device, Index >::getRowLength( const IndexType row ) const
{
	TNL_ASSERT( row >= 0 && row < this->getRows(),
                    std::cerr << "row = " << row << " this->getRows() = " << this->getRows() );

	const IndexType strip = row / this->warpSize;
	const IndexType groupBegin = strip * ( this->logWarpSize + 1 );
	const IndexType rowStripPerm = this->rowPermArray.getElement( row ) - strip * this->warpSize;
	IndexType elementPtr = this->groupPointers.getElement( groupBegin ) * this->warpSize + rowStripPerm;
	IndexType rowMultiplicator = 1;
	IndexType step = this->warpSize;
	IndexType rowLength = 0;

	for( IndexType group = 0; group < this->getNumberOfGroups( row ); group++ )
	{
		for( IndexType i = 0; i < rowMultiplicator * this->getGroupLength( strip, group ); i++ )
		{
			if( this->values.getElement( elementPtr ) == 0.0 )
				return rowLength;
			else
				rowLength++;
			elementPtr += step;
		}
		rowMultiplicator *= 2;
		step /= 2;
	}
	return rowLength;
}

template< typename Real,
		  typename Device,
		  typename Index >
	template< typename Real2,
			  typename Device2,
			  typename Index2 >
void BiEllpack< Real, Device, Index >::setLike( const BiEllpack< Real2, Device2, Index2 >& matrix )
{
	Sparse< Real, Device, Index >::setLike( matrix );
	this->rowPermArray.setLike( matrix.rowPermArray );
	this->groupPointers.setLike( matrix.groupPointers );
}

template< typename Real,
		  typename Device,
		  typename Index >
void BiEllpack< Real, Device, Index >::reset()
{
	Sparse< Real, Device, Index >::reset();
	this->rowPermArray.reset();
	this->groupPointers.reset();
}

template< typename Real,
          typename Device,
          typename Index >
   template< typename Real2,
             typename Device2,
             typename Index2 >
bool BiEllpack< Real, Device, Index >::operator == ( const BiEllpack< Real2, Device2, Index2 >& matrix ) const
{
   TNL_ASSERT( this->getRows() == matrix.getRows() &&
               this->getColumns() == matrix.getColumns(),
               std::cerr << "this->getRows() = " << this->getRows()
                    << " matrix.getRows() = " << matrix.getRows()
                    << " this->getColumns() = " << this->getColumns()
                    << " matrix.getColumns() = " << matrix.getColumns() );

   TNL_ASSERT_TRUE( false, "operator == is not yet implemented for BiEllpack.");

   // TODO: implement this
   return false;
}

template< typename Real,
          typename Device,
          typename Index >
   template< typename Real2,
             typename Device2,
             typename Index2 >
bool BiEllpack< Real, Device, Index >::operator != ( const BiEllpack< Real2, Device2, Index2 >& matrix ) const
{
   return ! ( ( *this ) == matrix );
}

template< typename Real,
		  typename Device,
		  typename Index >
void BiEllpack< Real, Device, Index >::getRowLengths( RowsCapacitiesType& rowLengths) const
{
    // WHAT IS THIS??!
    // It's called getRowLengths, but takes an argument that it fill up with this matrix's row lengths???
    for( IndexType row = 0; row < this->getRows(); row++ )
        rowLengths.setElement( row, this->getRowLength( row ) );
}

template< typename Real,
		  typename Device,
		  typename Index >
bool
BiEllpack< Real, Device, Index >::
setElement( const IndexType row,
            const IndexType column,
            const RealType& value )
{
    TNL_ASSERT( ( row >= 0 && row < this->getRows() ) ||
                        ( column >= 0 && column < this->getColumns() ),
                  std::cerr << "row = " << row
                       << " this->getRows() = " << this->getRows()
                       << " this->getColumns() = " << this->getColumns() );

    return this->addElement( row, column, value, 0.0 );
}

template< typename Real,
          typename Device,
          typename Index >
__cuda_callable__
bool BiEllpack< Real, Device, Index >::setElementFast( const IndexType row,
								  const IndexType column,
								  const RealType& value )
{
	TNL_ASSERT( ( row >= 0 && row < this->getRows() ) ||
			   ( column >= 0 && column < this->getColumns() ),
			     std::cerr << "row = " << row
			     	  << " this->getRows() = " << this->getRows()
			     	  << " this->getColumns() = " << this->getColumns() );

	return this->addElementFast( row, column, value, 0.0 );
}

template< typename Real,
		  typename Device,
		  typename Index >
bool BiEllpack< Real, Device, Index >::addElement( const IndexType row,
                                                              const IndexType column,
                                                              const RealType& value,
                                                              const RealType& thisElementMultiplicator )
{
    const IndexType strip = row / this->warpSize;
    const IndexType groupBegin = strip * ( this->logWarpSize + 1 );
    const IndexType rowStripPerm = this->rowPermArray.getElement( row ) - strip * this->warpSize;
    IndexType elementPtr = this->groupPointers.getElement( groupBegin ) * this->warpSize + rowStripPerm;
    IndexType rowMultiplicator = 1;
    IndexType step = this->warpSize;

    for( IndexType group = 0; group < this->getNumberOfGroups( row ); group++ )
    {
        for( IndexType i = 0; i < rowMultiplicator * this->getGroupLength( strip, group ); i++ )
        {
            if( this->columnIndexes.getElement( elementPtr ) == this->getPaddingIndex() )
            {
                this->columnIndexes.setElement( elementPtr, column );
                this->values.setElement( elementPtr, value );
                return true;
            }
            if( this->columnIndexes.getElement( elementPtr ) == column )
            {
                this->values.setElement( elementPtr, value + thisElementMultiplicator * this->values.getElement( elementPtr ) );
                return true;
            }
            elementPtr += step;
        }
        step /= 2;
        rowMultiplicator *= 2;
    }
    return false;
}

template< typename Real,
          typename Device,
          typename Index >
__cuda_callable__
bool BiEllpack< Real, Device, Index >::addElementFast( const IndexType row,
								  const IndexType column,
								  const RealType& value,
								  const RealType& thisElementMultiplicator )
{
    const IndexType strip = row / this->warpSize;
    const IndexType groupBegin = strip * ( this->logWarpSize + 1 );
    const IndexType rowStripPerm = this->rowPermArray[ row ] - strip * this->warpSize;
    IndexType elementPtr = this->groupPointers[ groupBegin ] * this->warpSize + rowStripPerm;
    IndexType rowMultiplicator = 1;
    IndexType step = this->warpSize;

    IndexType numberOfGroups = this->logWarpSize + 1;
    IndexType bisection = 1;
    for( IndexType i = 0; i < this->logWarpSize + 1; i++ )
    {
        if( rowStripPerm < bisection )
        {
            numberOfGroups -= i;
            break;
        }
        bisection *= 2;
    }

    for( IndexType group = 0; group < numberOfGroups; group++ )
    {
        IndexType groupLength = this->groupPointers[ strip * ( this->logWarpSize + 1 ) + group + 1 ]
                - this->groupPointers[ strip * ( this->logWarpSize + 1 ) + group ];

        for( IndexType i = 0; i < rowMultiplicator * groupLength; i++ )
        {
            if( this->columnIndexes[ elementPtr ] == this->getPaddingIndex() )
            {
                this->columnIndexes[ elementPtr ] = column ;
                this->values[ elementPtr ] = value;
                return true;
            }
            if( this->columnIndexes[ elementPtr ] == column )
            {
                this->values[ elementPtr ] = thisElementMultiplicator * this->values[ elementPtr ] + value ;
                return true;
            }
            elementPtr += step;
        }
        step /= 2;
        rowMultiplicator *= 2;
    }
    return false;
}

template< typename Real,
		  typename Device,
		  typename Index >
bool
BiEllpack< Real, Device, Index >::
setRow( const IndexType row,
	const IndexType* columns,
	const RealType* values,
	const IndexType numberOfElements )
{
	TNL_ASSERT( row >= 0 && row < this->getRows(),
                    std::cerr <<"row = " << row << " this->getRows() = " << this->getRows() );

	const IndexType strip = row / this->warpSize;
	const IndexType groupBegin = strip * ( this->logWarpSize + 1 );
	const IndexType rowStripPerm = this->rowPermArray.getElement( row ) - strip * this->warpSize;
	IndexType elementPtr = this->groupPointers.getElement( groupBegin ) * this->warpSize + rowStripPerm;
	IndexType thisElementPtr = 0;
	IndexType rowMultiplicator = 1;
	IndexType step = this->warpSize;

	for( IndexType group = 0; ( group < this->getNumberOfGroups( row ) ) && ( thisElementPtr < numberOfElements ); group++ )
	{
		for( IndexType i = 0; ( i <  rowMultiplicator * this->getGroupLength( strip, group ) ) && ( thisElementPtr < numberOfElements ); i++ )
		{
			this->columnIndexes.setElement( elementPtr, columns[ thisElementPtr ] );
			this->values.setElement( elementPtr, values[ thisElementPtr ] );
			thisElementPtr++;
			elementPtr += step;
		}
		step /= 2;
		rowMultiplicator *= 2;
	}
	if( thisElementPtr == numberOfElements )
		return true;
   return false;
}

template< typename Real,
		  typename Device,
		  typename Index >
bool
BiEllpack< Real, Device, Index >::
addRow( const IndexType row,
        const IndexType* columns,
        const RealType* values,
        const IndexType numberOfElements,
        const RealType& thisElementMultiplicator )
{
	TNL_ASSERT( row >=0 && row < this->getRows(),
	            std::cerr << "row = " << row << " this->getRows() = " << this->getRows() );

	const IndexType strip = row / this->warpSize;
	const IndexType groupBegin = strip * ( this->logWarpSize + 1 );
	const IndexType rowStripPerm = this->rowPermArray.getElement( row ) - this->warpSize * strip;
	IndexType elementPtr = this->groupPointers.getElement( groupBegin ) * this->warpSize + rowStripPerm;
	IndexType rowMultiplicator = 1;
	IndexType step = this->warpSize;
	IndexType thisElementPtr = 0;

	while( thisElementPtr < numberOfElements )
	{
		for( IndexType group = 0; group < this->getNumberOfGroups( row ); group++ )
		{
			for( IndexType i = 0; ( i < rowMultiplicator * this->getGroupLength( strip, group ) ) && ( thisElementPtr < numberOfElements ); i++ )
			{
				if( this->columnIndexes.getElement( elementPtr ) == columns[ thisElementPtr ] )
				{
					RealType result = this->values.getElement( elementPtr ) + values[ thisElementPtr ] * thisElementMultiplicator;
					this->values.setElement( elementPtr, result );
					thisElementPtr++;
				}
				elementPtr += step;
			}
			step /= 2;
			rowMultiplicator *= 2;
		}
	}
	return ( thisElementPtr == numberOfElements );
}

template< typename Real,
		  typename Device,
		  typename Index >
Real BiEllpack< Real, Device, Index >::getElement( const IndexType row,
                                                              const IndexType column ) const
{
	TNL_ASSERT( ( row >= 0 && row < this->getRows() ) ||
				( column >= 0 && column < this->getColumns() ),
				  std::cerr << "row = " << row
				  	   << " this->getRows() = " << this->getRows()
				  	   << " this->getColumns() = " << this->getColumns() );

	const IndexType strip = row / this->warpSize;
	const IndexType groupBegin = strip * ( this->logWarpSize + 1 );
	const IndexType rowStripPerm = this->rowPermArray.getElement( row ) - strip * this->warpSize;
	IndexType elementPtr = this->groupPointers.getElement( groupBegin ) * this->warpSize + rowStripPerm;
	IndexType rowMultiplicator = 1;
	IndexType step = this->warpSize;

	for( IndexType group = 0; group < this->getNumberOfGroups( row ); group++ )
	{
		for( IndexType i = 0; i < rowMultiplicator * this->getGroupLength( strip, group ); i++ )
		{
			if( this->columnIndexes.getElement( elementPtr ) == column )
				return this->values.getElement( elementPtr );
			elementPtr += step;
		}
		step /= 2;
		rowMultiplicator *= 2;
	}
	return 0.0;
}

template< typename Real,
          typename Device,
          typename Index >
__cuda_callable__
Real BiEllpack< Real, Device, Index >::getElementFast( const IndexType row,
								  const IndexType column ) const
{
    const IndexType strip = row / this->warpSize;
    const IndexType groupBegin = strip * ( this->logWarpSize + 1 );
    const IndexType rowStripPerm = this->rowPermArray[ row ] - strip * this->warpSize;
    IndexType elementPtr = this->groupPointers[ groupBegin ] * this->warpSize + rowStripPerm;
    IndexType rowMultiplicator = 1;
    IndexType step = this->warpSize;

    IndexType numberOfGroups = this->logWarpSize + 1;
    IndexType bisection = 1;
    for( IndexType i = 0; i < this->logWarpSize + 1; i++ )
    {
        if( rowStripPerm < bisection )
        {
            numberOfGroups -= i;
            break;
        }
        bisection *= 2;
    }

    for( IndexType group = 0; group < numberOfGroups; group++ )
    {
        IndexType groupLength = this->groupPointers[ strip * ( this->logWarpSize + 1 ) + group + 1 ]
                - this->groupPointers[ strip * ( this->logWarpSize + 1 ) + group ];

        for( IndexType i = 0; i < rowMultiplicator * groupLength; i++ )
        {
            if( this->columnIndexes[ elementPtr ] == column )
                return this->values[ elementPtr ];
            elementPtr += step;
        }
        step /= 2;
        rowMultiplicator *= 2;
    }
    return false;
}

template< typename Real,
		  typename Device,
		  typename Index >
void BiEllpack< Real, Device, Index >::getRow( const IndexType row,
							  IndexType* columns,
							  RealType* values ) const
{
	TNL_ASSERT( row >=0 && row < this->getRows(),
	              std::cerr << "row = " << row
	                   << " this->getRows() = " << this->getRows() );

	bool padding = false;
	const IndexType strip = row / this->warpSize;
	const IndexType groupBegin = strip * ( this->logWarpSize + 1 );
	const IndexType rowStripPerm = this->rowPermArray.getElement( row ) - this->warpSize * strip;
	IndexType elementPtr = this->groupPointers.getElement( groupBegin ) * this->warpSize + rowStripPerm;
	IndexType rowMultiplicator = 1;
	IndexType step = this->warpSize;
	IndexType thisElementPtr = 0;

	for( IndexType group = 0; group < this->getNumberOfGroups( row ) && !padding; group++ )
	{
		for( IndexType i = 0; ( i < rowMultiplicator * this->getGroupLength( strip, group ) ) && !padding; i++ )
		{
			if( this->columnIndexes.getElement( elementPtr ) == this->getPaddingIndex() )
			{
				padding = true;
				break;
			}
			values[ thisElementPtr ] = this->values.getElement( elementPtr );
			columns[ thisElementPtr ] = this->columnIndexes.getElement( elementPtr );
			thisElementPtr++;
			elementPtr += step;
		}
		step /= 2;
		rowMultiplicator *= 2;
	}
}

template< typename Real,
		  typename Device,
		  typename Index >
void BiEllpack< Real, Device, Index >::setVirtualRows(const IndexType rows)
{
    this->virtualRows = rows;
}

template< typename Real,
          typename Device,
          typename Index >
__cuda_callable__
Index BiEllpack< Real, Device, Index >::getGroupLength( const Index strip,
                                                                   const Index group ) const
{
    return this->groupPointers.getElement( strip * ( this->logWarpSize + 1 ) + group + 1 )
            - this->groupPointers.getElement( strip * ( this->logWarpSize + 1 ) + group );
}

template< typename Real,
          typename Device,
          typename Index >
template< typename InVector,
	  	  typename OutVector >
void BiEllpack< Real, Device, Index >::vectorProduct( const InVector& inVector,
								 OutVector& outVector ) const
{
    DeviceDependentCode::vectorProduct( *this, inVector, outVector );
}

template< typename Real,
          typename Device,
          typename Index >
template< typename InVector,
		  typename OutVector >
void BiEllpack< Real, Device, Index >::vectorProductHost( const InVector& inVector,
                                                                     OutVector& outVector ) const
{
	const IndexType cudaBlockSize = 256;
	const IndexType cudaBlocks = roundUpDivision( this->getRows(), cudaBlockSize );
	for( IndexType blockIdx = 0; blockIdx < cudaBlocks; blockIdx++ )
	{
		Containers::Vector< Real, Device, Index > tempStripOutVector;
		tempStripOutVector.setSize( cudaBlockSize );
		for( IndexType i = 0; i < tempStripOutVector.getSize(); i++ )
			tempStripOutVector.setElement( i, 0 );

		for( IndexType threadIdx = 0; threadIdx < cudaBlockSize; threadIdx++ )
		{
			IndexType globalIdx = cudaBlockSize * blockIdx + threadIdx;
			IndexType warpStart = this->warpSize * ( globalIdx / this->warpSize );
			IndexType inWarpIdx = globalIdx % this->warpSize;
			if( warpStart >= this->getRows() )
				break;
			IndexType strip = warpStart / this->warpSize;
			const IndexType groupBegin = strip * ( this->logWarpSize + 1 );

			IndexType row = warpStart + inWarpIdx;
			IndexType currentRow = row;
			IndexType elementPtr = this->groupPointers.getElement( groupBegin ) * this->warpSize + ( row - warpStart );
			IndexType bisection = this->warpSize;
			for( IndexType group = 0; group < this->logWarpSize + 1; group++ )
			{
				if( !( currentRow - warpStart < bisection ) )
					currentRow -= bisection;
				IndexType groupLength = this->groupPointers[ strip * ( this->logWarpSize + 1 ) + group + 1 ]
				               		      - this->groupPointers[ strip * ( this->logWarpSize + 1 ) + group ];
				for( IndexType i = 0; i < groupLength; i++ )
				{
					if( this->columnIndexes.getElement( elementPtr ) == this->getPaddingIndex() )
					{
						elementPtr += this->warpSize;
						continue;
					}
					RealType result = tempStripOutVector.getElement( currentRow % cudaBlockSize );
					result += inVector[ this->columnIndexes.getElement( elementPtr ) ] * this->values.getElement( elementPtr );
					tempStripOutVector.setElement( currentRow % cudaBlockSize, result );
					elementPtr += this->warpSize;
				}
				bisection /= 2;
			}
		}
		IndexType end = cudaBlockSize * ( blockIdx + 1 );
		if( end > this->getRows() )
			end = this->getRows();
		for( IndexType i = cudaBlockSize * blockIdx; i < end; i++ )
			outVector[ i ] = tempStripOutVector.getElement( this->rowPermArray.getElement( i ) % cudaBlockSize );
		tempStripOutVector.reset();
	}
}

// copy assignment
template< typename Real,
          typename Device,
          typename Index >
BiEllpack< Real, Device, Index >&
BiEllpack< Real, Device, Index >::operator=( const BiEllpack& matrix )
{
   this->setLike( matrix );
   this->values = matrix.values;
   this->columnIndexes = matrix.columnIndexes;
   this->warpSize = matrix.warpSize;
   this->logWarpSize = matrix.logWarpSize;
   this->virtualRows = matrix.virtualRows;
   this->rowPermArray = matrix.rowPermArray;
   this->groupPointers = matrix.groupPointers;
   return *this;
}

// cross-device copy assignment
template< typename Real,
          typename Device,
          typename Index >
   template< typename Real2, typename Device2, typename Index2, typename >
BiEllpack< Real, Device, Index >&
BiEllpack< Real, Device, Index >::operator=( const BiEllpack< Real2, Device2, Index2 >& matrix )
{
   static_assert( std::is_same< Device, Devices::Host >::value || std::is_same< Device, Devices::Cuda >::value,
                  "unknown device" );
   static_assert( std::is_same< Device2, Devices::Host >::value || std::is_same< Device2, Devices::Cuda >::value,
                  "unknown device" );

   this->setLike( matrix );
   this->values = matrix.values;
   this->columnIndexes = matrix.columnIndexes;
   this->warpSize = matrix.warpSize;
   this->logWarpSize = matrix.logWarpSize;
   this->virtualRows = matrix.virtualRows;
   this->rowPermArray = matrix.rowPermArray;
   this->groupPointers = matrix.groupPointers;
   return *this;
}

template< typename Real,
		  typename Device,
		  typename Index >
void BiEllpack< Real, Device, Index >::save( File& file ) const
{
   Sparse< Real, Device, Index >::save( file );
   file << this->groupPointers << this->rowPermArray;
}

template< typename Real,
		  typename Device,
		  typename Index >
void BiEllpack< Real, Device, Index >::load( File& file )
{
   Sparse< Real, Device, Index >::load( file );
   file >> this->groupPointers >> this->rowPermArray;
}

template< typename Real,
		  typename Device,
		  typename Index >
void BiEllpack< Real, Device, Index >::save( const String& fileName ) const
{
   Object::save( fileName );
}

template< typename Real,
		  typename Device,
		  typename Index >
void BiEllpack< Real, Device, Index >::load( const String& fileName )
{
   Object::load( fileName );
}

template< typename Real,
		  typename Device,
		  typename Index >
void BiEllpack< Real, Device, Index >::print( std::ostream& str ) const
{
	for( IndexType row = 0; row < this->getRows(); row++ )
	{
		str <<"Row: " << row << " -> ";
//                str << row << ": ";
		bool padding = false;
		const IndexType strip = row / this->warpSize;
		const IndexType groupBegin = strip * ( this->logWarpSize + 1 );
		const IndexType rowStripPerm = this->rowPermArray.getElement( row ) - this->warpSize * strip;
		IndexType elementPtr = this->groupPointers.getElement( groupBegin ) * this->warpSize + rowStripPerm;
		IndexType rowMultiplicator = 1;
		IndexType step = this->warpSize;

		for( IndexType group = 0; group < this->getNumberOfGroups( row ) && !padding; group++ )
		{
			for( IndexType i = 0; ( i < rowMultiplicator * this->getGroupLength( strip, group ) ) && !padding; i++ )
			{
				if( this->columnIndexes.getElement( elementPtr ) == this->getPaddingIndex() )
				{
					padding = true;
					break;
				}
				RealType value = this->values.getElement( elementPtr );
				IndexType column = this->columnIndexes.getElement( elementPtr );
				str << " Col:" << column << "->" << value << "\t";
//                                str << value << " ";
				elementPtr += step;
			}
			step /= 2;
			rowMultiplicator *= 2;
		}
		str << std::endl;
	}
}

template< typename Real,
          typename Device,
          typename Index >
void BiEllpack< Real, Device, Index >::printValues() const
{
    for( Index i = 0; i < this->values.getSize(); i++ ) {
        if( this->columnIndexes.getElement( i ) != this->getColumns() )
            std::cout << "values.getElement( " << i << " ) = " << this->values.getElement( i )
             << "\tcolumnIndexes.getElement( " << i << " ) = " << this->columnIndexes.getElement( i ) << std::endl;
    }

    for( Index i = 0; i < this->rowPermArray.getSize(); i++ ) {
        std::cout << "rowPermArray[ " << i << " ] = " << this->rowPermArray.getElement( i ) << std::endl;
    }

    for( Index i = 0; i < this->groupPointers.getSize(); i++ ) {
        std::cout << "groupPointers[ " << i << " ] = " << this->groupPointers.getElement( i ) << std::endl;
    }
}

template< typename Real,
		  typename Device,
		  typename Index >
void BiEllpack< Real, Device, Index >::performRowBubbleSort( Containers::Vector< Index, Device, Index >& tempRowLengths )
{
    Index strips = this->virtualRows / this->warpSize;
    for( Index i = 0; i < strips; i++ )
    {
        Index begin = i * this->warpSize;
        Index end = ( i + 1 ) * this->warpSize - 1;
        if( this->getRows() - 1 < end)
            end = this->getRows() - 1;
        bool sorted = false;
        Index permIndex1 = 0;
        Index permIndex2 = 0;
        Index offset = 0;
        while( !sorted )
        {
            sorted = true;
            for( Index j = begin + offset; j < end - offset; j++ )
                if( tempRowLengths.getElement( j ) < tempRowLengths.getElement( j + 1 ) )
                {
                    for( Index k = begin; k < end + 1; k++ )
                    {
                    	if( this->rowPermArray.getElement( k ) == j )
                    		permIndex1 = k;
                    	if( this->rowPermArray.getElement( k ) == j + 1 )
                    		permIndex2 = k;
                    }
                    Index temp = tempRowLengths.getElement( j );
                    tempRowLengths.setElement( j, tempRowLengths.getElement( j + 1 ) );
                    tempRowLengths.setElement( j + 1, temp );
                    temp = this->rowPermArray.getElement( permIndex1 );
                    this->rowPermArray.setElement( permIndex1, this->rowPermArray.getElement( permIndex2 ) );
                    this->rowPermArray.setElement( permIndex2, temp );
                    sorted = false;
                }
            for( Index j = end - 1 - offset; j > begin + offset; j-- )
                if( tempRowLengths.getElement( j ) > tempRowLengths.getElement( j - 1 ) )
                {
                    for( Index k = begin; k < end + 1; k++ )
                    {
                    	if( this->rowPermArray.getElement( k ) == j )
                    		permIndex1 = k;
                    	if( this->rowPermArray.getElement( k ) == j - 1 )
                    		permIndex2 = k;
                    }
                    Index temp = tempRowLengths.getElement( j );
                    tempRowLengths.setElement( j, tempRowLengths.getElement( j - 1 ) );
                    tempRowLengths.setElement( j - 1, temp );
                    temp = this->rowPermArray.getElement( permIndex1 );
                    this->rowPermArray.setElement( permIndex1, this->rowPermArray.getElement( permIndex2 ) );
                    this->rowPermArray.setElement( permIndex2, temp );
                    sorted = false;
                }
            offset++;
        }
    }
}

template< typename Real,
		  typename Device,
		  typename Index >
void BiEllpack< Real, Device, Index >::computeColumnSizes( Containers::Vector< Index, Device, Index >& tempRowLengths )
{
    Index numberOfStrips = this->virtualRows / this->warpSize;
    for( Index strip = 0; strip < numberOfStrips; strip++ )
    {
        Index i = 0;
        Index rowBegin = strip * this->warpSize;
        Index groupBegin = strip * ( this->logWarpSize + 1 );
        Index emptyGroups = 0;
        if( strip == numberOfStrips - 1 )
        {
            Index lastRows = this->getRows() - rowBegin;
            while( !( lastRows > this->power( 2, this->logWarpSize - 1 - emptyGroups ) ) )
                emptyGroups++;
            for( Index group = groupBegin; group < groupBegin + emptyGroups; group++ )
                this->groupPointers.setElement( group, 0 );
        }
        i += emptyGroups;
        for( Index group = groupBegin + emptyGroups; group < groupBegin + this->logWarpSize; group++ )
        {
            Index row = this->power( 2, 4 - i );
            Index temp = tempRowLengths.getElement( row + rowBegin );
            for( Index prevGroups = groupBegin; prevGroups < group; prevGroups++ )
                temp -= this->power( 2, prevGroups - groupBegin ) * this->groupPointers.getElement( prevGroups );
            temp =  ceil( ( float ) temp / this->power( 2, i ) );
            this->groupPointers.setElement( group, temp );
            i++;
        }
        Index temp = tempRowLengths.getElement( rowBegin );
        for( Index prevGroups = groupBegin; prevGroups < groupBegin + this->logWarpSize; prevGroups++ )
            temp -= this->power( 2, prevGroups - groupBegin ) * this->groupPointers.getElement( prevGroups );
        temp = ceil( ( float ) temp / this->power( 2, this->logWarpSize ) );
        this->groupPointers.setElement( groupBegin + this->logWarpSize, temp );
    }
}

template<>
class BiEllpackDeviceDependentCode< Devices::Host >
{
public:

	typedef Devices::Host Device;

	template< typename Real,
			  typename Index >
	static void verifyRowLengths( const BiEllpack< Real, Device, Index >& matrix,
                                      const typename BiEllpack< Real, Device, Index >::RowsCapacitiesType& rowLengths )
	{
		bool ok = true;
		for( Index row = 0; row < matrix.getRows(); row++ )
		{
			const Index strip = row / matrix.warpSize;
			const Index stripLength = matrix.getStripLength( strip );
			const Index groupBegin = ( matrix.logWarpSize + 1 ) * strip;
			const Index rowStripPerm = matrix.rowPermArray.getElement( row ) - strip * matrix.warpSize;
			const Index begin = matrix.groupPointers.getElement( groupBegin ) * matrix.warpSize + rowStripPerm * stripLength;
			Index elementPtr = begin;
			Index rowLength = 0;
			for( Index group = 0; group < matrix.getNumberOfGroups( row ); group++ )
			{
				for( Index i = 0; i < matrix.getGroupLength( strip, group ); i++ )
				{
					Index biElementPtr = elementPtr;
					for( Index j = 0; j < matrix.power( 2, group ); j++ )
					{
						rowLength++;
						biElementPtr += matrix.power( 2, matrix.logWarpSize - group ) * stripLength;
					}
					elementPtr++;
				}
			}
			if( rowLengths.getElement( row ) > rowLength )
				ok = false;
		}
		if( ok )
                {
//                    std::cout << "row lengths OK" << std::endl;
                }
	}

	template< typename Real,
			  typename Index >
	static void verifyRowPerm( const BiEllpack< Real, Device, Index >& matrix,
                                   const typename BiEllpack< Real, Device, Index >::RowsCapacitiesType& rowLengths )
	{
		bool ok = true;
		Index numberOfStrips = matrix.virtualRows / matrix.warpSize;
		for( Index strip = 0; strip < numberOfStrips; strip++ )
		{
			Index begin = strip * matrix.warpSize;
			Index end = ( strip + 1 ) * matrix.warpSize;
			if( matrix.getRows() < end )
				end = matrix.getRows();
			for( Index i = begin; i < end - 1; i++ )
			{
				Index permIndex1 = 0;
				Index permIndex2 = 0;
				bool first = false;
				bool second = false;
				for( Index j = begin; j < end; j++ )
				{
					if( matrix.rowPermArray.getElement( j ) == i )
					{
						permIndex1 = j;
						first = true;
					}
					if( matrix.rowPermArray.getElement( j ) == i + 1 )
					{
						permIndex2 = j;
						second = true;
					}
				}
				if( !first || !second )
					std::cout << "Wrong permutation!" << std::endl;
				if( rowLengths.getElement( permIndex1 ) >= rowLengths.getElement( permIndex2 ) )
					continue;
				else
					ok = false;
			}
		}
		if( ok )
                {
//                    std::cout << "Permutation OK" << std::endl;
                }
	}

	template< typename Real,
			  typename Index,
			  typename InVector,
			  typename OutVector >
	static void vectorProduct( const BiEllpack< Real, Device, Index >& matrix,
                                   const InVector& inVector,
                                   OutVector& outVector )
	{
		matrix.vectorProductHost( inVector, outVector );
	}

	template< typename Real,
			  typename Index >
	static void computeColumnSizes( BiEllpack< Real, Device, Index >& matrix,
			 	 	const typename BiEllpack< Real, Device, Index >::RowsCapacitiesType& rowLengths )
	{
		Index numberOfStrips = matrix.virtualRows / matrix.warpSize;
		for( Index strip = 0; strip < numberOfStrips; strip++ )
		{
			Index i = 0;
			Index rowBegin = strip * matrix.warpSize;
			Index groupBegin = strip * ( matrix.logWarpSize + 1 );
			Index emptyGroups = 0;
			if( strip == numberOfStrips - 1 )
			{
				Index lastRows = matrix.getRows() - rowBegin;
				while( !( lastRows > matrix.power( 2, matrix.logWarpSize - 1 - emptyGroups ) ) )
					emptyGroups++;
				for( Index group = groupBegin; group < groupBegin + emptyGroups; group++ )
					matrix.groupPointers.setElement( group, 0 );
			}
			i += emptyGroups;
			for( Index group = groupBegin + emptyGroups; group < groupBegin + matrix.logWarpSize; group++ )
			{
				Index row = matrix.power( 2, 4 - i );
				Index permRow = 0;
				while( matrix.rowPermArray.getElement( permRow + rowBegin ) != row + rowBegin )
					permRow++;
				Index temp = rowLengths.getElement( permRow + rowBegin );
				for( Index prevGroups = groupBegin; prevGroups < group; prevGroups++ )
					temp -= matrix.power( 2, prevGroups - groupBegin ) * matrix.groupPointers.getElement( prevGroups );
				temp =  ceil( ( float ) temp / matrix.power( 2, i ) );
				matrix.groupPointers.setElement( group, temp );
				i++;
			}
			Index permRow = rowBegin;
			while( matrix.rowPermArray.getElement( permRow ) != rowBegin )
				permRow++;
			Index temp = rowLengths.getElement( permRow );
			for( Index prevGroups = groupBegin; prevGroups < groupBegin + matrix.logWarpSize; prevGroups++ )
				temp -= matrix.power( 2, prevGroups - groupBegin ) * matrix.groupPointers.getElement( prevGroups );
			temp = ceil( ( float ) temp / matrix.power( 2, matrix.logWarpSize ) );
			matrix.groupPointers.setElement( groupBegin + matrix.logWarpSize, temp );
		}
	}

	template< typename Real,
			  typename Index >
	static void performRowBubbleSort( BiEllpack< Real, Device, Index >& matrix,
					  const typename BiEllpack< Real, Device, Index >::RowsCapacitiesType& rowLengths
					/*Containers::Vector< Index, Device, Index >& tempRowLengths*/ )
	{
		Index strips = matrix.virtualRows / matrix.warpSize;
		for( Index i = 0; i < strips; i++ )
		{
			Index begin = i * matrix.warpSize;
			Index end = ( i + 1 ) * matrix.warpSize - 1;
			if(matrix.getRows() - 1 < end)
				end = matrix.getRows() - 1;
			bool sorted = false;
			Index permIndex1 = 0;
			Index permIndex2 = 0;
			Index offset = 0;
			while( !sorted )
			{
				sorted = true;
				for( Index j = begin + offset; j < end - offset; j++ )
				{
					for( Index k = begin; k < end + 1; k++ )
					{
						if( matrix.rowPermArray.getElement( k ) == j )
							permIndex1 = k;
						if( matrix.rowPermArray.getElement( k ) == j + 1 )
							permIndex2 = k;
					}
					if( rowLengths.getElement( permIndex1 ) < rowLengths.getElement( permIndex2 ) )
					{
						Index temp = matrix.rowPermArray.getElement( permIndex1 );
						matrix.rowPermArray.setElement( permIndex1, matrix.rowPermArray.getElement( permIndex2 ) );
						matrix.rowPermArray.setElement( permIndex2, temp );
						sorted = false;
					}
				}
				for( Index j = end - 1 - offset; j > begin + offset; j-- )
				{
					for( Index k = begin; k < end + 1; k++ )
					{
						if( matrix.rowPermArray.getElement( k ) == j )
							permIndex1 = k;
						if( matrix.rowPermArray.getElement( k ) == j - 1 )
							permIndex2 = k;
					}
					if( rowLengths.getElement( permIndex2 ) < rowLengths.getElement( permIndex1 ) )
					{
						Index temp = matrix.rowPermArray.getElement( permIndex1 );
						matrix.rowPermArray.setElement( permIndex1, matrix.rowPermArray.getElement( permIndex2 ) );
						matrix.rowPermArray.setElement( permIndex2, temp );
						sorted = false;
					}
				}
				offset++;
			}
		}
	}
};

#ifdef __CUDACC__
template< typename Real,
          typename Device,
          typename Index >
template< typename InVector,
          typename OutVector >
__device__
void BiEllpack< Real, Device, Index >::spmvCuda( const InVector& inVector,
					  	  	    OutVector& outVector,
                                                            int globalIdx ) const
{
    const IndexType strip = globalIdx >> this->logWarpSize;
    const IndexType warpStart = strip << this->logWarpSize;
    const IndexType inWarpIdx = globalIdx & ( this->warpSize - 1 );

    if( warpStart >= this->getRows() )
    return;

    const IndexType cudaBlockSize = 256;
    IndexType bisection = this->warpSize;
    IndexType groupBegin = strip * ( this->logWarpSize + 1 );

    Real* temp = Cuda::getSharedMemory< Real >();
    __shared__ Real results[ cudaBlockSize ];
    results[ threadIdx.x ] = 0.0;
    IndexType elementPtr = ( this->groupPointers[ groupBegin ] << this->logWarpSize ) + inWarpIdx;

    for( IndexType group = 0; group < this->logWarpSize + 1; group++ )
    {
        temp[ threadIdx.x ] = 0.0;
        IndexType groupLength = this->groupPointers[ strip * ( this->logWarpSize + 1 ) + group + 1 ]
                                  - this->groupPointers[ strip * ( this->logWarpSize + 1 ) + group ];

        if( groupLength > 0 )
        {
            for( IndexType i = 0; i < groupLength; i++ )
            {
                if( this->columnIndexes[ elementPtr ] < this->getColumns() )
                temp[ threadIdx.x ] += inVector[ this->columnIndexes[ elementPtr ] ] * this->values[ elementPtr ];
                elementPtr += this->warpSize;
            }
            IndexType bisection2 = this->warpSize;
            for( IndexType i = 0; i < group; i++ )
            {
                bisection2 >>= 1;
                if( inWarpIdx < bisection2 )
                temp[ threadIdx.x ] += temp[ threadIdx.x + bisection2 ];
            }
            if( inWarpIdx < bisection )
                results[ threadIdx.x ] += temp[ threadIdx.x ];
        }
        bisection >>= 1;
    }
    __syncthreads();
    if( warpStart + inWarpIdx >= this->getRows() )
        return;

    outVector[ warpStart + inWarpIdx ] = results[ this->rowPermArray[ warpStart + inWarpIdx ] & ( cudaBlockSize - 1 ) ];
}
#endif

#ifdef __CUDACC__
template< typename Real,
          typename Index,
          typename InVector,
          typename OutVector >
__global__
void BiEllpackVectorProductCuda( const BiEllpack< Real, Devices::Cuda, Index >* matrix,
				 const InVector* inVector,
				 OutVector* outVector,
				 int gridIdx,
				 const int warpSize )
{
	Index globalIdx = ( gridIdx * Cuda::getMaxGridXSize() + blockIdx.x ) * blockDim.x + threadIdx.x;
	matrix->spmvCuda( *inVector, *outVector, globalIdx );
}
#endif

#ifdef __CUDACC__
template< typename Real,
          typename Device,
          typename Index >
__cuda_callable__
void BiEllpack< Real, Device, Index >::performRowBubbleSortCudaKernel( const typename BiEllpack< Real, Device, Index >::RowsCapacitiesType& rowLengths,
										  const IndexType strip )
{
    IndexType begin = strip * this->warpSize;
    IndexType end = ( strip + 1 ) * this->warpSize - 1;
    if( this->getRows() - 1 < end )
        end = this->getRows() - 1;
    bool sorted = false;
    IndexType permIndex1 = 0;
    IndexType permIndex2 = 0;
    IndexType offset = 0;
    while( !sorted )
    {
        sorted = true;
        for( IndexType j = begin + offset; j < end - offset; j++ )
        {
            for( IndexType k = begin; k < end + 1; k++)
            {
                if( this->rowPermArray[ k ] == j )
                    permIndex1 = k;
                if( this->rowPermArray[ k ] == j + 1 )
                    permIndex2 = k;
            }
            if( rowLengths[ permIndex1 ] < rowLengths[ permIndex2 ] )
            {
                IndexType temp = this->rowPermArray[ permIndex1 ];
                this->rowPermArray[ permIndex1 ] = this->rowPermArray[ permIndex2 ];
                this->rowPermArray[ permIndex2 ] = temp;
                sorted = false;
            }
        }
        for( IndexType j = end - 1 - offset; j > begin + offset; j-- )
        {
            for( IndexType k = begin; k < end + 1; k++ )
            {
                if( this->rowPermArray[ k ] == j )
                    permIndex1 = k;
                if( this->rowPermArray[ k ] == j - 1)
                    permIndex2 = k;
            }
            if( rowLengths[ permIndex2 ] < rowLengths[ permIndex1 ] )
            {
                IndexType temp = this->rowPermArray[ permIndex1 ];
                this->rowPermArray[ permIndex1 ] = this->rowPermArray[ permIndex2 ];
                this->rowPermArray[ permIndex2 ] = temp;
                sorted = false;
            }
        }
        offset++;
    }
}
#endif

#ifdef __CUDACC__
template< typename Real,
          typename Device,
          typename Index >
__cuda_callable__
void BiEllpack< Real, Device, Index >::computeColumnSizesCudaKernel( const typename BiEllpack< Real, Device, Index >::RowsCapacitiesType& rowLengths,
										const IndexType numberOfStrips,
										const IndexType strip )
{
    if( strip >= numberOfStrips )
        return;
    IndexType i = 0;
    IndexType rowBegin = strip * this->warpSize;
    IndexType groupBegin = strip * ( this->logWarpSize + 1 );
    IndexType emptyGroups = 0;
    if( strip == numberOfStrips - 1 )
    {
        IndexType lastRows = this->getRows() - rowBegin;
        while( !( lastRows > this->power( 2, this->logWarpSize - 1 - emptyGroups ) ) )
            emptyGroups++;
        for( IndexType group = groupBegin; group < groupBegin + emptyGroups; group++ )
            this->groupPointers[ group ] = 0;
    }
    i += emptyGroups;
    for( IndexType group = groupBegin + emptyGroups; group < groupBegin + this->logWarpSize; group++ )
    {
        IndexType row = this->power( 2, 4 - i );
        IndexType permRow = 0;
        while( this->rowPermArray[ permRow + rowBegin ] != row + rowBegin && permRow < this->warpSize )
            permRow++;
        IndexType temp = rowLengths[ permRow + rowBegin ];
        for( IndexType prevGroups = groupBegin; prevGroups < group; prevGroups++ )
            temp -= this->power( 2, prevGroups - groupBegin ) * this->groupPointers[ prevGroups ];
        temp =  ceil( ( float ) temp / this->power( 2, i ) );
        this->groupPointers[ group ] = temp;
        i++;
    }
    IndexType permRow = rowBegin;
    while( this->rowPermArray[ permRow ] != rowBegin && permRow < this->warpSize + rowBegin )
        permRow++;
    IndexType temp = rowLengths[ permRow ];
    for( IndexType prevGroups = groupBegin; prevGroups < groupBegin + this->logWarpSize; prevGroups++ )
        temp -= this->power( 2, prevGroups - groupBegin ) * this->groupPointers[ prevGroups ];
    temp = ceil( ( float ) temp / this->power( 2, this->logWarpSize ) );
    this->groupPointers[ groupBegin + this->logWarpSize ] = temp;
}
#endif

#ifdef __CUDACC__
template< typename Real,
          typename Index >
__global__
void performRowBubbleSortCuda( BiEllpack< Real, Devices::Cuda, Index >* matrix,
                               const typename BiEllpack< Real, Devices::Cuda, Index >::RowsCapacitiesType* rowLengths,
                               int gridIdx )
{
	const Index stripIdx = gridIdx * Cuda::getMaxGridXSize() * blockDim.x + blockIdx.x * blockDim.x + threadIdx.x;
	matrix->performRowBubbleSortCudaKernel( *rowLengths, stripIdx );
}
#endif

#ifdef __CUDACC__
template< typename Real,
          typename Index >
__global__
void computeColumnSizesCuda( BiEllpack< Real, Devices::Cuda, Index >* matrix,
                             const typename BiEllpack< Real, Devices::Cuda, Index >::RowsCapacitiesType* rowLengths,
                             const Index numberOfStrips,
                             int gridIdx )
{
	const Index stripIdx = gridIdx * Cuda::getMaxGridXSize() * blockDim.x + blockIdx.x * blockDim.x + threadIdx.x;
	matrix->computeColumnSizesCudaKernel( *rowLengths, numberOfStrips, stripIdx );
}
#endif

template<>
class BiEllpackDeviceDependentCode< Devices::Cuda >
{
public:

	typedef Devices::Cuda Device;

	template< typename Real,
		  typename Index >
	static void verifyRowLengths( const BiEllpack< Real, Device, Index >& matrix,
                                      const typename BiEllpack< Real, Device, Index >::RowsCapacitiesType& rowLengths )
	{
		bool ok = true;
		for( Index row = 0; row < matrix.getRows(); row++ )
		{
                    const Index strip = row / matrix.warpSize;
                    const Index stripLength = matrix.getStripLength( strip );
                    const Index groupBegin = ( matrix.logWarpSize + 1 ) * strip;
                    const Index rowStripPerm = matrix.rowPermArray.getElement( row ) - strip * matrix.warpSize;
                    const Index begin = matrix.groupPointers.getElement( groupBegin ) * matrix.warpSize + rowStripPerm * stripLength;
                    Index elementPtr = begin;
                    Index rowLength = 0;

                    for( Index group = 0; group < matrix.getNumberOfGroups( row ); group++ )
                    {
                        for( Index i = 0; i < matrix.getGroupLength( strip, group ); i++ )
                        {
                            Index biElementPtr = elementPtr;
                            for( Index j = 0; j < matrix.power( 2, group ); j++ )
                            {
                                rowLength++;
                                biElementPtr += matrix.power( 2, matrix.logWarpSize - group ) * stripLength;
                            }
                            elementPtr++;
                        }
                    }
                    if( rowLengths.getElement( row ) > rowLength )
                        ok = false;
		}
		if( ok )
                {
//                    std::cout << "row lengths OK" << std::endl;
                }
	}

	template< typename Real,
			  typename Index >
	static void verifyRowPerm( const BiEllpack< Real, Device, Index >& matrix,
                                   const typename BiEllpack< Real, Device, Index >::RowsCapacitiesType& rowLengths )
	{
		bool ok = true;
		Index numberOfStrips = matrix.virtualRows / matrix.warpSize;
		for( Index strip = 0; strip < numberOfStrips; strip++ )
		{
			Index begin = strip * matrix.warpSize;
			Index end = ( strip + 1 ) * matrix.warpSize;
			if( matrix.getRows() < end )
				end = matrix.getRows();
			for( Index i = begin; i < end - 1; i++ )
			{
				Index permIndex1 = 0;
				Index permIndex2 = 0;
				bool first = false;
				bool second = false;
				for( Index j = begin; j < end; j++ )
				{
					if( matrix.rowPermArray.getElement( j ) == i )
					{
						permIndex1 = j;
						first = true;
					}
					if( matrix.rowPermArray.getElement( j ) == i + 1 )
					{
						permIndex2 = j;
						second = true;
					}
				}
				if( !first || !second )
					std::cout << "nenasel jsem spravne indexy" << std::endl;
				if( rowLengths.getElement( permIndex1 ) >= rowLengths.getElement( permIndex2 ) )
					continue;
				else
					ok = false;
			}
		}
		if( ok )
                {
//                    std::cout << "perm OK" << std::endl;
                }
	}

	template< typename Real,
			  typename Index >
	static void performRowBubbleSort( BiEllpack< Real, Device, Index >& matrix,
                                          const typename BiEllpack< Real, Device, Index >::RowsCapacitiesType& rowLengths )
	{
#ifdef __CUDACC__
		Index numberOfStrips = matrix.virtualRows / matrix.warpSize;
		typedef BiEllpack< Real, Devices::Cuda, Index > Matrix;
		typedef typename Matrix::RowsCapacitiesType RowsCapacitiesType;
		Matrix* kernel_this = Cuda::passToDevice( matrix );
		RowsCapacitiesType* kernel_rowLengths = Cuda::passToDevice( rowLengths );
		dim3 cudaBlockSize( 256 ), cudaGridSize( Cuda::getMaxGridXSize() );
		const Index cudaBlocks = roundUpDivision( numberOfStrips, cudaBlockSize.x );
		const Index cudaGrids = roundUpDivision( cudaBlocks, Cuda::getMaxGridXSize() );
		for( int gridIdx = 0; gridIdx < cudaGrids; gridIdx++ )
		{
		     if( gridIdx == cudaGrids - 1 )
		         cudaGridSize.x = cudaBlocks % Cuda::getMaxGridXSize();
		     performRowBubbleSortCuda< Real, Index >
		     	 	 	 	 	 	 <<< cudaGridSize, cudaBlockSize >>>
		                             ( kernel_this,
		                               kernel_rowLengths,
		                               gridIdx );
		}
		Cuda::freeFromDevice( kernel_this );
		Cuda::freeFromDevice( kernel_rowLengths );
		TNL_CHECK_CUDA_DEVICE;
#endif
	}

	template< typename Real,
			  typename Index >
	static void computeColumnSizes( BiEllpack< Real, Device, Index >& matrix,
			 	 	const typename BiEllpack< Real, Device, Index >::RowsCapacitiesType& rowLengths )
	{
#ifdef __CUDACC__
		const Index numberOfStrips = matrix.virtualRows / matrix.warpSize;
		typedef BiEllpack< Real, Devices::Cuda, Index > Matrix;
		typedef typename Matrix::RowsCapacitiesType RowsCapacitiesType;
		Matrix* kernel_this = Cuda::passToDevice( matrix );
		RowsCapacitiesType* kernel_rowLengths = Cuda::passToDevice( rowLengths );
		dim3 cudaBlockSize( 256 ), cudaGridSize( Cuda::getMaxGridXSize() );
		const Index cudaBlocks = roundUpDivision( numberOfStrips, cudaBlockSize.x );
		const Index cudaGrids = roundUpDivision( cudaBlocks, Cuda::getMaxGridXSize() );
		for( int gridIdx = 0; gridIdx < cudaGrids; gridIdx++ )
		{
		     if( gridIdx == cudaGrids - 1 )
		         cudaGridSize.x = cudaBlocks % Cuda::getMaxGridXSize();
		     computeColumnSizesCuda< Real, Index >
		     	 	 	 	 	   <<< cudaGridSize, cudaBlockSize >>>
		                           ( kernel_this,
		                             kernel_rowLengths,
		                             numberOfStrips,
		                             gridIdx );
        }
		Cuda::freeFromDevice( kernel_this );
		Cuda::freeFromDevice( kernel_rowLengths );
		TNL_CHECK_CUDA_DEVICE;
#endif
	}


	template< typename Real,
			  typename Index,
			  typename InVector,
			  typename OutVector >
	static void vectorProduct( const BiEllpack< Real, Device, Index >& matrix,
			   	   	   	   	   const InVector& inVector,
			   	   	   	   	   OutVector& outVector )
	{
#ifdef __CUDACC__
		typedef BiEllpack< Real, Devices::Cuda, Index > Matrix;
		typedef typename Matrix::IndexType IndexType;
		Matrix* kernel_this = Cuda::passToDevice( matrix );
		InVector* kernel_inVector = Cuda::passToDevice( inVector );
		OutVector* kernel_outVector = Cuda::passToDevice( outVector );
		dim3 cudaBlockSize( 256 ), cudaGridSize( Cuda::getMaxGridXSize() );
		const IndexType cudaBlocks = roundUpDivision( matrix.getRows(), cudaBlockSize.x );
		const IndexType cudaGrids = roundUpDivision( cudaBlocks, Cuda::getMaxGridXSize() );
		for( IndexType gridIdx = 0; gridIdx < cudaGrids; gridIdx++ )
		{
			if( gridIdx == cudaGrids - 1 )
				cudaGridSize.x = cudaBlocks % Cuda::getMaxGridXSize();
			const int sharedMemory = cudaBlockSize.x * sizeof( Real );
			BiEllpackVectorProductCuda< Real, Index, InVector, OutVector >
			                                   <<< cudaGridSize, cudaBlockSize, sharedMemory >>>
			                                   ( kernel_this,
			                                     kernel_inVector,
			                                     kernel_outVector,
			                                     gridIdx,
			                                     matrix.warpSize );
		}
		Cuda::freeFromDevice( kernel_this );
		Cuda::freeFromDevice( kernel_inVector );
		Cuda::freeFromDevice( kernel_outVector );
		TNL_CHECK_CUDA_DEVICE;
#endif
    }

};
      			} //namespace Legacy
            } //namespace ReferenceFormats
        } //namespace SpMV
    } //namespace Benchmarks
} // namespace TNL

