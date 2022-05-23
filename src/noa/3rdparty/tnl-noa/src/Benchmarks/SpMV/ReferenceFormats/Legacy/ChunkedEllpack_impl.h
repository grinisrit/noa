#pragma once

#include <Benchmarks/SpMV/ReferenceFormats/Legacy/ChunkedEllpack.h>
#include <TNL/Containers/Vector.h>
#include <TNL/Algorithms/scan.h>
#include <TNL/Math.h>
#include <TNL/Exceptions/NotImplementedError.h>

namespace TNL {
    namespace Benchmarks {
        namespace SpMV {
            namespace ReferenceFormats {
               namespace Legacy {

template< typename Real,
          typename Index,
          typename Vector >
void ChunkedEllpackVectorProductCuda( const ChunkedEllpack< Real, Devices::Cuda, Index >& matrix,
                                               const Vector& inVector,
                                               Vector& outVector );


template< typename Real,
          typename Device,
          typename Index >
ChunkedEllpack< Real, Device, Index >::ChunkedEllpack()
: chunksInSlice( 256 ),
  desiredChunkSize( 16 ),
  numberOfSlices( 0 )
{
};

template< typename Real,
          typename Device,
          typename Index >
std::string ChunkedEllpack< Real, Device, Index >::getSerializationType()
{
   return "Matrices::ChunkedEllpack< " +
          getType< Real >() +
          ", [any device], " +
          TNL::getType< Index >() +
          " >";
}

template< typename Real,
          typename Device,
          typename Index >
std::string ChunkedEllpack< Real, Device, Index >::getSerializationTypeVirtual() const
{
   return this->getSerializationType();
}

template< typename Real,
          typename Device,
          typename Index >
void ChunkedEllpack< Real, Device, Index >::setDimensions( const IndexType rows,
                                                           const IndexType columns )
{
   TNL_ASSERT( rows > 0 && columns > 0,
              std::cerr << "rows = " << rows
                   << " columns = " << columns << std::endl );
   Sparse< Real, Device, Index >::setDimensions( rows, columns );

   /****
    * Allocate slice info array. Note that there cannot be
    * more slices than rows.
    */
   this->slices.setSize( this->rows );
   this->rowToChunkMapping.setSize( this-> rows );
   this->rowToSliceMapping.setSize( this->rows );
   this->rowPointers.setSize( this->rows + 1 );
}

template< typename Real,
          typename Device,
          typename Index >
void ChunkedEllpack< Real, Device, Index >::resolveSliceSizes( ConstRowsCapacitiesTypeView rowLengths )
{
   /****
    * Iterate over rows and allocate slices so that each slice has
    * approximately the same number of allocated elements
    */
   const IndexType desiredElementsInSlice =
            this->chunksInSlice * this->desiredChunkSize;

   IndexType row( 0 ),
             sliceSize( 0 ),
             allocatedElementsInSlice( 0 );
   numberOfSlices = 0;
   while( row < this->rows )
   {
      /****
       * Add one row to the current slice until we reach the desired
       * number of elements in a slice.
       */
      allocatedElementsInSlice += rowLengths[ row ];
      sliceSize++;
      row++;
      if( allocatedElementsInSlice < desiredElementsInSlice  )
          if( row < this->rows && sliceSize < chunksInSlice ) continue;
      TNL_ASSERT( sliceSize >0, );
      this->slices[ numberOfSlices ].size = sliceSize;
      this->slices[ numberOfSlices ].firstRow = row - sliceSize;
      this->slices[ numberOfSlices ].pointer = allocatedElementsInSlice; // this is only temporary
      sliceSize = 0;
      numberOfSlices++;
      allocatedElementsInSlice = 0;
   }
}

template< typename Real,
          typename Device,
          typename Index >
bool ChunkedEllpack< Real, Device, Index >::setSlice( ConstRowsCapacitiesTypeView rowLengths,
                                                               const IndexType sliceIndex,
                                                               IndexType& elementsToAllocation )
{
   /****
    * Now, compute the number of chunks per each row.
    * Each row get one chunk by default.
    * Then each row will get additional chunks w.r. to the
    * number of the elements in the row. If there are some
    * free chunks left, repeat it again.
    */
   const IndexType sliceSize = this->slices[ sliceIndex ].size;
   const IndexType sliceBegin = this->slices[ sliceIndex ].firstRow;
   const IndexType allocatedElementsInSlice = this->slices[ sliceIndex ].pointer;
   const IndexType sliceEnd = sliceBegin + sliceSize;

   IndexType freeChunks = this->chunksInSlice - sliceSize;
   for( IndexType i = sliceBegin; i < sliceEnd; i++ )
      this->rowToChunkMapping.setElement( i, 1 );

   int totalAddedChunks( 0 );
   int maxRowLength( rowLengths[ sliceBegin ] );
   for( IndexType i = sliceBegin; i < sliceEnd; i++ )
   {
      double rowRatio( 0.0 );
      if( allocatedElementsInSlice != 0 )
         rowRatio = ( double ) rowLengths[ i ] / ( double ) allocatedElementsInSlice;
      const IndexType addedChunks = freeChunks * rowRatio;
      totalAddedChunks += addedChunks;
      this->rowToChunkMapping[ i ] += addedChunks;
      if( maxRowLength < rowLengths[ i ] )
         maxRowLength = rowLengths[ i ];
   }
   freeChunks -= totalAddedChunks;
   while( freeChunks )
      for( IndexType i = sliceBegin; i < sliceEnd && freeChunks; i++ )
         if( rowLengths[ i ] == maxRowLength )
         {
            this->rowToChunkMapping[ i ]++;
            freeChunks--;
         }

   /****
    * Compute the chunk size
    */
   IndexType maxChunkInSlice( 0 );
   for( IndexType i = sliceBegin; i < sliceEnd; i++ )
   {
       maxChunkInSlice = max( maxChunkInSlice,
                          roundUpDivision( rowLengths[ i ], this->rowToChunkMapping[ i ] ) );
   }
      TNL_ASSERT( maxChunkInSlice > 0,
              std::cerr << " maxChunkInSlice = " << maxChunkInSlice << std::endl );

   /****
    * Set-up the slice info.
    */
   this->slices[ sliceIndex ].chunkSize = maxChunkInSlice;
   this->slices[ sliceIndex ].pointer = elementsToAllocation;
   elementsToAllocation += this->chunksInSlice * maxChunkInSlice;

   for( IndexType i = sliceBegin; i < sliceEnd; i++ )
      this->rowToSliceMapping[ i ] = sliceIndex;

   for( IndexType i = sliceBegin; i < sliceEnd; i++ )
   {
      this->rowPointers[ i + 1 ] = maxChunkInSlice*rowToChunkMapping[ i ];
      TNL_ASSERT( this->rowPointers[ i ] >= 0,
                 std::cerr << "this->rowPointers[ i ] = " << this->rowPointers[ i ] );
      TNL_ASSERT( this->rowPointers[ i + 1 ] >= 0,
                 std::cerr << "this->rowPointers[ i + 1 ] = " << this->rowPointers[ i + 1 ] );
   }

   /****
    * Finish the row to chunk mapping by computing the prefix sum.
    */
   for( IndexType j = sliceBegin + 1; j < sliceEnd; j++ )
      rowToChunkMapping[ j ] += rowToChunkMapping[ j - 1 ];
   return true;
}

template< typename Real,
          typename Device,
          typename Index >
void ChunkedEllpack< Real, Device, Index >::setCompressedRowLengths( ConstRowsCapacitiesTypeView rowLengths )
{
   TNL_ASSERT_GT( this->getRows(), 0, "cannot set row lengths of an empty matrix" );
   TNL_ASSERT_GT( this->getColumns(), 0, "cannot set row lengths of an empty matrix" );
   TNL_ASSERT_EQ( this->getRows(), rowLengths.getSize(), "wrong size of the rowLengths vector" );

   IndexType elementsToAllocation( 0 );

   if( std::is_same< Device, Devices::Host >::value )
   {
      DeviceDependentCode::resolveSliceSizes( *this, rowLengths );
      this->rowPointers.setElement( 0, 0 );
      for( IndexType sliceIndex = 0; sliceIndex < numberOfSlices; sliceIndex++ )
         this->setSlice( rowLengths, sliceIndex, elementsToAllocation );
      Algorithms::inplaceInclusiveScan( this->rowPointers );
   }

   if( std::is_same< Device, Devices::Cuda >::value )
   {
      ChunkedEllpack< RealType, Devices::Host, IndexType > hostMatrix;
      hostMatrix.setDimensions( this->getRows(), this->getColumns() );
      Containers::Vector< IndexType, Devices::Host, IndexType > hostCompressedRowLengths;
      hostCompressedRowLengths.setLike( rowLengths);
      hostCompressedRowLengths = rowLengths;
      hostMatrix.setNumberOfChunksInSlice( this->chunksInSlice );
      hostMatrix.setDesiredChunkSize( this->desiredChunkSize );
      hostMatrix.setCompressedRowLengths( hostCompressedRowLengths );

      this->rowToChunkMapping.setLike( hostMatrix.rowToChunkMapping );
      this->rowToChunkMapping = hostMatrix.rowToChunkMapping;
      this->rowToSliceMapping.setLike( hostMatrix.rowToSliceMapping );
      this->rowToSliceMapping = hostMatrix.rowToSliceMapping;
      this->rowPointers.setLike( hostMatrix.rowPointers );
      this->rowPointers = hostMatrix.rowPointers;
      this->slices.setLike( hostMatrix.slices );
      this->slices = hostMatrix.slices;
      this->numberOfSlices = hostMatrix.numberOfSlices;
      elementsToAllocation = hostMatrix.values.getSize();
   }
   this->maxRowLength = max( rowLengths );
   Sparse< Real, Device, Index >::allocateMatrixElements( elementsToAllocation );
}

template< typename Real,
          typename Device,
          typename Index >
void ChunkedEllpack< Real, Device, Index >::setRowCapacities( ConstRowsCapacitiesTypeView rowLengths )
{
   setCompressedRowLengths( rowLengths );
}

template< typename Real,
          typename Device,
          typename Index >
Index ChunkedEllpack< Real, Device, Index >::getRowLength( const IndexType row ) const
{
   const IndexType& sliceIndex = rowToSliceMapping.getElement( row );
   TNL_ASSERT( sliceIndex < this->rows, );
   const IndexType& chunkSize = slices.getElement( sliceIndex ).chunkSize;
   return rowPointers.getElement( row + 1 ) - rowPointers.getElement( row );
}

template< typename Real,
          typename Device,
          typename Index >
__cuda_callable__
Index ChunkedEllpack< Real, Device, Index >::getRowLengthFast( const IndexType row ) const
{
   const IndexType& sliceIndex = rowToSliceMapping[ row ];
   TNL_ASSERT( sliceIndex < this->rows, );
   const IndexType& chunkSize = slices[ sliceIndex ].chunkSize;
   return rowPointers[ row + 1 ] - rowPointers[ row ];
}

template< typename Real,
          typename Device,
          typename Index >
Index ChunkedEllpack< Real, Device, Index >::getNonZeroRowLength( const IndexType row ) const
{
    ConstMatrixRow matrixRow = getRow( row );
    return matrixRow.getNonZeroElementsCount( Device::getDeviceType() );
}

template< typename Real,
          typename Device,
          typename Index >
   template< typename Real2,
             typename Device2,
             typename Index2 >
void ChunkedEllpack< Real, Device, Index >::setLike( const ChunkedEllpack< Real2, Device2, Index2 >& matrix )
{
   this->chunksInSlice = matrix.chunksInSlice;
   this->desiredChunkSize = matrix.desiredChunkSize;
   Sparse< Real, Device, Index >::setLike( matrix );
   this->rowToChunkMapping.setLike( matrix.rowToChunkMapping );
   this->rowToSliceMapping.setLike( matrix.rowToSliceMapping );
   this->slices.setLike( matrix.slices );
}

template< typename Real,
          typename Device,
          typename Index >
void ChunkedEllpack< Real, Device, Index >::reset()
{
   Sparse< Real, Device, Index >::reset();
   this->slices.reset();
   this->rowToChunkMapping.reset();
   this->rowToSliceMapping.reset();
}

template< typename Real,
          typename Device,
          typename Index >
void ChunkedEllpack< Real, Device, Index >::setNumberOfChunksInSlice( const IndexType chunksInSlice )
{
   this->chunksInSlice = chunksInSlice;
}

template< typename Real,
          typename Device,
          typename Index >
__cuda_callable__
Index ChunkedEllpack< Real, Device, Index >::getNumberOfChunksInSlice() const
{
   return this->chunksInSlice;
}

template< typename Real,
          typename Device,
          typename Index >
void ChunkedEllpack< Real, Device, Index >::setDesiredChunkSize( const IndexType desiredChunkSize )
{
   this->desiredChunkSize = desiredChunkSize;
}

template< typename Real,
          typename Device,
          typename Index >
Index ChunkedEllpack< Real, Device, Index >::getDesiredChunkSize() const
{
   return this->desiredChunkSize;
}

template< typename Real,
          typename Device,
          typename Index >
__cuda_callable__
Index ChunkedEllpack< Real, Device, Index >::getNumberOfSlices() const
{
   return this->numberOfSlices;
}

template< typename Real,
          typename Device,
          typename Index >
   template< typename Real2,
             typename Device2,
             typename Index2 >
bool ChunkedEllpack< Real, Device, Index >::operator == ( const ChunkedEllpack< Real2, Device2, Index2 >& matrix ) const
{
   TNL_ASSERT( this->getRows() == matrix.getRows() &&
              this->getColumns() == matrix.getColumns(),
              std::cerr << "this->getRows() = " << this->getRows()
                   << " matrix.getRows() = " << matrix.getRows()
                   << " this->getColumns() = " << this->getColumns()
                   << " matrix.getColumns() = " << matrix.getColumns() );
   // TODO: implement this
   return false;
}

template< typename Real,
          typename Device,
          typename Index >
   template< typename Real2,
             typename Device2,
             typename Index2 >
bool ChunkedEllpack< Real, Device, Index >::operator != ( const ChunkedEllpack< Real2, Device2, Index2 >& matrix ) const
{
   return ! ( ( *this ) == matrix );
}

template< typename Real,
          typename Device,
          typename Index >
__cuda_callable__
bool ChunkedEllpack< Real, Device, Index >::setElementFast( const IndexType row,
                                                                     const IndexType column,
                                                                     const Real& value )
{
   return this->addElementFast( row, column, value, 0.0 );
}

template< typename Real,
          typename Device,
          typename Index >
bool ChunkedEllpack< Real, Device, Index >::setElement( const IndexType row,
                                                                 const IndexType column,
                                                                 const Real& value )
{
   return this->addElement( row, column, value, 0.0 );
}

template< typename Real,
          typename Device,
          typename Index >
__cuda_callable__
bool ChunkedEllpack< Real, Device, Index >::addElementFast( const IndexType row,
                                                                     const IndexType _column,
                                                                     const RealType& _value,
                                                                     const RealType& _thisElementMultiplicator )
{
   // TODO: return this back when CUDA kernels support std::cerr
   /*TNL_ASSERT( row >= 0 && row < this->rows &&
              _column >= 0 && _column <= this->columns,
              std::cerr << " row = " << row
                   << " column = " << _column
                   << " this->rows = " << this->rows
                   << " this->columns = " << this-> columns );*/

   const IndexType& sliceIndex = rowToSliceMapping[ row ];
   TNL_ASSERT( sliceIndex < this->rows, );
   IndexType chunkIndex( 0 );
   if( row != slices[ sliceIndex ].firstRow )
      chunkIndex = rowToChunkMapping[ row - 1 ];
   const IndexType lastChunk = rowToChunkMapping[ row ];
   const IndexType sliceOffset = slices[ sliceIndex ].pointer;
   const IndexType chunkSize = slices[ sliceIndex ].chunkSize;
   IndexType column( _column );
   RealType value( _value ), thisElementMultiplicator( _thisElementMultiplicator );
   while( chunkIndex < lastChunk - 1 &&
          ! addElementToChunkFast( sliceOffset, chunkIndex, chunkSize, column, value, thisElementMultiplicator ) )
      chunkIndex++;
   if( chunkIndex < lastChunk - 1 )
      return true;
   return addElementToChunkFast( sliceOffset, chunkIndex, chunkSize, column, value, thisElementMultiplicator );
}

template< typename Real,
          typename Device,
          typename Index >
__cuda_callable__
bool ChunkedEllpack< Real, Device, Index >::addElementToChunkFast( const IndexType sliceOffset,
                                                                            const IndexType chunkIndex,
                                                                            const IndexType chunkSize,
                                                                            IndexType& column,
                                                                            RealType& value,
                                                                            RealType& thisElementMultiplicator )
{
   IndexType elementPtr, chunkEnd, step;

   DeviceDependentCode::initChunkTraverse( sliceOffset,
                                           chunkIndex,
                                           chunkSize,
                                           this->getNumberOfChunksInSlice(),
                                           elementPtr,
                                           chunkEnd,
                                           step );
   IndexType col = 0;
   while( elementPtr < chunkEnd &&
          ( col = this->columnIndexes[ elementPtr ] ) < column &&
          col != this->getPaddingIndex() )
      elementPtr += step;

   if( col == column )
   {
      if( thisElementMultiplicator != 0.0 )
         this->values[ elementPtr ] = value + thisElementMultiplicator * this->values[ elementPtr ];
      else
         this->values[ elementPtr ] = value;
      return true;
   }
   if( col < column )
      return false;

   IndexType i( chunkEnd - step );

   /****
    * Check if the chunk is already full. In this case, the last element
    * will be inserted to the next chunk.
    */
   IndexType elementColumn( column );
   RealType elementValue( value );
   bool chunkOverflow( false );
   if( ( col = this->columnIndexes[ i ] ) != this->getPaddingIndex() )
   {
      chunkOverflow = true;
      column = col;
      value = this->values[ i ];
      thisElementMultiplicator = 0;
   }

   while( i > elementPtr )
   {
      this->columnIndexes[ i ] = this->columnIndexes[ i - step ];
      this->values[ i ] = this->values[ i - step ];
      i -= step;
   }
   this->values[ elementPtr ] = elementValue;
   this->columnIndexes[ elementPtr ] = elementColumn;
   return ! chunkOverflow;
}


template< typename Real,
          typename Device,
          typename Index >
bool ChunkedEllpack< Real, Device, Index >::addElement( const IndexType row,
                                                                 const IndexType _column,
                                                                 const RealType& _value,
                                                                 const RealType& _thisElementMultiplicator )
{
   TNL_ASSERT( row >= 0 && row < this->rows &&
              _column >= 0 && _column <= this->columns,
              std::cerr << " row = " << row
                   << " column = " << _column
                   << " this->rows = " << this->rows
                   << " this->columns = " << this-> columns );

   const IndexType& sliceIndex = rowToSliceMapping.getElement( row );
   TNL_ASSERT( sliceIndex < this->rows, );
   IndexType chunkIndex( 0 );
   if( row != slices.getElement( sliceIndex ).firstRow )
      chunkIndex = rowToChunkMapping.getElement( row - 1 );
   const IndexType lastChunk = rowToChunkMapping.getElement( row );
   const IndexType sliceOffset = slices.getElement( sliceIndex ).pointer;
   const IndexType chunkSize = slices.getElement( sliceIndex ).chunkSize;
   IndexType column( _column );
   RealType value( _value ), thisElementMultiplicator( _thisElementMultiplicator );
   while( chunkIndex < lastChunk - 1 &&
          ! addElementToChunk( sliceOffset, chunkIndex, chunkSize, column, value, thisElementMultiplicator ) )
      chunkIndex++;
   if( chunkIndex < lastChunk - 1 )
      return true;
   return addElementToChunk( sliceOffset, chunkIndex, chunkSize, column, value, thisElementMultiplicator );
}

template< typename Real,
          typename Device,
          typename Index >
bool ChunkedEllpack< Real, Device, Index >::addElementToChunk( const IndexType sliceOffset,
                                                                        const IndexType chunkIndex,
                                                                        const IndexType chunkSize,
                                                                        IndexType& column,
                                                                        RealType& value,
                                                                        RealType& thisElementMultiplicator )
{
   IndexType elementPtr, chunkEnd, step;
   DeviceDependentCode::initChunkTraverse( sliceOffset,
                                           chunkIndex,
                                           chunkSize,
                                           this->getNumberOfChunksInSlice(),
                                           elementPtr,
                                           chunkEnd,
                                           step );
   IndexType col = 0;
   while( elementPtr < chunkEnd &&
          ( col = this->columnIndexes.getElement( elementPtr ) ) < column &&
          col != this->getPaddingIndex() )
      elementPtr += step;
   if( col == column )
   {
      if( thisElementMultiplicator != 0.0 )
         this->values.setElement( elementPtr, value + thisElementMultiplicator * this->values.getElement( elementPtr ) );
      else
         this->values.setElement( elementPtr, value );
      return true;
   }
   if( col < column )
      return false;

   IndexType i( chunkEnd - step );

   /****
    * Check if the chunk is already full. In this case, the last element
    * will be inserted to the next chunk.
    */
   IndexType elementColumn( column );
   RealType elementValue( value );
   bool chunkOverflow( false );
   if( ( col = this->columnIndexes.getElement( i ) ) != this->getPaddingIndex() )
   {
      chunkOverflow = true;
      column = col;
      value = this->values.getElement( i );
      thisElementMultiplicator = 0;
   }

   while( i > elementPtr )
   {
      this->columnIndexes.setElement( i, this->columnIndexes.getElement( i - step ) );
      this->values.setElement( i, this->values.getElement( i - step ) );
      i -= step;
   }
   this->values.setElement( elementPtr, elementValue );
   this->columnIndexes.setElement( elementPtr, elementColumn );
   return ! chunkOverflow;
}

template< typename Real,
          typename Device,
          typename Index >
__cuda_callable__
bool ChunkedEllpack< Real, Device, Index >::setRowFast( const IndexType row,
                                                                 const IndexType* columnIndexes,
                                                                 const RealType* values,
                                                                 const IndexType elements )
{
   // TODO: return this back when CUDA kernels support std::cerr
   /*TNL_ASSERT( row >= 0 && row < this->rows,
              std::cerr << " row = " << row
                   << " this->rows = " << this->rows );*/
   const IndexType sliceIndex = rowToSliceMapping[ row ];
   //TNL_ASSERT( sliceIndex < this->rows, );
   IndexType chunkIndex( 0 );
   if( row != slices[ sliceIndex ].firstRow )
      chunkIndex = rowToChunkMapping[ row - 1 ];
   const IndexType lastChunk = rowToChunkMapping[ row ];
   const IndexType sliceOffset = slices[ sliceIndex ].pointer;
   const IndexType chunkSize = slices[ sliceIndex ].chunkSize;
   if( chunkSize * ( lastChunk - chunkIndex ) < elements )
      return false;
   IndexType offset( 0 );
   while( chunkIndex < lastChunk )
   {
      /****
       * Note, if elements - offset is non-positive then setChunkFast
       * just erase the chunk.
       */
      setChunkFast( sliceOffset,
                    chunkIndex,
                    chunkSize,
                    &columnIndexes[ offset ],
                    &values[ offset ],
                    elements - offset );
      chunkIndex++;
      offset += chunkSize;
   }
   return true;
}

template< typename Real,
          typename Device,
          typename Index >
__cuda_callable__
void ChunkedEllpack< Real, Device, Index >::setChunkFast( const IndexType sliceOffset,
                                                                   const IndexType chunkIndex,
                                                                   const IndexType chunkSize,
                                                                   const IndexType* columnIndexes,
                                                                   const RealType* values,
                                                                   const IndexType elements )
{
   IndexType elementPtr, chunkEnd, step;
   DeviceDependentCode::initChunkTraverse( sliceOffset,
                                           chunkIndex,
                                           chunkSize,
                                           this->getNumberOfChunksInSlice(),
                                           elementPtr,
                                           chunkEnd,
                                           step );
   IndexType i( 0 );
   while( i < chunkSize && i < elements )
   {
      this->values[ elementPtr ] = values[ i ];
      this->columnIndexes[ elementPtr ] = columnIndexes[ i ];
      i++;
      elementPtr += step;
   }
   while( i < chunkSize )
   {
      this->columnIndexes[ elementPtr ] = this->getPaddingIndex();
      elementPtr += step;
      i++;
   }
}

template< typename Real,
          typename Device,
          typename Index >
bool ChunkedEllpack< Real, Device, Index >::setRow( const IndexType row,
                                                             const IndexType* columnIndexes,
                                                             const RealType* values,
                                                             const IndexType elements )
{
   TNL_ASSERT( row >= 0 && row < this->rows,
              std::cerr << " row = " << row
                   << " this->rows = " << this->rows );

   const IndexType sliceIndex = rowToSliceMapping.getElement( row );
   TNL_ASSERT( sliceIndex < this->rows, );
   IndexType chunkIndex( 0 );
   if( row != slices.getElement( sliceIndex ).firstRow )
      chunkIndex = rowToChunkMapping.getElement( row - 1 );
   const IndexType lastChunk = rowToChunkMapping.getElement( row );
   const IndexType sliceOffset = slices.getElement( sliceIndex ).pointer;
   const IndexType chunkSize = slices.getElement( sliceIndex ).chunkSize;
   if( chunkSize * ( lastChunk - chunkIndex ) < elements )
      return false;
   IndexType offset( 0 );
   while( chunkIndex < lastChunk )
   {
      /****
       * Note, if elements - offset is non-positive then setChunkFast
       * just erase the chunk.
       */
      setChunk( sliceOffset,
                chunkIndex,
                chunkSize,
                &columnIndexes[ offset ],
                &values[ offset ],
                elements - offset );
      chunkIndex++;
      offset += chunkSize;
   }
   return true;
}

template< typename Real,
          typename Device,
          typename Index >
void ChunkedEllpack< Real, Device, Index >::setChunk( const IndexType sliceOffset,
                                                               const IndexType chunkIndex,
                                                               const IndexType chunkSize,
                                                               const IndexType* columnIndexes,
                                                               const RealType* values,
                                                               const IndexType elements )
{
   IndexType elementPtr, chunkEnd, step;
   DeviceDependentCode::initChunkTraverse( sliceOffset,
                                           chunkIndex,
                                           chunkSize,
                                           this->getNumberOfChunksInSlice(),
                                           elementPtr,
                                           chunkEnd,
                                           step );
   IndexType i( 0 );
   while( i < chunkSize && i < elements )
   {
      this->values.setElement( elementPtr, values[ i ] );
      this->columnIndexes.setElement( elementPtr, columnIndexes[ i ] );
      i++;
      elementPtr += step;
   }
   while( i < chunkSize )
   {
      this->columnIndexes.setElement( elementPtr, this->getPaddingIndex() );
      elementPtr += step;
      i++;
   }
}

template< typename Real,
          typename Device,
          typename Index >
__cuda_callable__
bool ChunkedEllpack< Real, Device, Index > :: addRowFast( const IndexType row,
                                                                   const IndexType* columns,
                                                                   const RealType* values,
                                                                   const IndexType numberOfElements,
                                                                   const RealType& thisElementMultiplicator )
{
   // TODO: implement
   return false;
}

template< typename Real,
          typename Device,
          typename Index >
bool ChunkedEllpack< Real, Device, Index > :: addRow( const IndexType row,
                                                               const IndexType* columns,
                                                               const RealType* values,
                                                               const IndexType numberOfElements,
                                                               const RealType& thisElementMultiplicator )
{
   return this->addRowFast( row, columns, values, numberOfElements, thisElementMultiplicator );
}

template< typename Real,
          typename Device,
          typename Index >
__cuda_callable__
Real ChunkedEllpack< Real, Device, Index >::getElementFast( const IndexType row,
                                                                     const IndexType column ) const
{
   const IndexType sliceIndex = rowToSliceMapping[ row ];
   TNL_ASSERT( sliceIndex < this->rows, );
   IndexType chunkIndex( 0 );
   if( row != slices[ sliceIndex ].firstRow )
      chunkIndex = rowToChunkMapping[ row - 1 ];
   const IndexType lastChunk = rowToChunkMapping[ row ];
   const IndexType sliceOffset = slices[ sliceIndex ].pointer;
   const IndexType chunkSize = slices[ sliceIndex ].chunkSize;
   RealType value( 0.0 );
   while( chunkIndex < lastChunk &&
          ! getElementInChunk( sliceOffset, chunkIndex, chunkSize, column, value ) )
      chunkIndex++;
   return value;
}

template< typename Real,
          typename Device,
          typename Index >
__cuda_callable__
bool ChunkedEllpack< Real, Device, Index >::getElementInChunkFast( const IndexType sliceOffset,
                                                                            const IndexType chunkIndex,
                                                                            const IndexType chunkSize,
                                                                            const IndexType column,
                                                                            RealType& value) const
{
   IndexType elementPtr, chunkEnd, step;
   DeviceDependentCode::initChunkTraverse( sliceOffset, chunkIndex, chunkSize, elementPtr, chunkEnd, step );
   while( elementPtr < chunkEnd )
   {
      const IndexType col = this->columnIndexes[ elementPtr ];
      if( col == column )
         value = this->values[ elementPtr ];
      if( col >= column || col == this->getPaddingIndex() )
         return true;
      elementPtr += step;
   }
   return false;
}


template< typename Real,
          typename Device,
          typename Index >
Real ChunkedEllpack< Real, Device, Index >::getElement( const IndexType row,
                                                                 const IndexType column ) const
{
   const IndexType& sliceIndex = rowToSliceMapping.getElement( row );
   TNL_ASSERT( sliceIndex < this->rows,
              std::cerr << " sliceIndex = " << sliceIndex
                   << " this->rows = " << this->rows << std::endl; );
   IndexType chunkIndex( 0 );
   if( row != slices.getElement( sliceIndex ).firstRow )
      chunkIndex = rowToChunkMapping.getElement( row - 1 );
   const IndexType lastChunk = rowToChunkMapping.getElement( row );
   const IndexType sliceOffset = slices.getElement( sliceIndex ).pointer;
   const IndexType chunkSize = slices.getElement( sliceIndex ).chunkSize;
   RealType value( 0.0 );
   while( chunkIndex < lastChunk &&
          ! getElementInChunk( sliceOffset, chunkIndex, chunkSize, column, value ) )
      chunkIndex++;
   return value;
}

template< typename Real,
          typename Device,
          typename Index >
bool ChunkedEllpack< Real, Device, Index >::getElementInChunk( const IndexType sliceOffset,
                                                                        const IndexType chunkIndex,
                                                                        const IndexType chunkSize,
                                                                        const IndexType column,
                                                                        RealType& value) const
{
   IndexType elementPtr, chunkEnd, step;
   DeviceDependentCode::initChunkTraverse( sliceOffset,
                                           chunkIndex,
                                           chunkSize,
                                           this->getNumberOfChunksInSlice(),
                                           elementPtr,
                                           chunkEnd,
                                           step );
   while( elementPtr < chunkEnd )
   {
      const IndexType col = this->columnIndexes.getElement( elementPtr );
      if( col == column )
         value = this->values.getElement( elementPtr );
      if( col >= column || col == this->getPaddingIndex() )
         return true;
      elementPtr += step;
   }
   return false;
}


template< typename Real,
          typename Device,
          typename Index >
__cuda_callable__
void ChunkedEllpack< Real, Device, Index >::getRowFast( const IndexType row,
                                                                 IndexType* columns,
                                                                 RealType* values ) const
{
   const IndexType& sliceIndex = rowToSliceMapping[ row ];
   TNL_ASSERT( sliceIndex < this->rows, );
   IndexType chunkIndex( 0 );
   if( row != slices[ sliceIndex ].firstRow )
      chunkIndex = rowToChunkMapping[ row - 1 ];
   const IndexType lastChunk = rowToChunkMapping[ row ];
   const IndexType sliceOffset = slices[ sliceIndex ].pointer;
   const IndexType chunkSize = slices[ sliceIndex ].chunkSize;
   RealType value( 0.0 );
   IndexType offset( 0 );
   while( chunkIndex < lastChunk )
   {
      getChunk( sliceOffset,
                chunkIndex,
                min( chunkSize, this->getColumns - offset ),
                &columns[ offset ],
                &values[ offset ] );
      chunkIndex++;
      offset += chunkSize;
   }
}

template< typename Real,
          typename Device,
          typename Index >
__cuda_callable__
void ChunkedEllpack< Real, Device, Index >::getChunkFast( const IndexType sliceOffset,
                                                                   const IndexType chunkIndex,
                                                                   const IndexType chunkSize,
                                                                   IndexType* columnIndexes,
                                                                   RealType* values ) const
{
   IndexType elementPtr, chunkEnd, step;
   DeviceDependentCode::initChunkTraverse( sliceOffset, chunkIndex, chunkSize, elementPtr, chunkEnd, step );
   IndexType i( 0 );
   while( i < chunkSize )
   {
      columnIndexes[ i ] = this->columnIndexes[ elementPtr ];
      values[ i ] = this->values[ elementPtr ];
      i++;
      elementPtr += step;
   }
}

template< typename Real,
          typename Device,
          typename Index >
__cuda_callable__
typename ChunkedEllpack< Real, Device, Index >::MatrixRow
ChunkedEllpack< Real, Device, Index >::
getRow( const IndexType rowIndex )
{
   const IndexType rowOffset = this->rowPointers[ rowIndex ];
   const IndexType rowLength = this->rowPointers[ rowIndex + 1 ] - rowOffset;
   return MatrixRow( &this->columnIndexes[ rowOffset ],
                     &this->values[ rowOffset ],
                     rowLength,
                     1 );
}

template< typename Real,
          typename Device,
          typename Index >
__cuda_callable__
typename ChunkedEllpack< Real, Device, Index >::ConstMatrixRow
ChunkedEllpack< Real, Device, Index >::
getRow( const IndexType rowIndex ) const
{
   const IndexType rowOffset = this->rowPointers[ rowIndex ];
   const IndexType rowLength = this->rowPointers[ rowIndex + 1 ] - rowOffset;
   return ConstMatrixRow( &this->columnIndexes[ rowOffset ],
                          &this->values[ rowOffset ],
                          rowLength,
                          1 );
}


/*template< typename Real,
          typename Device,
          typename Index >
void ChunkedEllpack< Real, Device, Index >::getRow( const IndexType row,
                                                             IndexType* columns,
                                                             RealType* values ) const
{
   const IndexType& sliceIndex = rowToSliceMapping.getElement( row );
   TNL_ASSERT( sliceIndex < this->rows, );
   IndexType chunkIndex( 0 );
   if( row != slices.getElement( sliceIndex ).firstRow )
      chunkIndex = rowToChunkMapping.getElement( row - 1 );
   const IndexType lastChunk = rowToChunkMapping.getElement( row );
   const IndexType sliceOffset = slices.getElement( sliceIndex ).pointer;
   const IndexType chunkSize = slices.getElement( sliceIndex ).chunkSize;
   RealType value( 0.0 );
   IndexType offset( 0 );
   while( chunkIndex < lastChunk )
   {
      getChunk( sliceOffset,
                chunkIndex,
                min( chunkSize, this->getColumns() - offset ),
                &columns[ offset ],
                &values[ offset ] );
      chunkIndex++;
      offset += chunkSize;
   }
}*/

template< typename Real,
          typename Device,
          typename Index >
void ChunkedEllpack< Real, Device, Index >::getChunk( const IndexType sliceOffset,
                                                               const IndexType chunkIndex,
                                                               const IndexType chunkSize,
                                                               IndexType* columnIndexes,
                                                               RealType* values ) const
{
   IndexType elementPtr, chunkEnd, step;
   DeviceDependentCode::initChunkTraverse( sliceOffset,
                                           chunkIndex,
                                           chunkSize,
                                           this->getNumberOfChunksInSlice(),
                                           elementPtr,
                                           chunkEnd,
                                           step );
   IndexType i( 0 );
   while( i < chunkSize )
   {
      columnIndexes[ i ] = this->columnIndexes.getElement( elementPtr );
      values[ i ] = this->values.getElement( elementPtr );
      i++;
      elementPtr += step;
   }
}

template< typename Real,
          typename Device,
          typename Index >
   template< typename Vector >
__cuda_callable__
typename Vector::RealType ChunkedEllpack< Real, Device, Index >::rowVectorProduct( const IndexType row,
                                                                                            const Vector& vector ) const
{
   /*TNL_ASSERT( row >=0 && row < this->rows,
            std::cerr << " row = " << row << " this->rows = " << this->rows );*/

   const IndexType sliceIndex = rowToSliceMapping[ row ];
   //TNL_ASSERT( sliceIndex < this->rows, );
   IndexType chunkIndex( 0 );
   if( row != slices[ sliceIndex ].firstRow )
      chunkIndex = rowToChunkMapping[ row - 1 ];
   const IndexType lastChunk = rowToChunkMapping[ row ];
   const IndexType sliceOffset = slices[ sliceIndex ].pointer;
   const IndexType chunkSize = slices[ sliceIndex ].chunkSize;
   RealType result( 0.0 );
   while( chunkIndex < lastChunk )
   {
      result += chunkVectorProduct( sliceOffset,
                                    chunkIndex,
                                    chunkSize,
                                    vector );
      chunkIndex++;
   }
   return result;
}

template< typename Real,
          typename Device,
          typename Index >
   template< typename Vector >
__cuda_callable__
typename Vector::RealType ChunkedEllpack< Real, Device, Index >::chunkVectorProduct( const IndexType sliceOffset,
                                                                                              const IndexType chunkIndex,
                                                                                              const IndexType chunkSize,
                                                                                              const Vector& vector ) const
{
   IndexType elementPtr, chunkEnd, step;
   DeviceDependentCode::initChunkTraverse( sliceOffset,
                                           chunkIndex,
                                           chunkSize,
                                           this->getNumberOfChunksInSlice(),
                                           elementPtr,
                                           chunkEnd,
                                           step );
   IndexType i( 0 ), col( 0 );
   typename Vector::RealType result( 0.0 );
   while( i < chunkSize && ( col = this->columnIndexes[ elementPtr ] ) != this->getPaddingIndex() )
   {
      result += this->values[ elementPtr ] * vector[ col ];
      i++;
      elementPtr += step;
   }
   return result;
}


#ifdef HAVE_CUDA
template< typename Real,
          typename Device,
          typename Index >
   template< typename InVector,
             typename OutVector >
__device__ void ChunkedEllpack< Real, Device, Index >::computeSliceVectorProduct( const InVector* inVector,
                                                                                           OutVector* outVector,
                                                                                           int sliceIdx  ) const
{
   static_assert( std::is_same < DeviceType, Devices::Cuda >::value, "" );

   RealType* chunkProducts = Cuda::getSharedMemory< RealType >();
   ChunkedEllpackSliceInfo* sliceInfo = ( ChunkedEllpackSliceInfo* ) & chunkProducts[ blockDim.x ];

   if( threadIdx.x == 0 )
      ( *sliceInfo ) = this->slices[ sliceIdx ];
   __syncthreads();
   chunkProducts[ threadIdx.x ] = this->chunkVectorProduct( sliceInfo->pointer,
                                                            threadIdx.x,
                                                            sliceInfo->chunkSize,
                                                            *inVector );
   __syncthreads();
   if( threadIdx.x < sliceInfo->size )
   {
      const IndexType row = sliceInfo->firstRow + threadIdx.x;
      IndexType chunkIndex( 0 );
      if( threadIdx.x != 0 )
         chunkIndex = this->rowToChunkMapping[ row - 1 ];
      const IndexType lastChunk = this->rowToChunkMapping[ row ];
      RealType result( 0.0 );
      while( chunkIndex < lastChunk )
         result += chunkProducts[ chunkIndex++ ];
      ( *outVector )[ row ] = result;
   }
}
#endif

template< typename Real,
          typename Device,
          typename Index >
   template< typename InVector,
             typename OutVector >
void ChunkedEllpack< Real, Device, Index >::vectorProduct( const InVector& inVector,
                                                                    OutVector& outVector ) const
{
   DeviceDependentCode::vectorProduct( *this, inVector, outVector );
}


template< typename Real,
          typename Device,
          typename Index >
   template< typename Real2,
             typename Index2 >
void ChunkedEllpack< Real, Device, Index >::addMatrix( const ChunkedEllpack< Real2, Device, Index2 >& matrix,
                                                                          const RealType& matrixMultiplicator,
                                                                          const RealType& thisMatrixMultiplicator )
{
   throw Exceptions::NotImplementedError( "ChunkedEllpack::addMatrix is not implemented." );
   // TODO: implement
}

template< typename Real,
          typename Device,
          typename Index >
   template< typename Real2,
             typename Index2 >
void ChunkedEllpack< Real, Device, Index >::getTransposition( const ChunkedEllpack< Real2, Device, Index2 >& matrix,
                                                                       const RealType& matrixMultiplicator )
{
   throw Exceptions::NotImplementedError( "ChunkedEllpack::getTransposition is not implemented." );
   // TODO: implement
}

template< typename Real,
          typename Device,
          typename Index >
   template< typename Vector1, typename Vector2 >
bool ChunkedEllpack< Real, Device, Index >::performSORIteration( const Vector1& b,
                                                                 const IndexType row,
                                                                 Vector2& x,
                                                                 const RealType& omega ) const
{
   TNL_ASSERT( row >=0 && row < this->getRows(),
              std::cerr << "row = " << row
                   << " this->getRows() = " << this->getRows() << std::endl );

   RealType diagonalValue( 0.0 );
   RealType sum( 0.0 );

   const IndexType& sliceIndex = rowToSliceMapping[ row ];
   TNL_ASSERT( sliceIndex < this->rows, );
   const IndexType& chunkSize = slices.getElement( sliceIndex ).chunkSize;
   IndexType elementPtr = rowPointers[ row ];
   const IndexType rowEnd = rowPointers[ row + 1 ];
   IndexType column;
   while( elementPtr < rowEnd && ( column = this->columnIndexes[ elementPtr ] ) < this->columns )
   {
      if( column == row )
         diagonalValue = this->values.getElement( elementPtr );
      else
         sum += this->values.getElement( row * this->diagonalsShift.getSize() + elementPtr ) * x. getElement( column );
      elementPtr++;
   }
   if( diagonalValue == ( Real ) 0.0 )
   {
      std::cerr << "There is zero on the diagonal in " << row << "-th row of a matrix. I cannot perform SOR iteration." << std::endl;
      return false;
   }
   x. setElement( row, x[ row ] + omega / diagonalValue * ( b[ row ] - sum ) );
   return true;
}


// copy assignment
template< typename Real,
          typename Device,
          typename Index >
ChunkedEllpack< Real, Device, Index >&
ChunkedEllpack< Real, Device, Index >::operator=( const ChunkedEllpack& matrix )
{
   this->setLike( matrix );
   this->values = matrix.values;
   this->columnIndexes = matrix.columnIndexes;
   this->chunksInSlice = matrix.chunksInSlice;
   this->desiredChunkSize = matrix.desiredChunkSize;
   this->rowToChunkMapping = matrix.rowToChunkMapping;
   this->rowToSliceMapping = matrix.rowToSliceMapping;
   this->rowPointers = matrix.rowPointers;
   this->slices = matrix.slices;
   this->numberOfSlices = matrix.numberOfSlices;
   return *this;
}

// cross-device copy assignment
template< typename Real,
          typename Device,
          typename Index >
   template< typename Real2, typename Device2, typename Index2, typename >
ChunkedEllpack< Real, Device, Index >&
ChunkedEllpack< Real, Device, Index >::operator=( const ChunkedEllpack< Real2, Device2, Index2 >& matrix )
{
   static_assert( std::is_same< Device, Devices::Host >::value || std::is_same< Device, Devices::Cuda >::value,
                  "unknown device" );
   static_assert( std::is_same< Device2, Devices::Host >::value || std::is_same< Device2, Devices::Cuda >::value,
                  "unknown device" );

   this->setLike( matrix );
   this->chunksInSlice = matrix.chunksInSlice;
   this->desiredChunkSize = matrix.desiredChunkSize;
   this->rowToChunkMapping = matrix.rowToChunkMapping;
   this->rowToSliceMapping = matrix.rowToSliceMapping;
   this->rowPointers = matrix.rowPointers;
   this->slices = matrix.slices;
   this->numberOfSlices = matrix.numberOfSlices;

   // host -> cuda
   if( std::is_same< Device, Devices::Cuda >::value ) {
       typename ValuesVector::template Self< typename ValuesVector::RealType, Devices::Host > tmpValues;
       typename ColumnIndexesVector::template Self< typename ColumnIndexesVector::RealType, Devices::Host > tmpColumnIndexes;
       tmpValues.setLike( matrix.values );
       tmpColumnIndexes.setLike( matrix.columnIndexes );

#ifdef HAVE_OPENMP
#pragma omp parallel for if( Devices::Host::isOMPEnabled() )
#endif
       for( Index sliceIdx = 0; sliceIdx < matrix.numberOfSlices; sliceIdx++ ) {
           const Index chunkSize = matrix.slices.getElement( sliceIdx ).chunkSize;
           const Index offset = matrix.slices.getElement( sliceIdx ).pointer;

           for( Index j = 0; j < chunkSize; j++ )
               for( Index i = 0; i < matrix.chunksInSlice; i++ ) {
                   tmpValues[ offset + j * matrix.chunksInSlice + i ] = matrix.values[ offset + i * chunkSize + j ];
                   tmpColumnIndexes[ offset + j * matrix.chunksInSlice + i ] = matrix.columnIndexes[ offset + i * chunkSize + j ];
               }
       }

       this->values = tmpValues;
       this->columnIndexes = tmpColumnIndexes;
   }

   // cuda -> host
   if( std::is_same< Device, Devices::Host >::value ) {
       ValuesVector tmpValues;
       ColumnIndexesVector tmpColumnIndexes;
       tmpValues.setLike( matrix.values );
       tmpColumnIndexes.setLike( matrix.columnIndexes );
       tmpValues = matrix.values;
       tmpColumnIndexes = matrix.columnIndexes;

#ifdef HAVE_OPENMP
#pragma omp parallel for if( Devices::Host::isOMPEnabled() )
#endif
       for( Index sliceIdx = 0; sliceIdx < matrix.numberOfSlices; sliceIdx++ ) {
           const Index chunkSize = matrix.slices.getElement( sliceIdx ).chunkSize;
           const Index offset = matrix.slices.getElement( sliceIdx ).pointer;

           for( Index j = 0; j < chunkSize; j++ )
               for( Index i = 0; i < matrix.chunksInSlice; i++ ) {
                   this->values[ offset + i * chunkSize + j ] = tmpValues[ offset + j * matrix.chunksInSlice + i ];
                   this->columnIndexes[ offset + i * chunkSize + j ] = tmpColumnIndexes[ offset + j * matrix.chunksInSlice + i ];
               }
       }
   }
   return *this;
}


template< typename Real,
          typename Device,
          typename Index >
void ChunkedEllpack< Real, Device, Index >::save( File& file ) const
{
   Sparse< Real, Device, Index >::save( file );
   file << this->rowToChunkMapping << this->rowToSliceMapping << this->rowPointers << this->slices;
}

template< typename Real,
          typename Device,
          typename Index >
void ChunkedEllpack< Real, Device, Index >::load( File& file )
{
   Sparse< Real, Device, Index >::load( file );
   file >> this->rowToChunkMapping >> this->rowToSliceMapping >> this->rowPointers >> this->slices;
}

template< typename Real,
          typename Device,
          typename Index >
void ChunkedEllpack< Real, Device, Index >::save( const String& fileName ) const
{
   Object::save( fileName );
}

template< typename Real,
          typename Device,
          typename Index >
void ChunkedEllpack< Real, Device, Index >::load( const String& fileName )
{
   Object::load( fileName );
}

template< typename Real,
          typename Device,
          typename Index >
void ChunkedEllpack< Real, Device, Index >::print( std::ostream& str ) const
{
   for( IndexType row = 0; row < this->getRows(); row++ )
   {
      str <<"Row: " << row << " -> ";

      //const IndexType& sliceIndex = rowToSliceMapping.getElement( row );
      //TNL_ASSERT( sliceIndex < this->rows, );
      //const IndexType& chunkSize = slices.getElement( sliceIndex ).chunkSize;
      IndexType elementPtr = rowPointers.getElement( row );
      const IndexType rowEnd = rowPointers.getElement( row + 1 );

      while( elementPtr < rowEnd &&
             this->columnIndexes.getElement( elementPtr ) < this->columns &&
             this->columnIndexes.getElement( elementPtr ) != this->getPaddingIndex() )
      {
         const Index column = this->columnIndexes.getElement( elementPtr );
         str << " Col:" << column << "->" << this->values.getElement( elementPtr ) << "\t";
         elementPtr++;
      }
      str << std::endl;
   }
}

template< typename Real,
          typename Device,
          typename Index >
void ChunkedEllpack< Real, Device, Index >::printStructure( std::ostream& str,
                                                                     const String& name ) const
{
   const IndexType numberOfSlices = this->getNumberOfSlices();
   str << "Matrix type: " << getType( *this ) << std::endl
       << "Marix name: " << name << std::endl
       << "Rows: " << this->getRows() << std::endl
       << "Columns: " << this->getColumns() << std::endl
       << "Slices: " << numberOfSlices << std::endl;
   for( IndexType i = 0; i < numberOfSlices; i++ )
      str << "   Slice " << i
          << " : size = " << this->slices.getElement( i ).size
          << " chunkSize = " << this->slices.getElement( i ).chunkSize
          << " firstRow = " << this->slices.getElement( i ).firstRow
          << " pointer = " << this->slices.getElement( i ).pointer << std::endl;
   for( IndexType i = 0; i < this->getRows(); i++ )
      str << "Row " << i
          << " : slice = " << this->rowToSliceMapping.getElement( i )
          << " chunk = " << this->rowToChunkMapping.getElement( i ) << std::endl;
}

template<>
class ChunkedEllpackDeviceDependentCode< Devices::Host >
{
   public:

      typedef Devices::Host Device;

      template< typename Real,
                typename Index >
      static void resolveSliceSizes( ChunkedEllpack< Real, Device, Index >& matrix,
                                     typename ChunkedEllpack< Real, Device, Index >::ConstRowsCapacitiesTypeView rowLengths )
      {
         matrix.resolveSliceSizes( rowLengths );
      }

      template< typename Index >
      __cuda_callable__
      static void initChunkTraverse( const Index sliceOffset,
                                     const Index chunkIndex,
                                     const Index chunkSize,
                                     const Index chunksInSlice,
                                     Index& chunkBegining,
                                     Index& chunkEnd,
                                     Index& step )
      {
         chunkBegining = sliceOffset + chunkIndex * chunkSize;
         chunkEnd = chunkBegining + chunkSize;
         step = 1;
      }

      template< typename Real,
                typename Index,
                typename InVector,
                typename OutVector >
      static void vectorProduct( const ChunkedEllpack< Real, Device, Index >& matrix,
                                 const InVector& inVector,
                                 OutVector& outVector )
      {
         for( Index row = 0; row < matrix.getRows(); row ++ )
            outVector[ row ] = matrix.rowVectorProduct( row, inVector );
      }
};

#ifdef HAVE_CUDA
template< typename Real,
          typename Index,
          typename InVector,
          typename OutVector >
__global__ void ChunkedEllpackVectorProductCudaKernel( const ChunkedEllpack< Real, Devices::Cuda, Index >* matrix,
                                                                const InVector* inVector,
                                                                OutVector* outVector,
                                                                int gridIdx )
{
   const Index sliceIdx = gridIdx * Cuda::getMaxGridSize() + blockIdx.x;
   if( sliceIdx < matrix->getNumberOfSlices() )
      matrix->computeSliceVectorProduct( inVector, outVector, sliceIdx );

}
#endif


template<>
class ChunkedEllpackDeviceDependentCode< Devices::Cuda >
{
   public:

      typedef Devices::Cuda Device;

      template< typename Real,
                typename Index >
      static void resolveSliceSizes( ChunkedEllpack< Real, Device, Index >& matrix,
                                     typename ChunkedEllpack< Real, Device, Index >::ConstRowsCapacitiesTypeView rowLengths )
      {
      }

      template< typename Index >
      __cuda_callable__
      static void initChunkTraverse( const Index sliceOffset,
                                     const Index chunkIndex,
                                     const Index chunkSize,
                                     const Index chunksInSlice,
                                     Index& chunkBegining,
                                     Index& chunkEnd,
                                     Index& step )
      {
         chunkBegining = sliceOffset + chunkIndex;
         chunkEnd = chunkBegining + chunkSize * chunksInSlice;
         step = chunksInSlice;

         /*chunkBegining = sliceOffset + chunkIndex * chunkSize;
         chunkEnd = chunkBegining + chunkSize;
         step = 1;*/
      }

      template< typename Real,
                typename Index,
                typename InVector,
                typename OutVector >
      static void vectorProduct( const ChunkedEllpack< Real, Device, Index >& matrix,
                                 const InVector& inVector,
                                 OutVector& outVector )
      {
         #ifdef HAVE_CUDA
            typedef ChunkedEllpack< Real, Devices::Cuda, Index > Matrix;
            typedef Index IndexType;
            typedef Real RealType;
            Matrix* kernel_this = Cuda::passToDevice( matrix );
            InVector* kernel_inVector = Cuda::passToDevice( inVector );
            OutVector* kernel_outVector = Cuda::passToDevice( outVector );
            dim3 cudaBlockSize( matrix.getNumberOfChunksInSlice() ),
                 cudaGridSize( Cuda::getMaxGridSize() );
            const IndexType cudaBlocks = matrix.getNumberOfSlices();
            const IndexType cudaGrids = roundUpDivision( cudaBlocks, Cuda::getMaxGridSize() );
            const IndexType sharedMemory = cudaBlockSize.x * sizeof( RealType ) +
                                           sizeof( tnlChunkedEllpackSliceInfo< IndexType > );
            for( IndexType gridIdx = 0; gridIdx < cudaGrids; gridIdx++ )
            {
               if( gridIdx == cudaGrids - 1 )
                  cudaGridSize.x = cudaBlocks % Cuda::getMaxGridSize();
               ChunkedEllpackVectorProductCudaKernel< Real, Index, InVector, OutVector >
                                                             <<< cudaGridSize, cudaBlockSize, sharedMemory  >>>
                                                             ( kernel_this,
                                                               kernel_inVector,
                                                               kernel_outVector,
                                                               gridIdx );
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
