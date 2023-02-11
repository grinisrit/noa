#pragma once

#include "CSR.h"
#include <TNL/Containers/VectorView.h>
#include <TNL/Algorithms/scan.h>
#include <TNL/Math.h>
#include <TNL/Algorithms/AtomicOperations.h>
#include <TNL/Exceptions/NotImplementedError.h>
#include <TNL/Atomic.h>
#include <vector> // for blocks in CSR Adaptive

#ifdef HAVE_CUSPARSE
#include <cuda.h>
#include <cusparse.h>
#endif

constexpr size_t MAX_X_DIM = 2147483647;

namespace TNL {
    namespace Benchmarks {
        namespace SpMV {
            namespace ReferenceFormats {
               namespace Legacy {

#ifdef HAVE_CUSPARSE
template< typename Real, typename Index >
class tnlCusparseCSRWrapper {};
#endif


template< typename Real,
          typename Device,
          typename Index,
          CSRKernel KernelType >
CSR< Real, Device, Index, KernelType >::CSR()
: //spmvCudaKernel( hybrid ),
  cudaWarpSize( 32 ), //Cuda::getWarpSize() )
  hybridModeSplit( 4 )
{
}

template< typename Real,
          typename Device,
          typename Index,
          CSRKernel KernelType >
std::string CSR< Real, Device, Index, KernelType >::getSerializationType()
{
   return "Matrices::CSR< "+
          TNL::getType< Real>() +
          ", [any_device], " +
          TNL::getType< Index >() +
          " >";
}

template< typename Real,
          typename Device,
          typename Index,
          CSRKernel KernelType >
std::string CSR< Real, Device, Index, KernelType >::getSerializationTypeVirtual() const
{
   return this->getSerializationType();
}

template< typename Real,
          typename Device,
          typename Index,
          CSRKernel KernelType >
void CSR< Real, Device, Index, KernelType >::setDimensions( const IndexType rows,
                                                const IndexType columns )
{
   Sparse< Real, Device, Index >::setDimensions( rows, columns );
   this->rowPointers.setSize( this->rows + 1 );
   this->rowPointers.setValue( 0 );
}

template< typename Real,
          typename Device,
          typename Index,
          CSRKernel KernelType >
void CSR< Real, Device, Index, KernelType >::setCompressedRowLengths( ConstRowsCapacitiesTypeView rowLengths )
{
   TNL_ASSERT_GT( this->getRows(), 0, "cannot set row lengths of an empty matrix" );
   TNL_ASSERT_GT( this->getColumns(), 0, "cannot set row lengths of an empty matrix" );
   TNL_ASSERT_EQ( this->getRows(), rowLengths.getSize(), "wrong size of the rowLengths vector" );

   /****
    * Compute the rows pointers. The last one is
    * the end of the last row and so it says the
    * necessary length of the vectors this->values
    * and this->columnIndexes.
    */
   Containers::VectorView< IndexType, DeviceType, IndexType > rowPtrs;
   rowPtrs.bind( this->rowPointers.getData(), this->getRows() );
   rowPtrs = rowLengths;
   this->rowPointers.setElement( this->rows, 0 );
   Algorithms::inplaceExclusiveScan( this->rowPointers );
   this->maxRowLength = max( rowLengths );

   /****
    * Allocate values and column indexes
    */
   this->values.setSize( this->rowPointers.getElement( this->rows ) );
   this->columnIndexes.setSize( this->rowPointers.getElement( this->rows ) );
   this->columnIndexes.setValue( this->columns );

   if( KernelType == CSRAdaptive )
      this->setBlocks();
}

template< typename Real,
          typename Device,
          typename Index,
          CSRKernel KernelType >
void CSR< Real, Device, Index, KernelType >::setRowCapacities( ConstRowsCapacitiesTypeView rowLengths )
{
   setCompressedRowLengths( rowLengths );
}

/* Find limit of block */
template< typename Real,
          typename Index,
          typename Device,
          CSRKernel KernelType>
Index findLimit(const Index start,
               const CSR< Real, Device, Index, KernelType >& matrix,
               const Index size,
               Type &type,
               Index &sum) {
   sum = 0;
   for( Index current = start; current < size - 1; ++current) {
      Index elements = matrix.getRowPointers().getElement(current + 1) -
                       matrix.getRowPointers().getElement(current);
      sum += elements;
      if (sum > matrix.SHARED_PER_WARP) {
         if (current - start > 0) { // extra row
            type = Type::STREAM;
            return current;
         } else {                  // one long row
            if (sum <= 2 * matrix.MAX_ELEMENTS_PER_WARP_ADAPT)
               type = Type::VECTOR;
            else
               type = Type::LONG;
            return current + 1;
         }
      }
   }

   type = Type::STREAM;
   return size - 1; // return last row pointer
}

template< typename Real,
          typename Device,
          typename Index,
          CSRKernel KernelType >
void CSR< Real, Device, Index, KernelType >::setBlocks()
{
   const Index rows = this->getRowPointers().getSize();
   Index sum, start = 0, nextStart = 0;

   /* Fill blocks */
   std::vector<Block<Index>> inBlock;
   inBlock.reserve(rows); // reserve space to avoid reallocation

   while( nextStart != rows - 1 )
   {
      Type type;
      nextStart = findLimit<Real, Index, Device, KernelType>(
         start, *this, rows, type, sum
      );
      if (type == Type::LONG)
      {
         Index parts = roundUpDivision(sum, this->SHARED_PER_WARP);
         for (Index index = 0; index < parts; ++index)
         {
            inBlock.emplace_back(start, Type::LONG, index);
         }
      }
      else
      {
         inBlock.emplace_back(start, type,
            nextStart,
            this->rowPointers.getElement(nextStart),
            this->rowPointers.getElement(start)
         );
      }
      start = nextStart;
   }
   inBlock.emplace_back(nextStart);

   /* Copy values */
   this->blocks = inBlock;
   /*this->blocks.setSize(inBlock.size());
   for (size_t i = 0; i < inBlock.size(); ++i)
      this->blocks.setElement(i, inBlock[i]);*/
}

template< typename Real,
          typename Device,
          typename Index,
          CSRKernel KernelType >
void CSR< Real, Device, Index, KernelType >::getCompressedRowLengths( RowsCapacitiesTypeView rowLengths ) const
{
   TNL_ASSERT_EQ( rowLengths.getSize(), this->getRows(), "invalid size of the rowLengths vector" );
   for( IndexType row = 0; row < this->getRows(); row++ )
      rowLengths.setElement( row, this->getRowLength( row ) );
}

template< typename Real,
          typename Device,
          typename Index,
          CSRKernel KernelType >
Index CSR< Real, Device, Index, KernelType >::getRowLength( const IndexType row ) const
{
   return this->rowPointers.getElement( row + 1 ) - this->rowPointers.getElement( row );
}

template< typename Real,
          typename Device,
          typename Index,
          CSRKernel KernelType >
__cuda_callable__
Index CSR< Real, Device, Index, KernelType >::getRowLengthFast( const IndexType row ) const
{
   return this->rowPointers[ row + 1 ] - this->rowPointers[ row ];
}

template< typename Real,
          typename Device,
          typename Index,
          CSRKernel KernelType >
Index CSR< Real, Device, Index, KernelType >::getNonZeroRowLength( const IndexType row ) const
{
    // TODO: Fix/Implement
    TNL_ASSERT( false, std::cerr << "TODO: Fix/Implement" );
    return 0;
}

template< typename Real,
          typename Device,
          typename Index,
          CSRKernel KernelType >
__cuda_callable__
Index CSR< Real, Device, Index, KernelType >::getNonZeroRowLengthFast( const IndexType row ) const
{
   ConstMatrixRow matrixRow = this->getRow( row );
   return matrixRow.getNonZeroElementsCount();
}

template< typename Real,
          typename Device,
          typename Index,
          CSRKernel KernelType >
   template< typename Real2,
             typename Device2,
             typename Index2,
             CSRKernel KernelType2 >
void CSR< Real, Device, Index, KernelType >::setLike( const CSR< Real2, Device2, Index2, KernelType2 >& matrix )
{
   Sparse< Real, Device, Index >::setLike( matrix );
   this->rowPointers.setLike( matrix.rowPointers );
}

template< typename Real,
          typename Device,
          typename Index,
          CSRKernel KernelType >
void CSR< Real, Device, Index, KernelType >::reset()
{
   Sparse< Real, Device, Index >::reset();
   this->rowPointers.reset();
}

template< typename Real,
          typename Device,
          typename Index,
          CSRKernel KernelType >
__cuda_callable__
bool CSR< Real, Device, Index, KernelType >::setElementFast( const IndexType row,
                                                          const IndexType column,
                                                          const Real& value )
{
   return this->addElementFast( row, column, value, 0.0 );
}

template< typename Real,
          typename Device,
          typename Index,
          CSRKernel KernelType >
bool CSR< Real, Device, Index, KernelType >::setElement( const IndexType row,
                                                      const IndexType column,
                                                      const Real& value )
{
   return this->addElement( row, column, value, 0.0 );
}


template< typename Real,
          typename Device,
          typename Index,
          CSRKernel KernelType >
__cuda_callable__
bool CSR< Real, Device, Index, KernelType >::addElementFast( const IndexType row,
                                                          const IndexType column,
                                                          const RealType& value,
                                                          const RealType& thisElementMultiplicator )
{
   IndexType elementPtr = this->rowPointers[ row ];
   const IndexType rowEnd = this->rowPointers[ row + 1 ];
   IndexType col = 0;
   while( elementPtr < rowEnd &&
          ( col = this->columnIndexes[ elementPtr ] ) < column &&
          col != this->getPaddingIndex() ) elementPtr++;
   if( elementPtr == rowEnd )
      return false;
   if( col == column )
   {
      this->values[ elementPtr ] = thisElementMultiplicator * this->values[ elementPtr ] + value;
      return true;
   }
   else
      if( col == this->getPaddingIndex() )
      {
         this->columnIndexes[ elementPtr ] = column;
         this->values[ elementPtr ] = value;
         return true;
      }
      else
      {
         IndexType j = rowEnd - 1;
         while( j > elementPtr )
         {
            this->columnIndexes[ j ] = this->columnIndexes[ j - 1 ];
            this->values[ j ] = this->values[ j - 1 ];
            j--;
         }
         this->columnIndexes[ elementPtr ] = column;
         this->values[ elementPtr ] = value;
         return true;
      }
}

template< typename Real,
          typename Device,
          typename Index,
          CSRKernel KernelType >
bool CSR< Real, Device, Index, KernelType >::addElement( const IndexType row,
                                                      const IndexType column,
                                                      const RealType& value,
                                                      const RealType& thisElementMultiplicator )
{
   TNL_ASSERT( row >= 0 && row < this->rows &&
               column >= 0 && column < this->columns,
               std::cerr << " row = " << row
                    << " column = " << column
                    << " this->rows = " << this->rows
                    << " this->columns = " << this->columns );

    IndexType elementPtr = this->rowPointers.getElement( row );
    const IndexType rowEnd = this->rowPointers.getElement( row + 1 );
    IndexType col = 0;
    while( elementPtr < rowEnd &&
           ( col = this->columnIndexes.getElement( elementPtr ) ) < column &&
           col != this->getPaddingIndex() ) elementPtr++;
    if( elementPtr == rowEnd )
       return false;
    if( col == column )
    {
       this->values.setElement( elementPtr, thisElementMultiplicator * this->values.getElement( elementPtr ) + value );
       return true;
    }
    else
       if( col == this->getPaddingIndex() )
       {
          this->columnIndexes.setElement( elementPtr, column );
          this->values.setElement( elementPtr, value );
          return true;
       }
       else
       {
          IndexType j = rowEnd - 1;
          while( j > elementPtr )
          {
             this->columnIndexes.setElement( j, this->columnIndexes.getElement( j - 1 ) );
             this->values.setElement( j, this->values.getElement( j - 1 ) );
             j--;
          }
          this->columnIndexes.setElement( elementPtr, column );
          this->values.setElement( elementPtr, value );
          return true;
       }
}

template< typename Real,
          typename Device,
          typename Index,
          CSRKernel KernelType >
__cuda_callable__
bool CSR< Real, Device, Index, KernelType > :: setRowFast( const IndexType row,
                                                        const IndexType* columnIndexes,
                                                        const RealType* values,
                                                        const IndexType elements )
{
   IndexType elementPointer = this->rowPointers[ row ];
   const IndexType rowLength = this->rowPointers[ row + 1 ] - elementPointer;
   if( elements > rowLength )
      return false;

   for( IndexType i = 0; i < elements; i++ )
   {
      //printf( "Setting element row: %d column: %d value: %f \n", row, columnIndexes[ i ], values[ i ] );
      this->columnIndexes[ elementPointer ] = columnIndexes[ i ];
      this->values[ elementPointer ] = values[ i ];
      elementPointer++;
   }
   for( IndexType i = elements; i < rowLength; i++ )
      this->columnIndexes[ elementPointer++ ] = this->getPaddingIndex();
   return true;
}

template< typename Real,
          typename Device,
          typename Index,
          CSRKernel KernelType >
bool CSR< Real, Device, Index, KernelType > :: setRow( const IndexType row,
                                                    const IndexType* columnIndexes,
                                                    const RealType* values,
                                                    const IndexType elements )
{
   IndexType elementPointer = this->rowPointers.getElement( row );
   const IndexType rowLength = this->rowPointers.getElement( row + 1 ) - elementPointer;
   if( elements > rowLength )
      return false;

   for( IndexType i = 0; i < elements; i++ )
   {
      this->columnIndexes.setElement( elementPointer, columnIndexes[ i ] );
      this->values.setElement( elementPointer, values[ i ] );
      elementPointer++;
   }
   for( IndexType i = elements; i < rowLength; i++ )
      this->columnIndexes.setElement( elementPointer++, this->getPaddingIndex() );
   return true;
}

template< typename Real,
          typename Device,
          typename Index,
          CSRKernel KernelType >
__cuda_callable__
bool CSR< Real, Device, Index, KernelType > :: addRowFast( const IndexType row,
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
          typename Index,
          CSRKernel KernelType >
bool CSR< Real, Device, Index, KernelType > :: addRow( const IndexType row,
                                                    const IndexType* columns,
                                                    const RealType* values,
                                                    const IndexType numberOfElements,
                                                    const RealType& thisElementMultiplicator )
{
   return this->addRowFast( row, columns, values, numberOfElements, thisElementMultiplicator );
}

template< typename Real,
          typename Device,
          typename Index,
          CSRKernel KernelType >
__cuda_callable__
Real CSR< Real, Device, Index, KernelType >::getElementFast( const IndexType row,
                                                          const IndexType column ) const
{
   IndexType elementPtr = this->rowPointers[ row ];
   const IndexType rowEnd = this->rowPointers[ row + 1 ];
   IndexType col = 0;
   while( elementPtr < rowEnd &&
          ( col = this->columnIndexes[ elementPtr ] ) < column &&
          col != this->getPaddingIndex() )
      elementPtr++;
   if( elementPtr < rowEnd && col == column )
      return this->values[ elementPtr ];
   return 0.0;
}

template< typename Real,
          typename Device,
          typename Index,
          CSRKernel KernelType >
Real CSR< Real, Device, Index, KernelType >::getElement( const IndexType row,
                                                      const IndexType column ) const
{
   IndexType elementPtr = this->rowPointers.getElement( row );
   const IndexType rowEnd = this->rowPointers.getElement( row + 1 );
   IndexType col = 0;
   while( elementPtr < rowEnd &&
          ( col = this->columnIndexes.getElement( elementPtr ) ) < column &&
          col != this->getPaddingIndex() )
      elementPtr++;
   if( elementPtr < rowEnd && col == column )
      return this->values.getElement( elementPtr );
   return 0.0;
}

template< typename Real,
          typename Device,
          typename Index,
          CSRKernel KernelType >
__cuda_callable__
void CSR< Real, Device, Index, KernelType >::getRowFast( const IndexType row,
                                                      IndexType* columns,
                                                      RealType* values ) const
{
   IndexType elementPointer = this->rowPointers[ row ];
   const IndexType rowLength = this->rowPointers[ row + 1 ] - elementPointer;
   for( IndexType i = 0; i < rowLength; i++ )
   {
      columns[ i ] = this->columnIndexes[ elementPointer ];
      values[ i ] = this->values[ elementPointer ];
      elementPointer++;
   }
}

template< typename Real,
          typename Device,
          typename Index,
          CSRKernel KernelType >
__cuda_callable__
typename CSR< Real, Device, Index, KernelType >::MatrixRow
CSR< Real, Device, Index, KernelType >::
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
          typename Index,
          CSRKernel KernelType >
__cuda_callable__
typename CSR< Real, Device, Index, KernelType >::ConstMatrixRow
CSR< Real, Device, Index, KernelType >::
getRow( const IndexType rowIndex ) const
{
    const IndexType rowOffset = this->rowPointers[ rowIndex ];
    const IndexType rowLength = this->rowPointers[ rowIndex + 1 ] - rowOffset;
    return ConstMatrixRow( &this->columnIndexes[ rowOffset ],
                           &this->values[ rowOffset ],
                           rowLength,
                           1 );
}

template< typename Real,
          typename Device,
          typename Index,
          CSRKernel KernelType >
   template< typename Vector >
__cuda_callable__
typename Vector::RealType CSR< Real, Device, Index, KernelType >::rowVectorProduct( const IndexType row,
                                                                                 const Vector& vector ) const
{
   Real result = 0.0;
   IndexType elementPtr = this->rowPointers[ row ];
   const IndexType rowEnd = this->rowPointers[ row + 1 ];
   IndexType column;
   while( elementPtr < rowEnd &&
          ( column = this->columnIndexes[ elementPtr ] ) != this->getPaddingIndex() )
      result += this->values[ elementPtr++ ] * vector[ column ];
   return result;
}

template< typename Real,
          typename Device,
          typename Index,
          CSRKernel KernelType >
   template< typename InVector, typename OutVector >
void CSR< Real, Device, Index, KernelType >::vectorProduct( const InVector& inVector,
                                                OutVector& outVector ) const
{
   DeviceDependentCode::vectorProduct( *this, inVector, outVector );
}

template< typename Real,
          typename Device,
          typename Index,
          CSRKernel KernelType >
   template< typename Real2,
             typename Index2,
             CSRKernel KernelType2 >
void CSR< Real, Device, Index, KernelType >::addMatrix( const CSR< Real2, Device, Index2, KernelType2 >& matrix,
                                            const RealType& matrixMultiplicator,
                                            const RealType& thisMatrixMultiplicator )
{
   throw Exceptions::NotImplementedError( "CSR::addMatrix is not implemented." );
   // TODO: implement
}

template< typename Real,
          typename Device,
          typename Index,
          CSRKernel KernelType >
   template< typename Real2,
             typename Index2,
             CSRKernel KernelType2 >
void CSR< Real, Device, Index, KernelType >::getTransposition( const CSR< Real2, Device, Index2, KernelType2 >& matrix,
                                                                      const RealType& matrixMultiplicator )
{
   throw Exceptions::NotImplementedError( "CSR::getTransposition is not implemented." );
   // TODO: implement
}

// copy assignment
template< typename Real,
          typename Device,
          typename Index,
          CSRKernel KernelType >
CSR< Real, Device, Index, KernelType >&
CSR< Real, Device, Index, KernelType >::operator=( const CSR& matrix )
{
   this->setLike( matrix );
   this->values = matrix.values;
   this->columnIndexes = matrix.columnIndexes;
   this->rowPointers = matrix.rowPointers;
   this->blocks = matrix.blocks;
   return *this;
}

template< typename Real,
          typename Device,
          typename Index,
          CSRKernel KernelType >
   template< typename IndexType2, CSRKernel KernelType2 >
CSR< Real, Device, Index, KernelType >&
CSR< Real, Device, Index, KernelType >::
operator=( const CSR< Real, Device, IndexType2, KernelType2 >& matrix )
{
   this->setLike( matrix );
   this->values = matrix.values;
   this->columnIndexes = matrix.columnIndexes;
   this->rowPointers = matrix.rowPointers;
   this->blocks = matrix.blocks;
   return *this;
}

// cross-device copy assignment
template< typename Real,
          typename Device,
          typename Index,
          CSRKernel KernelType >
   template< typename Real2, typename Device2, typename Index2, CSRKernel KernelType2, typename >
CSR< Real, Device, Index, KernelType >&
CSR< Real, Device, Index, KernelType >::operator=( const CSR< Real2, Device2, Index2, KernelType2 >& matrix )
{
   this->setLike( matrix );
   this->values = matrix.values;
   this->columnIndexes = matrix.columnIndexes;
   this->rowPointers = matrix.rowPointers;
   if( KernelType == CSRAdaptive )
      this->setBlocks();
   return *this;
}


template< typename Real,
          typename Device,
          typename Index,
          CSRKernel KernelType >
void CSR< Real, Device, Index, KernelType >::save( File& file ) const
{
   Sparse< Real, Device, Index >::save( file );
   file << this->rowPointers;
}

template< typename Real,
          typename Device,
          typename Index,
          CSRKernel KernelType >
void CSR< Real, Device, Index, KernelType >::load( File& file )
{
   Sparse< Real, Device, Index >::load( file );
   file >> this->rowPointers;
}

template< typename Real,
          typename Device,
          typename Index,
          CSRKernel KernelType >
void CSR< Real, Device, Index, KernelType >::save( const String& fileName ) const
{
   Object::save( fileName );
}

template< typename Real,
          typename Device,
          typename Index,
          CSRKernel KernelType >
void CSR< Real, Device, Index, KernelType >::load( const String& fileName )
{
   Object::load( fileName );
}

template< typename Real,
          typename Device,
          typename Index,
          CSRKernel KernelType >
void CSR< Real, Device, Index, KernelType >::print( std::ostream& str ) const
{
   for( IndexType row = 0; row < this->getRows(); row++ )
   {
      str <<"Row: " << row << " -> ";
      IndexType elementPtr = this->rowPointers.getElement( row );
      const IndexType rowEnd = this->rowPointers.getElement( row + 1 );
      IndexType column;
      while( elementPtr < rowEnd &&
             ( column = this->columnIndexes.getElement( elementPtr ) ) < this->columns &&
             column != this->getPaddingIndex() )
         str << " Col:" << column << "->" << this->values.getElement( elementPtr++ ) << "\t";
      str << std::endl;
   }
}

/*template< typename Real,
          typename Device,
          typename Index >
void CSR< Real, Device, Index, KernelType >::setCudaKernelType( const SPMVCudaKernel kernel )
{
   this->spmvCudaKernel = kernel;
}

template< typename Real,
          typename Device,
          typename Index >
__cuda_callable__
typename CSR< Real, Device, Index, KernelType >::SPMVCudaKernel CSR< Real, Device, Index, KernelType >::getCudaKernelType() const
{
   return this->spmvCudaKernel;
}*/

template< typename Real,
          typename Device,
          typename Index,
          CSRKernel KernelType >
void CSR< Real, Device, Index, KernelType >::setCudaWarpSize( const int warpSize )
{
   this->cudaWarpSize = warpSize;
}

template< typename Real,
          typename Device,
          typename Index,
          CSRKernel KernelType >
int CSR< Real, Device, Index, KernelType >::getCudaWarpSize() const
{
   return this->cudaWarpSize;
}

template< typename Real,
          typename Device,
          typename Index,
          CSRKernel KernelType >
void CSR< Real, Device, Index, KernelType >::setHybridModeSplit( const IndexType hybridModeSplit )
{
   this->hybridModeSplit = hybridModeSplit;
}

template< typename Real,
          typename Device,
          typename Index,
          CSRKernel KernelType >
__cuda_callable__
Index CSR< Real, Device, Index, KernelType >::getHybridModeSplit() const
{
   return this->hybridModeSplit;
}

#ifdef __CUDACC__


template< typename Real,
          typename Index>
__global__
void SpMVCSRScalar( const Real *inVector,
                    Real* outVector,
                    const Index* rowPointers,
                    const Index* columnIndexes,
                    const Real* values,
                    const Index rows,
                    const Index gridID) {
   const Index row = (gridID * MAX_X_DIM) + (blockIdx.x * blockDim.x) + threadIdx.x;
   if (row >= rows)
      return;

   Real result = 0.0;
   const Index endID = rowPointers[row + 1];

   for (Index i = rowPointers[row]; i < endID; ++i)
      result += values[i] * inVector[columnIndexes[i]];

   outVector[row] = result;
}

template< typename Real,
          typename Index,
          int warpSize >
__global__
void SpMVCSRMultiVector( const Real *inVector,
                         Real* outVector,
                         const Index* rowPointers,
                         const Index* columnIndexes,
                         const Real* values,
                         const Index rows,
                         const Index warps, // warps per row
                         const Index gridID)
{
   const Index warpID =
      ((gridID * MAX_X_DIM) + (blockIdx.x * blockDim.x) + threadIdx.x) / warpSize;
   const Index rowID = warpID / warps;
   if (rowID >= rows)
      return;

   const Index laneID = threadIdx.x & 31; // & is cheaper than %
   const Index offset = warps * warpSize;

   Real result = 0.0;
   Index endID = rowPointers[rowID + 1];
   /* Calculate result */
   for (Index i = rowPointers[rowID] + (warpID % warps) * warpSize + laneID;
            i < endID; i += offset) {
      result += values[i] * inVector[columnIndexes[i]];
   }

   /* Reduction */
   result += __shfl_down_sync(0xFFFFFFFF, result, 16);
   result += __shfl_down_sync(0xFFFFFFFF, result, 8);
   result += __shfl_down_sync(0xFFFFFFFF, result, 4);
   result += __shfl_down_sync(0xFFFFFFFF, result, 2);
   result += __shfl_down_sync(0xFFFFFFFF, result, 1);
   /* Write result */
   if (laneID == 0) atomicAdd(&outVector[rowID], result);
}

template< typename Real,
          typename Index,
          int warpSize >
__global__
void SpMVCSRVector( const Real *inVector,
                    Real* outVector,
                    const Index* rowPointers,
                    const Index* columnIndexes,
                    const Real* values,
                    const Index rows,
                    const Index gridID)
{
   const Index warpID = ((gridID * MAX_X_DIM) + (blockIdx.x * blockDim.x) + threadIdx.x) / warpSize;
   if (warpID >= rows)
      return;

   Real result = 0.0;
   const Index laneID = threadIdx.x & 31; // & is cheaper than %
   Index endID = rowPointers[warpID + 1];

   /* Calculate result */
   for (Index i = rowPointers[warpID] + laneID; i < endID; i += warpSize)
      result += values[i] * inVector[columnIndexes[i]];

   /* Reduction */
   result += __shfl_down_sync(0xFFFFFFFF, result, 16);
   result += __shfl_down_sync(0xFFFFFFFF, result, 8);
   result += __shfl_down_sync(0xFFFFFFFF, result, 4);
   result += __shfl_down_sync(0xFFFFFFFF, result, 2);
   result += __shfl_down_sync(0xFFFFFFFF, result, 1);
   /* Write result */
   if (laneID == 0) outVector[warpID] = result;
}

template< typename Real,
          typename Index,
          int groupSize,
          int MAX_NUM_VECTORS_PER_BLOCK >
__global__
void SpMVCSRLight( const Real *inVector,
                   Real* outVector,
                   const Index* rowPointers,
                   const Index* columnIndexes,
                   const Real* values,
                   const Index rows,
                   unsigned *rowCnt) {
   Real sum;
   Index row = 0, i, rowStart, rowEnd;
   const Index laneId = threadIdx.x % groupSize; /*lane index in the vector*/
   const Index vectorId = threadIdx.x / groupSize; /*vector index in the thread block*/
   const Index warpLaneId = threadIdx.x & 31;	/*lane index in the warp*/
   const Index warpVectorId = warpLaneId / groupSize;	/*vector index in the warp*/

   __shared__ volatile Index space[MAX_NUM_VECTORS_PER_BLOCK][2];

   /*get the row index*/
   if (warpLaneId == 0) {
      row = atomicAdd(rowCnt, 32 / groupSize);
   }
   /*broadcast the value to other threads in the same warp and compute the row index of each vector*/
   row = __shfl_sync(0xFFFFFFFF, row, 0) + warpVectorId;

   /*check the row range*/
   while (row < rows) {

      /*use two threads to fetch the row offset*/
      if (laneId < 2)
         space[vectorId][laneId] = rowPointers[row + laneId];

      rowStart = space[vectorId][0];
      rowEnd = space[vectorId][1];

      /*there are non-zero elements in the current row*/
      sum = 0;
      /*compute dot product*/
      if (groupSize == 32) {

         /*ensure aligned memory access*/
         i = rowStart - (rowStart & (groupSize - 1)) + laneId;

         /*process the unaligned part*/
         if (i >= rowStart && i < rowEnd)
            sum += values[i] * inVector[columnIndexes[i]];

         /*process the aligned part*/
         for (i += groupSize; i < rowEnd; i += groupSize)
            sum += values[i] * inVector[columnIndexes[i]];
      } else {
         /*regardless of the global memory access alignment*/
         for (i = rowStart + laneId; i < rowEnd; i += groupSize)
            sum += values[i] * inVector[columnIndexes[i]];
      }
      /*intra-vector reduction*/
      for (i = groupSize >> 1; i > 0; i >>= 1)
         sum += __shfl_down_sync(0xFFFFFFFF, sum, i);

      /*save the results and get a new row*/
      if (laneId == 0)
         outVector[row] = sum;

      /*get a new row index*/
      if(warpLaneId == 0)
         row = atomicAdd(rowCnt, 32 / groupSize);

      /*broadcast the row index to the other threads in the same warp and compute the row index of each vetor*/
      row = __shfl_sync(0xFFFFFFFF, row, 0) + warpVectorId;

	}/*while*/
}

/* Original CSR Light without shared memory */
template< typename Real,
          typename Index,
          int groupSize >
__global__
void SpMVCSRLight2( const Real *inVector,
                   Real* outVector,
                   const Index* rowPointers,
                   const Index* columnIndexes,
                   const Real* values,
                   const Index rows,
                   unsigned *rowCnt) {
   Real sum;
   Index i, rowStart, rowEnd, row = 0;
   const Index laneId = threadIdx.x % groupSize; /*lane index in the vector*/
   const Index warpLaneId = threadIdx.x & 31;	/*lane index in the warp*/
   const Index warpVectorId = warpLaneId / groupSize;	/*vector index in the warp*/

   /*get the row index*/
   if (warpLaneId == 0)
      row = atomicAdd(rowCnt, 32 / groupSize);

   /*broadcast the value to other threads in the same warp and compute the row index of each vector*/
   row = __shfl_sync(0xFFFFFFFF, row, 0) + warpVectorId;

   /*check the row range*/
   while (row < rows) {

      rowStart = rowPointers[row];
      rowEnd = rowPointers[row + 1];

      /*there are non-zero elements in the current row*/
      sum = 0;
      /*compute dot product*/
      if (groupSize == 32) {

         /*ensure aligned memory access*/
         i = rowStart - (rowStart & (groupSize - 1)) + laneId;

         /*process the unaligned part*/
         if (i >= rowStart && i < rowEnd)
            sum += values[i] * inVector[columnIndexes[i]];

         /*process the aligned part*/
         for (i += groupSize; i < rowEnd; i += groupSize)
            sum += values[i] * inVector[columnIndexes[i]];
      } else {
         /*regardless of the global memory access alignment*/
         for (i = rowStart + laneId; i < rowEnd; i += groupSize)
            sum += values[i] * inVector[columnIndexes[i]];
      }
      /*intra-vector reduction*/
      for (i = groupSize >> 1; i > 0; i >>= 1)
         sum += __shfl_down_sync(0xFFFFFFFF, sum, i);

      /*save the results and get a new row*/
      if (laneId == 0)
         outVector[row] = sum;

      /*get a new row index*/
      if(warpLaneId == 0)
         row = atomicAdd(rowCnt, 32 / groupSize);

      /*broadcast the row index to the other threads in the same warp and compute the row index of each vetor*/
      row = __shfl_sync(0xFFFFFFFF, row, 0) + warpVectorId;

	}/*while*/
}

/* Original CSR Light without shared memory and allign memory access */
template< typename Real,
          typename Index,
          int groupSize >
__global__
void SpMVCSRLight3( const Real *inVector,
                   Real* outVector,
                   const Index* rowPointers,
                   const Index* columnIndexes,
                   const Real* values,
                   const Index rows,
                   unsigned *rowCnt) {
   Real sum;
   Index i, rowEnd, row = 0;
   const Index laneId = threadIdx.x % groupSize; /*lane index in the vector*/
   const Index warpLaneId = threadIdx.x & 31;	/*lane index in the warp*/
   const Index warpVectorId = warpLaneId / groupSize;	/*vector index in the warp*/

   /*get the row index*/
   if (warpLaneId == 0)
      row = atomicAdd(rowCnt, 32 / groupSize);

   /*broadcast the value to other threads in the same warp and compute the row index of each vector*/
   row = __shfl_sync(0xFFFFFFFF, row, 0) + warpVectorId;

   /*check the row range*/
   while (row < rows) {
      sum = 0;

      /*compute dot product*/
      rowEnd = rowPointers[row + 1];
      for (i = rowPointers[row] + laneId; i < rowEnd; i += groupSize)
         sum += values[i] * inVector[columnIndexes[i]];

      /*intra-vector reduction*/
      for (i = groupSize >> 1; i > 0; i >>= 1)
         sum += __shfl_down_sync(0xFFFFFFFF, sum, i);

      /*save the results and get a new row*/
      if (laneId == 0)
         outVector[row] = sum;

      /*get a new row index*/
      if(warpLaneId == 0)
         row = atomicAdd(rowCnt, 32 / groupSize);

      /*broadcast the row index to the other threads in the same warp and compute the row index of each vetor*/
      row = __shfl_sync(0xFFFFFFFF, row, 0) + warpVectorId;

	}/*while*/
}

/* Original CSR Light without shared memory, allign memory access and atomic instructions */
template< typename Real,
          typename Index,
          int groupSize >
__global__
void SpMVCSRLight4( const Real *inVector,
                   Real* outVector,
                   const Index* rowPointers,
                   const Index* columnIndexes,
                   const Real* values,
                   const Index rows,
                   const Index gridID) {
   const Index row = ((gridID * MAX_X_DIM) + (blockIdx.x * blockDim.x) + threadIdx.x) / groupSize;
   if (row >= rows)
      return;

   Real sum = 0;
   Index i;
   const Index laneId = threadIdx.x & (groupSize - 1);	/*lane index in the group*/

   /*compute dot product*/
   const Index rowEnd = rowPointers[row + 1];
   for (i = rowPointers[row] + laneId; i < rowEnd; i += groupSize)
      sum += values[i] * inVector[columnIndexes[i]];

   /*intra-vector reduction*/
   for (i = groupSize >> 1; i > 0; i >>= 1)
      sum += __shfl_down_sync(0xFFFFFFFF, sum, i);

   /*save the results and get a new row*/
   if (laneId == 0) outVector[row] = sum;
}

template< typename Real,
          typename Index>
__global__
void SpMVCSRLightWithoutAtomic2( const Real *inVector,
                                 Real* outVector,
                                 const Index* rowPointers,
                                 const Index* columnIndexes,
                                 const Real* values,
                                 const Index rows,
                                 const Index gridID) {
   const Index row =
      ((gridID * MAX_X_DIM) + (blockIdx.x * blockDim.x) + threadIdx.x) / 2;
   if (row >= rows)
      return;

   const Index inGroupID = threadIdx.x & 1; // & is cheaper than %
   const Index maxID = rowPointers[row + 1];

   Real result = 0.0;
   for (Index i = rowPointers[row] + inGroupID; i < maxID; i += 2)
      result += values[i] * inVector[columnIndexes[i]];

   /* Parallel reduction */
   result += __shfl_down_sync(0xFFFFFFFF, result, 1);

   /* Write result */
   if (inGroupID == 0) outVector[row] = result;
}

template< typename Real,
          typename Index>
__global__
void SpMVCSRLightWithoutAtomic4( const Real *inVector,
                                 Real* outVector,
                                 const Index* rowPointers,
                                 const Index* columnIndexes,
                                 const Real* values,
                                 const Index rows,
                                 const Index gridID) {
   const Index row =
      ((gridID * MAX_X_DIM) + (blockIdx.x * blockDim.x) + threadIdx.x) / 4;
   if (row >= rows)
      return;

   const Index inGroupID = threadIdx.x & 3; // & is cheaper than %
   const Index maxID = rowPointers[row + 1];

   Real result = 0.0;
   for (Index i = rowPointers[row] + inGroupID; i < maxID; i += 4)
      result += values[i] * inVector[columnIndexes[i]];

   /* Parallel reduction */
   result += __shfl_down_sync(0xFFFFFFFF, result, 2);
   result += __shfl_down_sync(0xFFFFFFFF, result, 1);

   /* Write result */
   if (inGroupID == 0) outVector[row] = result;
}

template< typename Real,
          typename Index>
__global__
void SpMVCSRLightWithoutAtomic8( const Real *inVector,
                                 Real* outVector,
                                 const Index* rowPointers,
                                 const Index* columnIndexes,
                                 const Real* values,
                                 const Index rows,
                                 const Index gridID) {
   const Index row =
      ((gridID * MAX_X_DIM) + (blockIdx.x * blockDim.x) + threadIdx.x) / 8;
   if (row >= rows)
      return;

   Index i;
   const Index inGroupID = threadIdx.x & 7; // & is cheaper than %
   const Index maxID = rowPointers[row + 1];

   Real result = 0.0;
   for (i = rowPointers[row] + inGroupID; i < maxID; i += 8)
      result += values[i] * inVector[columnIndexes[i]];

   /* Parallel reduction */
   result += __shfl_down_sync(0xFFFFFFFF, result, 4);
   result += __shfl_down_sync(0xFFFFFFFF, result, 2);
   result += __shfl_down_sync(0xFFFFFFFF, result, 1);

   /* Write result */
   if (inGroupID == 0) outVector[row] = result;
}

template< typename Real,
          typename Index>
__global__
void SpMVCSRLightWithoutAtomic16( const Real *inVector,
                                  Real* outVector,
                                  const Index* rowPointers,
                                  const Index* columnIndexes,
                                  const Real* values,
                                  const Index rows,
                                  const Index gridID) {
   const Index row =
      ((gridID * MAX_X_DIM) + (blockIdx.x * blockDim.x) + threadIdx.x) / 16;
   if (row >= rows)
      return;


   Index i;
   const Index inGroupID = threadIdx.x & 15; // & is cheaper than %
   const Index maxID = rowPointers[row + 1];

   Real result = 0.0;
   for (i = rowPointers[row] + inGroupID; i < maxID; i += 16)
      result += values[i] * inVector[columnIndexes[i]];

   /* Parallel reduction */
   result += __shfl_down_sync(0xFFFFFFFF, result, 8);
   result += __shfl_down_sync(0xFFFFFFFF, result, 4);
   result += __shfl_down_sync(0xFFFFFFFF, result, 2);
   result += __shfl_down_sync(0xFFFFFFFF, result, 1);

   /* Write result */
   if (inGroupID == 0) outVector[row] = result;
}

template< typename Real,
          typename Index,
          typename Device,
          CSRKernel KernelType>
void SpMVCSRScalarPrepare( const Real *inVector,
                           Real* outVector,
                           const CSR< Real, Device, Index, KernelType >& matrix) {
   const Index threads = matrix.THREADS_SCALAR; // block size
   size_t neededThreads = matrix.getRowPointers().getSize() - 1;
   Index blocks;
   /* Execute kernels on device */
   for (Index grid = 0; neededThreads != 0; ++grid) {
      if (MAX_X_DIM * threads >= neededThreads) {
         blocks = roundUpDivision(neededThreads, threads);
         neededThreads = 0;
      } else {
         blocks = MAX_X_DIM;
         neededThreads -= MAX_X_DIM * threads;
      }

      SpMVCSRScalar<Real, Index><<<blocks, threads>>>(
               inVector,
               outVector,
               matrix.getRowPointers().getData(),
               matrix.getColumnIndexes().getData(),
               matrix.getValues().getData(),
               matrix.getRowPointers().getSize() - 1,
               grid
      );
   }
}

template< typename Real,
          typename Index,
          typename Device,
          CSRKernel KernelType,
          int warpSize >
void SpMVCSRVectorPrepare( const Real *inVector,
                           Real* outVector,
                           const CSR< Real, Device, Index, KernelType >& matrix) {
   const Index threads = matrix.THREADS_VECTOR; // block size
   size_t neededThreads = matrix.getRowPointers().getSize() * warpSize;
   Index blocks;
   /* Execute kernels on device */
   for (Index grid = 0; neededThreads != 0; ++grid) {
      if (MAX_X_DIM * threads >= neededThreads) {
         blocks = roundUpDivision(neededThreads, threads);
         neededThreads = 0;
      } else {
         blocks = MAX_X_DIM;
         neededThreads -= MAX_X_DIM * threads;
      }

      SpMVCSRVector<Real, Index, warpSize><<<blocks, threads>>>(
               inVector,
               outVector,
               matrix.getRowPointers().getData(),
               matrix.getColumnIndexes().getData(),
               matrix.getValues().getData(),
               matrix.getRowPointers().getSize() - 1,
               grid
      );
   }
}

template< typename Real,
          typename Index,
          typename Device,
          CSRKernel KernelType,
          int warpSize >
void SpMVCSRLightPrepare( const Real *inVector,
                          Real* outVector,
                          const CSR< Real, Device, Index, KernelType >& matrix) {
   const Index threads = 1024; // max block size
   const Index rows = matrix.getRowPointers().getSize() - 1;
   /* Copy rowCnt to GPU */
   unsigned rowCnt = 0;
   unsigned *kernelRowCnt = nullptr;
   cudaMalloc((void **)&kernelRowCnt, sizeof(*kernelRowCnt));
   cudaMemcpy(kernelRowCnt, &rowCnt, sizeof(*kernelRowCnt), cudaMemcpyHostToDevice);
   /* Get info about GPU */
   cudaDeviceProp properties;
   cudaGetDeviceProperties( &properties, Cuda::DeviceInfo::getActiveDevice() );
   const Index blocks =
      properties.multiProcessorCount * properties.maxThreadsPerMultiProcessor / threads;

   const Index nnz = roundUpDivision(matrix.getValues().getSize(), rows); // non zeroes per row
   if (KernelType == CSRLight) { //-----------------------------------------
      if (nnz <= 2)
         SpMVCSRLight<Real, Index, 2, 1024 / 2><<<blocks, threads>>>(
            inVector, outVector, matrix.getRowPointers().getData(),
            matrix.getColumnIndexes().getData(), matrix.getValues().getData(),
            rows, kernelRowCnt
         );
      else if (nnz <= 4)
         SpMVCSRLight<Real, Index, 4, 1024 / 4><<<blocks, threads>>>(
            inVector, outVector, matrix.getRowPointers().getData(),
            matrix.getColumnIndexes().getData(), matrix.getValues().getData(),
            rows, kernelRowCnt
         );
      else if (nnz <= 64)
         SpMVCSRLight<Real, Index, 8, 1024 / 8><<<blocks, threads>>>(
            inVector, outVector, matrix.getRowPointers().getData(),
            matrix.getColumnIndexes().getData(), matrix.getValues().getData(),
            rows, kernelRowCnt
         );
      else
         SpMVCSRLight<Real, Index, 32, 1024 / 32><<<blocks, threads>>>(
            inVector, outVector, matrix.getRowPointers().getData(),
            matrix.getColumnIndexes().getData(), matrix.getValues().getData(),
            rows, kernelRowCnt
         );
   } else if(KernelType == CSRLight2) { //-----------------------------------------
      if (nnz <= 2)
         SpMVCSRLight2<Real, Index, 2><<<blocks, threads>>>(
            inVector, outVector, matrix.getRowPointers().getData(),
            matrix.getColumnIndexes().getData(), matrix.getValues().getData(),
            rows, kernelRowCnt
         );
      else if (nnz <= 4)
         SpMVCSRLight2<Real, Index, 4><<<blocks, threads>>>(
            inVector, outVector, matrix.getRowPointers().getData(),
            matrix.getColumnIndexes().getData(), matrix.getValues().getData(),
            rows, kernelRowCnt
         );
      else if (nnz <= 64)
         SpMVCSRLight2<Real, Index, 8><<<blocks, threads>>>(
            inVector, outVector, matrix.getRowPointers().getData(),
            matrix.getColumnIndexes().getData(), matrix.getValues().getData(),
            rows, kernelRowCnt
         );
      else
         SpMVCSRLight2<Real, Index, 32><<<blocks, threads>>>(
            inVector, outVector, matrix.getRowPointers().getData(),
            matrix.getColumnIndexes().getData(), matrix.getValues().getData(),
            rows, kernelRowCnt
         );
   } else if(KernelType == CSRLight3) { //-----------------------------------------
      if (nnz <= 2)
         SpMVCSRLight3<Real, Index, 2><<<blocks, threads>>>(
            inVector, outVector, matrix.getRowPointers().getData(),
            matrix.getColumnIndexes().getData(), matrix.getValues().getData(),
            rows, kernelRowCnt
         );
      else if (nnz <= 4)
         SpMVCSRLight3<Real, Index, 4><<<blocks, threads>>>(
            inVector, outVector, matrix.getRowPointers().getData(),
            matrix.getColumnIndexes().getData(), matrix.getValues().getData(),
            rows, kernelRowCnt
         );
      else if (nnz <= 64)
         SpMVCSRLight3<Real, Index, 8><<<blocks, threads>>>(
            inVector, outVector, matrix.getRowPointers().getData(),
            matrix.getColumnIndexes().getData(), matrix.getValues().getData(),
            rows, kernelRowCnt
         );
      else
         SpMVCSRLight3<Real, Index, 32><<<blocks, threads>>>(
            inVector, outVector, matrix.getRowPointers().getData(),
            matrix.getColumnIndexes().getData(), matrix.getValues().getData(),
            rows, kernelRowCnt
         );
   } else if(KernelType == CSRLight6) { //-----------------------------------------
      if (nnz <= 2)
         SpMVCSRLight3<Real, Index, 2><<<blocks, threads>>>(
            inVector, outVector, matrix.getRowPointers().getData(),
            matrix.getColumnIndexes().getData(), matrix.getValues().getData(),
            rows, kernelRowCnt
         );
      else if (nnz <= 4)
         SpMVCSRLight3<Real, Index, 4><<<blocks, threads>>>(
            inVector, outVector, matrix.getRowPointers().getData(),
            matrix.getColumnIndexes().getData(), matrix.getValues().getData(),
            rows, kernelRowCnt
         );
      else if (nnz <= 8)
         SpMVCSRLight3<Real, Index, 8><<<blocks, threads>>>(
            inVector, outVector, matrix.getRowPointers().getData(),
            matrix.getColumnIndexes().getData(), matrix.getValues().getData(),
            rows, kernelRowCnt
         );
      else if (nnz <= 16)
         SpMVCSRLight3<Real, Index, 16><<<blocks, threads>>>(
            inVector, outVector, matrix.getRowPointers().getData(),
            matrix.getColumnIndexes().getData(), matrix.getValues().getData(),
            rows, kernelRowCnt
         );
      else
         SpMVCSRLight3<Real, Index, 32><<<blocks, threads>>>(
            inVector, outVector, matrix.getRowPointers().getData(),
            matrix.getColumnIndexes().getData(), matrix.getValues().getData(),
            rows, kernelRowCnt
         );
   }

   cudaFree(kernelRowCnt);
}

template< typename Real,
          typename Index,
          typename Device,
          CSRKernel KernelType,
          int warpSize>
void SpMVCSRLightWithoutAtomicPrepare( const Real *inVector,
                                       Real* outVector,
                                       const CSR< Real, Device, Index, KernelType >& matrix) {
   const Index rows = matrix.getRowPointers().getSize() - 1;
   const Index threads = matrix.THREADS_LIGHT; // block size
   size_t neededThreads = rows * warpSize;
   Index blocks, groupSize;

   const Index nnz = roundUpDivision(matrix.getValues().getSize(), rows); // non zeroes per row
   if (nnz <= 2)
      groupSize = 2;
   else if (nnz <= 4)
      groupSize = 4;
   else if (nnz <= 8)
      groupSize = 8;
   else if (nnz <= 16)
      groupSize = 16;
   else if (nnz <= 2 * matrix.MAX_ELEMENTS_PER_WARP)
      groupSize = 32; // CSR Vector
   else
      groupSize = roundUpDivision(nnz, matrix.MAX_ELEMENTS_PER_WARP) * 32; // CSR MultiVector

   if (KernelType == CSRLightWithoutAtomic)
      neededThreads = groupSize * rows;
   else
      neededThreads = rows * (groupSize > 32 ? 32 : groupSize);

   /* Execute kernels on device */
   for (Index grid = 0; neededThreads != 0; ++grid) {
      if (MAX_X_DIM * threads >= neededThreads) {
         blocks = roundUpDivision(neededThreads, threads);
         neededThreads = 0;
      } else {
         blocks = MAX_X_DIM;
         neededThreads -= MAX_X_DIM * threads;
      }

      if (KernelType == CSRLightWithoutAtomic) { //-----------------------------------------
         if (groupSize == 2) {
            SpMVCSRLightWithoutAtomic2<Real, Index><<<blocks, threads>>>(
                     inVector, outVector, matrix.getRowPointers().getData(),
                     matrix.getColumnIndexes().getData(), matrix.getValues().getData(),
                     rows, grid
            );
         } else if (groupSize == 4) {
            SpMVCSRLightWithoutAtomic4<Real, Index><<<blocks, threads>>>(
                     inVector, outVector, matrix.getRowPointers().getData(),
                     matrix.getColumnIndexes().getData(), matrix.getValues().getData(),
                     rows, grid
            );
         } else if (groupSize == 8) {
            SpMVCSRLightWithoutAtomic8<Real, Index><<<blocks, threads>>>(
                     inVector, outVector, matrix.getRowPointers().getData(),
                     matrix.getColumnIndexes().getData(), matrix.getValues().getData(),
                     rows, grid
            );
         } else if (groupSize == 16) {
            SpMVCSRLightWithoutAtomic16<Real, Index><<<blocks, threads>>>(
                     inVector, outVector, matrix.getRowPointers().getData(),
                     matrix.getColumnIndexes().getData(), matrix.getValues().getData(),
                     rows, grid
            );
         } else if (groupSize == 32) { // CSR SpMV Light with groupsize = 32 is CSR Vector
            SpMVCSRVector<Real, Index, warpSize><<<blocks, threads>>>(
                     inVector, outVector, matrix.getRowPointers().getData(),
                     matrix.getColumnIndexes().getData(), matrix.getValues().getData(),
                     rows, grid
            );
         } else { // Execute CSR MultiVector
            SpMVCSRMultiVector<Real, Index, warpSize><<<blocks, threads>>>(
                     inVector, outVector, matrix.getRowPointers().getData(),
                     matrix.getColumnIndexes().getData(), matrix.getValues().getData(),
                     rows, groupSize / 32, grid
            );
         }
      } else if (KernelType == CSRLight5) { //-----------------------------------------
         if (groupSize == 2) {
            SpMVCSRLightWithoutAtomic2<Real, Index><<<blocks, threads>>>(
                     inVector, outVector, matrix.getRowPointers().getData(),
                     matrix.getColumnIndexes().getData(), matrix.getValues().getData(),
                     rows, grid
            );
         } else if (groupSize == 4) {
            SpMVCSRLightWithoutAtomic4<Real, Index><<<blocks, threads>>>(
                     inVector, outVector, matrix.getRowPointers().getData(),
                     matrix.getColumnIndexes().getData(), matrix.getValues().getData(),
                     rows, grid
            );
         } else if (groupSize == 8) {
            SpMVCSRLightWithoutAtomic8<Real, Index><<<blocks, threads>>>(
                     inVector, outVector, matrix.getRowPointers().getData(),
                     matrix.getColumnIndexes().getData(), matrix.getValues().getData(),
                     rows, grid
            );
         } else if (groupSize == 16) {
            SpMVCSRLightWithoutAtomic16<Real, Index><<<blocks, threads>>>(
                     inVector, outVector, matrix.getRowPointers().getData(),
                     matrix.getColumnIndexes().getData(), matrix.getValues().getData(),
                     rows, grid
            );
         } else { // CSR SpMV Light with groupsize = 32 is CSR Vector
            SpMVCSRVector<Real, Index, warpSize><<<blocks, threads>>>(
                     inVector, outVector, matrix.getRowPointers().getData(),
                     matrix.getColumnIndexes().getData(), matrix.getValues().getData(),
                     rows, grid
            );
         }
      } else if (KernelType == CSRLight4) { //-----------------------------------------
         if (groupSize == 2) {
            SpMVCSRLight4<Real, Index, 2><<<blocks, threads>>>(
                     inVector, outVector, matrix.getRowPointers().getData(),
                     matrix.getColumnIndexes().getData(), matrix.getValues().getData(),
                     rows, grid
            );
         } else if (groupSize == 4) {
            SpMVCSRLight4<Real, Index, 4><<<blocks, threads>>>(
                     inVector, outVector, matrix.getRowPointers().getData(),
                     matrix.getColumnIndexes().getData(), matrix.getValues().getData(),
                     rows, grid
            );
         } else if (groupSize == 8) {
            SpMVCSRLight4<Real, Index, 8><<<blocks, threads>>>(
                     inVector, outVector, matrix.getRowPointers().getData(),
                     matrix.getColumnIndexes().getData(), matrix.getValues().getData(),
                     rows, grid
            );
         } else if (groupSize == 16) {
            SpMVCSRLight4<Real, Index, 16><<<blocks, threads>>>(
                     inVector, outVector, matrix.getRowPointers().getData(),
                     matrix.getColumnIndexes().getData(), matrix.getValues().getData(),
                     rows, grid
            );
         } else { // CSR SpMV Light with groupsize = 32 is CSR Vector
            SpMVCSRVector<Real, Index, warpSize><<<blocks, threads>>>(
                     inVector, outVector, matrix.getRowPointers().getData(),
                     matrix.getColumnIndexes().getData(), matrix.getValues().getData(),
                     rows, grid
            );
         } //-----------------------------------------
      }
   }
}

template< typename Real,
          typename Index,
          typename Device,
          CSRKernel KernelType,
          int warpSize>
void SpMVCSRMultiVectorPrepare( const Real *inVector,
                                Real* outVector,
                                const CSR< Real, Device, Index, KernelType >& matrix) {
   const Index rows = matrix.getRowPointers().getSize() - 1;
   const Index threads = matrix.THREADS_VECTOR; // block size
   Index blocks;

   const Index nnz = roundUpDivision(matrix.getValues().getSize(), rows); // non zeroes per row
   const Index neededWarps = roundUpDivision(nnz, matrix.MAX_ELEMENTS_PER_WARP); // warps per row
   size_t neededThreads = warpSize * neededWarps * rows;
   /* Execute kernels on device */
   for (Index grid = 0; neededThreads != 0; ++grid) {
      if (MAX_X_DIM * threads >= neededThreads) {
         blocks = roundUpDivision(neededThreads, threads);
         neededThreads = 0;
      } else {
         blocks = MAX_X_DIM;
         neededThreads -= MAX_X_DIM * threads;
      }

      if (neededWarps == 1) { // one warp per row -> execute CSR Vector
         SpMVCSRVector<Real, Index, warpSize><<<blocks, threads>>>(
               inVector,
               outVector,
               matrix.getRowPointers().getData(),
               matrix.getColumnIndexes().getData(),
               matrix.getValues().getData(),
               rows,
               grid
         );
      } else {
         SpMVCSRMultiVector<Real, Index, warpSize><<<blocks, threads>>>(
                  inVector,
                  outVector,
                  matrix.getRowPointers().getData(),
                  matrix.getColumnIndexes().getData(),
                  matrix.getValues().getData(),
                  rows,
                  neededWarps,
                  grid
         );
      }
   }
}

template< typename Real, typename Index >
__device__ Real CSRFetch( const Index* columnIndexes, const Real* values, const Real* vector, const Index i )
{
   return values[ i ] * vector[ columnIndexes[ i ] ];
}

template< typename Real,
          typename Index,
          int warpSize,
          int WARPS,
          int SHARED_PER_WARP,
          int MAX_ELEM_PER_WARP,
         typename Fetch >
__global__
void SpMVCSRAdaptive( const Real *inVector,
                      Real *outVector,
                      const Index* rowPointers,
                      const Index* columnIndexes,
                      const Real* values,
                      const Block<Index> *blocks,
                      Index blocksSize,
                      Index gridID,
                      const Fetch fetch )
{
   __shared__ Real shared[WARPS][SHARED_PER_WARP];
   const Index index = ( ( gridID * MAX_X_DIM + blockIdx.x ) * blockDim.x ) + threadIdx.x;
   const Index blockIdx = index / warpSize;
   if( blockIdx >= blocksSize )
      return;

   Real result = 0.0;
   const Index laneID = threadIdx.x & 31; // & is cheaper than %
   Block<Index> block = blocks[blockIdx];
   const Index minID = rowPointers[block.index[0]/* minRow */];
   Index i, to, maxID;

   if( block.byte[sizeof(Index) == 4 ? 7 : 15] & 0b1000000)
   {
      /////////////////////////////////////* CSR STREAM *//////////////
      const Index warpID = threadIdx.x / 32;
      maxID = minID + block.twobytes[sizeof(Index) == 4 ? 2 : 4];
      //              ^-> maxID - minID

      // Stream data to shared memory
      for (i = laneID + minID; i < maxID; i += warpSize)
         //shared[warpID][i - minID] = fetch( i, compute ); //CSRFetch( columnIndexes, values, inVector, i );
         shared[warpID][i - minID] = values[i] * inVector[columnIndexes[i]];

      const Index maxRow = block.index[0] + // minRow
         (block.twobytes[sizeof(Index) == 4 ? 3 : 5] & 0x3FFF); // maxRow - minRow
      // Calculate result
      for (i = block.index[0]+ laneID; i < maxRow; i += warpSize) // block.index[0] -> minRow
      {
         to = rowPointers[i + 1] - minID; // end of preprocessed data
         result = 0;
         // Scalar reduction
         for (Index sharedID = rowPointers[i] - minID; sharedID < to; ++sharedID)
            result += shared[warpID][sharedID];

         outVector[i] = result; // Write result
      }
   } else if (block.byte[sizeof(Index) == 4 ? 7 : 15] & 0b10000000) {
      //////////////////////////////////// CSR VECTOR /////////////
      maxID = minID + // maxID - minID
               block.twobytes[sizeof(Index) == 4 ? 2 : 4];

      for (i = minID + laneID; i < maxID; i += warpSize)
         result += values[i] * inVector[columnIndexes[i]];

      // Parallel reduction
      result += __shfl_down_sync(0xFFFFFFFF, result, 16);
      result += __shfl_down_sync(0xFFFFFFFF, result, 8);
      result += __shfl_down_sync(0xFFFFFFFF, result, 4);
      result += __shfl_down_sync(0xFFFFFFFF, result, 2);
      result += __shfl_down_sync(0xFFFFFFFF, result, 1);
      // Write result
      if (laneID == 0) outVector[block.index[0]] = result; // block.index[0] -> minRow
   } else {
      //////////////////////////////////// CSR VECTOR L ////////////
      // Number of elements processed by previous warps
      const Index offset = block.index[1] * MAX_ELEM_PER_WARP;
      //                   ^ warpInRow
      to = minID + (block.index[1] + 1) * MAX_ELEM_PER_WARP;
      //           ^ warpInRow
      maxID = rowPointers[block.index[0] + 1];
      //                  ^ minRow
      if (to > maxID) to = maxID;
      for (i = minID + offset + laneID; i < to; i += warpSize)
      {
         result += values[i] * inVector[columnIndexes[i]];
      }

      // Parallel reduction
      result += __shfl_down_sync(0xFFFFFFFF, result, 16);
      result += __shfl_down_sync(0xFFFFFFFF, result, 8);
      result += __shfl_down_sync(0xFFFFFFFF, result, 4);
      result += __shfl_down_sync(0xFFFFFFFF, result, 2);
      result += __shfl_down_sync(0xFFFFFFFF, result, 1);
      if (laneID == 0) atomicAdd(&outVector[block.index[0]], result);
      //                                    ^ minRow
   }
}

template< typename Real,
          typename Index,
          typename Device,
          CSRKernel KernelType,
          int warpSize>
void SpMVCSRAdaptivePrepare( const Real *inVector,
                             Real* outVector,
                             const CSR< Real, Device, Index, KernelType >& matrix) {
   Index blocks;
   const Index threads = matrix.THREADS_ADAPTIVE;

   const Index* columnIndexesData = matrix.getColumnIndexes().getData();
   const Real* valuesData = matrix.getValues().getData();
   auto fetch = [=] __cuda_callable__ ( Index globalIdx, bool& compute ) -> Real {
      return valuesData[ globalIdx ] * inVector[ columnIndexesData[ globalIdx ] ];
   };

   // Fill blocks
   size_t neededThreads = matrix.blocks.getSize() * warpSize; // one warp per block
   // Execute kernels on device
   for( Index grid = 0; neededThreads != 0; ++grid )
   {
      if( MAX_X_DIM * threads >= neededThreads )
      {
         blocks = roundUpDivision(neededThreads, threads);
         neededThreads = 0;
      }
      else
      {
         blocks = MAX_X_DIM;
         neededThreads -= MAX_X_DIM * threads;
      }
      using Matrix = CSR< Real, Device, Index, KernelType >;
      SpMVCSRAdaptive< Real, Index, warpSize,
            Matrix::WARPS,
            Matrix::SHARED_PER_WARP,
            Matrix::MAX_ELEMENTS_PER_WARP_ADAPT >
         <<<blocks, threads>>>(
               inVector,
               outVector,
               matrix.getRowPointers().getData(),
               matrix.getColumnIndexes().getData(),
               matrix.getValues().getData(),
               matrix.blocks.getData(),
               matrix.blocks.getSize() - 1, // last block shouldn't be used
               grid,
               fetch );
   }
}

#endif

template<>
class CSRDeviceDependentCode< Devices::Host >
{
   public:

      typedef Devices::Host Device;

      template< typename Real,
                typename Index,
                CSRKernel KernelType,
                typename InVector,
                typename OutVector >
      static void vectorProduct( const CSR< Real, Device, Index, KernelType >& matrix,
                                 const InVector& inVector,
                                 OutVector& outVector )
      {
         const Index rows = matrix.getRows();
         const CSR< Real, Device, Index, KernelType >* matrixPtr = &matrix;
         const InVector* inVectorPtr = &inVector;
         OutVector* outVectorPtr = &outVector;
#ifdef HAVE_OPENMP
#pragma omp parallel for firstprivate( matrixPtr, inVectorPtr, outVectorPtr ), schedule(dynamic,100), if( Devices::Host::isOMPEnabled() )
#endif
         for( Index row = 0; row < rows; row ++ )
            ( *outVectorPtr )[ row ] = matrixPtr->rowVectorProduct( row, *inVectorPtr );
      }

};


#ifdef HAVE_CUSPARSE
template<>
class tnlCusparseCSRWrapper< float, int >
{
   public:

      typedef float Real;
      typedef int Index;

      static void vectorProduct( const Index rows,
                                 const Index columns,
                                 const Index nnz,
                                 const Real* values,
                                 const Index* columnIndexes,
                                 const Index* rowPointers,
                                 const Real* x,
                                 Real* y )
      {
#if CUDART_VERSION >= 11000
         throw std::runtime_error("cusparseScsrmv was removed in CUDA 11.");
#else
         cusparseHandle_t   cusparseHandle;
         cusparseMatDescr_t cusparseMatDescr;
         cusparseCreate( &cusparseHandle );
         cusparseCreateMatDescr( &cusparseMatDescr );
         cusparseSetMatType( cusparseMatDescr, CUSPARSE_MATRIX_TYPE_GENERAL );
         cusparseSetMatIndexBase( cusparseMatDescr, CUSPARSE_INDEX_BASE_ZERO );
         Real alpha( 1.0 ), beta( 0.0 );
         cusparseScsrmv( cusparseHandle,
                         CUSPARSE_OPERATION_NON_TRANSPOSE,
                         rows,
                         columns,
                         nnz,
                         &alpha,
                         cusparseMatDescr,
                         values,
                         rowPointers,
                         columnIndexes,
                         x,
                         &beta,
                         y );
#endif
      };
};

template<>
class tnlCusparseCSRWrapper< double, int >
{
   public:

      typedef double Real;
      typedef int Index;

      static void vectorProduct( const Index rows,
                                 const Index columns,
                                 const Index nnz,
                                 const Real* values,
                                 const Index* columnIndexes,
                                 const Index* rowPointers,
                                 const Real* x,
                                 Real* y )
      {
#if CUDART_VERSION >= 11000
         throw std::runtime_error("cusparseDcsrmv was removed in CUDA 11.");
#else
         cusparseHandle_t   cusparseHandle;
         cusparseMatDescr_t cusparseMatDescr;
         cusparseCreate( &cusparseHandle );
         cusparseCreateMatDescr( &cusparseMatDescr );
         cusparseSetMatType( cusparseMatDescr, CUSPARSE_MATRIX_TYPE_GENERAL );
         cusparseSetMatIndexBase( cusparseMatDescr, CUSPARSE_INDEX_BASE_ZERO );
         Real alpha( 1.0 ), beta( 0.0 );
         cusparseDcsrmv( cusparseHandle,
                         CUSPARSE_OPERATION_NON_TRANSPOSE,
                         rows,
                         columns,
                         nnz,
                         &alpha,
                         cusparseMatDescr,
                         values,
                         rowPointers,
                         columnIndexes,
                         x,
                         &beta,
                         y );
#endif
      };
};

#endif

template<>
class CSRDeviceDependentCode< Devices::Cuda >
{
   public:

      typedef Devices::Cuda Device;

      template< typename Real,
                typename Index,
                CSRKernel KernelType,
                typename InVector,
                typename OutVector >
      static void vectorProduct( const CSR< Real, Device, Index, KernelType >& matrix,
                                 const InVector& inVector,
                                 OutVector& outVector )
      {
#ifdef __CUDACC__
#ifdef HAVE_CUSPARSE
         tnlCusparseCSRWrapper< Real, Index >::vectorProduct( matrix.getRows(),
                                                              matrix.getColumns(),
                                                              matrix.values.getSize(),
                                                              matrix.values.getData(),
                                                              matrix.columnIndexes.getData(),
                                                              matrix.rowPointers.getData(),
                                                              inVector.getData(),
                                                              outVector.getData() );
#else
         switch(KernelType)
         {
            case CSRScalar:
               SpMVCSRScalarPrepare<Real, Index, Device, KernelType>(
                  inVector.getData(), outVector.getData(), matrix
               );
               break;
            case CSRVector:
               SpMVCSRVectorPrepare<Real, Index, Device, KernelType, 32>(
                  inVector.getData(), outVector.getData(), matrix
               );
               break;
            case CSRLight:
            case CSRLight2:
            case CSRLight3:
            case CSRLight6:
               SpMVCSRLightPrepare<Real, Index, Device, KernelType, 32>(
                  inVector.getData(), outVector.getData(), matrix
               );
               break;
            case CSRAdaptive:
               SpMVCSRAdaptivePrepare<Real, Index, Device, KernelType, 32>(
                  inVector.getData(), outVector.getData(), matrix
               );
               break;
            case CSRMultiVector:
               SpMVCSRMultiVectorPrepare<Real, Index, Device, KernelType, 32>(
                  inVector.getData(), outVector.getData(), matrix
               );
               break;
            case CSRLight4:
            case CSRLight5:
            case CSRLightWithoutAtomic:
               SpMVCSRLightWithoutAtomicPrepare<Real, Index, Device, KernelType, 32>(
                  inVector.getData(), outVector.getData(), matrix
               );
               break;
         }
#endif /* __CUDACC__ */
#endif
      }
};

               } //namespace Legacy
            } //namespace ReferenceFormats
        } //namespace SpMV
    } //namespace Benchmarks
} // namespace TNL
