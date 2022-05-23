#pragma once

#include "Sparse.h"
#include <TNL/Exceptions/NotImplementedError.h>

namespace TNL {
    namespace Benchmarks {
        namespace SpMV {
            namespace ReferenceFormats {
               namespace Legacy {

template< typename Real,
          typename Device,
          typename Index >
Sparse< Real, Device, Index >::Sparse()
: maxRowLength( 0 )
{
}

template< typename Real,
          typename Device,
          typename Index >
   template< typename Real2,
             typename Device2,
             typename Index2 >
void Sparse< Real, Device, Index >::setLike( const Sparse< Real2, Device2, Index2 >& matrix )
{
   TNL::Matrices::Matrix< Real, Device, Index >::setLike( matrix );
   this->allocateMatrixElements( matrix.getAllocatedElementsCount() );
}


template< typename Real,
          typename Device,
          typename Index >
Index Sparse< Real, Device, Index >::getNumberOfNonzeroMatrixElements() const
{
   IndexType nonzeroElements( 0 );
   for( IndexType i = 0; i < this->values.getSize(); i++ )
      if( this->columnIndexes.getElement( i ) != this-> columns &&
          this->values.getElement( i ) != RealType( 0 ) )
         nonzeroElements++;
   return nonzeroElements;
}

template< typename Real,
          typename Device,
          typename Index >
Index
Sparse< Real, Device, Index >::
getMaxRowLength() const
{
   return this->maxRowLength;
}

template< typename Real,
          typename Device,
          typename Index >
__cuda_callable__
Index Sparse< Real, Device, Index >::getPaddingIndex() const
{
   return this->getColumns();
}

template< typename Real,
          typename Device,
          typename Index >
void Sparse< Real, Device, Index >::reset()
{
   TNL::Matrices::Matrix< Real, Device, Index >::reset();
   this->columnIndexes.reset();
}

template< typename Real,
          typename Device,
          typename Index >
void Sparse< Real, Device, Index >::save( File& file ) const
{
   TNL::Matrices::Matrix< Real, Device, Index >::save( file );
   file << this->values << this->columnIndexes;
}

template< typename Real,
          typename Device,
          typename Index >
void Sparse< Real, Device, Index >::load( File& file )
{
   TNL::Matrices::Matrix< Real, Device, Index >::load( file );
   file >> this->values >> this->columnIndexes;
}

template< typename Real,
          typename Device,
          typename Index >
void Sparse< Real, Device, Index >::allocateMatrixElements( const IndexType& numberOfMatrixElements )
{
   TNL_ASSERT_GE( numberOfMatrixElements, ( IndexType ) 0, "Number of matrix elements must be non-negative." );

   this->values.setSize( numberOfMatrixElements );
   this->columnIndexes.setSize( numberOfMatrixElements );

   /****
    * Setting a column index to this->columns means that the
    * index is undefined.
    */
   if( numberOfMatrixElements > 0 )
      this->columnIndexes.setValue( this->columns );
}

template< typename Real,
          typename Device,
          typename Index >
void Sparse< Real, Device, Index >::printStructure( std::ostream& str ) const
{
   throw Exceptions::NotImplementedError("Sparse::printStructure is not implemented yet.");
}

               } //namespace Legacy
            } //namespace ReferenceFormats
        } //namespace SpMV
    } //namespace Benchmarks
} // namespace TNL
