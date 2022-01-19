// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Matrices/Matrix.h>
#include <TNL/Assert.h>
#include <TNL/Cuda/LaunchHelpers.h>
#include <TNL/Cuda/MemoryHelpers.h>
#include <TNL/Cuda/SharedMemory.h>

namespace TNL {
namespace Matrices {

template< typename Real,
          typename Device,
          typename Index,
          typename RealAllocator >
Matrix< Real, Device, Index, RealAllocator >::
Matrix( const RealAllocatorType& allocator )
: rows( 0 ),
  columns( 0 ),
  values( allocator )
{
}

template< typename Real,
          typename Device,
          typename Index,
          typename RealAllocator >
Matrix< Real, Device, Index, RealAllocator >::
Matrix( const IndexType rows_, const IndexType columns_, const RealAllocatorType& allocator )
: rows( rows_ ),
  columns( columns_ ),
  values( allocator )
{
}

template< typename Real,
          typename Device,
          typename Index,
          typename RealAllocator >
void Matrix< Real, Device, Index, RealAllocator >::setDimensions( const IndexType rows,
                                                   const IndexType columns )
{
   TNL_ASSERT( rows >= 0 && columns >= 0,
               std::cerr << " rows = " << rows << " columns = " << columns );
   this->rows = rows;
   this->columns = columns;
}

template< typename Real,
          typename Device,
          typename Index,
          typename RealAllocator >
   template< typename Matrix_ >
void Matrix< Real, Device, Index, RealAllocator >::setLike( const Matrix_& matrix )
{
   setDimensions( matrix.getRows(), matrix.getColumns() );
}

template< typename Real,
          typename Device,
          typename Index,
          typename RealAllocator >
Index Matrix< Real, Device, Index, RealAllocator >::getAllocatedElementsCount() const
{
   return this->values.getSize();
}

template< typename Real,
          typename Device,
          typename Index,
          typename RealAllocator >
Index Matrix< Real, Device, Index, RealAllocator >::getNonzeroElementsCount() const
{
   const auto values_view = this->values.getConstView();
   auto fetch = [=] __cuda_callable__ ( const IndexType i ) -> IndexType {
      return ( values_view[ i ] != 0.0 );
   };
   return Algorithms::reduce< DeviceType >( 0, this->values.getSize(), fetch, std::plus<>{}, 0 );
}

template< typename Real,
          typename Device,
          typename Index,
          typename RealAllocator >
__cuda_callable__
Index Matrix< Real, Device, Index, RealAllocator >::getRows() const
{
   return this->rows;
}

template< typename Real,
          typename Device,
          typename Index,
          typename RealAllocator >
__cuda_callable__
Index Matrix< Real, Device, Index, RealAllocator >::getColumns() const
{
   return this->columns;
}

template< typename Real,
          typename Device,
          typename Index,
          typename RealAllocator >
auto
Matrix< Real, Device, Index, RealAllocator >::
getValues() const -> const ValuesType&
{
   return this->values;
}

template< typename Real,
          typename Device,
          typename Index,
          typename RealAllocator >
auto
Matrix< Real, Device, Index, RealAllocator >::
getValues() -> ValuesType&
{
   return this->values;
}

template< typename Real,
          typename Device,
          typename Index,
          typename RealAllocator >
void Matrix< Real, Device, Index, RealAllocator >::reset()
{
   this->rows = 0;
   this->columns = 0;
   this->values.reset();
}

template< typename Real,
          typename Device,
          typename Index,
          typename RealAllocator >
   template< typename MatrixT >
bool Matrix< Real, Device, Index, RealAllocator >::operator == ( const MatrixT& matrix ) const
{
   if( this->getRows() != matrix.getRows() ||
       this->getColumns() != matrix.getColumns() )
      return false;
   for( IndexType row = 0; row < this->getRows(); row++ )
      for( IndexType column = 0; column < this->getColumns(); column++ )
         if( this->getElement( row, column ) != matrix.getElement( row, column ) )
            return false;
   return true;
}

template< typename Real,
          typename Device,
          typename Index,
          typename RealAllocator >
   template< typename MatrixT >
bool Matrix< Real, Device, Index, RealAllocator >::operator != ( const MatrixT& matrix ) const
{
   return ! operator == ( matrix );
}

template< typename Real,
          typename Device,
          typename Index,
          typename RealAllocator >
void Matrix< Real, Device, Index, RealAllocator >::save( File& file ) const
{
   Object::save( file );
   file.save( &this->rows );
   file.save( &this->columns );
   file << this->values;
}

template< typename Real,
          typename Device,
          typename Index,
          typename RealAllocator >
void Matrix< Real, Device, Index, RealAllocator >::load( File& file )
{
   Object::load( file );
   file.load( &this->rows );
   file.load( &this->columns );
   file >> this->values;
}

template< typename Real,
          typename Device,
          typename Index,
          typename RealAllocator >
void Matrix< Real, Device, Index, RealAllocator >::print( std::ostream& str ) const
{
}

/*
template< typename Real,
          typename Device,
          typename Index,
          typename RealAllocator >
void
Matrix< Real, Device, Index, RealAllocator >::
computeColorsVector(Containers::Vector<Index, Device, Index> &colorsVector)
{
    for( IndexType i = this->getRows() - 1; i >= 0; i-- )
    {
        // init color array
        Containers::Vector< Index, Device, Index > usedColors;
        usedColors.setSize( this->numberOfColors );
        for( IndexType j = 0; j < this->numberOfColors; j++ )
            usedColors.setElement( j, 0 );

        // find all colors used in given row
        for( IndexType j = i + 1; j < this->getColumns(); j++ )
             if( this->getElement( i, j ) != 0.0 )
                 usedColors.setElement( colorsVector.getElement( j ), 1 );

        // find unused color
        bool found = false;
        for( IndexType j = 0; j < this->numberOfColors; j++ )
            if( usedColors.getElement( j ) == 0 )
            {
                colorsVector.setElement( i, j );
                found = true;
                break;
            }
        if( !found )
        {
            colorsVector.setElement( i, this->numberOfColors );
            this->numberOfColors++;
        }
    }
}*/

} // namespace Matrices
} // namespace TNL
