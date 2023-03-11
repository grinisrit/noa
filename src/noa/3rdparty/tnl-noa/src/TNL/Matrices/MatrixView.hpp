// Copyright (c) 2004-2023 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <noa/3rdparty/tnl-noa/src/TNL/Matrices/Matrix.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Assert.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Cuda/LaunchHelpers.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Cuda/SharedMemory.h>

#include <utility>

namespace noa::TNL {
namespace Matrices {

template< typename Real, typename Device, typename Index >
__cuda_callable__
MatrixView< Real, Device, Index >::MatrixView() : rows( 0 ), columns( 0 ) {}

template< typename Real, typename Device, typename Index >
__cuda_callable__
MatrixView< Real, Device, Index >::MatrixView( IndexType rows, IndexType columns, ValuesView values )
: rows( rows ), columns( columns ), values( std::move( values ) )
{}

template< typename Real, typename Device, typename Index >
Index
MatrixView< Real, Device, Index >::getAllocatedElementsCount() const
{
   return this->values.getSize();
}

template< typename Real, typename Device, typename Index >
Index
MatrixView< Real, Device, Index >::getNonzeroElementsCount() const
{
   const auto values_view = this->values.getConstView();
   auto fetch = [ = ] __cuda_callable__( const IndexType i ) -> IndexType
   {
      return ( values_view[ i ] != 0.0 );
   };
   return Algorithms::reduce< DeviceType >( (IndexType) 0, this->values.getSize(), fetch, std::plus<>{}, 0 );
}

template< typename Real, typename Device, typename Index >
__cuda_callable__
Index
MatrixView< Real, Device, Index >::getRows() const
{
   return this->rows;
}

template< typename Real, typename Device, typename Index >
__cuda_callable__
Index
MatrixView< Real, Device, Index >::getColumns() const
{
   return this->columns;
}

template< typename Real, typename Device, typename Index >
__cuda_callable__
const typename MatrixView< Real, Device, Index >::ValuesView&
MatrixView< Real, Device, Index >::getValues() const
{
   return this->values;
}

template< typename Real, typename Device, typename Index >
__cuda_callable__
typename MatrixView< Real, Device, Index >::ValuesView&
MatrixView< Real, Device, Index >::getValues()
{
   return this->values;
}
template< typename Real, typename Device, typename Index >
__cuda_callable__
MatrixView< Real, Device, Index >&
MatrixView< Real, Device, Index >::operator=( const MatrixView& view )
{
   rows = view.rows;
   columns = view.columns;
   values.bind( view.values );
   return *this;
}

template< typename Real, typename Device, typename Index >
template< typename MatrixT >
bool
MatrixView< Real, Device, Index >::operator==( const MatrixT& matrix ) const
{
   if( this->getRows() != matrix.getRows() || this->getColumns() != matrix.getColumns() )
      return false;
   for( IndexType row = 0; row < this->getRows(); row++ )
      for( IndexType column = 0; column < this->getColumns(); column++ )
         if( this->getElement( row, column ) != matrix.getElement( row, column ) )
            return false;
   return true;
}

template< typename Real, typename Device, typename Index >
template< typename MatrixT >
bool
MatrixView< Real, Device, Index >::operator!=( const MatrixT& matrix ) const
{
   return ! operator==( matrix );
}

template< typename Real, typename Device, typename Index >
void
MatrixView< Real, Device, Index >::save( File& file ) const
{
   file.save( magic_number, strlen( magic_number ) );
   file << this->getSerializationTypeVirtual();
   file.save( &this->rows );
   file.save( &this->columns );
   file << this->values;
}

template< typename Real, typename Device, typename Index >
void
MatrixView< Real, Device, Index >::save( const String& fileName ) const
{
   File file;
   file.open( fileName, std::ios_base::out );
   this->save( file );
}

template< typename Real, typename Device, typename Index >
void
MatrixView< Real, Device, Index >::print( std::ostream& str ) const
{}

}  // namespace Matrices
}  // namespace noa::TNL
