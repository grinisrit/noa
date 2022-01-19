// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Matrices/SparseMatrixRowView.h>
#include <TNL/Assert.h>

namespace TNL {
namespace Matrices {

template< typename RowView >
__cuda_callable__
MatrixRowViewIterator< RowView >::
MatrixRowViewIterator( RowViewType& rowView,
                             const IndexType& localIdx )
: rowView( rowView ), localIdx( localIdx )
{
}

template< typename RowView >
__cuda_callable__ bool
MatrixRowViewIterator< RowView >::
operator==( const MatrixRowViewIterator& other ) const
{
   if( &this->rowView == &other.rowView &&
       localIdx == other.localIdx )
      return true;
   return false;
}

template< typename RowView >
__cuda_callable__ bool
MatrixRowViewIterator< RowView >::
operator!=( const MatrixRowViewIterator& other ) const
{
   return ! ( other == *this );
}

template< typename RowView >
__cuda_callable__
MatrixRowViewIterator< RowView >&
MatrixRowViewIterator< RowView >::
operator++()
{
   if( localIdx < rowView.getSize() )
      localIdx ++;
   return *this;
}

template< typename RowView >
__cuda_callable__
MatrixRowViewIterator< RowView >&
MatrixRowViewIterator< RowView >::
operator--()
{
   if( localIdx > 0 )
      localIdx --;
   return *this;
}

template< typename RowView >
__cuda_callable__ auto
MatrixRowViewIterator< RowView >::
operator*() -> MatrixElementType
{
   return MatrixElementType(
      this->rowView.getValue( this->localIdx ),
      this->rowView.getRowIndex(),
      this->rowView.getColumnIndex( this->localIdx ),
      this->localIdx );
}

template< typename RowView >
__cuda_callable__ auto
MatrixRowViewIterator< RowView >::
operator*() const -> const MatrixElementType
{
   return MatrixElementType(
      this->rowView.getValue( this->localIdx ),
      this->rowView.getRowIndex(),
      this->rowView.getColumnIndex( this->localIdx ),
      this->localIdx );
}


   } // namespace Matrices
} // namespace TNL
