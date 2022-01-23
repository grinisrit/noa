// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

// Implemented by: Jakub Klinkovsky

#pragma once

#include <noa/3rdparty/TNL/Containers/NDArray.h>
#include <noa/3rdparty/TNL/Containers/StaticVector.h>

namespace noa::TNL {
namespace Matrices {

template< typename Value,
          std::size_t Rows,
          std::size_t Columns,
          typename Permutation = std::index_sequence< 0, 1 > >  // identity by default
class StaticMatrix
: public Containers::StaticNDArray< Value,
                                    // note that using std::size_t in SizesHolder does not make sense, since the
                                    // StaticNDArray is based on StaticArray, which uses int as IndexType
                                    Containers::SizesHolder< int, Rows, Columns >,
                                    Permutation >
{
   using Base = Containers::StaticNDArray< Value,
                                           Containers::SizesHolder< int, Rows, Columns >,
                                           Permutation >;

public:
   // inherit all assignment operators
   using Base::operator=;

   static constexpr std::size_t getRows()
   {
      return Rows;
   }

   __cuda_callable__
   static constexpr std::size_t getColumns()
   {
      return Columns;
   }

   __cuda_callable__
   Containers::StaticVector< Rows, Value >
   operator*( const Containers::StaticVector< Columns, Value > & vector ) const
   {
      Containers::StaticVector< Rows, Value > result;
      for( std::size_t i = 0; i < Rows; i++ ) {
         Value v = 0;
         for( std::size_t j = 0; j < Columns; j++ )
            v += (*this)( i, j ) * vector[ j ];
         result[ i ] = v;
      }
      return result;
   }
};

} // namespace Matrices
} // namespace noa::TNL
