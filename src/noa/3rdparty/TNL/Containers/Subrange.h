// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

// Implemented by: Jakub Klinkovský

#pragma once

#include <ostream>

#include <TNL/Assert.h>
#include <TNL/String.h>
#include <TNL/TypeInfo.h>

namespace TNL {
namespace Containers {

// Specifies a subrange [begin, end) of a range [0, globalSize).
template< typename Index >
class Subrange
{
public:
   using IndexType = Index;

   __cuda_callable__
   Subrange() = default;

   __cuda_callable__
   Subrange( Index begin, Index end )
   {
      setSubrange( begin, end );
   }

   // Sets the local subrange and global range size.
   __cuda_callable__
   void setSubrange( Index begin, Index end )
   {
      TNL_ASSERT_LE( begin, end, "begin must be before end" );
      TNL_ASSERT_GE( begin, 0, "begin must be non-negative" );
      this->begin = begin;
      this->end = end;
   }

   __cuda_callable__
   void reset()
   {
      begin = 0;
      end = 0;
   }

   // Checks if a global index is in the set of local indices.
   __cuda_callable__
   bool isLocal( Index i ) const
   {
      return begin <= i && i < end;
   }

   // Gets the begin of the subrange.
   __cuda_callable__
   Index getBegin() const
   {
      return begin;
   }

   // Gets the begin of the subrange.
   __cuda_callable__
   Index getEnd() const
   {
      return end;
   }

   // Gets number of local indices.
   __cuda_callable__
   Index getSize() const
   {
      return end - begin;
   }

   // Gets local index for given global index.
   __cuda_callable__
   Index getLocalIndex( Index i ) const
   {
      TNL_ASSERT_GE( i, getBegin(), "Given global index was not found in the local index set." );
      TNL_ASSERT_LT( i, getEnd(), "Given global index was not found in the local index set." );
      return i - begin;
   }

   // Gets global index for given local index.
   __cuda_callable__
   Index getGlobalIndex( Index i ) const
   {
      TNL_ASSERT_GE( i, 0, "Given local index was not found in the local index set." );
      TNL_ASSERT_LT( i, getSize(), "Given local index was not found in the local index set." );
      return i + begin;
   }

   __cuda_callable__
   bool operator==( const Subrange& other ) const
   {
      return begin == other.begin &&
             end == other.end;
   }

   __cuda_callable__
   bool operator!=( const Subrange& other ) const
   {
      return ! (*this == other);
   }

protected:
   Index begin = 0;
   Index end = 0;
};

// due to formatting in TNL::Assert
template< typename Index >
std::ostream& operator<<( std::ostream& str, const Subrange< Index >& range )
{
   return str << getType< Subrange< Index > >() << "( " << range.getBegin() << ", " << range.getEnd() << " )";
}

} // namespace Containers
} // namespace TNL
