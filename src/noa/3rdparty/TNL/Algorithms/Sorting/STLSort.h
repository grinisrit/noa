// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

// Implemented by: Tomas Oberhuber

#pragma once

#include <algorithm>

namespace noaTNL {
   namespace Algorithms {
      namespace Sorting {

struct STLSort
{
   template< typename Array >
   void static sort( Array& array )
   {
      std::sort( array.getData(), array.getData() + array.getSize() );
   }

   template< typename Array, typename Compare >
   void static sort( Array& array, const Compare& compare )
   {
      std::sort( array.getData(), array.getData() + array.getSize(), compare );
   }
};

      } // namespace Sorting
   } // namespace Algorithms
} //namespace noaTNL
