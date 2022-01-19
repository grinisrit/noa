// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/TypeTraits.h>

namespace TNL {
   namespace Matrices {
      namespace details {

template< typename VectorOrView,
          std::enable_if_t< HasSetSizeMethod< VectorOrView >::value, bool > = true >
static void set_size_if_resizable( VectorOrView& v, typename VectorOrView::IndexType size )
{
   v.setSize( size );
}

template< typename VectorOrView,
          std::enable_if_t< ! HasSetSizeMethod< VectorOrView >::value, bool > = true >
static void set_size_if_resizable( VectorOrView& v, typename VectorOrView::IndexType size )
{
   TNL_ASSERT_EQ( v.getSize(), size, "view has wrong size" );
}

      } //namespace details
   } //namepsace Matrices
} //namespace TNL
