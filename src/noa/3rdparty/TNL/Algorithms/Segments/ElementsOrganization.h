// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <noa/3rdparty/TNL/Devices/Host.h>

namespace noaTNL {
   namespace Algorithms {
      namespace Segments {

enum ElementsOrganization { ColumnMajorOrder = 0, RowMajorOrder };

template< typename Device >
struct DefaultElementsOrganization
{
   static constexpr ElementsOrganization getOrganization() {
      if( std::is_same< Device, Devices::Host >::value )
         return RowMajorOrder;
      else
         return ColumnMajorOrder;
   };
};

} // namespace Segments
} // namespace Algorithms

inline String getSerializationType( Algorithms::Segments::ElementsOrganization Organization )
{
   if( Organization == Algorithms::Segments::RowMajorOrder )
      return String( "RowMajorOrder" );
   else
      return String( "ColumnMajorOrder" );
}

} // namespace noaTNL
