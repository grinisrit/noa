// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Algorithms/staticFor.h>
#include <TNL/Meshes/EntityShapeGroup.h>

namespace TNL {
namespace Meshes {
namespace VTK {

template< EntityShape GeneralShape_ >
class EntityShapeGroupChecker
{
public:
   static constexpr EntityShape GeneralShape = GeneralShape_;

   static bool belong( EntityShape shape )
   {
      if( GeneralShape == shape )
      {
         return true;
      }
      else
      {
         bool result = false;

         Algorithms::staticFor< int, 0, EntityShapeGroup< GeneralShape >::size >(
            [&] ( auto index ) {
               EntityShape groupShape = EntityShapeGroupElement< GeneralShape, index >::shape;
               result = result || ( shape == groupShape );
            }
         );

         return result;
      }
   }

   static bool bothBelong( EntityShape shape1, EntityShape shape2 )
   {
      return belong( shape1 ) && belong( shape2 );
   }
};

} // namespace VTK
} // namespace Meshes
} // namespace TNL