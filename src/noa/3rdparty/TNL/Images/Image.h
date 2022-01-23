// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

namespace noa::TNL {
/**
 * \brief Namespace for image processing.
 */
namespace Images {

template< typename Index = int >
class Image
{
   public:
 
      typedef Index IndexType;
 
      Image() : width( 0 ), height( 0 ) {};
 
      IndexType getWidth() const
      {
         return this->width;
      }
 
      IndexType getHeight() const
      {
         return this->height;
      }
 
   protected:
 
      IndexType width, height;
 
};

} // namespace Images
} // namespace noa::TNL

