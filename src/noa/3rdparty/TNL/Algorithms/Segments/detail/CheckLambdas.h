// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once


namespace noaTNL {
   namespace Algorithms {
      namespace Segments {
         namespace detail {

template< typename Index,
          typename Lambda >
class CheckFetchLambda
{
   private:
      typedef char YesType[1];
      typedef char NoType[2];

      template< typename C > static YesType& test( decltype(std::declval< C >()( Index(), Index(), Index(), std::declval< bool& >() ) ) );
      template< typename C > static NoType& test(...);

      static constexpr bool value = ( sizeof( test< Lambda >(0) ) == sizeof( YesType ) );

   public:

      static constexpr bool hasAllParameters() { return value; };
};

         } // namespace detail
      } // namespace Segments
   }  // namespace Algorithms
} // namespace noaTNL
