// Copyright (c) 2004-2023 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <noa/3rdparty/tnl-noa/src/TNL/Containers/Expressions/TypeTraits.h>

namespace noa::TNL {
namespace Containers {
namespace Expressions {

enum ExpressionVariableType
{
   ArithmeticVariable,
   VectorExpressionVariable,
   OtherVariable
};

template< typename T, typename V = T >
constexpr ExpressionVariableType
getExpressionVariableType()
{
   if( std::is_arithmetic< T >::value )
      return ArithmeticVariable;
   // vectors must be considered as an arithmetic type when used as RealType in another vector
   if( IsArithmeticSubtype< T, V >::value )
      return ArithmeticVariable;
   if( HasEnabledExpressionTemplates< T >::value || HasEnabledStaticExpressionTemplates< T >::value
       || HasEnabledDistributedExpressionTemplates< T >::value )
      return VectorExpressionVariable;
   if( IsArrayType< T >::value || IsStaticArrayType< T >::value )
      return VectorExpressionVariable;
   return OtherVariable;
}

}  // namespace Expressions
}  // namespace Containers
}  // namespace noa::TNL
