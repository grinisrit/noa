// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <noa/3rdparty/tnl-noa/src/TNL/TypeTraits.h>

namespace noa::TNL {
namespace Containers {
namespace Expressions {

// trait classes used for the enabling of expression templates
template< typename T >
struct HasEnabledExpressionTemplates : std::false_type
{};

template< typename T >
struct HasEnabledStaticExpressionTemplates : std::false_type
{};

template< typename T >
struct HasEnabledDistributedExpressionTemplates : std::false_type
{};

// type aliases for enabling specific operators and functions using SFINAE
template< typename ET1, typename T = void >
using EnableIfStaticUnaryExpression_t =
   std::enable_if_t< HasEnabledStaticExpressionTemplates< std::decay_t< ET1 > >::value, T >;

template< typename ET1, typename ET2, typename T = void >
using EnableIfStaticBinaryExpression_t =
   std::enable_if_t< ( HasEnabledStaticExpressionTemplates< std::decay_t< ET1 > >::value
                       || HasEnabledStaticExpressionTemplates< std::decay_t< ET2 > >::value )
                        && ! ( HasEnabledExpressionTemplates< std::decay_t< ET2 > >::value
                               || HasEnabledExpressionTemplates< std::decay_t< ET1 > >::value
                               || HasEnabledDistributedExpressionTemplates< std::decay_t< ET2 > >::value
                               || HasEnabledDistributedExpressionTemplates< std::decay_t< ET1 > >::value ),
                     T >;

template< typename ET1, typename T = void >
using EnableIfUnaryExpression_t = std::enable_if_t< HasEnabledExpressionTemplates< std::decay_t< ET1 > >::value, T >;

template< typename ET1, typename ET2, typename T = void >
using EnableIfBinaryExpression_t = std::enable_if_t<
   // we need to avoid ambiguity with operators defined in Array (e.g. Array::operator==)
   // so the first operand must not be Array
   ( HasAddAssignmentOperator< std::decay_t< ET1 > >::value || HasEnabledExpressionTemplates< std::decay_t< ET1 > >::value
     || std::is_arithmetic< std::decay_t< ET1 > >::value )
      && ( HasEnabledExpressionTemplates< std::decay_t< ET2 > >::value
           || HasEnabledExpressionTemplates< std::decay_t< ET1 > >::value ),
   T >;

template< typename ET1, typename T = void >
using EnableIfDistributedUnaryExpression_t =
   std::enable_if_t< HasEnabledDistributedExpressionTemplates< std::decay_t< ET1 > >::value, T >;

template< typename ET1, typename ET2, typename T = void >
using EnableIfDistributedBinaryExpression_t = std::enable_if_t<
   // we need to avoid ambiguity with operators defined in Array (e.g. Array::operator==)
   // so the first operand must not be Array
   ( HasAddAssignmentOperator< std::decay_t< ET1 > >::value
     || HasEnabledDistributedExpressionTemplates< std::decay_t< ET1 > >::value
     || std::is_arithmetic< std::decay_t< ET1 > >::value )
      && ( HasEnabledDistributedExpressionTemplates< std::decay_t< ET2 > >::value
           || HasEnabledDistributedExpressionTemplates< std::decay_t< ET1 > >::value ),
   T >;

// helper trait class for recursively turning expression template classes into compatible vectors
template< typename R, typename Enable = void >
struct RemoveExpressionTemplate
{
   using type = std::decay_t< R >;
};

template< typename R >
struct RemoveExpressionTemplate< R, typename enable_if_type< typename std::decay_t< R >::VectorOperandType >::type >
{
   using type = typename RemoveExpressionTemplate< typename std::decay_t< R >::VectorOperandType >::type;
};

template< typename R >
using RemoveET = typename RemoveExpressionTemplate< R >::type;

template< typename T1, typename T2 >
constexpr std::enable_if_t< ! ( IsStaticArrayType< T1 >::value && IsStaticArrayType< T2 >::value )
                               && ! ( IsArrayType< T1 >::value && IsArrayType< T2 >::value ),
                            bool >
compatibleForVectorAssignment()
{
   return IsScalarType< T1 >::value && IsScalarType< T2 >::value;
}

template< typename T1, typename T2 >
constexpr std::enable_if_t< IsStaticArrayType< T1 >::value && IsStaticArrayType< T2 >::value, bool >
compatibleForVectorAssignment()
{
   return T1::getSize() == T2::getSize()
       && compatibleForVectorAssignment< typename RemoveET< T1 >::ValueType, typename RemoveET< T2 >::ValueType >();
}

template< typename T1, typename T2 >
constexpr std::enable_if_t< IsArrayType< T1 >::value && IsArrayType< T2 >::value, bool >
compatibleForVectorAssignment()
{
   return compatibleForVectorAssignment< typename RemoveET< T1 >::ValueType, typename RemoveET< T2 >::ValueType >();
}

// helper trait class for proper classification of expression operands using getExpressionVariableType
template< typename T,
          typename V,
          bool enabled = HasEnabledExpressionTemplates< V >::value || HasEnabledStaticExpressionTemplates< V >::value
                      || HasEnabledDistributedExpressionTemplates< V >::value >
struct IsArithmeticSubtype
: public std::integral_constant< bool,
                                 // Note that using std::is_same would not be general enough, because e.g.
                                 // StaticVector<3, int> may be assigned to StaticVector<3, double>
                                 compatibleForVectorAssignment< typename V::RealType, T >() >
{};

template< typename T >
struct IsArithmeticSubtype< T, T, true > : public std::false_type
{};

template< typename T >
struct IsArithmeticSubtype< T, T, false > : public std::false_type
{};

template< typename T, typename V >
struct IsArithmeticSubtype< T, V, false > : public std::is_arithmetic< T >
{};

// helper trait class for Static*ExpressionTemplates classes
template< typename R, typename Enable = void >
struct OperandMemberType
{
   using type = std::conditional_t< std::is_fundamental< R >::value,
                                    // non-reference for fundamental types
                                    std::add_const_t< std::remove_reference_t< R > >,
                                    // lvalue-reference for other types (especially StaticVector)
                                    std::add_lvalue_reference_t< std::add_const_t< R > > >;
   //   using type = std::add_const_t< std::remove_reference_t< R > >;
};

// assuming that only the StaticBinaryExpressionTemplate and StaticUnaryTemplate classes have a VectorOperandType type member
template< typename R >
struct OperandMemberType< R, typename enable_if_type< typename R::VectorOperandType >::type >
{
   // non-reference for StaticBinaryExpressionTemplate and StaticUnaryExpressionTemplate
   // (otherwise we would get segfaults - binding const-reference to temporary Static*ExpressionTemplate
   // objects does not work as expected...)
   using type = std::add_const_t< std::remove_reference_t< R > >;
};

}  // namespace Expressions
}  // namespace Containers
}  // namespace noa::TNL
