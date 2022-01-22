// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <type_traits>
#include <utility>

namespace noaTNL {

template< typename T, typename R = void >
struct enable_if_type
{
   using type = R;
};

/**
 * \brief Type trait for checking if T has getArrayData method.
 */
template< typename T >
class HasGetArrayDataMethod
{
private:
    typedef char YesType[1];
    typedef char NoType[2];

    template< typename C > static YesType& test( decltype(std::declval< C >().getArrayData()) );
    template< typename C > static NoType& test(...);

public:
    static constexpr bool value = ( sizeof( test< std::decay_t<T> >(0) ) == sizeof( YesType ) );
};

/**
 * \brief Type trait for checking if T has getSize method.
 */
template< typename T >
class HasGetSizeMethod
{
private:
    typedef char YesType[1];
    typedef char NoType[2];

    template< typename C > static YesType& test( decltype(std::declval< C >().getSize() ) );
    template< typename C > static NoType& test(...);

public:
    static constexpr bool value = ( sizeof( test< std::decay_t<T> >(0) ) == sizeof( YesType ) );
};

/**
 * \brief Type trait for checking if T has setSize method.
 */
template< typename T >
class HasSetSizeMethod
{
private:
   template< typename U >
   static constexpr auto check(U*)
   -> typename
      std::enable_if_t<
         std::is_same<
               decltype( std::declval<U>().setSize(0) ),
               void
            >::value,
         std::true_type
      >;

   template< typename >
   static constexpr std::false_type check(...);

   using type = decltype(check<std::decay_t<T>>(0));

public:
    static constexpr bool value = type::value;
};

/**
 * \brief Type trait for checking if T has operator[] taking one index argument.
 */
template< typename T >
class HasSubscriptOperator
{
private:
   template< typename U >
   static constexpr auto check(U*)
   -> typename
      std::enable_if_t<
         ! std::is_same<
               decltype( std::declval<U>()[ std::declval<U>().getSize() ] ),
               void
            >::value,
         std::true_type
      >;

   template< typename >
   static constexpr std::false_type check(...);

   using type = decltype(check<std::decay_t<T>>(0));

public:
    static constexpr bool value = type::value;
};

/**
 * \brief Type trait for checking if T has operator+= taking one argument of type T.
 */
template< typename T >
class HasAddAssignmentOperator
{
private:
   template< typename U >
   static constexpr auto check(U*)
   -> typename
      std::enable_if_t<
         ! std::is_same<
               decltype( std::declval<U>() += std::declval<U>() ),
               void
            >::value,
         std::true_type
      >;

   template< typename >
   static constexpr std::false_type check(...);

   using type = decltype(check<std::decay_t<T>>(0));

public:
    static constexpr bool value = type::value;
};

/**
 * \brief Type trait for checking if T is a [scalar type](https://en.wikipedia.org/wiki/Scalar_(mathematics))
 * (in the mathemtatical sense). Not to be confused with \ref std::is_scalar.
 *
 * For example, \ref std::is_arithmetic "arithmetic types" as defined by the STL
 * are scalar types. TNL also provides additional scalar types, e.g. for
 * extended precision arithmetics. Users may also define specializations of this
 * trait class for their custom scalar types.
 */
template< typename T >
struct IsScalarType
: public std::is_arithmetic< T >
{};

/**
 * \brief Type trait for checking if T is an array type, e.g.
 *        \ref Containers::Array or \ref Containers::Vector.
 *
 * The trait combines \ref HasGetArrayDataMethod, \ref HasGetSizeMethod,
 * and \ref HasSubscriptOperator.
 */
template< typename T >
struct IsArrayType
: public std::integral_constant< bool,
            HasGetArrayDataMethod< T >::value &&
            HasGetSizeMethod< T >::value &&
            HasSubscriptOperator< T >::value >
{};

/**
 * \brief Type trait for checking if T is a vector type, e.g.
 *        \ref Containers::Vector or \ref Containers::VectorView.
 */
template< typename T >
struct IsVectorType
: public std::integral_constant< bool,
            IsArrayType< T >::value &&
            HasAddAssignmentOperator< T >::value >
{};

/**
 * \brief Type trait for checking if T has a \e constexpr \e getSize method.
 */
template< typename T >
struct HasConstexprGetSizeMethod
{
private:
   // implementation adopted from here: https://stackoverflow.com/a/50169108
   template< bool hasGetSize = HasGetSizeMethod< T >::value, typename = void >
   struct impl
   {
      // disable nvcc warning: invalid narrowing conversion from "unsigned int" to "int"
      // (the implementation is based on the conversion)
      #ifdef __NVCC__
         #ifdef __NVCC_DIAG_PRAGMA_SUPPORT__
            #pragma nv_diagnostic push
            #pragma nv_diag_suppress 2361
         #else
            #pragma push
            #pragma diag_suppress 2361
         #endif
      #elif defined(__INTEL_COMPILER)
         #pragma warning(push)
         #pragma warning(disable:3291)
      #endif
      template< typename M, M method >
      static constexpr std::true_type is_constexpr_impl( decltype(int{((*method)(), 0U)}) );
      #ifdef __NVCC__
         #ifdef __NVCC_DIAG_PRAGMA_SUPPORT__
            #pragma nv_diagnostic pop
         #else
            #pragma pop
         #endif
      #elif defined(__INTEL_COMPILER)
         // FIXME: this does not work - warning would be shown again...
         //#pragma warning(pop)
      #endif

      template< typename M, M method >
      static constexpr std::false_type is_constexpr_impl(...);

      using type = decltype(is_constexpr_impl< decltype(&std::decay_t<T>::getSize), &std::decay_t<T>::getSize >(0));
   };

   // specialization for types which don't have getSize() method at all
   template< typename _ >
   struct impl< false, _ >
   {
      using type = std::false_type;
   };

   using type = typename impl<>::type;

public:
   static constexpr bool value = type::value;
};

/**
 * \brief Type trait for checking if T is a static array type.
 *
 * Static array types are array types which have a \e constexpr \e getSize
 * method.
 */
template< typename T >
struct IsStaticArrayType
: public std::integral_constant< bool,
            HasConstexprGetSizeMethod< T >::value &&
            HasSubscriptOperator< T >::value >
{};

/**
 * \brief Type trait for checking if T is a view type.
 */
template< typename T >
struct IsViewType
{
private:
   template< typename C > static constexpr auto test(C)
      -> std::integral_constant< bool,
               std::is_same< typename C::ViewType, C >::value
         >;
   static constexpr std::false_type test(...);

public:
   static constexpr bool value = decltype( test(std::decay_t<T>{}) )::value;
};

/**
 * \brief Type trait for checking if T has getCommunicator method.
 */
template< typename T >
class HasGetCommunicatorMethod
{
private:
    typedef char YesType[1];
    typedef char NoType[2];

    template< typename C > static YesType& test( decltype(std::declval< C >().getCommunicator()) );
    template< typename C > static NoType& test(...);

public:
    static constexpr bool value = ( sizeof( test< std::decay_t<T> >(0) ) == sizeof( YesType ) );
};

/**
 * \brief Copy const qualifier from Source type to Target type.
 */
template< typename Target >
struct copy_const
{
   template< typename Source >
   struct from
   {
      using type = typename std::conditional<
       std::is_const< Source >::value,
       std::add_const_t< Target >, Target >::type;
   };
};

/**
 * \brief Type trait for checking if T has count member
 */
template< typename T >
class HasCountMember
{
private:
    typedef char YesType[1];
    typedef char NoType[2];

    template< typename C > static YesType& test( decltype( &C::count ) );
    template< typename C > static NoType& test(...);

public:
    static constexpr bool value = ( sizeof( test< std::decay_t<T> >(0) ) == sizeof( YesType ) );
};

} //namespace noaTNL
