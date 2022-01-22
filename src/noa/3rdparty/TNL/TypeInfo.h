// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <typeinfo>
#include <type_traits>
#include <string>

#if defined( __has_include )
   #if __has_include(<cxxabi.h>)
      #define TNL_HAS_CXXABI_H
   #endif
#elif defined( __GLIBCXX__ ) || defined( __GLIBCPP__ )
   #define TNL_HAS_CXXABI_H
#endif

#if defined( TNL_HAS_CXXABI_H )
   #include <cxxabi.h>  // abi::__cxa_demangle
   #include <memory>  // std::unique_ptr
   #include <cstdlib>  // std::free
#endif

#include <noa/3rdparty/TNL/String.h>

namespace noaTNL {
namespace detail {

inline std::string
demangle( const char* name )
{
#if defined( TNL_HAS_CXXABI_H )
   int status = 0;
   std::size_t size = 0;
   std::unique_ptr<char[], void (*)(void*)> result(
      abi::__cxa_demangle( name, NULL, &size, &status ),
      std::free
   );
   if( result.get() )
      return result.get();
#endif
   return name;
}

/**
 * \brief Type trait for checking if T has a static getSerializationType method.
 */
template< typename T >
class HasStaticGetSerializationType
{
private:
   template< typename U >
   static constexpr auto check(U*)
   -> typename
      std::enable_if_t<
         ! std::is_same<
               decltype( U::getSerializationType() ),
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

} // namespace detail

/**
 * \brief Returns a human-readable string representation of given type.
 *
 * Note that since we use the \ref typeid operator internally, the top-level
 * cv-qualifiers are always ignored. See https://stackoverflow.com/a/8889143
 * for details.
 */
template< typename T >
String getType()
{
   return detail::demangle( typeid(T).name() );
}

/**
 * \brief Returns a human-readable string representation of given object's type.
 *
 * Note that since we use the \ref typeid operator internally, the top-level
 * cv-qualifiers are always ignored. See https://stackoverflow.com/a/8889143
 * for details.
 */
template< typename T >
String getType( T&& obj )
{
   return detail::demangle( typeid(obj).name() );
}

/**
 * \brief Returns a string identifying a type for the purpose of serialization.
 *
 * By default, this function returns the same string as \ref getType. However,
 * if a user-defined class has a static \e getSerializationType method, it is
 * called instead. This is useful for overriding the default \ref typeid name,
 * which may be necessary e.g. for class templates which should have the same
 * serialization type for multiple devices.
 */
template< typename T,
          std::enable_if_t< ! detail::HasStaticGetSerializationType< T >::value, bool > = true >
String getSerializationType()
{
   return getType< T >();
}

/**
 * \brief Specialization of \ref getSerializationType for types which provide a
 *        static \e getSerializationType method to override the default behaviour.
 */
template< typename T,
          std::enable_if_t< detail::HasStaticGetSerializationType< T >::value, bool > = true >
String getSerializationType()
{
   return T::getSerializationType();
}

} // namespace noaTNL
