#pragma once

#include <iostream>
#include <limits>

#include <TNL/Math.h>
#include <TNL/MPI/getDataType.h>

namespace TNL {

template< class T >
struct CustomScalar
{
   T value = 0;

public:
   constexpr CustomScalar() = default;

   constexpr CustomScalar( T value ) : value( value ) {}

   template< typename S >
   constexpr CustomScalar( const CustomScalar< S >& v ) : value( v.value ) {}

   constexpr CustomScalar( const CustomScalar& ) = default;

   constexpr CustomScalar( CustomScalar&& ) = default;

   constexpr CustomScalar& operator=( const CustomScalar& v ) = default;

   constexpr CustomScalar& operator=( CustomScalar&& v ) = default;

#define MAKE_ASSIGNMENT_OP(op) \
   template< typename S >                                            \
   constexpr CustomScalar& operator op( const CustomScalar< S >& v ) \
   {                                                                 \
      value op v.value;                                              \
      return *this;                                                  \
   }                                                                 \
   template< typename S >                                            \
   constexpr CustomScalar& operator op( const S& v )                 \
   {                                                                 \
      value op v;                                                    \
      return *this;                                                  \
   }                                                                 \

   MAKE_ASSIGNMENT_OP(+=)
   MAKE_ASSIGNMENT_OP(-=)
   MAKE_ASSIGNMENT_OP(*=)
   MAKE_ASSIGNMENT_OP(/=)
   MAKE_ASSIGNMENT_OP(%=)
   MAKE_ASSIGNMENT_OP(&=)
   MAKE_ASSIGNMENT_OP(|=)
   MAKE_ASSIGNMENT_OP(^=)
   MAKE_ASSIGNMENT_OP(<<=)
   MAKE_ASSIGNMENT_OP(>>=)

#undef MAKE_ASSIGNMENT_OP

   // bitwise negation
   constexpr bool operator~() const
   {
      return ~ value;
   }

   // logical negation
   constexpr bool operator!() const
   {
      return ! value;
   }

   // unary plus (integer promotion)
   constexpr auto operator+() const -> CustomScalar< decltype(+value) >
   {
      return +value;
   }

   // unary minus (additive inverse)
   constexpr CustomScalar operator-() const
   {
      return -value;
   }

   // prefix increment
   constexpr CustomScalar& operator++()
   {
      ++value;
      return *this;
   }

   // prefix decrement
   constexpr CustomScalar& operator--()
   {
      --value;
      return *this;
   }

   // postfix increment
   constexpr CustomScalar operator++(int)
   {
      CustomScalar result = *this;
      value++;
      return result;
   }

   // postfix decrement
   constexpr CustomScalar operator--(int)
   {
      CustomScalar result = *this;
      value--;
      return result;
   }

   // cast to T
   constexpr operator T() const
   {
      return value;
   }
};

#define MAKE_BINARY_OP(op)                                              \
template< class T, class S >                                            \
constexpr auto operator op( const CustomScalar< T >& v1,                \
                            const CustomScalar< S >& v2 )               \
   -> CustomScalar< decltype( v1.value op v2.value ) >                  \
{                                                                       \
   return v1.value op v2.value;                                         \
}                                                                       \
template< class T, class S >                                            \
constexpr auto operator op( const CustomScalar< T >& v1, const S& v2 )  \
   -> CustomScalar< decltype( v1.value op v2 ) >                        \
{                                                                       \
   return v1.value op v2;                                               \
}                                                                       \
template< class S, class T >                                            \
constexpr auto operator op( const S& v1, const CustomScalar< T >& v2 )  \
   -> CustomScalar< decltype( v1 op v2.value ) >                        \
{                                                                       \
   return v1 op v2.value;                                               \
}                                                                       \

MAKE_BINARY_OP(+)
MAKE_BINARY_OP(-)
MAKE_BINARY_OP(*)
MAKE_BINARY_OP(/)
MAKE_BINARY_OP(%)
MAKE_BINARY_OP(&)
MAKE_BINARY_OP(|)
MAKE_BINARY_OP(^)
MAKE_BINARY_OP(<<)
MAKE_BINARY_OP(>>)

#undef MAKE_BINARY_OP

#define MAKE_BOOL_BINARY_OP(op)                                         \
template< class T, class S >                                            \
constexpr bool operator op( const CustomScalar< T >& v1,                \
                            const CustomScalar< S >& v2 )               \
{                                                                       \
   return v1.value op v2.value;                                         \
}                                                                       \
template< class T, class S >                                            \
constexpr bool operator op( const CustomScalar< T >& v1, const S& v2 )  \
{                                                                       \
   return v1.value op v2;                                               \
}                                                                       \
template< class S, class T >                                            \
constexpr bool operator op( const S& v1, const CustomScalar< T >& v2 )  \
{                                                                       \
   return v1 op v2.value;                                               \
}                                                                       \

MAKE_BOOL_BINARY_OP(==)
MAKE_BOOL_BINARY_OP(!=)
MAKE_BOOL_BINARY_OP(<=)
MAKE_BOOL_BINARY_OP(>=)
MAKE_BOOL_BINARY_OP(<)
MAKE_BOOL_BINARY_OP(>)
MAKE_BOOL_BINARY_OP(&&)
MAKE_BOOL_BINARY_OP(||)

#undef MAKE_BOOL_BINARY_OP

template< class T >
std::istream& operator>>( std::istream& str, const CustomScalar< T >& v )
{
   return str >> v.value;
}

template< class T >
std::ostream& operator<<( std::ostream& str, const CustomScalar< T >& v )
{
   return str << v.value;
}

#define MAKE_UNARY_FUNC(fname)                              \
   template< class T >                                      \
   constexpr auto fname ( const CustomScalar< T >& v )      \
      -> CustomScalar< decltype(TNL::fname( v.value )) >    \
   { return TNL::fname( v.value ); }                        \

#define MAKE_BINARY_FUNC(fname)                                                     \
   template< class T, class S >                                                     \
   constexpr auto fname ( const CustomScalar< T >& v, const CustomScalar< S >& w )  \
      -> CustomScalar< decltype(TNL::fname( v.value, w.value )) >                   \
   { return TNL::fname( v.value, w.value ); }                                       \
   template< class T, class S >                                                     \
   constexpr auto fname ( const CustomScalar< T >& v, const S& w )                  \
      -> CustomScalar< decltype(TNL::fname( v.value, w )) >                         \
   { return TNL::fname( v.value, w ); }                                             \
   template< class S, class T >                                                     \
   constexpr auto fname ( const S& w, const CustomScalar< T >& v )                  \
      -> CustomScalar< decltype(TNL::fname( w, v.value )) >                         \
   { return TNL::fname( w, v.value ); }                                             \

MAKE_UNARY_FUNC( abs )
MAKE_UNARY_FUNC( sqrt )
MAKE_UNARY_FUNC( cbrt )
MAKE_UNARY_FUNC( exp )
MAKE_UNARY_FUNC( log )
MAKE_UNARY_FUNC( log10 )
MAKE_UNARY_FUNC( log2 )
MAKE_UNARY_FUNC( sin )
MAKE_UNARY_FUNC( cos )
MAKE_UNARY_FUNC( tan )
MAKE_UNARY_FUNC( asin )
MAKE_UNARY_FUNC( acos )
MAKE_UNARY_FUNC( atan )
MAKE_UNARY_FUNC( sinh )
MAKE_UNARY_FUNC( cosh )
MAKE_UNARY_FUNC( tanh )
MAKE_UNARY_FUNC( asinh )
MAKE_UNARY_FUNC( acosh )
MAKE_UNARY_FUNC( atanh )
MAKE_UNARY_FUNC( floor )
MAKE_UNARY_FUNC( ceil )

MAKE_BINARY_FUNC( min )
MAKE_BINARY_FUNC( max )
MAKE_BINARY_FUNC( argMin )
MAKE_BINARY_FUNC( argMax )
MAKE_BINARY_FUNC( argAbsMin )
MAKE_BINARY_FUNC( argAbsMax )
MAKE_BINARY_FUNC( pow )

#undef MAKE_UNARY_FUNC
#undef MAKE_BINARY_FUNC

} // namespace TNL

namespace std {
   template< typename T >
   struct numeric_limits< TNL::CustomScalar< T > > : public numeric_limits< T > {};
} // namespace std

namespace TNL {
   template< typename T >
   struct IsScalarType< CustomScalar< T > > : public std::true_type {};
} // namespace TNL

#ifdef HAVE_MPI
namespace TNL {
namespace MPI {
   template< typename T >
   struct TypeResolver< CustomScalar< T > > : public TypeResolver< T > {};
} // namespace MPI
} // namespace TNL
#endif
