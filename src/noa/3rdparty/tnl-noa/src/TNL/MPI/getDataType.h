// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#ifdef HAVE_MPI
   #include <mpi.h>
#endif

namespace noa::TNL {
namespace MPI {

#ifdef HAVE_MPI
template< typename T >
struct TypeResolver
{
   static inline MPI_Datatype
   getType()
   {
      static_assert( sizeof( T ) == sizeof( char ) || sizeof( T ) == sizeof( int ) || sizeof( T ) == sizeof( short int )
                        || sizeof( T ) == sizeof( long int ),
                     "Fatal Error - Unknown MPI Type" );
      switch( sizeof( T ) ) {
         case sizeof( char ):
            return MPI_CHAR;
         case sizeof( int ):
            return MPI_INT;
         case sizeof( short int ):
            return MPI_SHORT;
         case sizeof( long int ):
            return MPI_LONG;
      }
      // This will never happen thanks to the static_assert above, but icpc is
      // not that smart and complains about missing return statement at the end
      // of non-void function.
      throw 0;
   }
};

template<>
struct TypeResolver< char >
{
   static inline MPI_Datatype
   getType()
   {
      return MPI_CHAR;
   };
};

template<>
struct TypeResolver< int >
{
   static inline MPI_Datatype
   getType()
   {
      return MPI_INT;
   };
};

template<>
struct TypeResolver< short int >
{
   static inline MPI_Datatype
   getType()
   {
      return MPI_SHORT;
   };
};

template<>
struct TypeResolver< long int >
{
   static inline MPI_Datatype
   getType()
   {
      return MPI_LONG;
   };
};

template<>
struct TypeResolver< unsigned char >
{
   static inline MPI_Datatype
   getType()
   {
      return MPI_UNSIGNED_CHAR;
   };
};

template<>
struct TypeResolver< unsigned short int >
{
   static inline MPI_Datatype
   getType()
   {
      return MPI_UNSIGNED_SHORT;
   };
};

template<>
struct TypeResolver< unsigned int >
{
   static inline MPI_Datatype
   getType()
   {
      return MPI_UNSIGNED;
   };
};

template<>
struct TypeResolver< unsigned long int >
{
   static inline MPI_Datatype
   getType()
   {
      return MPI_UNSIGNED_LONG;
   };
};

template<>
struct TypeResolver< float >
{
   static inline MPI_Datatype
   getType()
   {
      return MPI_FLOAT;
   };
};

template<>
struct TypeResolver< double >
{
   static inline MPI_Datatype
   getType()
   {
      return MPI_DOUBLE;
   };
};

template<>
struct TypeResolver< long double >
{
   static inline MPI_Datatype
   getType()
   {
      return MPI_LONG_DOUBLE;
   };
};

template<>
struct TypeResolver< bool >
{
   // sizeof(bool) is implementation-defined: https://stackoverflow.com/a/4897859
   static_assert( sizeof( bool ) == 1, "The systems where sizeof(bool) != 1 are not supported by MPI." );
   static inline MPI_Datatype
   getType()
   {
      return MPI_C_BOOL;
   };
};

template< typename T >
MPI_Datatype
getDataType( const T& = T{} )
{
   return TypeResolver< T >::getType();
}
#endif

}  // namespace MPI
}  // namespace noa::TNL
