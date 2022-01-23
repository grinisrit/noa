// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <noa/3rdparty/TNL/TypeTraits.h>
#include <noa/3rdparty/TNL/Containers/Expressions/TypeTraits.h>
#include <noa/3rdparty/TNL/Algorithms/ParallelFor.h>

namespace noa::TNL {
namespace Containers {
namespace detail {

/**
 * \brief Vector assignment
 */
template< typename Vector,
          typename T,
          bool vectorVectorAssignment = HasSubscriptOperator< T >::value && ! Expressions::IsArithmeticSubtype< T, Vector >::value >
struct VectorAssignment;

/**
 * \brief Vector assignment with an operation: +=, -=, *=, /=, %=
 */
template< typename Vector,
          typename T,
          bool vectorVectorAssignment = HasSubscriptOperator< T >::value && ! Expressions::IsArithmeticSubtype< T, Vector >::value,
          bool hasSetSizeMethod = HasSetSizeMethod< T >::value >
struct VectorAssignmentWithOperation;

/**
 * \brief Specialization for vector-vector assignment.
 */
template< typename Vector,
          typename T >
struct VectorAssignment< Vector, T, true >
{
   static void resize( Vector& v, const T& t )
   {
      v.setSize( t.getSize() );
   }

   __cuda_callable__
   static void assignStatic( Vector& v, const T& t )
   {
      TNL_ASSERT_EQ( v.getSize(), t.getSize(), "The sizes of the vectors must be equal." );
      for( decltype( v.getSize() ) i = 0; i < v.getSize(); i++ )
         v[ i ] = t[ i ];
   }

   static void assign( Vector& v, const T& t )
   {
      static_assert( std::is_same< typename Vector::DeviceType, typename T::DeviceType >::value,
                     "Cannot assign an expression to a vector allocated on a different device." );
      TNL_ASSERT_EQ( v.getSize(), t.getSize(), "The sizes of the vectors must be equal." );
      using RealType = typename Vector::RealType;
      using DeviceType = typename Vector::DeviceType;
      using IndexType = typename Vector::IndexType;

      RealType* data = v.getData();
      auto assignment = [=] __cuda_callable__ ( IndexType i )
      {
         data[ i ] = t[ i ];
      };
      Algorithms::ParallelFor< DeviceType >::exec( ( IndexType ) 0, v.getSize(), assignment );
   }
};

/**
 * \brief Specialization for vector-value assignment. We assume that T is assignable to Vector::RealType.
 */
template< typename Vector,
          typename T >
struct VectorAssignment< Vector, T, false >
{
   static void resize( Vector& v, const T& t )
   {
   }

   __cuda_callable__
   static void assignStatic( Vector& v, const T& t )
   {
      for( decltype( v.getSize() ) i = 0; i < v.getSize(); i++ )
         v[ i ] = t;
   }

   static void assign( Vector& v, const T& t )
   {
      using RealType = typename Vector::RealType;
      using DeviceType = typename Vector::DeviceType;
      using IndexType = typename Vector::IndexType;

      RealType* data = v.getData();
      auto assignment = [=] __cuda_callable__ ( IndexType i )
      {
         data[ i ] = t;
      };
      Algorithms::ParallelFor< DeviceType >::exec( ( IndexType ) 0, v.getSize(), assignment );
   }
};

/**
 * \brief Specialization for types with subscript operator and setSize
 *        method - i.e. for Vectors.
 *
 * This is necessary, because Vectors cannot be passed-by-value to CUDA kernels
 * so we have to do it with views.
 */
template< typename Vector,
          typename T >
struct VectorAssignmentWithOperation< Vector, T, true, true >
{
   static void addition( Vector& v, const T& t )
   {
      VectorAssignmentWithOperation< Vector, typename T::ConstViewType >::addition( v, t.getConstView() );
   }

   static void subtraction( Vector& v, const T& t )
   {
      VectorAssignmentWithOperation< Vector, typename T::ConstViewType >::subtraction( v, t.getConstView() );
   }

   static void multiplication( Vector& v, const T& t )
   {
      VectorAssignmentWithOperation< Vector, typename T::ConstViewType >::multiplication( v, t.getConstView() );
   }

   static void division( Vector& v, const T& t )
   {
      VectorAssignmentWithOperation< Vector, typename T::ConstViewType >::subtraction( v, t.getConstView() );
   }
};

/**
 * \brief Specialization for types with subscript operator, but without setSize
 *        method - i.e. for expressions, views and static vectors.
 */
template< typename Vector,
          typename T >
struct VectorAssignmentWithOperation< Vector, T, true, false >
{
   __cuda_callable__
   static void additionStatic( Vector& v, const T& t )
   {
      TNL_ASSERT_EQ( v.getSize(), t.getSize(), "The sizes of the vectors must be equal." );
      for( decltype( v.getSize() ) i = 0; i < v.getSize(); i++ )
         v[ i ] += t[ i ];
   }

   static void addition( Vector& v, const T& t )
   {
      static_assert( std::is_same< typename Vector::DeviceType, typename T::DeviceType >::value,
                     "Cannot assign an expression to a vector allocated on a different device." );
      TNL_ASSERT_EQ( v.getSize(), t.getSize(), "The sizes of the vectors must be equal." );
      using RealType = typename Vector::RealType;
      using DeviceType = typename Vector::DeviceType;
      using IndexType = typename Vector::IndexType;

      RealType* data = v.getData();
      auto add = [=] __cuda_callable__ ( IndexType i )
      {
         data[ i ] += t[ i ];
      };
      Algorithms::ParallelFor< DeviceType >::exec( ( IndexType ) 0, v.getSize(), add );
   }

   __cuda_callable__
   static void subtractionStatic( Vector& v, const T& t )
   {
      TNL_ASSERT_EQ( v.getSize(), t.getSize(), "The sizes of the vectors must be equal." );
      for( decltype( v.getSize() ) i = 0; i < v.getSize(); i++ )
         v[ i ] -= t[ i ];
   }

   static void subtraction( Vector& v, const T& t )
   {
      static_assert( std::is_same< typename Vector::DeviceType, typename T::DeviceType >::value,
                     "Cannot assign an expression to a vector allocated on a different device." );
      TNL_ASSERT_EQ( v.getSize(), t.getSize(), "The sizes of the vectors must be equal." );
      using RealType = typename Vector::RealType;
      using DeviceType = typename Vector::DeviceType;
      using IndexType = typename Vector::IndexType;

      RealType* data = v.getData();
      auto subtract = [=] __cuda_callable__ ( IndexType i )
      {
         data[ i ] -= t[ i ];
      };
      Algorithms::ParallelFor< DeviceType >::exec( ( IndexType ) 0, v.getSize(), subtract );
   }

   __cuda_callable__
   static void multiplicationStatic( Vector& v, const T& t )
   {
      TNL_ASSERT_EQ( v.getSize(), t.getSize(), "The sizes of the vectors must be equal." );
      for( decltype( v.getSize() ) i = 0; i < v.getSize(); i++ )
         v[ i ] *= t[ i ];
   }

   static void multiplication( Vector& v, const T& t )
   {
      static_assert( std::is_same< typename Vector::DeviceType, typename T::DeviceType >::value,
                     "Cannot assign an expression to a vector allocated on a different device." );
      TNL_ASSERT_EQ( v.getSize(), t.getSize(), "The sizes of the vectors must be equal." );
      using RealType = typename Vector::RealType;
      using DeviceType = typename Vector::DeviceType;
      using IndexType = typename Vector::IndexType;

      RealType* data = v.getData();
      auto multiply = [=] __cuda_callable__ ( IndexType i )
      {
         data[ i ] *= t[ i ];
      };
      Algorithms::ParallelFor< DeviceType >::exec( ( IndexType ) 0, v.getSize(), multiply );
   }

   __cuda_callable__
   static void divisionStatic( Vector& v, const T& t )
   {
      TNL_ASSERT_EQ( v.getSize(), t.getSize(), "The sizes of the vectors must be equal." );
      for( decltype( v.getSize() ) i = 0; i < v.getSize(); i++ )
         v[ i ] /= t[ i ];
   }

   static void division( Vector& v, const T& t )
   {
      static_assert( std::is_same< typename Vector::DeviceType, typename T::DeviceType >::value,
                     "Cannot assign an expression to a vector allocated on a different device." );
      TNL_ASSERT_EQ( v.getSize(), t.getSize(), "The sizes of the vectors must be equal." );
      using RealType = typename Vector::RealType;
      using DeviceType = typename Vector::DeviceType;
      using IndexType = typename Vector::IndexType;

      RealType* data = v.getData();
      auto divide = [=] __cuda_callable__ ( IndexType i )
      {
         data[ i ] /= t[ i ];
      };
      Algorithms::ParallelFor< DeviceType >::exec( ( IndexType ) 0, v.getSize(), divide );
   }

   __cuda_callable__
   static void moduloStatic( Vector& v, const T& t )
   {
      TNL_ASSERT_EQ( v.getSize(), t.getSize(), "The sizes of the vectors must be equal." );
      for( decltype( v.getSize() ) i = 0; i < v.getSize(); i++ )
         v[ i ] %= t[ i ];
   }

   static void modulo( Vector& v, const T& t )
   {
      static_assert( std::is_same< typename Vector::DeviceType, typename T::DeviceType >::value,
                     "Cannot assign an expression to a vector allocated on a different device." );
      TNL_ASSERT_EQ( v.getSize(), t.getSize(), "The sizes of the vectors must be equal." );
      using RealType = typename Vector::RealType;
      using DeviceType = typename Vector::DeviceType;
      using IndexType = typename Vector::IndexType;

      RealType* data = v.getData();
      auto divide = [=] __cuda_callable__ ( IndexType i )
      {
         data[ i ] %= t[ i ];
      };
      Algorithms::ParallelFor< DeviceType >::exec( ( IndexType ) 0, v.getSize(), divide );
   }
};

/**
 * \brief Specialization for array-value assignment for other types. We assume
 * that T is convertible to Vector::ValueType.
 */
template< typename Vector,
          typename T >
struct VectorAssignmentWithOperation< Vector, T, false, false >
{
   __cuda_callable__
   static void additionStatic( Vector& v, const T& t )
   {
      for( decltype( v.getSize() ) i = 0; i < v.getSize(); i++ )
         v[ i ] += t;
   }

   static void addition( Vector& v, const T& t )
   {
      using RealType = typename Vector::RealType;
      using DeviceType = typename Vector::DeviceType;
      using IndexType = typename Vector::IndexType;

      RealType* data = v.getData();
      auto add = [=] __cuda_callable__ ( IndexType i )
      {
         data[ i ] += t;
      };
      Algorithms::ParallelFor< DeviceType >::exec( ( IndexType ) 0, v.getSize(), add );
   }

   __cuda_callable__
   static void subtractionStatic( Vector& v, const T& t )
   {
      for( decltype( v.getSize() ) i = 0; i < v.getSize(); i++ )
         v[ i ] -= t;
   }

   static void subtraction( Vector& v, const T& t )
   {
      using RealType = typename Vector::RealType;
      using DeviceType = typename Vector::DeviceType;
      using IndexType = typename Vector::IndexType;

      RealType* data = v.getData();
      auto subtract = [=] __cuda_callable__ ( IndexType i )
      {
         data[ i ] -= t;
      };
      Algorithms::ParallelFor< DeviceType >::exec( ( IndexType ) 0, v.getSize(), subtract );
   }

   __cuda_callable__
   static void multiplicationStatic( Vector& v, const T& t )
   {
      for( decltype( v.getSize() ) i = 0; i < v.getSize(); i++ )
         v[ i ] *= t;
   }

   static void multiplication( Vector& v, const T& t )
   {
      using RealType = typename Vector::RealType;
      using DeviceType = typename Vector::DeviceType;
      using IndexType = typename Vector::IndexType;

      RealType* data = v.getData();
      auto multiply = [=] __cuda_callable__ ( IndexType i )
      {
         data[ i ] *= t;
      };
      Algorithms::ParallelFor< DeviceType >::exec( ( IndexType ) 0, v.getSize(), multiply );
   }

   __cuda_callable__
   static void divisionStatic( Vector& v, const T& t )
   {
      for( decltype( v.getSize() ) i = 0; i < v.getSize(); i++ )
         v[ i ] /= t;
   }

   static void division( Vector& v, const T& t )
   {
      using RealType = typename Vector::RealType;
      using DeviceType = typename Vector::DeviceType;
      using IndexType = typename Vector::IndexType;

      RealType* data = v.getData();
      auto divide = [=] __cuda_callable__ ( IndexType i )
      {
         data[ i ] /= t;
      };
      Algorithms::ParallelFor< DeviceType >::exec( ( IndexType ) 0, v.getSize(), divide );
   }

   __cuda_callable__
   static void moduloStatic( Vector& v, const T& t )
   {
      for( decltype( v.getSize() ) i = 0; i < v.getSize(); i++ )
         v[ i ] %= t;
   }

   static void modulo( Vector& v, const T& t )
   {
      using RealType = typename Vector::RealType;
      using DeviceType = typename Vector::DeviceType;
      using IndexType = typename Vector::IndexType;

      RealType* data = v.getData();
      auto divide = [=] __cuda_callable__ ( IndexType i )
      {
         data[ i ] %= t;
      };
      Algorithms::ParallelFor< DeviceType >::exec( ( IndexType ) 0, v.getSize(), divide );
   }
};

} // namespace detail
} // namespace Containers
} // namespace noa::TNL
