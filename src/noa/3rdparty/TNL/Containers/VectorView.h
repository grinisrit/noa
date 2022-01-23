// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

// Implemented by: Jakub Klinkovský

#pragma once

#include <noa/3rdparty/TNL/Containers/ArrayView.h>
#include <noa/3rdparty/TNL/Containers/Expressions/ExpressionTemplates.h>

namespace noa::TNL {
namespace Containers {

/**
 * \brief \e VectorView extends \ref ArrayView with algebraic operations.
 *
 * The template parameters have the same meaning as in \ref ArrayView, with
 * \e Real corresponding to \e ArrayView's \e Value parameter.
 *
 * \tparam Real   An arithmetic type for the vector values, e.g. `float` or
 *                `double`.
 * \tparam Device The device to be used for the execution of vector operations.
 * \tparam Index  The indexing type.
 */
template< typename Real = double,
          typename Device = Devices::Host,
          typename Index = int >
class VectorView
: public ArrayView< Real, Device, Index >
{
   using BaseType = ArrayView< Real, Device, Index >;
   using NonConstReal = typename std::remove_const< Real >::type;
public:
   /**
    * \brief Type of elements stored in this vector.
    */
   using RealType = Real;

   /**
    * \brief Device where the vector is allocated.
    *
    * See \ref Devices::Host or \ref Devices::Cuda.
    */
   using DeviceType = Device;

   /**
    * \brief Type being used for the vector elements indexing.
    */
   using IndexType = Index;

   /**
    * \brief Compatible VectorView type.
    */
   using ViewType = VectorView< Real, Device, Index >;

   /**
    * \brief Compatible constant VectorView type.
    */
   using ConstViewType = VectorView< std::add_const_t< Real >, Device, Index >;

   /**
    * \brief A template which allows to quickly obtain a \ref VectorView type with changed template parameters.
    */
   template< typename _Real,
             typename _Device = Device,
             typename _Index = Index >
   using Self = VectorView< _Real, _Device, _Index >;


   // constructors and assignment operators inherited from the class ArrayView
   using ArrayView< Real, Device, Index >::ArrayView;
   using ArrayView< Real, Device, Index >::operator=;

   // In C++14, default constructors cannot be inherited, although Clang
   // and GCC since version 7.0 inherit them.
   // https://stackoverflow.com/a/51854172
   //! \brief Constructs an empty array view with zero size.
   __cuda_callable__
   VectorView() = default;

   /**
    * \brief Constructor for the initialization by a base class object.
    */
   // initialization by base class is not a copy constructor so it has to be explicit
   template< typename Real_ >  // template catches both const and non-const qualified Element
   __cuda_callable__
   VectorView( const ArrayView< Real_, Device, Index >& view )
   : BaseType( view ) {}

   /**
    * \brief Returns a modifiable view of the vector view.
    *
    * By default, a view for the whole vector is returned. If \e begin or
    * \e end is set to a non-zero value, a view only for the sub-interval
    * `[begin, end)` is returned.
    *
    * \param begin The beginning of the vector view sub-interval. It is 0 by
    *              default.
    * \param end The end of the vector view sub-interval. The default value is 0
    *            which is, however, replaced with the array size.
    */
   __cuda_callable__
   ViewType getView( IndexType begin = 0, IndexType end = 0 );

   /**
    * \brief Returns a non-modifiable view of the vector view.
    *
    * By default, a view for the whole vector is returned. If \e begin or
    * \e end is set to a non-zero value, a view only for the sub-interval
    * `[begin, end)` is returned.
    *
    * \param begin The beginning of the vector view sub-interval. It is 0 by
    *              default.
    * \param end The end of the vector view sub-interval. The default value is 0
    *            which is, however, replaced with the array size.
    */
   __cuda_callable__
   ConstViewType getConstView( IndexType begin = 0, IndexType end = 0 ) const;

   /**
    * \brief Assigns a vector expression to this vector view.
    *
    * The assignment is evaluated element-wise. The vector expression must
    * either evaluate to a scalar or a vector of the same size as this vector
    * view.
    *
    * \param expression The vector expression to be evaluated and assigned to
    *                   this vector view.
    * \return Reference to this vector view.
    */
   template< typename VectorExpression,
             typename...,
             typename = std::enable_if_t< Expressions::HasEnabledExpressionTemplates< VectorExpression >::value && ! IsArrayType< VectorExpression >::value > >
   VectorView& operator=( const VectorExpression& expression );

   /**
    * \brief Assigns a value or an array - same as \ref ArrayView::operator=.
    *
    * \return Reference to this vector view.
    */
   // operator= from the base class should be hidden according to the C++14 standard,
   // although GCC does not do that - see https://stackoverflow.com/q/57322624
#if !defined(__CUDACC_VER_MAJOR__) || __CUDACC_VER_MAJOR__ < 11
   template< typename T,
             typename...,
             typename = std::enable_if_t< std::is_convertible< T, Real >::value || IsArrayType< T >::value > >
   ArrayView< Real, Device, Index >&
   operator=( const T& data )
   {
      return ArrayView< Real, Device, Index >::operator=(data);
   }
#endif

   /**
    * \brief Adds elements of this vector view and a vector expression and
    * stores the result in this vector view.
    *
    * The addition is evaluated element-wise. The vector expression must
    * either evaluate to a scalar or a vector of the same size as this vector
    * view.
    *
    * \param expression Reference to a vector expression.
    * \return Reference to this vector view.
    */
   template< typename VectorExpression >
   VectorView& operator+=( const VectorExpression& expression );

   /**
    * \brief Subtracts elements of this vector view and a vector expression and
    * stores the result in this vector view.
    *
    * The subtraction is evaluated element-wise. The vector expression must
    * either evaluate to a scalar or a vector of the same size as this vector
    * view.
    *
    * \param expression Reference to a vector expression.
    * \return Reference to this vector view.
    */
   template< typename VectorExpression >
   VectorView& operator-=( const VectorExpression& expression );

   /**
    * \brief Multiplies elements of this vector view and a vector expression and
    * stores the result in this vector view.
    *
    * The multiplication is evaluated element-wise. The vector expression must
    * either evaluate to a scalar or a vector of the same size as this vector
    * view.
    *
    * \param expression Reference to a vector expression.
    * \return Reference to this vector view.
    */
   template< typename VectorExpression >
   VectorView& operator*=( const VectorExpression& expression );

   /**
    * \brief Divides elements of this vector view and a vector expression and
    * stores the result in this vector view.
    *
    * The division is evaluated element-wise. The vector expression must
    * either evaluate to a scalar or a vector of the same size as this vector
    * view.
    *
    * \param expression Reference to a vector expression.
    * \return Reference to this vector view.
    */
   template< typename VectorExpression >
   VectorView& operator/=( const VectorExpression& expression );

   /**
    * \brief Modulo assignment operator for vector view and a vector expression.
    *
    * The division is evaluated element-wise. The vector expression must
    * either evaluate to a scalar or a vector of the same size as this vector
    * view.
    *
    * \param expression Reference to a vector expression.
    * \return Reference to this vector.
    */
   template< typename VectorExpression >
   VectorView& operator%=( const VectorExpression& expression );
};

// Enable expression templates for VectorView
namespace Expressions {
   template< typename Real, typename Device, typename Index >
   struct HasEnabledExpressionTemplates< VectorView< Real, Device, Index > >
   : std::true_type
   {};
} // namespace Expressions

} // namespace Containers
} // namespace noa::TNL

#include <noa/3rdparty/TNL/Containers/VectorView.hpp>
