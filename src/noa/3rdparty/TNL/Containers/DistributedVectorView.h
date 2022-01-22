// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

// Implemented by: Jakub Klinkovský

#pragma once

#include <noa/3rdparty/TNL/Containers/DistributedArrayView.h>
#include <noa/3rdparty/TNL/Containers/Expressions/DistributedExpressionTemplates.h>
#include <noa/3rdparty/TNL/Containers/VectorView.h>

namespace noaTNL {
namespace Containers {

template< typename Real,
          typename Device = Devices::Host,
          typename Index = int >
class DistributedVectorView
: public DistributedArrayView< Real, Device, Index >
{
   using BaseType = DistributedArrayView< Real, Device, Index >;
   using NonConstReal = typename std::remove_const< Real >::type;
public:
   using RealType = Real;
   using DeviceType = Device;
   using IndexType = Index;
   using LocalViewType = Containers::VectorView< Real, Device, Index >;
   using ConstLocalViewType = Containers::VectorView< std::add_const_t< Real >, Device, Index >;
   using ViewType = DistributedVectorView< Real, Device, Index >;
   using ConstViewType = DistributedVectorView< std::add_const_t< Real >, Device, Index >;

   /**
    * \brief A template which allows to quickly obtain a \ref VectorView type with changed template parameters.
    */
   template< typename _Real,
             typename _Device = Device,
             typename _Index = Index >
   using Self = DistributedVectorView< _Real, _Device, _Index >;


   // inherit all constructors and assignment operators from ArrayView
   using BaseType::DistributedArrayView;
#if !defined(__CUDACC_VER_MAJOR__) || __CUDACC_VER_MAJOR__ < 11
   using BaseType::operator=;
#endif

   // In C++14, default constructors cannot be inherited, although Clang
   // and GCC since version 7.0 inherit them.
   // https://stackoverflow.com/a/51854172
   DistributedVectorView() = default;

   // initialization by base class is not a copy constructor so it has to be explicit
   template< typename Real_ >  // template catches both const and non-const qualified Element
   DistributedVectorView( const Containers::DistributedArrayView< Real_, Device, Index >& view )
   : BaseType( view ) {}

   /**
    * \brief Returns a modifiable view of the local part of the vector.
    */
   LocalViewType getLocalView();

   /**
    * \brief Returns a non-modifiable view of the local part of the vector.
    */
   ConstLocalViewType getConstLocalView() const;

   /**
    * \brief Returns a modifiable view of the local part of the vector,
    * including ghost values.
    */
   LocalViewType getLocalViewWithGhosts();

   /**
    * \brief Returns a non-modifiable view of the local part of the vector,
    * including ghost values.
    */
   ConstLocalViewType getConstLocalViewWithGhosts() const;

   /**
    * \brief Returns a modifiable view of the array view.
    */
   ViewType getView();

   /**
    * \brief Returns a non-modifiable view of the array view.
    */
   ConstViewType getConstView() const;

   /*
    * Usual Vector methods follow below.
    */
   template< typename Scalar,
             typename...,
             typename = std::enable_if_t< ! HasSubscriptOperator<Scalar>::value > >
   DistributedVectorView& operator=( Scalar c );

   template< typename Scalar,
             typename...,
             typename = std::enable_if_t< ! HasSubscriptOperator<Scalar>::value > >
   DistributedVectorView& operator+=( Scalar c );

   template< typename Scalar,
             typename...,
             typename = std::enable_if_t< ! HasSubscriptOperator<Scalar>::value > >
   DistributedVectorView& operator-=( Scalar c );

   template< typename Scalar,
             typename...,
             typename = std::enable_if_t< ! HasSubscriptOperator<Scalar>::value > >
   DistributedVectorView& operator*=( Scalar c );

   template< typename Scalar,
             typename...,
             typename = std::enable_if_t< ! HasSubscriptOperator<Scalar>::value > >
   DistributedVectorView& operator/=( Scalar c );

   template< typename Scalar,
             typename...,
             typename = std::enable_if_t< ! HasSubscriptOperator<Scalar>::value > >
   DistributedVectorView& operator%=( Scalar c );

   template< typename Vector,
             typename...,
             typename = std::enable_if_t< HasSubscriptOperator<Vector>::value > >
   DistributedVectorView& operator=( const Vector& vector );

   template< typename Vector,
             typename...,
             typename = std::enable_if_t< HasSubscriptOperator<Vector>::value > >
   DistributedVectorView& operator+=( const Vector& vector );

   template< typename Vector,
             typename...,
             typename = std::enable_if_t< HasSubscriptOperator<Vector>::value > >
   DistributedVectorView& operator-=( const Vector& vector );

   template< typename Vector,
             typename...,
             typename = std::enable_if_t< HasSubscriptOperator<Vector>::value > >
   DistributedVectorView& operator*=( const Vector& vector );

   template< typename Vector,
             typename...,
             typename = std::enable_if_t< HasSubscriptOperator<Vector>::value > >
   DistributedVectorView& operator/=( const Vector& vector );

   template< typename Vector,
             typename...,
             typename = std::enable_if_t< HasSubscriptOperator<Vector>::value > >
   DistributedVectorView& operator%=( const Vector& vector );
};

// Enable expression templates for DistributedVector
namespace Expressions {
   template< typename Real, typename Device, typename Index >
   struct HasEnabledDistributedExpressionTemplates< DistributedVectorView< Real, Device, Index > >
   : std::true_type
   {};
} // namespace Expressions

} // namespace Containers
} // namespace noaTNL

#include <noa/3rdparty/TNL/Containers/DistributedVectorView.hpp>
