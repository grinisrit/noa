// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

// Implemented by: Jakub Klinkovský

#pragma once

#include <noa/3rdparty/TNL/Containers/DistributedArray.h>
#include <noa/3rdparty/TNL/Containers/DistributedVectorView.h>

namespace noaTNL {
namespace Containers {

template< typename Real,
          typename Device = Devices::Host,
          typename Index = int,
          typename Allocator = typename Allocators::Default< Device >::template Allocator< Real > >
class DistributedVector
: public DistributedArray< Real, Device, Index, Allocator >
{
   using BaseType = DistributedArray< Real, Device, Index, Allocator >;
public:
   using RealType = Real;
   using DeviceType = Device;
   using IndexType = Index;
   using AllocatorType = Allocator;
   using LocalViewType = Containers::VectorView< Real, Device, Index >;
   using ConstLocalViewType = Containers::VectorView< std::add_const_t< Real >, Device, Index >;
   using ViewType = DistributedVectorView< Real, Device, Index >;
   using ConstViewType = DistributedVectorView< std::add_const_t< Real >, Device, Index >;

   /**
    * \brief A template which allows to quickly obtain a \ref Vector type with changed template parameters.
    */
   template< typename _Real,
             typename _Device = Device,
             typename _Index = Index,
             typename _Allocator = typename Allocators::Default< _Device >::template Allocator< _Real > >
   using Self = DistributedVector< _Real, _Device, _Index, _Allocator >;


   // inherit all constructors and assignment operators from Array
   using BaseType::DistributedArray;
#if !defined(__CUDACC_VER_MAJOR__) || __CUDACC_VER_MAJOR__ < 11
   using BaseType::operator=;
#endif

   DistributedVector() = default;

   /**
    * \brief Copy constructor (makes a deep copy).
    */
   explicit DistributedVector( const DistributedVector& ) = default;

   /**
    * \brief Copy constructor with a specific allocator (makes a deep copy).
    */
   explicit DistributedVector( const DistributedVector& vector, const AllocatorType& allocator );

   /**
    * \brief Default move constructor.
    */
   DistributedVector( DistributedVector&& ) = default;

   /**
    * \brief Copy-assignment operator for copying data from another vector.
    */
   DistributedVector& operator=( const DistributedVector& ) = default;

   /**
    * \brief Move-assignment operator for acquiring data from \e rvalues.
    */
   DistributedVector& operator=( DistributedVector&& ) = default;

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
    * \brief Returns a modifiable view of the vector.
    */
   ViewType getView();

   /**
    * \brief Returns a non-modifiable view of the vector.
    */
   ConstViewType getConstView() const;

   /**
    * \brief Conversion operator to a modifiable view of the vector.
    */
   operator ViewType();

   /**
    * \brief Conversion operator to a non-modifiable view of the vector.
    */
   operator ConstViewType() const;


   /*
    * Usual Vector methods follow below.
    */
   template< typename Scalar,
             typename...,
             typename = std::enable_if_t< ! HasSubscriptOperator<Scalar>::value > >
   DistributedVector& operator=( Scalar c );

   template< typename Scalar,
             typename...,
             typename = std::enable_if_t< ! HasSubscriptOperator<Scalar>::value > >
   DistributedVector& operator+=( Scalar c );

   template< typename Scalar,
             typename...,
             typename = std::enable_if_t< ! HasSubscriptOperator<Scalar>::value > >
   DistributedVector& operator-=( Scalar c );

   template< typename Scalar,
             typename...,
             typename = std::enable_if_t< ! HasSubscriptOperator<Scalar>::value > >
   DistributedVector& operator*=( Scalar c );

   template< typename Scalar,
             typename...,
             typename = std::enable_if_t< ! HasSubscriptOperator<Scalar>::value > >
   DistributedVector& operator/=( Scalar c );

   template< typename Scalar,
             typename...,
             typename = std::enable_if_t< ! HasSubscriptOperator<Scalar>::value > >
   DistributedVector& operator%=( Scalar c );

   template< typename Vector,
             typename...,
             typename = std::enable_if_t< HasSubscriptOperator<Vector>::value > >
   DistributedVector& operator=( const Vector& vector );

   template< typename Vector,
             typename...,
             typename = std::enable_if_t< HasSubscriptOperator<Vector>::value > >
   DistributedVector& operator+=( const Vector& vector );

   template< typename Vector,
             typename...,
             typename = std::enable_if_t< HasSubscriptOperator<Vector>::value > >
   DistributedVector& operator-=( const Vector& vector );

   template< typename Vector,
             typename...,
             typename = std::enable_if_t< HasSubscriptOperator<Vector>::value > >
   DistributedVector& operator*=( const Vector& vector );

   template< typename Vector,
             typename...,
             typename = std::enable_if_t< HasSubscriptOperator<Vector>::value > >
   DistributedVector& operator/=( const Vector& vector );

   template< typename Vector,
             typename...,
             typename = std::enable_if_t< HasSubscriptOperator<Vector>::value > >
   DistributedVector& operator%=( const Vector& vector );
};

// Enable expression templates for DistributedVector
namespace Expressions {
   template< typename Real, typename Device, typename Index, typename Allocator >
   struct HasEnabledDistributedExpressionTemplates< DistributedVector< Real, Device, Index, Allocator > >
   : std::true_type
   {};
} // namespace Expressions

} // namespace Containers
} // namespace noaTNL

#include <noa/3rdparty/TNL/Containers/DistributedVector.hpp>
