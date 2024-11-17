// Copyright (c) 2004-2023 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

// Implemented by: Jakub Klinkovský

#pragma once

#include <noa/3rdparty/tnl-noa/src/TNL/Containers/Vector.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Containers/VectorView.h>

namespace noa::TNL {
namespace Matrices {
namespace __DistributedSpMV_impl {

template< typename Real, typename Device = Devices::Host, typename Index = int >
class ThreePartVectorView
{
   using ConstReal = std::add_const_t< Real >;

public:
   using RealType = Real;
   using DeviceType = Device;
   using IndexType = Index;
   using VectorView = Containers::VectorView< Real, Device, Index >;

   ThreePartVectorView() = default;
   ThreePartVectorView( const ThreePartVectorView& ) = default;
   ThreePartVectorView( ThreePartVectorView&& ) noexcept = default;

   ThreePartVectorView( VectorView view_left, VectorView view_mid, VectorView view_right )
   {
      bind( view_left, view_mid, view_right );
   }

   void
   bind( VectorView view_left, VectorView view_mid, VectorView view_right )
   {
      left.bind( view_left );
      middle.bind( view_mid );
      right.bind( view_right );
   }

   void
   reset()
   {
      left.reset();
      middle.reset();
      right.reset();
   }

   IndexType
   getSize() const
   {
      return left.getSize() + middle.getSize() + right.getSize();
   }

   ThreePartVectorView< ConstReal, Device, Index >
   getConstView() const
   {
      return { left.getConstView(), middle, right.getConstView() };
   }

   //   __cuda_callable__
   //   Real& operator[]( Index i )
   //   {
   //      if( i < left.getSize() )
   //         return left[ i ];
   //      else if( i < left.getSize() + middle.getSize() )
   //         return middle[ i - left.getSize() ];
   //      else
   //         return right[ i - left.getSize() - middle.getSize() ];
   //   }

   __cuda_callable__
   const Real&
   operator[]( Index i ) const
   {
      if( i < left.getSize() )
         return left[ i ];
      else if( i < left.getSize() + middle.getSize() )
         return middle[ i - left.getSize() ];
      else
         return right[ i - left.getSize() - middle.getSize() ];
   }

   __cuda_callable__
   const Real*
   getPointer( Index i ) const
   {
      if( i < left.getSize() )
         return &left.getData()[ i ];
      else if( i < left.getSize() + middle.getSize() )
         return &middle.getData()[ i - left.getSize() ];
      else
         return &right.getData()[ i - left.getSize() - middle.getSize() ];
   }

   friend std::ostream&
   operator<<( std::ostream& str, const ThreePartVectorView& v )
   {
      str << "[\n\tleft: " << v.left << ",\n\tmiddle: " << v.middle << ",\n\tright: " << v.right << "\n]";
      return str;
   }

protected:
   VectorView left, middle, right;
};

template< typename Real, typename Device = Devices::Host, typename Index = int >
class ThreePartVector
{
   using ConstReal = std::add_const_t< Real >;

public:
   using RealType = Real;
   using DeviceType = Device;
   using IndexType = Index;
   using Vector = Containers::Vector< Real, Device, Index >;
   using VectorView = Containers::VectorView< Real, Device, Index >;
   using ConstVectorView = Containers::VectorView< ConstReal, Device, Index >;

   ThreePartVector() = default;
   ThreePartVector( ThreePartVector& ) = default;

   void
   init( Index size_left, ConstVectorView view_mid, Index size_right )
   {
      left.setSize( size_left );
      middle.bind( view_mid );
      right.setSize( size_right );
   }

   void
   reset()
   {
      left.reset();
      middle.reset();
      right.reset();
   }

   IndexType
   getSize() const
   {
      return left.getSize() + middle.getSize() + right.getSize();
   }

   ThreePartVectorView< ConstReal, Device, Index >
   getConstView() const
   {
      return { left.getConstView(), middle, right.getConstView() };
   }

   //   __cuda_callable__
   //   Real& operator[]( Index i )
   //   {
   //      if( i < left.getSize() )
   //         return left[ i ];
   //      else if( i < left.getSize() + middle.getSize() )
   //         return middle[ i - left.getSize() ];
   //      else
   //         return right[ i - left.getSize() - middle.getSize() ];
   //   }

   __cuda_callable__
   const Real&
   operator[]( Index i ) const
   {
      if( i < left.getSize() )
         return left[ i ];
      else if( i < left.getSize() + middle.getSize() )
         return middle[ i - left.getSize() ];
      else
         return right[ i - left.getSize() - middle.getSize() ];
   }

   __cuda_callable__
   const Real*
   getPointer( Index i ) const
   {
      if( i < left.getSize() )
         return &left.getData()[ i ];
      else if( i < left.getSize() + middle.getSize() )
         return &middle.getData()[ i - left.getSize() ];
      else
         return &right.getData()[ i - left.getSize() - middle.getSize() ];
   }

   friend std::ostream&
   operator<<( std::ostream& str, const ThreePartVector& v )
   {
      str << "[\n\tleft: " << v.left << ",\n\tmiddle: " << v.middle << ",\n\tright: " << v.right << "\n]";
      return str;
   }

protected:
   Vector left, right;
   ConstVectorView middle;
};

}  // namespace __DistributedSpMV_impl
}  // namespace Matrices
}  // namespace noa::TNL
