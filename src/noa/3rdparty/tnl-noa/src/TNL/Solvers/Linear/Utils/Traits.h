// Copyright (c) 2004-2023 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

// Implemented by: Jakub Klinkovský

#pragma once

#include <noa/3rdparty/tnl-noa/src/TNL/MPI/Comm.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Containers/Vector.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Containers/VectorView.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Containers/DistributedVector.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Containers/DistributedVectorView.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Matrices/DistributedMatrix.h>

namespace noa::TNL {
namespace Solvers {
namespace Linear {

template< typename Matrix >
struct Traits
{
   using VectorType = Containers::Vector< typename Matrix::RealType, typename Matrix::DeviceType, typename Matrix::IndexType >;
   using VectorViewType =
      Containers::VectorView< typename Matrix::RealType, typename Matrix::DeviceType, typename Matrix::IndexType >;
   using ConstVectorViewType = Containers::
      VectorView< std::add_const_t< typename Matrix::RealType >, typename Matrix::DeviceType, typename Matrix::IndexType >;

   // compatibility aliases
   using LocalVectorType = VectorType;
   using LocalViewType = VectorViewType;
   using ConstLocalViewType = ConstVectorViewType;

   // compatibility wrappers for some DistributedMatrix methods
   static const Matrix&
   getLocalMatrix( const Matrix& m )
   {
      return m;
   }

   static ConstLocalViewType
   getConstLocalView( ConstVectorViewType v )
   {
      return v;
   }

   static LocalViewType
   getLocalView( VectorViewType v )
   {
      return v;
   }

   static ConstLocalViewType
   getConstLocalViewWithGhosts( ConstVectorViewType v )
   {
      return v;
   }

   static LocalViewType
   getLocalViewWithGhosts( VectorViewType v )
   {
      return v;
   }

   static MPI::Comm
   getCommunicator( const Matrix& m )
   {
      return MPI_COMM_WORLD;
   }

   static void
   startSynchronization( VectorViewType v )
   {}

   static void
   waitForSynchronization( VectorViewType v )
   {}
};

template< typename Matrix >
struct Traits< Matrices::DistributedMatrix< Matrix > >
{
   using VectorType =
      Containers::DistributedVector< typename Matrix::RealType, typename Matrix::DeviceType, typename Matrix::IndexType >;
   using VectorViewType =
      Containers::DistributedVectorView< typename Matrix::RealType, typename Matrix::DeviceType, typename Matrix::IndexType >;
   using ConstVectorViewType = Containers::DistributedVectorView< std::add_const_t< typename Matrix::RealType >,
                                                                  typename Matrix::DeviceType,
                                                                  typename Matrix::IndexType >;

   using LocalVectorType =
      Containers::Vector< typename Matrix::RealType, typename Matrix::DeviceType, typename Matrix::IndexType >;
   using LocalViewType =
      Containers::VectorView< typename Matrix::RealType, typename Matrix::DeviceType, typename Matrix::IndexType >;
   using ConstLocalViewType = Containers::
      VectorView< std::add_const_t< typename Matrix::RealType >, typename Matrix::DeviceType, typename Matrix::IndexType >;

   // compatibility wrappers for some DistributedMatrix methods
   static const Matrix&
   getLocalMatrix( const Matrices::DistributedMatrix< Matrix >& m )
   {
      return m.getLocalMatrix();
   }

   static ConstLocalViewType
   getConstLocalView( ConstVectorViewType v )
   {
      return v.getConstLocalView();
   }

   static LocalViewType
   getLocalView( VectorViewType v )
   {
      return v.getLocalView();
   }

   static ConstLocalViewType
   getConstLocalViewWithGhosts( ConstVectorViewType v )
   {
      return v.getConstLocalViewWithGhosts();
   }

   static LocalViewType
   getLocalViewWithGhosts( VectorViewType v )
   {
      return v.getLocalViewWithGhosts();
   }

   static const MPI::Comm&
   getCommunicator( const Matrices::DistributedMatrix< Matrix >& m )
   {
      return m.getCommunicator();
   }

   static void
   startSynchronization( VectorViewType v )
   {
      v.startSynchronization();
   }

   static void
   waitForSynchronization( VectorViewType v )
   {
      v.waitForSynchronization();
   }
};

}  // namespace Linear
}  // namespace Solvers
}  // namespace noa::TNL
