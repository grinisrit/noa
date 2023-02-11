// Copyright (c) 2004-2023 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <noa/3rdparty/tnl-noa/src/TNL/String.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Matrices/DenseMatrix.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Matrices/DenseMatrixView.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Matrices/SparseMatrix.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Matrices/SparseMatrixView.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Matrices/Sandbox/SparseSandboxMatrix.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Algorithms/Segments/CSRView.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Algorithms/Segments/EllpackView.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Algorithms/Segments/SlicedEllpackView.h>

namespace noa::TNL {
/**
 * \brief Namespace for matrix formats.
 */
namespace Matrices {

template< typename Matrix >
struct MatrixInfo
{};

/// This is to prevent from appearing in Doxygen documentation.
/// \cond HIDDEN_CLASS
template< typename Real, typename Device, typename Index, ElementsOrganization Organization >
struct MatrixInfo< DenseMatrixView< Real, Device, Index, Organization > >
{
   static String
   getDensity()
   {
      return "dense";
   }

   static String
   getFormat()
   {
      return "Dense";
   }
};

template< typename Real, typename Device, typename Index, ElementsOrganization Organization, typename RealAllocator >
struct MatrixInfo< DenseMatrix< Real, Device, Index, Organization, RealAllocator > >
: public MatrixInfo< typename DenseMatrix< Real, Device, Index, Organization, RealAllocator >::ViewType >
{};

template< typename Real,
          typename Device,
          typename Index,
          typename MatrixType,
          template< typename Device_, typename Index_ >
          class SegmentsView >
struct MatrixInfo< SparseMatrixView< Real, Device, Index, MatrixType, SegmentsView > >
{
   static String
   getDensity()
   {
      return "sparse";
   }

   static String
   getFormat()
   {
      String prefix;
      if( MatrixType::isSymmetric() ) {
         if( std::is_same< Real, bool >::value )
            prefix = "Symmetric Binary ";
         else
            prefix = "Symmetric ";
      }
      else if( std::is_same< Real, bool >::value )
         prefix = "Binary ";
      return prefix + SegmentsView< Device, Index >::getSegmentsType();
   }
};

template< typename Real,
          typename Device,
          typename Index,
          typename MatrixType,
          template< typename Device_, typename Index_, typename IndexAllocator_ >
          class Segments,
          typename RealAllocator,
          typename IndexAllocator >
struct MatrixInfo< SparseMatrix< Real, Device, Index, MatrixType, Segments, RealAllocator, IndexAllocator > >
: public MatrixInfo<
     typename SparseMatrix< Real, Device, Index, MatrixType, Segments, RealAllocator, IndexAllocator >::ViewType >
{};

template< typename Real, typename Device, typename Index, typename MatrixType >
struct MatrixInfo< Sandbox::SparseSandboxMatrixView< Real, Device, Index, MatrixType > >
{
   static String
   getDensity()
   {
      return "sparse";
   }

   static String
   getFormat()
   {
      if( MatrixType::isSymmetric() )
         return "Symmetric Sandbox";
      else
         return "Sandbox";
   }
};

template< typename Real, typename Device, typename Index, typename MatrixType, typename RealAllocator, typename IndexAllocator >
struct MatrixInfo< Sandbox::SparseSandboxMatrix< Real, Device, Index, MatrixType, RealAllocator, IndexAllocator > >
: public MatrixInfo<
     typename Sandbox::SparseSandboxMatrix< Real, Device, Index, MatrixType, RealAllocator, IndexAllocator >::ViewType >
{};

/// \endcond
}  // namespace Matrices
}  // namespace noa::TNL
