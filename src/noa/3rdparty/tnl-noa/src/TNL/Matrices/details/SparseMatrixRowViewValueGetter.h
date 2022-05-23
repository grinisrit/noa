

// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

namespace noa::TNL {
namespace Matrices {
namespace details {

template< typename SegmentView,
          typename ValuesView,
          typename ColumnsIndexesView,
          typename Real = std::remove_const_t< typename ValuesView::RealType >,
          bool isBinary_ = std::is_same< std::remove_const_t< typename ValuesView::RealType >, bool >::value >
struct SparseMatrixRowViewValueGetter
{};

template< typename SegmentView, typename ValuesView, typename ColumnsIndexesView, typename Real >
struct SparseMatrixRowViewValueGetter< SegmentView, ValuesView, ColumnsIndexesView, Real, true >
{
   using RealType = typename ValuesView::RealType;

   using IndexType = typename ColumnsIndexesView::IndexType;

   using ResultType = bool;

   using ConstResultType = bool;

   __cuda_callable__
   static bool
   getValue( const IndexType& globalIdx,
             const ValuesView& values,
             const ColumnsIndexesView& columnIndexes,
             const IndexType& paddingIndex )
   {
      return columnIndexes[ globalIdx ] != paddingIndex;
   };
};

template< typename SegmentView, typename ValuesView, typename ColumnsIndexesView, typename Real >
struct SparseMatrixRowViewValueGetter< SegmentView, ValuesView, ColumnsIndexesView, Real, false >
{
   using RealType = typename ValuesView::RealType;

   using IndexType = typename ColumnsIndexesView::IndexType;

   using ResultType = RealType&;

   using ConstResultType = const RealType&;

   __cuda_callable__
   static const RealType&
   getValue( const IndexType& globalIdx,
             const ValuesView& values,
             const ColumnsIndexesView& columnIndexes,
             const IndexType& paddingIndex )
   {
      return values[ globalIdx ];
   };

   __cuda_callable__
   static RealType&
   getValue( const IndexType& globalIdx, ValuesView& values, ColumnsIndexesView& columnIndexes, const IndexType& paddingIndex )
   {
      return values[ globalIdx ];
   };
};

}  // namespace details
}  // namespace Matrices
}  // namespace noa::TNL
