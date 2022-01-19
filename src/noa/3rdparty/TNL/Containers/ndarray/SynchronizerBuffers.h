// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

// Implemented by: Jakub Klinkovsky

#pragma once

#include <TNL/Containers/NDArray.h>

namespace TNL {
namespace Containers {
namespace __ndarray_impl {

template< typename DistributedNDArray, std::size_t level >
struct SynchronizerBuffersLayer
{
   SynchronizerBuffersLayer& getDimBuffers( std::integral_constant< std::size_t, level > )
   {
      return *this;
   }

   using NDArrayType = NDArray< typename DistributedNDArray::ValueType,
                                typename DistributedNDArray::SizesHolderType,
                                typename DistributedNDArray::PermutationType,
                                typename DistributedNDArray::DeviceType >;
   NDArrayType left_send_buffer, left_recv_buffer, right_send_buffer, right_recv_buffer;
   typename NDArrayType::ViewType left_send_view, left_recv_view, right_send_view, right_recv_view;
   typename DistributedNDArray::LocalBeginsType left_send_offsets, left_recv_offsets, right_send_offsets, right_recv_offsets;

   int left_neighbor = -1;
   int right_neighbor = -1;

   void reset()
   {
      left_send_buffer.reset();
      left_recv_buffer.reset();
      right_send_buffer.reset();
      right_recv_buffer.reset();

      left_send_view.reset();
      left_recv_view.reset();
      right_send_view.reset();
      right_recv_view.reset();

      left_send_offsets = left_recv_offsets = right_send_offsets = right_recv_offsets = typename DistributedNDArray::LocalBeginsType{};

      left_neighbor = right_neighbor = -1;
   }
};

template< typename DistributedNDArray,
          typename LevelTag = std::integral_constant< std::size_t, DistributedNDArray::getDimension() > >
struct SynchronizerBuffersLayerHelper
{};

template< typename DistributedNDArray, std::size_t level >
struct SynchronizerBuffersLayerHelper< DistributedNDArray, std::integral_constant< std::size_t, level > >
: public SynchronizerBuffersLayerHelper< DistributedNDArray, std::integral_constant< std::size_t, level - 1 > >,
  public SynchronizerBuffersLayer< DistributedNDArray, level >
{
   using SynchronizerBuffersLayerHelper< DistributedNDArray, std::integral_constant< std::size_t, level - 1 > >::getDimBuffers;
   using SynchronizerBuffersLayer< DistributedNDArray, level >::getDimBuffers;
};

template< typename DistributedNDArray >
struct SynchronizerBuffersLayerHelper< DistributedNDArray, std::integral_constant< std::size_t, 0 > >
: public SynchronizerBuffersLayer< DistributedNDArray, 0 >
{
   using SynchronizerBuffersLayer< DistributedNDArray, 0 >::getDimBuffers;
};

template< typename DistributedNDArray >
struct SynchronizerBuffers
: public SynchronizerBuffersLayerHelper< DistributedNDArray >
{
   using SynchronizerBuffersLayerHelper< DistributedNDArray >::getDimBuffers;

   template< std::size_t level >
   auto& getDimBuffers()
   {
      return this->getDimBuffers( std::integral_constant< std::size_t, level >{} );
   }
};

} // namespace __ndarray_impl
} // namespace Containers
} // namespace TNL
