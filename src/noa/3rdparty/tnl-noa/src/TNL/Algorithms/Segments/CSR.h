// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <type_traits>

#include <noa/3rdparty/tnl-noa/src/TNL/Containers/Vector.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Algorithms/Segments/CSRView.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Algorithms/Segments/SegmentView.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Algorithms/Segments/ElementsOrganization.h>

namespace noa::TNL {
namespace Algorithms {
namespace Segments {

/**
 * \brief Data structure for CSR segments format.
 *
 * See \ref TNL::Algorithms::Segments for more details about segments.
 *
 * \tparam Device is type of device where the segments will be operating.
 * \tparam Index is type for indexing of the elements managed by the segments.
 * \tparam Kernel is type of kernel used for parallel operations with segments.
 *    It can be any of the following:
 *    \ref TNL::Containers::Segments::Kernels::CSRAdaptiveKernel,
 *    \ref TNL::Containers::Segments::Kernels::CSRHybridKernel,
 *    \ref TNL::Containers::Segments::Kernels::CSRScalarKernel,
 *    \ref TNL::Containers::Segments::Kernels::CSRVectorKernel
 *
 * \tparam IndexAllocator is allocator for supporting index containers.
 */
template< typename Device,
          typename Index,
          typename Kernel = CSRScalarKernel< Index, Device >,
          typename IndexAllocator = typename Allocators::Default< Device >::template Allocator< Index > >
class CSR
{
public:
   /**
    * \brief The device where the segments are operating.
    */
   using DeviceType = Device;

   /**
    * \brief The type used for indexing of segments elements.
    */
   using IndexType = std::remove_const_t< Index >;

   /**
    * \brief Type of kernel used for reduction operations.
    */
   using KernelType = Kernel;

   /**
    * \brief Type of container storing offsets of particular rows.
    */
   using OffsetsContainer = Containers::Vector< Index, DeviceType, IndexType, IndexAllocator >;

   /**
    * \brief Templated view type.
    *
    * \tparam Device_ is alternative device type for the view.
    * \tparam Index_ is alternative index type for the view.
    */
   template< typename Device_, typename Index_ >
   using ViewTemplate = CSRView< Device_, Index_, KernelType >;

   /**
    * \brief Type of segments view.1
    */
   using ViewType = CSRView< Device, Index, KernelType >;

   /**
    * \brief Type of constant segments view.
    */
   using ConstViewType = CSRView< Device, std::add_const_t< IndexType >, KernelType >;

   /**
    * \brief Accessor type fro one particular segment.
    */
   using SegmentViewType = SegmentView< IndexType, RowMajorOrder >;

   /**
    * \brief This functions says that CSR format is always organised in row-major order.
    */
   static constexpr ElementsOrganization
   getOrganization()
   {
      return RowMajorOrder;
   }

   /**
    * \brief This function says that CSR format does not use padding elements.
    */
   static constexpr bool
   havePadding()
   {
      return false;
   };

   /**
    * \brief Construct with no parameters to create empty segments.
    */
   CSR() = default;

   /**
    * \brief Construct with segments sizes.
    *
    * The number of segments is given by the size of \e segmentsSizes. Particular elements
    * of this container define sizes of particular segments.
    *
    * \tparam SizesContainer is a type of container for segments sizes.  It can be \ref TNL::Containers::Array or
    *  \ref TNL::Containers::Vector for example.
    * \param sizes is an instance of the container with the segments sizes.
    *
    * See the following example:
    *
    * \includelineno Algorithms/Segments/SegmentsExample_CSR_constructor_1.cpp
    *
    * The result looks as follows:
    *
    * \include SegmentsExample_CSR_constructor_1.out
    */
   template< typename SizesContainer >
   CSR( const SizesContainer& segmentsSizes );

   /**
    * \brief Construct with segments sizes in initializer list..
    *
    * The number of segments is given by the size of \e segmentsSizes. Particular elements
    * of this initializer list define sizes of particular segments.
    *
    * \tparam ListIndex is a type of indexes of the initializer list.
    * \param sizes is an instance of the container with the segments sizes.
    *
    * See the following example:
    *
    * \includelineno Algorithms/Segments/SegmentsExample_CSR_constructor_2.cpp
    *
    * The result looks as follows:
    *
    * \include SegmentsExample_CSR_constructor_2.out
    */
   template< typename ListIndex >
   CSR( const std::initializer_list< ListIndex >& segmentsSizes );

   /**
    * \brief Copy constructor.
    *
    * \param segments are the source segments.
    */
   CSR( const CSR& segments ) = default;

   /**
    * \brief Move constructor.
    *
    * \param segments  are the source segments.
    */
   CSR( CSR&& segments ) noexcept = default;

   /**
    * \brief Returns string with serialization type.
    *
    * The string has a form `Algorithms::Segments::CSR< IndexType,  [any_device], [any_kernel], [any_allocator] >`.
    *
    * \return String with the serialization type.
    *
    * \par Example
    * \include Algorithms/Segments/SegmentsExample_CSR_getSerializationType.cpp
    * \par Output
    * \include SegmentsExample_CSR_getSerializationType.out
    */
   static std::string
   getSerializationType();

   /**
    * \brief Returns string with segments type.
    *
    * The string has a form `CSR< KernelType >`.
    *
    * \return \ref String with the segments type.
    *
    * \par Example
    * \include Algorithms/Segments/SegmentsExample_CSR_getSegmentsType.cpp
    * \par Output
    * \include SegmentsExample_CSR_getSegmentsType.out
    */
   static String
   getSegmentsType();

   /**
    * \brief Set sizes of particular segments.
    *
    * \tparam SizesContainer is a container with segments sizes. It can be \ref TNL::Containers::Array or
    *  \ref TNL::Containers::Vector for example.
    *
    * \param segmentsSizes is an instance of the container with segments sizes.
    */
   template< typename SizesContainer >
   void
   setSegmentsSizes( const SizesContainer& segmentsSizes );

   /**
    * \brief Reset the segments to empty states.
    *
    * It means that there is no segment in the CSR segments.
    */
   void
   reset();

   /**
    * \brief Getter of a view object.
    *
    * \return View for this instance of CSR segments which can by used for example in
    *  lambda functions running in GPU kernels.
    */
   ViewType
   getView();

   /**
    * \brief Getter of a view object for constants instances.
    *
    * \return View for this instance of CSR segments which can by used for example in
    *  lambda functions running in GPU kernels.
    */
   const ConstViewType
   getConstView() const;

   /**
    * \brief Getter of number of segments.
    *
    * \return number of segments within this object.
    */
   __cuda_callable__
   IndexType
   getSegmentsCount() const;

   /**
    * \brief Returns size of particular segment.
    *
    * \return size of the segment number \e segmentIdx.
    */
   __cuda_callable__
   IndexType
   getSegmentSize( IndexType segmentIdx ) const;

   /***
    * \brief Returns number of elements managed by all segments.
    *
    * \return number of elements managed by all segments.
    */
   __cuda_callable__
   IndexType
   getSize() const;

   /**
    * \brief Returns number of elements that needs to be allocated by a container connected to this segments.
    *
    * \return size of container connected to this segments.
    */
   __cuda_callable__
   IndexType
   getStorageSize() const;

   /**
    * \brief Computes the global index of an element managed by the segments.
    *
    * The global index serves as a refernce on the element in its container.
    *
    * \param segmentIdx is index of a segment with the element.
    * \param localIdx is tha local index of the element within the segment.
    * \return global index of the element.
    */
   __cuda_callable__
   IndexType
   getGlobalIndex( Index segmentIdx, Index localIdx ) const;

   /**
    * \brief Returns segment view (i.e. segment accessor) of segment with given index.
    *
    * \param segmentIdx is index of the request segment.
    * \return segment view of given segment.
    *
    * \par Example
    * \include Algorithms/Segments/SegmentsExample_CSR_getSegmentView.cpp
    * \par Output
    * \include SegmentsExample_CSR_getSegmentView.out
    */
   __cuda_callable__
   SegmentViewType
   getSegmentView( IndexType segmentIdx ) const;

   /**
    * \brief Returns reference on constant vector with row offsets used in the CSR format.
    *
    * \return reference on constant vector with row offsets used in the CSR format.
    */
   const OffsetsContainer&
   getOffsets() const;

   /**
    * \brief Returns reference on vector with row offsets used in the CSR format.
    *
    * \return reference on vector with row offsets used in the CSR format.
    */
   OffsetsContainer&
   getOffsets();

   /**
    * \brief Iterate over all elements of given segments in parallel and call given lambda function.
    *
    * \tparam Function is a type of the lambda function to be performed on each element.
    * \param begin defines begining of an interval [ \e begin, \e end ) of segments on
    *    elements of which we want to apply the lambda function.
    * \param end defines end of an interval [ \e begin, \e end ) of segments on
    *    elements of which we want to apply the lambda function.
    * \param function is the lambda function to be applied on the elements of the segments.
    *
    * Declaration of the lambda function \e function is supposed to be
    *
    * ```
    * auto f = [=] __cuda_callable__ ( IndexType segmentIdx, IndexType localIdx, IndexType globalIdx ) {...}
    * ```
    * where \e segmentIdx is index of segment where given element belong to, \e localIdx is rank of the element
    * within the segment and \e globalIdx is index of the element within the related container.
    *
    * \par Example
    * \include Algorithms/Segments/SegmentsExample_CSR_forElements.cpp
    * \par Output
    * \include SegmentsExample_CSR_forElements.out
    */
   template< typename Function >
   void
   forElements( IndexType begin, IndexType end, Function&& function ) const;

   /**
    * \brief Call \ref TNL::Algorithms::Segments::CSR::forElements for all elements of the segments.
    *
    * See \ref TNL::Algorithms::Segments::CSR::forElements for more details.
    */
   template< typename Function >
   void
   forAllElements( Function&& function ) const;

   /**
    * \brief Iterate over all segments in parallel and call given lambda function.
    *
    * \tparam Function is a type of the lambda function to be performed on each segment.
    * \param begin defines begining of an interval [ \e begin, \e end ) of segments on
    *    elements of which we want to apply the lambda function.
    * \param end defines end of an interval [ \e begin, \e end ) of segments on
    *    elements of which we want to apply the lambda function.
    * \param function is the lambda function to be applied on the elements of the segments.
    *
    *  Declaration of the lambda function \e function is supposed to be
    *
    * ```
    * auto f = [=] __cuda_callable__ ( const SegmentView& segment ) {...}
    * ```
    * where \e segment represents given segment (see \ref TNL::Algorithms::Segments::SegmentView).
    * Its type is given by \ref SegmentViewType.
    *
    * \par Example
    * \include Algorithms/Segments/SegmentsExample_CSR_forSegments.cpp
    * \par Output
    * \include SegmentsExample_CSR_forSegments.out
    */
   template< typename Function >
   void
   forSegments( IndexType begin, IndexType end, Function&& function ) const;

   /**
    * \brief Call \ref TNL::Algorithms::Segments::CSR::forSegments for all segments.
    *
    * See \ref TNL::Algorithms::Segments::CSR::forSegments for more details.
    */
   template< typename Function >
   void
   forAllSegments( Function&& function ) const;

   /**
    * \brief Call \ref TNL::Algorithms::Segments::CSR::forSegments sequentially for particular segments.
    *
    * With this method, the given segments are processed sequentially one-by-one. This is usefull for example
    * for printing of segments based data structures or for debugging reasons.
    *
    * \param begin defines begining of an interval [ \e begin, \e end ) of segments on
    *    elements of which we want to apply the lambda function.
    * \param end defines end of an interval [ \e begin, \e end ) of segments on
    *    elements of which we want to apply the lambda function.
    * \param function is the lambda function to be applied on the elements of the segments.
    *
    * See \ref TNL::Algorithms::Segments::CSR::forSegments for more details.
    *
    * \par Example
    * \include Algorithms/Segments/SegmentsExample_CSR_sequentialForSegments.cpp
    * \par Output
    * \include SegmentsExample_CSR_sequentialForSegments.out
    */
   template< typename Function >
   void
   sequentialForSegments( IndexType begin, IndexType end, Function&& function ) const;

   /**
    * \brief Call \ref TNL::Algorithms::Segments::CSR::sequentialForSegments for all segments.
    *
    * See \ref TNL::Algorithms::Segments::CSR::sequentialForSegments for more details.
    */
   template< typename Function >
   void
   sequentialForAllSegments( Function&& f ) const;

   /**
    * \brief Compute reduction in each segment.
    *
    * \tparam Fetch is type of lambda function for data fetching.
    * \tparam Reduce is a reduction operation.
    * \tparam Keep is lambda function for storing results from particular segments.
    *
    * \param begin defines begining of an interval [ \e begin, \e end ) of segments in
    *    which we want to perform the reduction.
    * \param end defines and of an interval [ \e begin, \e end ) of segments in
    *    which we want to perform the reduction.
    * \param fetch is a lambda function for fetching of data. It is suppos have one of the
    *  following forms:
    * 1. Full form
    *  ```
    *  auto fetch = [=] __cuda_callable__ ( IndexType segmentIdx, IndexType localIdx, IndexType globalIdx, bool& compute ) { ...
    * }
    *  ```
    * 2. Brief form
    * ```
    * auto fetch = [=] __cuda_callable__ ( IndexType globalIdx, bool& compute ) { ... }
    * ```
    * where for both variants \e segmentIdx is segment index, \e localIdx is a rank of element in the segment, \e globalIdx is
    * index of the element in related container and \e compute is a boolean variable which serves for stopping the reduction if
    * it is set to \e false. It is however, only a hint and the real behaviour depends on type of kernel used ofr the redcution.
    * Some kernels are optimized so that they can be significantly faster with the brief variant of the \e fetch lambda
    * function. \param reduce is a lambda function representing the reduction opeartion. It is supposed to be defined as:
    *
    * ```
    * auto reduce = [=] __cuda_callable__ ( const Value& a, const Value& b ) -> Value { ... }
    * ```
    *
    * where \e a and \e b are values to be reduced and the lambda function returns result of the reduction.
    * \param keep is a lambda function for saving results from particular segments. It is supposed to be defined as:
    *
    * ```
    * auto keep = [=] __cuda_callable__ ( IndexType segmentIdx, const Value& value ) { ... }
    * ```
    *
    * where \e segmentIdx is an index of the segment and \e value is the result of the reduction in given segment to be stored.
    *
    * \par Example
    * \include Algorithms/Segments/SegmentsExample_CSR_reduceSegments.cpp
    * \par Output
    * \include SegmentsExample_CSR_reduceSegments.out
    */
   template< typename Fetch, typename Reduce, typename Keep, typename Value >
   void
   reduceSegments( IndexType begin, IndexType end, Fetch& fetch, const Reduce& reduce, Keep& keep, const Value& zero ) const;

   /**
    * \brief Call \ref TNL::Algorithms::Segments::CSR::reduceSegments for all segments.
    *
    * See \ref TNL::Algorithms::Segments::CSR::reduceSegments for more details.
    */
   template< typename Fetch, typename Reduce, typename Keep, typename Value >
   void
   reduceAllSegments( Fetch& fetch, const Reduce& reduce, Keep& keep, const Value& zero ) const;

   /**
    * \brief Assignment operator.
    *
    * It makes a deep copy of the source segments.
    *
    * \param source are the CSR segments to be assigned.
    * \return reference to this instance.
    */
   CSR&
   operator=( const CSR& source ) = default;

   /**
    * \brief Assignment operator with CSR segments with different template parameters.
    *
    * It makes a deep copy of the source segments.
    *
    * \tparam Device_ is device type of the source segments.
    * \tparam Index_ is the index type of the source segments.
    * \tparam Kernel_ is the kernel type of the source segments.
    * \tparam IndexAllocator_ is the index allocator of the source segments.
    * \param source is the source segments object.
    * \return reference to this instance.
    */
   template< typename Device_, typename Index_, typename Kernel_, typename IndexAllocator_ >
   CSR&
   operator=( const CSR< Device_, Index_, Kernel_, IndexAllocator_ >& source );

   /**
    * \brief Method for saving the segments to a file in a binary form.
    *
    * \param file is the target file.
    */
   void
   save( File& file ) const;

   /**
    * \brief Method for loading the segments from a file in a binary form.
    *
    * \param file is the source file.
    */
   void
   load( File& file );

   /**
    * \brief Return simple proxy object for insertion to output stream.
    *
    * The proxy object serves for wrapping segments with lambda function mediating access to data managed by the segments.
    *
    * \tparam Fetch is type of lambda function for data access.
    * \param fetch is an instance of lambda function for data access. It is supposed to be defined as
    *
    * ```
    * auto fetch = [=] __cuda_callable__ ( IndexType globalIdx ) -> ValueType { return data_view[ globalIdx ]; };
    * ```
    * \return Proxy object for insertion to output stream.
    *
    * \par Example
    * \include Algorithms/Segments/SegmentsPrintingExample-2.cpp
    * \par Output
    * \include SegmentsPrintingExample-2.out
    */
   template< typename Fetch >
   SegmentsPrinter< CSR, Fetch >
   print( Fetch&& fetch ) const;

   KernelType&
   getKernel()
   {
      return kernel;
   }

   const KernelType&
   getKernel() const
   {
      return kernel;
   }

protected:
   OffsetsContainer offsets;

   KernelType kernel;
};

/**
 * \brief Insertion operator of CSR segments to output stream.
 *
 * \tparam Device is the device type of the source segments.
 * \tparam Index is the index type of the source segments.
 * \tparam Kernel is kernel type of the source segments.
 * \tparam IndexAllocator is the index allocator of the source segments.
 * \param str is the output stream.
 * \param segments are the source segments.
 * \return reference to the output stream.
 */
template< typename Device, typename Index, typename Kernel, typename IndexAllocator >
std::ostream&
operator<<( std::ostream& str, const CSR< Device, Index, Kernel, IndexAllocator >& segments )
{
   return printSegments( segments, str );
}

template< typename Device,
          typename Index,
          typename IndexAllocator = typename Allocators::Default< Device >::template Allocator< Index > >
using CSRScalar = CSR< Device, Index, CSRScalarKernel< Index, Device >, IndexAllocator >;

template< typename Device,
          typename Index,
          typename IndexAllocator = typename Allocators::Default< Device >::template Allocator< Index > >
using CSRVector = CSR< Device, Index, CSRVectorKernel< Index, Device >, IndexAllocator >;

template< typename Device,
          typename Index,
          typename IndexAllocator = typename Allocators::Default< Device >::template Allocator< Index > >
using CSRHybrid = CSR< Device, Index, CSRHybridKernel< Index, Device >, IndexAllocator >;

template< typename Device,
          typename Index,
          typename IndexAllocator = typename Allocators::Default< Device >::template Allocator< Index > >
using CSRLight = CSR< Device, Index, CSRLightKernel< Index, Device >, IndexAllocator >;

template< typename Device,
          typename Index,
          typename IndexAllocator = typename Allocators::Default< Device >::template Allocator< Index > >
using CSRAdaptive = CSR< Device, Index, CSRAdaptiveKernel< Index, Device >, IndexAllocator >;

template< typename Device,
          typename Index,
          typename IndexAllocator = typename Allocators::Default< Device >::template Allocator< Index > >
using CSRDefault = CSRScalar< Device, Index, IndexAllocator >;

}  // namespace Segments
}  // namespace Algorithms
}  // namespace noa::TNL

#include <noa/3rdparty/tnl-noa/src/TNL/Algorithms/Segments/CSR.hpp>
