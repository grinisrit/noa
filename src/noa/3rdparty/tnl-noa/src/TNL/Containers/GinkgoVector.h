// Copyright (c) 2004-2023 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <noa/3rdparty/tnl-noa/src/TNL/Allocators/Default.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Containers/VectorView.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Devices/Host.h>

#include <ginkgo/ginkgo.hpp>

namespace noa::TNL {
namespace Containers {

/**
 * \defgroup Ginkgo  Wrappers for the Ginkgo library
 *
 * This group includes various wrapper classes and helper funct for data
 * structures and algorithms implemented in the [Ginkgo library][ginkgo]. See
 * the [examples][examples] for how these wrappers can be used.
 *
 * [ginkgo]: https://github.com/ginkgo-project/ginkgo
 * [examples]: https://gitlab.com/tnl-project/tnl/-/tree/main/src/Examples/Ginkgo
 */

/**
 * \brief Wrapper for representing vectors using Ginkgo data structure.
 *
 * \ingroup Ginkgo
 */
template< typename Value, typename Device = Devices::Host >
class GinkgoVector : public gko::matrix::Dense< Value >
{
public:
   using RealType = Value;
   using ValueType = Value;
   using DeviceType = Device;
   using IndexType = gko::size_type;
   using AllocatorType = typename Allocators::Default< Device >::template Allocator< ValueType >;

   using ViewType = Containers::VectorView< RealType, DeviceType, IndexType >;
   using ConstViewType = typename ViewType::ConstViewType;

   /**
    * \brief Bind or copy data into a new \e GinkgoVector object.
    *
    * \param exec Ginkgo executor to associate with this vector.
    * \param vector Input vector or vector view. See
    *       \ref TNL::Containers::Vector and \ref TNL::Containers::VectorView.
    * \param ownership controls whether or not we want Ginkgo to own the data.
    *       Normally, when we are wrapping a TNL Vector(View) created outside
    *       Ginkgo, we do not want ownership to be true. However, Ginkgo
    *       creates its own temporary vectors as part of its solvers, and these
    *       will be owned (and deleted) by Ginkgo.
    */
   template< typename Vector >
   GinkgoVector( std::shared_ptr< const gko::Executor > exec, Vector&& vector, bool ownership = false )
   : gko::matrix::Dense< ValueType >( exec,
                                      gko::dim< 2 >{ static_cast< std::size_t >( vector.getSize() ), 1 },
                                      gko::make_array_view( exec, vector.getSize(), vector.getData() ),
                                      1 ),  // stride
     wrapped_view( ViewType{ vector.getData(), static_cast< IndexType >( vector.getSize() ) } ), ownership( ownership )
   {
      static_assert( std::is_same< typename std::decay_t< Vector >::DeviceType, DeviceType >::value,
                     "the DeviceType passed to the constructor does not match the GinkgoVector type" );
   }

   virtual ~GinkgoVector()
   {
      if( ownership ) {
         AllocatorType allocator;
         allocator.deallocate( wrapped_view.getData(), wrapped_view.getSize() );
         wrapped_view.reset();
      }
   }

   template< typename Vector >
   static std::unique_ptr< GinkgoVector >
   create( std::shared_ptr< const gko::Executor > exec, Vector&& vector, bool ownership = false )
   {
      static_assert( std::is_same< typename std::decay_t< Vector >::DeviceType, DeviceType >::value,
                     "the DeviceType passed to the create function does not match the GinkgoVector type" );
      return std::unique_ptr< GinkgoVector >( new GinkgoVector( exec, std::forward< Vector >( vector ), ownership ) );
   }

   ViewType
   getView()
   {
      return wrapped_view.getView();
   }

   ConstViewType
   getConstView() const
   {
      return wrapped_view.getConstView();
   }

   /**
    * \brief Override base Dense class implementation for creating new vectors
    * with same executor and size as self
    */
   std::unique_ptr< gko::matrix::Dense< ValueType > >
   create_with_same_config() const override
   {
      AllocatorType allocator;
      ViewType view( allocator.allocate( this->get_size()[ 0 ] ), this->get_size()[ 0 ] );

      // If this function is called, Ginkgo is creating this object and should
      // control the memory, so ownership is set to true
      return GinkgoVector::create( this->get_executor(), view, true );
   }

   /**
    * \brief Override base Dense class implementation for creating new vectors
    * with same executor and type as self, but with a different size.
    *
    * This function will create "one large VectorWrapper" of size
    * size[0] * size[1], since MFEM Vectors only have one dimension.
    */
   std::unique_ptr< gko::matrix::Dense< ValueType > >
   create_with_type_of_impl( std::shared_ptr< const gko::Executor > exec,
                             const gko::dim< 2 >& size,
                             gko::size_type stride ) const override
   {
      // Only stride of 1 is allowed for VectorWrapper type
      if( stride > 1 )
         throw gko::Error( __FILE__, __LINE__, "GinkgoVector cannot be created with stride > 1" );

      // Compute total size of new Vector
      const gko::size_type total_size = size[ 0 ] * size[ 1 ];

      AllocatorType allocator;
      ViewType view( allocator.allocate( total_size ), total_size );

      // If this function is called, Ginkgo is creating this object and should
      // control the memory, so ownership is set to true
      return GinkgoVector::create( this->get_executor(), view, true );
   }

   /**
    * \brief Override base Dense class implementation for creating new
    * sub-vectors from a larger vector.
    */
   std::unique_ptr< gko::matrix::Dense< ValueType > >
   create_submatrix_impl( const gko::span& rows, const gko::span& columns, const gko::size_type stride ) override
   {
      const gko::size_type num_rows = rows.end - rows.begin;
      const gko::size_type num_cols = columns.end - columns.begin;
      // Data in the Dense matrix will be stored in row-major format.
      // Check that we only have one column, and that the stride = 1
      // (only allowed value for VectorWrappers).
      if( num_cols > 1 || stride > 1 )
         throw gko::BadDimension( __FILE__,
                                  __LINE__,
                                  __func__,
                                  "new_submatrix",
                                  num_rows,
                                  num_cols,
                                  "GinkgoVector submatrix must have one column and stride = 1" );

      // Create a view for the sub-vector
      ViewType view = wrapped_view.getView( rows.begin, rows.end );

      // Ownership is set to false as the created object represents a view into
      // the parent vector
      return GinkgoVector::create( this->get_executor(), view, false );
   }

private:
   ViewType wrapped_view;
   bool ownership;
};

}  // namespace Containers
}  // namespace noa::TNL
