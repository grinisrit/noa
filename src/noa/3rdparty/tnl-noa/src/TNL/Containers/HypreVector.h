// Copyright (c) 2004-2023 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

// Implemented by: Jakub Klinkovský

#pragma once

#ifdef HAVE_HYPRE

   #include <noa/3rdparty/tnl-noa/src/TNL/Hypre.h>

   #include <noa/3rdparty/tnl-noa/src/TNL/Containers/Vector.h>

namespace noa::TNL {
namespace Containers {

/**
 * \brief Wrapper for Hypre's sequential vector.
 *
 * Links to upstream sources:
 * - https://github.com/hypre-space/hypre/blob/master/src/seq_mv/vector.h
 * - https://github.com/hypre-space/hypre/blob/master/src/seq_mv/vector.c
 * - https://github.com/hypre-space/hypre/blob/master/src/seq_mv/seq_mv.h (catch-all interface)
 *
 * \ingroup Hypre
 */
class HypreVector
{
public:
   using RealType = HYPRE_Real;
   using ValueType = RealType;
   using DeviceType = HYPRE_Device;
   using IndexType = HYPRE_Int;

   using VectorType = Containers::Vector< RealType, DeviceType, IndexType >;
   using ViewType = typename VectorType::ViewType;
   using ConstViewType = typename VectorType::ConstViewType;

   HypreVector() = default;

   // TODO: behavior should depend on "owns_data" (shallow vs deep copy)
   HypreVector( const HypreVector& other ) = delete;

   HypreVector( HypreVector&& other ) noexcept : v( other.v ), owns_handle( other.owns_handle )
   {
      other.v = nullptr;
   }

   // TODO should do a deep copy
   HypreVector&
   operator=( const HypreVector& other ) = delete;

   HypreVector&
   operator=( HypreVector&& other ) noexcept
   {
      v = other.v;
      other.v = nullptr;
      owns_handle = other.owns_handle;
      return *this;
   }

   HypreVector( RealType* data, IndexType size )
   {
      bind( data, size );
   }

   HypreVector( ViewType view )
   {
      bind( view );
   }

   /**
    * \brief Convert Hypre's format to HypreVector
    *
    * \param handle is the Hypre vector handle.
    * \param take_ownership indicates if the vector should take ownership of
    * the handle, i.e. whether to call \e hypre_VectorDestroy when it does
    * not need it anymore.
    */
   explicit HypreVector( hypre_Vector* handle, bool take_ownership = true )
   {
      bind( handle, take_ownership );
   }

   operator const hypre_Vector*() const noexcept
   {
      return v;
   }

   operator hypre_Vector*() noexcept
   {
      return v;
   }

   // HYPRE_Vector is "equivalent" to pointer to hypre_Vector, but requires
   // ugly C-style cast on the pointer (which is done even in Hypre itself)
   // https://github.com/hypre-space/hypre/blob/master/src/seq_mv/HYPRE_vector.c
   operator HYPRE_Vector() const noexcept
   {
      return (HYPRE_Vector) v;
   }

   ~HypreVector()
   {
      reset();
   }

   const RealType*
   getData() const noexcept
   {
      if( v == nullptr )
         return nullptr;
      return hypre_VectorData( v );
   }

   RealType*
   getData() noexcept
   {
      if( v == nullptr )
         return nullptr;
      return hypre_VectorData( v );
   }

   IndexType
   getSize() const
   {
      if( v == nullptr )
         return 0;
      return hypre_VectorSize( v );
   }

   ConstViewType
   getConstView() const
   {
      if( v == nullptr )
         return {};
      return { getData(), getSize() };
   }

   ViewType
   getView()
   {
      if( v == nullptr )
         return {};
      return { getData(), getSize() };
   }

   /**
    * \brief Drop previously set data (deallocate if the vector was the owner)
    * and bind to the given data (i.e., the vector does not become the owner).
    */
   void
   bind( RealType* data, IndexType size )
   {
      // drop/deallocate the current data
      reset();

      // create handle for the vector
      v = hypre_SeqVectorCreate( 0 );
      hypre_VectorMemoryLocation( v ) = getHypreMemoryLocation();

      // set view data
      hypre_VectorOwnsData( v ) = 0;
      hypre_VectorData( v ) = data;
      hypre_VectorSize( v ) = size;
   }

   void
   bind( ViewType view )
   {
      bind( view.getData(), view.getSize() );
   }

   void
   bind( VectorType& vector )
   {
      bind( vector.getView() );
   }

   void
   bind( HypreVector& vector )
   {
      bind( vector.getData(), vector.getSize() );
   }

   /**
    * \brief Convert Hypre's format to HypreSeqVector
    *
    * \param handle is the Hypre vector handle.
    * \param take_ownership indicates if the vector should take ownership of
    * the handle, i.e. whether to call \e hypre_VectorDestroy when it does
    * not need it anymore.
    */
   void
   bind( hypre_Vector* handle, bool take_ownership = true )
   {
      // drop/deallocate the current data
      reset();

      // set the handle and ownership flag
      v = handle;
      owns_handle = take_ownership;
   }

   //! \brief Reset the vector to empty state.
   void
   reset()
   {
      if( owns_handle && v != nullptr ) {
         hypre_SeqVectorDestroy( v );
         v = nullptr;
      }
      else
         v = nullptr;
      owns_handle = true;
   }

   /**
    * \brief Set the new vector size.
    *
    * - if the vector previously owned data, they are deallocated
    * - new size is set
    * - the vector is initialized with \e hypre_SeqVectorInitialize
    *   (i.e., data are allocated)
    */
   void
   setSize( IndexType size )
   {
      hypre_SeqVectorDestroy( v );
      v = hypre_SeqVectorCreate( size );
      hypre_VectorMemoryLocation( v ) = getHypreMemoryLocation();
      hypre_SeqVectorInitialize( v );
   }

   //! \brief Equivalent to \ref setSize.
   void
   resize( IndexType size )
   {
      setSize( size );
   }

   //! \brief Equivalent to \ref setSize followed by \ref setValue.
   void
   resize( IndexType size, RealType value )
   {
      setSize( size );
      setValue( value );
   }

   //! \brief Set all elements of the vector to \e value.
   void
   setValue( RealType value )
   {
      hypre_SeqVectorSetConstantValues( v, value );
   }

protected:
   hypre_Vector* v = nullptr;
   bool owns_handle = true;
};

}  // namespace Containers
}  // namespace noa::TNL

#endif  // HAVE_HYPRE
