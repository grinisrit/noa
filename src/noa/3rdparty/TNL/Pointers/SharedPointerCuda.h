// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

// Implemented by: Tomas Oberhuber, Jakub Klinkovsky

#pragma once

#include "SharedPointer.h"

#include <TNL/Allocators/Default.h>
#include <TNL/Devices/Cuda.h>
#include <TNL/Pointers/SmartPointer.h>
#include <TNL/Pointers/SmartPointersRegister.h>

#include <cstring>   // std::memcpy, std::memcmp
#include <cstddef>   // std::nullptr_t
#include <algorithm> // swap

namespace TNL {
namespace Pointers {

//#define HAVE_CUDA_UNIFIED_MEMORY

#if ! defined HAVE_CUDA_UNIFIED_MEMORY

/**
 * \brief Specialization of the \ref SharedPointer for the CUDA device.
 *
 * \tparam  Object is a type of object to be owned by the pointer.
 */
template< typename Object >
class SharedPointer< Object, Devices::Cuda > : public SmartPointer
{
   private:
      /**
       * \typedef Enabler
       *
       * Convenient template alias for controlling the selection of copy- and
       * move-constructors and assignment operators using SFINAE.
       * The type Object_ is "enabled" iff Object_ and Object are not the same,
       * but after removing const and volatile qualifiers they are the same.
       */
      template< typename Object_ >
      using Enabler = std::enable_if< ! std::is_same< Object_, Object >::value &&
                                      std::is_same< typename std::remove_cv< Object >::type, Object_ >::value >;

      // friend class will be needed for templated assignment operators
      template< typename Object_, typename Device_ >
      friend class SharedPointer;

   public:

      /**
       * \typedef ObjectType is the type of object owned by the pointer.
       */
      using ObjectType = Object;

      /**
       * \typedef DeviceType is the type of device where the object is to be
       * mirrored.
       */
      using DeviceType = Devices::Cuda;

      /**
       * \typedef AllocatorType is the type of the allocator for \e DeviceType.
       */
      using AllocatorType = typename Allocators::Default< DeviceType >::Allocator< ObjectType >;

      /**
       * \brief Constructor of empty pointer.
       */
      SharedPointer( std::nullptr_t )
      : pd( nullptr ),
        cuda_pointer( nullptr )
      {}

      /**
       * \brief Constructor with parameters of the Object constructor.
       *
       * \tparam Args is variadic template type of arguments of the Object constructor.
       * \tparam args are arguments passed to the Object constructor.
       */
      template< typename... Args >
      explicit  SharedPointer( Args... args )
      : pd( nullptr ),
        cuda_pointer( nullptr )
      {
         this->allocate( args... );
      }

      /**
       * \brief Constructor with initializer list.
       *
       * \tparam Value is type of the initializer list elements.
       * \param list is the instance of the initializer list..
       */
      template< typename Value >
      explicit  SharedPointer( std::initializer_list< Value > list )
      : pd( nullptr ),
        cuda_pointer( nullptr )
      {
         this->allocate( list );
      }

      /**
       * \brief Constructor with nested initializer lists.
       *
       * \tparam Value is type of the nested initializer list elements.
       * \param list is the instance of the nested initializer list..
       */
      template< typename Value >
      explicit  SharedPointer( std::initializer_list< std::initializer_list< Value > > list )
      : pd( nullptr ),
        cuda_pointer( nullptr )
      {
         this->allocate( list );
      }

      /**
       * \brief Copy constructor.
       *
       * \param pointer is the source shared pointer.
       */
      SharedPointer( const SharedPointer& pointer ) // this is needed only to avoid the default compiler-generated constructor
      : pd( (PointerData*) pointer.pd ),
        cuda_pointer( pointer.cuda_pointer )
      {
         this->pd->counter += 1;
      }

      /**
       * \brief Copy constructor.
       *
       * This is specialization for compatible object types.
       *
       * See \ref Enabler.
       *
       * \param pointer is the source shared pointer.
       */
      template< typename Object_,
                typename = typename Enabler< Object_ >::type >
      SharedPointer( const SharedPointer<  Object_, DeviceType >& pointer ) // conditional constructor for non-const -> const data
      : pd( (PointerData*) pointer.pd ),
        cuda_pointer( pointer.cuda_pointer )
      {
         this->pd->counter += 1;
      }

      /**
       * \brief Move constructor.
       *
       * \param pointer is the source shared pointer.
       */
      SharedPointer( SharedPointer&& pointer ) // this is needed only to avoid the default compiler-generated constructor
      : pd( (PointerData*) pointer.pd ),
        cuda_pointer( pointer.cuda_pointer )
      {
         pointer.pd = nullptr;
         pointer.cuda_pointer = nullptr;
      }

      /**
       * \brief Move constructor.
       *
       * This is specialization for compatible object types.
       *
       * See \ref Enabler.
       *
       * \param pointer is the source shared pointer.
       */
      template< typename Object_,
                typename = typename Enabler< Object_ >::type >
      SharedPointer( SharedPointer<  Object_, DeviceType >&& pointer ) // conditional constructor for non-const -> const data
      : pd( (PointerData*) pointer.pd ),
        cuda_pointer( pointer.cuda_pointer )
      {
         pointer.pd = nullptr;
         pointer.cuda_pointer = nullptr;
      }

      /**
       * \brief Create new object based in given constructor parameters.
       *
       * \tparam Args is variadic template type of arguments to be passed to the
       *    object constructor.
       * \param args are arguments to be passed to the object constructor.
       * \return true if recreation was successful, false otherwise.
       */
      template< typename... Args >
      bool recreate( Args... args )
      {
#ifdef TNL_DEBUG_SHARED_POINTERS
         std::cerr << "Recreating shared pointer to " << getType< ObjectType >() << std::endl;
#endif
         if( ! this->pd )
            return this->allocate( args... );

         if( this->pd->counter == 1 )
         {
            /****
             * The object is not shared -> recreate it in-place, without reallocation
             */
            this->pd->data.~Object();
            new ( &this->pd->data ) Object( args... );
#ifdef HAVE_CUDA
            cudaMemcpy( (void*) this->cuda_pointer, (void*) &this->pd->data, sizeof( Object ), cudaMemcpyHostToDevice );
#endif
            this->set_last_sync_state();
            return true;
         }

         // free will just decrement the counter
         this->free();

         return this->allocate( args... );
      }

      /**
       * \brief Arrow operator for accessing the object owned by constant smart pointer.
       *
       * \return constant pointer to the object owned by this smart pointer. It
       * returns pointer to object image on the CUDA device if it is called from CUDA
       * kernel and pointer to host image otherwise.
       */
      __cuda_callable__
      const Object* operator->() const
      {
         TNL_ASSERT_TRUE( this->pd, "Attempt to dereference a null pointer" );
#ifdef __CUDA_ARCH__
         return this->cuda_pointer;
#else
         return &this->pd->data;
#endif
      }

      /**
       * \brief Arrow operator for accessing the object owned by non-constant smart pointer.
       *
       * \return pointer to the object owned by this smart pointer. It
       * returns pointer to object image on the CUDA device if it is called from CUDA
       * kernel and pointer to host image otherwise.
       */
      __cuda_callable__
      Object* operator->()
      {
         TNL_ASSERT_TRUE( this->pd, "Attempt to dereference a null pointer" );
#ifdef __CUDA_ARCH__
         return this->cuda_pointer;
#else
         this->pd->maybe_modified = true;
         return &this->pd->data;
#endif
      }

      /**
       * \brief Dereferencing operator for accessing the object owned by constant smart pointer.
       *
       * \return constant reference to the object owned by this smart pointer. It
       * returns reference to object image on the CUDA device if it is called from CUDA
       * kernel and reference to host image otherwise.
       */
      __cuda_callable__
      const Object& operator *() const
      {
         TNL_ASSERT_TRUE( this->pd, "Attempt to dereference a null pointer" );
#ifdef __CUDA_ARCH__
         return *( this->cuda_pointer );
#else
         return this->pd->data;
#endif
      }

      /**
       * \brief Dereferencing operator for accessing the object owned by non-constant smart pointer.
       *
       * \return reference to the object owned by this smart pointer.  It
       * returns reference to object image on the CUDA device if it is called from CUDA
       * kernel and reference to host image otherwise.
       */
      __cuda_callable__
      Object& operator *()
      {
         TNL_ASSERT_TRUE( this->pd, "Attempt to dereference a null pointer" );
#ifdef __CUDA_ARCH__
         return *( this->cuda_pointer );
#else
         this->pd->maybe_modified = true;
         return this->pd->data;
#endif
      }

      /**
       * \brief Conversion to boolean type.
       *
       * \return Returns true if the pointer is not empty, false otherwise.
       */
      __cuda_callable__
      operator bool() const
      {
         return this->pd;
      }

      /**
       * \brief Negation operator.
       *
       * \return Returns false if the pointer is not empty, true otherwise.
       */
      __cuda_callable__
      bool operator!() const
      {
         return ! this->pd;
      }

      /**
       * \brief Constant object reference getter.
       *
       * No synchronization of this pointer will be performed due to calling
       * this method.
       *
       * \tparam Device says what image of the object one want to dereference. It
       * can be either \ref DeviceType or Devices::Host.
       * \return constant reference to the object image on given device.
       */
      template< typename Device = Devices::Host >
      __cuda_callable__
      const Object& getData() const
      {
         static_assert( std::is_same< Device, Devices::Host >::value || std::is_same< Device, Devices::Cuda >::value, "Only Devices::Host or Devices::Cuda devices are accepted here." );
         TNL_ASSERT_TRUE( this->pd, "Attempt to dereference a null pointer" );
         TNL_ASSERT_TRUE( this->cuda_pointer, "Attempt to dereference a null pointer" );
         if( std::is_same< Device, Devices::Host >::value )
            return this->pd->data;
         if( std::is_same< Device, Devices::Cuda >::value )
            return *( this->cuda_pointer );
      }

      /**
       * \brief Non-constant object reference getter.
       *
       * After calling this method, the object owned by the pointer might need
       * to be synchronized. One should not forget to call
       * \ref Pointers::synchronizeSmartPointersOnDevice< Devices::Cuda >()
       * before calling CUDA kernel using object from this smart pointer.
       *
       * \tparam Device says what image of the object one want to dereference. It
       * can be either \ref DeviceType or Devices::Host.
       * \return constant reference to the object image on given device.
       */
      template< typename Device = Devices::Host >
      __cuda_callable__
      Object& modifyData()
      {
         static_assert( std::is_same< Device, Devices::Host >::value || std::is_same< Device, Devices::Cuda >::value, "Only Devices::Host or Devices::Cuda devices are accepted here." );
         TNL_ASSERT_TRUE( this->pd, "Attempt to dereference a null pointer" );
         TNL_ASSERT_TRUE( this->cuda_pointer, "Attempt to dereference a null pointer" );
         if( std::is_same< Device, Devices::Host >::value )
         {
            this->pd->maybe_modified = true;
            return this->pd->data;
         }
         if( std::is_same< Device, Devices::Cuda >::value )
            return *( this->cuda_pointer );
      }

      /**
       * \brief Assignment operator.
       *
       * It assigns object owned by the pointer \ref ptr to \ref this pointer.
       *
       * \param ptr input pointer
       * \return constant reference to \ref this
       */
      const SharedPointer& operator=( const SharedPointer& ptr ) // this is needed only to avoid the default compiler-generated operator
      {
         this->free();
         this->pd = (PointerData*) ptr.pd;
         this->cuda_pointer = ptr.cuda_pointer;
         if( this->pd != nullptr )
            this->pd->counter += 1;
#ifdef TNL_DEBUG_SHARED_POINTERS
         std::cerr << "Copy-assigned shared pointer: counter = " << this->pd->counter << ", type: " << getType< ObjectType >() << std::endl;
#endif
         return *this;
      }

      /**
       * \brief Assignment operator for compatible object types.
       *
       * It assigns object owned by the pointer \ref ptr to \ref this pointer.
       *
       * See \ref Enabler.
       *
       * \param ptr input pointer
       * \return constant reference to \ref this
       */
      template< typename Object_,
                typename = typename Enabler< Object_ >::type >
      const SharedPointer& operator=( const SharedPointer<  Object_, DeviceType >& ptr ) // conditional operator for non-const -> const data
      {
         this->free();
         this->pd = (PointerData*) ptr.pd;
         this->cuda_pointer = ptr.cuda_pointer;
         if( this->pd != nullptr )
            this->pd->counter += 1;
#ifdef TNL_DEBUG_SHARED_POINTERS
         std::cerr << "Copy-assigned shared pointer: counter = " << this->pd->counter << ", type: " << getType< ObjectType >() << std::endl;
#endif
         return *this;
      }

      /**
       * \brief Move operator.
       *
       * It assigns object owned by the pointer \ref ptr to \ref this pointer.
       *
       * \param ptr input pointer
       * \return constant reference to \ref this
       */
      const SharedPointer& operator=( SharedPointer&& ptr ) // this is needed only to avoid the default compiler-generated operator
      {
         this->free();
         this->pd = (PointerData*) ptr.pd;
         this->cuda_pointer = ptr.cuda_pointer;
         ptr.pd = nullptr;
         ptr.cuda_pointer = nullptr;
#ifdef TNL_DEBUG_SHARED_POINTERS
         std::cerr << "Move-assigned shared pointer: counter = " << this->pd->counter << ", type: " << getType< ObjectType >() << std::endl;
#endif
         return *this;
      }

      /**
       * \brief Move operator.
       *
       * It assigns object owned by the pointer \ref ptr to \ref this pointer.
       *
       * See \ref Enabler.
       *
       * \param ptr input pointer
       * \return constant reference to \ref this
       */
      template< typename Object_,
                typename = typename Enabler< Object_ >::type >
      const SharedPointer& operator=( SharedPointer<  Object_, DeviceType >&& ptr ) // conditional operator for non-const -> const data
      {
         this->free();
         this->pd = (PointerData*) ptr.pd;
         this->cuda_pointer = ptr.cuda_pointer;
         ptr.pd = nullptr;
         ptr.cuda_pointer = nullptr;
#ifdef TNL_DEBUG_SHARED_POINTERS
         std::cerr << "Move-assigned shared pointer: counter = " << this->pd->counter << ", type: " << getType< ObjectType >() << std::endl;
#endif
         return *this;
      }

      /**
       * \brief Cross-device pointer synchronization.
       *
       * This method is usually called by the smart pointers register when calling
       * \ref Pointers::synchronizeSmartPointersOnDevice< Devices::Cuda >()
       *
       * \return true if the synchronization was successful, false otherwise.
       */
      bool synchronize()
      {
         if( ! this->pd )
            return true;
#ifdef HAVE_CUDA
         if( this->modified() )
         {
#ifdef TNL_DEBUG_SHARED_POINTERS
            std::cerr << "Synchronizing shared pointer: counter = " << this->pd->counter << ", type: " << getType< ObjectType >() << std::endl;
            std::cerr << "   ( " << sizeof( Object ) << " bytes, CUDA adress " << this->cuda_pointer << " )" << std::endl;
#endif
            TNL_ASSERT( this->cuda_pointer, );
            cudaMemcpy( (void*) this->cuda_pointer, (void*) &this->pd->data, sizeof( Object ), cudaMemcpyHostToDevice );
            TNL_CHECK_CUDA_DEVICE;
            this->set_last_sync_state();
            return true;
         }
         return true;
#else
         return false;
#endif
      }

      /**
       * \brief Reset the pointer to empty state.
       */
      void clear()
      {
         this->free();
      }

      /**
       * \brief Swap the owned object with another pointer.
       *
       * \param ptr2 the other shared pointer for swapping.
       */
      void swap( SharedPointer& ptr2 )
      {
         std::swap( this->pd, ptr2.pd );
         std::swap( this->cuda_pointer, ptr2.cuda_pointer );
      }

      /**
       * \brief Destructor.
       */
      ~SharedPointer()
      {
         this->free();
         getSmartPointersRegister< DeviceType >().remove( this );
      }

   protected:

      struct PointerData
      {
         Object data;
         char data_image[ sizeof(Object) ];
         int counter;
         bool maybe_modified;

         template< typename... Args >
         explicit PointerData( Args... args )
         : data( args... ),
           counter( 1 ),
           maybe_modified( true )
         {}
      };

      template< typename... Args >
      bool allocate( Args... args )
      {
         this->pd = new PointerData( args... );
         // allocate on device
         this->cuda_pointer = AllocatorType{}.allocate(1);
         // synchronize
         this->synchronize();
#ifdef TNL_DEBUG_SHARED_POINTERS
         std::cerr << "Created shared pointer to " << getType< ObjectType >() << " (cuda_pointer = " << this->cuda_pointer << ")" << std::endl;
#endif
         getSmartPointersRegister< DeviceType >().insert( this );
         return true;
      }

      void set_last_sync_state()
      {
         TNL_ASSERT_TRUE( this->pd, "Attempt to dereference a null pointer" );
         std::memcpy( (void*) &this->pd->data_image, (void*) &this->pd->data, sizeof( Object ) );
         this->pd->maybe_modified = false;
      }

      bool modified()
      {
         TNL_ASSERT_TRUE( this->pd, "Attempt to dereference a null pointer" );
         // optimization: skip bitwise comparison if we're sure that the data is the same
         if( ! this->pd->maybe_modified )
            return false;
         return std::memcmp( (void*) &this->pd->data_image, (void*) &this->pd->data, sizeof( Object ) ) != 0;
      }

      void free()
      {
         if( this->pd )
         {
#ifdef TNL_DEBUG_SHARED_POINTERS
            std::cerr << "Freeing shared pointer: counter = " << this->pd->counter << ", cuda_pointer = " << this->cuda_pointer << ", type: " << getType< ObjectType >() << std::endl;
#endif
            if( ! --this->pd->counter )
            {
               delete this->pd;
               this->pd = nullptr;
               if( this->cuda_pointer )
                  AllocatorType{}.deallocate( this->cuda_pointer, 1 );
#ifdef TNL_DEBUG_SHARED_POINTERS
               std::cerr << "...deleted data." << std::endl;
#endif
            }
         }
      }

      PointerData* pd;

      // cuda_pointer can't be part of PointerData structure, since we would be
      // unable to dereference this-pd on the device
      Object* cuda_pointer;
};

#else
// Implementation with CUDA unified memory. It is very slow, we keep it only for experimental reasons.
template< typename Object >
class SharedPointer< Object, Devices::Cuda > : public SmartPointer
{
   private:
      // Convenient template alias for controlling the selection of copy- and
      // move-constructors and assignment operators using SFINAE.
      // The type Object_ is "enabled" iff Object_ and Object are not the same,
      // but after removing const and volatile qualifiers they are the same.
      template< typename Object_ >
      using Enabler = std::enable_if< ! std::is_same< Object_, Object >::value &&
                                      std::is_same< typename std::remove_cv< Object >::type, Object_ >::value >;

      // friend class will be needed for templated assignment operators
      template< typename Object_, typename Device_ >
      friend class SharedPointer;

   public:

      using ObjectType = Object;
      using DeviceType = Devices::Cuda;

      SharedPointer( std::nullptr_t )
      : pd( nullptr )
      {}

      template< typename... Args >
      explicit  SharedPointer( Args... args )
      : pd( nullptr )
      {
#ifdef TNL_DEBUG_SHARED_POINTERS
         std::cerr << "Creating shared pointer to " << getType< ObjectType >() << std::endl;
#endif
         this->allocate( args... );
      }

      // this is needed only to avoid the default compiler-generated constructor
      SharedPointer( const SharedPointer& pointer )
      : pd( (PointerData*) pointer.pd )
      {
         this->pd->counter += 1;
      }

      // conditional constructor for non-const -> const data
      template< typename Object_,
                typename = typename Enabler< Object_ >::type >
      SharedPointer( const SharedPointer<  Object_, DeviceType >& pointer )
      : pd( (PointerData*) pointer.pd )
      {
         this->pd->counter += 1;
      }

      // this is needed only to avoid the default compiler-generated constructor
      SharedPointer( SharedPointer&& pointer )
      : pd( (PointerData*) pointer.pd )
      {
         pointer.pd = nullptr;
      }

      // conditional constructor for non-const -> const data
      template< typename Object_,
                typename = typename Enabler< Object_ >::type >
      SharedPointer( SharedPointer<  Object_, DeviceType >&& pointer )
      : pd( (PointerData*) pointer.pd )
      {
         pointer.pd = nullptr;
      }

      template< typename... Args >
      bool recreate( Args... args )
      {
#ifdef TNL_DEBUG_SHARED_POINTERS
         std::cerr << "Recreating shared pointer to " << getType< ObjectType >() << std::endl;
#endif
         if( ! this->counter )
            return this->allocate( args... );

         if( *this->pd->counter == 1 )
         {
            /****
             * The object is not shared -> recreate it in-place, without reallocation
             */
            this->pd->data.~ObjectType();
            new ( this->pd->data ) ObjectType( args... );
            return true;
         }

         // free will just decrement the counter
         this->free();

         return this->allocate( args... );
      }

      const Object* operator->() const
      {
         TNL_ASSERT_TRUE( this->pd, "Attempt to dereference a null pointer" );
         return &this->pd->data;
      }

      Object* operator->()
      {
         TNL_ASSERT_TRUE( this->pd, "Attempt to dereference a null pointer" );
         return &this->pd->data;
      }

      const Object& operator *() const
      {
         TNL_ASSERT_TRUE( this->pd, "Attempt to dereference a null pointer" );
         return this->pd->data;
      }

      Object& operator *()
      {
         TNL_ASSERT_TRUE( this->pd, "Attempt to dereference a null pointer" );
         return this->pd->data;
      }

      __cuda_callable__
      operator bool() const
      {
         return this->pd;
      }

      __cuda_callable__
      bool operator!() const
      {
         return ! this->pd;
      }

      template< typename Device = Devices::Host >
      __cuda_callable__
      const Object& getData() const
      {
         TNL_ASSERT_TRUE( this->pd, "Attempt to dereference a null pointer" );
         return this->pd->data;
      }

      template< typename Device = Devices::Host >
      __cuda_callable__
      Object& modifyData()
      {
         TNL_ASSERT_TRUE( this->pd, "Attempt to dereference a null pointer" );
         return this->pd->data;
      }

      // this is needed only to avoid the default compiler-generated operator
      const SharedPointer& operator=( const SharedPointer& ptr )
      {
         this->free();
         this->pd = (PointerData*) ptr.pd;
         if( this->pd != nullptr )
            this->pd->counter += 1;
         return *this;
      }

      // conditional operator for non-const -> const data
      template< typename Object_,
                typename = typename Enabler< Object_ >::type >
      const SharedPointer& operator=( const SharedPointer<  Object_, DeviceType >& ptr )
      {
         this->free();
         this->pd = (PointerData*) ptr.pd;
         if( this->pd != nullptr )
            this->pd->counter += 1;
         return *this;
      }

      // this is needed only to avoid the default compiler-generated operator
      const SharedPointer& operator=( SharedPointer&& ptr )
      {
         this->free();
         this->pd = (PointerData*) ptr.pd;
         ptr.pd = nullptr;
         return *this;
      }

      // conditional operator for non-const -> const data
      template< typename Object_,
                typename = typename Enabler< Object_ >::type >
      const SharedPointer& operator=( SharedPointer<  Object_, DeviceType >&& ptr )
      {
         this->free();
         this->pd = (PointerData*) ptr.pd;
         ptr.pd = nullptr;
         return *this;
      }

      bool synchronize()
      {
         return true;
      }

      void clear()
      {
         this->free();
      }

      void swap( SharedPointer& ptr2 )
      {
         std::swap( this->pd, ptr2.pd );
      }

      ~SharedPointer()
      {
         this->free();
      }


   protected:

      struct PointerData
      {
         Object data;
         int counter;

         template< typename... Args >
         explicit PointerData( Args... args )
         : data( args... ),
           counter( 1 )
         {}
      };

      template< typename... Args >
      bool allocate( Args... args )
      {
#ifdef HAVE_CUDA
         if( cudaMallocManaged( ( void** ) &this->pd, sizeof( PointerData ) != cudaSuccess ) )
            return false;
         new ( this->pd ) PointerData( args... );
         return true;
#else
         return false;
#endif
      }

      void free()
      {
         if( this->pd )
         {
            if( ! --this->pd->counter )
            {
#ifdef HAVE_CUDA
               cudaFree( this->pd );
#endif
               this->pd = nullptr;
            }
         }
      }

      PointerData* pd;
};

#endif // ! HAVE_CUDA_UNIFIED_MEMORY

} // namespace Pointers
} // namespace TNL
