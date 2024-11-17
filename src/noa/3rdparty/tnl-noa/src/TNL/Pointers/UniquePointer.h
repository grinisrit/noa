// Copyright (c) 2004-2023 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

// Implemented by: Tomas Oberhuber, Jakub Klinkovsky

#pragma once

#include <noa/3rdparty/tnl-noa/src/TNL/Allocators/Default.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Devices/Host.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Devices/Cuda.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Pointers/SmartPointer.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Pointers/SmartPointersRegister.h>

#include <cstring>  // std::memcpy, std::memcmp
#include <cstddef>  // std::nullptr_t

namespace noa::TNL {
namespace Pointers {

/**
 * \brief Cross-device unique smart pointer.
 *
 * This smart pointer is inspired by std::unique_ptr from STL library. It means
 * that the object owned by the smart pointer is accessible only through this
 * smart pointer. One cannot make any copy of this smart pointer. In addition,
 * the smart pointer is able to work across different devices which means that the
 * object owned by the smart pointer is mirrored on both host and device.
 *
 * **NOTE: When using smart pointers to pass objects on GPU, one must call
 * \ref Pointers::synchronizeSmartPointersOnDevice< Devices::Cuda >()
 * before calling a CUDA kernel working with smart pointers.**
 *
 * \tparam Object is a type of object to be owned by the pointer.
 * \tparam Device is device where the object is to be allocated. The object is
 * always allocated on the host system as well for easier object manipulation.
 *
 * See also \ref SharedPointer and \ref DevicePointer.
 *
 * See also \ref UniquePointer< Object, Devices::Host > and \ref UniquePointer< Object, Devices::Cuda >.
 *
 * \par Example
 * \include Pointers/UniquePointerExample.cpp
 * \par Output
 * \include UniquePointerExample.out
 */
template< typename Object, typename Device = typename Object::DeviceType >
class UniquePointer
{};

/**
 * \brief Specialization of the \ref UniquePointer for the host system.
 *
 * \tparam  Object is a type of object to be owned by the pointer.
 */
template< typename Object >
class UniquePointer< Object, Devices::Host > : public SmartPointer
{
public:
   /**
    * \brief Type of the object owned by the pointer.
    */
   using ObjectType = Object;

   /**
    * \brief Type of the device where the object is to be mirrored.
    */
   using DeviceType = Devices::Host;

   /**
    * \brief Constructor of empty pointer.
    */
   UniquePointer( std::nullptr_t ) : pointer( nullptr ) {}

   /**
    * \brief Constructor with parameters of the Object constructor.
    *
    * \tparam Args is variadic template type of arguments of the Object constructor.
    * \tparam args are arguments passed to the Object constructor.
    */
   template< typename... Args >
   explicit UniquePointer( const Args... args )
   {
      this->pointer = new Object( args... );
   }

   /**
    * \brief Constructor with initializer list.
    *
    * \tparam Value is type of the initializer list elements.
    * \param list is the instance of the initializer list..
    */
   template< typename Value >
   explicit UniquePointer( std::initializer_list< Value > list )
   {
      this->pointer = new Object( list );
   }

   /**
    * \brief Constructor with nested initializer lists.
    *
    * \tparam Value is type of the nested initializer list elements.
    * \param list is the instance of the nested initializer list..
    */
   template< typename Value >
   explicit UniquePointer( std::initializer_list< std::initializer_list< Value > > list )
   {
      this->pointer = new Object( list );
   }

   /**
    * \brief Arrow operator for accessing the object owned by constant smart pointer.
    *
    * \return constant pointer to the object owned by this smart pointer.
    */
   const Object*
   operator->() const
   {
      TNL_ASSERT_TRUE( this->pointer, "Attempt to dereference a null pointer" );
      return this->pointer;
   }

   /**
    * \brief Arrow operator for accessing the object owned by non-constant smart pointer.
    *
    * \return pointer to the object owned by this smart pointer.
    */
   Object*
   operator->()
   {
      TNL_ASSERT_TRUE( this->pointer, "Attempt to dereference a null pointer" );
      return this->pointer;
   }

   /**
    * \brief Dereferencing operator for accessing the object owned by constant smart pointer.
    *
    * \return constant reference to the object owned by this smart pointer.
    */
   const Object&
   operator*() const
   {
      TNL_ASSERT_TRUE( this->pointer, "Attempt to dereference a null pointer" );
      return *( this->pointer );
   }

   /**
    * \brief Dereferencing operator for accessing the object owned by non-constant smart pointer.
    *
    * \return reference to the object owned by this smart pointer.
    */
   Object&
   operator*()
   {
      TNL_ASSERT_TRUE( this->pointer, "Attempt to dereference a null pointer" );
      return *( this->pointer );
   }

   /**
    * \brief Conversion to boolean type.
    *
    * \return Returns true if the pointer is not empty, false otherwise.
    */
   __cuda_callable__
   operator bool() const
   {
      return this->pointer;
   }

   /**
    * \brief Negation operator.
    *
    * \return Returns false if the pointer is not empty, true otherwise.
    */
   __cuda_callable__
   bool
   operator!() const
   {
      return ! this->pointer;
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
   const Object&
   getData() const
   {
      TNL_ASSERT_TRUE( this->pointer, "Attempt to dereference a null pointer" );
      return *( this->pointer );
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
   Object&
   modifyData()
   {
      TNL_ASSERT_TRUE( this->pointer, "Attempt to dereference a null pointer" );
      return *( this->pointer );
   }

   /**
    * \brief Assignment operator.
    *
    * It assigns object owned by the pointer \e ptr to \e this pointer.
    * The original pointer \e ptr is reset to empty state.
    *
    * \param ptr input pointer
    * \return constant reference to \e this
    */
   const UniquePointer&
   operator=( UniquePointer& ptr )
   {
      if( this->pointer )
         delete this->pointer;
      this->pointer = ptr.pointer;
      ptr.pointer = nullptr;
      return *this;
   }

   /**
    * \brief Move operator.
    *
    * It assigns object owned by the pointer \e ptr to \e this pointer.
    * The original pointer \e ptr is reset to empty state.
    *
    * \param ptr input pointer
    * \return constant reference to \e this
    */
   const UniquePointer&
   operator=( UniquePointer&& ptr ) noexcept
   {
      return this->operator=( ptr );
   }

   /**
    * \brief Cross-device pointer synchronization.
    *
    * For the smart pointers in the host, this method does nothing.
    *
    * \return true.
    */
   bool
   synchronize() override
   {
      return true;
   }

   /**
    * \brief Destructor.
    */
   ~UniquePointer() override
   {
      if( this->pointer )
         delete this->pointer;
   }

protected:
   Object* pointer;
};

/**
 * \brief Specialization of the \ref UniquePointer for the CUDA device.
 *
 * \tparam  Object is a type of object to be owned by the pointer.
 */
template< typename Object >
class UniquePointer< Object, Devices::Cuda > : public SmartPointer
{
public:
   /**
    * \brief Type of the object owned by the pointer.
    */
   using ObjectType = Object;

   /**
    * \brief Type of the device where the object is to be mirrored.
    */
   using DeviceType = Devices::Cuda;

   /**
    * \brief Type of the allocator for \e DeviceType.
    */
   using AllocatorType = typename Allocators::Default< DeviceType >::Allocator< ObjectType >;

   /**
    * \brief Constructor of empty pointer.
    */
   UniquePointer( std::nullptr_t ) : pd( nullptr ), cuda_pointer( nullptr ) {}

   /**
    * \brief Constructor with parameters of the Object constructor.
    *
    * \tparam Args is variadic template type of arguments of the Object constructor.
    * \tparam args are arguments passed to the Object constructor.
    */
   template< typename... Args >
   explicit UniquePointer( const Args... args ) : pd( nullptr ), cuda_pointer( nullptr )
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
   explicit UniquePointer( std::initializer_list< Value > list ) : pd( nullptr ), cuda_pointer( nullptr )
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
   explicit UniquePointer( std::initializer_list< std::initializer_list< Value > > list )
   : pd( nullptr ), cuda_pointer( nullptr )
   {
      this->allocate( list );
   }

   /**
    * \brief Arrow operator for accessing the object owned by constant smart pointer.
    *
    * \return constant pointer to the object owned by this smart pointer.
    */
   const Object*
   operator->() const
   {
      TNL_ASSERT_TRUE( this->pd, "Attempt to dereference a null pointer" );
      return &this->pd->data;
   }

   /**
    * \brief Arrow operator for accessing the object owned by non-constant smart pointer.
    *
    * \return pointer to the object owned by this smart pointer.
    */
   Object*
   operator->()
   {
      TNL_ASSERT_TRUE( this->pd, "Attempt to dereference a null pointer" );
      this->pd->maybe_modified = true;
      return &this->pd->data;
   }

   /**
    * \brief Dereferencing operator for accessing the object owned by constant smart pointer.
    *
    * \return constant reference to the object owned by this smart pointer.
    */
   const Object&
   operator*() const
   {
      TNL_ASSERT_TRUE( this->pd, "Attempt to dereference a null pointer" );
      return this->pd->data;
   }

   /**
    * \brief Dereferencing operator for accessing the object owned by non-constant smart pointer.
    *
    * \return reference to the object owned by this smart pointer.
    */
   Object&
   operator*()
   {
      TNL_ASSERT_TRUE( this->pd, "Attempt to dereference a null pointer" );
      this->pd->maybe_modified = true;
      return this->pd->data;
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
   bool
   operator!() const
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
   const Object&
   getData() const
   {
      static_assert( std::is_same< Device, Devices::Host >::value || std::is_same< Device, Devices::Cuda >::value,
                     "Only Devices::Host or Devices::Cuda devices are accepted here." );
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
   Object&
   modifyData()
   {
      static_assert( std::is_same< Device, Devices::Host >::value || std::is_same< Device, Devices::Cuda >::value,
                     "Only Devices::Host or Devices::Cuda devices are accepted here." );
      TNL_ASSERT_TRUE( this->pd, "Attempt to dereference a null pointer" );
      TNL_ASSERT_TRUE( this->cuda_pointer, "Attempt to dereference a null pointer" );
      if( std::is_same< Device, Devices::Host >::value ) {
         this->pd->maybe_modified = true;
         return this->pd->data;
      }
      if( std::is_same< Device, Devices::Cuda >::value )
         return *( this->cuda_pointer );
   }

   /**
    * \brief Assignment operator.
    *
    * It assigns object owned by the pointer \e ptr to \e this pointer.
    * The original pointer \e ptr is reset to empty state.
    *
    * \param ptr input pointer
    * \return constant reference to \e this
    */
   const UniquePointer&
   operator=( UniquePointer& ptr )
   {
      this->free();
      this->pd = ptr.pd;
      this->cuda_pointer = ptr.cuda_pointer;
      ptr.pd = nullptr;
      ptr.cuda_pointer = nullptr;
      return *this;
   }

   /**
    * \brief Move operator.
    *
    * It assigns object owned by the pointer \e ptr to \e this pointer.
    * The original pointer \e ptr is reset to empty state.
    *
    * \param ptr input pointer
    * \return constant reference to \e this
    */
   const UniquePointer&
   operator=( UniquePointer&& ptr ) noexcept
   {
      return this->operator=( ptr );
   }

   /**
    * \brief Cross-device pointer synchronization.
    *
    * This method is usually called by the smart pointers register when calling
    * \ref Pointers::synchronizeSmartPointersOnDevice< Devices::Cuda >()
    *
    * \return true if the synchronization was successful, false otherwise.
    */
   bool
   synchronize() override
   {
      if( ! this->pd )
         return true;
#ifdef __CUDACC__
      if( this->modified() ) {
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
    * \brief Destructor.
    */
   ~UniquePointer() override
   {
      this->free();
      getSmartPointersRegister< DeviceType >().remove( this );
   }

protected:
   struct PointerData
   {
      Object data;
      char data_image[ sizeof( Object ) ];
      bool maybe_modified = true;

      template< typename... Args >
      explicit PointerData( Args... args ) : data( args... )
      {}
   };

   template< typename... Args >
   bool
   allocate( Args... args )
   {
      this->pd = new PointerData( args... );
      // allocate on device
      this->cuda_pointer = AllocatorType{}.allocate( 1 );
      // synchronize
      this->synchronize();
      getSmartPointersRegister< DeviceType >().insert( this );
      return true;
   }

   void
   set_last_sync_state()
   {
      TNL_ASSERT_TRUE( this->pd, "Attempt to dereference a null pointer" );
      std::memcpy( (void*) &this->pd->data_image, (void*) &this->pd->data, sizeof( ObjectType ) );
      this->pd->maybe_modified = false;
   }

   bool
   modified()
   {
      TNL_ASSERT_TRUE( this->pd, "Attempt to dereference a null pointer" );
      // optimization: skip bitwise comparison if we're sure that the data is the same
      if( ! this->pd->maybe_modified )
         return false;
      return std::memcmp( (void*) &this->pd->data_image, (void*) &this->pd->data, sizeof( ObjectType ) ) != 0;
   }

   void
   free()
   {
      if( this->pd )
         delete this->pd;
      if( this->cuda_pointer )
         AllocatorType{}.deallocate( this->cuda_pointer, 1 );
   }

   PointerData* pd;

   // cuda_pointer can't be part of PointerData structure, since we would be
   // unable to dereference this-pd on the device
   Object* cuda_pointer;
};

}  // namespace Pointers

#ifndef NDEBUG
namespace Assert {

template< typename Object, typename Device >
struct Formatter< Pointers::UniquePointer< Object, Device > >
{
   static std::string
   printToString( const Pointers::UniquePointer< Object, Device >& value )
   {
      ::std::stringstream ss;
      ss << "(" + getType< Pointers::UniquePointer< Object, Device > >() << " > object at " << &value << ")";
      return ss.str();
   }
};

}  // namespace Assert
#endif

}  // namespace noa::TNL
