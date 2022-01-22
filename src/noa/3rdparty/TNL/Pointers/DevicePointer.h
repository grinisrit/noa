// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

// Implemented by: Jakub Klinkovsky

#pragma once

#include <noa/3rdparty/TNL/Allocators/Default.h>
#include <noa/3rdparty/TNL/Devices/Host.h>
#include <noa/3rdparty/TNL/Devices/Cuda.h>
#include <noa/3rdparty/TNL/Pointers/SmartPointer.h>
#include <noa/3rdparty/TNL/Pointers/SmartPointersRegister.h>
#include <noa/3rdparty/TNL/TypeInfo.h>

#include <cstring>  // std::memcpy, std::memcmp

namespace noaTNL {
namespace Pointers {

/**
 * \brief The DevicePointer is like SharedPointer, except it takes an existing host
 * object - there is no call to the ObjectType's constructor nor destructor.
 *
 * **NOTE: When using smart pointers to pass objects on GPU, one must call
 * \ref Pointers::synchronizeSmartPointersOnDevice< Devices::Cuda >()
 * before calling a CUDA kernel working with smart pointers.**
 *
 * \tparam Object is a type of object to be owned by the pointer.
 * \tparam Device is device where the object is to be allocated. The object is
 * always allocated on the host system as well for easier object manipulation.
 *
 * See also \ref UniquePointer and \ref SharedPointer.
 *
 * See also \ref DevicePointer< Object, Devices::Host > and \ref DevicePointer< Object, Devices::Cuda >.
 *
 * \par Example
 * \include Pointers/DevicePointerExample.cpp
 * \par Output
 * \include DevicePointerExample.out
 */
template< typename Object,
          typename Device = typename Object::DeviceType >
class DevicePointer
{
   static_assert( ! std::is_same< Device, void >::value, "The device cannot be void. You need to specify the device explicitly in your code." );
};

/**
 * \brief Specialization of the \ref DevicePointer for the host system.
 *
 * \tparam  Object is a type of object to be owned by the pointer.
 */
template< typename Object >
class DevicePointer< Object, Devices::Host > : public SmartPointer
{
   private:
      /**
       * \typedef Enabler
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
      friend class DevicePointer;

   public:

      /**
       * \typedef ObjectType is the type of object owned by the pointer.
       */
      using ObjectType = Object;

      /**
       * \typedef DeviceType is the type of device where the object is to be
       * mirrored.
       */
      using DeviceType = Devices::Host;

      /**
       * \brief Constructor of an empty pointer.
       */
      DevicePointer( std::nullptr_t )
      : pointer( nullptr )
      {}

      /**
       * \brief Constructor with an object reference.
       *
       * \param obj reference to an object to be managed by the pointer.
       */
      explicit  DevicePointer( ObjectType& obj )
      : pointer( nullptr )
      {
         this->pointer = &obj;
      }

      /**
       * \brief Copy constructor.
       *
       * \param pointer is the source device pointer.
       */
      DevicePointer( const DevicePointer& pointer ) // this is needed only to avoid the default compiler-generated constructor
      : pointer( pointer.pointer )
      {
      }

      /**
       * \brief Copy constructor.
       *
       * This is specialization for compatible object types.
       *
       * See \ref Enabler.
       *
       * \param pointer is the source device pointer.
       */
      template< typename Object_,
                typename = typename Enabler< Object_ >::type >
      DevicePointer( const DevicePointer< Object_, DeviceType >& pointer ) // conditional constructor for non-const -> const data
      : pointer( pointer.pointer )
      {
      }

      /**
       * \brief Move constructor.
       *
       * \param pointer is the source device pointer.
       */
      DevicePointer( DevicePointer&& pointer ) // this is needed only to avoid the default compiler-generated constructor
      : pointer( pointer.pointer )
      {
         pointer.pointer = nullptr;
      }

      /**
       * \brief Move constructor.
       *
       * This is specialization for compatible object types.
       *
       * See \ref Enabler.
       *
       * \param pointer is the source device pointer.
       */
      template< typename Object_,
                typename = typename Enabler< Object_ >::type >
      DevicePointer( DevicePointer< Object_, DeviceType >&& pointer ) // conditional constructor for non-const -> const data
      : pointer( pointer.pointer )
      {
         pointer.pointer = nullptr;
      }

      /**
       * \brief Arrow operator for accessing the object owned by constant smart pointer.
       *
       * \return constant pointer to the object owned by this smart pointer.
       */
      const Object* operator->() const
      {
         return this->pointer;
      }

      /**
       * \brief Arrow operator for accessing the object owned by non-constant smart pointer.
       *
       * \return pointer to the object owned by this smart pointer.
       */
      Object* operator->()
      {
         return this->pointer;
      }

      /**
       * \brief Dereferencing operator for accessing the object owned by constant smart pointer.
       *
       * \return constant reference to the object owned by this smart pointer.
       */
      const Object& operator *() const
      {
         return *( this->pointer );
      }

      /**
       * \brief Dereferencing operator for accessing the object owned by non-constant smart pointer.
       *
       * \return reference to the object owned by this smart pointer.
       */
      Object& operator *()
      {
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
      bool operator!() const
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
      __cuda_callable__
      const Object& getData() const
      {
         return *( this->pointer );
      }

      /**
       * \brief Non-constant object reference getter.
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
      Object& modifyData()
      {
         return *( this->pointer );
      }

      /**
       * \brief Assignment operator.
       *
       * It assigns object owned by the pointer \ref ptr to \ref this pointer.
       *
       * \param ptr input pointer
       * \return constant reference to \ref this
       */
      const DevicePointer& operator=( const DevicePointer& ptr ) // this is needed only to avoid the default compiler-generated operator
      {
         this->pointer = ptr.pointer;
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
      const DevicePointer& operator=( const DevicePointer< Object_, DeviceType >& ptr ) // conditional operator for non-const -> const data
      {
         this->pointer = ptr.pointer;
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
      const DevicePointer& operator=( DevicePointer&& ptr ) // this is needed only to avoid the default compiler-generated operator
      {
         this->pointer = ptr.pointer;
         ptr.pointer = nullptr;
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
      const DevicePointer& operator=( DevicePointer< Object_, DeviceType >&& ptr ) // conditional operator for non-const -> const data
      {
         this->pointer = ptr.pointer;
         ptr.pointer = nullptr;
         return *this;
      }

      /**
       * \brief Cross-device pointer synchronization.
       *
       * For the smart pointers on the host, this method does nothing.
       *
       * \return true.
       */
      bool synchronize()
      {
         return true;
      }

      /**
       * \brief Swap the owned object with another pointer.
       *
       * \param ptr2 the other device pointer for swapping.
       */
      void swap( DevicePointer& ptr2 )
      {
         std::swap( this->pointer, ptr2.pointer );
      }

      /**
       * \brief Destructor.
       */
      ~DevicePointer()
      {
      }


   protected:

      Object* pointer;
};

/**
 * \brief Specialization of the \ref DevicePointer for the CUDA device.
 *
 * \tparam  Object is a type of object to be owned by the pointer.
 */
template< typename Object >
class DevicePointer< Object, Devices::Cuda > : public SmartPointer
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
      friend class DevicePointer;

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
      DevicePointer( std::nullptr_t )
      : pointer( nullptr ),
        pd( nullptr ),
        cuda_pointer( nullptr ) {}

      /**
       * \brief Constructor with an object reference.
       *
       * \param obj is a reference on an object to be managed by the pointer.
       */
      explicit  DevicePointer( ObjectType& obj )
      : pointer( nullptr ),
        pd( nullptr ),
        cuda_pointer( nullptr )
      {
         this->allocate( obj );
      }

      /**
       * \brief Copy constructor.
       *
       * \param pointer is the source device pointer.
       */
      DevicePointer( const DevicePointer& pointer ) // this is needed only to avoid the default compiler-generated constructor
      : pointer( pointer.pointer ),
        pd( (PointerData*) pointer.pd ),
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
       * \param pointer is the source device pointer.
       */
      template< typename Object_,
                typename = typename Enabler< Object_ >::type >
      DevicePointer( const DevicePointer< Object_, DeviceType >& pointer ) // conditional constructor for non-const -> const data
      : pointer( pointer.pointer ),
        pd( (PointerData*) pointer.pd ),
        cuda_pointer( pointer.cuda_pointer )
      {
         this->pd->counter += 1;
      }

      /**
       * \brief Move constructor.
       *
       * \param pointer is the source device pointer.
       */
      DevicePointer( DevicePointer&& pointer ) // this is needed only to avoid the default compiler-generated constructor
      : pointer( pointer.pointer ),
        pd( (PointerData*) pointer.pd ),
        cuda_pointer( pointer.cuda_pointer )
      {
         pointer.pointer = nullptr;
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
       * \param pointer is the source device pointer.
       */
      template< typename Object_,
                typename = typename Enabler< Object_ >::type >
      DevicePointer( DevicePointer< Object_, DeviceType >&& pointer ) // conditional constructor for non-const -> const data
      : pointer( pointer.pointer ),
        pd( (PointerData*) pointer.pd ),
        cuda_pointer( pointer.cuda_pointer )
      {
         pointer.pointer = nullptr;
         pointer.pd = nullptr;
         pointer.cuda_pointer = nullptr;
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
#ifdef __CUDA_ARCH__
         return this->cuda_pointer;
#else
         return this->pointer;
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
#ifdef __CUDA_ARCH__
         return this->cuda_pointer;
#else
         this->pd->maybe_modified = true;
         return this->pointer;
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
#ifdef __CUDA_ARCH__
         return *( this->cuda_pointer );
#else
         return *( this->pointer );
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
#ifdef __CUDA_ARCH__
         return *( this->cuda_pointer );
#else
         this->pd->maybe_modified = true;
         return *( this->pointer );
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
         TNL_ASSERT( this->pointer, );
         TNL_ASSERT( this->pd, );
         TNL_ASSERT( this->cuda_pointer, );
         if( std::is_same< Device, Devices::Host >::value )
            return *( this->pointer );
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
         TNL_ASSERT( this->pointer, );
         TNL_ASSERT( this->pd, );
         TNL_ASSERT( this->cuda_pointer, );
         if( std::is_same< Device, Devices::Host >::value )
         {
            this->pd->maybe_modified = true;
            return *( this->pointer );
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
      const DevicePointer& operator=( const DevicePointer& ptr ) // this is needed only to avoid the default compiler-generated operator
      {
         this->free();
         this->pointer = ptr.pointer;
         this->pd = (PointerData*) ptr.pd;
         this->cuda_pointer = ptr.cuda_pointer;
         if( this->pd )
            this->pd->counter += 1;
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
      const DevicePointer& operator=( const DevicePointer< Object_, DeviceType >& ptr ) // conditional operator for non-const -> const data
      {
         this->free();
         this->pointer = ptr.pointer;
         this->pd = (PointerData*) ptr.pd;
         this->cuda_pointer = ptr.cuda_pointer;
         if( this->pd )
            this->pd->counter += 1;
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
      const DevicePointer& operator=( DevicePointer&& ptr ) // this is needed only to avoid the default compiler-generated operator
      {
         this->free();
         this->pointer = ptr.pointer;
         this->pd = (PointerData*) ptr.pd;
         this->cuda_pointer = ptr.cuda_pointer;
         ptr.pointer = nullptr;
         ptr.pd = nullptr;
         ptr.cuda_pointer = nullptr;
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
      const DevicePointer& operator=( DevicePointer< Object_, DeviceType >&& ptr ) // conditional operator for non-const -> const data
      {
         this->free();
         this->pointer = ptr.pointer;
         this->pd = (PointerData*) ptr.pd;
         this->cuda_pointer = ptr.cuda_pointer;
         ptr.pointer = nullptr;
         ptr.pd = nullptr;
         ptr.cuda_pointer = nullptr;
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
            TNL_ASSERT( this->pointer, );
            TNL_ASSERT( this->cuda_pointer, );
            cudaMemcpy( (void*) this->cuda_pointer, (void*) this->pointer, sizeof( ObjectType ), cudaMemcpyHostToDevice );
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
       * \brief Swap the owned object with another pointer.
       *
       * \param ptr2 the other device pointer for swapping.
       */
      void swap( DevicePointer& ptr2 )
      {
         std::swap( this->pointer, ptr2.pointer );
         std::swap( this->pd, ptr2.pd );
         std::swap( this->cuda_pointer, ptr2.cuda_pointer );
      }

      /**
       * \brief Destructor.
       */
      ~DevicePointer()
      {
         this->free();
         getSmartPointersRegister< DeviceType >().remove( this );
      }

   protected:

      struct PointerData
      {
         char data_image[ sizeof(Object) ];
         int counter = 1;
         bool maybe_modified = true;
      };

      bool allocate( ObjectType& obj )
      {
         this->pointer = &obj;
         this->pd = new PointerData();
         // allocate on device
         this->cuda_pointer = AllocatorType{}.allocate(1);
         // synchronize
         this->synchronize();
         getSmartPointersRegister< DeviceType >().insert( this );
         return true;
      }

      void set_last_sync_state()
      {
         TNL_ASSERT( this->pointer, );
         TNL_ASSERT( this->pd, );
         std::memcpy( (void*) &this->pd->data_image, (void*) this->pointer, sizeof( Object ) );
         this->pd->maybe_modified = false;
      }

      bool modified()
      {
         TNL_ASSERT( this->pointer, );
         TNL_ASSERT( this->pd, );
         // optimization: skip bitwise comparison if we're sure that the data is the same
         if( ! this->pd->maybe_modified )
            return false;
         return std::memcmp( (void*) &this->pd->data_image, (void*) this->pointer, sizeof( Object ) ) != 0;
      }

      void free()
      {
         if( this->pd )
         {
            if( ! --this->pd->counter )
            {
               delete this->pd;
               this->pd = nullptr;
               if( this->cuda_pointer )
                  AllocatorType{}.deallocate( this->cuda_pointer, 1 );
            }
         }
      }

      Object* pointer;

      PointerData* pd;

      // cuda_pointer can't be part of PointerData structure, since we would be
      // unable to dereference this-pd on the device
      Object* cuda_pointer;
};

} // namespace Pointers

#ifndef NDEBUG
namespace Assert {

template< typename Object, typename Device >
struct Formatter< Pointers::DevicePointer< Object, Device > >
{
   static std::string
   printToString( const Pointers::DevicePointer< Object, Device >& value )
   {
      ::std::stringstream ss;
      ss << "(" + getType< Pointers::DevicePointer< Object, Device > >()
         << " object at " << &value << ")";
      return ss.str();
   }
};

} // namespace Assert
#endif

} // namespace noaTNL
