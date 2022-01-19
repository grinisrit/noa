// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

// Implemented by: Tomas Oberhuber, Jakub Klinkovsky

#pragma once

#include "SharedPointer.h"

#include <TNL/Devices/Host.h>
#include <TNL/Cuda/CudaCallable.h>
#include <TNL/Pointers/SmartPointer.h>

#include <cstddef>   // std::nullptr_t
#include <algorithm> // swap

namespace TNL {
namespace Pointers {

/**
 * \brief Specialization of the \ref SharedPointer for the host system.
 *
 * \tparam  Object is a type of object to be owned by the pointer.
 */
template< typename Object >
class SharedPointer< Object, Devices::Host > : public SmartPointer
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
      using DeviceType = Devices::Host;

      /**
       * \brief Constructor of an empty pointer.
       */
      SharedPointer( std::nullptr_t )
      : pd( nullptr )
      {}

      /**
       * \brief Constructor with parameters of the Object constructor.
       *
       * \tparam Args is variadic template type of arguments of the Object constructor.
       * \param args are arguments passed to the Object constructor.
       */
      template< typename... Args >
      explicit  SharedPointer( Args... args )
      : pd( nullptr )
      {
#ifdef TNL_DEBUG_SHARED_POINTERS
         std::cerr << "Creating shared pointer to " << getType< ObjectType >() << std::endl;
#endif
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
      : pd( nullptr )
      {
#ifdef TNL_DEBUG_SHARED_POINTERS
         std::cerr << "Creating shared pointer to " << getType< ObjectType >() << std::endl;
#endif
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
      : pd( nullptr )
      {
#ifdef TNL_DEBUG_SHARED_POINTERS
         std::cerr << "Creating shared pointer to " << getType< ObjectType >() << std::endl;
#endif
         this->allocate( list );
      }

      /**
       * \brief Copy constructor.
       *
       * \param pointer is the source shared pointer.
       */
      SharedPointer( const SharedPointer& pointer ) // this is needed only to avoid the default compiler-generated constructor
      : pd( (PointerData*) pointer.pd )
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
      : pd( (PointerData*) pointer.pd )
      {
         this->pd->counter += 1;
      }

      /**
       * \brief Move constructor.
       *
       * \param pointer is the source shared pointer.
       */
      SharedPointer( SharedPointer&& pointer ) // this is needed only to avoid the default compiler-generated constructor
      : pd( (PointerData*) pointer.pd )
      {
         pointer.pd = nullptr;
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
      : pd( (PointerData*) pointer.pd )
      {
         pointer.pd = nullptr;
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

      /**
       * \brief Arrow operator for accessing the object owned by constant smart pointer.
       *
       * \return constant pointer to the object owned by this smart pointer.
       */
      const Object* operator->() const
      {
         TNL_ASSERT_TRUE( this->pd, "Attempt to dereference a null pointer" );
         return &this->pd->data;
      }

      /**
       * \brief Arrow operator for accessing the object owned by non-constant smart pointer.
       *
       * \return pointer to the object owned by this smart pointer.
       */
      Object* operator->()
      {
         TNL_ASSERT_TRUE( this->pd, "Attempt to dereference a null pointer" );
         return &this->pd->data;
      }

      /**
       * \brief Dereferencing operator for accessing the object owned by constant smart pointer.
       *
       * \return constant reference to the object owned by this smart pointer.
       */
      const Object& operator *() const
      {
         TNL_ASSERT_TRUE( this->pd, "Attempt to dereference a null pointer" );
         return this->pd->data;
      }

      /**
       * \brief Dereferencing operator for accessing the object owned by non-constant smart pointer.
       *
       * \return reference to the object owned by this smart pointer.
       */
      Object& operator *()
      {
         TNL_ASSERT_TRUE( this->pd, "Attempt to dereference a null pointer" );
         return this->pd->data;
      }

      /**
       * \brief Conversion to boolean type.
       *
       * \return Returns true if the pointer is not empty, false otherwise.
       */
      operator bool() const
      {
         return this->pd;
      }

      /**
       * \brief Negation operator.
       *
       * \return Returns false if the pointer is not empty, true otherwise.
       */
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
         TNL_ASSERT_TRUE( this->pd, "Attempt to dereference a null pointer" );
         return this->pd->data;
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
         TNL_ASSERT_TRUE( this->pd, "Attempt to dereference a null pointer" );
         return this->pd->data;
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
         if( this->pd != nullptr )
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
      const SharedPointer& operator=( const SharedPointer<  Object_, DeviceType >& ptr ) // conditional operator for non-const -> const data
      {
         this->free();
         this->pd = (PointerData*) ptr.pd;
         if( this->pd != nullptr )
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
      const SharedPointer& operator=( SharedPointer&& ptr ) // this is needed only to avoid the default compiler-generated operator
      {
         this->free();
         this->pd = (PointerData*) ptr.pd;
         ptr.pd = nullptr;
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
         ptr.pd = nullptr;
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
      }

      /**
       * \brief Destructor.
       */
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
         this->pd = new PointerData( args... );
         return this->pd;
      }

      void free()
      {
         if( this->pd )
         {
            if( ! --this->pd->counter )
            {
               delete this->pd;
               this->pd = nullptr;
            }
         }

      }

      PointerData* pd;
};

} // namespace Pointers
} // namespace TNL
