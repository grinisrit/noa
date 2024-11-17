// Copyright (c) 2004-2023 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <stdexcept>
#include <memory>

#include "Wrappers.h"

namespace noa::TNL {
namespace MPI {

inline int
GetRank( MPI_Comm communicator = MPI_COMM_WORLD )
{
   TNL_ASSERT_NE( communicator, MPI_COMM_NULL, "GetRank cannot be called with MPI_COMM_NULL" );
#ifdef HAVE_MPI
   TNL_ASSERT_TRUE( Initialized() && ! Finalized(), "Fatal Error - MPI is not initialized" );
   int rank;
   MPI_Comm_rank( communicator, &rank );
   return rank;
#else
   return 0;
#endif
}

inline int
GetSize( MPI_Comm communicator = MPI_COMM_WORLD )
{
   TNL_ASSERT_NE( communicator, MPI_COMM_NULL, "GetSize cannot be called with MPI_COMM_NULL" );
#ifdef HAVE_MPI
   TNL_ASSERT_TRUE( Initialized() && ! Finalized(), "Fatal Error - MPI is not initialized" );
   int size;
   MPI_Comm_size( communicator, &size );
   return size;
#else
   return 1;
#endif
}

/**
 * \brief An RAII wrapper for custom MPI communicators.
 *
 * This is an RAII wrapper for custom MPI communicators created by calls to
 * `MPI_Comm_create`, `MPI_Comm_split`, or similar functions. It is based on
 * \ref std::shared_ptr so copy-constructible and copy-assignable, copies of
 * the object represent the same communicator that is deallocated only when the
 * internal reference counter drops to zero.
 *
 * Note that predefined communicators (i.e. `MPI_COMM_WORLD`, `MPI_COMM_NULL`
 * and `MPI_COMM_SELF`) can be used to initialize this class, but other handles
 * of the `MPI_Comm` type _cannot_ be used to initialize this class.
 *
 * This class follows the factory pattern, i.e. it provides static methods such
 * as \ref Comm::duplicate or \ref Comm::split that return an instance of a new
 * communicator.
 */
class Comm
{
private:
   struct Wrapper
   {
      MPI_Comm comm = MPI_COMM_NULL;

      Wrapper() = default;
      Wrapper( const Wrapper& other ) = delete;
      Wrapper( Wrapper&& other ) = default;
      Wrapper&
      operator=( const Wrapper& other ) = delete;
      Wrapper&
      operator=( Wrapper&& other ) = default;

      Wrapper( MPI_Comm comm ) : comm( comm ) {}

      ~Wrapper()
      {
#ifdef HAVE_MPI
         // cannot free a predefined handle
         if( comm != MPI_COMM_NULL && comm != MPI_COMM_WORLD && comm != MPI_COMM_SELF )
            MPI_Comm_free( &comm );
#endif
      }
   };

   std::shared_ptr< Wrapper > wrapper;

   //! \brief Internal constructor for the factory methods - initialization by the wrapper.
   Comm( std::shared_ptr< Wrapper >&& wrapper ) : wrapper( std::move( wrapper ) ) {}

public:
   //! \brief Constructs an empty communicator with a null handle (`MPI_COMM_NULL`).
   Comm() = default;

   //! \brief Default copy-constructor.
   Comm( const Comm& other ) = default;

   //! \brief Default move-constructor.
   Comm( Comm&& other ) = default;

   //! \brief Default copy-assignment operator.
   Comm&
   operator=( const Comm& other ) = default;

   //! \brief Default move-assignment operator.
   Comm&
   operator=( Comm&& other ) = default;

   /**
    * \brief Constructs a communicator initialized by given predefined communicator.
    *
    * Note that only predefined communicators (i.e. `MPI_COMM_WORLD`,
    * `MPI_COMM_NULL` and `MPI_COMM_SELF`) can be used to initialize this
    * class. Other handles of the `MPI_Comm` type _cannot_ be used to
    * initialize this class.
    *
    * \throws std::logic_error when the \e comm handle is not a predefined
    * communicator.
    */
   Comm( MPI_Comm comm )
   {
      if( comm != MPI_COMM_NULL && comm != MPI_COMM_WORLD && comm != MPI_COMM_SELF )
         throw std::logic_error( "Only predefined communicators (MPI_COMM_WORLD, MPI_COMM_NULL and "
                                 "MPI_COMM_SELF) can be used to initialize this class. Other "
                                 "handles of the MPI_Comm type *cannot* be used to initialize "
                                 "the TNL::MPI::Comm class." );
      wrapper = std::make_shared< Wrapper >( comm );
   }

   //! \brief Factory method – wrapper for `MPI_Comm_dup`
   static Comm
   duplicate( MPI_Comm comm )
   {
#ifdef HAVE_MPI
      MPI_Comm newcomm;
      MPI_Comm_dup( comm, &newcomm );
      return { std::make_shared< Wrapper >( newcomm ) };
#else
      return { std::make_shared< Wrapper >( comm ) };
#endif
   }

   //! \brief Non-static factory method – wrapper for `MPI_Comm_dup`
   Comm
   duplicate() const
   {
      return duplicate( *this );
   }

   //! \brief Factory method – wrapper for `MPI_Comm_split`
   static Comm
   split( MPI_Comm comm, int color, int key )
   {
#ifdef HAVE_MPI
      MPI_Comm newcomm;
      MPI_Comm_split( comm, color, key, &newcomm );
      return { std::make_shared< Wrapper >( newcomm ) };
#else
      return { std::make_shared< Wrapper >( comm ) };
#endif
   }

   //! \brief Non-static factory method – wrapper for `MPI_Comm_split`
   Comm
   split( int color, int key ) const
   {
      return split( *this, color, key );
   }

   //! \brief Factory method – wrapper for `MPI_Comm_split_type`
   static Comm
   split_type( MPI_Comm comm, int split_type, int key, MPI_Info info )
   {
#ifdef HAVE_MPI
      MPI_Comm newcomm;
      MPI_Comm_split_type( comm, split_type, key, info, &newcomm );
      return { std::make_shared< Wrapper >( newcomm ) };
#else
      return { std::make_shared< Wrapper >( comm ) };
#endif
   }

   //! \brief Non-static factory method – wrapper for `MPI_Comm_split_type`
   Comm
   split_type( int split_type, int key, MPI_Info info ) const
   {
      return Comm::split_type( *this, split_type, key, info );
   }

   /**
    * \brief Access the MPI communicator associated with this object.
    *
    * This routine permits the implicit conversion from \ref Comm to
    * `MPI_Comm`.
    *
    * \warning The obtained `MPI_Comm` handle becomes invalid when the
    * originating \ref Comm object is destroyed. For example, the following
    * code is invalid, because the \ref Comm object managing the lifetime of
    * the communicator is destroyed as soon as it is cast to `MPI_Comm`:
    *
    * \code{.cpp}
    * const MPI_Comm comm = MPI::Comm::duplicate( MPI_COMM_WORLD );
    * const int nproc = MPI::GetSize( comm );
    * \endcode
    */
   operator const MPI_Comm&() const
   {
      return wrapper->comm;
   }

   //! \brief Determines the rank of the calling process in the communicator.
   int
   rank() const
   {
      return GetRank( *this );
   }

   //! \brief Returns the size of the group associated with a communicator.
   int
   size() const
   {
      return GetSize( *this );
   }

   //! \brief Compares two communicators – wrapper for `MPI_Comm_compare`.
   int
   compare( MPI_Comm comm2 ) const
   {
#ifdef HAVE_MPI
      int result;
      MPI_Comm_compare( *this, comm2, &result );
      return result;
#else
      return MPI_IDENT;
#endif
   }

   /**
    * \brief Wait for all processes within a communicator to reach the barrier.
    *
    * This routine is a collective operation that blocks each process until all
    * processes have entered it, then releases all of the processes
    * "simultaneously". It is equivalent to calling `MPI_Barrier` with the
    * MPI communicator associated with this object.
    */
   void
   barrier() const
   {
      Barrier( *this );
   }
};

/**
 * \brief Returns a local rank ID of the current process within a group of
 * processes running on a shared-memory node.
 *
 * The given MPI communicator is split into groups according to the
 * `MPI_COMM_TYPE_SHARED` type (from MPI-3) and the rank ID of the process
 * within the group is returned.
 */
inline int
getRankOnNode( MPI_Comm communicator = MPI_COMM_WORLD )
{
#ifdef HAVE_MPI
   const int rank = GetRank( communicator );
   const MPI::Comm local_comm = MPI::Comm::split_type( communicator, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL );
   return local_comm.rank();
#else
   return 0;
#endif
}

}  // namespace MPI
}  // namespace noa::TNL
