// Copyright (c) 2004-2023 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

// Implemented by: Jakub Klinkovský

#pragma once

#ifdef HAVE_HYPRE

   #include <noa/3rdparty/tnl-noa/src/TNL/Containers/HypreVector.h>
   #include <noa/3rdparty/tnl-noa/src/TNL/Containers/DistributedVector.h>

namespace noa::TNL {
namespace Containers {

/**
 * \brief Wrapper for Hypre's parallel vector.
 *
 * Links to upstream sources:
 * - https://github.com/hypre-space/hypre/blob/master/src/parcsr_mv/par_vector.h
 * - https://github.com/hypre-space/hypre/blob/master/src/parcsr_mv/par_vector.c
 * - https://github.com/hypre-space/hypre/blob/master/src/parcsr_mv/_hypre_parcsr_mv.h (catch-all interface)
 *
 * \ingroup Hypre
 */
class HypreParVector
{
public:
   using RealType = HYPRE_Real;
   using ValueType = RealType;
   using DeviceType = HYPRE_Device;
   using IndexType = HYPRE_Int;
   using LocalRangeType = Subrange< IndexType >;

   using VectorType = Containers::DistributedVector< RealType, DeviceType, IndexType >;
   using ViewType = typename VectorType::ViewType;
   using ConstViewType = typename VectorType::ConstViewType;

   using LocalVectorType = Containers::Vector< RealType, DeviceType, IndexType >;
   using LocalViewType = typename LocalVectorType::ViewType;
   using ConstLocalViewType = typename LocalVectorType::ConstViewType;

   using SynchronizerType = typename ViewType::SynchronizerType;

   // default constructor, no underlying \e hypre_ParVector is created.
   HypreParVector() = default;

   // TODO: behavior should depend on "owns_data" (shallow vs deep copy)
   HypreParVector( const HypreParVector& other ) = delete;

   HypreParVector( HypreParVector&& other ) noexcept
   : v( other.v ), owns_handle( other.owns_handle ), localData( std::move( other.localData ) ), ghosts( other.ghosts ),
     synchronizer( std::move( other.synchronizer ) ), valuesPerElement( other.valuesPerElement )
   {
      other.v = nullptr;
   }

   // TODO should do a deep copy
   HypreParVector&
   operator=( const HypreParVector& other ) = delete;

   HypreParVector&
   operator=( HypreParVector&& other ) noexcept
   {
      v = other.v;
      other.v = nullptr;
      owns_handle = other.owns_handle;
      localData = std::move( other.localData );
      ghosts = other.ghosts;
      synchronizer = std::move( other.synchronizer );
      valuesPerElement = other.valuesPerElement;
      return *this;
   }

   //! \brief Initialization by raw data
   HypreParVector( const LocalRangeType& localRange,
                   IndexType ghosts,
                   IndexType globalSize,
                   MPI_Comm communicator,
                   LocalViewType localData )
   {
      bind( localRange, ghosts, globalSize, communicator, localData );
   }

   /**
    * \brief Convert Hypre's format to HypreParVector
    *
    * \param handle is the Hypre vector handle.
    * \param take_ownership indicates if the vector should take ownership of
    * the handle, i.e. whether to call \e hypre_VectorDestroy when it does
    * not need it anymore.
    */
   explicit HypreParVector( hypre_ParVector* handle, bool take_ownership = true )
   {
      bind( handle, take_ownership );
   }

   //! Typecasting to Hypre's \e hypre_ParVector* (which is equivalent to \e HYPRE_ParVector*)
   operator hypre_ParVector*() const
   {
      return v;
   }

   ~HypreParVector()
   {
      reset();
   }

   LocalRangeType
   getLocalRange() const
   {
      if( v == nullptr )
         return {};
      return { hypre_ParVectorPartitioning( v )[ 0 ], hypre_ParVectorPartitioning( v )[ 1 ] };
   }

   IndexType
   getGhosts() const
   {
      return ghosts;
   }

   //! \brief Return the MPI communicator.
   MPI_Comm
   getCommunicator() const
   {
      if( v == nullptr )
         return MPI_COMM_NULL;
      return hypre_ParVectorComm( v );
   }

   //! \brief Returns the global size of the vector.
   HYPRE_Int
   getSize() const
   {
      if( v == nullptr )
         return 0;
      return hypre_ParVectorGlobalSize( v );
   }

   ConstLocalViewType
   getConstLocalView() const
   {
      return { localData.getData(), getLocalRange().getSize() };
   }

   LocalViewType
   getLocalView()
   {
      return { localData.getData(), getLocalRange().getSize() };
   }

   ConstLocalViewType
   getConstLocalViewWithGhosts() const
   {
      return localData.getConstView();
   }

   LocalViewType
   getLocalViewWithGhosts()
   {
      return localData.getView();
   }

   ConstViewType
   getConstView() const
   {
      return { getLocalRange(), ghosts, getSize(), getCommunicator(), getConstLocalViewWithGhosts() };
   }

   ViewType
   getView()
   {
      return { getLocalRange(), ghosts, getSize(), getCommunicator(), getLocalViewWithGhosts() };
   }

   /**
    * \brief Drop previously set data (deallocate if the vector was the owner)
    * and bind to the given data (i.e., the vector does not become the owner).
    */
   void
   bind( const LocalRangeType& localRange,
         IndexType ghosts,
         IndexType globalSize,
         MPI_Comm communicator,
         LocalViewType localData )
   {
      TNL_ASSERT_EQ( localData.getSize(),
                     localRange.getSize() + ghosts,
                     "The local array size does not match the local range of the distributed array." );
      TNL_ASSERT_GE( ghosts, 0, "The ghosts count must be non-negative." );

      // drop/deallocate the current data
      reset();

      // create handle for the vector
      IndexType partitioning[ 2 ];
      partitioning[ 0 ] = localRange.getBegin();
      partitioning[ 1 ] = localRange.getEnd();
      v = hypre_ParVectorCreate( communicator, globalSize, partitioning );
      // hypre_ParVectorMemoryLocation is read-only
      hypre_VectorMemoryLocation( hypre_ParVectorLocalVector( v ) ) = getHypreMemoryLocation();

      // set view data
      this->localData.bind( localData );
      hypre_ParVectorOwnsData( v ) = 0;
      hypre_ParVectorLocalVector( v ) = this->localData;
      hypre_ParVectorActualLocalSize( v ) = this->localData.getSize();

      this->ghosts = ghosts;
   }

   void
   bind( ViewType view )
   {
      bind( view.getLocalRange(), view.getGhosts(), view.getSize(), view.getCommunicator(), view.getLocalViewWithGhosts() );
   }

   void
   bind( VectorType& vector )
   {
      bind( vector.getView() );
   }

   void
   bind( HypreParVector& vector )
   {
      bind( vector.getLocalRange(),
            vector.getGhosts(),
            vector.getSize(),
            vector.getCommunicator(),
            vector.getLocalViewWithGhosts() );
   }

   /**
    * \brief Convert Hypre's format to HypreParVector
    *
    * \param handle is the Hypre vector handle.
    * \param take_ownership indicates if the vector should take ownership of
    * the handle, i.e. whether to call \e hypre_VectorDestroy when it does
    * not need it anymore.
    */
   void
   bind( hypre_ParVector* handle, bool take_ownership = true )
   {
      // drop/deallocate the current data
      if( v != nullptr )
         hypre_ParVectorDestroy( v );

      // set the handle and ownership flag
      v = handle;
      owns_handle = take_ownership;

      // set view data
      localData.bind( hypre_ParVectorLocalVector( v ) );
      hypre_ParVectorOwnsData( v ) = 0;
      hypre_ParVectorLocalVector( v ) = this->localData;
      hypre_ParVectorActualLocalSize( v ) = this->localData.getSize();

      const HYPRE_Int actual_size = hypre_ParVectorActualLocalSize( v );
      const HYPRE_Int local_size = hypre_ParVectorPartitioning( v )[ 1 ] - hypre_ParVectorPartitioning( v )[ 0 ];
      this->ghosts = actual_size - local_size;
   }

   //! \brief Reset the vector to empty state.
   void
   reset()
   {
      // drop/deallocate the current data
      if( owns_handle && v != nullptr ) {
         hypre_ParVectorDestroy( v );
         v = nullptr;
      }
      else
         v = nullptr;
      owns_handle = true;

      localData.reset();
      ghosts = 0;
      synchronizer = nullptr;
      valuesPerElement = 1;
   }

   void
   setDistribution( LocalRangeType localRange, IndexType ghosts, IndexType globalSize, const MPI::Comm& communicator )
   {
      TNL_ASSERT_LE( localRange.getEnd(), globalSize, "end of the local range is outside of the global range" );

      // drop/deallocate the current data
      reset();

      // create handle for the vector
      IndexType partitioning[ 2 ];
      partitioning[ 0 ] = localRange.getBegin();
      partitioning[ 1 ] = localRange.getEnd();
      v = hypre_ParVectorCreate( communicator, globalSize, partitioning );
      // hypre_ParVectorMemoryLocation is read-only
      hypre_VectorMemoryLocation( hypre_ParVectorLocalVector( v ) ) = getHypreMemoryLocation();

      // set view data
      this->localData.setSize( localRange.getSize() + ghosts );
      hypre_ParVectorOwnsData( v ) = 0;
      hypre_ParVectorLocalVector( v ) = this->localData;
      hypre_ParVectorActualLocalSize( v ) = this->localData.getSize();

      this->ghosts = ghosts;
   }

   //! \brief Set all elements of the vector to \e value.
   void
   setValue( RealType value )
   {
      if( v != nullptr )
         hypre_ParVectorSetConstantValues( v, value );
   }

   // synchronizer stuff
   void
   setSynchronizer( std::shared_ptr< SynchronizerType > synchronizer, int valuesPerElement = 1 )
   {
      this->synchronizer = std::move( synchronizer );
      this->valuesPerElement = valuesPerElement;
   }

   std::shared_ptr< SynchronizerType >
   getSynchronizer() const
   {
      return synchronizer;
   }

   int
   getValuesPerElement() const
   {
      return valuesPerElement;
   }

   // Note that this method is not thread-safe - only the thread which created
   // and "owns" the instance of this object can call this method.
   void
   startSynchronization()
   {
      if( ghosts == 0 )
         return;
      // TODO: assert does not play very nice with automatic synchronizations from operations like
      //       assignment of scalars
      // (Maybe we should just drop all automatic syncs? But that's not nice for high-level codes
      // like linear solvers...)
      TNL_ASSERT_TRUE( synchronizer, "the synchronizer was not set" );

      typename SynchronizerType::ByteArrayView bytes;
      bytes.bind( reinterpret_cast< std::uint8_t* >( localData.getData() ), sizeof( ValueType ) * localData.getSize() );
      synchronizer->synchronizeByteArrayAsync( bytes, sizeof( ValueType ) * valuesPerElement );
   }

   void
   waitForSynchronization() const
   {
      if( synchronizer && synchronizer->async_op.valid() ) {
         synchronizer->async_wait_timer.start();
         synchronizer->async_op.wait();
         synchronizer->async_wait_timer.stop();
      }
   }

protected:
   hypre_ParVector* v = nullptr;
   bool owns_handle = true;
   HypreVector localData;
   HYPRE_Int ghosts = 0;

   std::shared_ptr< SynchronizerType > synchronizer = nullptr;
   int valuesPerElement = 1;
};

}  // namespace Containers
}  // namespace noa::TNL

#endif  // HAVE_HYPRE
