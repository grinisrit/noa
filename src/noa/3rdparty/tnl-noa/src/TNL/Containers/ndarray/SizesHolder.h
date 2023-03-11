// Copyright (c) 2004-2023 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

// Implemented by: Jakub Klinkovsky

#pragma once

#include <noa/3rdparty/tnl-noa/src/TNL/Assert.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Cuda/CudaCallable.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Algorithms/staticFor.h>

#include <noa/3rdparty/tnl-noa/src/TNL/Containers/ndarray/Meta.h>

namespace noa::TNL {
namespace Containers {

namespace detail {

template< typename Index, typename LevelTag, std::size_t size >
class SizeHolder
{
public:
   __cuda_callable__
   constexpr Index
   getSize( LevelTag ) const
   {
      return size;
   }

   __cuda_callable__
   void
   setSize( LevelTag, Index newSize )
   {
      TNL_ASSERT_EQ( newSize, 0, "Dynamic size for a static dimension must be 0." );
   }

   __cuda_callable__
   bool
   operator==( const SizeHolder& ) const
   {
      return true;
   }
};

template< typename Index, typename LevelTag >
class SizeHolder< Index, LevelTag, 0 >
{
public:
   __cuda_callable__
   Index
   getSize( LevelTag ) const
   {
      return size;
   }

   __cuda_callable__
   void
   setSize( LevelTag, Index size )
   {
      this->size = size;
   }

   __cuda_callable__
   bool
   operator==( const SizeHolder& other ) const
   {
      return size == other.size;
   }

private:
   Index size = 0;
};

template< typename Index, std::size_t currentSize, std::size_t... otherSizes >
class SizesHolderLayer
// Doxygen complains about recursive classes
//! \cond
: public SizesHolderLayer< Index, otherSizes... >,
  public SizeHolder< Index,
                     IndexTag< sizeof...( otherSizes ) >,  // LevelTag
                     currentSize >
//! \endcond
{
   using BaseType = SizesHolderLayer< Index, otherSizes... >;
   using Layer = SizeHolder< Index,
                             IndexTag< sizeof...( otherSizes ) >,  // LevelTag
                             currentSize >;

protected:
   using BaseType::getSize;
   using BaseType::setSize;
   using Layer::getSize;
   using Layer::setSize;

   __cuda_callable__
   bool
   operator==( const SizesHolderLayer& other ) const
   {
      return BaseType::operator==( other ) && Layer::operator==( other );
   }
};

// specializations to terminate the recursive inheritance
template< typename Index, std::size_t currentSize >
class SizesHolderLayer< Index, currentSize > : public SizeHolder< Index,
                                                                  IndexTag< 0 >,  // LevelTag
                                                                  currentSize >
{
   using Layer = SizeHolder< Index,
                             IndexTag< 0 >,  // LevelTag
                             currentSize >;

protected:
   using Layer::getSize;
   using Layer::setSize;

   __cuda_callable__
   bool
   operator==( const SizesHolderLayer& other ) const
   {
      return Layer::operator==( other );
   }
};

}  // namespace detail

/**
 * \brief Holds static and dynamic sizes of an N-dimensional array.
 *
 * The dimension of the array and static sizes are specified as
 * \ref std::size_t, the type of dynamic sizes is configurable with \e Index.
 *
 * \tparam Index Integral type used for storing dynamic sizes.
 * \tparam sizes Sequence of integers specifying static and dynamic sizes. The
 *         number of integers in the sequence specifies the dimension of the
 *         array.  Positive values specify static sizes, zeros specify dynamic
 *         sizes that must be set at run-time via \ref setSize.
 *
 * \ingroup ndarray
 */
template< typename Index, std::size_t... sizes >
class SizesHolder : public detail::SizesHolderLayer< Index, sizes... >
{
   using BaseType = detail::SizesHolderLayer< Index, sizes... >;

public:
   using IndexType = Index;

   //! \brief Default constructor.
   SizesHolder() = default;

   //! \brief Constructs the holder from given pack of sizes.
   template< typename... Indices, std::enable_if_t< sizeof...( Indices ) == sizeof...( sizes ), bool > = true >
   explicit SizesHolder( Indices... _sizes )
   {
      Algorithms::staticFor< std::size_t, 0, getDimension() >(
         [ & ]( auto i )
         {
            setSize< i >( detail::get_from_pack< i >( _sizes... ) );
         } );
   }

   //! \brief Returns the dimension of the array, i.e. number of \e sizes
   //! specified in the template parameters.
   static constexpr std::size_t
   getDimension()
   {
      return sizeof...( sizes );
   }

   //! \brief Returns the _static_ size of a specific dimension.
   template< std::size_t level >
   static constexpr std::size_t
   getStaticSize()
   {
      static_assert( level < sizeof...( sizes ), "Invalid dimension passed to getStaticSize()." );
      return detail::get_from_pack< level >( sizes... );
   }

   //! \brief Returns the _dynamic_ size along a specific axis.
   template< std::size_t level >
   __cuda_callable__
   Index
   getSize() const
   {
      static_assert( level < sizeof...( sizes ), "Invalid level passed to getSize()." );
      return BaseType::getSize( detail::IndexTag< getDimension() - level - 1 >() );
   }

   //! \brief Sets the _dynamic_ size along a specific axis.
   template< std::size_t level >
   __cuda_callable__
   void
   setSize( Index size )
   {
      static_assert( level < sizeof...( sizes ), "Invalid level passed to setSize()." );
      BaseType::setSize( detail::IndexTag< getDimension() - level - 1 >(), size );
   }

   //! \brief Compares the sizes with another instance of the holder.
   __cuda_callable__
   bool
   operator==( const SizesHolder& other ) const
   {
      return BaseType::operator==( other );
   }

   //! \brief Compares the sizes with another instance of the holder.
   __cuda_callable__
   bool
   operator!=( const SizesHolder& other ) const
   {
      return ! operator==( other );
   }
};

/**
 * \brief Combines the sizes of two instance of \ref SizesHolder with the operator `+`.
 *
 * \ingroup ndarray
 */
template< typename Index, std::size_t... sizes, typename OtherHolder >
SizesHolder< Index, sizes... >
operator+( const SizesHolder< Index, sizes... >& lhs, const OtherHolder& rhs )
{
   SizesHolder< Index, sizes... > result;
   Algorithms::staticFor< std::size_t, 0, sizeof...( sizes ) >(
      [ &result, &lhs, &rhs ]( auto level )
      {
         if( result.template getStaticSize< level >() == 0 )
            result.template setSize< level >( lhs.template getSize< level >() + rhs.template getSize< level >() );
      } );
   return result;
}

/**
 * \brief Combines the sizes of two instance of \ref SizesHolder with the operator `-`.
 *
 * \ingroup ndarray
 */
template< typename Index, std::size_t... sizes, typename OtherHolder >
SizesHolder< Index, sizes... >
operator-( const SizesHolder< Index, sizes... >& lhs, const OtherHolder& rhs )
{
   SizesHolder< Index, sizes... > result;
   Algorithms::staticFor< std::size_t, 0, sizeof...( sizes ) >(
      [ &result, &lhs, &rhs ]( auto level )
      {
         if( result.template getStaticSize< level >() == 0 )
            result.template setSize< level >( lhs.template getSize< level >() - rhs.template getSize< level >() );
      } );
   return result;
}

/**
 * \brief Prints the sizes contained in an instance of \ref SizesHolder to the
 * given output stream.
 *
 * \ingroup ndarray
 */
template< typename Index, std::size_t... sizes >
std::ostream&
operator<<( std::ostream& str, const SizesHolder< Index, sizes... >& holder )
{
   str << "SizesHolder< ";
   Algorithms::staticFor< std::size_t, 0, sizeof...( sizes ) - 1 >(
      [ &str, &holder ]( auto dimension )
      {
         str << holder.template getStaticSize< dimension >() << ", ";
      } );
   str << holder.template getStaticSize< sizeof...( sizes ) - 1 >() << " >( ";
   Algorithms::staticFor< std::size_t, 0, sizeof...( sizes ) - 1 >(
      [ &str, &holder ]( auto dimension )
      {
         str << holder.template getSize< dimension >() << ", ";
      } );
   str << holder.template getSize< sizeof...( sizes ) - 1 >() << " )";
   return str;
}

namespace detail {

// helper for the methods forAll, forInterior, etc.
template< typename Index, std::size_t dimension, Index constSize >
class ConstStaticSizesHolder
{
public:
   using IndexType = Index;

   static constexpr std::size_t
   getDimension()
   {
      return dimension;
   }

   template< std::size_t level >
   static constexpr std::size_t
   getStaticSize()
   {
      static_assert( level < getDimension(), "Invalid level passed to getStaticSize()." );
      return constSize;
   }

   template< std::size_t level >
   __cuda_callable__
   Index
   getSize() const
   {
      static_assert( level < getDimension(), "Invalid dimension passed to getSize()." );
      return constSize;
   }

   // methods for convenience
   __cuda_callable__
   bool
   operator==( const ConstStaticSizesHolder& other ) const
   {
      return true;
   }

   __cuda_callable__
   bool
   operator!=( const ConstStaticSizesHolder& other ) const
   {
      return false;
   }
};

// helper for the forInterior method
template< typename SizesHolder, std::size_t ConstValue >
struct SubtractedSizesHolder
{};

template< typename Index, std::size_t ConstValue, std::size_t... sizes >
struct SubtractedSizesHolder< SizesHolder< Index, sizes... >, ConstValue >
{
   //   using type = SizesHolder< Index, std::max( (std::size_t) 0, sizes - ConstValue )... >;
   using type = SizesHolder< Index, ( ( sizes >= ConstValue ) ? sizes - ConstValue : 0 )... >;
};

// wrapper for localBegins in DistributedNDArray (static sizes cannot be distributed, begins are always 0)
template< typename SizesHolder,
          // overridable value is useful in the forInterior method
          std::size_t ConstValue = 0 >
struct LocalBeginsHolder : public SizesHolder
{
   template< std::size_t dimension >
   static constexpr std::size_t
   getStaticSize()
   {
      static_assert( dimension < SizesHolder::getDimension(), "Invalid dimension passed to getStaticSize()." );
      return ConstValue;
   }

   template< std::size_t level >
   __cuda_callable__
   typename SizesHolder::IndexType
   getSize() const
   {
      if( SizesHolder::template getStaticSize< level >() != 0 )
         return ConstValue;
      return SizesHolder::template getSize< level >();
   }

   template< std::size_t level >
   __cuda_callable__
   void
   setSize( typename SizesHolder::IndexType newSize )
   {
      if( SizesHolder::template getStaticSize< level >() == 0 )
         SizesHolder::template setSize< level >( newSize );
      else
         TNL_ASSERT_EQ( newSize,
                        (typename SizesHolder::IndexType) ConstValue,
                        "Dynamic size for a static dimension must be equal to the specified ConstValue." );
   }
};

template< typename SizesHolder, std::size_t ConstValue, typename OtherHolder >
LocalBeginsHolder< SizesHolder, ConstValue >
operator+( const LocalBeginsHolder< SizesHolder, ConstValue >& lhs, const OtherHolder& rhs )
{
   LocalBeginsHolder< SizesHolder, ConstValue > result;
   Algorithms::staticFor< std::size_t, 0, SizesHolder::getDimension() >(
      [ &result, &lhs, &rhs ]( auto level )
      {
         if( SizesHolder::template getStaticSize< level >() == 0 )
            result.template setSize< level >( lhs.template getSize< level >() + rhs.template getSize< level >() );
      } );
   return result;
}

template< typename SizesHolder, std::size_t ConstValue, typename OtherHolder >
LocalBeginsHolder< SizesHolder, ConstValue >
operator-( const LocalBeginsHolder< SizesHolder, ConstValue >& lhs, const OtherHolder& rhs )
{
   LocalBeginsHolder< SizesHolder, ConstValue > result;
   Algorithms::staticFor< std::size_t, 0, SizesHolder::getDimension() >(
      [ &result, &lhs, &rhs ]( auto level )
      {
         if( SizesHolder::template getStaticSize< level >() == 0 )
            result.template setSize< level >( lhs.template getSize< level >() - rhs.template getSize< level >() );
      } );
   return result;
}

template< typename SizesHolder, std::size_t ConstValue >
std::ostream&
operator<<( std::ostream& str, const LocalBeginsHolder< SizesHolder, ConstValue >& holder )
{
   str << "LocalBeginsHolder< SizesHolder< ";
   Algorithms::staticFor< std::size_t, 0, SizesHolder::getDimension() - 1 >(
      [ &str, &holder ]( auto dimension )
      {
         str << holder.template getStaticSize< dimension >() << ", ";
      } );
   str << holder.template getStaticSize< SizesHolder::getDimension() - 1 >() << " >, ";
   str << ConstValue << " >( ";
   Algorithms::staticFor< std::size_t, 0, SizesHolder::getDimension() - 1 >(
      [ &str, &holder ]( auto dimension )
      {
         str << holder.template getSize< dimension >() << ", ";
      } );
   str << holder.template getSize< SizesHolder::getDimension() - 1 >() << " )";
   return str;
}

}  // namespace detail

}  // namespace Containers
}  // namespace noa::TNL
