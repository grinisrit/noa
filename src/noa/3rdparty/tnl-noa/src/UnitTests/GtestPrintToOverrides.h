#pragma once

// Overrides due to GTest's fuckup...
// https://stackoverflow.com/a/25265174

#include <TNL/Containers/Vector.h>
#include <TNL/Containers/VectorView.h>
#include <TNL/Containers/StaticVector.h>

namespace TNL {
namespace Containers {

template< typename Value, typename Device, typename Index, typename Allocator >
void PrintTo( const Vector< Value, Device, Index, Allocator >& vec,
              std::ostream *str )
{
   *str << vec;
}

template< typename Value, typename Device, typename Index >
void PrintTo( const VectorView< Value, Device, Index >& vec,
              std::ostream *str )
{
   *str << vec;
}

template< int Size, typename Value >
void PrintTo( const StaticVector< Size, Value >& vec,
              std::ostream *str )
{
   *str << vec;
}

} // namespace Containers
} // namespace TNL
