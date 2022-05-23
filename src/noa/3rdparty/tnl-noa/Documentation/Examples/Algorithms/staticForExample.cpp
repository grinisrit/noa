#include <iostream>
#include <array>
#include <tuple>
#include <TNL/Algorithms/staticFor.h>

/*
 * Example function printing members of std::tuple using staticFor
 * using lambda with capture.
 */
template< typename... Ts >
void printTuple( const std::tuple<Ts...>& tupleVar )
{
   std::cout << "{ ";
   TNL::Algorithms::staticFor<size_t, 0, sizeof... (Ts)>( [&](auto i) {
      std::cout << std::get<i>(tupleVar);
      if( i < sizeof... (Ts) - 1 )
         std::cout << ", ";
   });
   std::cout << " }" << std::endl;
}

struct TuplePrinter
{
   constexpr TuplePrinter() = default;

   template< typename Index, typename... Ts >
   void operator()( Index i, const std::tuple<Ts...>& tupleVar )
   {
      std::cout << std::get<i>( tupleVar );
      if( i < sizeof... (Ts) - 1 )
         std::cout << ", ";
   }
};

/*
 * Example function printing members of std::tuple using staticFor
 * and a structure with templated operator().
 */
template< typename... Ts >
void printTupleCallableStruct( const std::tuple<Ts...>& tupleVar )
{
   std::cout << "{ ";
   TNL::Algorithms::staticFor< size_t, 0, sizeof... (Ts) >( TuplePrinter(), tupleVar );
   std::cout << " }" << std::endl;
}


int main( int argc, char* argv[] )
{
   // initiate std::array
   std::array< int, 5 > a{ 1, 2, 3, 4, 5 };

   // print out the array using template parameters for indexing
   TNL::Algorithms::staticFor< int, 0, 5 >(
      [&a] ( auto i ) {
         std::cout << "a[ " << i << " ] = " << std::get< i >( a ) << std::endl;
      }
   );

   // example of printing a tuple using staticFor and a lambda function
   printTuple( std::make_tuple( "Hello", 3, 2.1 ) );
   // example of printing a tuple using staticFor and a structure with templated operator()
   printTupleCallableStruct( std::make_tuple( "Hello", 3, 2.1 ) );
}
