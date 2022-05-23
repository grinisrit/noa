#include <iostream>
#include <TNL/Containers/Array.h>
#include <list>
#include <vector>

using namespace TNL;
using namespace TNL::Containers;

int main( int argc, char* argv[] )
{
   /****
    * Create one array on host and one array on device.
    */
   Array< int > host_array( 10 );
   Array< int, Devices::Cuda > device_array;

   /***
    * Initiate the host array with number three and assign it to the device array.
    * NOTE: Of course, you may do directly 'device_array = 3' as well.
    */
   host_array = 3;
   device_array = host_array;

   /****
    * Print both arrays.
    */
   std::cout << "host_array = " << host_array << std::endl;
   std::cout << "device_array = " << device_array << std::endl;
   std::cout << std::endl;

   /****
    * There are few other ways how to initialize arrays...
    */
   std::list< int > list { 1, 2, 3, 4, 5 };
   std::vector< int > vector { 6, 7, 8, 9, 10 };

   Array< int, Devices::Cuda > device_array_list( list );
   Array< int, Devices::Cuda > device_array_vector( vector );
   Array< int, Devices::Cuda > device_array_init_list { 11, 12, 13, 14, 15 };

   /****
    * ... and print them all
    */
   std::cout << "device_array_list = " << device_array_list << std::endl;
   std::cout << "device_array_vector = " << device_array_vector << std::endl;
   std::cout << "device_array_init_list = " << device_array_init_list << std::endl;
}
