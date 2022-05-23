#include <iostream>
#include <TNL/FileName.h>
#include <TNL/Containers/StaticVector.h>

using namespace TNL;

int main()
{
   /**
    * Create file name with filename base 'velocity' and extension 'vtk'.
    */
   FileName fileName( "velocity-", "vtk" );

   /***
    * Set the distributed system node ID to 0-0-0.
    */
   using CoordinatesType = Containers::StaticVector< 3, int >;
   CoordinatesType coordinates( 0, 0, 0 );
   fileName.setDistributedSystemNodeCoordinates( coordinates );

   /**
    * Now set the file name index digits count to 2 and print file names
    * for indexes 0 to 10.
    */
   fileName.setDigitsCount( 2 );
   for( int i = 0; i <= 10; i ++ )
   {
      fileName.setIndex( i );
      std::cout << fileName.getFileName() << std::endl;
   }
}
