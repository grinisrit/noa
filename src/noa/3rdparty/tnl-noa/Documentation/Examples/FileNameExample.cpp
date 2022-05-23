#include <iostream>
#include <TNL/FileName.h>

using namespace TNL;

int main()
{
   /***
    * Create file name with filename base 'velocity' and extension 'vtk'.
    */
   FileName fileName( "velocity-", "vtk" );

   /**
    * Set the number of digits for the index to 2 and print file names for
    * indexes 0 to 10.
    */
   fileName.setDigitsCount( 2 );
   for( int i = 0; i <= 10; i ++ )
   {
      fileName.setIndex( i );
      std::cout << fileName.getFileName() << std::endl;
   }

   /***
    * Now set the number if index digits to 3 and do the same.
    */
   fileName.setDigitsCount( 3 );
   for( int i = 0; i <= 10; i ++ )
   {
      fileName.setIndex( i );
      std::cout << fileName.getFileName() << std::endl;
   }
}
