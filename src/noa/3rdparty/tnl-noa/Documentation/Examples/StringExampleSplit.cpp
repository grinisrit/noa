#include <iostream>
#include <TNL/String.h>

using namespace TNL;

int main()
{
   String dates("3/4/2005;8/7/2011;11/12/2019");
   std::vector< String > list = dates.split(';');
   std::cout << "list_dates = " << list[0] << ", " << list[1] << ", " << list[2] << std::endl;

   String cars("Subaru,Mazda,,Skoda," );
   std::vector< String > list3 = cars.split(',', String::SplitSkip::SkipEmpty );
   std::cout << "split with String::SkipEmpty = " << list3[0] << ", " << list3[1] << ", " << list3[2] << std::endl;
   std::vector<String> list5 = cars.split(',');
   std::cout << "split without  String::SkipEmpty = " << list5[0] << ", " << list5[1] << ", " << list5[2] << ", " << list5[3] << std::endl;
}
