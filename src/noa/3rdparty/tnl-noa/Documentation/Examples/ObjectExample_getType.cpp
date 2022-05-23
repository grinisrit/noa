#include <iostream>
#include <TNL/TypeInfo.h>
#include <TNL/Object.h>
#include <TNL/Devices/Host.h>
#include <TNL/Devices/Cuda.h>

using namespace TNL;

template< typename Value,
          typename Device >
class MyArray : public Object
{
   public:

      static std::string getSerializationType()
      {
         return "MyArray< " + TNL::getType< Value >() + ", " + getType< Devices::Host >() + " >";
      }

      virtual std::string getSerializationTypeVirtual() const override
      {
         return getSerializationType();
      }
};

int main()
{
   using HostArray = MyArray< int, Devices::Host >;
   using CudaArray = MyArray< int, Devices::Cuda >;

   HostArray hostArray;
   CudaArray cudaArray;
   Object* hostArrayPtr = &hostArray;
   Object* cudaArrayPtr = &cudaArray;

   // Object types
   std::cout << "HostArray type is                  " << getType< HostArray >() << std::endl;
   std::cout << "hostArrayPtr type is               " << getType( *hostArrayPtr ) << std::endl;

   std::cout << "CudaArray type is                  " << getType< CudaArray >() << std::endl;
   std::cout << "cudaArrayPtr type is               " << getType( *cudaArrayPtr ) << std::endl;

   // Object serialization types
   std::cout << "HostArray serialization type is    " << HostArray::getSerializationType() << std::endl;
   std::cout << "hostArrayPtr serialization type is " << hostArrayPtr->getSerializationTypeVirtual() << std::endl;

   std::cout << "CudaArray serialization type is    " << CudaArray::getSerializationType() << std::endl;
   std::cout << "cudaArrayPtr serialization type is " << cudaArrayPtr->getSerializationTypeVirtual() << std::endl;
}
