// Implemented by: Tomas Oberhuber

#pragma once

#include <TNL/Pointers/SharedPointer.h>

namespace TNL {
   namespace Benchmarks {
      namespace Traversers {

template< typename MeshFunction >
class BenchmarkTraverserUserData
{
   public:

      using MeshType = typename MeshFunction::MeshType;
      using RealType = typename MeshType::RealType;
      using DeviceType = typename MeshType::DeviceType;
      using MeshFunctionPointer = Pointers::SharedPointer< MeshFunction >;

      BenchmarkTraverserUserData( MeshFunctionPointer& f )
         : u( &f.template modifyData< DeviceType >() ), data( f->getData().getData() ){}

      MeshFunction* u;
      RealType* data;
};


      } // namespace Traversers
   } // namespace Benchmarks
} // namespace TNL
