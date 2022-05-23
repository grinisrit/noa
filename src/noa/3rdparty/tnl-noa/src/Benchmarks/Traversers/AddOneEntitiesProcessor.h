// Implemented by: Tomas Oberhuber

#pragma once

#include <TNL/Devices/Cuda.h>

namespace TNL {
   namespace Benchmarks {
      namespace Traversers {

template< typename TraverserUserData >
class AddOneEntitiesProcessor
{
   public:
      
      using MeshType = typename TraverserUserData::MeshType;
      using DeviceType = typename MeshType::DeviceType;
      using RealType = typename MeshType::RealType;

      template< typename GridEntity >
      __cuda_callable__
      static inline void processEntity( const MeshType& mesh,
                                        TraverserUserData& userData,
                                        const GridEntity& entity )
      {
         auto& u = *userData.u;
         u( entity ) += ( RealType ) 1.0;
      }
};

      } // namespace Traversers
   } // namespace Benchmarks
} // namespace TNL
