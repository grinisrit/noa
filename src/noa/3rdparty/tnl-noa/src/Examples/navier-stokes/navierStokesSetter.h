#ifndef SIMPLEPROBLEMTYPESSETTER_H_
#define SIMPLEPROBLEMTYPESSETTER_H_

#include <TNL/Config/ParameterContainer.h>
#include "navierStokesSolver.h"

template< typename MeshType,
          typename SolverStarter >
class navierStokesSetter
{
   public:

   template< typename RealType,
             typename DeviceType,
             typename IndexType >
   static bool run( const Config::ParameterContainer& parameters );
};

template< typename MeshReal, typename Device, typename MeshIndex, typename SolverStarter >
class navierStokesSetter< Meshes::Grid< 2, MeshReal, Device, MeshIndex >, SolverStarter >
{
   public:

   typedef Meshes::Grid< 2, MeshReal, Device, MeshIndex > MeshType;

   template< typename RealType,
             typename DeviceType,
             typename IndexType >
   static bool run( const Config::ParameterContainer& parameters );
};


#include "navierStokesSetter_impl.h"


#endif /* SIMPLEPROBLEMSETTER_H_ */
