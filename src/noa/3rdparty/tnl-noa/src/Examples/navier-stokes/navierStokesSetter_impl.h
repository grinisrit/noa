#ifndef NAVIERSTOKESSETTER_IMPL_H_
#define NAVIERSTOKESSETTER_IMPL_H_

#include <TNL/Meshes/Grid.h>
#include <TNL/Meshes/tnlLinearGridGeometry.h>
#include <TNL/Operators/euler/fvm/LaxFridrichs.h>
#include <TNL/Operators/gradient/tnlCentralFDMGradient.h>

template< typename MeshType, typename SolverStarter >
   template< typename RealType,
             typename DeviceType,
             typename IndexType >
bool navierStokesSetter< MeshType, SolverStarter > :: run( const Config::ParameterContainer& parameters )
{
   std::cerr << "The solver is not implemented for the mesh " << getType< MeshType >() << "." << std::endl;
   return false;
}

template< typename MeshReal, typename Device, typename MeshIndex, typename SolverStarter >
template< typename RealType,
          typename DeviceType,
          typename IndexType >
bool navierStokesSetter< Meshes::Grid< 2, MeshReal, Device, MeshIndex >, SolverStarter >::run( const Config::ParameterContainer& parameters )
{
   SolverStarter solverStarter;
   const String& schemeName = parameters. getParameter< String >( "scheme" );
   if( schemeName == "lax-fridrichs" )
      return solverStarter. run< navierStokesSolver< MeshType,
                                                     LaxFridrichs< MeshType,
                                                                     tnlCentralFDMGradient< MeshType > > > >
                                                     ( parameters );
};

#endif /* NAVIERSTOKESSETTER_IMPL_H_ */
