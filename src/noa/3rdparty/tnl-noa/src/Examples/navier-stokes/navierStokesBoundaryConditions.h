#ifndef NAVIERSTOKESBOUNDARYCONDITIONS_H_
#define NAVIERSTOKESBOUNDARYCONDITIONS_H_

#include <TNL/Config/ParameterContainer.h>

template< typename Mesh >
class navierStokesBoundaryConditions
{
   public:

   typedef Mesh MeshType;
   typedef typename Mesh::RealType RealType;
   typedef typename Mesh::DeviceType DeviceType;
   typedef typename Mesh::IndexType IndexType;

   navierStokesBoundaryConditions();

   bool setup( const Config::ParameterContainer& parameters );

   void setMesh( const MeshType& mesh );

   template< typename Vector >
   void apply( const RealType& time,
               const RealType& tau,
               Vector& rho,
               Vector& u1,
               Vector& u2,
               Vector& temperature );

   protected:

   const MeshType* mesh;

   RealType maxInflowVelocity, startUp, T, R, p0, gamma;
};

#include "navierStokesBoundaryConditions_impl.h"

#endif /* NAVIERSTOKESBOUNDARYCONDITIONS_H_ */
