#ifndef LaxFridrichs3D_H
#define LaxFridrichs3D_H

#include <TNL/Containers/Vector.h>
#include <TNL/Meshes/Grid.h>
#include "LaxFridrichsContinuity.h"
#include "LaxFridrichsEnergy.h"
#include "LaxFridrichsMomentumX.h"
#include "LaxFridrichsMomentumY.h"
#include "LaxFridrichsMomentumZ.h"
#include "EulerPressureGetter.h"
#include "Euler2DVelXGetter.h"
#include "EulerVelGetter.h"

namespace TNL {

template< typename Mesh,
          typename Real = typename Mesh::RealType,
          typename Index = typename Mesh::IndexType >
class LaxFridrichs3D
{
   public:
      typedef Real RealType;
      typedef typename Mesh::DeviceType DeviceType;
      typedef Index IndexType;
      typedef Functions::MeshFunctionView< Mesh > MeshFunctionType;
 
      typedef LaxFridrichsContinuity< Mesh, Real, Index > Continuity;
      typedef LaxFridrichsMomentumX< Mesh, Real, Index > MomentumX;
      typedef LaxFridrichsMomentumY< Mesh, Real, Index > MomentumY;
      typedef LaxFridrichsMomentumZ< Mesh, Real, Index > MomentumZ;
      typedef LaxFridrichsEnergy< Mesh, Real, Index > Energy;
      typedef EulerVelXGetter< Mesh, Real, Index > VelocityX;
      typedef EulerVelGetter< Mesh, Real, Index > Velocity;
      typedef EulerPressureGetter< Mesh, Real, Index > Pressure;
   
};

} //namespace TNL

#endif	/* LaxFridrichs3D_H */
