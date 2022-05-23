#ifndef NAVIERSTOKESSOLVERMONITOR_H_
#define NAVIERSTOKESSOLVERMONITOR_H_

#include <TNL/Solvers/ODE/ODESolverMonitor.h>

template< typename Real, typename Index >
class navierStokesSolverMonitor : public ODESolverMonitor< Real, Index >
{
   public:

   navierStokesSolverMonitor();

   void refresh();

   Real uMax, uAvg, rhoMax, rhoAvg, rhoUMax, rhoUAvg, eMax, eAvg;

};

#include "navierStokesSolverMonitor_impl.h"

#endif /* NAVIERSTOKESSOLVERMONITOR_H_ */
