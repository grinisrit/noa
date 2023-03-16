#ifndef TNLNAVIERSTOKESSOLVERMONITOR_IMPL_H_
#define TNLNAVIERSTOKESSOLVERMONITOR_IMPL_H_

#include <fstream>

using namespace std;

template< typename Real, typename Index >
navierStokesSolverMonitor< Real, Index > :: navierStokesSolverMonitor()
{
}

template< typename Real, typename Index >
void navierStokesSolverMonitor< Real, Index > :: refresh()
{
   if( this->verbose > 0 && this->refresRate % this->refreshRate == 0 )
   {
     std::cout << "V=( " << uMax
           << " , " << uAvg
           << " ) E=( " << eMax
           << ", " << eAvg << " ) ";
   }
   ODESolverMonitor< Real, Index > :: refresh();
}

#endif /* TNLNAVIERSTOKESSOLVERMONITOR_IMPL_H_ */
