#pragma once

#include <TNL/Solvers/ODE/Merson.h>

template< typename DofVector >
using ODETestSolver = TNL::Solvers::ODE::Merson< DofVector >;

#include "ODESolverTest.h"
