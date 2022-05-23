#pragma once

#include <TNL/Solvers/ODE/Euler.h>

template< typename DofVector >
using ODETestSolver = TNL::Solvers::ODE::Euler< DofVector >;

#include "ODESolverTest.h"
