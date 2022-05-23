#pragma once

#include <math.h>
#include <TNL/Config/ConfigDescription.h>
#include <TNL/Solvers/ODE/ExplicitSolver.h>
#include <TNL/Config/ParameterContainer.h>
#include <TNL/Timer.h>

#include "../BLAS/CommonVectorOperations.h"

namespace TNL {
namespace Benchmarks {

template< typename Problem,
          typename SolverMonitor = Solvers::IterativeSolverMonitor< typename Problem::RealType, typename Problem::IndexType > >
class Euler : public Solvers::ODE::ExplicitSolver< Problem, SolverMonitor >
{
   public:
   using ProblemType = Problem;
   using DofVectorType = typename Problem::DofVectorType;
   using RealType = typename Problem::RealType;
   using DeviceType = typename Problem::DeviceType;
   using IndexType = typename Problem::IndexType;
   using DofVectorPointer = Pointers::SharedPointer< DofVectorType, DeviceType >;
   using VectorOperations = CommonVectorOperations< DeviceType >;


   Euler();

   static void configSetup( Config::ConfigDescription& config,
                            const String& prefix = "" );

   bool setup( const Config::ParameterContainer& parameters,
              const String& prefix = "" );

   void setCFLCondition( const RealType& cfl );

   const RealType& getCFLCondition() const;

   bool solve( DofVectorPointer& u );

   protected:
   void computeNewTimeLevel( DofVectorPointer& u,
                             RealType tau,
                             RealType& currentResidue );

   
   DofVectorPointer k1;

   RealType cflCondition;
 
   //Timer timer, updateTimer;
};

} // namespace Benchmarks
} // namespace TNL

#include "Euler.hpp"
