#pragma once

#include <math.h>
#include <TNL/Config/ConfigDescription.h>
#include <TNL/Solvers/ODE/ExplicitSolver.h>

#include "../BLAS/CommonVectorOperations.h"

namespace TNL {
namespace Benchmarks {

template< class Problem,
          typename SolverMonitor = Solvers::IterativeSolverMonitor< typename Problem::RealType, typename Problem::IndexType > >
class Merson : public Solvers::ODE::ExplicitSolver< Problem, SolverMonitor >
{
   public:
   using ProblemType = Problem;
   using DofVectorType = typename Problem::DofVectorType;
   using RealType = typename Problem::RealType;
   using DeviceType = typename Problem::DeviceType;
   using IndexType = typename Problem::IndexType;
   using DofVectorPointer = Pointers::SharedPointer< DofVectorType, DeviceType >;
   using VectorOperations = CommonVectorOperations< DeviceType >;
   
   Merson();

   static void configSetup( Config::ConfigDescription& config,
                            const String& prefix = "" );

   bool setup( const Config::ParameterContainer& parameters,
              const String& prefix = "" );

   void setAdaptivity( const RealType& a );

   bool solve( DofVectorPointer& u );

   protected:
 
   //! Compute the Runge-Kutta coefficients
   /****
    * The parameter u is not constant because one often
    * needs to correct u on the boundaries to be able to compute
    * the RHS.
    */
   void computeKFunctions( DofVectorPointer& u,
                           const RealType& time,
                           RealType tau );

   RealType computeError( const RealType tau );

   void computeNewTimeLevel( const RealType time,
                             const RealType tau,
                             DofVectorPointer& u,
                             RealType& currentResidue );

   void writeGrids( const DofVectorPointer& u );

   DofVectorPointer k1, k2, k3, k4, k5, kAux;

   /****
    * This controls the accuracy of the solver
    */
   RealType adaptivity;
   
   Containers::Vector< RealType, DeviceType, IndexType > openMPErrorEstimateBuffer;
};

} // namespace Benchmarks
} // namespace TNL

#include "Merson.hpp"
