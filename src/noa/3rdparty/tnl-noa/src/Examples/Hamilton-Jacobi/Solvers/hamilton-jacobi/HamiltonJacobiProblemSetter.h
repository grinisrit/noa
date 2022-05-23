#pragma once 

#include <TNL/Config/ParameterContainer.h>
#include "HamiltonJacobiProblem.h"

template< typename RealType,
		  typename DeviceType,
		  typename IndexType,
		  typename MeshType,
		  typename ConfigTag,
          typename SolverStarter >
class HamiltonJacobiProblemSetter
{
   public:
   static bool run( const Config::ParameterContainer& parameters );
};

#include "HamiltonJacobiProblemSetter_impl.h"

