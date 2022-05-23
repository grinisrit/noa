#pragma once 

#include <TNL/Config/ConfigDescription.h>

template< typename ConfigTag >
class HamiltonJacobiProblemConfig
{
   public:
      static void configSetup( tnlConfigDescription& config )
      {
         config.addDelimiter( "Hamilton-Jacobi solver settings:" );
         config.addEntry        < String > ( "problem-name", "This defines particular problem.", "hamilton-jacobi" );
         config.addEntry       < String > ( "scheme", "This defines scheme used for discretization.", "godunov" );
         config.addEntryEnum( "godunov" );
         config.addEntryEnum( "upwind" );
         config.addEntry        < double > ( "epsilon", "This defines epsilon for smoothening of sign().", 3.0 );
         config.addEntry        < double > ( "-value", "Constant value of RHS.", 0.0 );
      }
};

