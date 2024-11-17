#include <TNL/Config/parseCommandLine.h>
#include <TNL/CFD/TurbulenceGenerator.h>
#include <TNL/Meshes/NDMetaGrid.h>
#include <TNL/Meshes/Writers/VTIWriter.h>

using Real = double;
#ifdef __CUDACC__
using Device = TNL::Devices::Cuda;
#else
using Device = TNL::Devices::Host;
#endif

void
generate( Real X, Real Y, Real Z, int Nx, int Ny, int Nz, int ntimes, double timeStep, double timeScale )
{
   using Turbgen = TNL::CFD::TurbulenceGenerator< Real, Device, int >;
   Turbgen turbgen;

   // ensure deterministic results
   turbgen.rng.seed( 0 );

   // time correlation between individual realizations
   typename Turbgen::TimeCorrelationMethod timeCorrelation = Turbgen::TimeCorrelationMethod::initialNone;

   // generate grid coordinates in x, y, z
   using Array1D = typename decltype( turbgen )::Array1D;
   const Array1D xc = Array1D( TNL::linspace( Real( 0 ), X, Nx ) );
   const Array1D yc = Array1D( TNL::linspace( Real( 0 ), Y, Ny ) );
   const Array1D zc = Array1D( TNL::linspace( Real( 0 ), Z, Nz ) );

   const Real minSpaceStep = TNL::min( X / Nx, Y / ( Ny - 1 ), Z / ( Nz - 1 ) );

   auto result = turbgen.getFluctuations( xc, yc, zc, minSpaceStep, ntimes, timeStep, timeScale, timeCorrelation );

   using Grid = TNL::Meshes::NDMetaGrid< 3, Real, int >;
   Grid grid;
   grid.setDimensions( { Nx, Ny, Nz } );
   grid.setDomain( { 0, 0, 0 }, { X, Y, Z } );

   for( int t = 0; t < ntimes; t++ ) {
      std::ofstream file( "test." + std::to_string( t ) + ".vti" );
      using Writer = TNL::Meshes::Writers::VTIWriter< Grid >;
      Writer writer( file );
      writer.writeImageData( grid );

      // get subarrays for u, v, w
      auto u = std::get< 0 >( result ).getSubarrayView< 1, 2, 3 >( t, 0, 0, 0 );
      auto v = std::get< 1 >( result ).getSubarrayView< 1, 2, 3 >( t, 0, 0, 0 );
      auto w = std::get< 2 >( result ).getSubarrayView< 1, 2, 3 >( t, 0, 0, 0 );

      using ViewType = TNL::Containers::VectorView< Real, Device, int >;
      ViewType view;
      view.bind( u.getData(), u.getStorageSize() );
      writer.writeCellData( view, "u", 1 );
      std::cout << "u_" << t << " mean = " << TNL::sum( view ) / view.getSize() << std::endl;

      view.bind( v.getData(), v.getStorageSize() );
      writer.writeCellData( view, "v", 1 );
      std::cout << "v_" << t << " mean = " << TNL::sum( view ) / view.getSize() << std::endl;

      view.bind( w.getData(), w.getStorageSize() );
      writer.writeCellData( view, "w", 1 );
      std::cout << "w_" << t << " mean = " << TNL::sum( view ) / view.getSize() << std::endl;
   }
}

void
configSetup( TNL::Config::ConfigDescription& config )
{
   config.addEntry< double >( "x", "Physical grid size along the x-axis.", 1 );
   config.addEntry< double >( "y", "Physical grid size along the y-axis.", 1 );
   config.addEntry< double >( "z", "Physical grid size along the z-axis.", 1 );
   config.addEntry< int >( "nx", "Number of grid cells along the x-axis.", 1 );
   config.addEntry< int >( "ny", "Number of grid cells along the y-axis.", 128 );
   config.addEntry< int >( "nz", "Number of grid cells along the z-axis.", 256 );
   config.addEntry< int >( "ntimes", "Number of time steps to generate.", 100 );
   config.addEntry< double >( "time-step", "Physical time step between snapshots.", 0.015 );
   config.addEntry< double >( "time-scale", "Turbulence integral time scale.", 0.36 );
}

int
main( int argc, char* argv[] )
{
   TNL::Config::ParameterContainer parameters;
   TNL::Config::ConfigDescription conf_desc;

   configSetup( conf_desc );

   if( ! parseCommandLine( argc, argv, conf_desc, parameters ) )
      return EXIT_FAILURE;

   const double X = parameters.getParameter< double >( "x" );
   const double Y = parameters.getParameter< double >( "y" );
   const double Z = parameters.getParameter< double >( "z" );
   const int Nx = parameters.getParameter< int >( "nx" );
   const int Ny = parameters.getParameter< int >( "ny" );
   const int Nz = parameters.getParameter< int >( "nz" );
   const int ntimes = parameters.getParameter< int >( "ntimes" );
   const double timeStep = parameters.getParameter< double >( "time-step" );
   const double timeScale = parameters.getParameter< double >( "time-scale" );

   generate( X, Y, Z, Nx, Ny, Nz, ntimes, timeStep, timeScale );

   return EXIT_SUCCESS;
}
