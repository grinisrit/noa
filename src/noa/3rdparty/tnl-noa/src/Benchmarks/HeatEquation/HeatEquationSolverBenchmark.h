// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

// Implemented by: Tomáš Oberhuber, Yury Hayeu

#pragma once

#include <TNL/Config/parseCommandLine.h>
#include <TNL/Benchmarks/Benchmarks.h>

template< typename Real = double,
          typename Device = TNL::Devices::Host,
          typename Index = int >
struct HeatEquationSolverBenchmark
{
   static void configSetup( TNL::Config::ConfigDescription& config )
   {
      config.addDelimiter("Benchmark settings:");
      config.addEntry<TNL::String>("id", "Identifier of the run", "unknown");
      config.addEntry<TNL::String>("log-file", "Log file name.", "tnl-benchmark-heat-equation.log");
      config.addEntry<TNL::String>("output-mode", "Mode for opening the log file.", "overwrite");
      config.addEntryEnum("append");
      config.addEntryEnum("overwrite");
      config.addEntry<int>("min-x-dimension", "Minimum dimension over x axis used in the benchmark.", 100);
      config.addEntry<int>("max-x-dimension", "Maximum dimension over x axis used in the benchmark.", 200);
      config.addEntry<int>("x-size-step-factor", "Factor determining the dimension grows over x axis. First size is min-x-dimension and each following size is stepFactor*previousSize, up to max-x-dimension.", 2);

      config.addEntry<int>("min-y-dimension", "Minimum dimension over x axis used in the benchmark.", 100);
      config.addEntry<int>("max-y-dimension", "Maximum dimension over x axis used in the benchmark.", 200);
      config.addEntry<int>("y-size-step-factor", "Factor determining the dimension grows over y axis. First size is min-y-dimension and each following size is stepFactor*previousSize, up to max-y-dimension.", 2);

      config.addEntry<int>("loops", "Number of iterations for every computation.", 10);

      config.addEntry<int>("verbose", "Verbose mode.", 1);
      config.addEntry<bool>("write-data", "Write initial condition and final state to a file.", false );

      config.addDelimiter("Problem settings:");
      config.addEntry<double>("domain-x-size", "Domain size along x-axis.", 2.0);
      config.addEntry<double>("domain-y-size", "Domain size along y-axis.", 2.0);

      config.addDelimiter( "Initial condition settings ( (x^2/alpha + y^2/beta) + gamma)):" );
      config.addEntry< double >( "alpha", "Alpha value in initial condition", -0.05 );
      config.addEntry< double >( "beta", "Beta value in initial condition", -0.05 );
      config.addEntry< double >( "gamma", "Gamma key in initial condition", 5 );

      config.addEntry<double>("sigma", "Sigma in exponential initial condition.", 1.0);

      config.addEntry<double>("time-step", "Time step. By default it is proportional to one over space step square.", 0.0 );
      config.addEntry<double>("final-time", "Final time of the simulation.", 0.01);
      config.addEntry<int>("max-iterations", "Maximum time iterations.", 0 );

   }

   void init( const Index xSize, const Index ySize )
   {
      this->ux.setSize( xSize * ySize );
      this->aux.setSize( xSize * ySize );

      this->ux = 0;
      this->aux = 0;

      const Real hx = this->xDomainSize / (Real) xSize;
      const Real hy = this->yDomainSize / (Real) ySize;

      auto uxView = this->ux.getView();
      auto xDomainSize_ = this->xDomainSize;
      auto yDomainSize_ = this->yDomainSize;
      auto alpha_ = this->alpha;
      auto beta_ = this->beta;
      auto gamma_ = this->gamma;
      auto init = [=] __cuda_callable__(int i, int j) mutable
      {
         auto index = j * xSize + i;

         auto x = i * hx - xDomainSize_ / 2.;
         auto y = j * hy - yDomainSize_ / 2.;

         uxView[index] = TNL::max( ( ( ( x*x / alpha_ )  + ( y*y / beta_ ) ) + gamma_ ) * 0.2, 0.0 );
      };
      TNL::Algorithms::ParallelFor2D<Device>::exec( 1, 1, xSize - 1, ySize - 1, init );
   }

   bool writeGnuplot( const std::string &filename,
                      const Index xSize, const Index ySize ) const
   {
      std::ofstream out(filename, std::ios::out);
      if( !out.is_open() )
         return false;
      const Real hx = this->xDomainSize / (Real) xSize;
      const Real hy = this->yDomainSize / (Real) ySize;
      for( Index j = 0; j < ySize; j++)
         for( Index i = 0; i < xSize; i++)
            out << i * hx - this->xDomainSize / 2. << " "
               << j * hy - this->yDomainSize / 2. << " "
               << this->ux.getElement( j * xSize + i ) << std::endl;
      return out.good();
   }

   virtual void exec( const Index xSize, const Index ySize ) = 0;

   bool runBenchmark( const TNL::Config::ParameterContainer& parameters )
   {
      auto implementation = parameters.getParameter< TNL::String >( "implementation" );
      const TNL::String logFileName = parameters.getParameter< TNL::String >( "log-file" );
      const TNL::String outputMode = parameters.getParameter< TNL::String >( "output-mode" );
      bool writeData = parameters.getParameter< bool >( "write-data" );

      const Index minXDimension = parameters.getParameter< int >("min-x-dimension");
      const Index maxXDimension = parameters.getParameter< int >("max-x-dimension");
      const Index xSizeStepFactor = parameters.getParameter< int >("x-size-step-factor");

      if( xSizeStepFactor <= 1 ) {
         std::cerr << "The value of --x-size-step-factor must be greater than 1." << std::endl;
         return false;
      }

      const Index minYDimension = parameters.getParameter< int >("min-y-dimension");
      const Index maxYDimension = parameters.getParameter< int >("max-y-dimension");
      const Index ySizeStepFactor = parameters.getParameter< int >("y-size-step-factor");

      const int loops = parameters.getParameter< int >("loops");
      const int verbose = parameters.getParameter< int >("verbose");

      if( ySizeStepFactor <= 1 ) {
         std::cerr << "The value of --y-size-step-factor must be greater than 1." << std::endl;
         return false;
      }

      auto mode = std::ios::out;
      if( outputMode == "append" )
         mode |= std::ios::app;
      std::ofstream logFile( logFileName.getString(), mode );
      TNL::Benchmarks::Benchmark<> benchmark(logFile, loops, verbose);

      // write global metadata into a separate file
      std::map< std::string, std::string > metadata = TNL::Benchmarks::getHardwareMetadata();
      TNL::Benchmarks::writeMapAsJson( metadata, logFileName, ".metadata.json" );

      this->xDomainSize = parameters.getParameter<Real>( "domain-x-size" );
      this->yDomainSize = parameters.getParameter<Real>( "domain-y-size" );
      this->alpha = parameters.getParameter<Real>( "alpha" );
      this->beta = parameters.getParameter<Real>( "beta" );
      this->gamma = parameters.getParameter<Real>( "gamma" );
      this->timeStep = parameters.getParameter<Real>( "time-step" );
      this->finalTime = parameters.getParameter<Real>( "final-time" );
      this->maxIterations = parameters.getParameter< int >( "max-iterations" );

      auto precision = TNL::getType<Real>();
      TNL::String device;
      if( std::is_same< Device, TNL::Devices::Sequential >::value )
         device = "sequential";
      if( std::is_same< Device, TNL::Devices::Host >::value )
         device = "host";
      if( std::is_same< Device, TNL::Devices::Cuda >::value )
         device = "cuda";

      std::cout << "Heat equation benchmark  with (" << precision << ", " << device << ")" << std::endl;

      for( Index xSize = minXDimension; xSize <= maxXDimension; xSize *= xSizeStepFactor ) {
         for( Index ySize = minYDimension; ySize <= maxYDimension; ySize *= ySizeStepFactor ) {
            benchmark.setMetadataColumns( TNL::Benchmarks::Benchmark<>::MetadataColumns( {
               { "precision", precision },
               { "xSize", TNL::convertToString( xSize ) },
               { "ySize", TNL::convertToString( ySize ) },
               { "implementation", implementation }
            }));

            benchmark.setDatasetSize( xSize * ySize );
            this->init( xSize, ySize );
            if( writeData ) {
               TNL::String fileName = TNL::String( "initial-" ) + implementation +
                  "-" + TNL::convertToString( xSize) + "-" + TNL::convertToString( ySize ) + ".gplt";
               writeGnuplot( fileName.data(), xSize, ySize );
            }
            auto lambda = [&]() {
               this->exec( xSize, ySize );
            };
            benchmark.time<Device>(device, lambda);
            if( writeData ) {
               TNL::String fileName = TNL::String( "final-" ) + implementation +
                  "-" + TNL::convertToString( xSize) + "-" + TNL::convertToString( ySize ) + ".gplt";
               writeGnuplot( fileName.data(), xSize, ySize );
            }
         }
      }
      return true;
   }

protected:

   Real xDomainSize = 0.0, yDomainSize = 0.0;
   Real alpha = 0.0, beta = 0.0, gamma = 0.0;
   Real timeStep = 0.0, finalTime = 0.0;
   bool outputData = false;
   bool verbose = false;
   Index maxIterations = 0;

   TNL::Containers::Vector<Real, Device> ux, aux;
};
