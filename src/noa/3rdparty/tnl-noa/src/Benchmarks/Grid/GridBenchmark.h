
// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

// Implemented by: Tomáš Oberhuber, Yury Hayeu

#pragma once

#include "Operations.h"

#include <vector>

#include <TNL/Meshes/Grid.h>
#include <TNL/Config/parseCommandLine.h>

#include <TNL/Devices/Host.h>
#include <TNL/Devices/Cuda.h>

#include <TNL/Benchmarks/Benchmarks.h>

static std::vector<TNL::String> dimensionParameterIds = { "x-dimension", "y-dimension", "z-dimension" };

template< typename Real = double,
          typename Device = TNL::Devices::Host,
           typename Index = int >
class GridBenchmark {
   public:
      using Benchmark = typename TNL::Benchmarks::Benchmark<>;

      static void setupConfig( TNL::Config::ConfigDescription& config ) {

         config.addDelimiter("Benchmark settings:");
         config.addEntry<TNL::String>("log-file", "Log file name.", "output.log");
         config.addEntry<TNL::String>("output-mode", "Mode for opening the log file.", "overwrite");
         config.addEntryEnum("append");
         config.addEntryEnum("overwrite");
         for (int i = 0; i < 3; i++)
            config.addEntry<int>( dimensionParameterIds[i], "Grid resolution.", 100 );

         config.addEntry<int>("loops", "Number of iterations for every computation.", 10);
         config.addEntry<int>("verbose", "Verbose mode.", 1 );
      }

      template< int GridDimension >
      int runBenchmark(const TNL::Config::ParameterContainer& parameters) const {
         if (!TNL::Devices::Host::setup( parameters ) || !TNL::Devices::Cuda::setup( parameters ) )
            return EXIT_FAILURE;

         const TNL::String logFileName = parameters.getParameter<TNL::String>( "log-file" );
         const TNL::String outputMode = parameters.getParameter<TNL::String>( "output-mode" );

         const int verbose = parameters.getParameter< int >("verbose");
         const int loops = parameters.getParameter< int >("loops");

         auto mode = std::ios::out;

         if( outputMode == "append" )
            mode |= std::ios::app;

         std::ofstream logFile( logFileName.getString(), mode );

         Benchmark benchmark(logFile, loops, verbose);

         // write global metadata into a separate file
         std::map< std::string, std::string > metadata = TNL::Benchmarks::getHardwareMetadata();
         TNL::Benchmarks::writeMapAsJson( metadata, logFileName, ".metadata.json" );
         time< GridDimension >(benchmark, parameters);
         return 0;
      }

      template< int GridDimension >
      void time(Benchmark& benchmark, const TNL::Config::ParameterContainer& parameters) const {
         using Grid = typename TNL::Meshes::Grid<GridDimension, Real, Device, int>;
         using CoordinatesType = typename Grid::CoordinatesType;

         CoordinatesType dimensions;

         for (int i = 0; i < GridDimension; i++)
            dimensions[i] = parameters.getParameter<int>(dimensionParameterIds[i]);

         Grid grid;

         grid.setDimensions(dimensions);

         auto forEachEntityDimension = [&](const auto entityDimension) {
            timeTraverse<entityDimension, Grid, VoidOperation>(benchmark, grid);

            timeTraverse<entityDimension, Grid, GetEntityIsBoundaryOperation>(benchmark, grid);
            timeTraverse<entityDimension, Grid, GetEntityCoordinateOperation>(benchmark, grid);
            timeTraverse<entityDimension, Grid, GetEntityIndexOperation>(benchmark, grid);
            timeTraverse<entityDimension, Grid, GetEntityNormalsOperation>(benchmark, grid);
            timeTraverse<entityDimension, Grid, RefreshEntityOperation>(benchmark, grid);

            timeTraverse<entityDimension, Grid, GetMeshDimensionOperation>(benchmark, grid);
            timeTraverse<entityDimension, Grid, GetOriginOperation>(benchmark, grid);
            timeTraverse<entityDimension, Grid, GetEntitiesCountsOperation>(benchmark, grid);
         };
         TNL::Algorithms::staticFor< int, 0, GridDimension + 1 >(forEachEntityDimension);
      }

      template<int EntityDimension, typename Grid, typename Operation>
      void timeTraverse(Benchmark& benchmark, const Grid& grid) const {
         auto exec = [] __cuda_callable__ (typename Grid::template EntityType<EntityDimension>& entity) mutable {
            Operation::exec(entity);
         };

         TNL::String device;
         if( std::is_same< Device, TNL::Devices::Sequential >::value )
            device = "sequential";
         if( std::is_same< Device, TNL::Devices::Host >::value )
            device = "host";
         if( std::is_same< Device, TNL::Devices::Cuda >::value )
            device = "cuda";

         auto operation = TNL::getType<Operation>();

         const Benchmark::MetadataColumns columns = {
            { "dimensions", TNL::convertToString(grid.getDimensions()) },
            { "entity_dimension", TNL::convertToString(EntityDimension) },
            { "entitiesCounts", TNL::convertToString(grid.getEntitiesCount(EntityDimension)) },
            { "operation_id", operation }
         };

         Benchmark::MetadataColumns forAllColumns( columns );
         forAllColumns.push_back( { "traverse_id", "forAll" } );
         benchmark.setMetadataColumns(forAllColumns);
         auto measureAll = [=]() {
            grid.template forAllEntities<EntityDimension>(exec);
         };
         benchmark.time<typename Grid::DeviceType>(device, measureAll);

         Benchmark::MetadataColumns forInteriorColumns( columns );
         forInteriorColumns.push_back( { "traverse_id", "forInterior" } );
         benchmark.setMetadataColumns(forInteriorColumns);
         auto measureInterior = [=]() {
            grid.template forInteriorEntities<EntityDimension>(exec);
         };
         benchmark.time<typename Grid::DeviceType>(device, measureInterior);


         Benchmark::MetadataColumns forBoundaryColumns( columns );
         forBoundaryColumns.push_back( { "traverse_id", "forBoundary" } );
         benchmark.setMetadataColumns(forInteriorColumns);
         auto measureBoundary = [=]() {
            grid.template forBoundaryEntities<EntityDimension>(exec);
         };
         benchmark.time<typename Grid::DeviceType>(device, measureBoundary);
      }
};
