#pragma once

#include <TNL/Config/ParameterContainer.h>
#include <TNL/Meshes/Grid.h>
#include <TNL/Meshes/Writers/VTIWriter.h>

using namespace TNL;

template< typename RealType, typename IndexType >
bool setupGrid( const Config::ParameterContainer& parameters )
{
   const String outputFile = parameters.getParameter< String >( "output-file" );
   const int dimension = parameters.getParameter< int >( "dimension" );
   if( dimension == 1 )
   {
      RealType originX = parameters.getParameter< double >( "origin-x" );
      RealType proportionsX = parameters.getParameter< double >( "proportions-x" );
      if( ! parameters.checkParameter( "size-x" ) ) {
         std::cerr << "The parameter size-x is required when the grid dimension is 1." << std::endl;
         return false;
      }
      IndexType sizeX = parameters.getParameter< int >( "size-x" );

      using GridType = Meshes::Grid< 1, RealType, Devices::Host, IndexType >;
      using PointType = typename GridType::PointType;
      using CoordinatesType = typename GridType::CoordinatesType;
      GridType grid;
      grid.setDomain( PointType( originX ), PointType( proportionsX ) );
      grid.setDimensions( CoordinatesType( sizeX ) );
      std::cout << "Setting dimensions to  ... " << sizeX << std::endl;
      std::cout << "Writing the grid to the file " << outputFile << std::endl;

      std::ofstream file( outputFile );
      Meshes::Writers::VTIWriter< GridType > writer( file );
      writer.writeImageData( grid );
   }
   if( dimension == 2 )
   {
      RealType originX = parameters.getParameter< double >( "origin-x" );
      RealType originY = parameters.getParameter< double >( "origin-y" );
      RealType proportionsX = parameters.getParameter< double >( "proportions-x" );
      RealType proportionsY = parameters.getParameter< double >( "proportions-y" );
      if( ! parameters.checkParameters( {"size-x", "size-y"} ) ) {
         std::cerr << "The parameters size-x and size-y are required when the grid dimension is 2." << std::endl;
         return false;
      }
      IndexType sizeX = parameters.getParameter< int >( "size-x" );
      IndexType sizeY = parameters.getParameter< int >( "size-y" );
      using GridType = Meshes::Grid< 2, RealType, Devices::Host, IndexType >;
      using PointType = typename GridType::PointType;
      using CoordinatesType = typename GridType::CoordinatesType;
      GridType grid;
      grid.setDomain( PointType( originX, originY ), PointType( proportionsX, proportionsY ) );
      grid.setDimensions( CoordinatesType( sizeX, sizeY ) );
      if( parameters.getParameter< bool >( "equal-space-steps" ) )
      {
         if( grid.getSpaceSteps().x() != grid.getSpaceSteps().y() )
         {
            double h = min( grid.getSpaceSteps().x(), grid.getSpaceSteps().y() );
            grid.setSpaceSteps( PointType( h, h ) );
            std::cout << "Adjusting grid space steps to " << grid.getSpaceSteps()
                      << " and grid proportions to " << grid.getProportions() << "."  << std::endl;

         }
      }
      std::cout << "Setting dimensions to  ... " << grid.getDimensions() << std::endl;
      std::cout << "Writing the grid to the file " << outputFile << std::endl;

      std::ofstream file( outputFile );
      Meshes::Writers::VTIWriter< GridType > writer( file );
      writer.writeImageData( grid );
   }
   if( dimension == 3 )
   {
      RealType originX = parameters.getParameter< double >( "origin-x" );
      RealType originY = parameters.getParameter< double >( "origin-y" );
      RealType originZ = parameters.getParameter< double >( "origin-z" );
      RealType proportionsX = parameters.getParameter< double >( "proportions-x" );
      RealType proportionsY = parameters.getParameter< double >( "proportions-y" );
      RealType proportionsZ = parameters.getParameter< double >( "proportions-z" );
      if( ! parameters.checkParameters( {"size-x", "size-y", "size-z"} ) ) {
         std::cerr << "The parameters size-x, size-y and size-z are required when the grid dimension is 3." << std::endl;
         return false;
      }
      IndexType sizeX = parameters.getParameter< int >( "size-x" );
      IndexType sizeY = parameters.getParameter< int >( "size-y" );
      IndexType sizeZ = parameters.getParameter< int >( "size-z" );

      using GridType = Meshes::Grid< 3, RealType, Devices::Host, IndexType >;
      using PointType = typename GridType::PointType;
      using CoordinatesType = typename GridType::CoordinatesType;
      GridType grid;
      grid.setDomain( PointType( originX, originY, originZ ), PointType( proportionsX, proportionsY, proportionsZ ) );
      grid.setDimensions( CoordinatesType( sizeX, sizeY, sizeZ ) );
      if( parameters.getParameter< bool >( "equal-space-steps" ) )
      {
         if( grid.getSpaceSteps().x() != grid.getSpaceSteps().y() ||
             grid.getSpaceSteps().x() != grid.getSpaceSteps().z() )
         {
            double h = min( grid.getSpaceSteps().x(), min( grid.getSpaceSteps().y(), grid.getSpaceSteps().z() ) );
            grid.setSpaceSteps( PointType( h, h, h ) );
            std::cout << "Adjusting grid space steps to " << grid.getSpaceSteps()
                      << " and grid proportions to " << grid.getProportions() << "." << std::endl;
         }
      }
      std::cout << "Setting dimensions to  ... " << grid.getDimensions() << std::endl;
      std::cout << "Writing the grid to the file " << outputFile << std::endl;

      std::ofstream file( outputFile );
      Meshes::Writers::VTIWriter< GridType > writer( file );
      writer.writeImageData( grid );
   }
   std::cout << "[ OK ] " << std::endl;
   return true;
}

template< typename RealType >
bool resolveIndexType( const Config::ParameterContainer& parameters )
{
   const String& indexType = parameters.getParameter< String >( "index-type" );
   std::cout << "Setting index type to  ... " << indexType << std::endl;
   if( indexType == "int" )
      return setupGrid< RealType, int >( parameters );
   if( indexType == "long-int" )
      return setupGrid< RealType, long int >( parameters );
   std::cerr << "The index type '" << indexType << "' is not defined." << std::endl;
   return false;
}

inline bool resolveRealType( const Config::ParameterContainer& parameters )
{
   String realType = parameters.getParameter< String >( "real-type" );
   std::cout << "Setting real type to   ... " << realType << std::endl;
   if( realType == "float" )
      return resolveIndexType< float >( parameters );
   if( realType == "double" )
      return resolveIndexType< double >( parameters );
   if( realType == "long-double" )
      return resolveIndexType< long double >( parameters );
   std::cerr << "The real type '" << realType << "' is not supported." << std::endl;
   return false;
}
