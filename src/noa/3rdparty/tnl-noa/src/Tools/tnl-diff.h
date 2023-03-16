#pragma once

#include <iomanip>
#include <TNL/Config/ParameterContainer.h>
#include <TNL/FileName.h>
#include <TNL/Containers/Vector.h>
#include <TNL/Containers/StaticVector.h>
#include <TNL/Meshes/TypeResolver/resolveMeshType.h>
#include <TNL/Functions/MeshFunction.h>
#include <TNL/Exceptions/NotImplementedError.h>

using namespace TNL;

template< typename MeshFunction,
          typename Mesh = typename MeshFunction::MeshType,
          int EntitiesDimension = MeshFunction::getEntitiesDimension() >
class ExactMatchTest
{
   public:
      static void run( const MeshFunction& f1,
                       const MeshFunction& f2,
                       const String& f1Name,
                       const String& f2Name,
                       std::fstream& outputFile,
                       bool verbose = false)
      {
         throw Exceptions::NotImplementedError("Not implemented yet.");
      }
};

template< typename MeshFunction,
          typename Real,
          typename Device,
          typename Index >
class ExactMatchTest< MeshFunction, Meshes::Grid< 1, Real, Device, Index >, 1 >
{
   public:
      static void run( const MeshFunction& f1,
                       const MeshFunction& f2,
                       const String& f1Name,
                       const String& f2Name,
                       std::fstream& outputFile,
                       bool verbose = false )
      {
         const int Dimension = 1;
         const int EntityDimension = 1;
         using Grid = Meshes::Grid< Dimension, Real, Device, Index >;
         using Entity = typename Grid::template EntityType< EntityDimension >;
         if( f1.getMesh().getDimensions() != f2.getMesh().getDimensions() )
         {
            outputFile << f1Name << " and " << f2Name << " are defined on different meshes. "  << std::endl;
            if( verbose )
               std::cout << f1Name << " and " << f2Name << " are defined on different meshes. "  << std::endl;
         }

         Entity entity( f1.getMesh() );
         for( entity.getCoordinates().x() = 0;
              entity.getCoordinates().x() < f1.getMesh().getDimensions().x();
              entity.getCoordinates().x()++ )
         {
            entity.refresh();
            if( f1.getValue( entity ) != f2.getValue( entity ) )
            {
               outputFile << f1Name << " and " << f2Name << " differs at " << entity.getCoordinates()
                          << " " << f1.getValue( entity ) << " != " << f2.getValue( entity ) <<std::endl;
               if( verbose )
                  std::cout << f1Name << " and " << f2Name << " differs at " << entity.getCoordinates()
                          << " " << f1.getValue( entity ) << " != " << f2.getValue( entity ) <<std::endl;
            }
         }
      }
};

template< typename MeshFunction,
          typename Real,
          typename Device,
          typename Index >
class ExactMatchTest< MeshFunction, Meshes::Grid< 2, Real, Device, Index >, 2 >
{
   public:
      static void run( const MeshFunction& f1,
                       const MeshFunction& f2,
                       const String& f1Name,
                       const String& f2Name,
                       std::fstream& outputFile,
                       bool verbose = false )
      {
         const int Dimension = 2;
         const int EntityDimension = 2;
         using Grid = Meshes::Grid< Dimension, Real, Device, Index >;
         using Entity = typename Grid::template EntityType< EntityDimension >;
         if( f1.getMesh().getDimensions() != f2.getMesh().getDimensions() )
         {
            outputFile << f1Name << " and " << f2Name << " are defined on different meshes. "  << std::endl;
            if( verbose )
               std::cout << f1Name << " and " << f2Name << " are defined on different meshes. "  << std::endl;
         }

         Entity entity( f1.getMesh() );
         for( entity.getCoordinates().y() = 0;
              entity.getCoordinates().y() < f1.getMesh().getDimensions().y();
              entity.getCoordinates().y()++ )
            for( entity.getCoordinates().x() = 0;
                 entity.getCoordinates().x() < f1.getMesh().getDimensions().x();
                 entity.getCoordinates().x()++ )
            {
               entity.refresh();
               if( f1.getValue( entity ) != f2.getValue( entity ) )
               {
                  outputFile << f1Name << " and " << f2Name << " differs at " << entity.getCoordinates()
                             << " " << f1.getValue( entity ) << " != " << f2.getValue( entity ) <<std::endl;
                  if( verbose )
                     std::cout << f1Name << " and " << f2Name << " differs at " << entity.getCoordinates()
                               << " " << f1.getValue( entity ) << " != " << f2.getValue( entity ) <<std::endl;
               }
            }
      }
};

template< typename MeshFunction,
          typename Real,
          typename Device,
          typename Index >
class ExactMatchTest< MeshFunction, Meshes::Grid< 3, Real, Device, Index >, 3 >
{
   public:
      static void run( const MeshFunction& f1,
                       const MeshFunction& f2,
                       const String& f1Name,
                       const String& f2Name,
                       std::fstream& outputFile,
                       bool verbose = false )
      {
         const int Dimension = 3;
         const int EntityDimension = 3;
         using Grid = Meshes::Grid< Dimension, Real, Device, Index >;
         using Entity = typename Grid::template EntityType< EntityDimension >;
         if( f1.getMesh().getDimensions() != f2.getMesh().getDimensions() )
         {
            outputFile << f1Name << " and " << f2Name << " are defined on different meshes. "  << std::endl;
            if( verbose )
               std::cout << f1Name << " and " << f2Name << " are defined on different meshes. "  << std::endl;
         }

         Entity entity( f1.getMesh() );
         for( entity.getCoordinates().z() = 0;
              entity.getCoordinates().z() < f1.getMesh().getDimensions().z();
              entity.getCoordinates().z()++ )
            for( entity.getCoordinates().y() = 0;
                 entity.getCoordinates().y() < f1.getMesh().getDimensions().y();
                 entity.getCoordinates().y()++ )
               for( entity.getCoordinates().x() = 0;
                    entity.getCoordinates().x() < f1.getMesh().getDimensions().x();
                    entity.getCoordinates().x()++ )
               {
                  entity.refresh();
                  if( f1.getValue( entity ) != f2.getValue( entity ) )
                  {
                     outputFile << f1Name << " and " << f2Name << " differs at " << entity.getCoordinates()
                                << " " << f1.getValue( entity ) << " != " << f2.getValue( entity ) <<std::endl;
                     if( verbose )
                        std::cout << f1Name << " and " << f2Name << " differs at " << entity.getCoordinates()
                                << " " << f1.getValue( entity ) << " != " << f2.getValue( entity ) <<std::endl;
                  }
               }
      }
};



template< typename MeshPointer, typename Value, typename Real, typename Index >
bool computeDifferenceOfMeshFunctions( const MeshPointer& meshPointer, const Config::ParameterContainer& parameters )
{
   bool verbose = parameters. getParameter< bool >( "verbose" );
   std::vector< String > inputFiles = parameters. getParameter< std::vector< String > >( "input-files" );
   const String meshFunctionName = parameters.getParameter< String >( "mesh-function-name" );
   String mode = parameters. getParameter< String >( "mode" );
   String outputFileName = parameters. getParameter< String >( "output-file" );
   double snapshotPeriod = parameters. getParameter< double >( "snapshot-period" );
   bool writeDifference = parameters. getParameter< bool >( "write-difference" );
   bool exactMatch = parameters. getParameter< bool >( "exact-match" );

   std::fstream outputFile;
   outputFile.open( outputFileName.getString(), std::fstream::out );
   if( ! outputFile )
   {
      std::cerr << "Unable to open the file " << outputFileName << "." << std::endl;
      return false;
   }
   if( ! exactMatch )
   {
      outputFile << "#";
      outputFile << std::setw( 6 ) << "Time";
      outputFile << std::setw( 18 ) << "L1 diff."
                 << std::setw( 18 ) << "L2 diff."
                 << std::setw( 18 ) << "Max. diff."
                 << std::setw( 18 ) << "Total L1 diff."
                 << std::setw( 18 ) << "Total L2 diff."
                 << std::setw( 18 ) << "Total Max. diff."
                 << std::endl;
   }
   if( verbose )
      std::cout << std::endl;

   using Mesh = typename MeshPointer::ObjectType;
   using MeshFunctionType = Functions::MeshFunction< Mesh, Mesh::getMeshDimension(), Real >;
   MeshFunctionType v1( meshPointer );
   MeshFunctionType v2( meshPointer );
   MeshFunctionType diff( meshPointer );
   Real totalL1Diff = 0;
   Real totalL2Diff = 0;
   Real totalMaxDiff = 0;
   for( int i = 0; i < (int) inputFiles.size(); i ++ )
   {
      String file1;
      String file2;
      if( mode == "couples" )
      {
         if( i + 1 == (int) inputFiles.size() )
         {
            std::cerr << std::endl << "Skipping the file " << inputFiles[ i ] << " since there is no file to couple it with." << std::endl;
            outputFile.close();
            return false;
         }
         if( verbose )
           std::cout << "Processing files " << inputFiles[ i ] << " and " << inputFiles[ i + 1 ] << "...           \r" << std::flush;
         try
         {
            Functions::readMeshFunction( v1, meshFunctionName, inputFiles[ i ], "auto" );
            Functions::readMeshFunction( v2, meshFunctionName, inputFiles[ i + 1 ], "auto" );
         }
         catch(...)
         {
            std::cerr << "Unable to read the files " << inputFiles[ i ] << " and " << inputFiles[ i + 1 ] << "." << std::endl;
            outputFile.close();
            return false;
         }
         if( ! exactMatch )
            outputFile << std::setw( 6 ) << i/2 * snapshotPeriod << " ";
         file1 = inputFiles[ i ];
         file2 = inputFiles[ i + 1 ];
         i++;
      }
      if( mode == "sequence" )
      {
         if( i == 0 )
         {
            if( verbose )
              std::cout << "Reading the file " << inputFiles[ 0 ] << "...               \r" << std::flush;
            Functions::readMeshFunction( v1, meshFunctionName, inputFiles[ 0 ], "auto" );
            file1 = inputFiles[ 0 ];
         }
         if( verbose )
           std::cout << "Processing the files " << inputFiles[ 0 ] << " and " << inputFiles[ i ] << "...             \r" << std::flush;
         Functions::readMeshFunction( v2, meshFunctionName, inputFiles[ i ], "auto" );
         if( ! exactMatch )
            outputFile << std::setw( 6 ) << ( i - 1 ) * snapshotPeriod << " ";
         file2 = inputFiles[ i ];
      }
      if( mode == "halves" )
      {
         const int half = inputFiles.size() / 2;
         if( i == 0 )
            i = half;
         if( verbose )
           std::cout << "Processing files " << inputFiles[ i - half ] << " and " << inputFiles[ i ] << "...                 \r" << std::flush;
         Functions::readMeshFunction( v1, meshFunctionName, inputFiles[ i - half ], "auto" );
         Functions::readMeshFunction( v2, meshFunctionName, inputFiles[ i ], "auto" );
         //if( snapshotPeriod != 0.0 )
         if( ! exactMatch )
            outputFile << std::setw( 6 ) << ( i - half ) * snapshotPeriod << " ";
         file1 = inputFiles[ i - half ];
         file2 = inputFiles[ i ];
      }
      diff = v1;
      diff -= v2;
      if( exactMatch )
         ExactMatchTest< MeshFunctionType >::run( v1, v2, file1, file2, outputFile, verbose );
      else
      {
         Real l1Diff = diff.getLpNorm( 1.0 );
         Real l2Diff = diff.getLpNorm( 2.0 );
         Real maxDiff = diff.getMaxNorm();
         if( snapshotPeriod != 0.0 )
         {
            totalL1Diff += snapshotPeriod * l1Diff;
            totalL2Diff += snapshotPeriod * l2Diff * l2Diff;
         }
         else
         {
            totalL1Diff += l1Diff;
            totalL2Diff += l2Diff * l2Diff;
         }
         totalMaxDiff = max( totalMaxDiff, maxDiff );
         outputFile << std::setw( 18 ) << l1Diff
                    << std::setw( 18 ) << l2Diff
                    << std::setw( 18 ) << maxDiff
                    << std::setw( 18 ) << totalL1Diff
                    << std::setw( 18 ) << ::sqrt( totalL2Diff )
                    << std::setw( 18 ) << totalMaxDiff << std::endl;

         if( writeDifference )
         {
            String differenceFileName = removeFileNameExtension( inputFiles[ i ] ) + ".diff.vti";
            //diff.setLike( v1 );
            diff = v1;
            diff -= v2;
            diff.write( "diff", differenceFileName );
         }
      }
   }
   outputFile.close();

   if( verbose )
     std::cout << std::endl;
   return true;
}


template< typename MeshPointer, typename Value, typename Real, typename Index >
bool computeDifferenceOfVectors( const MeshPointer& meshPointer, const Config::ParameterContainer& parameters )
{
   bool verbose = parameters.getParameter< bool >( "verbose" );
   std::vector< String > inputFiles = parameters.getParameter< std::vector< String > >( "input-files" );
   String mode = parameters.getParameter< String >( "mode" );
   String outputFileName = parameters.getParameter< String >( "output-file" );
   double snapshotPeriod = parameters.getParameter< double >( "snapshot-period" );
   bool writeDifference = parameters.getParameter< bool >( "write-difference" );

   std::fstream outputFile;
   outputFile.open( outputFileName.getString(), std::fstream::out );
   if( ! outputFile )
   {
      std::cerr << "Unable to open the file " << outputFileName << "." << std::endl;
      return false;
   }
   outputFile << "#";
   outputFile << std::setw( 6 ) << "Time";
   outputFile << std::setw( 18 ) << "L1 diff."
              << std::setw( 18 ) << "L2 diff."
              << std::setw( 18 ) << "Max. diff."
              << std::setw( 18 ) << "Total L1 diff."
              << std::setw( 18 ) << "Total L2 diff."
              << std::setw( 18 ) << "Total Max. diff."
              << std::endl;
   if( verbose )
     std::cout << std::endl;

   using VectorType = Containers::Vector< Real, Devices::Host, Index >;
   VectorType v1;
   VectorType v2;
   Real totalL1Diff = 0;
   Real totalL2Diff = 0;
   Real totalMaxDiff = 0;
   for( int i = 0; i < (int) inputFiles.size(); i++ )
   {
      if( mode == "couples" )
      {
         if( i + 1 == (int) inputFiles.size() )
         {
            std::cerr << std::endl << "Skipping the file " << inputFiles[ i ] << " since there is no file to couple it with." << std::endl;
            outputFile.close();
            return false;
         }
         if( verbose )
           std::cout << "Processing files " << inputFiles[ i ] << " and " << inputFiles[ i + 1 ] << "...           \r" << std::flush;
         File( inputFiles[ i ], std::ios_base::in ) >> v1;
         File( inputFiles[ i + 1 ], std::ios_base::in ) >> v2;
         outputFile << std::setw( 6 ) << i/2 * snapshotPeriod << " ";
         i++;
      }
      if( mode == "sequence" )
      {
         if( i == 0 )
         {
            if( verbose )
              std::cout << "Reading the file " << inputFiles[ 0 ] << "...               \r" << std::flush;
            File( inputFiles[ 0 ], std::ios_base::in ) >> v1;
         }
         if( verbose )
           std::cout << "Processing the files " << inputFiles[ 0 ] << " and " << inputFiles[ i ] << "...             \r" << std::flush;
         File( inputFiles[ i ], std::ios_base::in ) >> v2;
         outputFile << std::setw( 6 ) << ( i - 1 ) * snapshotPeriod << " ";
      }
      if( mode == "halves" )
      {
         const int half = inputFiles.size() / 2;
         if( i == 0 )
            i = half;
         if( verbose )
           std::cout << "Processing files " << inputFiles[ i - half ] << " and " << inputFiles[ i ] << "...                 \r" << std::flush;
         File( inputFiles[ i - half ], std::ios_base::in ) >> v1;
         File( inputFiles[ i ], std::ios_base::in ) >> v2;
         //if( snapshotPeriod != 0.0 )
         outputFile << std::setw( 6 ) << ( i - half ) * snapshotPeriod << " ";
      }
      Real cellVolume = meshPointer->getCellMeasure();
//      Real l1Diff = meshPointer->getDifferenceLpNorm( v1, v2, 1.0 );
      Real l1Diff = cellVolume * sum( abs( v1 - v2 ) );
//      Real l2Diff = meshPointer->getDifferenceLpNorm( v1, v2, 2.0 );
      Real l2Diff = cellVolume * std::sqrt( dot(v1 - v2, v1 - v2) );
      Real maxDiff = max( abs( v1 - v2 ) );
      if( snapshotPeriod != 0.0 )
      {
         totalL1Diff += snapshotPeriod * l1Diff;
         totalL2Diff += snapshotPeriod * l2Diff * l2Diff;
      }
      else
      {
         totalL1Diff += l1Diff;
         totalL2Diff += l2Diff * l2Diff;
      }
      totalMaxDiff = max( totalMaxDiff, maxDiff );
      outputFile << std::setw( 18 ) << l1Diff
                 << std::setw( 18 ) << l2Diff
                 << std::setw( 18 ) << maxDiff
                 << std::setw( 18 ) << totalL1Diff
                 << std::setw( 18 ) << ::sqrt( totalL2Diff )
                 << std::setw( 18 ) << totalMaxDiff << std::endl;

      if( writeDifference )
      {
         String differenceFileName = removeFileNameExtension( inputFiles[ i ] ) + ".diff.tnl";
         Containers::Vector< Real, Devices::Host, Index > diff;
         diff.setLike( v1 );
         diff = v1;
         diff -= v2;
         File( differenceFileName, std::ios_base::out ) << diff;
      }
   }
   outputFile.close();

   if( verbose )
     std::cout << std::endl;
   return true;
}

template< typename MeshPointer, typename Value, typename Real, typename Index >
bool computeDifference( const MeshPointer& meshPointer, const String& objectType, const Config::ParameterContainer& parameters )
{
   if( objectType == "Functions::MeshFunction" )
      return computeDifferenceOfMeshFunctions< MeshPointer, Value, Real, Index >( meshPointer, parameters );
   if( objectType == "Containers::Array" ||
       objectType == "Containers::Vector" )  // TODO: remove deprecated names (Vector is saved as Array)
      return computeDifferenceOfVectors< MeshPointer, Value, Real, Index >( meshPointer, parameters );
   std::cerr << "Unknown object type " << objectType << "." << std::endl;
   return false;
}


template< typename MeshPointer, typename Value, typename Real >
bool setIndexType( const MeshPointer& meshPointer,
                   const String& inputFileName,
                   const std::vector< String >& parsedObjectType,
                   const Config::ParameterContainer& parameters )
{
   String indexType;
   if( parsedObjectType[ 0 ] == "Containers::Array" ||
       parsedObjectType[ 0 ] == "Containers::Vector" )  // TODO: remove deprecated names (Vector is saved as Array)
      indexType = parsedObjectType[ 3 ];

   if( parsedObjectType[ 0 ] == "Functions::MeshFunction" )
      return computeDifference< MeshPointer, Value, Real, typename MeshPointer::ObjectType::IndexType >( meshPointer, parsedObjectType[ 0 ], parameters );

   if( indexType == "int" )
      return computeDifference< MeshPointer, Value, Real, int >( meshPointer, parsedObjectType[ 0 ], parameters );
   if( indexType == "long-int" )
      return computeDifference< MeshPointer, Value, Real, long int >( meshPointer, parsedObjectType[ 0 ], parameters );
   std::cerr << "Unknown index type " << indexType << "." << std::endl;
   return false;
}

template< typename MeshPointer >
bool setTupleType( const MeshPointer& meshPointer,
                   const String& inputFileName,
                   const std::vector< String >& parsedObjectType,
                   const std::vector< String >& parsedValueType,
                   const Config::ParameterContainer& parameters )
{
   int dimensions = atoi( parsedValueType[ 1 ].getString() );
   const String& dataType = parsedValueType[ 2 ];
   if( dataType == "float" )
      switch( dimensions )
      {
         case 1:
            return setIndexType< MeshPointer, Containers::StaticVector< 1, float >, float >( meshPointer, inputFileName, parsedObjectType, parameters );
            break;
         case 2:
            return setIndexType< MeshPointer, Containers::StaticVector< 2, float >, float >( meshPointer, inputFileName, parsedObjectType, parameters );
            break;
         case 3:
            return setIndexType< MeshPointer, Containers::StaticVector< 3, float >, float >( meshPointer, inputFileName, parsedObjectType, parameters );
            break;
      }
   if( dataType == "double" )
      switch( dimensions )
      {
         case 1:
            return setIndexType< MeshPointer, Containers::StaticVector< 1, double >, double >( meshPointer, inputFileName, parsedObjectType, parameters );
            break;
         case 2:
            return setIndexType< MeshPointer, Containers::StaticVector< 2, double >, double >( meshPointer, inputFileName, parsedObjectType, parameters );
            break;
         case 3:
            return setIndexType< MeshPointer, Containers::StaticVector< 3, double >, double >( meshPointer, inputFileName, parsedObjectType, parameters );
            break;
      }
//   if( dataType == "long double" )
//      switch( dimensions )
//      {
//         case 1:
//            return setIndexType< MeshPointer, Containers::StaticVector< 1, long double >, long double >( meshPointer, inputFileName, parsedObjectType, parameters );
//            break;
//         case 2:
//            return setIndexType< MeshPointer, Containers::StaticVector< 2, long double >, long double >( meshPointer, inputFileName, parsedObjectType, parameters );
//            break;
//         case 3:
//            return setIndexType< MeshPointer, Containers::StaticVector< 3, long double >, long double >( meshPointer, inputFileName, parsedObjectType, parameters );
//            break;
//      }
   return false;
}

template< typename MeshPointer >
bool setValueType( const MeshPointer& meshPointer,
                     const String& inputFileName,
                     const std::vector< String >& parsedObjectType,
                     const Config::ParameterContainer& parameters )
{
   String elementType;

   if( parsedObjectType[ 0 ] == "Functions::MeshFunction" )
      elementType = parsedObjectType[ 3 ];
   if( parsedObjectType[ 0 ] == "Containers::Array" ||
       parsedObjectType[ 0 ] == "Containers::Vector" )  // TODO: remove deprecated names (Vector is saved as Array)
      elementType = parsedObjectType[ 1 ];


   if( elementType == "float" )
      return setIndexType< MeshPointer, float, float >( meshPointer, inputFileName, parsedObjectType, parameters );
   if( elementType == "double" )
      return setIndexType< MeshPointer, double, double >( meshPointer, inputFileName, parsedObjectType, parameters );
//   if( elementType == "long double" )
//      return setIndexType< MeshPointer, long double, long double >( meshPointer, inputFileName, parsedObjectType, parameters );
   const std::vector< String > parsedValueType = parseObjectType( elementType );
   if( parsedValueType.empty() )
   {
      std::cerr << "Unable to parse object type " << elementType << "." << std::endl;
      return false;
   }
   if( parsedValueType[ 0 ] == "Containers::StaticVector" )
      return setTupleType< MeshPointer >( meshPointer, inputFileName, parsedObjectType, parsedValueType, parameters );

   std::cerr << "Unknown element type " << elementType << "." << std::endl;
   return false;
}

template< typename Mesh >
bool processFiles( const Config::ParameterContainer& parameters )
{
   int verbose = parameters.getParameter< int >( "verbose");
   const String meshFile = parameters.getParameter< String >( "mesh" );
   const String meshFileFormat = parameters.getParameter< String >( "mesh-format" );

   using MeshPointer = Pointers::SharedPointer< Mesh >;
   MeshPointer meshPointer;
   if( ! Meshes::loadMesh( *meshPointer, meshFile, meshFileFormat ) )
      return false;

   std::vector< String > inputFiles = parameters.getParameter< std::vector< String > >( "input-files" );

   String objectType;
   try
   {
      objectType = getObjectType( inputFiles[ 0 ] );
   }
   catch( const std::ios_base::failure& exception )
   {
      std::cerr << "Cannot open file " << inputFiles[ 0 ] << std::endl;
   }

   if( verbose )
     std::cout << objectType << " detected ... ";

   const std::vector< String > parsedObjectType = parseObjectType( objectType );
   if( parsedObjectType.empty() )
   {
      std::cerr << "Unable to parse object type " << objectType << "." << std::endl;
      return false;
   }
   setValueType< MeshPointer >( meshPointer, inputFiles[ 0 ], parsedObjectType, parameters );
   return true;
}
