#include <TNL/Config/parseCommandLine.h>
#include <TNL/FileName.h>
#include <TNL/Meshes/Grid.h>
#include <TNL/Pointers/SharedPointer.h>
#include <TNL/Functions/MeshFunction.h>
#include <TNL/Images/PGMImage.h>
#include <TNL/Images/PNGImage.h>
#include <TNL/Images/JPEGImage.h>
#include <TNL/Images/RegionOfInterest.h>

using namespace TNL;

void configSetup( Config::ConfigDescription& config )
{
   config.addDelimiter( "General parameters" );
   config.addList< String >( "input-images", "Input images for conversion to VTI files." );
   config.addList< String >( "input-files", "Input VTI files for conversion to images." );
   config.addEntry< String >( "image-format", "Output images file format.", "pgm" );
   config.addEntry< String >( "mesh-function-name", "Name of the mesh function in the VTI files.", "image" );
   config.addEntry< String >( "real-type", "Output mesh function real type.", "double" );
      config.addEntryEnum< String >( "float" );
      config.addEntryEnum< String >( "double" );
      config.addEntryEnum< String >( "long-double" );
   config.addEntry< int >( "roi-top",    "Top (smaller number) line of the region of interest.", -1 );
   config.addEntry< int >( "roi-bottom", "Bottom (larger number) line of the region of interest.", -1 );
   config.addEntry< int >( "roi-left",   "Left (smaller number) column of the region of interest.", -1 );
   config.addEntry< int >( "roi-right",  "Right (larger number) column of the region of interest.", -1 );
}

template< typename Real >
bool processImages( const Config::ParameterContainer& parameters )
{
   const std::vector< std::string > inputImages = parameters.getParameter< std::vector< std::string > >( "input-images" );
   const std::string meshFunctionName = parameters.getParameter< std::string >( "mesh-function-name" );

   using GridType = Meshes::Grid< 2, Real, Devices::Host, int >;
   using GridPointer = Pointers::SharedPointer< GridType >;
   using MeshFunctionType = Functions::MeshFunction< GridType >;
   GridPointer grid;
   MeshFunctionType meshFunction;

   Images::RegionOfInterest< int > roi;
   for( const auto& fileName : inputImages ) {
      const String outputFileName = removeFileNameExtension( fileName ) + ".vti";
      std::cout << "Processing image file " << fileName << "... ";
      Images::PGMImage< int > pgmImage;
      if( pgmImage.openForRead( fileName ) )
      {
         std::cout << "PGM format detected ...";
         if( ! roi.check( &pgmImage ) )
            return false;
         meshFunction.setMesh( grid );
         if( ! pgmImage.read( roi, meshFunction ) )
            return false;
         std::cout << "Writing image data to " << outputFileName << std::endl;
         meshFunction.write( meshFunctionName, outputFileName );
         pgmImage.close();
         continue;
      }
      Images::PNGImage< int > pngImage;
      if( pngImage.openForRead( fileName ) )
      {
         std::cout << "PNG format detected ...";
         if( ! roi.check( &pngImage ) )
            return false;
         meshFunction.setMesh( grid );
         if( ! pngImage.read( roi, meshFunction ) )
            return false;
         std::cout << "Writing image data to " << outputFileName << std::endl;
         meshFunction.write( meshFunctionName, outputFileName );
         pngImage.close();
         continue;
      }
      Images::JPEGImage< int > jpegImage;
      if( jpegImage.openForRead( fileName ) )
      {
         std::cout << "JPEG format detected ...";
         if( ! roi.check( &jpegImage ) )
            return false;
         meshFunction.setMesh( grid );
         if( ! jpegImage.read( roi, meshFunction ) )
            return false;
         std::cout << "Writing image data to " << outputFileName << std::endl;
         meshFunction.write( meshFunctionName, outputFileName );
         jpegImage.close();
         continue;
      }
   }
   return true;
}

bool processFiles( const Config::ParameterContainer& parameters )
{
   const std::vector< std::string > inputFiles = parameters.getParameter< std::vector< std::string > >( "input-files" );
   const std::string imageFormat = parameters.getParameter< std::string >( "image-format" );
   const std::string meshFunctionName = parameters.getParameter< std::string >( "mesh-function-name" );

   for( const auto& fileName : inputFiles ) {
      std::cout << "Processing file " << fileName << "... ";
      using Real = double;
      using GridType = Meshes::Grid< 2, Real, Devices::Host, int >;
      using GridPointer = Pointers::SharedPointer< GridType >;
      using MeshFunctionType = Functions::MeshFunction< GridType >;
      GridPointer grid;
      MeshFunctionType meshFunction;
      if( ! Functions::readMeshFunction( meshFunction, meshFunctionName, fileName ) )
         return false;

      if( imageFormat == "pgm" || imageFormat == "pgm-binary" || imageFormat == "pgm-ascii" )
      {
         Images::PGMImage< int > image;
         const String outputFileName = removeFileNameExtension( fileName ) + ".pgm";
         if ( imageFormat == "pgm" || imageFormat == "pgm-binary")
            image.openForWrite( outputFileName, *grid, true );
         if ( imageFormat == "pgm-ascii" )
            image.openForWrite( outputFileName, *grid, false );
         image.write( *grid, meshFunction.getData() );
         image.close();
         continue;
      }
      if( imageFormat == "png" )
      {
         Images::PNGImage< int > image;
         const String outputFileName = removeFileNameExtension( fileName ) + ".png";
         image.openForWrite( outputFileName, *grid );
         image.write( *grid, meshFunction.getData() );
         image.close();
      }
      if( imageFormat == "jpg" )
      {
         Images::JPEGImage< int > image;
         const String outputFileName = removeFileNameExtension( fileName ) + ".jpg";
         image.openForWrite( outputFileName, *grid );
         image.write( *grid, meshFunction.getData() );
         image.close();
      }
   }
   return true;
}

int main( int argc, char* argv[] )
{
   Config::ParameterContainer parameters;
   Config::ConfigDescription configDescription;
   configSetup( configDescription );
   if( ! parseCommandLine( argc, argv, configDescription, parameters ) )
      return EXIT_FAILURE;
   if( ! parameters.checkParameter( "input-images" ) &&
       ! parameters.checkParameter( "input-files") )
   {
       std::cerr << "Neither input images nor input .tnl files are given." << std::endl;
       Config::printUsage( configDescription, argv[ 0 ] );
       return EXIT_FAILURE;
   }
   if( parameters.checkParameter( "input-images" ) )
   {
      const String& realType = parameters.getParameter< String >( "real-type" );
      if( realType == "float" &&  ! processImages< float >( parameters ) )
         return EXIT_FAILURE;
      if( realType == "double" &&  ! processImages< double >( parameters ) )
         return EXIT_FAILURE;
   }
   if( parameters.checkParameter( "input-files" ) && ! processFiles( parameters ) )
      return EXIT_FAILURE;

   return EXIT_SUCCESS;
}
