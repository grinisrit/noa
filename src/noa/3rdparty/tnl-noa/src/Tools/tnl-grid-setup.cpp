#include "tnl-grid-setup.h"
#include <TNL/Config/parseCommandLine.h>

void configSetup( Config::ConfigDescription& config )
{
   config.addDelimiter( "General parameters" );
   config.addEntry        < String >( "output-file",   "Output file.", "mesh.vti" );
   config.addEntry        < int >   ( "verbose",       "Set the verbosity of the program.", 1 );

   config.addDelimiter              ( "Grid parameters" );
   config.addRequiredEntry< int >   ( "dimension",     "The grid dimension." );
      config.addEntryEnum< int >( 1 );
      config.addEntryEnum< int >( 2 );
      config.addEntryEnum< int >( 3 );
   config.addEntry        < String >( "real-type",     "Precision of the real type describing the grid.", "double" );
      config.addEntryEnum < String >( "float" );
      config.addEntryEnum < String >( "double" );
      config.addEntryEnum < String >( "long-double" );
   config.addEntry        < String >( "index-type",    "Type for the indexing of the grid elements." ,"int" );
      config.addEntryEnum < String >( "int" );
      config.addEntryEnum < String >( "long-int" );
   config.addEntry        < double >( "origin-x",          "The x-coordinate of the origin.", 0.0 );
   config.addEntry        < double >( "origin-y",          "The y-coordinate of the origin.", 0.0 );
   config.addEntry        < double >( "origin-z",          "The z-coordinate of the origin.", 0.0 );
   config.addEntry        < double >( "proportions-x",     "The proportions of the grid along the x axis.", 1.0 );
   config.addEntry        < double >( "proportions-y",     "The proportions of the grid along the y axis.", 1.0 );
   config.addEntry        < double >( "proportions-z",     "The proportions of the grid along the z axis.", 1.0 );
   config.addEntry        < int >   ( "size-x",            "Number of elements along the x axis." );
   config.addEntry        < int >   ( "size-y",            "Number of elements along the y axis." );
   config.addEntry        < int >   ( "size-z",            "Number of elements along the z axis." );
   config.addEntry        < bool >  ( "equal-space-steps", "All space steps will be equivalent.", false );
}

int main( int argc, char* argv[] )
{
   Config::ParameterContainer parameters;
   Config::ConfigDescription conf_desc;
   configSetup( conf_desc );
   if( ! parseCommandLine( argc, argv, conf_desc, parameters ) )
      return EXIT_FAILURE;
   if( ! resolveRealType( parameters ) )
      return EXIT_FAILURE;
   return EXIT_SUCCESS;
}
