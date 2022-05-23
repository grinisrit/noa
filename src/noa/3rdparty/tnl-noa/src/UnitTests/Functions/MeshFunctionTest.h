#pragma once

#ifdef HAVE_GTEST
#include <gtest/gtest.h>
#include <sstream>
#include <TNL/Functions/MeshFunction.h>
#include <TNL/Meshes/Grid.h>
#include <TNL/Pointers/SharedPointer.h>

TEST( MeshFunctionTest, BasicConstructor )
{
   using Grid = TNL::Meshes::Grid< 2 >;
   TNL::Functions::MeshFunction< Grid > meshFunction;
}

TEST( MeshFunctionTest, OstreamOperatorTest )
{
   using GridType = TNL::Meshes::Grid< 2 >;
   using GridPointer = TNL::Pointers::SharedPointer< GridType >;
   using CoordinatesType = typename GridType::CoordinatesType;
   using MeshFunctionType = TNL::Functions::MeshFunction< GridType >;
   GridPointer grid;
   grid->setDimensions( CoordinatesType( 3, 3 ) );
   MeshFunctionType meshFunction( grid );
   meshFunction.getData().setValue( 1.0 );

   const char* str = "[ 1, 1, 1, 1, 1, 1, 1, 1, 1 ]";
   std::stringstream string_stream1, string_stream2( str );
   string_stream1 << meshFunction;
   EXPECT_EQ( string_stream1.str(), string_stream2.str() );
}


#endif


#include "../main.h"
