
#pragma once

#ifdef HAVE_GTEST

   #include <gtest/gtest.h>
   #include "support.h"
   #include <TNL/Meshes/Grid.h>

using Index = int;
using Real = float;
using Device = TNL::Devices::Host;

template< typename Grid, typename Iterator >
void
test( Iterator& iterator,
      Grid& grid,
      typename Grid::CoordinatesType coordinate,
      typename Grid::CoordinatesType normals,
      typename Grid::PointType center,
      bool isBoundary,
      typename Iterator::Index index,
      typename Iterator::Real measure,
      bool next )
{
   static Real precision = 9e-5;

   EXPECT_EQ( coordinate, iterator.getCoordinate() ) << coordinate;
   EXPECT_EQ( normals, iterator.getNormals() ) << coordinate;
   EXPECT_EQ( isBoundary, iterator.isBoundary( grid ) ) << coordinate;
   EXPECT_EQ( index, iterator.getIndex( grid ) ) << coordinate;

   EXPECT_NEAR( measure, iterator.getMeasure( grid ), precision ) << coordinate;

   typename Grid::PointType iteratorCenter = iterator.getCenter( grid );

   for( Index i = 0; i < center.getSize(); i++ )
      EXPECT_NEAR( center[ i ], iteratorCenter[ i ], precision ) << coordinate << " " << iteratorCenter;

   EXPECT_EQ( next, iterator.next() ) << coordinate;
}

TEST( GridCoordinateIteratorTest, Grid1DEntity0DOrientation0Test )
{
   constexpr int Dimension = 1;
   constexpr int EntityDimension = 0;
   constexpr int Orientation = 0;
   using Grid = TNL::Meshes::Grid< Dimension, Real, Device, Index >;
   using Iterator = GridCoordinateIterator< Grid, EntityDimension, Orientation >;

   Grid grid;
   grid.setDimensions( 3 );
   grid.setSpaceSteps( 1. );

   Iterator iterator( grid.getDimensions() );

   test( iterator, grid, { 0 }, { 1 }, { 0. }, true, 0, 0., false );
   test( iterator, grid, { 1 }, { 1 }, { 1. }, false, 1, 0., false );
   test( iterator, grid, { 2 }, { 1 }, { 2. }, false, 2, 0., false );
   test( iterator, grid, { 3 }, { 1 }, { 3. }, true, 3, 0., true );
}

TEST( GridCoordinateIteratorTest, Grid1DEntity1DOrientation0Test )
{
   constexpr int Dimension = 1;
   constexpr int EntityDimension = 1;
   constexpr int Orientation = 0;
   using Grid = TNL::Meshes::Grid< Dimension, Real, Device, Index >;
   using Iterator = GridCoordinateIterator< Grid, EntityDimension, Orientation >;

   Grid grid;
   grid.setDimensions( 3 );
   grid.setSpaceSteps( 1. );

   Iterator iterator( grid.getDimensions() );

   test( iterator, grid, { 0 }, { 0 }, { 0.5 }, true, 0, 1., false );
   test( iterator, grid, { 1 }, { 0 }, { 1.5 }, false, 1, 1., false );
   test( iterator, grid, { 2 }, { 0 }, { 2.5 }, true, 2, 1., true );
}

TEST( GridCoordinateIteratorTest, Grid2DEntity0DOrientation0Test )
{
   constexpr int Dimension = 2;
   constexpr int EntityDimension = 0;
   constexpr int Orientation = 0;
   using Grid = TNL::Meshes::Grid< Dimension, Real, Device, Index >;
   using Iterator = GridCoordinateIterator< Grid, EntityDimension, Orientation >;

   Grid grid;
   grid.setDimensions( 3, 3 );
   grid.setSpaceSteps( 1., 1. );
   grid.setOrigin( 0., 0. );

   Iterator iterator( grid.getDimensions() );

   test( iterator, grid, { 0, 0 }, { 1, 1 }, { 0., 0. }, true, 0, 0., false );
   test( iterator, grid, { 1, 0 }, { 1, 1 }, { 1., 0. }, true, 1, 0., false );
   test( iterator, grid, { 2, 0 }, { 1, 1 }, { 2., 0. }, true, 2, 0., false );
   test( iterator, grid, { 3, 0 }, { 1, 1 }, { 3., 0. }, true, 3, 0., false );

   test( iterator, grid, { 0, 1 }, { 1, 1 }, { 0., 1. }, true, 4, 0., false );
   test( iterator, grid, { 1, 1 }, { 1, 1 }, { 1., 1. }, false, 5, 0., false );
   test( iterator, grid, { 2, 1 }, { 1, 1 }, { 2., 1. }, false, 6, 0., false );
   test( iterator, grid, { 3, 1 }, { 1, 1 }, { 3., 1. }, true, 7, 0., false );

   test( iterator, grid, { 0, 2 }, { 1, 1 }, { 0., 2. }, true, 8, 0., false );
   test( iterator, grid, { 1, 2 }, { 1, 1 }, { 1., 2. }, false, 9, 0., false );
   test( iterator, grid, { 2, 2 }, { 1, 1 }, { 2., 2. }, false, 10, 0., false );
   test( iterator, grid, { 3, 2 }, { 1, 1 }, { 3., 2. }, true, 11, 0., false );

   test( iterator, grid, { 0, 3 }, { 1, 1 }, { 0., 3. }, true, 12, 0., false );
   test( iterator, grid, { 1, 3 }, { 1, 1 }, { 1., 3. }, true, 13, 0., false );
   test( iterator, grid, { 2, 3 }, { 1, 1 }, { 2., 3. }, true, 14, 0., false );
   test( iterator, grid, { 3, 3 }, { 1, 1 }, { 3., 3. }, true, 15, 0., true );

}

TEST( GridCoordinateIteratorTest, Grid2DEntity1DOrientation0Test )
{
   constexpr int Dimension = 2;
   constexpr int EntityDimension = 1;
   constexpr int Orientation = 0;
   using Grid = TNL::Meshes::Grid< Dimension, Real, Device, Index >;
   using Iterator = GridCoordinateIterator< Grid, EntityDimension, Orientation >;

   Grid grid;
   grid.setDimensions( 3, 3 );
   grid.setSpaceSteps( 1., 1. );

   Iterator iterator( grid.getDimensions() );

   test( iterator, grid, { 0, 0 }, { 0, 1 }, { 0.5, 0. }, true, 0, 1., false );
   test( iterator, grid, { 1, 0 }, { 0, 1 }, { 1.5, 0. }, true, 1, 1., false );
   test( iterator, grid, { 2, 0 }, { 0, 1 }, { 2.5, 0. }, true, 2, 1., false );

   test( iterator, grid, { 0, 1 }, { 0, 1 }, { 0.5, 1. }, false, 3, 1., false );
   test( iterator, grid, { 1, 1 }, { 0, 1 }, { 1.5, 1. }, false, 4, 1., false );
   test( iterator, grid, { 2, 1 }, { 0, 1 }, { 2.5, 1. }, false, 5, 1., false );

   test( iterator, grid, { 0, 2 }, { 0, 1 }, { 0.5, 2. }, false, 6, 1., false );
   test( iterator, grid, { 1, 2 }, { 0, 1 }, { 1.5, 2. }, false, 7, 1., false );
   test( iterator, grid, { 2, 2 }, { 0, 1 }, { 2.5, 2. }, false, 8, 1., false );

   test( iterator, grid, { 0, 3 }, { 0, 1 }, { 0.5, 3. }, true, 9, 1., false );
   test( iterator, grid, { 1, 3 }, { 0, 1 }, { 1.5, 3. }, true, 10, 1., false );
   test( iterator, grid, { 2, 3 }, { 0, 1 }, { 2.5, 3. }, true, 11, 1., true );
}

TEST( GridCoordinateIteratorTest, Grid2DEntity1DOrientation1Test )
{
   constexpr int Dimension = 2;
   constexpr int EntityDimension = 1;
   constexpr int Orientation = 1;
   using Grid = TNL::Meshes::Grid< Dimension, Real, Device, Index >;
   using Iterator = GridCoordinateIterator< Grid, EntityDimension, Orientation >;

   Grid grid;
   grid.setDimensions( 3, 3 );
   grid.setSpaceSteps( 1., 1. );

   Iterator iterator( grid.getDimensions() );

   test( iterator, grid, { 0, 0 }, { 1, 0 }, { 0., 0.5  }, true, 12, 1., false );
   test( iterator, grid, { 1, 0 }, { 1, 0 }, { 1., 0.5  }, false, 13, 1., false );
   test( iterator, grid, { 2, 0 }, { 1, 0 }, { 2., 0.5 }, false, 14, 1., false );
   test( iterator, grid, { 3, 0 }, { 1, 0 }, { 3., 0.5 }, true, 15, 1., false );

   test( iterator, grid, { 0, 1 }, { 1, 0 }, { 0., 1.5  }, true, 16, 1., false );
   test( iterator, grid, { 1, 1 }, { 1, 0 }, { 1., 1.5  }, false, 17, 1., false );
   test( iterator, grid, { 2, 1 }, { 1, 0 }, { 2., 1.5 }, false, 18, 1., false );
   test( iterator, grid, { 3, 1 }, { 1, 0 }, { 3., 1.5 }, true, 19, 1., false );

   test( iterator, grid, { 0, 2 }, { 1, 0 }, { 0., 2.5  }, true, 20, 1., false );
   test( iterator, grid, { 1, 2 }, { 1, 0 }, { 1., 2.5  }, false, 21, 1., false );
   test( iterator, grid, { 2, 2 }, { 1, 0 }, { 2., 2.5 }, false, 22, 1., false );
   test( iterator, grid, { 3, 2 }, { 1, 0 }, { 3., 2.5 }, true, 23, 1., true );
}

TEST( GridCoordinateIteratorTest, Grid2DEntity2DOrientation0Test )
{
   constexpr int Dimension = 2;
   constexpr int EntityDimension = 2;
   constexpr int Orientation = 0;
   using Grid = TNL::Meshes::Grid< Dimension, Real, Device, Index >;
   using Iterator = GridCoordinateIterator< Grid, EntityDimension, Orientation >;

   Grid grid;
   grid.setDimensions( 3, 3 );
   grid.setSpaceSteps( 1., 1. );

   Iterator iterator( grid.getDimensions() );

   test( iterator, grid, { 0, 0 }, { 0, 0 }, { 0.5, 0.5 }, true, 0, 1., false );
   test( iterator, grid, { 1, 0 }, { 0, 0 }, { 1.5, 0.5 }, true, 1, 1., false );
   test( iterator, grid, { 2, 0 }, { 0, 0 }, { 2.5, 0.5 }, true, 2, 1., false );

   test( iterator, grid, { 0, 1 }, { 0, 0 }, { 0.5, 1.5 }, true, 3, 1., false );
   test( iterator, grid, { 1, 1 }, { 0, 0 }, { 1.5, 1.5 }, false, 4, 1., false );
   test( iterator, grid, { 2, 1 }, { 0, 0 }, { 2.5, 1.5 }, true, 5, 1., false );

   test( iterator, grid, { 0, 2 }, { 0, 0 }, { 0.5, 2.5 }, true, 6, 1., false );
   test( iterator, grid, { 1, 2 }, { 0, 0 }, { 1.5, 2.5 }, true, 7, 1., false );
   test( iterator, grid, { 2, 2 }, { 0, 0 }, { 2.5, 2.5 }, true, 8, 1., true );
}

TEST( GridCoordinateIteratorTest, Grid3DEntity0DOrientation0Test )
{
   constexpr int Dimension = 3;
   constexpr int EntityDimension = 0;
   constexpr int Orientation = 0;
   using Grid = TNL::Meshes::Grid< Dimension, Real, Device, Index >;
   using Iterator = GridCoordinateIterator< Grid, EntityDimension, Orientation >;

   Grid grid;
   grid.setDimensions( 2, 2, 2 );
   grid.setSpaceSteps( 1., 1., 1. );
   grid.setOrigin( 0., 0., 0.);

   Iterator iterator( grid.getDimensions() );

   test( iterator, grid, { 0, 0, 0 }, { 1, 1, 1 }, { 0., 0., 0. }, true, 0, 0., false );
   test( iterator, grid, { 1, 0, 0 }, { 1, 1, 1 }, { 1., 0., 0. }, true, 1, 0., false );
   test( iterator, grid, { 2, 0, 0 }, { 1, 1, 1 }, { 2., 0., 0. }, true, 2, 0., false );

   test( iterator, grid, { 0, 1, 0 }, { 1, 1, 1 }, { 0., 1., 0. }, true, 3, 0., false );
   test( iterator, grid, { 1, 1, 0 }, { 1, 1, 1 }, { 1., 1., 0. }, true, 4, 0., false );
   test( iterator, grid, { 2, 1, 0 }, { 1, 1, 1 }, { 2., 1., 0. }, true, 5, 0., false );

   test( iterator, grid, { 0, 2, 0 }, { 1, 1, 1 }, { 0., 2., 0. }, true, 6, 0., false );
   test( iterator, grid, { 1, 2, 0 }, { 1, 1, 1 }, { 1., 2., 0. }, true, 7, 0., false );
   test( iterator, grid, { 2, 2, 0 }, { 1, 1, 1 }, { 2., 2., 0. }, true, 8, 0., false );


   test( iterator, grid, { 0, 0, 1 }, { 1, 1, 1 }, { 0., 0., 1. }, true, 9,  0., false );
   test( iterator, grid, { 1, 0, 1 }, { 1, 1, 1 }, { 1., 0., 1. }, true, 10, 0., false );
   test( iterator, grid, { 2, 0, 1 }, { 1, 1, 1 }, { 2., 0., 1. }, true, 11, 0., false );

   test( iterator, grid, { 0, 1, 1 }, { 1, 1, 1 }, { 0., 1., 1. }, true, 12, 0., false );
   test( iterator, grid, { 1, 1, 1 }, { 1, 1, 1 }, { 1., 1., 1. }, false, 13, 0., false );
   test( iterator, grid, { 2, 1, 1 }, { 1, 1, 1 }, { 2., 1., 1. }, true, 14, 0., false );

   test( iterator, grid, { 0, 2, 1 }, { 1, 1, 1 }, { 0., 2., 1. }, true, 15, 0., false );
   test( iterator, grid, { 1, 2, 1 }, { 1, 1, 1 }, { 1., 2., 1. }, true, 16, 0., false );
   test( iterator, grid, { 2, 2, 1 }, { 1, 1, 1 }, { 2., 2., 1. }, true, 17, 0., false );


   test( iterator, grid, { 0, 0, 2 }, { 1, 1, 1 }, { 0., 0., 2. }, true, 18, 0., false );
   test( iterator, grid, { 1, 0, 2 }, { 1, 1, 1 }, { 1., 0., 2. }, true, 19, 0., false );
   test( iterator, grid, { 2, 0, 2 }, { 1, 1, 1 }, { 2., 0., 2. }, true, 20, 0., false );

   test( iterator, grid, { 0, 1, 2 }, { 1, 1, 1 }, { 0., 1., 2. }, true, 21, 0., false );
   test( iterator, grid, { 1, 1, 2 }, { 1, 1, 1 }, { 1., 1., 2. }, true, 22, 0., false );
   test( iterator, grid, { 2, 1, 2 }, { 1, 1, 1 }, { 2., 1., 2. }, true, 23, 0., false );

   test( iterator, grid, { 0, 2, 2 }, { 1, 1, 1 }, { 0., 2., 2. }, true, 24, 0., false );
   test( iterator, grid, { 1, 2, 2 }, { 1, 1, 1 }, { 1., 2., 2. }, true, 25, 0., false );
   test( iterator, grid, { 2, 2, 2 }, { 1, 1, 1 }, { 2., 2., 2. }, true, 26, 0., true );
}

TEST( GridCoordinateIteratorTest, Grid3DEntity1DOrientation0Test )
{
   constexpr int Dimension = 3;
   constexpr int EntityDimension = 1;
   constexpr int Orientation = 0;
   using Grid = TNL::Meshes::Grid< Dimension, Real, Device, Index >;
   using Iterator = GridCoordinateIterator< Grid, EntityDimension, Orientation >;

   Grid grid;
   grid.setDimensions( 2, 2, 2 );
   grid.setSpaceSteps( 1., 1., 1. );

   Iterator iterator( grid.getDimensions() );

   test( iterator, grid, { 0, 0, 0 }, { 0, 1, 1 }, { 0.5, 0., 0. }, true, 0, 1., false );
   test( iterator, grid, { 1, 0, 0 }, { 0, 1, 1 }, { 1.5, 0., 0. }, true, 1, 1., false );

   test( iterator, grid, { 0, 1, 0 }, { 0, 1, 1 }, { 0.5, 1., 0. }, true, 2, 1., false );
   test( iterator, grid, { 1, 1, 0 }, { 0, 1, 1 }, { 1.5, 1., 0. }, true, 3, 1., false );

   test( iterator, grid, { 0, 2, 0 }, { 0, 1, 1 }, { 0.5, 2., 0. }, true, 4, 1., false );
   test( iterator, grid, { 1, 2, 0 }, { 0, 1, 1 }, { 1.5, 2., 0. }, true, 5, 1., false );


   test( iterator, grid, { 0, 0, 1 }, { 0, 1, 1 }, { 0.5, 0., 1. }, true, 6, 1., false );
   test( iterator, grid, { 1, 0, 1 }, { 0, 1, 1 }, { 1.5, 0., 1. }, true, 7, 1., false );

   test( iterator, grid, { 0, 1, 1 }, { 0, 1, 1 }, { 0.5, 1., 1. }, false, 8, 1., false );
   test( iterator, grid, { 1, 1, 1 }, { 0, 1, 1 }, { 1.5, 1., 1. }, false, 9, 1., false );

   test( iterator, grid, { 0, 2, 1 }, { 0, 1, 1 }, { 0.5, 2., 1. }, true, 10, 1., false );
   test( iterator, grid, { 1, 2, 1 }, { 0, 1, 1 }, { 1.5, 2., 1. }, true, 11, 1., false );


   test( iterator, grid, { 0, 0, 2 }, { 0, 1, 1 }, { 0.5, 0., 2. }, true, 12, 1., false );
   test( iterator, grid, { 1, 0, 2 }, { 0, 1, 1 }, { 1.5, 0., 2. }, true, 13, 1., false );

   test( iterator, grid, { 0, 1, 2 }, { 0, 1, 1 }, { 0.5, 1., 2. }, true, 14, 1., false );
   test( iterator, grid, { 1, 1, 2 }, { 0, 1, 1 }, { 1.5, 1., 2. }, true, 15, 1., false );

   test( iterator, grid, { 0, 2, 2 }, { 0, 1, 1 }, { 0.5, 2., 2. }, true, 16, 1., false );
   test( iterator, grid, { 1, 2, 2 }, { 0, 1, 1 }, { 1.5, 2., 2. }, true, 17, 1., true );
}

TEST( GridCoordinateIteratorTest, Grid3DEntity1DOrientation1Test )
{
   constexpr int Dimension = 3;
   constexpr int EntityDimension = 1;
   constexpr int Orientation = 1;
   using Grid = TNL::Meshes::Grid< Dimension, Real, Device, Index >;
   using Iterator = GridCoordinateIterator< Grid, EntityDimension, Orientation >;

   Grid grid;
   grid.setDimensions( 2, 2, 2 );
   grid.setSpaceSteps( 1., 1., 1. );

   Iterator iterator( grid.getDimensions() );

   test( iterator, grid, { 0, 0, 0 }, { 1, 0, 1 }, { 0., 0.5, 0. }, true, 18, 1., false );
   test( iterator, grid, { 1, 0, 0 }, { 1, 0, 1 }, { 1., 0.5, 0. }, true, 19, 1., false );
   test( iterator, grid, { 2, 0, 0 }, { 1, 0, 1 }, { 2., 0.5, 0. }, true, 20, 1., false );

   test( iterator, grid, { 0, 1, 0 }, { 1, 0, 1 }, { 0., 1.5, 0. }, true, 21, 1., false );
   test( iterator, grid, { 1, 1, 0 }, { 1, 0, 1 }, { 1., 1.5, 0. }, true, 22,  1., false );
   test( iterator, grid, { 2, 1, 0 }, { 1, 0, 1 }, { 2., 1.5, 0. }, true, 23,  1., false );

   test( iterator, grid, { 0, 0, 1 }, { 1, 0, 1 }, { 0., 0.5, 1. }, true, 24,  1., false );
   test( iterator, grid, { 1, 0, 1 }, { 1, 0, 1 }, { 1., 0.5, 1. }, false, 25, 1., false );
   test( iterator, grid, { 2, 0, 1 }, { 1, 0, 1 }, { 2., 0.5, 1. }, true, 26, 1., false );

   test( iterator, grid, { 0, 1, 1 }, { 1, 0, 1 }, { 0., 1.5, 1. }, true, 27,  1., false );
   test( iterator, grid, { 1, 1, 1 }, { 1, 0, 1 }, { 1., 1.5, 1. }, false, 28, 1., false );
   test( iterator, grid, { 2, 1, 1 }, { 1, 0, 1 }, { 2., 1.5, 1. }, true, 29,  1., false );

   test( iterator, grid, { 0, 0, 2 }, { 1, 0, 1 }, { 0., 0.5, 2. }, true, 30, 1., false );
   test( iterator, grid, { 1, 0, 2 }, { 1, 0, 1 }, { 1., 0.5, 2. }, true, 31, 1., false );
   test( iterator, grid, { 2, 0, 2 }, { 1, 0, 1 }, { 2., 0.5, 2. }, true, 32, 1., false );

   test( iterator, grid, { 0, 1, 2 }, { 1, 0, 1 }, { 0., 1.5, 2. }, true, 33, 1., false );
   test( iterator, grid, { 1, 1, 2 }, { 1, 0, 1 }, { 1., 1.5, 2. }, true, 34, 1., false );
   test( iterator, grid, { 2, 1, 2 }, { 1, 0, 1 }, { 2., 1.5, 2. }, true, 35, 1., true );
}

TEST( GridCoordinateIteratorTest, Grid3DEntity1DOrientation2Test )
{
   constexpr int Dimension = 3;
   constexpr int EntityDimension = 1;
   constexpr int Orientation = 2;
   using Grid = TNL::Meshes::Grid< Dimension, Real, Device, Index >;
   using Iterator = GridCoordinateIterator< Grid, EntityDimension, Orientation >;

   Grid grid;
   grid.setDimensions( 2, 2, 2 );
   grid.setSpaceSteps( 1., 1., 1. );

   Iterator iterator( grid.getDimensions() );

   test( iterator, grid, { 0, 0, 0 }, { 1, 1, 0 }, { 0., 0., 0.5 }, true, 36, 1., false );
   test( iterator, grid, { 1, 0, 0 }, { 1, 1, 0 }, { 1., 0., 0.5 }, true, 37, 1., false );
   test( iterator, grid, { 2, 0, 0 }, { 1, 1, 0 }, { 2., 0., 0.5 }, true, 38, 1., false );

   test( iterator, grid, { 0, 1, 0 }, { 1, 1, 0 }, { 0., 1., 0.5 }, true, 39, 1., false );
   test( iterator, grid, { 1, 1, 0 }, { 1, 1, 0 }, { 1., 1., 0.5 }, false, 40, 1., false );
   test( iterator, grid, { 2, 1, 0 }, { 1, 1, 0 }, { 2., 1., 0.5 }, true, 41, 1., false );

   test( iterator, grid, { 0, 2, 0 }, { 1, 1, 0 }, { 0., 2., 0.5 }, true, 42, 1., false );
   test( iterator, grid, { 1, 2, 0 }, { 1, 1, 0 }, { 1., 2., 0.5 }, true, 43, 1., false );
   test( iterator, grid, { 2, 2, 0 }, { 1, 1, 0 }, { 2., 2., 0.5 }, true, 44, 1., false );


   test( iterator, grid, { 0, 0, 1 }, { 1, 1, 0 }, { 0., 0., 1.5 }, true, 45, 1., false );
   test( iterator, grid, { 1, 0, 1 }, { 1, 1, 0 }, { 1., 0., 1.5 }, true, 46, 1., false );
   test( iterator, grid, { 2, 0, 1 }, { 1, 1, 0 }, { 2., 0., 1.5 }, true, 47, 1., false );

   test( iterator, grid, { 0, 1, 1 }, { 1, 1, 0 }, { 0., 1., 1.5 }, true, 48, 1., false );
   test( iterator, grid, { 1, 1, 1 }, { 1, 1, 0 }, { 1., 1., 1.5 }, false, 49, 1., false );
   test( iterator, grid, { 2, 1, 1 }, { 1, 1, 0 }, { 2., 1., 1.5 }, true, 50, 1., false );

   test( iterator, grid, { 0, 2, 1 }, { 1, 1, 0 }, { 0., 2., 1.5 }, true, 51, 1., false );
   test( iterator, grid, { 1, 2, 1 }, { 1, 1, 0 }, { 1., 2., 1.5 }, true, 52, 1., false );
   test( iterator, grid, { 2, 2, 1 }, { 1, 1, 0 }, { 2., 2., 1.5 }, true, 53, 1., true );
}

TEST( GridCoordinateIteratorTest, Grid3DEntity2DOrientation0Test )
{
   constexpr int Dimension = 3;
   constexpr int EntityDimension = 2;
   constexpr int Orientation = 0;
   using Grid = TNL::Meshes::Grid< Dimension, Real, Device, Index >;
   using Iterator = GridCoordinateIterator< Grid, EntityDimension, Orientation >;

   Grid grid;
   grid.setDimensions( 2, 2, 2 );
   grid.setSpaceSteps( 1., 1., 1. );

   Iterator iterator( grid.getDimensions() );

   test( iterator, grid, { 0, 0, 0 }, { 0, 0, 1 }, { 0.5, 0.5, 0. }, true, 0, 1., false );
   test( iterator, grid, { 1, 0, 0 }, { 0, 0, 1 }, { 1.5, 0.5, 0. }, true, 1, 1., false );

   test( iterator, grid, { 0, 1, 0 }, { 0, 0, 1 }, { 0.5, 1.5, 0. }, true, 2, 1., false );
   test( iterator, grid, { 1, 1, 0 }, { 0, 0, 1 }, { 1.5, 1.5, 0. }, true, 3, 1., false );

   test( iterator, grid, { 0, 0, 1 }, { 0, 0, 1 }, { 0.5, 0.5, 1. }, false, 4, 1., false );
   test( iterator, grid, { 1, 0, 1 }, { 0, 0, 1 }, { 1.5, 0.5, 1. }, false, 5, 1., false );

   test( iterator, grid, { 0, 1, 1 }, { 0, 0, 1 }, { 0.5, 1.5, 1. }, false, 6, 1., false );
   test( iterator, grid, { 1, 1, 1 }, { 0, 0, 1 }, { 1.5, 1.5, 1. }, false, 7, 1., false );

   test( iterator, grid, { 0, 0, 2 }, { 0, 0, 1 }, { 0.5, 0.5, 2. }, true, 8, 1., false );
   test( iterator, grid, { 1, 0, 2 }, { 0, 0, 1 }, { 1.5, 0.5, 2. }, true, 9, 1., false );

   test( iterator, grid, { 0, 1, 2 }, { 0, 0, 1 }, { 0.5, 1.5, 2. }, true, 10, 1., false );
   test( iterator, grid, { 1, 1, 2 }, { 0, 0, 1 }, { 1.5, 1.5, 2. }, true, 11, 1., true );
}

TEST( GridCoordinateIteratorTest, Grid3DEntity2DOrientation1Test )
{
   constexpr int Dimension = 3;
   constexpr int EntityDimension = 2;
   constexpr int Orientation = 1;
   using Grid = TNL::Meshes::Grid< Dimension, Real, Device, Index >;
   using Iterator = GridCoordinateIterator< Grid, EntityDimension, Orientation >;

   Grid grid;
   grid.setDimensions( 2, 2, 2 );
   grid.setSpaceSteps( 1., 1., 1. );

   Iterator iterator( grid.getDimensions() );

   test( iterator, grid, { 0, 0, 0 }, { 0, 1, 0 }, { 0.5, 0., 0.5 }, true, 12, 1., false );
   test( iterator, grid, { 1, 0, 0 }, { 0, 1, 0 }, { 1.5, 0., 0.5 }, true, 13, 1., false );

   test( iterator, grid, { 0, 1, 0 }, { 0, 1, 0 }, { 0.5, 1., 0.5 }, false, 14, 1., false );
   test( iterator, grid, { 1, 1, 0 }, { 0, 1, 0 }, { 1.5, 1., 0.5 }, false, 15, 1., false );

   test( iterator, grid, { 0, 2, 0 }, { 0, 1, 0 }, { 0.5, 2., 0.5 }, true, 16, 1., false );
   test( iterator, grid, { 1, 2, 0 }, { 0, 1, 0 }, { 1.5, 2., 0.5 }, true, 17, 1., false );

   test( iterator, grid, { 0, 0, 1 }, { 0, 1, 0 }, { 0.5, 0., 1.5 }, true, 18, 1., false );
   test( iterator, grid, { 1, 0, 1 }, { 0, 1, 0 }, { 1.5, 0., 1.5 }, true, 19, 1., false );

   test( iterator, grid, { 0, 1, 1 }, { 0, 1, 0 }, { 0.5, 1., 1.5 }, false, 20, 1., false );
   test( iterator, grid, { 1, 1, 1 }, { 0, 1, 0 }, { 1.5, 1., 1.5 }, false, 21, 1., false );

   test( iterator, grid, { 0, 2, 1 }, { 0, 1, 0 }, { 0.5, 2., 1.5 }, true, 22, 1., false );
   test( iterator, grid, { 1, 2, 1 }, { 0, 1, 0 }, { 1.5, 2., 1.5 }, true, 23, 1., true );
}

TEST( GridCoordinateIteratorTest, Grid3DEntity2DOrientation2Test )
{
   constexpr int Dimension = 3;
   constexpr int EntityDimension = 2;
   constexpr int Orientation = 2;
   using Grid = TNL::Meshes::Grid< Dimension, Real, Device, Index >;
   using Iterator = GridCoordinateIterator< Grid, EntityDimension, Orientation >;

   Grid grid;
   grid.setDimensions( 2, 2, 2 );
   grid.setSpaceSteps( 1., 1., 1. );

   Iterator iterator( grid.getDimensions() );

   test( iterator, grid, { 0, 0, 0 }, { 1, 0, 0 }, { 0., 0.5, 0.5 }, true, 24, 1., false );
   test( iterator, grid, { 1, 0, 0 }, { 1, 0, 0 }, { 1., 0.5, 0.5 }, false, 25, 1., false );
   test( iterator, grid, { 2, 0, 0 }, { 1, 0, 0 }, { 2., 0.5, 0.5 }, true, 26, 1., false );

   test( iterator, grid, { 0, 1, 0 }, { 1, 0, 0 }, { 0., 1.5, 0.5 }, true, 27, 1., false );
   test( iterator, grid, { 1, 1, 0 }, { 1, 0, 0 }, { 1., 1.5, 0.5 }, false, 28, 1., false );
   test( iterator, grid, { 2, 1, 0 }, { 1, 0, 0 }, { 2., 1.5, 0.5 }, true, 29, 1., false );

   test( iterator, grid, { 0, 0, 1 }, { 1, 0, 0 }, { 0., 0.5, 1.5 }, true, 30, 1., false );
   test( iterator, grid, { 1, 0, 1 }, { 1, 0, 0 }, { 1., 0.5, 1.5 }, false, 31, 1., false );
   test( iterator, grid, { 2, 0, 1 }, { 1, 0, 0 }, { 2., 0.5, 1.5 }, true, 32, 1., false );

   test( iterator, grid, { 0, 1, 1 }, { 1, 0, 0 }, { 0., 1.5, 1.5 }, true, 33, 1., false );
   test( iterator, grid, { 1, 1, 1 }, { 1, 0, 0 }, { 1., 1.5, 1.5 }, false, 34, 1., false );
   test( iterator, grid, { 2, 1, 1 }, { 1, 0, 0 }, { 2., 1.5, 1.5 }, true, 35, 1., true );
}

TEST( GridCoordinateIteratorTest, Grid3DEntity3DOrientation0Test )
{
   constexpr int Dimension = 3;
   constexpr int EntityDimension = 3;
   constexpr int Orientation = 0;
   using Grid = TNL::Meshes::Grid< Dimension, Real, Device, Index >;
   using Iterator = GridCoordinateIterator< Grid, EntityDimension, Orientation >;

   Grid grid;
   grid.setDimensions( 2, 2, 2 );
   grid.setSpaceSteps( 1., 1., 1. );

   Iterator iterator( grid.getDimensions() );

   test( iterator, grid, { 0, 0, 0 }, { 0, 0, 0 }, { 0.5, 0.5, 0.5 }, true, 0, 1., false );
   test( iterator, grid, { 1, 0, 0 }, { 0, 0, 0 }, { 1.5, 0.5, 0.5 }, true, 1, 1., false );

   test( iterator, grid, { 0, 1, 0 }, { 0, 0, 0 }, { 0.5, 1.5, 0.5 }, true, 2, 1., false );
   test( iterator, grid, { 1, 1, 0 }, { 0, 0, 0 }, { 1.5, 1.5, 0.5 }, true, 3, 1., false );

   test( iterator, grid, { 0, 0, 1 }, { 0, 0, 0 }, { 0.5, 0.5, 1.5 }, true, 4, 1., false );
   test( iterator, grid, { 1, 0, 1 }, { 0, 0, 0 }, { 1.5, 0.5, 1.5 }, true, 5, 1., false );

   test( iterator, grid, { 0, 1, 1 }, { 0, 0, 0 }, { 0.5, 1.5, 1.5 }, true, 6, 1., false );
   test( iterator, grid, { 1, 1, 1 }, { 0, 0, 0 }, { 1.5, 1.5, 1.5 }, true, 7, 1., true );
}

#endif

#include "../../../main.h"
