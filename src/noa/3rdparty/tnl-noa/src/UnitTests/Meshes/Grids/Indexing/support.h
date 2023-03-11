#pragma once

template<typename Grid, int EntityDimension>
void testGetEntityFromIndex( Grid& grid,
                             const typename Grid::CoordinatesType& dimensions,
                             const typename Grid::PointType& origin = typename Grid::PointType(0),
                             const typename Grid::PointType& spaceSteps = typename Grid::PointType(1)) {
   SCOPED_TRACE("Grid Dimension: " + TNL::convertToString(Grid::getMeshDimension()));
   SCOPED_TRACE("Entity Dimension: " + TNL::convertToString(EntityDimension));
   SCOPED_TRACE("Dimension: " + TNL::convertToString(dimensions));
   SCOPED_TRACE("Origin:" + TNL::convertToString(origin));
   SCOPED_TRACE("Space steps:" + TNL::convertToString(spaceSteps));

   EXPECT_NO_THROW(grid.setDimensions(dimensions)) << "Verify, that the set of" << dimensions << " doesn't cause assert";
   EXPECT_NO_THROW(grid.setOrigin(origin)) << "Verify, that the set of" << origin << "doesn't cause assert";
   EXPECT_NO_THROW(grid.setSpaceSteps(spaceSteps)) << "Verify, that the set of" << spaceSteps << "doesn't cause assert";

   grid.template forAllEntities< EntityDimension >( [=] ( TNL::Meshes::GridEntity< Grid, EntityDimension >& entity ) mutable {
      auto new_entity = TNL::Meshes::GridEntity< Grid, EntityDimension >( entity.getGrid(), entity.getIndex() );
      EXPECT_EQ( new_entity.getCoordinates(), entity.getCoordinates() );
      EXPECT_EQ( new_entity.getNormals(), entity.getNormals() );
   } );
}
