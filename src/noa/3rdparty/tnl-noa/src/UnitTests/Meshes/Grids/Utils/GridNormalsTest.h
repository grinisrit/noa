#pragma once

#ifdef HAVE_GTEST

#include <gtest/gtest.h>

#include <TNL/Containers/StaticVector.h>
#include <TNL/Meshes/GridDetails/NormalsGetter.h>

template<int GridDimension, int EntityOrientation, int EntityDimension>
void compare(const TNL::Containers::StaticVector<GridDimension, int>& expectation) {
   auto normals = TNL::Meshes::NormalsGetter<int, EntityDimension, GridDimension>::template getNormals<EntityOrientation>();

   EXPECT_EQ(normals, expectation) << "Grid Dimension: [" << GridDimension << "], "
                                 << "Entity Orientation: [" << EntityOrientation << "], "
                                 << "Entity Dimension: [" << EntityDimension << "]";
}

TEST(NormalsTestSuite, Normals1DTest) {
   // Grid Dimension, EntityOrientation, EntityDimension
   compare<1, 0, 0>({ 1 });
   compare<1, 0, 1>({ 0 });
}

TEST(NormalsTestSuite, Normals2DTest) {
   // Grid Dimension, EntityOrientation, EntityDimension
   compare<2, 0, 0>({ 1, 1 });

   compare<2, 0, 1>({ 0, 1 });
   compare<2, 1, 1>({ 1, 0 });

   compare<2, 0, 2>({ 0, 0 });
}

TEST(NormalsTestSuite, Normals3DTest) {
    // Grid Dimension, EntityOrientation, EntityDimension
   compare<3, 0, 0>({ 1, 1, 1 });

   compare<3, 0, 1>({ 0, 1, 1 });
   compare<3, 1, 1>({ 1, 0, 1 });
   compare<3, 2, 1>({ 1, 1, 0 });

   compare<3, 0, 2>({ 0, 0, 1 });
   compare<3, 1, 2>({ 0, 1, 0 });
   compare<3, 2, 2>({ 1, 0, 0 });

   compare<3, 0, 3>({ 0, 0, 0 });
}

TEST(NormalsTestSuite, Normals4DTest) {
   // Grid Dimension, EntityOrientation, EntityDimension
   compare<4, 0, 0>({ 1, 1, 1, 1 });

   compare<4, 0, 1>({ 0, 1, 1, 1 });
   compare<4, 1, 1>({ 1, 0, 1, 1 });
   compare<4, 2, 1>({ 1, 1, 0, 1 });
   compare<4, 3, 1>({ 1, 1, 1, 0 });

   compare<4, 0, 2>({ 0, 0, 1, 1 });
   compare<4, 1, 2>({ 0, 1, 0, 1 });
   compare<4, 2, 2>({ 0, 1, 1, 0 });
   compare<4, 3, 2>({ 1, 0, 0, 1 });
   compare<4, 4, 2>({ 1, 0, 1, 0 });
   compare<4, 5, 2>({ 1, 1, 0, 0 });

   compare<4, 0, 3>({ 0, 0, 0, 1 });
   compare<4, 1, 3>({ 0, 0, 1, 0 });
   compare<4, 2, 3>({ 0, 1, 0, 0 });
   compare<4, 3, 3>({ 1, 0, 0, 0 });

   compare<4, 0, 4>({ 0, 0, 0, 0 });
}


#endif
