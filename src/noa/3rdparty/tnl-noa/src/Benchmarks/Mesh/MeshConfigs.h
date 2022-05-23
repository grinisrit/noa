// Implemented by: Jakub Klinkovsky

#pragma once

#include <TNL/String.h>
#include <TNL/Meshes/Topologies/SubentityVertexMap.h>

template< typename Cell,
          int SpaceDimension = Cell::dimension,
          typename Real = double,
          typename GlobalIndex = int,
          typename LocalIndex = GlobalIndex >
struct FullConfig
{
   using CellTopology = Cell;
   using RealType = Real;
   using GlobalIndexType = GlobalIndex;
   using LocalIndexType = LocalIndex;

   static constexpr int spaceDimension = SpaceDimension;
   static constexpr int meshDimension = Cell::dimension;

   static TNL::String getConfigType()
   {
      return "Full";
   }

   /****
    * Storage of subentities of mesh entities.
    */
   static constexpr bool subentityStorage( int entityDimension, int subentityDimension )
   {
      return true;
   }

   /****
    * Storage of superentities of mesh entities.
    */
   static constexpr bool superentityStorage( int entityDimension, int superentityDimension )
   {
      return true;
   }

   /****
    * Storage of mesh entity tags. Boundary tags are necessary for the mesh traverser.
    */
   static constexpr bool entityTagsStorage( int entityDimension )
   {
      return true;
   }

   /****
    * Storage of the dual graph.
    *
    * If enabled, links from vertices to cells must be stored.
    */
   static constexpr bool dualGraphStorage()
   {
      return true;
   }

   /****
    * Cells must have at least this number of common vertices to be considered
    * as neighbors in the dual graph.
    */
   static constexpr int dualGraphMinCommonVertices = meshDimension;
};

template< typename Cell,
          int SpaceDimension = Cell::dimension,
          typename Real = double,
          typename GlobalIndex = int,
          typename LocalIndex = GlobalIndex >
struct MinimalConfig
{
   using CellTopology = Cell;
   using RealType = Real;
   using GlobalIndexType = GlobalIndex;
   using LocalIndexType = LocalIndex;

   static constexpr int spaceDimension = SpaceDimension;
   static constexpr int meshDimension = Cell::dimension;

   static TNL::String getConfigType()
   {
      return "Minimal";
   }

   /****
    * Storage of subentities of mesh entities.
    */
   static constexpr bool subentityStorage( int entityDimension, int subentityDimension )
   {
      return subentityDimension == 0 || ( subentityDimension == meshDimension - 1 && entityDimension == meshDimension );
   }

   /****
    * Storage of superentities of mesh entities.
    */
   static constexpr bool superentityStorage( int entityDimension, int superentityDimension )
   {
      return ( entityDimension == 0 || entityDimension == meshDimension - 1 ) && superentityDimension == meshDimension;
   }

   /****
    * Storage of mesh entity tags. Boundary tags are necessary for the mesh traverser.
    */
   static constexpr bool entityTagsStorage( int entityDimension )
   {
      return false;
   }

   /****
    * Storage of the dual graph.
    *
    * If enabled, links from vertices to cells must be stored.
    */
   static constexpr bool dualGraphStorage()
   {
      return false;
   }

   /****
    * Cells must have at least this number of common vertices to be considered
    * as neighbors in the dual graph.
    */
   static constexpr int dualGraphMinCommonVertices = meshDimension;
};
