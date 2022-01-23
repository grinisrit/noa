// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

/***
 * Authors:
 * Oberhuber Tomas, tomas.oberhuber@fjfi.cvut.cz
 * Zabka Vitezslav, zabkav@gmail.com
 */

#pragma once

#include <ostream>
#include <iomanip>

namespace noa::TNL {
namespace Meshes {
namespace Writers {

template< typename Mesh >
class NetgenWriter
{
   using GlobalIndexType = typename Mesh::GlobalIndexType;
   using PointType = typename Mesh::PointType;
   using Cell = typename Mesh::Cell;
   static constexpr int meshDimension = Mesh::getMeshDimension();

public:
   static void writeMesh( const Mesh& mesh, std::ostream& str )
   {
      str << std::setprecision( 6 );
      str << std::fixed;

      const GlobalIndexType numberOfVertices = mesh.template getEntitiesCount< typename Mesh::Vertex >();
      str << numberOfVertices << std::endl;
      for( GlobalIndexType i = 0; i < numberOfVertices; i++ )
      {
         const PointType& point = mesh.template getEntity< typename Mesh::Vertex >( i ).getPoint();
         str << " ";
         for( int d = 0; d < meshDimension; d++ )
            str << " " << point[ d ];
         str << std::endl;
      }

      const GlobalIndexType numberOfCells = mesh.template getEntitiesCount< typename Mesh::Cell >();
      str << numberOfCells << std::endl;
      for( GlobalIndexType cellIdx = 0; cellIdx < numberOfCells; cellIdx++ )
      {
         const Cell& cell = mesh.template getEntity< typename Mesh::Cell >( cellIdx );
         str << "   1";
         for( int cellVertexIdx = 0;
              cellVertexIdx < meshDimension + 1;
              cellVertexIdx++ )
            // note: Netgen has 1-based indices
            str << " " << cell.template getSubentityIndex< 0 >( cellVertexIdx ) + 1;
         str << "\n";
      }
   }
};

} // namespace Writers
} // namespace Meshes
} // namespace noa::TNL
