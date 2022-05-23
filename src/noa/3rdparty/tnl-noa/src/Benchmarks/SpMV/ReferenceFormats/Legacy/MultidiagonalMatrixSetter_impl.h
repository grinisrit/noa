#pragma once

namespace TNL {
    namespace Benchmarks {
        namespace SpMV {
            namespace ReferenceFormats {
               namespace Legacy {

template< typename MeshReal,
          typename Device,
          typename MeshIndex >
   template< typename Real,
             typename Index >
bool
MultidiagonalMatrixSetter< Meshes::Grid< 1, MeshReal, Device, MeshIndex > >::
setupMatrix( const MeshType& mesh,
             Multidiagonal< Real, Device, Index >& matrix,
             int stencilSize,
             bool crossStencil )
{
   const Index dofs = mesh.template getEntitiesCount< typename MeshType::Cell >();
   matrix.setDimensions( dofs, dofs );
   CoordinatesType centerCell( stencilSize );
   Containers::Vector< Index, Device, Index > diagonals;
   diagonals.setSize( 3 );
   Index centerCellIndex = mesh.getCellIndex( CoordinatesType( stencilSize ) );
   diagonals.setElement( 0, mesh.getCellIndex( CoordinatesType( stencilSize - 1 ) ) - centerCellIndex );
   diagonals.setElement( 1, 0 );
   diagonals.setElement( 2, mesh.getCellIndex( CoordinatesType( stencilSize + 1 ) ) - centerCellIndex );
   //cout << "Setting the multidiagonal matrix offsets to: " << diagonals << std::endl;
   matrix.setDiagonals( diagonals );
   return true;
}

template< typename MeshReal,
          typename Device,
          typename MeshIndex >
   template< typename Real,
             typename Index >
bool
MultidiagonalMatrixSetter< Meshes::Grid< 2, MeshReal, Device, MeshIndex > >::
setupMatrix( const MeshType& mesh,
             Multidiagonal< Real, Device, Index >& matrix,
             int stencilSize,
             bool crossStencil )
{
   const Index dofs = mesh.template getEntitiesCount< typename MeshType::Cell >();
   matrix.setDimensions( dofs, dofs );
   CoordinatesType centerCell( stencilSize );
   Containers::Vector< Index, Device, Index > diagonals;
   diagonals.setSize( 5 );
   Index centerCellIndex = mesh.getCellIndex( CoordinatesType( stencilSize, stencilSize ) );
   diagonals.setElement( 0, mesh.getCellIndex( CoordinatesType( stencilSize, stencilSize - 1 ) ) - centerCellIndex );
   diagonals.setElement( 1, mesh.getCellIndex( CoordinatesType( stencilSize - 1, stencilSize ) ) - centerCellIndex );
   diagonals.setElement( 2, 0 );
   diagonals.setElement( 3, mesh.getCellIndex( CoordinatesType( stencilSize + 1, stencilSize ) ) - centerCellIndex );
   diagonals.setElement( 4, mesh.getCellIndex( CoordinatesType( stencilSize, stencilSize + 1 ) ) - centerCellIndex );
   //cout << "Setting the multidiagonal matrix offsets to: " << diagonals << std::endl;
   matrix.setDiagonals( diagonals );
   return true;
}

template< typename MeshReal,
          typename Device,
          typename MeshIndex >
   template< typename Real,
             typename Index >
bool
MultidiagonalMatrixSetter< Meshes::Grid< 3, MeshReal, Device, MeshIndex > >::
setupMatrix( const MeshType& mesh,
             Multidiagonal< Real, Device, Index >& matrix,
             int stencilSize,
             bool crossStencil )
{
   const Index dofs = mesh.template getEntitiesCount< typename MeshType::Cell >();
   matrix.setDimensions( dofs, dofs );
   CoordinatesType centerCell( stencilSize );
   Containers::Vector< Index, Device, Index > diagonals;
   diagonals.setSize( 7 );
   Index centerCellIndex = mesh.getCellIndex( CoordinatesType( stencilSize, stencilSize, stencilSize ) );
   diagonals.setElement( 0, mesh.getCellIndex( CoordinatesType( stencilSize, stencilSize, stencilSize - 1 ) ) - centerCellIndex );
   diagonals.setElement( 1, mesh.getCellIndex( CoordinatesType( stencilSize, stencilSize - 1, stencilSize ) ) - centerCellIndex );
   diagonals.setElement( 2, mesh.getCellIndex( CoordinatesType( stencilSize - 1, stencilSize, stencilSize ) ) - centerCellIndex );
   diagonals.setElement( 3, 0 );
   diagonals.setElement( 4, mesh.getCellIndex( CoordinatesType( stencilSize + 1, stencilSize, stencilSize ) ) - centerCellIndex );
   diagonals.setElement( 5, mesh.getCellIndex( CoordinatesType( stencilSize, stencilSize + 1, stencilSize ) ) - centerCellIndex );
   diagonals.setElement( 6, mesh.getCellIndex( CoordinatesType( stencilSize, stencilSize, stencilSize + 1 ) ) - centerCellIndex );
   //cout << "Setting the multidiagonal matrix offsets to: " << diagonals << std::endl;
   matrix.setDiagonals( diagonals );
   return true;
}

               } //namespace Legacy
            } //namespace ReferenceFormats
        } //namespace SpMV
    } //namespace Benchmarks
} // namespace TNL
