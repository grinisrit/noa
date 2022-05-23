#ifndef BenchmarkLaplace_IMPL_H
#define BenchmarkLaplace_IMPL_H

/****
 * 1D problem
 */
template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
template< typename MeshFunction, typename MeshEntity >
__cuda_callable__
Real
BenchmarkLaplace< Meshes::Grid< 1, MeshReal, Device, MeshIndex >, Real, Index >::
operator()( const MeshFunction& u,
            const MeshEntity& entity,
            const Real& time ) const
{
   /****
    * Implement your explicit form of the differential operator here.
    * The following example is the Laplace operator approximated 
    * by the Finite difference method.
    */
    static_assert( MeshEntity::getEntityDimension() == 1, "Wrong mesh entity dimension." ); 
    static_assert( MeshFunction::getEntitiesDimension() == 1, "Wrong preimage function" ); 
    const typename MeshEntity::template NeighborEntities< 1 >& neighborEntities = entity.getNeighborEntities(); 

   const RealType& hxSquareInverse = entity.getMesh().template getSpaceStepsProducts< -2 >(); 
   const IndexType& center = entity.getIndex(); 
   const IndexType& east = neighborEntities.template getEntityIndex< 1 >(); 
   const IndexType& west = neighborEntities.template getEntityIndex< -1 >(); 
   return ( u[ west ] - 2.0 * u[ center ]  + u[ east ] ) * hxSquareInverse;
}

template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
template< typename MeshEntity >
__cuda_callable__
Index
BenchmarkLaplace< Meshes::Grid< 1, MeshReal, Device, MeshIndex >, Real, Index >::
getLinearSystemRowLength( const MeshType& mesh,
                          const IndexType& index,
                          const MeshEntity& entity ) const
{
   /****
    * Return a number of non-zero elements in a line (associated with given grid element) of
    * the linear system.
    * The following example is the Laplace operator approximated 
    * by the Finite difference method.
    */

   return 2*Dimension + 1;
}

template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
   template< typename MeshEntity, typename Vector, typename MatrixRow >
__cuda_callable__
void
BenchmarkLaplace< Meshes::Grid< 1, MeshReal, Device, MeshIndex >, Real, Index >::
setMatrixElements( const RealType& time,
                    const RealType& tau,
                    const MeshType& mesh,
                    const IndexType& index,
                    const MeshEntity& entity,
                    const MeshFunctionType& u,
                    Vector& b,
                    MatrixRow& matrixRow ) const
{
   /****
    * Setup the non-zero elements of the linear system here.
    * The following example is the Laplace operator appriximated 
    * by the Finite difference method.
    */

    const typename MeshEntity::template NeighborEntities< 1 >& neighborEntities = entity.getNeighborEntities(); 
   const RealType& lambdaX = tau * entity.getMesh().template getSpaceStepsProducts< -2 >(); 
   const IndexType& center = entity.getIndex(); 
   const IndexType& east = neighborEntities.template getEntityIndex< 1 >(); 
   const IndexType& west = neighborEntities.template getEntityIndex< -1 >(); 
   matrixRow.setElement( 0, west,   - lambdaX );
   matrixRow.setElement( 1, center, 2.0 * lambdaX );
   matrixRow.setElement( 2, east,   - lambdaX );
}

/****
 * 2D problem
 */
template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
template< typename MeshFunction, typename MeshEntity >
__cuda_callable__
Real
BenchmarkLaplace< Meshes::Grid< 2, MeshReal, Device, MeshIndex >, Real, Index >::
operator()( const MeshFunction& u,
            const MeshEntity& entity,
            const Real& time ) const
{
   /****
    * Implement your explicit form of the differential operator here.
    * The following example is the Laplace operator approximated 
    * by the Finite difference method.
    */
   /*static_assert( MeshEntity::getEntityDimension() == 2, "Wrong mesh entity dimension." ); 
   static_assert( MeshFunction::getEntitiesDimension() == 2, "Wrong preimage function" ); 
   const typename MeshEntity::template NeighborEntities< 2 >& neighborEntities = entity.getNeighborEntities(); 

   const RealType& hxSquareInverse = entity.getMesh().template getSpaceStepsProducts< -2, 0 >(); 
   const RealType& hySquareInverse = entity.getMesh().template getSpaceStepsProducts< 0, -2 >(); 
   const IndexType& center = entity.getIndex(); 
   const IndexType& east  = neighborEntities.template getEntityIndex<  1,  0 >(); 
   const IndexType& west  = neighborEntities.template getEntityIndex< -1,  0 >(); 
   const IndexType& north = neighborEntities.template getEntityIndex<  0,  1 >(); 
   const IndexType& south = neighborEntities.template getEntityIndex<  0, -1 >(); */

   const IndexType& xSize = entity.getMesh().getDimensions().x();
   const IndexType& c = entity.getIndex();
   const RealType& hxSquareInverse = entity.getMesh().template getSpaceStepsProducts< -2, 0 >(); 
   const RealType& hySquareInverse = entity.getMesh().template getSpaceStepsProducts< 0, -2 >(); 
   return ( u[ c - 1 ] - 2.0 * u[ c ] + u[ c + 1 ]  ) * hxSquareInverse +
          ( u[ c - xSize ] - 2.0 * u[ c ] + u[ c + xSize ] ) * hySquareInverse;
}

template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
template< typename MeshEntity >
__cuda_callable__
Real
BenchmarkLaplace< Meshes::Grid< 2, MeshReal, Device, MeshIndex >, Real, Index >::
operator()( const RealType* u,
            const MeshEntity& entity,
            const Real& time ) const
{
   /****
    * Implement your explicit form of the differential operator here.
    * The following example is the Laplace operator approximated 
    * by the Finite difference method.
    */
   /*static_assert( MeshEntity::getEntityDimension() == 2, "Wrong mesh entity dimension." ); 
   static_assert( MeshFunction::getEntitiesDimension() == 2, "Wrong preimage function" ); 
   const typename MeshEntity::template NeighborEntities< 2 >& neighborEntities = entity.getNeighborEntities(); 

   const RealType& hxSquareInverse = entity.getMesh().template getSpaceStepsProducts< -2, 0 >(); 
   const RealType& hySquareInverse = entity.getMesh().template getSpaceStepsProducts< 0, -2 >(); 
   const IndexType& center = entity.getIndex(); 
   const IndexType& east  = neighborEntities.template getEntityIndex<  1,  0 >(); 
   const IndexType& west  = neighborEntities.template getEntityIndex< -1,  0 >(); 
   const IndexType& north = neighborEntities.template getEntityIndex<  0,  1 >(); 
   const IndexType& south = neighborEntities.template getEntityIndex<  0, -1 >(); */

   const IndexType& xSize = entity.getMesh().getDimensions().x();
   const IndexType& c = entity.getIndex();
   const RealType& hxSquareInverse = entity.getMesh().template getSpaceStepsProducts< -2, 0 >(); 
   const RealType& hySquareInverse = entity.getMesh().template getSpaceStepsProducts< 0, -2 >(); 
   return ( u[ c - 1 ] - 2.0 * u[ c ] + u[ c + 1 ]  ) * hxSquareInverse +
          ( u[ c - xSize ] - 2.0 * u[ c ] + u[ c + xSize ] ) * hySquareInverse;
}

template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
   //template< typename MeshFunction > //, typename MeshEntity >
__cuda_callable__
Real
BenchmarkLaplace< Meshes::Grid< 2, MeshReal, Device, MeshIndex >, Real, Index >::
operator()( const MeshType& mesh,
            const Real* u,
            const IndexType& entityIndex,
            const typename MeshType::CoordinatesType coordinates,
            const RealType& time ) const
{
   //const MeshType& mesh = u.template getMesh< Device >();
   const IndexType& xSize = mesh.getDimensions().x();
   const IndexType& c = entityIndex;
   const RealType& hxSquareInverse = mesh.template getSpaceStepsProducts< -2, 0 >(); 
   const RealType& hySquareInverse = mesh.template getSpaceStepsProducts< 0, -2 >(); 
   return ( u[ c - 1 ] - 2.0 * u[ c ] + u[ c + 1 ]  ) * hxSquareInverse +
          ( u[ c - xSize ] - 2.0 * u[ c ] + u[ c + xSize ] ) * hySquareInverse;
}

template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
template< typename MeshEntity >
__cuda_callable__
Index
BenchmarkLaplace< Meshes::Grid< 2, MeshReal, Device, MeshIndex >, Real, Index >::
getLinearSystemRowLength( const MeshType& mesh,
                          const IndexType& index,
                          const MeshEntity& entity ) const
{
   /****
    * Return a number of non-zero elements in a line (associated with given grid element) of
    * the linear system.
    * The following example is the Laplace operator approximated 
    * by the Finite difference method.
    */

   return 2*Dimension + 1;
}

template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
   template< typename MeshEntity, typename Vector, typename MatrixRow >
__cuda_callable__
void
BenchmarkLaplace< Meshes::Grid< 2, MeshReal, Device, MeshIndex >, Real, Index >::
setMatrixElements( const RealType& time,
                    const RealType& tau,
                    const MeshType& mesh,
                    const IndexType& index,
                    const MeshEntity& entity,
                    const MeshFunctionType& u,
                    Vector& b,
                    MatrixRow& matrixRow ) const
{
   /****
    * Setup the non-zero elements of the linear system here.
    * The following example is the Laplace operator appriximated 
    * by the Finite difference method.
    */

    const typename MeshEntity::template NeighborEntities< 2 >& neighborEntities = entity.getNeighborEntities(); 
   const RealType& lambdaX = tau * entity.getMesh().template getSpaceStepsProducts< -2, 0 >(); 
   const RealType& lambdaY = tau * entity.getMesh().template getSpaceStepsProducts< 0, -2 >(); 
   const IndexType& center = entity.getIndex(); 
   const IndexType& east  = neighborEntities.template getEntityIndex<  1,  0 >(); 
   const IndexType& west  = neighborEntities.template getEntityIndex< -1,  0 >(); 
   const IndexType& north = neighborEntities.template getEntityIndex<  0,  1 >(); 
   const IndexType& south = neighborEntities.template getEntityIndex<  0, -1 >(); 
   matrixRow.setElement( 0, south,  -lambdaY );
   matrixRow.setElement( 1, west,   -lambdaX );
   matrixRow.setElement( 2, center, 2.0 * ( lambdaX + lambdaY ) );
   matrixRow.setElement( 3, east,   -lambdaX );
   matrixRow.setElement( 4, north,  -lambdaY );
}

/****
 * 3D problem
 */
template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
template< typename MeshFunction, typename MeshEntity >
__cuda_callable__
Real
BenchmarkLaplace< Meshes::Grid< 3, MeshReal, Device, MeshIndex >, Real, Index >::
operator()( const MeshFunction& u,
            const MeshEntity& entity,
            const Real& time ) const
{
   /****
    * Implement your explicit form of the differential operator here.
    * The following example is the Laplace operator approximated 
    * by the Finite difference method.
    */
    static_assert( MeshEntity::getEntityDimension() == 3, "Wrong mesh entity dimension." ); 
    static_assert( MeshFunction::getEntitiesDimension() == 3, "Wrong preimage function" ); 
    const typename MeshEntity::template NeighborEntities< 3 >& neighborEntities = entity.getNeighborEntities(); 

   const RealType& hxSquareInverse = entity.getMesh().template getSpaceStepsProducts< -2,  0,  0 >(); 
   const RealType& hySquareInverse = entity.getMesh().template getSpaceStepsProducts<  0, -2,  0 >(); 
   const RealType& hzSquareInverse = entity.getMesh().template getSpaceStepsProducts<  0,  0, -2 >(); 
   const IndexType& center = entity.getIndex(); 
   const IndexType& east  = neighborEntities.template getEntityIndex<  1,  0,  0 >(); 
   const IndexType& west  = neighborEntities.template getEntityIndex< -1,  0,  0 >(); 
   const IndexType& north = neighborEntities.template getEntityIndex<  0,  1,  0 >(); 
   const IndexType& south = neighborEntities.template getEntityIndex<  0, -1,  0 >(); 
   const IndexType& up    = neighborEntities.template getEntityIndex<  0,  0,  1 >(); 
   const IndexType& down  = neighborEntities.template getEntityIndex<  0,  0, -1 >(); 
   return ( u[ west ] - 2.0 * u[ center ] + u[ east ]  ) * hxSquareInverse +
          ( u[ south ] - 2.0 * u[ center ] + u[ north ] ) * hySquareInverse +
          ( u[ up ] - 2.0 * u[ center ] + u[ down ] ) * hzSquareInverse;
}

template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
template< typename MeshEntity >
__cuda_callable__
Index
BenchmarkLaplace< Meshes::Grid< 3, MeshReal, Device, MeshIndex >, Real, Index >::
getLinearSystemRowLength( const MeshType& mesh,
                          const IndexType& index,
                          const MeshEntity& entity ) const
{
   /****
    * Return a number of non-zero elements in a line (associated with given grid element) of
    * the linear system.
    * The following example is the Laplace operator approximated 
    * by the Finite difference method.
    */

   return 2*Dimension + 1;
}

template< typename MeshReal,
          typename Device,
          typename MeshIndex,
          typename Real,
          typename Index >
   template< typename MeshEntity, typename Vector, typename MatrixRow >
__cuda_callable__
void
BenchmarkLaplace< Meshes::Grid< 3, MeshReal, Device, MeshIndex >, Real, Index >::
setMatrixElements( const RealType& time,
                    const RealType& tau,
                    const MeshType& mesh,
                    const IndexType& index,
                    const MeshEntity& entity,
                    const MeshFunctionType& u,
                    Vector& b,
                    MatrixRow& matrixRow ) const
{
   /****
    * Setup the non-zero elements of the linear system here.
    * The following example is the Laplace operator appriximated 
    * by the Finite difference method.
    */

    const typename MeshEntity::template NeighborEntities< 3 >& neighborEntities = entity.getNeighborEntities(); 
   const RealType& lambdaX = tau * entity.getMesh().template getSpaceStepsProducts< -2,  0,  0 >(); 
   const RealType& lambdaY = tau * entity.getMesh().template getSpaceStepsProducts<  0, -2,  0 >(); 
   const RealType& lambdaZ = tau * entity.getMesh().template getSpaceStepsProducts<  0,  0, -2 >(); 
   const IndexType& center = entity.getIndex(); 
   const IndexType& east  = neighborEntities.template getEntityIndex<  1,  0,  0 >(); 
   const IndexType& west  = neighborEntities.template getEntityIndex< -1,  0,  0 >(); 
   const IndexType& north = neighborEntities.template getEntityIndex<  0,  1,  0 >(); 
   const IndexType& south = neighborEntities.template getEntityIndex<  0, -1,  0 >(); 
   const IndexType& up    = neighborEntities.template getEntityIndex<  0,  0,  1 >(); 
   const IndexType& down  = neighborEntities.template getEntityIndex<  0,  0, -1 >(); 
   matrixRow.setElement( 0, down,   -lambdaZ );
   matrixRow.setElement( 1, south,  -lambdaY );
   matrixRow.setElement( 2, west,   -lambdaX );
   matrixRow.setElement( 3, center, 2.0 * ( lambdaX + lambdaY + lambdaZ ) );
   matrixRow.setElement( 4, east,   -lambdaX );
   matrixRow.setElement( 5, north,  -lambdaY );
   matrixRow.setElement( 6, up,     -lambdaZ );
}

#endif	/* BenchmarkLaplaceIMPL_H */

