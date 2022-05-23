#ifndef PURE_C_RHS_H
#define	PURE_C_RHS_H

#ifdef HAVE_CUDA
#include<cuda.h>
#endif

/****
 * Just testing data for measuring performance
 * with different ways of passing data to kernels.
 */
struct Data
{
   double time, tau;
   TNL::Containers::StaticVector< 2, double > c1, c2, c3, c4;
   TNL::Meshes::Grid< 2, double > grid;
};

#ifdef HAVE_CUDA

template< typename Real, typename Index >
__global__ void boundaryConditionsKernel( Real* u,
                                          Real* aux,
                                          const Index gridXSize, const Index gridYSize )
{
   const Index i = ( blockIdx.x ) * blockDim.x + threadIdx.x;
   const Index j = ( blockIdx.y ) * blockDim.y + threadIdx.y;
   if( i == 0 && j < gridYSize )
   {
      aux[ j * gridXSize ] = 0.0;
      u[ j * gridXSize ] = 0.0; //u[ j * gridXSize + 1 ];
   }
   if( i == gridXSize - 1 && j < gridYSize )
   {
      aux[ j * gridXSize + gridYSize - 1 ] = 0.0;
      u[ j * gridXSize + gridYSize - 1 ] = 0.0; //u[ j * gridXSize + gridXSize - 1 ];      
   }
   if( j == 0 && i < gridXSize )
   {
      aux[ i ] = 0.0; //u[ j * gridXSize + 1 ];
      u[ i ] = 0.0; //u[ j * gridXSize + 1 ];
   }
   if( j == gridYSize -1  && i < gridXSize )
   {
      aux[ j * gridXSize + i ] = 0.0; //u[ j * gridXSize + gridXSize - 1 ];      
      u[ j * gridXSize + i ] = 0.0; //u[ j * gridXSize + gridXSize - 1 ];      
   }         
}


template< typename Real, typename Index >
__global__ void heatEquationKernel( const Real* u, 
                                    Real* aux,
                                    const Real tau,
                                    const Real hx_inv,
                                    const Real hy_inv,
                                    const Index gridXSize,
                                    const Index gridYSize )
{
   const Index i = blockIdx.x * blockDim.x + threadIdx.x;
   const Index j = blockIdx.y * blockDim.y + threadIdx.y;
   if( i > 0 && i < gridXSize - 1 &&
       j > 0 && j < gridYSize - 1 )
   {
      const Index c = j * gridXSize + i;
      aux[ c ] = ( ( u[ c - 1 ]         - 2.0 * u[ c ] + u[ c + 1 ]         ) * hx_inv +
                   ( u[ c - gridXSize ] - 2.0 * u[ c ] + u[ c + gridXSize ] ) * hy_inv );
      //aux[ c ] += 0.1;
      //aux[ c ] = ( ( __ldg( &u[ c - 1 ] ) - 2.0 * __ldg( &u[ c ] ) + __ldg( &u[ c + 1 ] ) ) * hx_inv +
      //                   ( __ldg( &u[ c - gridXSize ] ) - 2.0 * __ldg( &u[ c ] ) + __ldg( &u[ c + gridXSize ] ) ) * hy_inv );
   }  
}

template< typename RealType >
bool pureCRhsCuda( dim3 cudaGridSize,
                   dim3 cudaBlockSize,
                   RealType* cuda_u,
                   RealType* cuda_aux,
                   const RealType& tau,
                   const RealType& hx_inv,
                   const RealType& hy_inv,
                   int gridXSize,
                   int gridYSize )
{
   int cudaErr;
   /****
    * Neumann boundary conditions
    */
   //cout << "Setting boundary conditions ... " <<std::endl;
   boundaryConditionsKernel<<< cudaGridSize, cudaBlockSize >>>( cuda_u, cuda_aux, gridXSize, gridYSize );
   if( ( cudaErr = cudaGetLastError() ) != cudaSuccess )
   {
      std::cerr << "Setting of boundary conditions failed. " << cudaErr << std::endl;
      return false;
   }

   /****
    * Laplace operator
    */
   //cout << "Laplace operator ... " <<std::endl;
   heatEquationKernel<<< cudaGridSize, cudaBlockSize >>>
      ( cuda_u, cuda_aux, tau, hx_inv, hy_inv, gridXSize, gridYSize );
   if( cudaGetLastError() != cudaSuccess )
   {
      std::cerr << "Laplace operator failed." << std::endl;
      return false;
   }
   return true;
}

#endif

#endif	/* PURE_C_RHS_H */

