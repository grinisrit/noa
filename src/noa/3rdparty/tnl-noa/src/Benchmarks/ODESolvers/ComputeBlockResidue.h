// Implemented by: Tomas Oberhuber


#pragma once

namespace TNL {
   namespace Benchmarks {

#ifdef __CUDACC__
template< typename Real, typename Index >
__device__ void computeBlockResidue( Real* du,
                                     Real* blockResidue,
                                     Index n )
{
   if( n == 128 || n ==  64 || n ==  32 || n ==  16 ||
       n ==   8 || n ==   4 || n ==   2 || n == 256 ||
       n == 512 )
    {
       if( blockDim.x >= 512 )
       {
          if( threadIdx.x < 256 )
             du[ threadIdx.x ] += du[ threadIdx.x  + 256 ];
          __syncthreads();
       }
       if( blockDim.x >= 256 )
       {
          if( threadIdx.x < 128 )
             du[ threadIdx.x ] += du[ threadIdx.x  + 128 ];
          __syncthreads();
       }
       if( blockDim.x >= 128 )
       {
          if( threadIdx.x < 64 )
             du[ threadIdx.x ] += du[ threadIdx.x  + 64 ];
          __syncthreads();
       }

       /***
        * This runs in one warp so it is synchronized implicitly.
        */
       if ( threadIdx.x < 32)
       {
          if( blockDim.x >= 64 )
             du[ threadIdx.x ] += du[ threadIdx.x  + 32 ];
          if( blockDim.x >= 32 )
             du[ threadIdx.x ] += du[ threadIdx.x  + 16 ];
          if( blockDim.x >= 16 )
             du[ threadIdx.x ] += du[ threadIdx.x  + 8 ];
          if( blockDim.x >=  8 )
             du[ threadIdx.x ] += du[ threadIdx.x  + 4 ];
          if( blockDim.x >=  4 )
             du[ threadIdx.x ] += du[ threadIdx.x  + 2 ];
          if( blockDim.x >=  2 )
             du[ threadIdx.x ] += du[ threadIdx.x  + 1 ];
       }
    }
    else
    {
       int s;
       if( n >= 512 )
       {
          s = n / 2;
          if( threadIdx.x < s )
             du[ threadIdx.x ] += du[ threadIdx.x + s ];

          __syncthreads();
          if( 2 * s < n  && threadIdx.x == n - 1 )
             du[ 0 ] += du[ threadIdx.x ];
          n = s;
          __syncthreads();
       }
       if( n >= 256 )
       {
          s = n / 2;
          if( threadIdx.x < s )
             du[ threadIdx.x ] += du[ threadIdx.x + s ];

          __syncthreads();
          if( 2 * s < n  && threadIdx.x == n - 1 )
             du[ 0 ] += du[ threadIdx.x ];
          n = s;
          __syncthreads();
       }
       if( n >= 128 )
       {
          s = n / 2;
          if( threadIdx.x < s )
             du[ threadIdx.x ] += du[ threadIdx.x + s ];

          __syncthreads();
          if( 2 * s < n  && threadIdx.x == n - 1 )
             du[ 0 ] += du[ threadIdx.x ];
          n = s;
          __syncthreads();
       }
       if( n >= 64 )
       {
          s = n / 2;
          if( threadIdx.x < s )
             du[ threadIdx.x ] += du[ threadIdx.x + s ];

          __syncthreads();
          if( 2 * s < n  && threadIdx.x == n - 1 )
             du[ 0 ] += du[ threadIdx.x ];
          n = s;
          __syncthreads();

       }
       if( n >= 32 )
       {
          s = n / 2;
          if( threadIdx.x < s )
             du[ threadIdx.x ] += du[ threadIdx.x + s ];

          __syncthreads();
          if( 2 * s < n  && threadIdx.x == n - 1 )
             du[ 0 ] += du[ threadIdx.x ];
          n = s;
          __syncthreads();
       }
       /***
        * This runs in one warp so it is synchronised implicitly.
        */
       if( n >= 16 )
       {
          s = n / 2;
          if( threadIdx.x < s )
             du[ threadIdx.x ] += du[ threadIdx.x + s ];
          if( 2 * s < n  && threadIdx.x == n - 1 )
             du[ 0 ] += du[ threadIdx.x ];
          n = s;
       }
       if( n >= 8 )
       {
          s = n / 2;
          if( threadIdx.x < s )
             du[ threadIdx.x ] += du[ threadIdx.x + s ];
          if( 2 * s < n  && threadIdx.x == n - 1 )
             du[ 0 ] += du[ threadIdx.x ];
          n = s;
       }
       if( n >= 4 )
       {
          s = n / 2;
          if( threadIdx.x < s )
             du[ threadIdx.x ] += du[ threadIdx.x + s ];
          if( 2 * s < n  && threadIdx.x == n - 1 )
             du[ 0 ] += du[ threadIdx.x ];
          n = s;
       }
       if( n >= 2 )
       {
          s = n / 2;
          if( threadIdx.x < s )
             du[ threadIdx.x ] += du[ threadIdx.x + s ];
          if( 2 * s < n  && threadIdx.x == n - 1 )
             du[ 0 ] += du[ threadIdx.x ];
          n = s;
       }
    }

   if( threadIdx.x == 0 )
      blockResidue[ blockIdx.x ] = du[ 0 ];

}
#endif

   } // namespace Benchmarks
} // namespace TNL