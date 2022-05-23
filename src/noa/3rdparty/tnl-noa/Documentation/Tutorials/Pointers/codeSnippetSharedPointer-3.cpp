__global__ void cudaKernel( Array a )
{
   if( thredadIdx.x. < a.size )
      a.data[ threadIdx.x ] = 0;
}