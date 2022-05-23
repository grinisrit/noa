__global__ tupleKernel( ArrayTuple tuple )
{
   if( threadIdx.x < tuple.a1->size )
      tuple.a1->data[ threadIdx.x ] = 0;
   if( threadIdx.x < tuple.a2->size )
      tuple.a2->data[ threadIdx.x ] = 0;
}
