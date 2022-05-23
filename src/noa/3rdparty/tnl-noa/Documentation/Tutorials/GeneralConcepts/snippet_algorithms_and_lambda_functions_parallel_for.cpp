template< typename Device >
void vectorAddition( double* v1, double* v2, double* sum, const int size )
{
    auto sum_lambda = [=] __cuda_callable__ ( int i ) mutable {
        sum[ i ] = v1[ i ] + v2[ i ];
    }
    TNL::Algorithms::ParalellFor< Device >::exec( 0, size, sum_lambda );
}