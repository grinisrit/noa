template< typename Device >
void scalarProduct( double* u1, double* u2,
                    double* v1, double* v2,
                    double* product, const int size )
{
    auto fetch = [=] __cuda_callable__ ( int i ) -> double {
        return = ( u1[ i ] + u2[ i ] ) * ( v1[ i ] + v2[ i ] );
    }
    auto reduce = [] __cuda_callable__ ( const double& a, const double& b ) {
        return a + b; };
    TNL::Algorithms::reduce< Device >( 0, size, fetch, reduce, 0.0 );
}