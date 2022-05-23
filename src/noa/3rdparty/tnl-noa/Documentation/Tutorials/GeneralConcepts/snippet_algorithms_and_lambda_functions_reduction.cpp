template< typename Device >
void scalarProduct( double* v1, double* v2, double* product, const int size )
{
    auto fetch = [=] __cuda_callable__ ( int i ) -> double {
        return = v1[ i ] * v2[ i ];
    }
    auto reduce = [] __cuda_callable__ ( const double& a, const double& b ) {
        return a + b; };
    TNL::Algorithms::reduce< Device >( 0, size, fetch, reduce, 0.0 );
}