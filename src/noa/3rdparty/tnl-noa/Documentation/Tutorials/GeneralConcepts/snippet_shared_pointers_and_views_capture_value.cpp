template< typename Device >
void lambda_capture_by_value( int size )
{
    TNL::Containers::Array< int, Device > a( size );
    auto f = [=] __cuda_callable__ ( int i ) mutable {
        a[ i ] = 1;
    };
    TNL::Algorithms::ParallelFor< Device >::exec( 0, size, f );
}


