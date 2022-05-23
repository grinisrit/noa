double sequentialSum( const double* a, const int size )
{
   double sum( 0.0 );
   for( int i = 0; i < size; i++ )
      sum += a[ i ];
   return sum;
}
