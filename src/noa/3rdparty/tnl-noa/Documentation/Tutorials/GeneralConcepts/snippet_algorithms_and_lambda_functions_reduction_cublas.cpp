void scalarProduct( double* u1, double* u2,
                    double* v1, double* v2,
                    double* product, const int size )
{
    cublasHandle_t handle;
    cublasSaxpy( handle, size, 1.0, u1, 1, u2, 1 );
    cublasSaxpy( handle, size, 1.0, v1, 1, v2, 1 );
    cublasSdot ( handle, size, 1.0, u1, 1, v1, 1, &product );
}