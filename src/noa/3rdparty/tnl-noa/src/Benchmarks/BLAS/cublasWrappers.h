#pragma once

#ifdef HAVE_CUDA

#include <cublas_v2.h>

inline cublasStatus_t
cublasIgamax( cublasHandle_t handle, int n,
              const float           *x, int incx, int *result )
{
   return cublasIsamax( handle, n, x, incx, result );
}

inline cublasStatus_t
cublasIgamax( cublasHandle_t handle, int n,
              const double          *x, int incx, int *result )
{
   return cublasIdamax( handle, n, x, incx, result );
}


inline cublasStatus_t
cublasIgamin( cublasHandle_t handle, int n,
              const float           *x, int incx, int *result )
{
   return cublasIsamin( handle, n, x, incx, result );
}

inline cublasStatus_t
cublasIgamin( cublasHandle_t handle, int n,
              const double          *x, int incx, int *result )
{
   return cublasIdamin( handle, n, x, incx, result );
}


inline cublasStatus_t
cublasGasum( cublasHandle_t handle, int n,
             const float           *x, int incx, float  *result )
{
   return cublasSasum( handle, n, x, incx, result );
}

inline cublasStatus_t
cublasGasum( cublasHandle_t handle, int n,
             const double          *x, int incx, double *result )
{
   return cublasDasum( handle, n, x, incx, result );
}


inline cublasStatus_t
cublasGaxpy( cublasHandle_t handle, int n,
             const float           *alpha,
             const float           *x, int incx,
             float                 *y, int incy )
{
   return cublasSaxpy( handle, n, alpha, x, incx, y, incy );
}

inline cublasStatus_t
cublasGaxpy( cublasHandle_t handle, int n,
             const double          *alpha,
             const double          *x, int incx,
             double                *y, int incy )
{
   return cublasDaxpy( handle, n, alpha, x, incx, y, incy );
}


inline cublasStatus_t
cublasGdot( cublasHandle_t handle, int n,
            const float        *x, int incx,
            const float        *y, int incy,
            float         *result )
{
   return cublasSdot( handle, n, x, incx, y, incy, result );
}

inline cublasStatus_t
cublasGdot( cublasHandle_t handle, int n,
            const double       *x, int incx,
            const double       *y, int incy,
            double        *result )
{
   return cublasDdot( handle, n, x, incx, y, incy, result );
}


inline cublasStatus_t
cublasGnrm2( cublasHandle_t handle, int n,
             const float           *x, int incx, float  *result )
{
   return cublasSnrm2( handle, n, x, incx, result );
}

inline cublasStatus_t
cublasGnrm2( cublasHandle_t handle, int n,
             const double          *x, int incx, double *result )
{
   return cublasDnrm2( handle, n, x, incx, result );
}


inline cublasStatus_t
cublasGscal( cublasHandle_t handle, int n,
             const float           *alpha,
             float           *x, int incx )
{
   return cublasSscal( handle, n, alpha, x, incx );
}

inline cublasStatus_t
cublasGscal( cublasHandle_t handle, int n,
             const double          *alpha,
             double          *x, int incx )
{
   return cublasDscal( handle, n, alpha, x, incx );
}


inline cublasStatus_t
cublasGemv( cublasHandle_t handle, cublasOperation_t trans,
            int m, int n,
            const float           *alpha,
            const float           *A, int lda,
            const float           *x, int incx,
            const float           *beta,
            float           *y, int incy )
{
   return cublasSgemv( handle, trans, m, n, alpha, A, lda, x, incx, beta, y, incy );
}

inline cublasStatus_t
cublasGemv( cublasHandle_t handle, cublasOperation_t trans,
            int m, int n,
            const double          *alpha,
            const double          *A, int lda,
            const double          *x, int incx,
            const double          *beta,
            double          *y, int incy )
{
   return cublasDgemv( handle, trans, m, n, alpha, A, lda, x, incx, beta, y, incy );
}

#endif
