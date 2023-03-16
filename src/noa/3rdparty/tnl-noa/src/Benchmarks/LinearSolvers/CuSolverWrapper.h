#pragma once

#ifdef __CUDACC__

#include <cusolverSp.h>

#include <TNL/Config/ConfigDescription.h>
#include <TNL/Matrices/CSR.h>
#include <TNL/Solvers/Linear/LinearSolver.h>


template< typename Matrix >
struct is_csr_matrix
{
    static constexpr bool value = false;
};

template< typename Real, typename Device, typename Index >
struct is_csr_matrix< TNL::Matrices::CSR< Real, Device, Index > >
{
    static constexpr bool value = true;
};


namespace TNL {
namespace Solvers {
namespace Linear {

template< typename Matrix >
class CuSolverWrapper
    : public LinearSolver< Matrix >
{
    static_assert( ::is_csr_matrix< Matrix >::value, "The CuSolverWrapper solver is available only for CSR matrices." );
    static_assert( std::is_same< typename Matrix::DeviceType, Devices::Cuda >::value, "CuSolverWrapper is available only on CUDA" );
    static_assert( std::is_same< typename Matrix::RealType, float >::value ||
                   std::is_same< typename Matrix::RealType, double >::value, "unsupported RealType" );
    static_assert( std::is_same< typename Matrix::IndexType, int >::value, "unsupported IndexType" );

    using Base = LinearSolver< Matrices::CSR< double, Devices::Cuda, int > >;
public:
    using RealType = typename Base::RealType;
    using DeviceType = typename Base::DeviceType;
    using IndexType = typename Base::IndexType;
    using VectorViewType = typename Base::VectorViewType;
    using ConstVectorViewType = typename Base::ConstVectorViewType;

    bool solve( ConstVectorViewType b, VectorViewType x ) override
    {
        TNL_ASSERT_EQ( this->matrix->getRows(), this->matrix->getColumns(), "matrix must be square" );
        TNL_ASSERT_EQ( this->matrix->getColumns(), x.getSize(), "wrong size of the solution vector" );
        TNL_ASSERT_EQ( this->matrix->getColumns(), b.getSize(), "wrong size of the right hand side" );

        const IndexType size = this->matrix->getRows();

        this->resetIterations();
        this->setResidue( this->getConvergenceResidue() + 1.0 );

        cusolverSpHandle_t handle;
        cusolverStatus_t status;
        cusparseStatus_t cusparse_status;
        cusparseMatDescr_t mat_descr;

        status = cusolverSpCreate( &handle );
        if( status != CUSOLVER_STATUS_SUCCESS ) {
            std::cerr << "cusolverSpCreate failed: " << status << std::endl;
            return false;
        }

        cusparse_status = cusparseCreateMatDescr( &mat_descr );
        if( cusparse_status != CUSPARSE_STATUS_SUCCESS ) {
            std::cerr << "cusparseCreateMatDescr failed: " << cusparse_status << std::endl;
            return false;
        }

        cusparseSetMatType( mat_descr, CUSPARSE_MATRIX_TYPE_GENERAL );
        cusparseSetMatIndexBase( mat_descr, CUSPARSE_INDEX_BASE_ZERO );

        const RealType tol = 1e-16;
        const int reorder = 0;
        int singularity = 0;

        if( std::is_same< typename Matrix::RealType, float >::value )
        {
            status = cusolverSpScsrlsvqr( handle,
                                          size,
                                          this->matrix->getValues().getSize(),
                                          mat_descr,
                                          (const float*) this->matrix->getValues().getData(),
                                          this->matrix->getRowPointers().getData(),
                                          this->matrix->getColumnIndexes().getData(),
                                          (const float*) b.getData(),
                                          tol,
                                          reorder,
                                          (float*) x.getData(),
                                          &singularity );

            if( status != CUSOLVER_STATUS_SUCCESS ) {
                std::cerr << "cusolverSpScsrlsvqr failed: " << status << std::endl;
                return false;
            }
        }

        if( std::is_same< typename Matrix::RealType, double >::value )
        {
            status = cusolverSpDcsrlsvqr( handle,
                                          size,
                                          this->matrix->getValues().getSize(),
                                          mat_descr,
                                          (const double*) this->matrix->getValues().getData(),
                                          this->matrix->getRowPointers().getData(),
                                          this->matrix->getColumnIndexes().getData(),
                                          (const double*) b.getData(),
                                          tol,
                                          reorder,
                                          (double*) x.getData(),
                                          &singularity );

            if( status != CUSOLVER_STATUS_SUCCESS ) {
                std::cerr << "cusolverSpDcsrlsvqr failed: " << status << std::endl;
                return false;
            }
        }

        cusparseDestroyMatDescr( mat_descr );
        cusolverSpDestroy( handle );

        return true;
    }
};

} // namespace Linear
} // namespace Solvers
} // namespace TNL

#endif
