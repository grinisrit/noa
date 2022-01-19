// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

// Implemented by: Jakub Klinkovsky

#pragma once

#ifdef HAVE_UMFPACK

#include "UmfpackWrapper.h"

#include <TNL/Solvers/Linear/Utils/LinearResidueGetter.h>

namespace TNL {
namespace Solvers {
namespace Linear {

bool UmfpackWrapper< Matrices::CSR< double, Devices::Host, int > >::
solve( ConstVectorViewType b, VectorViewType x )
{
    TNL_ASSERT_EQ( this->matrix->getRows(), this->matrix->getColumns(), "matrix must be square" );
    TNL_ASSERT_EQ( this->matrix->getColumns(), x.getSize(), "wrong size of the solution vector" );
    TNL_ASSERT_EQ( this->matrix->getColumns(), b.getSize(), "wrong size of the right hand side" );

    const IndexType size = this->matrix->getRows();

    this->resetIterations();
    this->setResidue( this -> getConvergenceResidue() + 1.0 );

    RealType bNorm = b. lpNorm( ( RealType ) 2.0 );

    // UMFPACK objects
    void* Symbolic = nullptr;
    void* Numeric = nullptr;

    int status = UMFPACK_OK;
    double Control[ UMFPACK_CONTROL ];
    double Info[ UMFPACK_INFO ];

    // umfpack expects Compressed Sparse Column format, we have Compressed Sparse Row
    // so we need to solve  A^T * x = rhs
    int system_type = UMFPACK_Aat;

    // symbolic reordering of the sparse matrix
    status = umfpack_di_symbolic( size, size,
                                  this->matrix->getRowPointers().getData(),
                                  this->matrix->getColumnIndexes().getData(),
                                  this->matrix->getValues().getData(),
                                  &Symbolic, Control, Info );
    if( status != UMFPACK_OK ) {
        std::cerr << "error: symbolic reordering failed" << std::endl;
        goto finished;
    }

    // numeric factorization
    status = umfpack_di_numeric( this->matrix->getRowPointers().getData(),
                                 this->matrix->getColumnIndexes().getData(),
                                 this->matrix->getValues().getData(),
                                 Symbolic, &Numeric, Control, Info );
    if( status != UMFPACK_OK ) {
        std::cerr << "error: numeric factorization failed" << std::endl;
        goto finished;
    }

    // solve with specified right-hand-side
    status = umfpack_di_solve( system_type,
                               this->matrix->getRowPointers().getData(),
                               this->matrix->getColumnIndexes().getData(),
                               this->matrix->getValues().getData(),
                               x.getData(),
                               b.getData(),
                               Numeric, Control, Info );
    if( status != UMFPACK_OK ) {
        std::cerr << "error: umfpack_di_solve failed" << std::endl;
        goto finished;
    }

finished:
    if( status != UMFPACK_OK ) {
        // increase print level for reports
        Control[ UMFPACK_PRL ] = 2;
        umfpack_di_report_status( Control, status );
//        umfpack_di_report_control( Control );
//        umfpack_di_report_info( Control, Info );
    }

    if( Symbolic )
        umfpack_di_free_symbolic( &Symbolic );
    if( Numeric )
        umfpack_di_free_numeric( &Numeric );

    this->setResidue( LinearResidueGetter::getResidue( *this->matrix, x, b, bNorm ) );
    this->refreshSolverMonitor( true );
    return status == UMFPACK_OK;
}

} // namespace Linear
} // namespace Solvers
} // namespace TNL

#endif
