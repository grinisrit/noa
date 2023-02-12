#pragma once

#include <ginkgo/ginkgo.hpp>

template< typename Value >
Value
get_scalar_value( const gko::LinOp* op )
{
   auto mtx = gko::as< gko::matrix::Dense< Value > >( op );
   auto exec = mtx->get_executor();
   return exec->copy_val_to_host( mtx->get_const_values() );
}
