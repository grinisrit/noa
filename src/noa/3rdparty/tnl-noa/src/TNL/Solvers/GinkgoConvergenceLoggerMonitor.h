// Copyright (c) 2004-2023 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <noa/3rdparty/tnl-noa/src/TNL/Solvers/IterativeSolverMonitor.h>

#include <ginkgo/ginkgo.hpp>

namespace noa::TNL {
namespace Solvers {

/**
 * \brief A Ginkgo \e Convergence logger with a TNL iterative solver monitor.
 *
 * \ingroup Ginkgo
 */
template< typename ValueType = gko::default_precision, typename IndexType = int >
class GinkgoConvergenceLoggerMonitor : public gko::log::Convergence< ValueType >
{
private:
   IterativeSolverMonitor< ValueType, IndexType >* solver_monitor = nullptr;

public:
   GinkgoConvergenceLoggerMonitor( std::shared_ptr< const gko::Executor > exec,
                                   const gko::log::Logger::mask_type& enabled_events = gko::log::Logger::all_events_mask,
                                   IterativeSolverMonitor< ValueType, IndexType >* solver_monitor = nullptr )
   : gko::log::Convergence< ValueType >( exec, enabled_events ), solver_monitor( solver_monitor )
   {}

   GinkgoConvergenceLoggerMonitor( const gko::log::Logger::mask_type& enabled_events = gko::log::Logger::all_events_mask,
                                   IterativeSolverMonitor< ValueType, IndexType >* solver_monitor = nullptr )
   : gko::log::Convergence< ValueType >( enabled_events ), solver_monitor( solver_monitor )
   {}

   static std::unique_ptr< GinkgoConvergenceLoggerMonitor >
   create( std::shared_ptr< const gko::Executor > exec,
           const gko::log::Logger::mask_type& enabled_events = gko::log::Logger::all_events_mask,
           IterativeSolverMonitor< ValueType, IndexType >* solver_monitor = nullptr )
   {
      return std::unique_ptr< GinkgoConvergenceLoggerMonitor >(
         new GinkgoConvergenceLoggerMonitor( exec, enabled_events, solver_monitor ) );
   }

   static std::unique_ptr< GinkgoConvergenceLoggerMonitor >
   create( const gko::log::Logger::mask_type& enabled_events = gko::log::Logger::all_events_mask,
           IterativeSolverMonitor< ValueType, IndexType >* solver_monitor = nullptr )
   {
      return std::unique_ptr< GinkgoConvergenceLoggerMonitor >(
         new GinkgoConvergenceLoggerMonitor( enabled_events, solver_monitor ) );
   }

   void
   set_solver_monitor( IterativeSolverMonitor< ValueType, IndexType >* solver_monitor )
   {
      this->solver_monitor = solver_monitor;
   }

   void
   on_criterion_check_completed( const gko::stop::Criterion* criterion,
                                 const gko::size_type& num_iterations,
                                 const gko::LinOp* r,
                                 const gko::LinOp* tau,
                                 const gko::LinOp* implicit_tau_sq,
                                 const gko::LinOp* x,
                                 const gko::uint8& stopping_id,
                                 const bool& set_finalized,
                                 const gko::array< gko::stopping_status >* status,
                                 const bool& one_changed,
                                 const bool& all_converged ) const override
   {
      // call the parent implementation
      gko::log::Convergence< ValueType >::on_criterion_check_completed( criterion,
                                                                        num_iterations,
                                                                        r,
                                                                        tau,
                                                                        implicit_tau_sq,
                                                                        x,
                                                                        stopping_id,
                                                                        set_finalized,
                                                                        status,
                                                                        one_changed,
                                                                        all_converged );

      if( ! solver_monitor )
         return;

      // Set current iteration number
      solver_monitor->setIterations( num_iterations );

      // If the solver shares a residual norm, log its value
      if( tau ) {
         auto norm = gko::as< gko::matrix::Dense< ValueType > >( tau );
         solver_monitor->setResidue( get_first_element( gko::lend( norm ) ) );
      }
      // Otherwise, if the solver shares the implicit squared residual norm, log its value
      else if( implicit_tau_sq ) {
         auto sq_norm = gko::as< gko::matrix::Dense< ValueType > >( implicit_tau_sq );
         solver_monitor->setResidue( std::sqrt( std::abs( get_first_element( gko::lend( sq_norm ) ) ) ) );
      }
      // Otherwise, compute the norm of the recurrent residual vector
      else {
         auto dense_residual = gko::as< gko::matrix::Dense< ValueType > >( r );
         solver_monitor->setResidue( compute_norm( gko::lend( dense_residual ) ) );
      }
   }

   void
   on_criterion_check_completed( const gko::stop::Criterion* criterion,
                                 const gko::size_type& num_iterations,
                                 const gko::LinOp* r,
                                 const gko::LinOp* tau,
                                 const gko::LinOp* x,
                                 const gko::uint8& stopping_id,
                                 const bool& set_finalized,
                                 const gko::array< gko::stopping_status >* status,
                                 const bool& one_changed,
                                 const bool& all_converged ) const override
   {
      // call the parent implementation
      gko::log::Convergence< ValueType >::on_criterion_check_completed(
         criterion, num_iterations, r, tau, x, stopping_id, set_finalized, status, one_changed, all_converged );

      if( ! solver_monitor )
         return;

      // Set current iteration number
      solver_monitor->setIterations( num_iterations );

      // If the solver shares a residual norm, log its value
      if( tau ) {
         auto norm = gko::as< gko::matrix::Dense< ValueType > >( tau );
         solver_monitor->setResidue( get_first_element( gko::lend( norm ) ) );
      }
      // Otherwise, compute the norm of the recurrent residual vector
      else {
         auto dense_residual = gko::as< gko::matrix::Dense< ValueType > >( r );
         solver_monitor->setResidue( compute_norm( gko::lend( dense_residual ) ) );
      }
   }

private:
   /**
    * \brief Utility function which returns the first element (position [0, 0])
    * from a given \e gko::matrix::Dense matrix / vector.
    */
   template< typename Value >
   static Value
   get_first_element( const gko::matrix::Dense< Value >* mtx )
   {
      // Copy the matrix / vector to the host device before accessing the value in
      // case it is stored in a GPU.
      return mtx->get_executor()->copy_val_to_host( mtx->get_const_values() );
   }

   /**
    * \brief Utility function which computes the norm of a Ginkgo
    * \e gko::matrix::Dense vector.
    */
   template< typename Value >
   static Value
   compute_norm( const gko::matrix::Dense< Value >* b )
   {
      // Initialize storage for the result
      auto b_norm = gko::initialize< gko::matrix::Dense< ValueType > >( { 0 }, b->get_executor() );
      // Compute the norm
      b->compute_norm2( gko::lend( b_norm ) );
      // Use the other utility function to return the value
      return get_first_element( gko::lend( b_norm ) );
   }
};

}  // namespace Solvers
}  // namespace noa::TNL
