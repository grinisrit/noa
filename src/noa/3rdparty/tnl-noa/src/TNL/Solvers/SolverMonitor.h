// Copyright (c) 2004-2023 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <thread>
#include <atomic>

#include <noa/3rdparty/tnl-noa/src/TNL/Timer.h>

namespace noa::TNL {
namespace Solvers {

/**
 * \brief Base class for solver monitors.
 *
 * The solver monitors serve for monitoring a convergence and status of various solvers.
 * The solver monitor uses separate thread for monitoring the solver status in preset time period.
 */
class SolverMonitor
{
public:
   /**
    * \brief Basic construct with no arguments
    */
   SolverMonitor() = default;

   /**
    * \brief This abstract method is responsible for printing or visualizing the status of the solver.
    */
   virtual void
   refresh() = 0;

   /**
    * \brief Set the time interval between two consecutive calls of \ref SolverMonitor::refresh.
    *
    * \param refreshRate refresh rate in miliseconds.
    */
   void
   setRefreshRate( const int& refreshRate )
   {
      timeout_milliseconds = refreshRate;
   }

   /**
    * \brief Set a timer object for the solver monitor.
    *
    * If a timer is set, the monitor can measure real elapsed time since the start of the solver.
    *
    * \param timer is an instance of \ref TNL::Timer.
    */
   void
   setTimer( Timer& timer )
   {
      this->timer = &timer;
   }

   /**
    * \brief Starts the main loop from which the method \ref SolverMonitor::refresh is called in given time periods.
    */
   void
   runMainLoop()
   {
      // We need to use both 'started' and 'stopped' to avoid a deadlock
      // when the loop thread runs this method delayed after the
      // SolverMonitorThread's destructor has already called stopMainLoop()
      // from the main thread.
      started = true;

      const int timeout_base = 100;
      const std::chrono::milliseconds timeout( timeout_base );

      while( ! stopped ) {
         refresh();

         // make sure to detect changes to refresh rate
         int steps = timeout_milliseconds / timeout_base;
         if( steps <= 0 )
            steps = 1;

         int i = 0;
         while( ! stopped && i++ < steps ) {
            std::this_thread::sleep_for( timeout );
         }
      }

      // reset to initial state
      started = false;
      stopped = false;
   }

   /**
    * \brief Stops the main loop of the monitor. See \ref runMainLoop.
    */
   void
   stopMainLoop()
   {
      stopped = true;
   }

   /**
    * \brief Checks whether the main loop was stopped.
    *
    * \return true if the main loop was stopped.
    * \return false if the main loop was not stopped yet.
    */
   bool
   isStopped() const
   {
      return stopped;
   }

protected:
   double
   getElapsedTime()
   {
      if( timer == nullptr )
         return 0.0;
      return timer->getRealTime();
   }

   std::atomic_int timeout_milliseconds{ 500 };

   std::atomic_bool started{ false };
   std::atomic_bool stopped{ false };

   Timer* timer = nullptr;
};

/**
 * \brief A RAII wrapper for launching the SolverMonitor's main loop in a separate thread.
 */
class SolverMonitorThread
{
public:
   /**
    * \brief Constructor with instance of solver monitor.
    *
    * \param solverMonitor is a reference to an instance of a solver monitor.
    */
   SolverMonitorThread( SolverMonitor& solverMonitor )
   : solverMonitor( solverMonitor ), t( &SolverMonitor::runMainLoop, &solverMonitor )
   {}

   /**
    * \brief Destructor.
    */
   ~SolverMonitorThread()
   {
      solverMonitor.stopMainLoop();
      if( t.joinable() )
         t.join();
   }

private:
   SolverMonitor& solverMonitor;

   std::thread t;
};

}  // namespace Solvers
}  // namespace noa::TNL
