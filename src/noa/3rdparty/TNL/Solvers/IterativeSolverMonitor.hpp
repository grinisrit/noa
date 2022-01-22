// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <iomanip>
#include <limits>

// check if we are on a POSIX system or Windows,
// see https://stackoverflow.com/a/4575466
#if !defined(_WIN32) && !defined(_WIN64)
   #include <sys/ioctl.h>
   #include <unistd.h>
#endif

#include <noa/3rdparty/TNL/Solvers/IterativeSolver.h>

namespace noaTNL {
namespace Solvers {

template< typename Real, typename Index>
IterativeSolverMonitor< Real, Index > :: IterativeSolverMonitor()
: SolverMonitor(),
  stage( "" ),
  saved_stage( "" ),
  saved( false ),
  attributes_changed( false ),
  time( 0.0 ),
  saved_time( 0.0 ),
  timeStep( 0.0 ),
  saved_timeStep( 0.0 ),
  residue( 0.0 ),
  saved_residue( 0.0 ),
  elapsed_time_before_refresh( 0.0 ),
  iterations( 0 ),
  saved_iterations( 0 ),
  iterations_before_refresh( 0 ),
  verbose( 2 ),
  nodesPerIteration( 0 )
{
}

template< typename Real, typename Index>
void IterativeSolverMonitor< Real, Index > :: setStage( const std::string& stage )
{
   // save the items after a complete stage
   if( iterations > 0 ) {
      saved_stage = this->stage;
      saved_time = time;
      saved_timeStep = timeStep;
      saved_iterations = iterations;
      saved_residue = residue;
   }

   // reset the current items
   iterations = 0;
   residue = 0.0;

   this->stage = stage;
   saved = true;
   attributes_changed = true;
}

template< typename Real, typename Index>
void IterativeSolverMonitor< Real, Index > :: setTime( const RealType& time )
{
   this->time = time;
   attributes_changed = true;
}

template< typename Real, typename Index>
void IterativeSolverMonitor< Real, Index > :: setTimeStep( const RealType& timeStep )
{
   this->timeStep = timeStep;
   attributes_changed = true;
}

template< typename Real, typename Index>
void IterativeSolverMonitor< Real, Index > :: setIterations( const Index& iterations )
{
   this->iterations = iterations;
   attributes_changed = true;
}

template< typename Real, typename Index>
void IterativeSolverMonitor< Real, Index > :: setResidue( const Real& residue )
{
   this->residue = residue;
   attributes_changed = true;
}

template< typename Real, typename Index>
void IterativeSolverMonitor< Real, Index > :: setVerbose( const Index& verbose )
{
   this->verbose = verbose;
   attributes_changed = true;
}

template< typename Real, typename Index>
void IterativeSolverMonitor< Real, Index > :: setNodesPerIteration( const IndexType& nodes )
{
   this->nodesPerIteration = nodes;
   attributes_changed = true;
}

template< typename Real, typename Index>
void IterativeSolverMonitor< Real, Index > :: refresh()
{
   // NOTE: We can't check if stdout is attached to a terminal or not, because
   // isatty(STDOUT_FILENO) *always* reports 1 under mpirun, regardless if it
   // runs from terminal or under SLURM or PBS. Hence, we use the verbose flag
   // to determine interactivity: verbose=2 expects stdout to be attached to a
   // terminal, verbose=1 does not use terminal features (can be used with
   // stdout redirected to a file, or with a terminal) and verbose=0 disables
   // the output completely.

   if( this->verbose > 0 )
   {
      // Check if we should display the current values or the values saved after
      // the previous stage. If the iterations cycle much faster than the solver
      // monitor refreshes, we display only the values saved after the whole
      // cycle to hide the irrelevant partial progress.
      const bool saved = this->saved;
      this->saved = false;

      const int line_width = getLineWidth();
      int free = line_width ? line_width : std::numeric_limits<int>::max();

      auto real_to_string = []( Real value, int precision = 6 ) {
         std::stringstream stream;
         stream << std::setprecision( precision ) << value;
         return stream.str();
      };

      auto print_item = [&free]( const std::string& item, int width = 0 ) {
         width = min( free, (width) ? width : item.length() );
         std::cout << std::setw( width ) << item.substr( 0, width );
         free -= width;
      };

      if( this->verbose >= 2 )
         // \33[2K erases the current line, see https://stackoverflow.com/a/35190285
         std::cout << "\33[2K\r";
      else if( ! attributes_changed )
         // verbose == 1, attributes were not updated since the last refresh
         return;

      if( timer != nullptr ) {
         print_item( " ELA:" );
         print_item( real_to_string( getElapsedTime(), 5 ), 8 );
      }
      if( time > 0 ) {
         print_item( " T:" );
         print_item( real_to_string( (saved) ? saved_time : time, 5 ), 8 );
         if( (saved) ? saved_timeStep : timeStep > 0 ) {
            print_item( " TAU:" );
            print_item( real_to_string( (saved) ? saved_timeStep : timeStep, 5 ), 10 );
         }
      }

      const std::string displayed_stage = (saved) ? saved_stage : stage;
      if( displayed_stage.length() && free > 5 ) {
         if( (int) displayed_stage.length() <= free - 2 ) {
            std::cout << "  " << displayed_stage;
            free -= ( 2 + displayed_stage.length() );
         }
         else {
            std::cout << "  " << displayed_stage.substr( 0, free - 5 ) << "...";
            free = 0;
         }
      }

      if( (saved) ? saved_iterations : iterations > 0 && free >= 14 ) {
         print_item( " ITER:" );
         print_item( std::to_string( (saved) ? saved_iterations : iterations ), 8 );
      }
      if( (saved) ? saved_residue : residue && free >= 17 ) {
         print_item( " RES:" );
         print_item( real_to_string( (saved) ? saved_residue : residue, 5 ), 12 );
      }

      if( nodesPerIteration ) // otherwise MLUPS: 0 is printed
      {
         const RealType mlups = nodesPerIteration * (iterations - iterations_before_refresh) / (getElapsedTime() - elapsed_time_before_refresh) * 1e-6;
         print_item( " MLUPS:", 0 );
         if( mlups > 0 )
         {
            print_item( real_to_string( mlups, 5 ), 7 );
            last_mlups = mlups;
         }
         else print_item( real_to_string( last_mlups, 5 ), 7 );
      }
      iterations_before_refresh = iterations;
      elapsed_time_before_refresh = getElapsedTime();

      if( this->verbose >= 2 )
         // return to the beginning of the line
         std::cout << "\r" << std::flush;
      else
         // linebreak and flush
         std::cout << "\n" << std::flush;

      // reset the changed flag
      attributes_changed = false;
   }
}

template< typename Real, typename Index>
int IterativeSolverMonitor< Real, Index > :: getLineWidth()
{
#if !defined(_WIN32) && !defined(_WIN64)
   struct winsize w;
   ioctl(STDOUT_FILENO, TIOCGWINSZ, &w);
   return w.ws_col;
#else
   return 0;
#endif
}

} // namespace Solvers
} // namespace noaTNL
