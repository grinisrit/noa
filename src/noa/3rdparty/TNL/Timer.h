// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <chrono>

namespace noa::TNL {

class Logger;

/**
 * \brief Class for real time, CPU time and CPU cycles measuring.
 *
 * It measures the elapsed real time, CPU time (in seconds) and CPU cycles
 * elapsed on the timer. The timer can be paused by calling \ref stop and \ref
 * start methods and reseted by calling \ref reset.
 *
 * \par Example
 * \include TimerExample.cpp
 * \par Output
 * \include TimerExample.out
 */
class Timer
{
   public:

      /**
       * \brief Basic constructor.
       *
       * This function creates a new timer and resets it.
       */
      Timer();

      /**
       * \brief Resets timer.
       *
       * Resets all time and cycle measurements such as real time, CPU time and 
       * CPU cycles. Sets all of them to zero.
       */
      void reset();

      /**
       * \brief Stops (pauses) the timer.
       *
       * Pauses all time and cycle measurements such as real time, CPU time and
       * CPU cycles, but does not set them to zero.
       */
      void stop();

      /**
       * \brief Starts timer.
       *
       * Starts all time and cycle measurements such as real time, CPU time and
       * CPU cycles. This method can be used also after using the \ref stop
       * method. The timer then continues measuring the time without reseting.
       */
      void start();

      /**
       * \brief Returns the elapsed real time on this timer.
       *
       * This method returns the real time elapsed so far (in seconds).
       * This method can be called while the timer is running, there is no
       * need to use \ref stop method first.
       */
      double getRealTime() const;

      /**
       * \brief Returns the elapsed CPU time on this timer.
       *
       * This method returns the CPU time (i.e. time the CPU spent by processing
       * this process) elapsed so far (in seconds). This method can be called
       * while the timer is running, there is no need to use \ref stop method
       * first.
       */
      double getCPUTime() const;

      /**
       * \brief Returns the number of CPU cycles (machine cycles) elapsed on this timer.
       *
       * CPU cycles are counted by adding the number of CPU cycles between
       * \ref start and \ref stop methods together.
       */
      unsigned long long int getCPUCycles() const;

      /**
       * \brief Writes a record into the \e logger.
       *
       * \param logger Name of Logger object.
       * \param logLevel A non-negative integer recording the log record indent.
       * 
       * \par Example
       * \include TimerExampleLogger.cpp
       * \par Output
       * \include TimerExampleLogger.out
       */
      bool writeLog( Logger& logger, int logLevel = 0 ) const;

   protected:

      using TimePoint = typename std::chrono::high_resolution_clock::time_point;
      using Duration = typename std::chrono::high_resolution_clock::duration;

      /**
       * \brief Function for measuring the real time.
       */
      TimePoint readRealTime() const;

      /**
       * \brief Function for measuring the CPU time.
       */
      double readCPUTime() const;

      /**
       * \brief Function for counting the number of CPU cycles (machine cycles).
       */
      unsigned long long int readCPUCycles() const;

      /**
       * \brief Converts the real time into seconds as a floating point number.
       */
      double durationToDouble( const Duration& duration ) const;

      /**
       * \brief Time Stamp Counter returning number of CPU cycles since reset.
       *
       * Only for x86 compatible CPUs.
       */
      inline unsigned long long rdtsc() const;

      TimePoint initialRealTime;

      Duration totalRealTime;

      double initialCPUTime, totalCPUTime;

      unsigned long long int initialCPUCycles, totalCPUCycles;

      bool stopState;
};

} // namespace noa::TNL

#include <noa/3rdparty/TNL/Timer.hpp>
