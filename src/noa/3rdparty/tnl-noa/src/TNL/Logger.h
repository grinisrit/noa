// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <ostream>

#include <noa/3rdparty/tnl-noa/src/TNL/String.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Config/ParameterContainer.h>

namespace noa::TNL {

/// Creates calculations log in the form of a table.
class Logger
{
public:
   /////
   /// \brief Basic constructor.
   ///
   /// \param _width Integer that defines the width of the log.
   /// \param _stream Defines output stream where the log will be printed out.
   Logger( int width, std::ostream& stream ) : width( width ), stream( stream ) {}

   /////
   /// \brief Creates header in given log.
   ///
   /// The header usually contains title of the program.
   ///
   /// \param title A String containing the header title.
   void
   writeHeader( const String& title );

   /// \brief Creates separator used as a log structure.
   void
   writeSeparator();

   /// \brief Inserts information about various system parameters into the log.
   ///
   /// \param printGPUInfo When \e true, prints information about available GPUs.
   bool
   writeSystemInformation( bool printGPUInfo = false );

   /////
   /// \brief Inserts a line with current time into the log.
   ///
   /// \param label Label to be printed to the log together with the current time.
   void
   writeCurrentTime( const char* label );

   /// \brief Inserts parameter information into the log.
   ///
   /// \tparam ParameterType Type of the parameter.
   /// \param label Description/label of the line.
   /// \param parameterName Name of the parameter.
   /// \param parameters A container with configuration parameters.
   /// \param parameterLevel Integer defining the indent used in the log.

   // TODO: add units
   template< typename ParameterType >
   void
   writeParameter( const String& label,
                   const String& parameterName,
                   const Config::ParameterContainer& parameters,
                   int parameterLevel = 0 );

   /// \brief Inserts parameter information into the log.
   ///
   /// \tparam ParameterType Type of the parameter.
   /// \param label Description/label of the line.
   /// \param value Parameter value.
   /// \param parameterLevel Integer defining the indent used in the log.
   template< typename ParameterType >
   void
   writeParameter( const String& label, const ParameterType& value, int parameterLevel = 0 );

protected:
   /// \brief Integer defining the width of the log.
   int width;

   /// \brief Output stream where the log will be printed out.
   std::ostream& stream;
};

}  // namespace noa::TNL

#include <noa/3rdparty/tnl-noa/src/TNL/Logger_impl.h>
