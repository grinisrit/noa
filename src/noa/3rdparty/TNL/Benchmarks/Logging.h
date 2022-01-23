// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

// Implemented by: Jakub Klinkovsky,
//                 Tomas Oberhuber

#pragma once

#include <list>
#include <map>
#include <vector>
#include <iostream>
#include <iomanip>
#include <string>
#include <fstream>

namespace noa::TNL {
namespace Benchmarks {

class LoggingRowElements
{
   public:

      LoggingRowElements()
      {
         stream << std::setprecision( 6 ) << std::fixed;
      }

      template< typename T >
      LoggingRowElements& operator << ( const T& b )
      {
         stream << b;
         elements.push_back( stream.str() );
         stream.str( std::string() );
         return *this;
      }

      LoggingRowElements& operator << ( decltype( std::setprecision( 2 ) )& setprec )
      {
         stream << setprec;
         return *this;
      }

      LoggingRowElements& operator << ( decltype( std::fixed )& setfixed ) // the same works also for std::scientific
      {
         stream << setfixed;
         return *this;
      }

      std::size_t size() const noexcept { return elements.size(); };

      // iterators
      auto begin() noexcept { return elements.begin(); }

      auto begin() const noexcept { return elements.begin(); }

      auto cbegin() const noexcept { return elements.cbegin(); }

      auto end() noexcept { return elements.end(); }

      auto end() const noexcept { return elements.end(); }

      auto cend() const noexcept { return elements.cend(); }

   protected:
      std::list< std::string > elements;

      std::stringstream stream;
};

class Logging
{
public:
   using MetadataElement = std::pair< std::string, std::string >;
   using MetadataColumns = std::vector< MetadataElement >;

   using HeaderElements = std::vector< std::string >;
   using RowElements = LoggingRowElements;
   using WidthHints = std::vector< int >;

   Logging( std::ostream& log, int verbose = true )
   : log(log), verbose(verbose)
   {
      try {
         // check if we got an open file
         std::ofstream& file = dynamic_cast< std::ofstream& >( log );
         if( file.is_open() )
            // enable exceptions, but only if we got an open file
            // (under MPI, only the master rank typically opens the log file and thus
            // logs from other ranks are ignored here)
            file.exceptions( std::ostream::failbit | std::ostream::badbit | std::ostream::eofbit );
      }
      catch( std::bad_cast& ) {
         // also enable exceptions if we did not get a file
         log.exceptions( std::ostream::failbit | std::ostream::badbit | std::ostream::eofbit );
      }
   }

   void
   setVerbose( int verbose )
   {
      this->verbose = verbose;
   }

   int getVerbose() const
   {
      return verbose;
   }

   virtual void setMetadataColumns( const MetadataColumns& elements )
   {
      // check if a header element changed (i.e. a first item of the pairs)
      if( metadataColumns.size() != elements.size() )
         header_changed = true;
      else
         for( std::size_t i = 0; i < metadataColumns.size(); i++ )
            if( metadataColumns[ i ].first != elements[ i ].first ) {
               header_changed = true;
               break;
            }
      metadataColumns = elements;
   }

   virtual void
   setMetadataElement( const typename MetadataColumns::value_type & element,
                       int insertPosition = -1 /* negative values insert from the end */ )
   {
      bool found = false;
      for( auto & it : metadataColumns )
         if( it.first == element.first ) {
            if( it.second != element.second )
               it.second = element.second;
            found = true;
            break;
         }
      if( ! found ) {
         if( insertPosition < 0 )
            metadataColumns.insert( metadataColumns.end() + insertPosition + 1, element );
         else
            metadataColumns.insert( metadataColumns.begin() + insertPosition, element );
         header_changed = true;
      }
   }

   virtual void
   setMetadataWidths( const std::map< std::string, int > & widths )
   {
      for( auto & it : widths )
         if( metadataWidths.count( it.first ) )
            metadataWidths[ it.first ] = it.second;
         else
            metadataWidths.insert( it );
   }

   virtual void
   logResult( const std::string& performer,
              const HeaderElements& headerElements,
              const RowElements& rowElements,
              const WidthHints& columnWidthHints,
              const std::string& errorMessage = "" ) = 0;

   virtual void writeErrorMessage( const std::string& message ) = 0;

protected:
   std::ostream& log;
   int verbose = 0;

   MetadataColumns metadataColumns;
   std::map< std::string, int > metadataWidths;
   bool header_changed = true;
};

} // namespace Benchmarks
} // namespace noa::TNL
