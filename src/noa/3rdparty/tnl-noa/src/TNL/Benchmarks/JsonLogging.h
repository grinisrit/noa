// Copyright (c) 2004-2023 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

// Implemented by: Jakub Klinkovsky,
//                 Tomas Oberhuber

#pragma once

#include "Logging.h"
#include <noa/3rdparty/tnl-noa/src/TNL/Assert.h>

namespace noa::TNL {
namespace Benchmarks {

class JsonLogging : public Logging
{
public:
   // inherit constructors
   using Logging::Logging;

   void
   writeHeader( const HeaderElements& headerElements, const WidthHints& widths )
   {
      TNL_ASSERT_EQ( headerElements.size(), widths.size(), "elements must have equal sizes" );
      if( verbose > 0 && ( header_changed || headerElements != lastHeaderElements ) ) {
         for( const auto& lg : metadataColumns ) {
            const int width = ( metadataWidths.count( lg.first ) > 0 ) ? metadataWidths[ lg.first ] : 14;
            std::cout << std::setw( width ) << lg.first;
         }
         for( std::size_t i = 0; i < headerElements.size(); i++ )
            std::cout << std::setw( widths[ i ] ) << headerElements[ i ];
         std::cout << std::endl;
         header_changed = false;
         lastHeaderElements = headerElements;
      }
   }

   void
   writeRow( const HeaderElements& headerElements,
             const RowElements& rowElements,
             const WidthHints& widths,
             const std::string& errorMessage )
   {
      TNL_ASSERT_EQ( headerElements.size(), rowElements.size(), "elements must have equal sizes" );
      TNL_ASSERT_EQ( headerElements.size(), widths.size(), "elements must have equal sizes" );

      log << "{";

      // write common logs
      int idx( 0 );
      for( const auto& lg : this->metadataColumns ) {
         if( verbose > 0 ) {
            const int width = ( metadataWidths.count( lg.first ) > 0 ) ? metadataWidths[ lg.first ] : 14;
            std::cout << std::setw( width ) << lg.second;
         }
         if( idx++ > 0 )
            log << ", ";
         log << "\"" << lg.first << "\": \"" << lg.second << "\"";
      }

      std::size_t i = 0;
      for( const auto& el : rowElements ) {
         if( verbose > 0 )
            std::cout << std::setw( widths[ i ] ) << el;
         if( idx++ > 0 )
            log << ", ";
         log << "\"" << headerElements[ i ] << "\": \"" << el << "\"";
         i++;
      }
      if( ! errorMessage.empty() ) {
         if( idx++ > 0 )
            log << ", ";
         log << "\"error\": \"" << escape_json( errorMessage ) << "\"";
      }
      log << "}" << std::endl;
      if( verbose > 0 )
         std::cout << std::endl;
   }

   void
   logResult( const std::string& performer,
              const HeaderElements& headerElements,
              const RowElements& rowElements,
              const WidthHints& columnWidthHints,
              const std::string& errorMessage = "" ) override
   {
      setMetadataElement( { "performer", performer } );
      writeHeader( headerElements, columnWidthHints );
      writeRow( headerElements, rowElements, columnWidthHints, errorMessage );
   }

   void
   writeErrorMessage( const std::string& message ) override
   {
      log << "{";

      // write common logs
      int idx( 0 );
      for( const auto& lg : this->metadataColumns ) {
         if( idx++ > 0 )
            log << ", ";
         log << "\"" << lg.first << "\": \"" << lg.second << "\"";
      }

      if( idx++ > 0 )
         log << ", ";
      log << "\"error\": \"" << escape_json( message ) << "\"";

      log << "}" << std::endl;
   }

protected:
   // manual double -> string conversion with fixed precision
   static std::string
   _to_string( double num, int precision = 0, bool fixed = false )
   {
      std::stringstream str;
      if( fixed )
         str << std::fixed;
      if( precision > 0 )
         str << std::setprecision( precision );
      str << num;
      return str.str();
   }

   // https://stackoverflow.com/a/33799784
   static std::string
   escape_json( const std::string& s )
   {
      std::ostringstream o;
      for( auto c : s ) {
         switch( c ) {
            case '"':
               o << "\\\"";
               break;
            case '\\':
               o << "\\\\";
               break;
            case '\b':
               o << "\\b";
               break;
            case '\f':
               o << "\\f";
               break;
            case '\n':
               o << "\\n";
               break;
            case '\r':
               o << "\\r";
               break;
            case '\t':
               o << "\\t";
               break;
            default:
               if( '\x00' <= c && c <= '\x1f' )
                  o << "\\u" << std::hex << std::setw( 4 ) << std::setfill( '0' ) << (int) c;
               else
                  o << c;
         }
      }
      return o.str();
   }

   // for tracking changes of the header elements written to the terminal
   HeaderElements lastHeaderElements;
};

}  // namespace Benchmarks
}  // namespace noa::TNL
