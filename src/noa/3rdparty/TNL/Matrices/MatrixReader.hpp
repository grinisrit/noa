// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <iomanip>
#include <sstream>
#include <TNL/String.h>
#include <TNL/Containers/Vector.h>
#include <TNL/Timer.h>
#include <TNL/Matrices/MatrixReader.h>

namespace TNL {
namespace Matrices {


template< typename Matrix, typename Device >
void
MatrixReader< Matrix, Device >::
readMtx( const TNL::String& fileName,
         Matrix& matrix,
         bool verbose )
{
   HostMatrix hostMatrix;
   MatrixReader< HostMatrix >::readMtx( fileName, hostMatrix, verbose );
   matrix = hostMatrix;
}

template< typename Matrix, typename Device >
void
MatrixReader< Matrix, Device >::
readMtx( std::istream& str,
         Matrix& matrix,
         bool verbose )
{
   HostMatrix hostMatrix;
   MatrixReader< HostMatrix >::readMtx( str, hostMatrix, verbose );
   matrix = hostMatrix;
}

/**
 * MatrixReader specialization for TNL::Devices::Host.
 */

template< typename Matrix >
void
MatrixReader< Matrix, TNL::Devices::Host >::
readMtx( const String& fileName,
         Matrix& matrix,
         bool verbose )
{
   std::fstream file;
   file.open( fileName.getString(), std::ios::in );
   if( ! file )
      throw std::runtime_error( std::string( "I am not able to open the file " ) + fileName.getString() );
   readMtx( file, matrix, verbose );
}

template< typename Matrix >
void
MatrixReader< Matrix, TNL::Devices::Host >::
readMtx( std::istream& file,
         Matrix& matrix,
         bool verbose )
{
   IndexType rows, columns;
   bool symmetricSourceMatrix( false );

   readMtxHeader( file, rows, columns, symmetricSourceMatrix, verbose );

   if( Matrix::isSymmetric() && !symmetricSourceMatrix )
      throw std::runtime_error( "Matrix is not symmetric, but flag for symmetric matrix is given. Aborting." );

   if( verbose )
      std::cout << "Matrix dimensions are " << rows << " x " << columns << std::endl;
   matrix.setDimensions( rows, columns );
   typename Matrix::RowsCapacitiesType rowLengths( rows );

   computeCompressedRowLengthsFromMtxFile( file, rowLengths, columns, rows, symmetricSourceMatrix, Matrix::isSymmetric(), verbose );

   matrix.setRowCapacities( rowLengths );

   readMatrixElementsFromMtxFile( file, matrix, symmetricSourceMatrix, verbose );
}

template< typename Matrix >
void
MatrixReader< Matrix, TNL::Devices::Host >::
verifyMtxFile( std::istream& file, const Matrix& matrix, bool verbose )
{
   bool symmetricSourceMatrix( false );
   IndexType rows, columns;
   readMtxHeader( file, rows, columns, symmetricSourceMatrix, false );
   file.clear();
   file.seekg( 0, std::ios::beg );
   String line;
   bool dimensionsLine( false );
   IndexType processedElements( 0 );
   Timer timer;
   timer.start();
   while( std::getline( file, line ) )
   {
      if( line[ 0 ] == '%' ) continue;
      if( ! dimensionsLine )
      {
         dimensionsLine = true;
         continue;
      }
      IndexType row( 1 ), column( 1 );
      RealType value;
      parseMtxLineWithElement( line, row, column, value );
      if( value != matrix.getElement( row-1, column-1 ) ||
          ( symmetricSourceMatrix && value != matrix.getElement( column-1, row-1 ) ) )
      {
         std::stringstream str;
         str << "*** !!! VERIFICATION ERROR !!! *** " << std::endl
             << "The elements differ at " << row-1 << " row " << column-1 << " column." << std::endl
             << "The matrix value is " << matrix.getElement( row-1, column-1 )
             << " while the file value is " << value << "." << std::endl;
         throw std::runtime_error( str.str() );
      }
      processedElements++;
      if( symmetricSourceMatrix && row != column )
         processedElements++;
      if( verbose )
        std::cout << " Verifying the matrix elements ... " << processedElements << " / " << matrix.getNumberOfMatrixElements() << "                       \r" << std::flush;
   }
   file.clear();
   long int fileSize = file.tellg();
   timer.stop();
   if( verbose )
     std::cout << " Verifying the matrix elements ... " << processedElements << " / " << matrix.getNumberOfMatrixElements()
           << " -> " << timer.getRealTime()
           << " sec. i.e. " << fileSize / ( timer.getRealTime() * ( 1 << 20 ))  << "MB/s." << std::endl;
}

template< typename Matrix >
bool
MatrixReader< Matrix, TNL::Devices::Host >::
findLineByElement( std::istream& file,
                   const IndexType& row,
                   const IndexType& column,
                   String& line,
                   IndexType& lineNumber )
{
   file.clear();
   file.seekg( 0, std::ios::beg );
   bool symmetricSourceMatrix( false );
   bool dimensionsLine( false );
   lineNumber = 0;
   while( std::getline( file, line ) )
   {
      lineNumber++;
      if( line[ 0 ] == '%' ) continue;
      if( ! dimensionsLine )
      {
         dimensionsLine = true;
         continue;
      }
      IndexType currentRow( 1 ), currentColumn( 1 );
      RealType value;
      parseMtxLineWithElement( line, currentRow, currentColumn, value );
      if( ( currentRow == row + 1 && currentColumn == column + 1 ) ||
          ( symmetricSourceMatrix && currentRow == column + 1 && currentColumn == row + 1 ) )
         return true;
   }
   return false;
}

template< typename Matrix >
void
MatrixReader< Matrix, TNL::Devices::Host >::checkMtxHeader( const String& header, bool& symmetric )
{
   std::vector< String > parsedLine = header.split( ' ', String::SplitSkip::SkipEmpty );
   if( (int) parsedLine.size() < 5 || parsedLine[ 0 ] != "%%MatrixMarket" )
      throw std::runtime_error( "Unknown format of the source file. We expect line like this: %%MatrixMarket matrix coordinate real general" );
   if( parsedLine[ 1 ] != "matrix" )
      throw std::runtime_error( std::string( "Keyword 'matrix' is expected in the header line: " ) + header.getString() );
   if( parsedLine[ 2 ] != "coordinates" &&
       parsedLine[ 2 ] != "coordinate" )
      throw std::runtime_error( std::string( "Error: Only 'coordinates' format is supported now, not " ) + parsedLine[ 2 ].getString() );
   if( parsedLine[ 3 ] != "real" )
      throw std::runtime_error( std::string( "Only 'real' matrices are supported, not " ) + parsedLine[ 3 ].getString() );
   if( parsedLine[ 4 ] != "general" )
   {
      if( parsedLine[ 4 ] == "symmetric" )
         symmetric = true;
      else
         throw std::runtime_error(  std::string( "Only 'general' matrices are supported, not "  ) + parsedLine[ 4 ].getString() );
   }
}

template< typename Matrix >
void
MatrixReader< Matrix, TNL::Devices::Host >::readMtxHeader( std::istream& file,
                                       IndexType& rows,
                                       IndexType& columns,
                                       bool& symmetric,
                                       bool verbose )
{
   file.clear();
   file.seekg( 0, std::ios::beg );
   String line;
   bool headerParsed( false );
   std::vector< String > parsedLine;
   while( true )
   {
      std::getline( file, line );
      if( ! headerParsed )
      {
         checkMtxHeader( line, symmetric );
         headerParsed = true;
         if( verbose && symmetric )
           std::cout << "The matrix is SYMMETRIC ... ";
         continue;
      }
      if( line[ 0 ] == '%' ) continue;

      parsedLine = line.split( ' ', String::SplitSkip::SkipEmpty );
      if( (int) parsedLine.size() != 3 )
         throw std::runtime_error( "Wrong number of parameters in the matrix header - should be 3." );
      rows = atoi( parsedLine[ 0 ].getString() );
      columns = atoi( parsedLine[ 1 ].getString() );
      if( verbose )
        std::cout << " The matrix has " << rows
              << " rows and " << columns << " columns. " << std::endl;

      if( rows <= 0 || columns <= 0 )
         throw std::runtime_error( "Row or column index is negative."  );
      break;
   }
}

template< typename Matrix >
void
MatrixReader< Matrix, TNL::Devices::Host >::
computeCompressedRowLengthsFromMtxFile( std::istream& file,
                                        Containers::Vector< int, DeviceType, int >& rowLengths,
                                        const int columns,
                                        const int rows,
                                        bool symmetricSourceMatrix,
                                        bool symmetricTargetMatrix,
                                        bool verbose )
{
   file.clear();
   file.seekg( 0,  std::ios::beg );
   rowLengths.setValue( 0 );
   String line;
   bool dimensionsLine( false );
   IndexType numberOfElements( 0 );
   Timer timer;
   timer.start();
   while( std::getline( file, line ) )
   {
      if( ! line.getSize() || line[ 0 ] == '%' ) continue;
      if( ! dimensionsLine )
      {
         dimensionsLine = true;
         continue;
      }
      IndexType row( 1 ), column( 1 );
      RealType value;
      parseMtxLineWithElement( line, row, column, value );
      numberOfElements++;
      if( column > columns || row > rows )
      {
         std::stringstream str;
         str << "There is an element at position " << row << ", " << column << " out of the matrix dimensions " << rows << " x " << columns << ".";
         throw std::runtime_error( str.str() );
      }
      if( verbose )
         std::cout << " Counting the matrix elements ... " << numberOfElements / 1000 << " thousands      \r" << std::flush;

      if( !symmetricTargetMatrix ||
          ( symmetricTargetMatrix && row >= column ) )
         rowLengths[ row - 1 ]++;
      else if( symmetricTargetMatrix && row < column )
         rowLengths[ column - 1 ]++;

      if( rowLengths[ row - 1 ] > columns )
      {
         std::stringstream str;
         str << "There are more elements ( " << rowLengths[ row - 1 ] << " ) than the matrix columns ( " << columns << " ) at the row " << row << ".";
         throw std::runtime_error( str.str() );
      }
      if( symmetricSourceMatrix && row != column && symmetricTargetMatrix )
      {
         rowLengths[ column - 1 ]++;
         if( rowLengths[ column - 1 ] > columns )
         {
            std::stringstream str;
            str << "There are more elements ( " << rowLengths[ row - 1 ] << " ) than the matrix columns ( " << columns << " ) at the row " << column << " .";
            throw std::runtime_error( str.str() );
         }
         continue;
      }
      else if( symmetricSourceMatrix && row != column && !symmetricTargetMatrix )
          rowLengths[ column - 1 ]++;
   }
   file.clear();
   long int fileSize = file.tellg();
   timer.stop();
   if( verbose )
     std::cout << " Counting the matrix elements ... " << numberOfElements / 1000
           << " thousands  -> " << timer.getRealTime()
           << " sec. i.e. " << fileSize / ( timer.getRealTime() * ( 1 << 20 ))  << "MB/s." << std::endl;
}

template< typename Matrix >
void
MatrixReader< Matrix, TNL::Devices::Host >::
readMatrixElementsFromMtxFile( std::istream& file,
                               Matrix& matrix,
                               bool symmetricSourceMatrix,
                               bool verbose )
{
   file.clear();
   file.seekg( 0,  std::ios::beg );
   String line;
   bool dimensionsLine( false );
   IndexType processedElements( 0 );
   Timer timer;
   timer.start();

   while( std::getline( file, line ) )
   {
      if( ! line.getSize() || line[ 0 ] == '%' ) continue;
      if( ! dimensionsLine )
      {
         dimensionsLine = true;
         continue;
      }
      IndexType row( 1 ), column( 1 );
      RealType value;
      parseMtxLineWithElement( line, row, column, value );

      if( ! Matrix::isSymmetric() || ( Matrix::isSymmetric() && row >= column ) )
         matrix.setElement( row - 1, column - 1, value );
      else if( Matrix::isSymmetric() && row < column )
         matrix.setElement( column - 1, row - 1, value );

      processedElements++;
      if( symmetricSourceMatrix && row != column && Matrix::isSymmetric() )
          continue;
      else if( symmetricSourceMatrix && row != column && ! Matrix::isSymmetric() )
      {
          matrix.setElement( column - 1, row - 1, value );
          processedElements++;
      }
   }

   file.clear();
   long int fileSize = file.tellg();
   timer.stop();
   if( verbose )
     std::cout << " Reading the matrix elements ... " << processedElements << " / " << matrix.getAllocatedElementsCount()
              << " -> " << timer.getRealTime()
              << " sec. i.e. " << fileSize / ( timer.getRealTime() * ( 1 << 20 ))  << "MB/s." << std::endl;
}

template< typename Matrix >
void
MatrixReader< Matrix, TNL::Devices::Host >::
parseMtxLineWithElement( const String& line,
                         IndexType& row,
                         IndexType& column,
                         RealType& value )
{
   std::vector< String > parsedLine = line.split( ' ', String::SplitSkip::SkipEmpty );
   if( (int) parsedLine.size() != 3 )
   {
      std::stringstream str;
      str << "Wrong number of parameters in the matrix row at line:" << line;
      throw std::runtime_error( str.str() );
   }
   row = atoi( parsedLine[ 0 ].getString() );
   column = atoi( parsedLine[ 1 ].getString() );
   value = ( RealType ) atof( parsedLine[ 2 ].getString() );
}

} // namespace Matrices
} // namespace TNL
