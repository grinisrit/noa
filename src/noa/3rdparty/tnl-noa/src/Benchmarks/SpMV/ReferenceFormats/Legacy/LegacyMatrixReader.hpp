#pragma once

#include <iomanip>
#include <sstream>
#include <TNL/String.h>
#include <TNL/Containers/Vector.h>
#include <TNL/Timer.h>
#include <TNL/Matrices/MatrixReader.h>

namespace TNL {
   namespace Benchmarks {
      namespace SpMV {
         namespace ReferenceFormats {
            namespace Legacy {


template< typename Matrix >
void LegacyMatrixReader< Matrix >::readMtxFile( const String& fileName,
                                             Matrix& matrix,
                                             bool verbose,
                                             bool symReader )
{
   std::fstream file;
   file.open( fileName.getString(), std::ios::in );
   if( ! file )
      throw std::runtime_error( std::string( "I am not able to open the file " ) + fileName.getString() );
   readMtxFile( file, matrix, verbose, symReader );
}

template< typename Matrix >
void LegacyMatrixReader< Matrix >::readMtxFile( std::istream& file,
                                             Matrix& matrix,
                                             bool verbose,
                                             bool symReader )
{
   MatrixReaderDeviceDependentCode< typename Matrix::DeviceType >::readMtxFile( file, matrix, verbose, symReader );
}

template< typename Matrix >
void LegacyMatrixReader< Matrix >::readMtxFileHostMatrix( std::istream& file,
                                                          Matrix& matrix,
                                                          typename Matrix::RowsCapacitiesType& rowLengths,
                                                          bool verbose,
                                                          bool symReader )
{
   IndexType rows, columns;
   bool symmetricMatrix( false );

   readMtxHeader( file, rows, columns, symmetricMatrix, verbose );

   if( symReader && !symmetricMatrix )
      throw std::runtime_error( "Matrix is not symmetric, but flag for symmetric matrix is given. Aborting." );

   matrix.setDimensions( rows, columns );
   rowLengths.setSize( rows );

   computeCompressedRowLengthsFromMtxFile( file, rowLengths, columns, rows, symmetricMatrix, verbose );

   matrix.setRowCapacities( rowLengths );

   readMatrixElementsFromMtxFile( file, matrix, symmetricMatrix, verbose, symReader );
}

template< typename Matrix >
void LegacyMatrixReader< Matrix >::verifyMtxFile( std::istream& file,
                                               const Matrix& matrix,
                                               bool verbose )
{
   bool symmetricMatrix( false );
   IndexType rows, columns;
   readMtxHeader( file, rows, columns, symmetricMatrix, false );
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
          ( symmetricMatrix && value != matrix.getElement( column-1, row-1 ) ) )
      {
         std::stringstream str;
         str << "*** !!! VERIFICATION ERROR !!! *** " << std::endl
             << "The elements differ at " << row-1 << " row " << column-1 << " column." << std::endl
             << "The matrix value is " << matrix.getElement( row-1, column-1 )
             << " while the file value is " << value << "." << std::endl;
         throw std::runtime_error( str.str() );
      }
      processedElements++;
      if( symmetricMatrix && row != column )
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
bool LegacyMatrixReader< Matrix >::findLineByElement( std::istream& file,
                                                   const IndexType& row,
                                                   const IndexType& column,
                                                   String& line,
                                                   IndexType& lineNumber )
{
   file.clear();
   file.seekg( 0, std::ios::beg );
   bool symmetricMatrix( false );
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
          ( symmetricMatrix && currentRow == column + 1 && currentColumn == row + 1 ) )
         return true;
   }
   return false;
}

template< typename Matrix >
bool LegacyMatrixReader< Matrix >::checkMtxHeader( const String& header,
                                                bool& symmetric )
{
   std::vector< String > parsedLine = header.split( ' ', String::SplitSkip::SkipEmpty );
   if( (int) parsedLine.size() < 5 || parsedLine[ 0 ] != "%%MatrixMarket" )
      return false;
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
   return true;
}

template< typename Matrix >
void LegacyMatrixReader< Matrix >::readMtxHeader( std::istream& file,
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
         headerParsed = checkMtxHeader( line, symmetric );
         if( verbose && symmetric )
           std::cout << "The matrix is SYMMETRIC ... ";
         continue;
      }
      if( line[ 0 ] == '%' ) continue;
      if( ! headerParsed )
         throw std::runtime_error( "Unknown format of the file. We expect line like this: %%MatrixMarket matrix coordinate real general" );

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
void LegacyMatrixReader< Matrix >::computeCompressedRowLengthsFromMtxFile( std::istream& file,
                                                              Containers::Vector< int, DeviceType, int >& rowLengths,
                                                              const int columns,
                                                              const int rows,
                                                              bool symmetricMatrix,
                                                              bool verbose,
                                                              bool symReader )
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

      if( !symReader ||
          ( symReader && row >= column ) )
         rowLengths[ row - 1 ]++;
      else if( symReader && row < column )
         rowLengths[ column - 1 ]++;

      if( rowLengths[ row - 1 ] > columns )
      {
         std::stringstream str;
         str << "There are more elements ( " << rowLengths[ row - 1 ] << " ) than the matrix columns ( " << columns << " ) at the row " << row << ".";
         throw std::runtime_error( str.str() );
      }
      if( symmetricMatrix && row != column && symReader )
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
      else if( symmetricMatrix && row != column && !symReader )
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
void LegacyMatrixReader< Matrix >::readMatrixElementsFromMtxFile( std::istream& file,
                                                               Matrix& matrix,
                                                               bool symmetricMatrix,
                                                               bool verbose,
                                                               bool symReader )
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

      if( !symReader ||
          ( symReader && row >= column ) )
         matrix.setElement( row - 1, column - 1, value );
      else if( symReader && row < column )
         matrix.setElement( column - 1, row - 1, value );

      processedElements++;
      if( symmetricMatrix && row != column && symReader )
          continue;
      else if( symmetricMatrix && row != column && !symReader )
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
void LegacyMatrixReader< Matrix >::parseMtxLineWithElement( const String& line,
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

/// This is to prevent from appearing in Doxygen documentation.
/// \cond HIDDEN_CLASS
template<>
class MatrixReaderDeviceDependentCode< Devices::Host >
{
   public:

   template< typename Matrix >
   static void readMtxFile( std::istream& file,
                            Matrix& matrix,
                            bool verbose,
                            bool symReader )
   {
      typename Matrix::RowsCapacitiesType rowLengths;
      LegacyMatrixReader< Matrix >::readMtxFileHostMatrix( file, matrix, rowLengths, verbose, symReader );
   }
};

template<>
class MatrixReaderDeviceDependentCode< Devices::Cuda >
{
   public:

   template< typename Matrix >
   static void readMtxFile( std::istream& file,
                            Matrix& matrix,
                            bool verbose,
                            bool symReader )
   {
      using HostMatrixType = typename Matrix::template Self< typename Matrix::RealType, Devices::Sequential >;
      using RowsCapacitiesType = typename HostMatrixType::RowsCapacitiesType;

      HostMatrixType hostMatrix;
      RowsCapacitiesType rowLengths;
      LegacyMatrixReader< Matrix >::readMtxFileHostMatrix( file, matrix, rowLengths, verbose, symReader );
   }
};
/// \endcond

            }// namespace Legacy
         }// namespace ReferenceFormats
      }// namespace SpMV
   } // namespace Benchmarks
} // namespace TNL
