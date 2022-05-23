#pragma once

#include <istream>
#include <TNL/String.h>
#include <TNL/Containers/Vector.h>

namespace TNL {
   namespace Benchmarks {
      namespace SpMV {
         namespace ReferenceFormats {
            namespace Legacy {


/// This is to prevent from appearing in Doxygen documentation.
/// \cond HIDDEN_CLASS
template< typename Device >
class MatrixReaderDeviceDependentCode
{};
/// \endcond

template< typename Matrix >
class LegacyMatrixReader
{
   public:

   typedef typename Matrix::IndexType IndexType;
   typedef typename Matrix::DeviceType DeviceType;
   typedef typename Matrix::RealType RealType;

   static void readMtxFile( const String& fileName,
                            Matrix& matrix,
                            bool verbose = false,
                            bool symReader = false );

   static void readMtxFile( std::istream& file,
                            Matrix& matrix,
                            bool verbose = false,
                            bool symReader = false );

   static void readMtxFileHostMatrix( std::istream& file,
                                      Matrix& matrix,
                                      typename Matrix::RowsCapacitiesType& rowLengths,
                                      bool verbose,
                                      bool symReader );


   static void verifyMtxFile( std::istream& file,
                              const Matrix& matrix,
                              bool verbose = false );

   static bool findLineByElement( std::istream& file,
                                  const IndexType& row,
                                  const IndexType& column,
                                  String& line,
                                  IndexType& lineNumber );
   protected:

   static bool checkMtxHeader( const String& header,
                               bool& symmetric );

   static void readMtxHeader( std::istream& file,
                              IndexType& rows,
                              IndexType& columns,
                              bool& symmetricMatrix,
                              bool verbose );

   static void computeCompressedRowLengthsFromMtxFile( std::istream& file,
                                             Containers::Vector< int, DeviceType, int >& rowLengths,
                                             const int columns,
                                             const int rows,
                                             bool symmetricMatrix,
                                             bool verbose,
                                             bool symReader = false );

   static void readMatrixElementsFromMtxFile( std::istream& file,
                                              Matrix& matrix,
                                              bool symmetricMatrix,
                                              bool verbose,
                                              bool symReader );

   static void parseMtxLineWithElement( const String& line,
                                        IndexType& row,
                                        IndexType& column,
                                        RealType& value );
};

            }// namespace Legacy
         }// namespace ReferenceFormats
      }// namespace SpMV
   } // namespace Benchmarks
} // namespace TNL

#include <Benchmarks/SpMV/ReferenceFormats/Legacy/LegacyMatrixReader.hpp>
