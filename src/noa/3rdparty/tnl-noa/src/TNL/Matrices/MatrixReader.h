// Copyright (c) 2004-2023 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <istream>
#include <noa/3rdparty/tnl-noa/src/TNL/String.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Containers/Vector.h>

namespace noa::TNL {
namespace Matrices {

/**
 * \brief Helper class for importing of matrices from different input formats.
 *
 * Currently it supports:
 *
 * 1. [Coordinate MTX Format](https://math.nist.gov/MatrixMarket/formats.html#coord) is supported.
 *
 * \tparam Matrix is a type of matrix into which we want to import the MTX file.
 * \tparam Device is used only for the purpose of template specialization.
 *
 * \par Example
 * \include Matrices/MatrixWriterReaderExample.cpp
 * \par Output
 * \include MatrixWriterReaderExample.out
 */
template< typename Matrix, typename Device = typename Matrix::DeviceType >
class MatrixReader
{
public:
   /**
    * \brief Type of matrix elements values.
    */
   using RealType = typename Matrix::RealType;

   /**
    * \brief Device where the matrix is allocated.
    */
   using DeviceType = typename Matrix::RealType;

   /**
    * \brief Type used for indexing of matrix elements.
    */
   using IndexType = typename Matrix::IndexType;

   /**
    * \brief Method for importing matrix from file with given filename.
    *
    * \param fileName is the name of the source file.
    * \param matrix is the target matrix.
    * \param verbose controls verbosity of the matrix import.
    */
   static void
   readMtx( const String& fileName, Matrix& matrix, bool verbose = false );

   /**
    * \brief Method for importing matrix from STL input stream.
    *
    * \param str is the input stream.
    * \param matrix is the target matrix.
    * \param verbose controls verbosity of the matrix import.
    */
   static void
   readMtx( std::istream& str, Matrix& matrix, bool verbose = false );

protected:
   using HostMatrix = typename Matrix::template Self< RealType, TNL::Devices::Host >;
};

// This is to prevent from appearing in Doxygen documentation.
/// \cond
template< typename Matrix >
class MatrixReader< Matrix, TNL::Devices::Host >
{
public:
   /**
    * \brief Type of matrix elements values.
    */
   using RealType = typename Matrix::RealType;

   /**
    * \brief Device where the matrix is allocated.
    */
   using DeviceType = typename Matrix::DeviceType;

   /**
    * \brief Type used for indexing of matrix elements.
    */
   using IndexType = typename Matrix::IndexType;

   /**
    * \brief Method for importing matrix from file with given filename.
    *
    * \param fileName is the name of the source file.
    * \param matrix is the target matrix.
    * \param verbose controls verbosity of the matrix import.
    *
    * \par Example
    * \include Matrices/MatrixWriterReaderExample.cpp
    * \par Output
    * \include Matrices/MatrixWriterReaderExample.out
    *
    */
   static void
   readMtx( const String& fileName, Matrix& matrix, bool verbose = false );

   /**
    * \brief Method for importing matrix from STL input stream.
    *
    * \param file is the input stream.
    * \param matrix is the target matrix.
    * \param verbose controls verbosity of the matrix import.
    */
   static void
   readMtx( std::istream& file, Matrix& matrix, bool verbose = false );

protected:
   static void
   verifyMtxFile( std::istream& file, const Matrix& matrix, bool verbose = false );

   static bool
   findLineByElement( std::istream& file, const IndexType& row, const IndexType& column, String& line, IndexType& lineNumber );

   static void
   checkMtxHeader( const String& header, bool& symmetric );

   static void
   readMtxHeader( std::istream& file, IndexType& rows, IndexType& columns, bool& symmetricMatrix, bool verbose );

   static void
   computeCompressedRowLengthsFromMtxFile( std::istream& file,
                                           Containers::Vector< int, DeviceType, int >& rowLengths,
                                           int columns,
                                           int rows,
                                           bool symmetricSourceMatrix,
                                           bool symmetricTargetMatrix,
                                           bool verbose );

   static void
   readMatrixElementsFromMtxFile( std::istream& file, Matrix& matrix, bool symmetricMatrix, bool verbose );

   static void
   parseMtxLineWithElement( const String& line, IndexType& row, IndexType& column, RealType& value );
};
/// \endcond

}  // namespace Matrices
}  // namespace noa::TNL

#include <noa/3rdparty/tnl-noa/src/TNL/Matrices/MatrixReader.hpp>
