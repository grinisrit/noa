// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <ostream>
#include <iostream>
#include <noa/3rdparty/tnl-noa/src/TNL/String.h>

namespace noa::TNL {
namespace Matrices {

/**
 * \brief Helper class for exporting of matrices to different output formats.
 *
 * Currently it supports:
 *
 * 1. [Coordinate MTX Format](https://math.nist.gov/MatrixMarket/formats.html#coord) is supported.
 * 2. Gnuplot format for matrix visualization in [Gnuplot](http://www.gnuplot.info).
 * 3. EPS format for matrix pattern visualization in [Encapsulated
 * PostScript](https://en.wikipedia.org/wiki/Encapsulated_PostScript)
 *
 * \tparam Matrix is a type of matrix into which we want to import the MTX file.
 * \tparam Device is used only for the purpose of template specialization.
 *
 * \par Example
 * \include Matrices/MatrixWriterReaderExample.cpp
 * \par Output
 * \include MatrixWriterReaderExample.out
 *
 */
template< typename Matrix, typename Device = typename Matrix::DeviceType >
class MatrixWriter
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
    * \brief Method for exporting matrix to file with given filename using Gnuplot format.
    *
    * \param fileName is the name of the target file.
    * \param matrix is the source matrix.
    * \param verbose controls verbosity of the matrix export.
    */
   static void
   writeGnuplot( const TNL::String& fileName, const Matrix& matrix, bool verbose = false );

   /**
    * \brief Method for exporting matrix to STL output stream using Gnuplot format.
    *
    * \param file is the output stream.
    * \param matrix is the source matrix.
    * \param verbose controls verbosity of the matrix export.
    */
   static void
   writeGnuplot( std::ostream& str, const Matrix& matrix, bool verbose = false );

   /**
    * \brief Method for exporting matrix to file with given filename using EPS format.
    *
    * \param fileName is the name of the target file.
    * \param matrix is the source matrix.
    * \param verbose controls verbosity of the matrix export.
    */
   static void
   writeEps( const TNL::String& fileName, const Matrix& matrix, bool verbose = false );

   /**
    * \brief Method for exporting matrix to STL output stream using EPS format.
    *
    * \param file is the output stream.
    * \param matrix is the source matrix.
    * \param verbose controls verbosity of the matrix export.
    */
   static void
   writeEps( std::ostream& str, const Matrix& matrix, bool verbose = false );

   /**
    * \brief Method for exporting matrix to file with given filename using MTX format.
    *
    * \param fileName is the name of the target file.
    * \param matrix is the source matrix.
    * \param verbose controls verbosity of the matrix export.
    */
   static void
   writeMtx( const TNL::String& fileName, const Matrix& matrix, bool verbose = false );

   /**
    * \brief Method for exporting matrix to STL output stream using MTX format.
    *
    * \param file is the output stream.
    * \param matrix is the source matrix.
    * \param verbose controls verbosity of the matrix export.
    */
   static void
   writeMtx( std::ostream& str, const Matrix& matrix, bool verbose = false );

protected:
   using HostMatrix = typename Matrix::template Self< RealType, TNL::Devices::Host >;
};

/// This is to prevent from appearing in Doxygen documentation.
/// \cond HIDDEN_CLASS
template< typename Matrix >
class MatrixWriter< Matrix, TNL::Devices::Host >
{
public:
   typedef typename Matrix::IndexType IndexType;
   typedef typename Matrix::RealType RealType;

   static void
   writeGnuplot( const TNL::String& fileName, const Matrix& matrix, bool verbose = false );

   static void
   writeGnuplot( std::ostream& str, const Matrix& matrix, bool verbose = false );

   static void
   writeEps( const TNL::String& fileName, const Matrix& matrix, bool verbose = false );

   static void
   writeEps( std::ostream& str, const Matrix& matrix, bool verbose = false );

   static void
   writeMtx( const TNL::String& fileName, const Matrix& matrix, bool verbose = false );

   static void
   writeMtx( std::ostream& str, const Matrix& matrix, bool verbose = false );

protected:
   static void
   writeEpsHeader( std::ostream& str, const Matrix& matrix, const int elementSize );

   static void
   writeEpsBody( std::ostream& str, const Matrix& matrix, const int elementSize, bool verbose );
};
/// \endcond

}  // namespace Matrices
}  // namespace noa::TNL

#include <noa/3rdparty/tnl-noa/src/TNL/Matrices/MatrixWriter.hpp>
