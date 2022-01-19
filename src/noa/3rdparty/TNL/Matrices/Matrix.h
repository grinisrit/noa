// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Object.h>
#include <TNL/Allocators/Default.h>
#include <TNL/Devices/Host.h>
#include <TNL/Containers/Vector.h>
#include <TNL/Containers/VectorView.h>
#include <TNL/Matrices/MatrixView.h>
#include <TNL/Algorithms/Segments/ElementsOrganization.h>

namespace TNL {
/**
 * \brief Namespace for different matrix types.
 */
namespace Matrices {

using Algorithms::Segments::ElementsOrganization;

/**
 * \brief Base class for other matrix types.
 *
 * \tparam Real is a type of matrix elements.
 * \tparam Device is a device where the matrix is allocated.
 * \tparam Index is a type for indexing of the matrix elements.
 * \tparam RealAllocator is allocator for the matrix elements values.
 */
template< typename Real = double,
          typename Device = Devices::Host,
          typename Index = int,
          typename RealAllocator = typename Allocators::Default< Device >::template Allocator< std::remove_const_t< Real > > >
class Matrix : public Object
{
   public:
      using RealAllocatorType = RealAllocator;
      using RowsCapacitiesType = Containers::Vector< Index, Device, Index >;
      using RowsCapacitiesView = Containers::VectorView< Index, Device, Index >;
      using ConstRowsCapacitiesView = typename RowsCapacitiesView::ConstViewType;

      /**
       * \brief The type of matrix elements.
       */
      using RealType = Real;

      /**
       * \brief The device where the matrix is allocated.
       */
      using DeviceType = Device;

      /**
       * \brief The type used for matrix elements indexing.
       */
      using IndexType = Index;

      /**
       * \brief Type of base matrix view.
       *
       */
      using ViewType = MatrixView< Real, Device, Index >;

      /**
       * \brief Type of base matrix view for constant instances.
       *
       */
      using ConstViewType = typename MatrixView< Real, Device, Index >::ConstViewType;

      /**
       * \brief Type of vector holding values of matrix elements.
       */
      using ValuesType = Containers::Vector< Real, Device, Index, RealAllocator >;

      /**
       * \brief Type of constant vector holding values of matrix elements.
       */
      using ConstValuesType = Containers::Vector< std::add_const_t< Real >, Device, Index, RealAllocator >;

      /**
       * \brief Type of vector view holding values of matrix elements.
       */
      using ValuesView = typename ViewType::ValuesView;

      /**
       * \brief Type of constant vector view holding values of matrix elements.
       */
      using ConstValuesView = typename ViewType::ConstValuesView;

      /**
       * \brief Construct a new Matrix object possibly with user defined allocator of the matrix values.
       *
       * \param allocator is is a user defined allocator of the matrix values.
       */
      Matrix( const RealAllocatorType& allocator = RealAllocatorType() );

      /**
       * \brief Construct a new Matrix object with given dimensions and possibly user defined allocator of the matrix values.
       *
       * \param rows is a number of matrix rows.
       * \param columns is a number of matrix columns.
       * \param allocator is a user defined allocator of the matrix values.
       */
      Matrix( const IndexType rows,
            const IndexType columns,
            const RealAllocatorType& allocator = RealAllocatorType() );

      /**
       * \brief Method for setting or changing of the matrix dimensions.
       *
       * \param rows is a number of matrix rows.
       * \param columns is a number of matrix columns.
       */
      virtual void setDimensions( const IndexType rows,
                                  const IndexType columns );

      /**
       * \brief Set the matrix dimensions to be equal to those of the input matrix.
       *
       * \tparam Matrix_ is a type if the input matrix.
       * \param matrix is an instance of the matrix.
       */
      template< typename Matrix_ >
      void setLike( const Matrix_& matrix );

      /**
       * \brief Tells the number of allocated matrix elements.
       *
       * In the case of dense matrices, this is just product of the number of rows and the number of columns.
       * But for other matrix types like sparse matrices, this can be different.
       *
       * \return Number of allocated matrix elements.
       */
      IndexType getAllocatedElementsCount() const;

      /**
       * \brief Computes a current number of nonzero matrix elements.
       *
       * \return number of nonzero matrix elements.
       */
      IndexType getNonzeroElementsCount() const;

      /**
       * \brief Reset the matrix.
       *
       * The matrix dimensions are set to zero and all matrix elements are freed from the memrory.
       */
      void reset();

      /**
       * \brief Returns number of matrix rows.
       *
       * \return number of matrix row.
       */
      __cuda_callable__
      IndexType getRows() const;

      /**
       * \brief Returns number of matrix columns.
       *
       * @return number of matrix columns.
       */
      __cuda_callable__
      IndexType getColumns() const;

      /**
       * \brief Returns a constant reference to a vector with the matrix elements values.
       *
       * \return constant reference to a vector with the matrix elements values.
       */
      const ValuesType& getValues() const;

      /**
       * \brief Returns a reference to a vector with the matrix elements values.
       *
       * \return constant reference to a vector with the matrix elements values.
       */
      ValuesType& getValues();

      /**
       * \brief Comparison operator with another arbitrary matrix type.
       *
       * \param matrix is the right-hand side matrix.
       * \return \e true if the RHS matrix is equal, \e false otherwise.
       */
      template< typename Matrix >
      bool operator == ( const Matrix& matrix ) const;

      /**
       * \brief Comparison operator with another arbitrary matrix type.
       *
       * \param matrix is the right-hand side matrix.
       * \return \e true if the RHS matrix is equal, \e false otherwise.
       */
      template< typename Matrix >
      bool operator != ( const Matrix& matrix ) const;

      /**
       * \brief Method for saving the matrix to a file.
       *
       * \param file is the output file.
       */
      virtual void save( File& file ) const;

      /**
       * \brief Method for loading the matrix from a file.
       *
       * \param file is the input file.
       */
      virtual void load( File& file );

      /**
       * \brief Method for printing the matrix to output stream.
       *
       * \param str is the output stream.
       */
      virtual void print( std::ostream& str ) const;


      // TODO: method for symmetric matrices, should not be in general Matrix interface
      //[[deprecated]]
      //__cuda_callable__
      //const IndexType& getNumberOfColors() const;

      // TODO: method for symmetric matrices, should not be in general Matrix interface
      //[[deprecated]]
      //void computeColorsVector(Containers::Vector<Index, Device, Index> &colorsVector);

      protected:

      IndexType rows, columns;

      // TODO: remove
      //IndexType numberOfColors;

      ValuesType values;
};

/**
 * \brief Overloaded insertion operator for printing a matrix to output stream.
 *
 * \tparam Real is a type of the matrix elements.
 * \tparam Device is a device where the matrix is allocated.
 * \tparam Index is a type used for the indexing of the matrix elements.
 *
 * \param str is a output stream.
 * \param matrix is the matrix to be printed.
 *
 * \return a reference on the output stream \ref std::ostream&.
 */
template< typename Real, typename Device, typename Index >
std::ostream& operator << ( std::ostream& str, const Matrix< Real, Device, Index >& matrix )
{
   matrix.print( str );
   return str;
}

} // namespace Matrices
} // namespace TNL

#include <TNL/Matrices/Matrix.hpp>
