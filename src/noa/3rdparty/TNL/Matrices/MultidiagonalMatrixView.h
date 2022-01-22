// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <noa/3rdparty/TNL/Matrices/MatrixView.h>
#include <noa/3rdparty/TNL/Containers/Vector.h>
#include <noa/3rdparty/TNL/Matrices/MultidiagonalMatrixRowView.h>
#include <noa/3rdparty/TNL/Algorithms/Segments/Ellpack.h>
#include <noa/3rdparty/TNL/Matrices/details/MultidiagonalMatrixIndexer.h>

namespace noaTNL {
namespace Matrices {

/**
 * \brief Implementation of sparse multidiagonal matrix.
 *
 * It serves as an accessor to \ref SparseMatrix for example when passing the
 * matrix to lambda functions. SparseMatrix view can be also created in CUDA kernels.
 *
 * See \ref MultidiagonalMatrix for more details.
 *
 * \tparam Real is a type of matrix elements.
 * \tparam Device is a device where the matrix is allocated.
 * \tparam Index is a type for indexing of the matrix elements.
 * \tparam Organization tells the ordering of matrix elements. It is either RowMajorOrder
 *         or ColumnMajorOrder.
 */
template< typename Real = double,
          typename Device = Devices::Host,
          typename Index = int,
          ElementsOrganization Organization = Algorithms::Segments::DefaultElementsOrganization< Device >::getOrganization() >
class MultidiagonalMatrixView : public MatrixView< Real, Device, Index >
{
   public:

      // Supporting types - they are not important for the user
      using BaseType = MatrixView< Real, Device, Index >;
      using ValuesViewType = typename BaseType::ValuesView;
      using IndexerType = details::MultidiagonalMatrixIndexer< Index, Organization >;
      using DiagonalsOffsetsView = Containers::VectorView< Index, Device, Index >;
      using HostDiagonalsOffsetsView = Containers::VectorView< Index, Devices::Host, Index >;

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
       * \brief Type of related matrix view.
       */
      using ViewType = MultidiagonalMatrixView< Real, Device, Index, Organization >;

      /**
       * \brief Matrix view type for constant instances.
       */
      using ConstViewType = MultidiagonalMatrixView< typename std::add_const< Real >::type, Device, Index, Organization >;

      /**
       * \brief Type for accessing matrix rows.
       */
      using RowView = MultidiagonalMatrixRowView< ValuesViewType, IndexerType, DiagonalsOffsetsView >;

      /**
       * \brief Type for accessing constant matrix rows.
       */
      using ConstRowView = typename RowView::ConstRowView;

      /**
       * \brief Helper type for getting self type or its modifications.
       */
      template< typename _Real = Real,
                typename _Device = Device,
                typename _Index = Index,
                ElementsOrganization Organization_ = Algorithms::Segments::DefaultElementsOrganization< Device >::getOrganization() >
      using Self = MultidiagonalMatrixView< _Real, _Device, _Index, Organization_ >;

      /**
       * \brief Constructor with no parameters.
       */
      __cuda_callable__
      MultidiagonalMatrixView();

      /**
       * \brief Constructor with all necessary data and views.
       *
       * \param values is a vector view with matrix elements values
       * \param diagonalsOffsets is a vector view with diagonals offsets
       * \param hostDiagonalsOffsets is a vector view with a copy of diagonals offsets on the host
       * \param indexer is an indexer of matrix elements
       */
      __cuda_callable__
      MultidiagonalMatrixView( const ValuesViewType& values,
                               const DiagonalsOffsetsView& diagonalsOffsets,
                               const HostDiagonalsOffsetsView& hostDiagonalsOffsets,
                               const IndexerType& indexer );

      /**
       * \brief Copy constructor.
       *
       * \param matrix is an input multidiagonal matrix view.
       */
      __cuda_callable__
      MultidiagonalMatrixView( const MultidiagonalMatrixView& view ) = default;

      /**
       * \brief Move constructor.
       *
       * \param matrix is an input multidiagonal matrix view.
       */
      __cuda_callable__
      MultidiagonalMatrixView( MultidiagonalMatrixView&& view ) = default;

      /**
       * \brief Returns a modifiable view of the multidiagonal matrix.
       *
       * \return multidiagonal matrix view.
       */
      ViewType getView();

      /**
       * \brief Returns a non-modifiable view of the multidiagonal matrix.
       *
       * \return multidiagonal matrix view.
       */
      ConstViewType getConstView() const;

      /**
       * \brief Returns string with serialization type.
       *
       * The string has a form `Matrices::MultidiagonalMatrix< RealType,  [any_device], IndexType, Organization, [any_allocator], [any_allocator] >`.
       *
       * See \ref MultidiagonalMatrix::getSerializationType.
       *
       * \return \ref String with the serialization type.
       */
      static String getSerializationType();

      /**
       * \brief Returns string with serialization type.
       *
       * See \ref MultidiagonalMatrix::getSerializationType.
       *
       * \return \ref String with the serialization type.
       */
      virtual String getSerializationTypeVirtual() const;

      /**
       * \brief Returns number of diagonals.
       *
       * \return Number of diagonals.
       */
      __cuda_callable__
      const IndexType getDiagonalsCount() const;

      /**
       * \brief Compute capacities of all rows.
       *
       * The row capacities are not stored explicitly and must be computed.
       *
       * \param rowCapacities is a vector where the row capacities will be stored.
       */
      template< typename Vector >
      void getRowCapacities( Vector& rowCapacities ) const;

      /**
       * \brief Computes number of non-zeros in each row.
       *
       * \param rowLengths is a vector into which the number of non-zeros in each row
       * will be stored.
       *
       * \par Example
       * \include Matrices/MultidiagonalMatrix/MultidiagonalMatrixViewExample_getCompressedRowLengths.cpp
       * \par Output
       * \include MultidiagonalMatrixViewExample_getCompressedRowLengths.out
       */
      template< typename Vector >
      void getCompressedRowLengths( Vector& rowLengths ) const;

      [[deprecated]]
      IndexType getRowLength( const IndexType row ) const;

      /**
       * \brief Returns number of non-zero matrix elements.
       *
       * This method really counts the non-zero matrix elements and so
       * it returns zero for matrix having all allocated elements set to zero.
       *
       * \return number of non-zero matrix elements.
       */
      IndexType getNonzeroElementsCount() const;

      /**
       * \brief Comparison operator with another multidiagonal matrix.
       *
       * \tparam Real_ is \e Real type of the source matrix.
       * \tparam Device_ is \e Device type of the source matrix.
       * \tparam Index_ is \e Index type of the source matrix.
       * \tparam Organization_ is \e Organization of the source matrix.
       *
       * \return \e true if both matrices are identical and \e false otherwise.
       */
      template< typename Real_,
                typename Device_,
                typename Index_,
                ElementsOrganization Organization_ >
      bool operator == ( const MultidiagonalMatrixView< Real_, Device_, Index_, Organization_ >& matrix ) const;

      /**
       * \brief Comparison operator with another multidiagonal matrix.
       *
       * \tparam Real_ is \e Real type of the source matrix.
       * \tparam Device_ is \e Device type of the source matrix.
       * \tparam Index_ is \e Index type of the source matrix.
       * \tparam Organization_ is \e Organization of the source matrix.
       *
       * \param matrix is the source matrix.
       *
       * \return \e true if both matrices are NOT identical and \e false otherwise.
       */
      template< typename Real_,
                typename Device_,
                typename Index_,
                ElementsOrganization Organization_ >
      bool operator != ( const MultidiagonalMatrixView< Real_, Device_, Index_, Organization_ >& matrix ) const;

      /**
       * \brief Non-constant getter of simple structure for accessing given matrix row.
       *
       * \param rowIdx is matrix row index.
       *
       * \return RowView for accessing given matrix row.
       *
       * \par Example
       * \include Matrices/MultidiagonalMatrix/MultidiagonalMatrixViewExample_getRow.cpp
       * \par Output
       * \include MultidiagonalMatrixViewExample_getRow.out
       *
       * See \ref MultidiagonalMatrixRowView.
       */
      __cuda_callable__
      RowView getRow( const IndexType& rowIdx );

      /**
       * \brief Constant getter of simple structure for accessing given matrix row.
       *
       * \param rowIdx is matrix row index.
       *
       * \return RowView for accessing given matrix row.
       *
       * \par Example
       * \include Matrices/MultidiagonalMatrix/MultidiagonalMatrixViewExample_getConstRow.cpp
       * \par Output
       * \include MultidiagonalMatrixViewExample_getConstRow.out
       *
       * See \ref MultidiagonalMatrixRowView.
       */
      __cuda_callable__
      const ConstRowView getRow( const IndexType& rowIdx ) const;

      /**
       * \brief Set all matrix elements to given value.
       *
       * \param value is the new value of all matrix elements.
       */
      void setValue( const RealType& v );

      /**
       * \brief Sets element at given \e row and \e column to given \e value.
       *
       * This method can be called from the host system (CPU) no matter
       * where the matrix is allocated. If the matrix is allocated on GPU this method
       * can be called even from device kernels. If the matrix is allocated in GPU device
       * this method is called from CPU, it transfers values of each matrix element separately and so the
       * performance is very low. For higher performance see. \ref MultidiagonalMatrix::getRow
       * or \ref MultidiagonalMatrix::forElements and \ref MultidiagonalMatrix::forAllElements.
       * The call may fail if the matrix row capacity is exhausted.
       *
       * \param row is row index of the element.
       * \param column is columns index of the element.
       * \param value is the value the element will be set to.
       *
       * \par Example
       * \include Matrices/MultidiagonalMatrix/MultidiagonalMatrixViewExample_setElement.cpp
       * \par Output
       * \include MultidiagonalMatrixViewExample_setElement.out
       */
      __cuda_callable__
      void setElement( const IndexType row,
                       const IndexType column,
                       const RealType& value );

      /**
       * \brief Add element at given \e row and \e column to given \e value.
       *
       * This method can be called from the host system (CPU) no matter
       * where the matrix is allocated. If the matrix is allocated on GPU this method
       * can be called even from device kernels. If the matrix is allocated in GPU device
       * this method is called from CPU, it transfers values of each matrix element separately and so the
       * performance is very low. For higher performance see. \ref MultidiagonalMatrix::getRow
       * or \ref MultidiagonalMatrix::forElements and \ref MultidiagonalMatrix::forAllElements.
       * The call may fail if the matrix row capacity is exhausted.
       *
       * \param row is row index of the element.
       * \param column is columns index of the element.
       * \param value is the value the element will be set to.
       * \param thisElementMultiplicator is multiplicator the original matrix element
       *   value is multiplied by before addition of given \e value.
       *
       * \par Example
       * \include Matrices/MultidiagonalMatrix/MultidiagonalMatrixViewExample_addElement.cpp
       * \par Output
       * \include MultidiagonalMatrixViewExample_addElement.out
       */
      __cuda_callable__
      void addElement( const IndexType row,
                       const IndexType column,
                       const RealType& value,
                       const RealType& thisElementMultiplicator = 1.0 );

      /**
       * \brief Returns value of matrix element at position given by its row and column index.
       *
       * This method can be called from the host system (CPU) no matter
       * where the matrix is allocated. If the matrix is allocated on GPU this method
       * can be called even from device kernels. If the matrix is allocated in GPU device
       * this method is called from CPU, it transfers values of each matrix element separately and so the
       * performance is very low. For higher performance see. \ref MultidiagonalMatrix::getRow
       * or \ref MultidiagonalMatrix::forElements and \ref MultidiagonalMatrix::forAllElements.
       *
       * \param row is a row index of the matrix element.
       * \param column i a column index of the matrix element.
       *
       * \return value of given matrix element.
       *
       * \par Example
       * \include Matrices/MultidiagonalMatrix/MultidiagonalMatrixViewExample_getElement.cpp
       * \par Output
       * \include MultidiagonalMatrixViewExample_getElement.out
       */
      __cuda_callable__
      RealType getElement( const IndexType row,
                           const IndexType column ) const;

      /**
       * \brief Method for performing general reduction on matrix rows for constant instances.
       *
       * \tparam Fetch is a type of lambda function for data fetch declared as
       *
       * ```
       * auto fetch = [=] __cuda_callable__ ( IndexType rowIdx, IndexType& columnIdx, RealType& elementValue ) -> FetchValue { ... };
       * ```
       *
       *  The return type of this lambda can be any non void.
       * \tparam Reduce is a type of lambda function for reduction declared as
       *
       * ```
       * auto reduce = [=] __cuda_callable__ ( const FetchValue& v1, const FetchValue& v2 ) -> FetchValue { ... };
       * ```
       *
       * \tparam Keep is a type of lambda function for storing results of reduction in each row. It is declared as
       *
       * ```
       * auto keep = [=] __cuda_callable__ ( const IndexType rowIdx, const double& value ) { ... };
       * ```
       *
       * \tparam FetchValue is type returned by the Fetch lambda function.
       *
       * \param begin defines beginning of the range [ \e begin, \e end ) of rows to be processed.
       * \param end defines ending of the range [ \e begin, \e end ) of rows to be processed.
       * \param fetch is an instance of lambda function for data fetch.
       * \param reduce is an instance of lambda function for reduction.
       * \param keep in an instance of lambda function for storing results.
       * \param identity is the [identity element](https://en.wikipedia.org/wiki/Identity_element)
       *                 for the reduction operation, i.e. element which does not
       *                 change the result of the reduction.
       *
       * \par Example
       * \include Matrices/MultidiagonalMatrix/MultidiagonalMatrixViewExample_reduceRows.cpp
       * \par Output
       * \include MultidiagonalMatrixViewExample_reduceRows.out
       */
      template< typename Fetch, typename Reduce, typename Keep, typename FetchReal >
      void reduceRows( IndexType begin, IndexType end, Fetch& fetch, Reduce& reduce, Keep& keep, const FetchReal& zero ) const;

      /**
       * \brief Method for performing general reduction on matrix rows.
       *
       * \tparam Fetch is a type of lambda function for data fetch declared as
       *
       * ```
       * auto fetch = [=] __cuda_callable__ ( IndexType rowIdx, IndexType& columnIdx, RealType& elementValue ) -> FetchValue { ... };
       * ```
       *
       * The return type of this lambda can be any non void.
       * \tparam Reduce is a type of lambda function for reduction declared as
       *
       * ```
       * auto reduce = [=] __cuda_callable__ ( const FetchValue& v1, const FetchValue& v2 ) -> FetchValue { ... };
       * ```
       *
       * \tparam Keep is a type of lambda function for storing results of reduction in each row. It is declared as
       *
       * ```
       * auto keep = [=] __cuda_callable__ ( const IndexType rowIdx, const double& value ) { ... };
       * ```
       *
       * \tparam FetchValue is type returned by the Fetch lambda function.
       *
       * \param begin defines beginning of the range [ \e begin, \e end ) of rows to be processed.
       * \param end defines ending of the range [ \e begin, \e end ) of rows to be processed.
       * \param fetch is an instance of lambda function for data fetch.
       * \param reduce is an instance of lambda function for reduction.
       * \param keep in an instance of lambda function for storing results.
       * \param identity is the [identity element](https://en.wikipedia.org/wiki/Identity_element)
       *                 for the reduction operation, i.e. element which does not
       *                 change the result of the reduction.
       *
       * \par Example
       * \include Matrices/MultidiagonalMatrix/MultidiagonalMatrixViewExample_reduceRows.cpp
       * \par Output
       * \include MultidiagonalMatrixViewExample_reduceRows.out
       */
      template< typename Fetch, typename Reduce, typename Keep, typename FetchReal >
      void reduceRows( IndexType begin, IndexType end, Fetch& fetch, Reduce& reduce, Keep& keep, const FetchReal& zero );

      /**
       * \brief Method for performing general reduction on all matrix rows for constant instances.
       *
       * \tparam Fetch is a type of lambda function for data fetch declared as
       *
       * ```
       * auto fetch = [=] __cuda_callable__ ( IndexType rowIdx, IndexType& columnIdx, RealType& elementValue ) -> FetchValue { ... };
       * ```
       *
       * The return type of this lambda can be any non void.
       * \tparam Reduce is a type of lambda function for reduction declared as
       *
       * ```
       * auto reduce = [=] __cuda_callable__ ( const FetchValue& v1, const FetchValue& v2 ) -> FetchValue { ... };
       * ```
       *
       * \tparam Keep is a type of lambda function for storing results of reduction in each row. It is declared as
       *
       * ```
       * auto keep = [=] __cuda_callable__ ( const IndexType rowIdx, const double& value ) { ... };
       * ```
       *
       * \tparam FetchValue is type returned by the Fetch lambda function.
       *
       * \param fetch is an instance of lambda function for data fetch.
       * \param reduce is an instance of lambda function for reduction.
       * \param keep in an instance of lambda function for storing results.
       * \param identity is the [identity element](https://en.wikipedia.org/wiki/Identity_element)
       *                 for the reduction operation, i.e. element which does not
       *                 change the result of the reduction.
       *
       * \par Example
       * \include Matrices/MultidiagonalMatrix/MultidiagonalMatrixViewExample_reduceAllRows.cpp
       * \par Output
       * \include MultidiagonalMatrixViewExample_reduceAllRows.out
       */
      template< typename Fetch, typename Reduce, typename Keep, typename FetchReal >
      void reduceAllRows( Fetch& fetch, Reduce& reduce, Keep& keep, const FetchReal& identity ) const;

      /**
       * \brief Method for performing general reduction on all matrix rows.
       *
       * \tparam Fetch is a type of lambda function for data fetch declared as
       *
       * ```
       * auto fetch = [=] __cuda_callable__ ( IndexType rowIdx, IndexType& columnIdx, RealType& elementValue ) -> FetchValue { ... };
       * ```
       *
       * The return type of this lambda can be any non void.
       * \tparam Reduce is a type of lambda function for reduction declared as
       *
       * ```
       * auto reduce = [=] __cuda_callable__ ( const FetchValue& v1, const FetchValue& v2 ) -> FetchValue { ... };
       * ```
       *
       * \tparam Keep is a type of lambda function for storing results of reduction in each row.
       *          It is declared as `keep( const IndexType rowIdx, const double& value )`.
       * \tparam FetchValue is type returned by the Fetch lambda function.
       *
       * \param fetch is an instance of lambda function for data fetch.
       * \param reduce is an instance of lambda function for reduction.
       * \param keep in an instance of lambda function for storing results.
       * \param identity is the [identity element](https://en.wikipedia.org/wiki/Identity_element)
       *                 for the reduction operation, i.e. element which does not
       *                 change the result of the reduction.
       *
       * \par Example
       * \include Matrices/MultidiagonalMatrix/MultidiagonalMatrixViewExample_reduceAllRows.cpp
       * \par Output
       * \include MultidiagonalMatrixViewExample_reduceAllRows.out
       */
      template< typename Fetch, typename Reduce, typename Keep, typename FetchReal >
      void reduceAllRows( Fetch& fetch, Reduce& reduce, Keep& keep, const FetchReal& identity );

      /**
       * \brief Method for iteration over all matrix rows for constant instances.
       *
       * \tparam Function is type of lambda function that will operate on matrix elements. It is should have form like
       *
       * ```
       * auto function = [=] __cuda_callable__ ( IndexType rowIdx, IndexType localIdx, IndexType columnIdx, const RealType& value ) { ... };
       * ```
       *
       * where
       *
       * \e rowIdx is an index of the matrix row.
       *
       * \e localIdx parameter is a rank of the non-zero element in given row. It is also, in fact,
       *  index of the matrix subdiagonal.
       *
       * \e columnIdx is a column index of the matrix element.
       *
       * \e value is the matrix element value.
       *
       * \param begin defines beginning of the range [ \e begin, \e end ) of rows to be processed.
       * \param end defines ending of the range [ \e begin, \e end ) of rows to be processed.
       * \param function is an instance of the lambda function to be called in each row.
       *
       * \par Example
       * \include Matrices/MultidiagonalMatrix/MultidiagonalMatrixViewExample_forRows.cpp
       * \par Output
       * \include MultidiagonalMatrixViewExample_forRows.out
       */
      template< typename Function >
      void forElements( IndexType begin, IndexType end, Function& function ) const;

      /**
       * \brief Method for iteration over all matrix rows for non-constant instances.
       *
       * \tparam Function is type of lambda function that will operate on matrix elements. It is should have form like
       *
       * ```
       * auto function = [=] __cuda_callable__ ( IndexType rowIdx, IndexType localIdx, IndexType columnIdx, const RealType& value ) { ... };
       * ```
       *
       * where
       *
       * \e rowIdx is an index of the matrix row.
       *
       * \e localIdx parameter is a rank of the non-zero element in given row. It is also, in fact,
       *  index of the matrix subdiagonal.
       *
       * \e columnIdx is a column index of the matrix element.
       *
       * \e value is a reference to the matrix element value. It can be used even for changing the matrix element value.
       *
       * \param begin defines beginning of the range [ \e begin, \e end ) of rows to be processed.
       * \param end defines ending of the range [ \e begin, \e end ) of rows to be processed.
       * \param function is an instance of the lambda function to be called in each row.
       *
       * \par Example
       * \include Matrices/MultidiagonalMatrix/MultidiagonalMatrixViewExample_forRows.cpp
       * \par Output
       * \include MultidiagonalMatrixViewExample_forRows.out
       */
      template< typename Function >
      void forElements( IndexType begin, IndexType end, Function& function );

      /**
       * \brief This method calls \e forElements for all matrix rows (for constant instances).
       *
       * See \ref MultidiagonalMatrix::forElements.
       *
       * \tparam Function is a type of lambda function that will operate on matrix elements.
       * \param function  is an instance of the lambda function to be called in each row.
       *
       * \par Example
       * \include Matrices/MultidiagonalMatrix/MultidiagonalMatrixViewExample_forAllRows.cpp
       * \par Output
       * \include MultidiagonalMatrixViewExample_forAllRows.out
       */
      template< typename Function >
      void forAllElements( Function& function ) const;

      /**
       * \brief This method calls \e forElements for all matrix rows.
       *
       * See \ref MultidiagonalMatrix::forElements.
       *
       * \tparam Function is a type of lambda function that will operate on matrix elements.
       * \param function  is an instance of the lambda function to be called in each row.
       *
       * \par Example
       * \include Matrices/MultidiagonalMatrix/MultidiagonalMatrixViewExample_forAllRows.cpp
       * \par Output
       * \include MultidiagonalMatrixViewExample_forAllRows.out
       */
      template< typename Function >
      void forAllElements( Function& function );

      /**
       * \brief Method for parallel iteration over matrix rows from interval [ \e begin, \e end).
       *
       * In each row, given lambda function is performed. Each row is processed by at most one thread unlike the method
       * \ref MultidiagonalMatrixView::forElements where more than one thread can be mapped to each row.
       *
       * \tparam Function is type of the lambda function.
       *
       * \param begin defines beginning of the range [ \e begin,\e end ) of rows to be processed.
       * \param end defines ending of the range [ \e begin, \e end ) of rows to be processed.
       * \param function is an instance of the lambda function to be called for each row.
       *
       * ```
       * auto function = [] __cuda_callable__ ( RowView& row ) mutable { ... };
       * ```
       *
       * \e RowView represents matrix row - see \ref noaTNL::Matrices::MultidiagonalMatrixView::RowView.
       *
       * \par Example
       * \include Matrices/MultidiagonalMatrix/MultidiagonalMatrixViewExample_forRows.cpp
       * \par Output
       * \include MultidiagonalMatrixViewExample_forRows.out
       */
      template< typename Function >
      void forRows( IndexType begin, IndexType end, Function&& function );

      /**
       * \brief Method for parallel iteration over matrix rows from interval [ \e begin, \e end) for constant instances.
       *
       * In each row, given lambda function is performed. Each row is processed by at most one thread unlike the method
       * \ref MultidiagonalMatrixView::forElements where more than one thread can be mapped to each row.
       *
       * \tparam Function is type of the lambda function.
       *
       * \param begin defines beginning of the range [ \e begin,\e end ) of rows to be processed.
       * \param end defines ending of the range [ \e begin, \e end ) of rows to be processed.
       * \param function is an instance of the lambda function to be called for each row.
       *
       * ```
       * auto function = [] __cuda_callable__ ( RowView& row ) { ... };
       * ```
       *
       * \e RowView represents matrix row - see \ref noaTNL::Matrices::MultidiagonalMatrixView::RowView.
       *
       * \par Example
       * \include Matrices/MultidiagonalMatrix/MultidiagonalMatrixViewExample_forRows.cpp
       * \par Output
       * \include MultidiagonalMatrixViewExample_forRows.out
       */
      template< typename Function >
      void forRows( IndexType begin, IndexType end, Function&& function ) const;

      /**
       * \brief Method for parallel iteration over all matrix rows.
       *
       * In each row, given lambda function is performed. Each row is processed by at most one thread unlike the method
       * \ref MultidiagonalMatrixView::forAllElements where more than one thread can be mapped to each row.
       *
       * \tparam Function is type of the lambda function.
       *
       * \param function is an instance of the lambda function to be called for each row.
       *
       * ```
       * auto function = [] __cuda_callable__ ( RowView& row ) mutable { ... };
       * ```
       *
       * \e RowView represents matrix row - see \ref noaTNL::Matrices::MultidiagonalMatrixView::RowView.
       *
       * \par Example
       * \include Matrices/MultidiagonalMatrix/MultidiagonalMatrixViewExample_forRows.cpp
       * \par Output
       * \include MultidiagonalMatrixViewExample_forRows.out
       */
      template< typename Function >
      void forAllRows( Function&& function );

      /**
       * \brief Method for parallel iteration over all matrix rows for constant instances.
       *
       * In each row, given lambda function is performed. Each row is processed by at most one thread unlike the method
       * \ref MultidiagonalMatrixView::forAllElements where more than one thread can be mapped to each row.
       *
       * \tparam Function is type of the lambda function.
       *
       * \param function is an instance of the lambda function to be called for each row.
       *
       * ```
       * auto function = [] __cuda_callable__ ( RowView& row ) { ... };
       * ```
       *
       * \e RowView represents matrix row - see \ref noaTNL::Matrices::MultidiagonalMatrixView::RowView.
       *
       * \par Example
       * \include Matrices/MultidiagonalMatrix/MultidiagonalMatrixViewExample_forRows.cpp
       * \par Output
       * \include MultidiagonalMatrixViewExample_forRows.out
       */
      template< typename Function >
      void forAllRows( Function&& function ) const;

      /**
       * \brief Method for sequential iteration over all matrix rows for constant instances.
       *
       * \tparam Function is type of lambda function that will operate on matrix elements. It is should have form like
       *
       * ```
       * auto function = [] __cuda_callable__ ( const RowView& row ) { ... };
       * ```
       *
       * \e RowView represents matrix row - see \ref noaTNL::Matrices::MultidiagonalMatrixView::RowView.
       *
       * \param begin defines beginning of the range [ \e begin, \e end ) of rows to be processed.
       * \param end defines ending of the range [ \e begin, \e end ) of rows to be processed.
       * \param function is an instance of the lambda function to be called in each row.
       */
      template< typename Function >
      void sequentialForRows( IndexType begin, IndexType end, Function& function ) const;

      /**
       * \brief Method for sequential iteration over all matrix rows for non-constant instances.
       *
       * \tparam Function is type of lambda function that will operate on matrix elements. It is should have form like
       *
       * ```
       * auto function = [] __cuda_callable__ ( RowView& row ) { ... };
       * ```
       *
       * \e RowView represents matrix row - see \ref noaTNL::Matrices::MultidiagonalMatrixView::RowView.
       *
       * \param begin defines beginning of the range [ \e  begin, \e end ) of rows to be processed.
       * \param end defines ending of the range [ \e begin, \e end ) of rows to be processed.
       * \param function is an instance of the lambda function to be called in each row.
       */
      template< typename Function >
      void sequentialForRows( IndexType begin, IndexType end, Function& function );

      /**
       * \brief This method calls \e sequentialForRows for all matrix rows (for constant instances).
       *
       * See \ref MultidiagonalMatrixView::sequentialForRows.
       *
       * \tparam Function is a type of lambda function that will operate on matrix elements.
       * \param function  is an instance of the lambda function to be called in each row.
       */
      template< typename Function >
      void sequentialForAllRows( Function& function ) const;

      /**
       * \brief This method calls \e sequentialForRows for all matrix rows.
       *
       * See \ref MultidiagonalMatrixView::sequentialForAllRows.
       *
       * \tparam Function is a type of lambda function that will operate on matrix elements.
       * \param function  is an instance of the lambda function to be called in each row.
       */
      template< typename Function >
      void sequentialForAllRows( Function& function );

      /**
       * \brief Computes product of matrix and vector.
       *
       * More precisely, it computes:
       *
       * ```
       * outVector = matrixMultiplicator * ( * this ) * inVector + outVectorMultiplicator * outVector
       * ```
       *
       * \tparam InVector is type of input vector.  It can be \ref Vector,
       *     \ref VectorView, \ref Array, \ref ArraView or similar container.
       * \tparam OutVector is type of output vector. It can be \ref Vector,
       *     \ref VectorView, \ref Array, \ref ArraView or similar container.
       *
       * \param inVector is input vector.
       * \param outVector is output vector.
       * \param matrixMultiplicator is a factor by which the matrix is multiplied. It is one by default.
       * \param outVectorMultiplicator is a factor by which the outVector is multiplied before added
       *    to the result of matrix-vector product. It is zero by default.
       * \param begin is the beginning of the rows range for which the vector product
       *    is computed. It is zero by default.
       * \param end is the end of the rows range for which the vector product
       *    is computed. It is number if the matrix rows by default.
       */
      template< typename InVector,
                typename OutVector >
      void vectorProduct( const InVector& inVector,
                          OutVector& outVector,
                          const RealType matrixMultiplicator = 1.0,
                          const RealType outVectorMultiplicator = 0.0,
                          const IndexType begin = 0,
                          IndexType end = 0 ) const;

      template< typename Real_, typename Device_, typename Index_, ElementsOrganization Organization_ >
      void addMatrix( const MultidiagonalMatrixView< Real_, Device_, Index_, Organization_ >& matrix,
                      const RealType& matrixMultiplicator = 1.0,
                      const RealType& thisMatrixMultiplicator = 1.0 );

      template< typename Real2, typename Index2 >
      void getTransposition( const MultidiagonalMatrixView< Real2, Device, Index2 >& matrix,
                             const RealType& matrixMultiplicator = 1.0 );

      /**
       * \brief Assignment of exactly the same matrix type.
       *
       * \param matrix is input matrix for the assignment.
       * \return reference to this matrix.
       */
      MultidiagonalMatrixView& operator=( const MultidiagonalMatrixView& view );

      /**
       * \brief Method for saving the matrix to a file.
       *
       * \param file is the output file.
       */
      void save( File& file ) const;

      /**
       * \brief Method for saving the matrix to the file with given filename.
       *
       * \param fileName is name of the file.
       */
      void save( const String& fileName ) const;

      /**
       * \brief Method for printing the matrix to output stream.
       *
       * \param str is the output stream.
       */
      void print( std::ostream& str ) const;

      /**
       * \brief This method returns matrix elements indexer used by this matrix.
       *
       * \return constant reference to the indexer.
       */
      __cuda_callable__
      const IndexerType& getIndexer() const;

      /**
       * \brief This method returns matrix elements indexer used by this matrix.
       *
       * \return non-constant reference to the indexer.
       */
      __cuda_callable__
      IndexerType& getIndexer();

      /**
       * \brief Returns padding index denoting padding zero elements.
       *
       * These elements are used for efficient data alignment in memory.
       *
       * \return value of the padding index.
       */
      __cuda_callable__
      IndexType getPaddingIndex() const;

   protected:

      DiagonalsOffsetsView diagonalsOffsets;

      HostDiagonalsOffsetsView hostDiagonalsOffsets;

      IndexerType indexer;
};

} // namespace Matrices
} // namespace noaTNL

#include <noa/3rdparty/TNL/Matrices/MultidiagonalMatrixView.hpp>
