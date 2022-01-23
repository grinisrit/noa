// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <noa/3rdparty/TNL/Matrices/Matrix.h>
#include <noa/3rdparty/TNL/Matrices/MatrixType.h>
#include <noa/3rdparty/TNL/Allocators/Default.h>
#include <noa/3rdparty/TNL/Algorithms/Segments/CSR.h>
#include <noa/3rdparty/TNL/Matrices/SparseMatrixRowView.h>
#include <noa/3rdparty/TNL/TypeTraits.h>

namespace noa::TNL {
namespace Matrices {

/// This is to prevent from appearing in Doxygen documentation.
/// \cond HIDDEN_CLASS
template< typename Real, typename Index = int >
struct ChooseSparseMatrixComputeReal
{
   using type = Real;
};

template< typename Index>
struct ChooseSparseMatrixComputeReal< bool, Index >
{
   using type = Index;
};
/// \endcond

/**
 * \brief Implementation of sparse matrix view.
 *
 * It serves as an accessor to \ref SparseMatrix for example when passing the
 * matrix to lambda functions. SparseMatrix view can be also created in CUDA kernels.
 *
 * \tparam Real is a type of matrix elements. If \e Real equals \e bool the matrix is treated
 *    as binary and so the matrix elements values are not stored in the memory since we need
 *    to remember only coordinates of non-zero elements( which equal one).
 * \tparam Device is a device where the matrix is allocated.
 * \tparam Index is a type for indexing of the matrix elements.
 * \tparam MatrixType specifies a symmetry of matrix. See \ref MatrixType. Symmetric
 *    matrices store only lower part of the matrix and its diagonal. The upper part is reconstructed on the fly.
 *    GeneralMatrix with no symmetry is used by default.
 * \tparam Segments is a structure representing the sparse matrix format. Depending on the pattern of the non-zero elements
 *    different matrix formats can perform differently especially on GPUs. By default \ref CSR format is used. See also
 *    \ref Ellpack, \ref SlicedEllpack, \ref ChunkedEllpack or \ref BiEllpack.
 * \tparam ComputeReal is the same as \e Real mostly but for binary matrices it is set to \e Index type. This can be changed
 *    bu the user, of course.
 *
 */
template< typename Real,
          typename Device = Devices::Host,
          typename Index = int,
          typename MatrixType = GeneralMatrix,
          template< typename Device_, typename Index_ > class SegmentsView = Algorithms::Segments::CSRViewDefault,
          typename ComputeReal = typename ChooseSparseMatrixComputeReal< Real, Index >::type >
class SparseMatrixView : public MatrixView< Real, Device, Index >
{
   static_assert(
      ! MatrixType::isSymmetric() ||
      ! std::is_same< Device, Devices::Cuda >::value ||
      ( std::is_same< Real, float >::value || std::is_same< Real, double >::value || std::is_same< Real, int >::value || std::is_same< Real, long long int >::value || std::is_same< Real, bool >::value ),
      "Given Real type is not supported by atomic operations on GPU which are necessary for symmetric operations." );

   public:

      // Supporting types - they are not important for the user
      using BaseType = MatrixView< Real, Device, Index >;
      using ValuesViewType = typename BaseType::ValuesView;
      using ConstValuesViewType = typename ValuesViewType::ConstViewType;
      using ColumnsIndexesViewType = Containers::VectorView< typename noa::TNL::copy_const< Index >::template from< Real >::type, Device, Index >;
      using ConstColumnsIndexesViewType = typename ColumnsIndexesViewType::ConstViewType;
      using RowsCapacitiesView = Containers::VectorView< Index, Device, Index >;
      using ConstRowsCapacitiesView = typename RowsCapacitiesView::ConstViewType;

      /**
       * \brief Test of symmetric matrix type.
       *
       * \return \e true if the matrix is stored as symmetric and \e false otherwise.
       */
      static constexpr bool isSymmetric() { return MatrixType::isSymmetric(); };

      /**
       * \brief Test of binary matrix type.
       *
       * \return \e true if the matrix is stored as binary and \e false otherwise.
       */
      static constexpr bool isBinary() { return std::is_same< Real, bool >::value; };

      /**
       * \brief The type of matrix elements.
       */
      using RealType = std::remove_const_t< Real >;

      using ComputeRealType = ComputeReal;

      /**
       * \brief The device where the matrix is allocated.
       */
      using DeviceType = Device;

      /**
       * \brief The type used for matrix elements indexing.
       */
      using IndexType = Index;

      /**
       * \brief Templated type of segments view, i.e. sparse matrix format.
       */
      template< typename Device_, typename Index_ >
      using SegmentsViewTemplate = SegmentsView< Device_, Index_ >;

      /**
       * \brief Type of segments view used by this matrix. It represents the sparse matrix format.
       */
      using SegmentsViewType = SegmentsView< Device, Index >;

      /**
       * \brief Type of related matrix view.
       */
      using ViewType = SparseMatrixView< Real, Device, Index, MatrixType, SegmentsViewTemplate >;

      /**
       * \brief Matrix view type for constant instances.
       */
      using ConstViewType = SparseMatrixView< std::add_const_t< Real >, Device, Index, MatrixType, SegmentsViewTemplate >;

      /**
       * \brief Type for accessing matrix rows.
       */
      using RowView = SparseMatrixRowView< typename SegmentsViewType::SegmentViewType, ValuesViewType, ColumnsIndexesViewType >;

      /**
       * \brief Type for accessing constant matrix rows.
       */
      using ConstRowView = SparseMatrixRowView< typename SegmentsViewType::SegmentViewType, ConstValuesViewType, ConstColumnsIndexesViewType >;;

      /**
       * \brief Helper type for getting self type or its modifications.
       */
      template< typename _Real = Real,
                typename _Device = Device,
                typename _Index = Index,
                typename _MatrixType = MatrixType,
                template< typename, typename > class _SegmentsView = SegmentsView,
                typename _ComputeReal = ComputeReal >
      using Self = SparseMatrixView< _Real, _Device, _Index, _MatrixType, _SegmentsView, _ComputeReal >;

      /**
       * \brief Constructor with no parameters.
       */
      __cuda_callable__
      SparseMatrixView();

      /**
       * \brief Constructor with all necessary data and views.
       *
       * \param rows is a number of matrix rows.
       * \param columns is a number of matrix columns.
       * \param values is a vector view with matrix elements values.
       * \param columnIndexes is a vector view with matrix elements column indexes.
       * \param segments is a segments view representing the sparse matrix format.
       */
      __cuda_callable__
      SparseMatrixView( const IndexType rows,
                        const IndexType columns,
                        const ValuesViewType& values,
                        const ColumnsIndexesViewType& columnIndexes,
                        const SegmentsViewType& segments );

      /**
       * \brief Copy constructor.
       *
       * \param matrix is an input sparse matrix view.
       */
      __cuda_callable__
      SparseMatrixView( const SparseMatrixView& matrix ) = default;

      /**
       * \brief Move constructor.
       *
       * \param matrix is an input sparse matrix view.
       */
      __cuda_callable__
      SparseMatrixView( SparseMatrixView&& matrix ) = default;

      /**
       * \brief Returns a modifiable view of the sparse matrix.
       *
       * \return sparse matrix view.
       */
      __cuda_callable__
      ViewType getView();

      /**
       * \brief Returns a non-modifiable view of the sparse matrix.
       *
       * \return sparse matrix view.
       */
      __cuda_callable__
      ConstViewType getConstView() const;

      /**
       * \brief Returns string with serialization type.
       *
       * The string has a form `Matrices::SparseMatrix< RealType,  [any_device], IndexType, General/Symmetric, Format, [any_allocator] >`.
       *
       * \return \ref String with the serialization type.
       *
       * \par Example
       * \include Matrices/SparseMatrix/SparseMatrixViewExample_getSerializationType.cpp
       * \par Output
       * \include SparseMatrixViewExample_getSerializationType.out
       */
      static String getSerializationType();

      /**
       * \brief Returns string with serialization type.
       *
       * See \ref SparseMatrix::getSerializationType.
       *
       * \return \e String with the serialization type.
       *
       * \par Example
       * \include Matrices/SparseMatrix/SparseMatrixExample_getSerializationType.cpp
       * \par Output
       * \include SparseMatrixExample_getSerializationType.out
       */
      virtual String getSerializationTypeVirtual() const;

      /**
       * \brief Computes number of non-zeros in each row.
       *
       * \param rowLengths is a vector into which the number of non-zeros in each row
       * will be stored.
       *
       * \par Example
       * \include Matrices/SparseMatrix/SparseMatrixViewExample_getCompressedRowLengths.cpp
       * \par Output
       * \include SparseMatrixViewExample_getCompressedRowLengths.out
       */
      template< typename Vector >
      void getCompressedRowLengths( Vector& rowLengths ) const;

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
       * \brief Returns capacity of given matrix row.
       *
       * \param row index of matrix row.
       * \return number of matrix elements allocated for the row.
       */
      __cuda_callable__
      IndexType getRowCapacity( const IndexType row ) const;

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
       * \brief Constant getter of simple structure for accessing given matrix row.
       *
       * \param rowIdx is matrix row index.
       *
       * \return RowView for accessing given matrix row.
       *
       * \par Example
       * \include Matrices/SparseMatrix/SparseMatrixViewExample_getConstRow.cpp
       * \par Output
       * \include SparseMatrixViewExample_getConstRow.out
       *
       * See \ref SparseMatrixRowView.
       */
      __cuda_callable__
      ConstRowView getRow( const IndexType& rowIdx ) const;

      /**
       * \brief Non-constant getter of simple structure for accessing given matrix row.
       *
       * \param rowIdx is matrix row index.
       *
       * \return RowView for accessing given matrix row.
       *
       * \par Example
       * \include Matrices/SparseMatrix/SparseMatrixViewExample_getRow.cpp
       * \par Output
       * \include SparseMatrixViewExample_getRow.out
       *
       * See \ref SparseMatrixRowView.
       */
      __cuda_callable__
      RowView getRow( const IndexType& rowIdx );

      /**
       * \brief Sets element at given \e row and \e column to given \e value.
       *
       * This method can be called from the host system (CPU) no matter
       * where the matrix is allocated. If the matrix is allocated on GPU this method
       * can be called even from device kernels. If the matrix is allocated in GPU device
       * this method is called from CPU, it transfers values of each matrix element separately and so the
       * performance is very low. For higher performance see. \ref SparseMatrix::getRow
       * or \ref SparseMatrix::forElements and \ref SparseMatrix::forAllElements.
       * The call may fail if the matrix row capacity is exhausted.
       *
       * \param row is row index of the element.
       * \param column is columns index of the element.
       * \param value is the value the element will be set to.
       *
       * \par Example
       * \include Matrices/SparseMatrix/SparseMatrixViewExample_setElement.cpp
       * \par Output
       * \include SparseMatrixViewExample_setElement.out
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
       * performance is very low. For higher performance see. \ref SparseMatrix::getRow
       * or \ref SparseMatrix::forElements and \ref SparseMatrix::forAllElements.
       * The call may fail if the matrix row capacity is exhausted.
       *
       * \param row is row index of the element.
       * \param column is columns index of the element.
       * \param value is the value the element will be set to.
       * \param thisElementMultiplicator is multiplicator the original matrix element
       *   value is multiplied by before addition of given \e value.
       *
       * \par Example
       * \include Matrices/SparseMatrix/SparseMatrixViewExample_addElement.cpp
       * \par Output
       * \include SparseMatrixViewExample_addElement.out
       */
      __cuda_callable__
      void addElement( IndexType row,
                       IndexType column,
                       const RealType& value,
                       const RealType& thisElementMultiplicator = 1.0 );

      /**
       * \brief Returns value of matrix element at position given by its row and column index.
       *
       * This method can be called from the host system (CPU) no matter
       * where the matrix is allocated. If the matrix is allocated on GPU this method
       * can be called even from device kernels. If the matrix is allocated in GPU device
       * this method is called from CPU, it transfers values of each matrix element separately and so the
       * performance is very low. For higher performance see. \ref SparseMatrix::getRow
       * or \ref SparseMatrix::forElements and \ref SparseMatrix::forAllElements.
       *
       * \param row is a row index of the matrix element.
       * \param column i a column index of the matrix element.
       *
       * \return value of given matrix element.
       *
       * \par Example
       * \include Matrices/SparseMatrix/SparseMatrixViewExample_getElement.cpp
       * \par Output
       * \include SparseMatrixViewExample_getElement.out
       *
       */
      __cuda_callable__
      RealType getElement( IndexType row,
                           IndexType column ) const;

      /**
       * \brief Method for performing general reduction on matrix rows.
       *
       * \tparam Fetch is a type of lambda function for data fetch declared as
       *
       * ```
       * auto fetch = [=] __cuda_callable__ ( IndexType rowIdx, IndexType& columnIdx, RealType& elementValue ) -> FetchValue
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
       * \include Matrices/SparseMatrix/SparseMatrixViewExample_reduceRows.cpp
       * \par Output
       * \include SparseMatrixViewExample_reduceRows.out
       */
      template< typename Fetch, typename Reduce, typename Keep, typename FetchReal >
      void reduceRows( IndexType begin, IndexType end, Fetch& fetch, const Reduce& reduce, Keep& keep, const FetchReal& identity );

      /**
       * \brief Method for performing general reduction on matrix rows for constant instances.
       *
       * \tparam Fetch is a type of lambda function for data fetch declared as
       *
       * ```
       * auto fetch = [=] __cuda_callable__ ( IndexType rowIdx, IndexType& columnIdx, RealType& elementValue ) -> FetchValue
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
       * auto keep = [=] __cuda_callable__ ( const IndexType rowIdx, const RealType& value ) { ... };
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
       * \include Matrices/SparseMatrix/SparseMatrixViewExample_reduceRows.cpp
       * \par Output
       * \include SparseMatrixViewExample_reduceRows.out
       */
      template< typename Fetch, typename Reduce, typename Keep, typename FetchReal >
      void reduceRows( IndexType begin, IndexType end, Fetch& fetch, const Reduce& reduce, Keep& keep, const FetchReal& identity ) const;

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
       * \include Matrices/SparseMatrix/SparseMatrixViewExample_reduceAllRows.cpp
       * \par Output
       * \include SparseMatrixViewExample_reduceAllRows.out
       */
      template< typename Fetch, typename Reduce, typename Keep, typename FetchReal >
      void reduceAllRows( Fetch& fetch, const Reduce& reduce, Keep& keep, const FetchReal& identity );

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
       * \include Matrices/SparseMatrix/SparseMatrixViewExample_reduceAllRows.cpp
       * \par Output
       * \include SparseMatrixViewExample_reduceAllRows.out
       */
      template< typename Fetch, typename Reduce, typename Keep, typename FetchReal >
      void reduceAllRows( Fetch& fetch, const Reduce& reduce, Keep& keep, const FetchReal& identity ) const;

      /**
       * \brief Method for iteration over all matrix rows for constant instances.
       *
       * \tparam Function is type of lambda function that will operate on matrix elements. It is should have form like
       *
       * ```
       * auto function = [=] __cuda_callable__ ( IndexType rowIdx, IndexType localIdx, IndexType columnIdx, const RealType& value ) { ... };
       * ```
       *
       *  The \e localIdx parameter is a rank of the non-zero element in given row.
       *
       * \param begin defines beginning of the range [ \e begin, \e end ) of rows to be processed.
       * \param end defines ending of the range [ \e begin,\e end ) of rows to be processed.
       * \param function is an instance of the lambda function to be called in each row.
       *
       * \par Example
       * \include Matrices/SparseMatrix/SparseMatrixViewExample_forRows.cpp
       * \par Output
       * \include SparseMatrixViewExample_forRows.out
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
       *  The \e localIdx parameter is a rank of the non-zero element in given row.
       *
       * \param begin defines beginning of the range [ \e begin, \e end ) of rows to be processed.
       * \param end defines ending of the range [ \e begin, \e end ) of rows to be processed.
       * \param function is an instance of the lambda function to be called in each row.
       *
       * \par Example
       * \include Matrices/SparseMatrix/SparseMatrixViewExample_forRows.cpp
       * \par Output
       * \include SparseMatrixViewExample_forRows.out
       */
      template< typename Function >
      void forElements( IndexType begin, IndexType end, Function& function );

      /**
       * \brief This method calls \e forElements for all matrix rows (for constant instances).
       *
       * See \ref SparseMatrix::forElements.
       *
       * \tparam Function is a type of lambda function that will operate on matrix elements.
       * \param function  is an instance of the lambda function to be called in each row.
       *
       * \par Example
       * \include Matrices/SparseMatrix/SparseMatrixViewExample_forAllRows.cpp
       * \par Output
       * \include SparseMatrixViewExample_forAllRows.out
       */
      template< typename Function >
      void forAllElements( Function& function ) const;

      /**
       * \brief This method calls \e forElements for all matrix rows.
       *
       * See \ref SparseMatrix::forElements.
       *
       * \tparam Function is a type of lambda function that will operate on matrix elements.
       * \param function  is an instance of the lambda function to be called in each row.
       *
       * \par Example
       * \include Matrices/SparseMatrix/SparseMatrixViewExample_forAllRows.cpp
       * \par Output
       * \include SparseMatrixViewExample_forAllRows.out
       */
      template< typename Function >
      void forAllElements( Function& function );

      /**
       * \brief Method for parallel iteration over matrix rows from interval [ \e begin, \e end).
       *
       * In each row, given lambda function is performed. Each row is processed by at most one thread unlike the method
       * \ref SparseMatrixView::forElements where more than one thread can be mapped to each row.

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
       * \e RowView represents matrix row - see \ref noa::TNL::Matrices::SparseMatrixView::RowView.
       *
       * \par Example
       * \include Matrices/SparseMatrix/SparseMatrixViewExample_forRows.cpp
       * \par Output
       * \include SparseMatrixViewExample_forRows.out
       */
      template< typename Function >
      void forRows( IndexType begin, IndexType end, Function&& function );

      /**
       * \brief Method for parallel iteration over matrix rows from interval [ \e begin, \e end) for constant instances.
       *
       * In each row, given lambda function is performed. Each row is processed by at most one thread unlike the method
       * \ref SparseMatrixView::forElements where more than one thread can be mapped to each row.
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
       * \e RowView represents matrix row - see \ref noa::TNL::Matrices::SparseMatrixView::RowView.
       *
       * \par Example
       * \include Matrices/SparseMatrix/SparseMatrixViewExample_forRows.cpp
       * \par Output
       * \include SparseMatrixViewExample_forRows.out
       */
      template< typename Function >
      void forRows( IndexType begin, IndexType end, Function&& function ) const;

      /**
       * \brief Method for parallel iteration over all matrix rows.
       *
       * In each row, given lambda function is performed. Each row is processed by at most one thread unlike the method
       * \ref SparseMatrixView::forAllElements where more than one thread can be mapped to each row.
       *
       * \tparam Function is type of the lambda function.
       *
       * \param function is an instance of the lambda function to be called for each row.
       *
       * ```
       * auto function = [] __cuda_callable__ ( RowView& row ) mutable { ... };
       * ```
       *
       * \e RowView represents matrix row - see \ref noa::TNL::Matrices::SparseMatrixView::RowView.
       *
       * \par Example
       * \include Matrices/SparseMatrix/SparseMatrixViewExample_forRows.cpp
       * \par Output
       * \include SparseMatrixViewExample_forRows.out
       */
      template< typename Function >
      void forAllRows( Function&& function );

      /**
       * \brief Method for parallel iteration over all matrix rows for constant instances.
       *
       * In each row, given lambda function is performed. Each row is processed by at most one thread unlike the method
       * \ref SparseMatrixView::forAllElements where more than one thread can be mapped to each row.
       *
       * \tparam Function is type of the lambda function.
       *
       * \param function is an instance of the lambda function to be called for each row.
       *
       * ```
       * auto function = [] __cuda_callable__ ( RowView& row ) { ... };
       * ```
       *
       * \e RowView represents matrix row - see \ref noa::TNL::Matrices::SparseMatrixView::RowView.
       *
       * \par Example
       * \include Matrices/SparseMatrix/SparseMatrixViewExample_forRows.cpp
       * \par Output
       * \include SparseMatrixViewExample_forRows.out
       */
      template< typename Function >
      void forAllRows( Function&& function ) const;

      /**
       * \brief Method for sequential iteration over all matrix rows for constant instances.
       *
       * \tparam Function is type of lambda function that will operate on matrix elements. It is should have form like
       *
       * ```
       * auto function = [] __cuda_callable__ ( RowView& row ) { ... };
       * ```
       *
       * \e RowView represents matrix row - see \ref noa::TNL::Matrices::SparseMatrixView::RowView.
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
       * \e RowView represents matrix row - see \ref noa::TNL::Matrices::SparseMatrixView::RowView.
       *
       * \param begin defines beginning of the range [ \e begin, \e end ) of rows to be processed.
       * \param end defines ending of the range [ \e begin, \e end ) of rows to be processed.
       * \param function is an instance of the lambda function to be called in each row.
       */
      template< typename Function >
      void sequentialForRows( IndexType begin, IndexType end, Function& function );

      /**
       * \brief This method calls \e sequentialForRows for all matrix rows (for constant instances).
       *
       * See \ref SparseMatrixView::sequentialForRows.
       *
       * \tparam Function is a type of lambda function that will operate on matrix elements.
       * \param function  is an instance of the lambda function to be called in each row.
       */
      template< typename Function >
      void sequentialForAllRows( Function& function ) const;

      /**
       * \brief This method calls \e sequentialForRows for all matrix rows.
       *
       * See \ref SparseMatrixView::sequentialForAllRows.
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
                          const ComputeRealType matrixMultiplicator = 1.0,
                          const ComputeRealType outVectorMultiplicator = 0.0,
                          const IndexType begin = 0,
                          IndexType end = 0 ) const;

      /**
       * \brief Assignment of any matrix type.
       * .
       * \param matrix is input matrix for the assignment.
       * \return reference to this matrix.
       */
      SparseMatrixView& operator=( const SparseMatrixView& matrix );

      /**
       * \brief Comparison operator with another arbitrary matrix type.
       *
       * \param matrix is the right-hand side matrix.
       * \return \e true if the RHS matrix is equal, \e false otherwise.
       */
      template< typename Matrix >
      bool operator==( const Matrix& m ) const;

      /**
       * \brief Comparison operator with another arbitrary matrix type.
       *
       * \param matrix is the right-hand side matrix.
       * \return \e true if the RHS matrix is equal, \e false otherwise.
       */
      template< typename Matrix >
      bool operator!=( const Matrix& m ) const;

      /**
       * \brief Method for saving the matrix to the file with given filename.
       *
       * \param fileName is name of the file.
       */
      void save( const String& fileName ) const;

      /**
       * \brief Method for saving the matrix to a file.
       *
       * \param file is the output file.
       */
      void save( File& file ) const;

      /**
       * \brief Method for printing the matrix to output stream.
       *
       * \param str is the output stream.
       */
      void print( std::ostream& str ) const;

      /**
       * \brief Getter of segments for non-constant instances.
       *
       * \e Segments are a structure for addressing the matrix elements columns and values.
       * In fact, \e Segments represent the sparse matrix format.
       *
       * \return Non-constant reference to segments.
       */
      SegmentsViewType& getSegments();

      /**
       * \brief Getter of segments for constant instances.
       *
       * \e Segments are a structure for addressing the matrix elements columns and values.
       * In fact, \e Segments represent the sparse matrix format.
       *
       * \return Constant reference to segments.
       */
      const SegmentsViewType& getSegments() const;

      /**
       * \brief Getter of column indexes for constant instances.
       *
       * \return Constant reference to a vector with matrix elements column indexes.
       */
      const ColumnsIndexesViewType& getColumnIndexes() const;

      /**
       * \brief Getter of column indexes for nonconstant instances.
       *
       * \return Reference to a vector with matrix elements column indexes.
       */
      ColumnsIndexesViewType& getColumnIndexes();

      /**
       * \brief Returns a padding index value.
       *
       * Padding index is used for column indexes of padding zeros. Padding zeros
       * are used in some sparse matrix formats for better data alignment in memory.
       *
       * \return value of the padding index.
       */
      __cuda_callable__
      IndexType getPaddingIndex() const;

   protected:

      ColumnsIndexesViewType columnIndexes;

      SegmentsViewType segments;

   private:
      // TODO: this should be probably moved into a detail namespace
      template< typename VectorOrView,
                std::enable_if_t< HasSetSizeMethod< VectorOrView >::value, bool > = true >
      static void set_size_if_resizable( VectorOrView& v, IndexType size )
      {
         v.setSize( size );
      }

      template< typename VectorOrView,
                std::enable_if_t< ! HasSetSizeMethod< VectorOrView >::value, bool > = true >
      static void set_size_if_resizable( VectorOrView& v, IndexType size )
      {
         TNL_ASSERT_EQ( v.getSize(), size, "view has wrong size" );
      }
};

   } // namespace Matrices
} // namespace noa::TNL

#include <noa/3rdparty/TNL/Matrices/SparseMatrixView.hpp>
