// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <noa/3rdparty/TNL/Allocators/Default.h>
#include <noa/3rdparty/TNL/Devices/Host.h>
#include <noa/3rdparty/TNL/Matrices/DenseMatrixRowView.h>
#include <noa/3rdparty/TNL/Matrices/Matrix.h>
#include <noa/3rdparty/TNL/Matrices/DenseMatrixView.h>
#include <noa/3rdparty/TNL/Algorithms/Segments/Ellpack.h>

namespace noa::TNL {
namespace Matrices {

/**
 * \brief Implementation of dense matrix, i.e. matrix storing explicitly all of its elements including zeros.
 *
 * \tparam Real is a type of matrix elements.
 * \tparam Device is a device where the matrix is allocated.
 * \tparam Index is a type for indexing of the matrix elements.
 * \tparam Organization tells the ordering of matrix elements. It is either RowMajorOrder
 *         or ColumnMajorOrder.
 * \tparam RealAllocator is allocator for the matrix elements.
 */
template< typename Real = double,
          typename Device = Devices::Host,
          typename Index = int,
          ElementsOrganization Organization = Algorithms::Segments::DefaultElementsOrganization< Device >::getOrganization(),
          typename RealAllocator = typename Allocators::Default< Device >::template Allocator< Real > >
class DenseMatrix : public Matrix< Real, Device, Index, RealAllocator >
{
   protected:
      using BaseType = Matrix< Real, Device, Index, RealAllocator >;
      using ValuesVectorType = typename BaseType::ValuesType;
      using ValuesViewType = typename ValuesVectorType::ViewType;
      using SegmentsType = Algorithms::Segments::Ellpack< Device, Index, typename Allocators::Default< Device >::template Allocator< Index >, Organization, 1 >;
      using SegmentViewType = typename SegmentsType::SegmentViewType;

   public:

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
       * \brief Matrix elements organization getter.
       *
       * \return matrix elements organization - RowMajorOrder of ColumnMajorOrder.
       */
      static constexpr ElementsOrganization getOrganization() { return Organization; };

      /**
       * \brief This is only for compatibility with sparse matrices.
       *
       * \return \e  \e false.
       */
      static constexpr bool isSymmetric() { return false; };

      /**
       * \brief The allocator for matrix elements.
       */
      using RealAllocatorType = RealAllocator;

      /**
       * \brief Type of related matrix view.
       *
       * See \ref DenseMatrixView.
       */
      using ViewType = DenseMatrixView< Real, Device, Index, Organization >;

      /**
       * \brief Matrix view type for constant instances.
       *
       * See \ref DenseMatrixView.
       */
      using ConstViewType = typename DenseMatrixView< Real, Device, Index, Organization >::ConstViewType;

      /**
       * \brief Type for accessing matrix rows.
       */
      using RowView = DenseMatrixRowView< SegmentViewType, ValuesViewType >;

      /**
       * \brief Type of vector holding values of matrix elements.
       */
      using typename Matrix< Real, Device, Index, RealAllocator >::ValuesType;

      /**
       * \brief Type of constant vector holding values of matrix elements.
       */
      using typename Matrix< Real, Device, Index, RealAllocator >::ConstValuesType;

      /**
       * \brief Type of vector view holding values of matrix elements.
       */
      using typename Matrix< Real, Device, Index, RealAllocator >::ValuesView;

      /**
       * \brief Type of constant vector view holding values of matrix elements.
       */
      using typename Matrix< Real, Device, Index, RealAllocator >::ConstValuesView;

      /**
       * \brief Helper type for getting self type or its modifications.
       */
      template< typename _Real = Real,
                typename _Device = Device,
                typename _Index = Index,
                ElementsOrganization _Organization = Algorithms::Segments::DefaultElementsOrganization< _Device >::getOrganization(),
                typename _RealAllocator = typename Allocators::Default< _Device >::template Allocator< _Real > >
      using Self = DenseMatrix< _Real, _Device, _Index, _Organization, _RealAllocator >;

      /**
       * \brief Constructor only with values allocator.
       *
       * \param allocator is used for allocation of matrix elements values.
       */
      DenseMatrix( const RealAllocatorType& allocator = RealAllocatorType() );

      /**
       * \brief Copy constructor.
       *
       * \param matrix is the source matrix
       */
      DenseMatrix( const DenseMatrix& matrix ) = default;

      /**
       * \brief Move constructor.
       *
       * \param matrix is the source matrix
       */
      DenseMatrix( DenseMatrix&& matrix ) = default;

      /**
       * \brief Constructor with matrix dimensions.
       *
       * \param rows is number of matrix rows.
       * \param columns is number of matrix columns.
       * \param allocator is used for allocation of matrix elements values.
       */
      DenseMatrix( const IndexType rows, const IndexType columns,
                   const RealAllocatorType& allocator = RealAllocatorType() );

      /**
       * \brief Constructor with 2D initializer list.
       *
       * The number of matrix rows is set to the outer list size and the number
       * of matrix columns is set to maximum size of inner lists. Missing elements
       * are filled in with zeros.
       *
       * \param data is a initializer list of initializer lists representing
       * list of matrix rows.
       * \param allocator is used for allocation of matrix elements values.
       *
       * \par Example
       * \include Matrices/DenseMatrix/DenseMatrixExample_Constructor_init_list.cpp
       * \par Output
       * \include DenseMatrixExample_Constructor_init_list.out
       */
      template< typename Value >
      DenseMatrix( std::initializer_list< std::initializer_list< Value > > data,
                  const RealAllocatorType& allocator = RealAllocatorType() );

      /**
       * \brief Returns a modifiable view of the dense matrix.
       *
       * See \ref DenseMatrixView.
       *
       * \return dense matrix view.
       */
      ViewType getView();

      /**
       * \brief Returns a non-modifiable view of the dense matrix.
       *
       * See \ref DenseMatrixView.
       *
       * \return dense matrix view.
       */
      ConstViewType getConstView() const;

      /**
       * \brief Returns string with serialization type.
       *
       * The string has a form \e `Matrices::DenseMatrix< RealType,  [any_device], IndexType, [any_allocator], true/false >`.
       *
       * \return \e String with the serialization type.
       */
      static String getSerializationType();

      /**
       * \brief Returns string with serialization type.
       *
       * See \ref DenseMatrix::getSerializationType.
       *
       * \return \e String with the serialization type.
       */
      virtual String getSerializationTypeVirtual() const;

      /**
       * \brief Set number of rows and columns of this matrix.
       *
       * \param rows is the number of matrix rows.
       * \param columns is the number of matrix columns.
       */
      void setDimensions( const IndexType rows,
                          const IndexType columns );

      /**
       * \brief Set the number of matrix rows and columns by the given matrix.
       *
       * \tparam Matrix is matrix type. This can be any matrix having methods
       *  \ref getRows and \ref getColumns.
       *
       * \param matrix in the input matrix dimensions of which are to be adopted.
       */
      template< typename Matrix >
      void setLike( const Matrix& matrix );

      /**
       * \brief This method is only for the compatibility with the sparse matrices.
       *
       * This method does nothing. In debug mode it contains assertions checking
       * that given rowCapacities are compatible with the current matrix dimensions.
       */
      template< typename RowCapacitiesVector >
      void setRowCapacities( const RowCapacitiesVector& rowCapacities );

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
       * \brief This method recreates the dense matrix from 2D initializer list.
       *
       * The number of matrix rows is set to the outer list size and the number
       * of matrix columns is set to maximum size of inner lists. Missing elements
       * are filled in with zeros.
       *
       * \param data is a initializer list of initializer lists representing
       * list of matrix rows.
       *
       * \par Example
       * \include Matrices/DenseMatrix/DenseMatrixExample_setElements.cpp
       * \par Output
       * \include DenseMatrixExample_setElements.out
       */
      template< typename Value >
      void setElements( std::initializer_list< std::initializer_list< Value > > data );

      /**
       * \brief Computes number of non-zeros in each row.
       *
       * \param rowLengths is a vector into which the number of non-zeros in each row
       * will be stored.
       *
       * \par Example
       * \include Matrices/DenseMatrix/DenseMatrixExample_getCompressedRowLengths.cpp
       * \par Output
       * \include DenseMatrixExample_getCompressedRowLengths.out
       */
      template< typename RowLengthsVector >
      void getCompressedRowLengths( RowLengthsVector& rowLengths ) const;

      /**
       * \brief Returns number of non-zero matrix elements.
       *
       * \return number of all non-zero matrix elements.
       *
       * \par Example
       * \include Matrices/DenseMatrix/DenseMatrixExample_getElementsCount.cpp
       * \par Output
       * \include DenseMatrixExample_getElementsCount.out
       */
      IndexType getNonzeroElementsCount() const;

      /**
       * \brief Resets the matrix to zero dimensions.
       */
      void reset();

      /**
       * \brief Constant getter of simple structure for accessing given matrix row.
       *
       * \param rowIdx is matrix row index.
       *
       * \return RowView for accessing given matrix row.
       *
       * \par Example
       * \include Matrices/DenseMatrix/DenseMatrixExample_getConstRow.cpp
       * \par Output
       * \include DenseMatrixExample_getConstRow.out
       *
       * See \ref DenseMatrixRowView.
       */
      __cuda_callable__
      const RowView getRow( const IndexType& rowIdx ) const;

      /**
       * \brief Non-constant getter of simple structure for accessing given matrix row.
       *
       * \param rowIdx is matrix row index.
       *
       * \return RowView for accessing given matrix row.
       *
       * \par Example
       * \include Matrices/DenseMatrix/DenseMatrixExample_getRow.cpp
       * \par Output
       * \include DenseMatrixExample_getRow.out
       *
       * See \ref DenseMatrixRowView.
       */
      __cuda_callable__
      RowView getRow( const IndexType& rowIdx );

      /**
       * \brief Sets all matrix elements to value \e v.
       *
       * \param v is value all matrix elements will be set to.
       */
      void setValue( const RealType& v );

      /**
       * \brief Returns non-constant reference to element at row \e row and column column.
       *
       * Since this method returns reference to the element, it cannot be called across
       * different address spaces. It means that it can be called only form CPU if the matrix
       * is allocated on CPU or only from GPU kernels if the matrix is allocated on GPU.
       *
       * \param row is a row index of the element.
       * \param column is a columns index of the element.
       * \return reference to given matrix element.
       */
      __cuda_callable__
      Real& operator()( const IndexType row,
                        const IndexType column );

      /**
       * \brief Returns constant reference to element at row \e row and column column.
       *
       * Since this method returns reference to the element, it cannot be called across
       * different address spaces. It means that it can be called only form CPU if the matrix
       * is allocated on CPU or only from GPU kernels if the matrix is allocated on GPU.
       *
       * \param row is a row index of the element.
       * \param column is a columns index of the element.
       * \return reference to given matrix element.
       */
      __cuda_callable__
      const Real& operator()( const IndexType row,
                              const IndexType column ) const;

      /**
       * \brief Sets element at given \e row and \e column to given \e value.
       *
       * This method can be called from the host system (CPU) no matter
       * where the matrix is allocated. If the matrix is allocated on GPU this method
       * can be called even from device kernels. If the matrix is allocated in GPU device
       * this method is called from CPU, it transfers values of each matrix element separately and so the
       * performance is very low. For higher performance see. \ref DenseMatrix::getRow
       * or \ref DenseMatrix::forElements and \ref DenseMatrix::forAllElements.
       *
       * \param row is row index of the element.
       * \param column is columns index of the element.
       * \param value is the value the element will be set to.
       *
       * \par Example
       * \include Matrices/DenseMatrix/DenseMatrixExample_setElement.cpp
       * \par Output
       * \include DenseMatrixExample_setElement.out
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
       * performance is very low. For higher performance see. \ref DenseMatrix::getRow
       * or \ref DenseMatrix::forElements and \ref DenseMatrix::forAllElements.
       *
       * \param row is row index of the element.
       * \param column is columns index of the element.
       * \param value is the value the element will be set to.
       * \param thisElementMultiplicator is multiplicator the original matrix element
       *   value is multiplied by before addition of given \e value.
       *
       * \par Example
       * \include Matrices/DenseMatrix/DenseMatrixExample_addElement.cpp
       * \par Output
       * \include DenseMatrixExample_addElement.out
       *
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
       * performance is very low. For higher performance see. \ref DenseMatrix::getRow
       * or \ref DenseMatrix::forElements and \ref DenseMatrix::forAllElements.
       *
       * \param row is a row index of the matrix element.
       * \param column i a column index of the matrix element.
       *
       * \return value of given matrix element.
       *
       * \par Example
       * \include Matrices/DenseMatrix/DenseMatrixExample_getElement.cpp
       * \par Output
       * \include DenseMatrixExample_getElement.out
       *
       */
      __cuda_callable__
      Real getElement( const IndexType row,
                       const IndexType column ) const;

      /**
       * \brief Method for iteration over all matrix rows for constant instances.
       *
       * \tparam Function is type of lambda function that will operate on matrix elements.
       *    It is should have form like
       *
       * ```
       * auto function = [=] __cuda_callable__ ( IndexType rowIdx, IndexType columnIdx, IndexType columnIdx_, const RealType& value ) { ... };
       * ```
       *
       *  The column index repeats twice only for compatibility with sparse matrices.
       *
       * \param begin defines beginning of the range [begin,end) of rows to be processed.
       * \param end defines ending of the range [begin,end) of rows to be processed.
       * \param function is an instance of the lambda function to be called in each row.
       *
       * \par Example
       * \include Matrices/DenseMatrix/DenseMatrixExample_forRows.cpp
       * \par Output
       * \include DenseMatrixExample_forRows.out
       */
      template< typename Function >
      void forElements( IndexType begin, IndexType end, Function&& function ) const;

      /**
       * \brief Method for iteration over all matrix rows for non-constant instances.
       *
       * \tparam Function is type of lambda function that will operate on matrix elements.
       *    It is should have form like
       *
       * ```
       * auto function = [=] __cuda_callable__ ( IndexType rowIdx, IndexType columnIdx, IndexType columnIdx_, RealType& value ) { ... };
       * ```
       *
       *  The column index repeats twice only for compatibility with sparse matrices.
       *
       * \param begin defines beginning of the range [begin,end) of rows to be processed.
       * \param end defines ending of the range [begin,end) of rows to be processed.
       * \param function is an instance of the lambda function to be called in each row.
       *
       * \par Example
       * \include Matrices/DenseMatrix/DenseMatrixExample_forRows.cpp
       * \par Output
       * \include DenseMatrixExample_forRows.out
       */
      template< typename Function >
      void forElements( IndexType begin, IndexType end, Function&& function );

      /**
       * \brief This method calls \e forElements for all matrix rows (for constant instances).
       *
       * See \ref DenseMatrix::forElements.
       *
       * \tparam Function is a type of lambda function that will operate on matrix elements.
       * \param function  is an instance of the lambda function to be called in each row.
       *
       * \par Example
       * \include Matrices/DenseMatrix/DenseMatrixExample_forAllRows.cpp
       * \par Output
       * \include DenseMatrixExample_forAllRows.out
       */
      template< typename Function >
      void forAllElements( Function&& function ) const;

      /**
       * \brief This method calls \e forElements for all matrix rows.
       *
       * See \ref DenseMatrix::forAllElements.
       *
       * \tparam Function is a type of lambda function that will operate on matrix elements.
       * \param function  is an instance of the lambda function to be called in each row.
       *
       * \par Example
       * \include Matrices/DenseMatrix/DenseMatrixExample_forAllRows.cpp
       * \par Output
       * \include DenseMatrixExample_forAllRows.out
       */
      template< typename Function >
      void forAllElements( Function&& function );

      /**
       * \brief Method for parallel iteration over matrix rows from interval [ \e begin, \e end).
       *
       * In each row, given lambda function is performed. Each row is processed by at most one thread unlike the method
       * \ref DenseMatrix::forElements where more than one thread can be mapped to each row.
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
       * \e RowView represents matrix row - see \ref noa::TNL::Matrices::DenseMatrix::RowView.
       *
       * \par Example
       * \include Matrices/DenseMatrix/DenseMatrixExample_forRows.cpp
       * \par Output
       * \include DenseMatrixExample_forRows.out
       */
      template< typename Function >
      void forRows( IndexType begin, IndexType end, Function&& function );

      /**
       * \brief Method for parallel iteration over matrix rows from interval [ \e begin, \e end) for constant instances.
       *
       * In each row, given lambda function is performed. Each row is processed by at most one thread unlike the method
       * \ref DenseMatrix::forElements where more than one thread can be mapped to each row.
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
       * \e RowView represents matrix row - see \ref noa::TNL::Matrices::DenseMatrix::RowView.
       *
       * \par Example
       * \include Matrices/DenseMatrix/DenseMatrixExample_forRows.cpp
       * \par Output
       * \include DenseMatrixExample_forRows.out
       */
      template< typename Function >
      void forRows( IndexType begin, IndexType end, Function&& function ) const;

      /**
       * \brief Method for parallel iteration over all matrix rows.
       *
       * In each row, given lambda function is performed. Each row is processed by at most one thread unlike the method
       * \ref DenseMatrix::forAllElements where more than one thread can be mapped to each row.
       *
       * \tparam Function is type of the lambda function.
       *
       * \param function is an instance of the lambda function to be called for each row.
       *
       * ```
       * auto function = [] __cuda_callable__ ( RowView& row ) mutable { ... };
       * ```
       *
       * \e RowView represents matrix row - see \ref noa::TNL::Matrices::DenseMatrix::RowView.
       *
       * \par Example
       * \include Matrices/DenseMatrix/DenseMatrixExample_forRows.cpp
       * \par Output
       * \include DenseMatrixExample_forRows.out
       */
      template< typename Function >
      void forAllRows( Function&& function );

      /**
       * \brief Method for parallel iteration over all matrix rows for constant instances.
       *
       * In each row, given lambda function is performed. Each row is processed by at most one thread unlike the method
       * \ref DenseMatrix::forAllElements where more than one thread can be mapped to each row.
       *
       * \tparam Function is type of the lambda function.
       *
       * \param function is an instance of the lambda function to be called for each row.
       *
       * ```
       * auto function = [] __cuda_callable__ ( RowView& row ) { ... };
       * ```
       *
       * \e RowView represents matrix row - see \ref noa::TNL::Matrices::DenseMatrix::RowView.
       *
       * \par Example
       * \include Matrices/DenseMatrix/DenseMatrixExample_forRows.cpp
       * \par Output
       * \include DenseMatrixExample_forRows.out
       */
      template< typename Function >
      void forAllRows( Function&& function ) const;

      /**
       * \brief Method for sequential iteration over all matrix rows for constant instances.
       *
       * \tparam Function is type of lambda function that will operate on matrix elements.
       *    It is should have form like
       *
       * ```
       * auto function = [] __cuda_callable__ ( RowView& row ) { ... };
       * ```
       *
       * \e RowView represents matrix row - see \ref noa::TNL::Matrices::DenseMatrix::RowView.
       *
       * \param begin defines beginning of the range [begin,end) of rows to be processed.
       * \param end defines ending of the range [begin,end) of rows to be processed.
       * \param function is an instance of the lambda function to be called in each row.
       */
      template< typename Function >
      void sequentialForRows( IndexType begin, IndexType end, Function&& function ) const;

      /**
       * \brief Method for sequential iteration over all matrix rows for non-constant instances.
       *
       * \tparam Function is type of lambda function that will operate on matrix elements.
       *    It is should have form like
       *
       * ```
       * auto function = [] __cuda_callable__ ( RowView& row ) { ... };
       * ```
       *
       * \e RowView represents matrix row - see \ref noa::TNL::Matrices::DenseMatrix::RowView.
       *
       * \param begin defines beginning of the range [begin,end) of rows to be processed.
       * \param end defines ending of the range [begin,end) of rows to be processed.
       * \param function is an instance of the lambda function to be called in each row.
       */
      template< typename Function >
      void sequentialForRows( IndexType begin, IndexType end, Function&& function );

      /**
       * \brief This method calls \e sequentialForRows for all matrix rows (for constant instances).
       *
       * See \ref DenseMatrix::sequentialForRows.
       *
       * \tparam Function is a type of lambda function that will operate on matrix elements.
       * \param function  is an instance of the lambda function to be called in each row.
       */
      template< typename Function >
      void sequentialForAllRows( Function&& function ) const;

      /**
       * \brief This method calls \e sequentialForRows for all matrix rows.
       *
       * See \ref DenseMatrix::sequentialForAllRows.
       *
       * \tparam Function is a type of lambda function that will operate on matrix elements.
       * \param function  is an instance of the lambda function to be called in each row.
       */
      template< typename Function >
      void sequentialForAllRows( Function&& function );

      /**
       * \brief Method for performing general reduction on matrix rows.
       *
       * \tparam Fetch is a type of lambda function for data fetch declared as
       *
       * ```
       * auto fetch = [=] __cuda_callable__ ( IndexType rowIdx, IndexType columnIdx, RealType elementValue ) -> FetchValue { ... };
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
       *  It is declared as
       *
       * ```
       * auto keep = [=] __cuda_callable__ ( const IndexType rowIdx, const double& value ) { ... };
       * ```
       *
       * \tparam FetchValue is type returned by the Fetch lambda function.
       *
       * \param begin defines beginning of the range [begin,end) of rows to be processed.
       * \param end defines ending of the range [begin,end) of rows to be processed.
       * \param fetch is an instance of lambda function for data fetch.
       * \param reduce is an instance of lambda function for reduction.
       * \param keep in an instance of lambda function for storing results.
       * \param identity is the [identity element](https://en.wikipedia.org/wiki/Identity_element)
       *                 for the reduction operation, i.e. element which does not
       *                 change the result of the reduction.
       *
       * \par Example
       * \include Matrices/DenseMatrix/DenseMatrixExample_reduceRows.cpp
       * \par Output
       * \include DenseMatrixExample_reduceRows.out
       */
      template< typename Fetch, typename Reduce, typename Keep, typename FetchValue >
      void reduceRows( IndexType begin, IndexType end, Fetch& fetch, const Reduce& reduce, Keep& keep, const FetchValue& identity );

      /**
       * \brief Method for performing general reduction on matrix rows for constant instances.
       *
       * \tparam Fetch is a type of lambda function for data fetch declared as
       *
       * ```
       * auto fetch = [=] __cuda_callable__ ( IndexType rowIdx, IndexType columnIdx, RealType elementValue ) -> FetchValue { ... };
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
       *          It is declared as
       *
       * ````
       * auto keep = [=] __cuda_callable__ ( const IndexType rowIdx, const double& value ) { ... };
       * ```
       *
       * \tparam FetchValue is type returned by the Fetch lambda function.
       *
       * \param begin defines beginning of the range [begin,end) of rows to be processed.
       * \param end defines ending of the range [begin,end) of rows to be processed.
       * \param fetch is an instance of lambda function for data fetch.
       * \param reduce is an instance of lambda function for reduction.
       * \param keep in an instance of lambda function for storing results.
       * \param identity is the [identity element](https://en.wikipedia.org/wiki/Identity_element)
       *                 for the reduction operation, i.e. element which does not
       *                 change the result of the reduction.
       *
       * \par Example
       * \include Matrices/DenseMatrix/DenseMatrixExample_reduceRows.cpp
       * \par Output
       * \include DenseMatrixExample_reduceRows.out
       */
      template< typename Fetch, typename Reduce, typename Keep, typename FetchValue >
      void reduceRows( IndexType begin, IndexType end, Fetch& fetch, const Reduce& reduce, Keep& keep, const FetchValue& identity ) const;

      /**
       * \brief Method for performing general reduction on ALL matrix rows.
       *
       * \tparam Fetch is a type of lambda function for data fetch declared as
       *
       * ```
       * auto fetch = [=] __cuda_callable__ ( IndexType rowIdx, IndexType columnIdx, RealType elementValue ) -> FetchValue { ... };
       * ```
       *
       *      The return type of this lambda can be any non void.
       * \tparam Reduce is a type of lambda function for reduction declared as
       *
       * ````
       * auto reduce = [=] __cuda_callable__ ( const FetchValue& v1, const FetchValue& v2 ) -> FetchValue { ... };
       * ```
       *
       * \tparam Keep is a type of lambda function for storing results of reduction in each row.
       *          It is declared as
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
       * \include Matrices/DenseMatrix/DenseMatrixExample_reduceAllRows.cpp
       * \par Output
       * \include DenseMatrixExample_reduceAllRows.out
       */
      template< typename Fetch, typename Reduce, typename Keep, typename FetchReal >
      void reduceAllRows( Fetch& fetch, const Reduce& reduce, Keep& keep, const FetchReal& identity );

      /**
       * \brief Method for performing general reduction on ALL matrix rows for constant instances.
       *
       * \tparam Fetch is a type of lambda function for data fetch declared as
       *
       * ```
       * auto fetch = [=] __cuda_callable__ ( IndexType rowIdx, IndexType columnIdx, RealType elementValue ) -> FetchValue { ... };
       * ```
       *
       *          The return type of this lambda can be any non void.
       * \tparam Reduce is a type of lambda function for reduction declared as
       *
       * ```
       * auto reduce = [=] __cuda_callable__ ( const FetchValue& v1, const FetchValue& v2 ) -> FetchValue { ... };
       * ```
       *
       * \tparam Keep is a type of lambda function for storing results of reduction in each row.
       *          It is declared as
       *
       *  ```
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
       * \include Matrices/DenseMatrix/DenseMatrixExample_reduceAllRows.cpp
       * \par Output
       * \include DenseMatrixExample_reduceAllRows.out
       */
      template< typename Fetch, typename Reduce, typename Keep, typename FetchReal >
      void reduceAllRows( Fetch& fetch, const Reduce& reduce, Keep& keep, const FetchReal& identity ) const;

      /**
       * \brief Computes product of matrix and vector.
       *
       * More precisely, it computes:
       *
       * ```
       * outVector = matrixMultiplicator * ( *this ) * inVector + outVectorMultiplicator * outVector
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
       *
       * Note that the ouput vector dimension must be the same as the number of matrix rows
       * no matter how we set `begin` and `end` parameters. These parameters just say that
       * some matrix rows and the output vector elements are omitted.
       */
      template< typename InVector, typename OutVector >
      void vectorProduct( const InVector& inVector,
                          OutVector& outVector,
                          const RealType& matrixMultiplicator = 1.0,
                          const RealType& outVectorMultiplicator = 0.0,
                          const IndexType begin = 0,
                          const IndexType end = 0 ) const;

      template< typename Matrix >
      void addMatrix( const Matrix& matrix,
                      const RealType& matrixMultiplicator = 1.0,
                      const RealType& thisMatrixMultiplicator = 1.0 );

      template< typename Matrix1, typename Matrix2, int tileDim = 32 >
      void getMatrixProduct( const Matrix1& matrix1,
                          const Matrix2& matrix2,
                          const RealType& matrix1Multiplicator = 1.0,
                          const RealType& matrix2Multiplicator = 1.0 );

      template< typename Matrix, int tileDim = 32 >
      void getTransposition( const Matrix& matrix,
                             const RealType& matrixMultiplicator = 1.0 );

      /**
       * \brief Assignment operator with exactly the same type of the dense matrix.
       *
       * \param matrix is the right-hand side matrix.
       * \return reference to this matrix.
       */
      DenseMatrix& operator=( const DenseMatrix& matrix );

      /**
       * \brief Assignment operator with the same organization.
       *
       * \param matrix is the right-hand side matrix.
       * \return reference to this matrix.
       */
      template< typename RHSReal, typename RHSDevice, typename RHSIndex, typename RHSRealAllocator >
      DenseMatrix& operator=( const DenseMatrix< RHSReal, RHSDevice, RHSIndex, Organization, RHSRealAllocator >& matrix );

      /**
       * \brief Assignment operator with matrix view having the same elements organization.
       *
       * \param matrix is the right-hand side matrix.
       * \return reference to this matrix.
       */
      template< typename RHSReal, typename RHSDevice, typename RHSIndex >
      DenseMatrix& operator=( const DenseMatrixView< RHSReal, RHSDevice, RHSIndex, Organization >& matrix );

      /**
       * \brief Assignment operator with other dense matrices.
       *
       * \param matrix is the right-hand side matrix.
       * \return reference to this matrix.
       */
      template< typename RHSReal, typename RHSDevice, typename RHSIndex,
                 ElementsOrganization RHSOrganization, typename RHSRealAllocator >
      DenseMatrix& operator=( const DenseMatrix< RHSReal, RHSDevice, RHSIndex, RHSOrganization, RHSRealAllocator >& matrix );

      /**
       * \brief Assignment operator with other dense matrices.
       *
       * \param matrix is the right-hand side matrix.
       * \return reference to this matrix.
       */
      template< typename RHSReal, typename RHSDevice, typename RHSIndex,
                 ElementsOrganization RHSOrganization >
      DenseMatrix& operator=( const DenseMatrixView< RHSReal, RHSDevice, RHSIndex, RHSOrganization >& matrix );

      /**
       * \brief Assignment operator with other (sparse) types of matrices.
       *
       * \param matrix is the right-hand side matrix.
       * \return reference to this matrix.
       */
      template< typename RHSMatrix >
      DenseMatrix& operator=( const RHSMatrix& matrix );

      /**
       * \brief Comparison operator with another dense matrix.
       *
       * \param matrix is the right-hand side matrix.
       * \return \e true if the RHS matrix is equal, \e false otherwise.
       */
      template< typename Real_, typename Device_, typename Index_, typename RealAllocator_ >
      bool operator==( const DenseMatrix< Real_, Device_, Index_, Organization, RealAllocator_ >& matrix ) const;

      /**
       * \brief Comparison operator with another dense matrix.
       *
       * \param matrix is the right-hand side matrix.
       * \return \e false if the RHS matrix is equal, \e true otherwise.
       */
      template< typename Real_, typename Device_, typename Index_, typename RealAllocator_ >
      bool operator!=( const DenseMatrix< Real_, Device_, Index_, Organization, RealAllocator_ >& matrix ) const;

      /**
       * \brief Comparison operator with another dense matrix view.
       *
       * \param matrix is the right-hand side matrix view.
       * \return \e true if the RHS matrix view is equal, \e false otherwise.
       */
      template< typename Real_, typename Device_, typename Index_ >
      bool operator==( const DenseMatrixView< Real_, Device_, Index_, Organization >& matrix ) const;

      /**
       * \brief Comparison operator with another dense matrix view.
       *
       * \param matrix is the right-hand side matrix view.
       * \return \e false if the RHS matrix view is equal, \e true otherwise.
       */
      template< typename Real_, typename Device_, typename Index_ >
      bool operator!=( const DenseMatrixView< Real_, Device_, Index_, Organization >& matrix ) const;

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
       * \brief Method for loading the matrix from the file with given filename.
       *
       * \param fileName is name of the file.
       */
      void load( const String& fileName );

      /**
       * \brief Method for saving the matrix to a file.
       *
       * \param fileName is name of the file.
       */
      void save( File& file ) const;

      /**
       * \brief Method for loading the matrix from a file.
       *
       * \param fileName is name of the file.
       */
      void load( File& file );

      /**
       * \brief Method for printing the matrix to output stream.
       *
       * \param str is the output stream.
       */
      void print( std::ostream& str ) const;

   protected:

      __cuda_callable__
      IndexType getElementIndex( const IndexType row,
                                 const IndexType column ) const;

      SegmentsType segments;

      ViewType view;
};

/**
 * \brief Insertion operator for dense matrix and output stream.
 *
 * \param str is the output stream.
 * \param matrix is the dense matrix.
 * \return  reference to the stream.
 */
template< typename Real,
          typename Device,
          typename Index,
          ElementsOrganization Organization,
          typename RealAllocator >
std::ostream& operator<< ( std::ostream& str, const DenseMatrix< Real, Device, Index, Organization, RealAllocator >& matrix );

/**
 * \brief Comparison operator with another dense matrix view.
 *
 * \param leftMatrix is the left-hand side matrix view.
 * \param rightMatrix is the right-hand side matrix.
 * \return \e true if the both matrices are is equal, \e false otherwise.
 */
template< typename Real, typename Device, typename Index,
          typename Real_, typename Device_, typename Index_,
          ElementsOrganization Organization, typename RealAllocator >
bool operator==( const DenseMatrixView< Real, Device, Index, Organization >& leftMatrix,
                 const DenseMatrix< Real_, Device_, Index_, Organization, RealAllocator >& rightMatrix );

/**
 * \brief Comparison operator with another dense matrix view.
 *
 * \param leftMatrix is the left-hand side matrix view.
 * \param rightMatrix is the right-hand side matrix.
 * \return \e false if the both matrices are is equal, \e true otherwise.
 */
template< typename Real, typename Device, typename Index,
          typename Real_, typename Device_, typename Index_,
          ElementsOrganization Organization, typename RealAllocator >
bool operator!=( const DenseMatrixView< Real, Device, Index, Organization >& leftMatrix,
                 const DenseMatrix< Real_, Device_, Index_, Organization, RealAllocator >& rightMatrix );


} // namespace Matrices
} // namespace noa::TNL

#include <noa/3rdparty/TNL/Matrices/DenseMatrix.hpp>
