// Copyright (c) 2004-2023 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <map>
#include <noa/3rdparty/tnl-noa/src/TNL/Matrices/Matrix.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Matrices/MatrixType.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Allocators/Default.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Algorithms/Segments/CSR.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Matrices/Sandbox/SparseSandboxMatrixRowView.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Matrices/Sandbox/SparseSandboxMatrixView.h>
#include <noa/3rdparty/tnl-noa/src/TNL/Matrices/DenseMatrix.h>

namespace noa::TNL {
namespace Matrices {
/**
 * \brief Namespace for sandbox matrices.
 */
namespace Sandbox {

/**
 * \brief Template of a sparse matrix that can be used for testing of new
 * sparse-matrix formats.
 *
 * \tparam Real is a type of matrix elements. If \e Real equals \e bool the
 *         matrix is treated as binary and so the matrix elements values are
 *         not stored in the memory since we need to remember only coordinates
 *         of non-zero elements( which equal one).
 * \tparam Device is a device where the matrix is allocated.
 * \tparam Index is a type for indexing of the matrix elements.
 * \tparam MatrixType specifies a symmetry of matrix. See \ref MatrixType.
 *         Symmetric matrices store only lower part of the matrix and its
 *         diagonal. The upper part is reconstructed on the fly.  GeneralMatrix
 *         with no symmetry is used by default.
 * \tparam RealAllocator is allocator for the matrix elements values.
 * \tparam IndexAllocator is allocator for the matrix elements column indexes.
 *
 * This class can be used for rapid testing and development of new formats for
 * sparse matrices. One may profit from several TNL tools compatible with
 * interface of this templated class like:
 *
 * 1. Large set of existing unit tests.
 * 3. Matrix reading from MTX files - to use \ref TNL::Matrices::MatrixReader,
 *    the following methods must be functional
 *    a. \ref SparseSandboxMatrix::setRowCapacities
 *    b. \ref SparseSandboxMatrix::setElement
 *    c. \ref SparseSandboxMatrix::operator= between different devices
 * 4. Matrix benchmarks - the following methods must be functional
 *    a. \ref SparseSandboxMatrix::vectorProduct - for SpMV benchmark
 * 5. Linear solvers
 * 6. Simple comparison of performance with other matrix formats
 *
 * In the core of this class there is:
 *
 * 1. Vector 'values` (\ref TNL::Matrices::Matrix::values) which is inheritted
 *    from \ref TNL::Matrices::Matrix. This vector is used for storing of matrix
 *    elements values.
 * 2. Vector `columnIndexes` (\ref SparseSandboxMatrix::columnIndexes). This
 *    vector is used for storing of matrix elements column indexes.
 *
 * This class contains fully functional implementation of CSR format and so the
 * user have to replace just what he needs to. Once you have successfully
 * implemented the sparse matrix format in this form, you may consider to
 * extract it into a form of segments to make it accessible even for other
 * algorithms then SpMV.
 *
 * Parts of the code, that need to be modified are marked by SANDBOX_TODO tag.
 * The whole implementation consits of the following classes:
 *
 * 1. \ref SparseSandboxMatrix - this class, it serves for matrix setup and
 *    performing of the main operations.
 * 2. \ref SparseSandboxMatrixView - view class which is necessary mainly for
 *    passing the matrix to GPU kernels. Most methods of `SparseSandboxMatrix`
 *    are common with `SparseSandboxMatrixView` and in this case they are
 *    implemented in the view class (and there is just redirection from this
 *    class). For this reason, `SparseSandboxMatrix` contains instance of the
 *    view class (\ref SparseSandboxMatrix::view) which needs to be regularly
 *    updated each time when metadata are changed. This is usually done by the
 *    means of method \ref SparseSandboxMatrix::getView.
 * 3. \ref SparseSandboxMatrixRowView - is a class for accessing particular
 *    matrix rows. It will, likely, require some changes as well.
 *
 * We suggest the following way of implementation of the new sparse matrix
 * format:
 *
 * 1. Add metadata required by your format next to
 *    \ref SparseSandboxMatrix::rowPointers but do not replace the row
 *    pointers. It will allow you to implement your new format next to the
 *    original CSR and to check/compare with the valid CSR implementation any
 *    time you get into troubles. The adventage is that all unit tests are
 *    working properly and you may just focus on modifying one method after
 *    another. The unit tests are called from
 *    `src/UnitTests/Matrices/SparseMatrixTests_SandboxMatrix.h` and
 *    `src/UnitTests/Matrices/SparseMatrixVectorProductTests_SandboxMatrix.h`
 * 2. Modify first the method \ref SparseSandboxMatrix::setRowCapacities which
 *    is responsible for the setup of the format metadata.
 * 3. Continue with modification of constructors, view class,
 *    \ref SparseSandboxMatrix::getView and \ref SparseSandboxMatrix::getConstView.
 * 4. Next you need to modify \ref SparseSandboxMatrix::setElement and
 *    \ref SparseSandboxMatrix::getElement methods and assignment operator at
 *    least for copying the matrix across different devices (i.e. from CPU to
 *    GPU). It will allow you to use \ref TNL::Matrices::MatrixReader.
 *    We recommend to have the same data layout on both CPU and GPU so that
 *    the transfer of the matrix from CPU to GPU is trivial.
 * 5. Finally proceed to \ref SparseSandboxMatrix::vectorProduct to implement
 *    SpMV operation. We recommend to implement first the CPU version which is
 *    easier to debug. Next proceed to GPU version.
 * 6. When SpMV works it is time to delete the original CSR implementation,
 *    i.e. everything around `rowPointers`.
 * 7. Optimize your implementation to the best performance and test with
 *    `tnl-benchmark-spmv` - you need to include your new matrix to
 *    `src/Benchmarks/SpMV/spmv.h` and modify this file accordingly.
 * 8. If you want, you may now generalize SpMV to
 *    \ref SparseSandboxMatrix::reduceRows method.
 * 9. If you have `reduceRows` implemented, you may use the original
 *    implementation of SpMV based just on the `reduceRows` method.
 * 10. You may implement \ref SparseSandboxMatrix::forRows and
 *     \ref SparseSandboxMatrix::forElements.
 * 11. Now you have complete implementation of new sparse matrix format. You
 *     may turn it into new type of segments (\ref TNL::Algorithms::Segments).
 *
 * During the implementation some unit tests may crash. If you do not need them
 * at the moment, you may comment them in files
 * `src/UnitTests/Matrices/SparseMatrixTests.h` and
 * `src/UnitTests/Matrices/SparseMatrixVectorProductTests.h`
 */
template< typename Real = double,
          typename Device = Devices::Host,
          typename Index = int,
          typename MatrixType = GeneralMatrix,
          typename RealAllocator = typename Allocators::Default< Device >::template Allocator< Real >,
          typename IndexAllocator = typename Allocators::Default< Device >::template Allocator< Index > >
class SparseSandboxMatrix : public Matrix< Real, Device, Index, RealAllocator >
{
   static_assert(
      ! MatrixType::isSymmetric() || ! std::is_same< Device, Devices::Cuda >::value
         || ( std::is_same< Real, float >::value || std::is_same< Real, double >::value || std::is_same< Real, int >::value
              || std::is_same< Real, long long int >::value ),
      "Given Real type is not supported by atomic operations on GPU which are necessary for symmetric operations." );

public:
   // Supporting types - they are not important for the user
   using BaseType = Matrix< Real, Device, Index, RealAllocator >;
   using ValuesVectorType = typename Matrix< Real, Device, Index, RealAllocator >::ValuesType;
   using ValuesViewType = typename ValuesVectorType::ViewType;
   using ConstValuesViewType = typename ValuesViewType::ConstViewType;
   using ColumnsIndexesVectorType =
      Containers::Vector< typename TNL::copy_const< Index >::template from< Real >::type, Device, Index, IndexAllocator >;
   using ColumnsIndexesViewType = typename ColumnsIndexesVectorType::ViewType;
   using ConstColumnsIndexesViewType = typename ColumnsIndexesViewType::ConstViewType;
   using RowsCapacitiesType = Containers::Vector< std::remove_const_t< Index >, Device, Index, IndexAllocator >;
   using RowsCapacitiesView = Containers::VectorView< std::remove_const_t< Index >, Device, Index >;
   using ConstRowsCapacitiesView = typename RowsCapacitiesView::ConstViewType;

   /**
    * \brief Test of symmetric matrix type.
    *
    * \return \e true if the matrix is stored as symmetric and \e false otherwise.
    */
   static constexpr bool
   isSymmetric()
   {
      return MatrixType::isSymmetric();
   }

   /**
    * \brief Test of binary matrix type.
    *
    * \return \e true if the matrix is stored as binary and \e false otherwise.
    */
   static constexpr bool
   isBinary()
   {
      return std::is_same< Real, bool >::value;
   }

   /**
    * \brief The type of matrix elements.
    */
   using RealType = std::remove_const_t< Real >;

   /**
    * \brief The device where the matrix is allocated.
    */
   using DeviceType = Device;

   /**
    * \brief The type used for matrix elements indexing.
    */
   using IndexType = Index;

   /**
    * \brief The allocator for matrix elements values.
    */
   using RealAllocatorType = RealAllocator;

   /**
    * \brief The allocator for matrix elements column indexes.
    */
   using IndexAllocatorType = IndexAllocator;

   /**
    * \brief Type of related matrix view.
    *
    * See \ref SparseSandboxMatrixView.
    */
   using ViewType = SparseSandboxMatrixView< Real, Device, Index, MatrixType >;

   /**
    * \brief Matrix view type for constant instances.
    *
    * See \ref SparseSandboxMatrixView.
    */
   using ConstViewType = SparseSandboxMatrixView< std::add_const_t< Real >, Device, Index, MatrixType >;

   /**
    * \brief Type for accessing matrix rows.
    */
   using RowView = SparseSandboxMatrixRowView< ValuesViewType, ColumnsIndexesViewType, isBinary() >;

   /**
    * \brief Type for accessing constant matrix rows.
    */
   using ConstRowView = SparseSandboxMatrixRowView< ConstValuesViewType, ConstColumnsIndexesViewType, isBinary() >;

   /**
    * \brief Helper type for getting self type or its modifications.
    */
   template< typename _Real = Real,
             typename _Device = Device,
             typename _Index = Index,
             typename _MatrixType = MatrixType,
             typename _RealAllocator = typename Allocators::Default< _Device >::template Allocator< _Real >,
             typename _IndexAllocator = typename Allocators::Default< _Device >::template Allocator< _Index > >
   using Self = SparseSandboxMatrix< _Real, _Device, _Index, _MatrixType, _RealAllocator, _IndexAllocator >;

   /**
    * \brief Type of container for CSR row pointers.
    *
    * SANDBOX_TODO: You may replace it with containers for metadata of your format.
    */
   using RowPointers = TNL::Containers::Vector< IndexType, DeviceType, IndexType >;

   /**
    * \brief Constructor only with values and column indexes allocators.
    *
    * \param realAllocator is used for allocation of matrix elements values.
    * \param indexAllocator is used for allocation of matrix elements column indexes.
    */
   SparseSandboxMatrix( const RealAllocatorType& realAllocator = RealAllocatorType(),
                        const IndexAllocatorType& indexAllocator = IndexAllocatorType() );

   /**
    * \brief Copy constructor.
    *
    * \param matrix is the source matrix
    */
   SparseSandboxMatrix( const SparseSandboxMatrix& matrix ) = default;

   /**
    * \brief Move constructor.
    *
    * \param matrix is the source matrix
    */
   SparseSandboxMatrix( SparseSandboxMatrix&& matrix ) noexcept = default;

   /**
    * \brief Constructor with matrix dimensions.
    *
    * \param rows is number of matrix rows.
    * \param columns is number of matrix columns.
    * \param realAllocator is used for allocation of matrix elements values.
    * \param indexAllocator is used for allocation of matrix elements column indexes.
    */
   template< typename Index_t, std::enable_if_t< std::is_integral< Index_t >::value, int > = 0 >
   SparseSandboxMatrix( Index_t rows,
                        Index_t columns,
                        const RealAllocatorType& realAllocator = RealAllocatorType(),
                        const IndexAllocatorType& indexAllocator = IndexAllocatorType() );

   /**
    * \brief Constructor with matrix rows capacities and number of columns.
    *
    * The number of matrix rows is given by the size of \e rowCapacities list.
    *
    * \tparam ListIndex is the initializer list values type.
    * \param rowCapacities is a list telling how many matrix elements must be
    *    allocated in each row.
    * \param columns is the number of matrix columns.
    * \param realAllocator is used for allocation of matrix elements values.
    * \param indexAllocator is used for allocation of matrix elements column indexes.
    *
    * \par Example
    * \include Matrices/SparseMatrix/SparseMatrixExample_Constructor_init_list_1.cpp
    * \par Output
    * \include SparseMatrixExample_Constructor_init_list_1.out
    */
   template< typename ListIndex >
   explicit SparseSandboxMatrix( const std::initializer_list< ListIndex >& rowCapacities,
                                 IndexType columns,
                                 const RealAllocatorType& realAllocator = RealAllocatorType(),
                                 const IndexAllocatorType& indexAllocator = IndexAllocatorType() );

   /**
    * \brief Constructor with matrix rows capacities given as a vector and number of columns.
    *
    * The number of matrix rows is given by the size of \e rowCapacities vector.
    *
    * \tparam RowCapacitiesVector is the row capacities vector type. Usually it is some of
    *    \ref TNL::Containers::Array, \ref TNL::Containers::ArrayView, \ref TNL::Containers::Vector or
    *    \ref TNL::Containers::VectorView.
    * \param rowCapacities is a vector telling how many matrix elements must be
    *    allocated in each row.
    * \param columns is the number of matrix columns.
    * \param realAllocator is used for allocation of matrix elements values.
    * \param indexAllocator is used for allocation of matrix elements column indexes.
    *
    * \par Example
    * \include Matrices/SparseMatrix/SparseMatrixExample_Constructor_rowCapacities_vector.cpp
    * \par Output
    * \include SparseMatrixExample_Constructor_rowCapacities_vector.out
    */
   template< typename RowCapacitiesVector, std::enable_if_t< TNL::IsArrayType< RowCapacitiesVector >::value, int > = 0 >
   explicit SparseSandboxMatrix( const RowCapacitiesVector& rowCapacities,
                                 IndexType columns,
                                 const RealAllocatorType& realAllocator = RealAllocatorType(),
                                 const IndexAllocatorType& indexAllocator = IndexAllocatorType() );

   /**
    * \brief Constructor with matrix dimensions and data in initializer list.
    *
    * The matrix elements values are given as a list \e data of triples:
    * { { row1, column1, value1 },
    *   { row2, column2, value2 },
    * ... }.
    *
    * \param rows is number of matrix rows.
    * \param columns is number of matrix columns.
    * \param data is a list of matrix elements values.
    * \param realAllocator is used for allocation of matrix elements values.
    * \param indexAllocator is used for allocation of matrix elements column indexes.
    *
    * \par Example
    * \include Matrices/SparseMatrix/SparseMatrixExample_Constructor_init_list_2.cpp
    * \par Output
    * \include SparseMatrixExample_Constructor_init_list_2.out
    */
   explicit SparseSandboxMatrix( IndexType rows,
                                 IndexType columns,
                                 const std::initializer_list< std::tuple< IndexType, IndexType, RealType > >& data,
                                 const RealAllocatorType& realAllocator = RealAllocatorType(),
                                 const IndexAllocatorType& indexAllocator = IndexAllocatorType() );

   /**
    * \brief Constructor with matrix dimensions and data in std::map.
    *
    * The matrix elements values are given as a map \e data where keys are
    * std::pair of matrix coordinates ( {row, column} ) and value is the
    * matrix element value.
    *
    * \tparam MapIndex is a type for indexing rows and columns.
    * \tparam MapValue is a type for matrix elements values in the map.
    *
    * \param rows is number of matrix rows.
    * \param columns is number of matrix columns.
    * \param map is std::map containing matrix elements.
    * \param realAllocator is used for allocation of matrix elements values.
    * \param indexAllocator is used for allocation of matrix elements column indexes.
    *
    * \par Example
    * \include Matrices/SparseMatrix/SparseMatrixExample_Constructor_std_map.cpp
    * \par Output
    * \include SparseMatrixExample_Constructor_std_map.out
    */
   template< typename MapIndex, typename MapValue >
   explicit SparseSandboxMatrix( IndexType rows,
                                 IndexType columns,
                                 const std::map< std::pair< MapIndex, MapIndex >, MapValue >& map,
                                 const RealAllocatorType& realAllocator = RealAllocatorType(),
                                 const IndexAllocatorType& indexAllocator = IndexAllocatorType() );

   /**
    * \brief Returns a modifiable view of the sparse matrix.
    *
    * See \ref SparseSandboxMatrixView.
    *
    * \return sparse matrix view.
    */
   ViewType
   getView();

   /**
    * \brief Returns a non-modifiable view of the sparse matrix.
    *
    * See \ref SparseSandboxMatrixView.
    *
    * \return sparse matrix view.
    */
   ConstViewType
   getConstView() const;

   /**
    * \brief Returns string with serialization type.
    *
    * The string has a form `Matrices::SparseSandboxMatrix< RealType,  [any_device], IndexType, General/Symmetric, Format,
    * [any_allocator] >`.
    *
    * \return \ref String with the serialization type.
    *
    * \par Example
    * \include Matrices/SparseMatrix/SparseMatrixExample_getSerializationType.cpp
    * \par Output
    * \include SparseMatrixExample_getSerializationType.out
    */
   static std::string
   getSerializationType();

   /**
    * \brief Returns string with serialization type.
    *
    * See \ref SparseSandboxMatrix::getSerializationType.
    *
    * \return \e String with the serialization type.
    *
    * \par Example
    * \include Matrices/SparseMatrix/SparseMatrixExample_getSerializationType.cpp
    * \par Output
    * \include SparseMatrixExample_getSerializationType.out
    */
   std::string
   getSerializationTypeVirtual() const override;

   /**
    * \brief Set number of rows and columns of this matrix.
    *
    * \param rows is the number of matrix rows.
    * \param columns is the number of matrix columns.
    */
   void
   setDimensions( IndexType rows, IndexType columns ) override;

   /**
    * \brief Set the number of matrix rows and columns by the given matrix.
    *
    * \tparam Matrix is matrix type. This can be any matrix having methods
    *  \ref getRows and \ref getColumns.
    *
    * \param matrix in the input matrix dimensions of which are to be adopted.
    */
   template< typename Matrix >
   void
   setLike( const Matrix& matrix );

   /**
    * \brief Allocates memory for non-zero matrix elements.
    *
    * The size of the input vector must be equal to the number of matrix rows.
    * The number of allocated matrix elements for each matrix row depends on
    * the sparse matrix format. Some formats may allocate more elements than
    * required.
    *
    * \tparam RowsCapacitiesVector is a type of vector/array used for row
    *    capacities setting.
    *
    * \param rowCapacities is a vector telling the number of required non-zero
    *    matrix elements in each row.
    *
    * \par Example
    * \include Matrices/SparseMatrix/SparseMatrixExample_setRowCapacities.cpp
    * \par Output
    * \include SparseMatrixExample_setRowCapacities.out
    */
   template< typename RowsCapacitiesVector >
   void
   setRowCapacities( const RowsCapacitiesVector& rowCapacities );

   /**
    * \brief Compute capacities of all rows.
    *
    * The row capacities are not stored explicitly and must be computed.
    *
    * \param rowCapacities is a vector where the row capacities will be stored.
    */
   template< typename Vector >
   void
   getRowCapacities( Vector& rowCapacities ) const;

   /**
    * \brief This method sets the sparse matrix elements from initializer list.
    *
    * The number of matrix rows and columns must be set already.
    * The matrix elements values are given as a list \e data of triples:
    * { { row1, column1, value1 },
    *   { row2, column2, value2 },
    * ... }.
    *
    * \param data is a initializer list of initializer lists representing
    * list of matrix rows.
    *
    * \par Example
    * \include Matrices/SparseMatrix/SparseMatrixExample_setElements.cpp
    * \par Output
    * \include SparseMatrixExample_setElements.out
    */
   void
   setElements( const std::initializer_list< std::tuple< IndexType, IndexType, RealType > >& data );

   /**
    * \brief This method sets the sparse matrix elements from std::map.
    *
    * The matrix elements values are given as a map \e data where keys are
    * std::pair of matrix coordinates ( {row, column} ) and value is the
    * matrix element value.
    *
    * \tparam MapIndex is a type for indexing rows and columns.
    * \tparam MapValue is a type for matrix elements values in the map.
    *
    * \param map is std::map containing matrix elements.
    *
    * \par Example
    * \include Matrices/SparseMatrix/SparseMatrixExample_setElements_map.cpp
    * \par Output
    * \include SparseMatrixExample_setElements_map.out
    */
   template< typename MapIndex, typename MapValue >
   void
   setElements( const std::map< std::pair< MapIndex, MapIndex >, MapValue >& map );

   /**
    * \brief Computes number of non-zeros in each row.
    *
    * \param rowLengths is a vector into which the number of non-zeros in each row
    * will be stored.
    *
    * \par Example
    * \include Matrices/SparseMatrix/SparseMatrixExample_getCompressedRowLengths.cpp
    * \par Output
    * \include SparseMatrixExample_getCompressedRowLengths.out
    */
   template< typename Vector >
   void
   getCompressedRowLengths( Vector& rowLengths ) const;

   /**
    * \brief Returns capacity of given matrix row.
    *
    * \param row index of matrix row.
    * \return number of matrix elements allocated for the row.
    */
   __cuda_callable__
   IndexType
   getRowCapacity( IndexType row ) const;

   /**
    * \brief Returns number of non-zero matrix elements.
    *
    * This method really counts the non-zero matrix elements and so
    * it returns zero for matrix having all allocated elements set to zero.
    *
    * \return number of non-zero matrix elements.
    */
   IndexType
   getNonzeroElementsCount() const override;

   /**
    * \brief Resets the matrix to zero dimensions.
    */
   void
   reset();

   /**
    * \brief Constant getter of simple structure for accessing given matrix row.
    *
    * \param rowIdx is matrix row index.
    *
    * \return RowView for accessing given matrix row.
    *
    * \par Example
    * \include Matrices/SparseMatrix/SparseMatrixExample_getConstRow.cpp
    * \par Output
    * \include SparseMatrixExample_getConstRow.out
    *
    * See \ref SparseMatrixRowView.
    */
   __cuda_callable__
   ConstRowView
   getRow( const IndexType& rowIdx ) const;

   /**
    * \brief Non-constant getter of simple structure for accessing given matrix row.
    *
    * \param rowIdx is matrix row index.
    *
    * \return RowView for accessing given matrix row.
    *
    * \par Example
    * \include Matrices/SparseMatrix/SparseMatrixExample_getRow.cpp
    * \par Output
    * \include SparseMatrixExample_getRow.out
    *
    * See \ref SparseMatrixRowView.
    */
   __cuda_callable__
   RowView
   getRow( const IndexType& rowIdx );

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
    * \include Matrices/SparseMatrix/SparseMatrixExample_setElement.cpp
    * \par Output
    * \include SparseMatrixExample_setElement.out
    */
   __cuda_callable__
   void
   setElement( IndexType row, IndexType column, const RealType& value );

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
    * \include Matrices/SparseMatrix/SparseMatrixExample_addElement.cpp
    * \par Output
    * \include SparseMatrixExample_addElement.out
    *
    */
   __cuda_callable__
   void
   addElement( IndexType row, IndexType column, const RealType& value, const RealType& thisElementMultiplicator );

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
    * \include Matrices/SparseMatrix/SparseMatrixExample_getElement.cpp
    * \par Output
    * \include SparseMatrixExample_getElement.out
    *
    */
   __cuda_callable__
   RealType
   getElement( IndexType row, IndexType column ) const;

   /**
    * \brief Method for performing general reduction on matrix rows.
    *
    * \tparam Fetch is a type of lambda function for data fetch declared as
    *          `fetch( IndexType rowIdx, IndexType& columnIdx, RealType& elementValue ) -> FetchValue`.
    *          The return type of this lambda can be any non void.
    * \tparam Reduce is a type of lambda function for reduction declared as
    *          `reduce( const FetchValue& v1, const FetchValue& v2 ) -> FetchValue`.
    * \tparam Keep is a type of lambda function for storing results of reduction in each row.
    *          It is declared as `keep( const IndexType rowIdx, const double& value )`.
    * \tparam FetchValue is type returned by the Fetch lambda function.
    *
    * \param begin defines beginning of the range [begin,end) of rows to be processed.
    * \param end defines ending of the range [begin,end) of rows to be processed.
    * \param fetch is an instance of lambda function for data fetch.
    * \param reduce is an instance of lambda function for reduction.
    * \param keep in an instance of lambda function for storing results.
    * \param zero is zero of given reduction operation also known as idempotent element.
    *
    * \par Example
    * \include Matrices/SparseMatrix/SparseMatrixExample_reduceRows.cpp
    * \par Output
    * \include SparseMatrixExample_reduceRows.out
    */
   template< typename Fetch, typename Reduce, typename Keep, typename FetchReal >
   void
   reduceRows( IndexType begin, IndexType end, Fetch& fetch, const Reduce& reduce, Keep& keep, const FetchReal& zero );

   /**
    * \brief Method for performing general reduction on matrix rows for constant instances.
    *
    * \tparam Fetch is a type of lambda function for data fetch declared as
    *          `fetch( IndexType rowIdx, IndexType& columnIdx, RealType& elementValue ) -> FetchValue`.
    *          The return type of this lambda can be any non void.
    * \tparam Reduce is a type of lambda function for reduction declared as
    *          `reduce( const FetchValue& v1, const FetchValue& v2 ) -> FetchValue`.
    * \tparam Keep is a type of lambda function for storing results of reduction in each row.
    *          It is declared as `keep( const IndexType rowIdx, const double& value )`.
    * \tparam FetchValue is type returned by the Fetch lambda function.
    *
    * \param begin defines beginning of the range [begin,end) of rows to be processed.
    * \param end defines ending of the range [begin,end) of rows to be processed.
    * \param fetch is an instance of lambda function for data fetch.
    * \param reduce is an instance of lambda function for reduction.
    * \param keep in an instance of lambda function for storing results.
    * \param zero is zero of given reduction operation also known as idempotent element.
    *
    * \par Example
    * \include Matrices/SparseMatrix/SparseMatrixExample_reduceRows.cpp
    * \par Output
    * \include SparseMatrixExample_reduceRows.out
    */
   template< typename Fetch, typename Reduce, typename Keep, typename FetchReal >
   void
   reduceRows( IndexType begin, IndexType end, Fetch& fetch, const Reduce& reduce, Keep& keep, const FetchReal& zero ) const;

   /**
    * \brief Method for performing general reduction on all matrix rows.
    *
    * \tparam Fetch is a type of lambda function for data fetch declared as
    *          `fetch( IndexType rowIdx, IndexType& columnIdx, RealType& elementValue ) -> FetchValue`.
    *          The return type of this lambda can be any non void.
    * \tparam Reduce is a type of lambda function for reduction declared as
    *          `reduce( const FetchValue& v1, const FetchValue& v2 ) -> FetchValue`.
    * \tparam Keep is a type of lambda function for storing results of reduction in each row.
    *          It is declared as `keep( const IndexType rowIdx, const double& value )`.
    * \tparam FetchValue is type returned by the Fetch lambda function.
    *
    * \param fetch is an instance of lambda function for data fetch.
    * \param reduce is an instance of lambda function for reduction.
    * \param keep in an instance of lambda function for storing results.
    * \param zero is zero of given reduction operation also known as idempotent element.
    *
    * \par Example
    * \include Matrices/SparseMatrix/SparseMatrixExample_reduceAllRows.cpp
    * \par Output
    * \include SparseMatrixExample_reduceAllRows.out
    */
   template< typename Fetch, typename Reduce, typename Keep, typename FetchReal >
   void
   reduceAllRows( Fetch&& fetch, const Reduce&& reduce, Keep&& keep, const FetchReal& zero );

   /**
    * \brief Method for performing general reduction on all matrix rows for constant instances.
    *
    * \tparam Fetch is a type of lambda function for data fetch declared as
    *          `fetch( IndexType rowIdx, IndexType& columnIdx, RealType& elementValue ) -> FetchValue`.
    *          The return type of this lambda can be any non void.
    * \tparam Reduce is a type of lambda function for reduction declared as
    *          `reduce( const FetchValue& v1, const FetchValue& v2 ) -> FetchValue`.
    * \tparam Keep is a type of lambda function for storing results of reduction in each row.
    *          It is declared as `keep( const IndexType rowIdx, const double& value )`.
    * \tparam FetchValue is type returned by the Fetch lambda function.
    *
    * \param fetch is an instance of lambda function for data fetch.
    * \param reduce is an instance of lambda function for reduction.
    * \param keep in an instance of lambda function for storing results.
    * \param zero is zero of given reduction operation also known as idempotent element.
    *
    * \par Example
    * \include Matrices/SparseMatrix/SparseMatrixExample_reduceAllRows.cpp
    * \par Output
    * \include SparseMatrixExample_reduceAllRows.out
    */
   template< typename Fetch, typename Reduce, typename Keep, typename FetchReal >
   void
   reduceAllRows( Fetch& fetch, const Reduce& reduce, Keep& keep, const FetchReal& zero ) const;

   /**
    * \brief Method for parallel iteration over matrix elements of given rows for constant instances.
    *
    * \tparam Function is type of lambda function that will operate on matrix elements.
    *
    * \param begin defines beginning of the range [ \e begin,\e end ) of rows to be processed.
    * \param end defines ending of the range [ \e begin, \e end ) of rows to be processed.
    * \param function is an instance of the lambda function to be called for element of given rows.
    *
    * The lambda function `function` should be declared like follows:
    *
    * ```
    * auto function = [] __cuda_callable__ ( IndexType rowIdx, IndexType localIdx, IndexType columnIdx, const RealType& value )
    * { ... };
    * ```
    *
    *  The \e localIdx parameter is a rank of the non-zero element in given row.
    *
    * \par Example
    * \include Matrices/SparseMatrix/SparseMatrixExample_forElements.cpp
    * \par Output
    * \include SparseMatrixExample_forElements.out
    */
   template< typename Function >
   void
   forElements( IndexType begin, IndexType end, Function&& function ) const;

   /**
    * \brief Method for parallel iteration over all matrix elements of given rows for non-constant instances.
    *
    * \tparam Function is type of lambda function that will operate on matrix elements.
    *
    * \param begin defines beginning of the range [ \e begin,\e end ) of rows to be processed.
    * \param end defines ending of the range [ \e begin, \e end ) of rows to be processed.
    * \param function is an instance of the lambda function to be called for each element of given rows.
    *
    * The lambda function `function` should be declared like follows:
    *
    * ```
    * auto function = [] __cuda_callable__ ( IndexType rowIdx, IndexType localIdx, IndexType columnIdx, const RealType& value )
    * mutable { ... }
    * ```
    *
    *  The \e localIdx parameter is a rank of the non-zero element in given row.
    *
    * \par Example
    * \include Matrices/SparseMatrix/SparseMatrixExample_forElements.cpp
    * \par Output
    * \include SparseMatrixExample_forElements.out
    */
   template< typename Function >
   void
   forElements( IndexType begin, IndexType end, Function&& function );

   /**
    * \brief Method for parallel iteration over all matrix elements for constant instances.
    *
    * See \ref SparseMatrix::forElements.
    *
    * \tparam Function is a type of lambda function that will operate on matrix elements.
    * \param function  is an instance of the lambda function to be called for each matrix element.
    *
    * \par Example
    * \include Matrices/SparseMatrix/SparseMatrixExample_forElements.cpp
    * \par Output
    * \include SparseMatrixExample_forElements.out
    */
   template< typename Function >
   void
   forAllElements( Function&& function ) const;

   /**
    * \brief Method for parallel iteration over all matrix elements for non-constant instances.
    *
    * See \ref SparseMatrix::forElements.
    *
    * \tparam Function is a type of lambda function that will operate on matrix elements.
    * \param function  is an instance of the lambda function to be called for each matrix element.
    *
    * \par Example
    * \include Matrices/SparseMatrix/SparseMatrixExample_forElements.cpp
    * \par Output
    * \include SparseMatrixExample_forElements.out
    */
   template< typename Function >
   void
   forAllElements( Function&& function );

   /**
    * \brief Method for parallel iteration over matrix rows from interval [ \e begin, \e end).
    *
    * In each row, given lambda function is performed. Each row is processed by at most one thread unlike the method
    * \ref SparseMatrix::forElements where more than one thread can be mapped to each row.
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
    * \e RowView represents matrix row - see \ref TNL::Matrices::SparseMatrix::RowView.
    *
    * \par Example
    * \include Matrices/SparseMatrix/SparseMatrixExample_forRows.cpp
    * \par Output
    * \include SparseMatrixExample_forRows.out
    */
   template< typename Function >
   void
   forRows( IndexType begin, IndexType end, Function&& function );

   /**
    * \brief Method for parallel iteration over matrix rows from interval [ \e begin, \e end) for constant instances.
    *
    * In each row, given lambda function is performed. Each row is processed by at most one thread unlike the method
    * \ref SparseMatrix::forElements where more than one thread can be mapped to each row.
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
    * \e RowView represents matrix row - see \ref TNL::Matrices::SparseMatrix::RowView.
    *
    * \par Example
    * \include Matrices/SparseMatrix/SparseMatrixExample_forRows.cpp
    * \par Output
    * \include SparseMatrixExample_forRows.out
    */
   template< typename Function >
   void
   forRows( IndexType begin, IndexType end, Function&& function ) const;

   /**
    * \brief Method for parallel iteration over all matrix rows.
    *
    * In each row, given lambda function is performed. Each row is processed by at most one thread unlike the method
    * \ref SparseMatrix::forAllElements where more than one thread can be mapped to each row.
    *
    * \tparam Function is type of the lambda function.
    *
    * \param function is an instance of the lambda function to be called for each row.
    *
    * ```
    * auto function = [] __cuda_callable__ ( RowView& row ) mutable { ... };
    * ```
    *
    * \e RowView represents matrix row - see \ref TNL::Matrices::SparseMatrix::RowView.
    *
    * \par Example
    * \include Matrices/SparseMatrix/SparseMatrixExample_forRows.cpp
    * \par Output
    * \include SparseMatrixExample_forRows.out
    */
   template< typename Function >
   void
   forAllRows( Function&& function );

   /**
    * \brief Method for parallel iteration over all matrix rows for constant instances.
    *
    * In each row, given lambda function is performed. Each row is processed by at most one thread unlike the method
    * \ref SparseMatrix::forAllElements where more than one thread can be mapped to each row.
    *
    * \tparam Function is type of the lambda function.
    *
    * \param function is an instance of the lambda function to be called for each row.
    *
    * ```
    * auto function = [] __cuda_callable__ ( RowView& row ) { ... };
    * ```
    *
    * \e RowView represents matrix row - see \ref TNL::Matrices::SparseMatrix::RowView.
    *
    * \par Example
    * \include Matrices/SparseMatrix/SparseMatrixExample_forRows.cpp
    * \par Output
    * \include SparseMatrixExample_forRows.out
    */
   template< typename Function >
   void
   forAllRows( Function&& function ) const;

   /**
    * \brief Method for sequential iteration over all matrix rows for constant instances.
    *
    * \tparam Function is type of lambda function that will operate on matrix elements.
    *    It is should have form like
    *  `function( IndexType rowIdx, IndexType columnIdx, IndexType columnIdx_, const RealType& value )`.
    *  The column index repeats twice only for compatibility with sparse matrices.
    *
    * \param begin defines beginning of the range [begin,end) of rows to be processed.
    * \param end defines ending of the range [begin,end) of rows to be processed.
    * \param function is an instance of the lambda function to be called in each row.
    */
   template< typename Function >
   void
   sequentialForRows( IndexType begin, IndexType end, Function& function ) const;

   /**
    * \brief Method for sequential iteration over all matrix rows for non-constant instances.
    *
    * \tparam Function is type of lambda function that will operate on matrix elements.
    *    It is should have form like
    *  `function( IndexType rowIdx, IndexType columnIdx, IndexType columnIdx_, RealType& value )`.
    *  The column index repeats twice only for compatibility with sparse matrices.
    *
    * \param begin defines beginning of the range [begin,end) of rows to be processed.
    * \param end defines ending of the range [begin,end) of rows to be processed.
    * \param function is an instance of the lambda function to be called in each row.
    */
   template< typename Function >
   void
   sequentialForRows( IndexType begin, IndexType end, Function& function );

   /**
    * \brief This method calls \e sequentialForRows for all matrix rows (for constant instances).
    *
    * See \ref SparseMatrix::sequentialForRows.
    *
    * \tparam Function is a type of lambda function that will operate on matrix elements.
    * \param function  is an instance of the lambda function to be called in each row.
    */
   template< typename Function >
   void
   sequentialForAllRows( Function& function ) const;

   /**
    * \brief This method calls \e sequentialForRows for all matrix rows.
    *
    * See \ref SparseMatrix::sequentialForAllRows.
    *
    * \tparam Function is a type of lambda function that will operate on matrix elements.
    * \param function  is an instance of the lambda function to be called in each row.
    */
   template< typename Function >
   void
   sequentialForAllRows( Function& function );

   /**
    * \brief Computes product of matrix and vector.
    *
    * More precisely, it computes:
    *
    * `outVector = matrixMultiplicator * ( * this ) * inVector + outVectorMultiplicator * outVector`
    *
    * \tparam InVector is type of input vector. It can be
    *         \ref TNL::Containers::Vector, \ref TNL::Containers::VectorView,
    *         \ref TNL::Containers::Array, \ref TNL::Containers::ArrayView,
    *         or similar container.
    * \tparam OutVector is type of output vector. It can be
    *         \ref TNL::Containers::Vector, \ref TNL::Containers::VectorView,
    *         \ref TNL::Containers::Array, \ref TNL::Containers::ArrayView,
    *         or similar container.
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
   template< typename InVector, typename OutVector >
   void
   vectorProduct( const InVector& inVector,
                  OutVector& outVector,
                  RealType matrixMultiplicator = 1.0,
                  RealType outVectorMultiplicator = 0.0,
                  IndexType begin = 0,
                  IndexType end = 0 ) const;

   /*template< typename Real2, typename Index2 >
   void addMatrix( const SparseMatrix< Real2, Segments, Device, Index2 >& matrix,
                   const RealType& matrixMultiplicator = 1.0,
                   const RealType& thisMatrixMultiplicator = 1.0 );
   */

   template< typename Real2, typename Index2 >
   void
   getTransposition( const SparseSandboxMatrix< Real2, Device, Index2, MatrixType >& matrix,
                     const RealType& matrixMultiplicator = 1.0 );

   /**
    * \brief Assignment of exactly the same matrix type.
    *
    * \param matrix is input matrix for the assignment.
    * \return reference to this matrix.
    */
   SparseSandboxMatrix&
   operator=( const SparseSandboxMatrix& matrix );

   /**
    * \brief Assignment of exactly the same matrix type but different device.
    *
    * \param matrix is input matrix for the assignment.
    * \return reference to this matrix.
    */
   template< typename Device_ >
   SparseSandboxMatrix&
   operator=( const SparseSandboxMatrix< RealType, Device_, IndexType, MatrixType, RealAllocator, IndexAllocator >& matrix );

   /**
    * \brief Assignment of dense matrix
    *
    * \param matrix is input matrix for the assignment.
    * \return reference to this matrix.
    */
   template< typename Real_, typename Device_, typename Index_, ElementsOrganization Organization, typename RealAllocator_ >
   SparseSandboxMatrix&
   operator=( const DenseMatrix< Real_, Device_, Index_, Organization, RealAllocator_ >& matrix );

   /**
    * \brief Assignment of any matrix type other then this and dense.
    *
    * **Warning: Assignment of symmetric sparse matrix to general sparse matrix does not give correct result, currently. Only
    * the diagonal and the lower part of the matrix is assigned.**
    *
    * \param matrix is input matrix for the assignment.
    * \return reference to this matrix.
    */
   template< typename RHSMatrix >
   SparseSandboxMatrix&
   operator=( const RHSMatrix& matrix );

   /**
    * \brief Comparison operator with another arbitrary matrix type.
    *
    * \param matrix is the right-hand side matrix.
    * \return \e true if the RHS matrix is equal, \e false otherwise.
    */
   template< typename Matrix >
   bool
   operator==( const Matrix& matrix ) const;

   /**
    * \brief Comparison operator with another arbitrary matrix type.
    *
    * \param matrix is the right-hand side matrix.
    * \return \e true if the RHS matrix is equal, \e false otherwise.
    */
   template< typename Matrix >
   bool
   operator!=( const Matrix& matrix ) const;

   /**
    * \brief Method for saving the matrix to the file with given filename.
    *
    * \param fileName is name of the file.
    */
   void
   save( const String& fileName ) const;

   /**
    * \brief Method for loading the matrix from the file with given filename.
    *
    * \param fileName is name of the file.
    */
   void
   load( const String& fileName );

   /**
    * \brief Method for saving the matrix to a file.
    *
    * \param file is the output file.
    */
   void
   save( File& file ) const override;

   /**
    * \brief Method for loading the matrix from a file.
    *
    * \param file is the input file.
    */
   void
   load( File& file ) override;

   /**
    * \brief Method for printing the matrix to output stream.
    *
    * \param str is the output stream.
    */
   void
   print( std::ostream& str ) const override;

   /**
    * \brief Returns a padding index value.
    *
    * Padding index is used for column indexes of padding zeros. Padding zeros
    * are used in some sparse matrix formats for better data alignment in memory.
    *
    * \return value of the padding index.
    */
   __cuda_callable__
   IndexType
   getPaddingIndex() const;

   /**
    * \brief Getter of segments for non-constant instances.
    *
    * \e Segments are a structure for addressing the matrix elements columns and values.
    * In fact, \e Segments represent the sparse matrix format.
    *
    * \return Non-constant reference to segments.
    */
   // SegmentsType& getSegments();

   /**
    * \brief Getter of segments for constant instances.
    *
    * \e Segments are a structure for addressing the matrix elements columns and values.
    * In fact, \e Segments represent the sparse matrix format.
    *
    * \return Constant reference to segments.
    */
   // const SegmentsType& getSegments() const;

   /**
    * \brief Getter of column indexes for constant instances.
    *
    * \return Constant reference to a vector with matrix elements column indexes.
    */
   const ColumnsIndexesVectorType&
   getColumnIndexes() const;

   /**
    * \brief Getter of column indexes for nonconstant instances.
    *
    * \return Reference to a vector with matrix elements column indexes.
    */
   ColumnsIndexesVectorType&
   getColumnIndexes();

protected:
   //! \brief Vector containing the column indices of non-zero matrix elements.
   ColumnsIndexesVectorType columnIndexes;

   IndexAllocator indexAllocator;

   //! \brief Instance of the sparse matrix view.
   ViewType view;

   /**
    * \brief Container for CSR row pointers.
    *
    * SANDBOX_TODO: You may replace it with containers and metadata required by you format.
    */

   RowPointers rowPointers;
};

}  // namespace Sandbox
}  // namespace Matrices
}  // namespace noa::TNL

#include <noa/3rdparty/tnl-noa/src/TNL/Matrices/Sandbox/SparseSandboxMatrix.hpp>
