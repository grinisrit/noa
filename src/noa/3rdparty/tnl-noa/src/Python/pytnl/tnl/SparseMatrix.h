#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/operators.h>
namespace py = pybind11;

#include <TNL/Containers/Vector.h>
#include <TNL/TypeTraits.h>

template< typename RowView, typename Scope >
std::enable_if_t< ! std::is_const< typename RowView::RealType >::value >
export_RowView_nonconst( Scope & s )
{
   using RealType = typename RowView::RealType;
   using IndexType = typename RowView::IndexType;

   s
      .def("getColumnIndex", []( RowView& row, IndexType localIdx ) -> IndexType& {
               return row.getColumnIndex( localIdx );
         }, py::return_value_policy::reference_internal)
      .def("getValue", []( RowView& row, IndexType localIdx ) -> RealType& {
               return row.getValue( localIdx );
         }, py::return_value_policy::reference_internal)
      .def("setValue",         &RowView::setValue)
      .def("setColumnIndex",   &RowView::setColumnIndex)
      .def("setElement",       &RowView::setElement)
   ;
}

template< typename RowView, typename Scope >
std::enable_if_t< std::is_const< typename RowView::RealType >::value >
export_RowView_nonconst( Scope & s )
{}

template< typename RowView, typename Scope >
void export_RowView( Scope & s, const char* name )
{
   using RealType = typename RowView::RealType;
   using IndexType = typename RowView::IndexType;

   auto rowView = py::class_< RowView >( s, name )
      .def("getSize",          &RowView::getSize)
      .def("getRowIndex",      &RowView::getRowIndex)
      .def("getColumnIndex", []( const RowView& row, IndexType localIdx ) -> const IndexType& {
               return row.getColumnIndex( localIdx );
         }, py::return_value_policy::reference_internal)
      .def("getValue", []( const RowView& row, IndexType localIdx ) -> const RealType& {
               return row.getValue( localIdx );
         }, py::return_value_policy::reference_internal)
      .def(py::self == py::self)
//      .def(py::self_ns::str(py::self_ns::self))
   ;
   export_RowView_nonconst< RowView >( rowView );
}

template< typename Segments, typename Enable = void >
struct export_CSR
{
   template< typename Scope >
   static void e( Scope & s ) {}
};

template< typename Segments >
struct export_CSR< Segments, typename TNL::enable_if_type< decltype(Segments{}.getOffsets()) >::type >
{
   template< typename Scope >
   static void e( Scope & s )
   {
      s
         .def("getOffsets", []( const Segments& segments ) -> const typename Segments::OffsetsContainer& {
                  return segments.getOffsets();
            }, py::return_value_policy::reference_internal)
      ;
   }
};

template< typename Segments, typename Scope >
void export_Segments( Scope & s, const char* name )
{
   auto segments = py::class_< Segments >( s, name )
      .def("getSegmentsCount", &Segments::getSegmentsCount)
      .def("getSegmentSize", &Segments::getSegmentSize)
      .def("getSize", &Segments::getSize)
      .def("getStorageSize", &Segments::getStorageSize)
      .def("getGlobalIndex", &Segments::getGlobalIndex)
      // FIXME: this does not compile
//      .def(py::self == py::self)
      // TODO: forElements, forAllElements, forSegments, forAllSegments, segmentsReduction, allReduction
   ;

   export_CSR< Segments >::e( segments );
}

template< typename Matrix >
void export_Matrix( py::module & m, const char* name )
{
   using RealType = typename Matrix::RealType;
   using DeviceType = typename Matrix::DeviceType;
   using IndexType = typename Matrix::IndexType;

   using VectorType = TNL::Containers::Vector< RealType, DeviceType, IndexType >;
   using IndexVectorType = TNL::Containers::Vector< IndexType, DeviceType, IndexType >;

   auto matrix = py::class_< Matrix, TNL::Object >( m, name )
      .def(py::init<>())
      // overloads (defined in Object)
      .def_static("getSerializationType", &Matrix::getSerializationType)
      .def("getSerializationTypeVirtual", &Matrix::getSerializationTypeVirtual)
      .def("print", &Matrix::print)
      .def("__str__", []( Matrix & m ) {
               std::stringstream ss;
               ss << m;
               return ss.str();
         })

      // Matrix
      .def("setDimensions",           &Matrix::setDimensions)
      // TODO: export for more types
      .def("setLike", []( Matrix& matrix, const Matrix& other ) -> void {
               matrix.setLike( other );
         })
      .def("getAllocatedElementsCount",   &Matrix::getAllocatedElementsCount)
      .def("getNonzeroElementsCount",     &Matrix::getNonzeroElementsCount)
      .def("reset",                       &Matrix::reset)
      .def("getRows",                     &Matrix::getRows)
      .def("getColumns",                  &Matrix::getColumns)
      // TODO: export for more types
      .def(py::self == py::self)
      .def(py::self != py::self)

      // SparseMatrix
      .def("setRowCapacities",         &Matrix::template setRowCapacities< IndexVectorType >)
      .def("getRowCapacities",         &Matrix::template getRowCapacities< IndexVectorType >)
      .def("getCompressedRowLengths",  &Matrix::template getCompressedRowLengths< IndexVectorType >)
      .def("getRowCapacity",           &Matrix::getRowCapacity)
      .def("getPaddingIndex",          &Matrix::getPaddingIndex)
      .def("getRow", []( Matrix& matrix, IndexType rowIdx ) -> typename Matrix::RowView {
               return matrix.getRow( rowIdx );
         })
      .def("getRow", []( const Matrix& matrix, IndexType rowIdx ) -> typename Matrix::ConstRowView {
               return matrix.getRow( rowIdx );
         })
      .def("setElement",               &Matrix::setElement)
      .def("addElement",               &Matrix::addElement)
      .def("getElement",               &Matrix::getElement)
      // TODO: reduceRows, reduceAllRows, forElements, forAllElements, forRows, forAllRows
      // TODO: export for more types
      .def("vectorProduct",       &Matrix::template vectorProduct< VectorType, VectorType >)
      // TODO: these two don't work
      //.def("addMatrix",           &Matrix::addMatrix)
      //.def("getTransposition",    &Matrix::getTransposition)
      // TODO: export for more types
      .def("assign", []( Matrix& matrix, const Matrix& other ) -> Matrix& {
               return matrix = other;
         })

      // accessors for internal vectors
      .def("getValues",        py::overload_cast<>(&Matrix::getValues),        py::return_value_policy::reference_internal)
      .def("getColumnIndexes", py::overload_cast<>(&Matrix::getColumnIndexes), py::return_value_policy::reference_internal)
      .def("getSegments",      py::overload_cast<>(&Matrix::getSegments),      py::return_value_policy::reference_internal)
   ;

   export_Segments< typename Matrix::SegmentsType >( matrix, "Segments" );
}
