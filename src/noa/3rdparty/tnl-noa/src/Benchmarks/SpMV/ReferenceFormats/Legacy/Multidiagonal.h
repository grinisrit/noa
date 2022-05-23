#pragma once

#include <TNL/Matrices/Matrix.h>
#include <TNL/Containers/Vector.h>
#include <TNL/Benchmarks/SpMV/ReferenceFormats/Legacy/MultidiagonalRow.h>

namespace TNL {
    namespace Benchmarks {
        namespace SpMV {
            namespace ReferenceFormats {
               namespace Legacy {

template< typename Device >
class MultidiagonalDeviceDependentCode;

template< typename Real, typename Device = Devices::Host, typename Index = int >
class Multidiagonal : public Matrix< Real, Device, Index >
{
private:
   // convenient template alias for controlling the selection of copy-assignment operator
   template< typename Device2 >
   using Enabler = std::enable_if< ! std::is_same< Device2, Device >::value >;

   // friend class will be needed for templated assignment operators
   template< typename Real2, typename Device2, typename Index2 >
   friend class Multidiagonal;

public:
   typedef Real RealType;
   typedef Device DeviceType;
   typedef Index IndexType;
   using RowsCapacitiesType = typename Sparse< RealType, DeviceType, IndexType >::RowsCapacitiesType;
   using RowsCapacitiesTypeView = typename Sparse< RealType, DeviceType, IndexType >::RowsCapacitiesView;
   using ConstRowsCapacitiesTypeView typename Sparse< RealType, DeviceType, IndexType >::ConstRowCapacitiesView;
   typedef Matrix< Real, Device, Index > BaseType;
   typedef MultidiagonalRow< Real, Index > MatrixRow;

   template< typename _Real = Real,
             typename _Device = Device,
             typename _Index = Index >
   using Self = Multidiagonal< _Real, _Device, _Index >;

   static constexpr bool isSymmetric() { return false; };

   Multidiagonal();

   static std::string getSerializationType();

   std::string getSerializationTypeVirtual() const override;

   void setDimensions( const IndexType rows,
                       const IndexType columns ) override;

   void setCompressedRowLengths( ConstRowsCapacitiesTypeView rowLengths );

   void setRowCapacities( ConstRowsCapacitiesTypeView rowLengths );

   IndexType getRowLength( const IndexType row ) const;

   __cuda_callable__
   IndexType getRowLengthFast( const IndexType row ) const;

   IndexType getMaxRowLength() const;

   template< typename Vector >
   void setDiagonals( const Vector& diagonals );

   const Containers::Vector< Index, Device, Index >& getDiagonals() const;

   template< typename Real2, typename Device2, typename Index2 >
   void setLike( const Multidiagonal< Real2, Device2, Index2 >& matrix );

   IndexType getNumberOfMatrixElements() const;

   IndexType getNumberOfNonzeroMatrixElements() const;

   IndexType getMaxRowlength() const;

   void reset();

   template< typename Real2, typename Device2, typename Index2 >
   bool operator == ( const Multidiagonal< Real2, Device2, Index2 >& matrix ) const;

   template< typename Real2, typename Device2, typename Index2 >
   bool operator != ( const Multidiagonal< Real2, Device2, Index2 >& matrix ) const;

   void setValue( const RealType& v );

   __cuda_callable__
   bool setElementFast( const IndexType row,
                        const IndexType column,
                        const RealType& value );

   bool setElement( const IndexType row,
                    const IndexType column,
                    const RealType& value );

   __cuda_callable__
   bool addElementFast( const IndexType row,
                        const IndexType column,
                        const RealType& value,
                        const RealType& thisElementMultiplicator = 1.0 );

   bool addElement( const IndexType row,
                    const IndexType column,
                    const RealType& value,
                    const RealType& thisElementMultiplicator = 1.0 );


   __cuda_callable__
   bool setRowFast( const IndexType row,
                    const IndexType* columns,
                    const RealType* values,
                    const IndexType numberOfElements );

   bool setRow( const IndexType row,
                const IndexType* columns,
                const RealType* values,
                const IndexType numberOfElements );


   __cuda_callable__
   bool addRowFast( const IndexType row,
                    const IndexType* columns,
                    const RealType* values,
                    const IndexType numberOfElements,
                    const RealType& thisElementMultiplicator = 1.0 );

   bool addRow( const IndexType row,
                const IndexType* columns,
                const RealType* values,
                const IndexType numberOfElements,
                const RealType& thisElementMultiplicator = 1.0 );

   __cuda_callable__
   RealType getElementFast( const IndexType row,
                            const IndexType column ) const;

   RealType getElement( const IndexType row,
                        const IndexType column ) const;

   __cuda_callable__
   void getRowFast( const IndexType row,
                    IndexType* columns,
                    RealType* values ) const;

   /*void getRow( const IndexType row,
                IndexType* columns,
                RealType* values ) const;*/

   __cuda_callable__
   MatrixRow getRow( const IndexType rowIndex );

   __cuda_callable__
   const MatrixRow getRow( const IndexType rowIndex ) const;

   template< typename Vector >
   __cuda_callable__
   typename Vector::RealType rowVectorProduct( const IndexType row,
                                               const Vector& vector ) const;

   template< typename InVector,
             typename OutVector >
   void vectorProduct( const InVector& inVector,
                       OutVector& outVector ) const;

   template< typename Real2, typename Index2 >
   void addMatrix( const Multidiagonal< Real2, Device, Index2 >& matrix,
                   const RealType& matrixMultiplicator = 1.0,
                   const RealType& thisMatrixMultiplicator = 1.0 );

   template< typename Real2, typename Index2 >
   void getTransposition( const Multidiagonal< Real2, Device, Index2 >& matrix,
                          const RealType& matrixMultiplicator = 1.0 );

   template< typename Vector1, typename Vector2 >
   bool performSORIteration( const Vector1& b,
                             const IndexType row,
                             Vector2& x,
                             const RealType& omega = 1.0 ) const;

   // copy assignment
   Multidiagonal& operator=( const Multidiagonal& matrix );

   // cross-device copy assignment
   template< typename Real2, typename Device2, typename Index2,
             typename = typename Enabler< Device2 >::type >
   Multidiagonal& operator=( const Multidiagonal< Real2, Device2, Index2 >& matrix );

   void save( File& file ) const override;

   void load( File& file ) override;

   void save( const String& fileName ) const;

   void load( const String& fileName );

   void print( std::ostream& str ) const override;

protected:

   bool getElementIndex( const IndexType row,
                         const IndexType column,
                         IndexType& index ) const;

   __cuda_callable__
   bool getElementIndexFast( const IndexType row,
                             const IndexType column,
                             IndexType& index ) const;

   Containers::Vector< Real, Device, Index > values;

   Containers::Vector< Index, Device, Index > diagonalsShift;

   typedef MultidiagonalDeviceDependentCode< DeviceType > DeviceDependentCode;
   friend class MultidiagonalDeviceDependentCode< DeviceType >;
};


               } //namespace Legacy
            } //namespace ReferenceFormats
        } //namespace SpMV
    } //namespace Benchmarks
} // namespace TNL

#include <TNL/Benchmarks/SpMV/ReferenceFormats/Legacy/Multidiagonal_impl.h>
