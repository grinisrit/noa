#pragma once

#include <TNL/Matrices/MatrixInfo.h>
#include "CSR.h"
#include "Ellpack.h"
#include "SlicedEllpack.h"
#include "ChunkedEllpack.h"
#include "BiEllpack.h"

namespace TNL {
namespace Matrices {

/////
// Legacy matrices
template< typename Real, typename Device, typename Index >
struct MatrixInfo< Benchmarks::SpMV::ReferenceFormats::Legacy::BiEllpack< Real, Device, Index > >
{
   static String
   getDensity()
   {
      return "sparse";
   }

   static String
   getFormat()
   {
      return "BiEllpack Legacy";
   }
};

template< typename Real, typename Device, typename Index >
struct MatrixInfo< Benchmarks::SpMV::ReferenceFormats::Legacy::
                      CSR< Real, Device, Index, Benchmarks::SpMV::ReferenceFormats::Legacy::CSRScalar > >
{
   static String
   getDensity()
   {
      return "sparse";
   }

   static String
   getFormat()
   {
      return "CSR Legacy Scalar";
   }
};

template< typename Real, typename Device, typename Index >
struct MatrixInfo< Benchmarks::SpMV::ReferenceFormats::Legacy::
                      CSR< Real, Device, Index, Benchmarks::SpMV::ReferenceFormats::Legacy::CSRVector > >
{
   static String
   getDensity()
   {
      return "sparse";
   }

   static String
   getFormat()
   {
      return "CSR Legacy Vector";
   }
};

template< typename Real, typename Device, typename Index >
struct MatrixInfo< Benchmarks::SpMV::ReferenceFormats::Legacy::
                      CSR< Real, Device, Index, Benchmarks::SpMV::ReferenceFormats::Legacy::CSRLight > >
{
   static String
   getDensity()
   {
      return "sparse";
   }

   static String
   getFormat()
   {
      return "CSR Legacy Light";
   }
};

template< typename Real, typename Device, typename Index >
struct MatrixInfo< Benchmarks::SpMV::ReferenceFormats::Legacy::
                      CSR< Real, Device, Index, Benchmarks::SpMV::ReferenceFormats::Legacy::CSRLight2 > >
{
   static String
   getDensity()
   {
      return "sparse";
   }

   static String
   getFormat()
   {
      return "CSR Legacy Light2";
   }
};

template< typename Real, typename Device, typename Index >
struct MatrixInfo< Benchmarks::SpMV::ReferenceFormats::Legacy::
                      CSR< Real, Device, Index, Benchmarks::SpMV::ReferenceFormats::Legacy::CSRLight3 > >
{
   static String
   getDensity()
   {
      return "sparse";
   }

   static String
   getFormat()
   {
      return "CSR Legacy Light3";
   }
};

template< typename Real, typename Device, typename Index >
struct MatrixInfo< Benchmarks::SpMV::ReferenceFormats::Legacy::
                      CSR< Real, Device, Index, Benchmarks::SpMV::ReferenceFormats::Legacy::CSRLight4 > >
{
   static String
   getDensity()
   {
      return "sparse";
   }

   static String
   getFormat()
   {
      return "CSR Legacy Light4";
   }
};

template< typename Real, typename Device, typename Index >
struct MatrixInfo< Benchmarks::SpMV::ReferenceFormats::Legacy::
                      CSR< Real, Device, Index, Benchmarks::SpMV::ReferenceFormats::Legacy::CSRLight5 > >
{
   static String
   getDensity()
   {
      return "sparse";
   }

   static String
   getFormat()
   {
      return "CSR Legacy Light5";
   }
};

template< typename Real, typename Device, typename Index >
struct MatrixInfo< Benchmarks::SpMV::ReferenceFormats::Legacy::
                      CSR< Real, Device, Index, Benchmarks::SpMV::ReferenceFormats::Legacy::CSRLight6 > >
{
   static String
   getDensity()
   {
      return "sparse";
   }

   static String
   getFormat()
   {
      return "CSR Legacy Light6";
   }
};

template< typename Real, typename Device, typename Index >
struct MatrixInfo< Benchmarks::SpMV::ReferenceFormats::Legacy::
                      CSR< Real, Device, Index, Benchmarks::SpMV::ReferenceFormats::Legacy::CSRAdaptive > >
{
   static String
   getDensity()
   {
      return "sparse";
   }

   static String
   getFormat()
   {
      return "CSR Legacy Adaptive";
   }
};

template< typename Real, typename Device, typename Index >
struct MatrixInfo< Benchmarks::SpMV::ReferenceFormats::Legacy::
                      CSR< Real, Device, Index, Benchmarks::SpMV::ReferenceFormats::Legacy::CSRMultiVector > >
{
   static String
   getDensity()
   {
      return "sparse";
   }

   static String
   getFormat()
   {
      return "CSR Legacy MultiVector";
   }
};

template< typename Real, typename Device, typename Index >
struct MatrixInfo< Benchmarks::SpMV::ReferenceFormats::Legacy::
                      CSR< Real, Device, Index, Benchmarks::SpMV::ReferenceFormats::Legacy::CSRLightWithoutAtomic > >
{
   static String
   getDensity()
   {
      return "sparse";
   }

   static String
   getFormat()
   {
      return "CSR Legacy LightWithoutAtomic";
   }
};

template< typename Real, typename Device, typename Index >
struct MatrixInfo< Benchmarks::SpMV::ReferenceFormats::Legacy::ChunkedEllpack< Real, Device, Index > >
{
   static String
   getDensity()
   {
      return "sparse";
   }

   static String
   getFormat()
   {
      return "ChunkedEllpack Legacy";
   }
};

template< typename Real, typename Device, typename Index >
struct MatrixInfo< Benchmarks::SpMV::ReferenceFormats::Legacy::Ellpack< Real, Device, Index > >
{
   static String
   getDensity()
   {
      return "sparse";
   }

   static String
   getFormat()
   {
      return "Ellpack Legacy";
   }
};

template< typename Real, typename Device, typename Index, int SliceSize >
struct MatrixInfo< Benchmarks::SpMV::ReferenceFormats::Legacy::SlicedEllpack< Real, Device, Index, SliceSize > >
{
   static String
   getDensity()
   {
      return "sparse";
   }

   static String
   getFormat()
   {
      return "SlicedEllpack Legacy";
   }
};

}  // namespace Matrices
}  // namespace TNL
