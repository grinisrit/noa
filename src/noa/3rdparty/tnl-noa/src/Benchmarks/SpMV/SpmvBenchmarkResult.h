#pragma once

#include <TNL/Benchmarks/Benchmarks.h>

namespace TNL {
namespace Benchmarks {

template< typename Real,
          typename Device,
          typename Index,
          typename ResultReal = Real,
          typename Logger = JsonLogging >
struct SpmvBenchmarkResult
: public BenchmarkResult
{
   using RealType = Real;
   using DeviceType = Device;
   using IndexType = Index;
   using HostVector = Containers::Vector< Real, Devices::Host, Index >;
   using BenchmarkVector = Containers::Vector< ResultReal, Device, Index >;

   using typename BenchmarkResult::HeaderElements;
   using typename BenchmarkResult::RowElements;
   using BenchmarkResult::stddev;
   using BenchmarkResult::bandwidth;
   using BenchmarkResult::speedup;
   using BenchmarkResult::time;


   SpmvBenchmarkResult( const HostVector& csrResult,
                        const BenchmarkVector& benchmarkResult )
   : csrResult( csrResult ), benchmarkResult( benchmarkResult )
   {}

   virtual HeaderElements getTableHeader() const override
   {
      return HeaderElements({ "time", "stddev", "stddev/time", "loops", "bandwidth", "speedup", "CSR Diff.Max", "CSR Diff.L2" });
   }

   virtual std::vector< int > getColumnWidthHints() const override
   {
      return std::vector< int >({ 14, 14, 14, 6, 14, 14, 14, 14 });
   }

   virtual RowElements getRowElements() const override
   {
      HostVector benchmarkResultCopy;
      benchmarkResultCopy = benchmarkResult;
      auto diff = csrResult - benchmarkResultCopy;
      RowElements elements;
      // write in scientific format to avoid precision loss
      elements << std::scientific << time << stddev << stddev/time << loops << bandwidth;
      if( speedup != 0.0 )
         elements << speedup;
      else
         elements << "N/A";
      elements << max( abs( diff ) ) << lpNorm( diff, 2.0 );
      return elements;
   }

   const HostVector& csrResult;
   const BenchmarkVector& benchmarkResult;
};

} //namespace Benchmarks
} //namespace TNL
