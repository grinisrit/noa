#include "../spmv.h"
namespace TNL {
namespace Benchmarks {
namespace SpMV {
template void dispatchSymmetric< double >( BenchmarkType&, const Containers::Vector< double, Devices::Host, int >&, const String&, bool, bool );
} // namespace TNL
} // namespace Benchmarks
} // namespace SpMV
