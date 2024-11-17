#include "../spmv.h"
namespace TNL {
namespace Benchmarks {
namespace SpMV {
template void dispatchSymmetric< double >( BenchmarkType&, const Containers::Vector< double, Devices::Host, int >&, const String&, const Config::ParameterContainer&, bool );
} // namespace TNL
} // namespace Benchmarks
} // namespace SpMV
