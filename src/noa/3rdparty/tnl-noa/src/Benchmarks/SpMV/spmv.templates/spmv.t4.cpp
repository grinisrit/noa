#include "../spmv.h"
namespace TNL {
namespace Benchmarks {
namespace SpMV {
template void dispatchSpMV< float >( BenchmarkType&, const Containers::Vector< float, Devices::Host, int >&, const String&, const Config::ParameterContainer&, bool );
} // namespace TNL
} // namespace Benchmarks
} // namespace SpMV
