#include "../spmv.h"
namespace TNL {
namespace Benchmarks {
namespace SpMV {
template void dispatchSymmetricBinary< float >( BenchmarkType&, const Matrices::SparseMatrix< float, Devices::Host >&, const Containers::Vector< float, Devices::Host, int >&, const String&, bool, bool );
} // namespace TNL
} // namespace Benchmarks
} // namespace SpMV
