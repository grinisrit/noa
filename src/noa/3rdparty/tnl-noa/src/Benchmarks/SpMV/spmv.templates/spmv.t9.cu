#include "../spmv.h"
namespace TNL {
namespace Benchmarks {
namespace SpMV {
template void dispatchSymmetricBinary< double >( BenchmarkType&, const Matrices::SparseMatrix< float, Devices::Host >&, const Containers::Vector< double, Devices::Host, int >&, const String&, const Config::ParameterContainer&, bool );
} // namespace TNL
} // namespace Benchmarks
} // namespace SpMV
