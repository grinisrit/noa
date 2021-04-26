#include "dcs-kernels.hh"

#include <noa/pms/dcs.cuh>
#include <noa/utils/common.cuh>


void bremsstrahlung_cuda(
        const torch::Tensor &result,
        const torch::Tensor &kinetic_energies,
        const torch::Tensor &recoil_energies,
        const AtomicElement <Scalar> &element,
        const ParticleMass &mass) {

    const Scalar *pq = recoil_energies.data_ptr<Scalar>();
    auto brems = [pq, element, mass] __device__(const int i, const Scalar &k) {
        return dcs::cuda::pumas::bremsstrahlung(k, pq[i], element, mass);
    };
    utils::cuda::vmapi<Scalar>(kinetic_energies, brems, result);
}
