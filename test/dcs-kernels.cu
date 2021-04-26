#include "dcs-kernels.hh"

#include <noa/pms/dcs.cuh>
#include <noa/utils/common.cuh>

namespace {

    __global__ void bremsstrahlung_cuda_kernel(
            Scalar *pres,
            const Scalar *pkin,
            const Scalar *pq,
            const AtomicElement <Scalar> element,
            Scalar mass,
            const int64_t n) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < n)
            pres[i] = dcs::cuda::pumas::bremsstrahlung(pkin[i], pq[i], element, mass);
    }
}


void bremsstrahlung_cuda(
        const torch::Tensor &result,
        const torch::Tensor &kinetic_energies,
        const torch::Tensor &recoil_energies,
        const AtomicElement <Scalar> &element,
        const ParticleMass &mass) {
    int64_t n = kinetic_energies.numel();

    int num_threads = utils::cuda::num_treads(n);
    int num_blocks = (n + num_threads - 1) / num_threads;

    const Scalar *pkin = kinetic_energies.data_ptr<Scalar>();
    const Scalar *pq = recoil_energies.data_ptr<Scalar>();
    Scalar *pres = result.data_ptr<Scalar>();

    bremsstrahlung_cuda_kernel<<<num_blocks, num_threads>>>(pres, pkin, pq, element, mass, n);
}

