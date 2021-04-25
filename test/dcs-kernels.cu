#include "dcs-kernels.hh"

#include <noa/pms/dcs.cuh>


namespace {
    template<typename Dtype>
    __global__ void bremsstrahlung_cuda_kernel(
            Dtype *res,
            Dtype *kinetic_energies,
            Dtype *recoil_energies,
            const int n, const AtomicElement<Dtype> element,  Dtype mass) {
        int index = blockIdx.x * blockDim.x + threadIdx.x;
        if (index < n)
            res[index] = dcs::pumas::cuda::bremsstrahlung(
                    kinetic_energies[index], recoil_energies[index], element, mass);
    }
}

template<>
void bremsstrahlung_cuda<double>(
        const torch::Tensor &result,
        const torch::Tensor &kinetic_energies,
        const torch::Tensor &recoil_energies,
        const AtomicElement<double> &element,
        const double &mass) {
    int nkin = kinetic_energies.size(0);
    int min_block = 32;
    int max_block = 1024;
    int thread_blocks = (nkin + min_block - 1) / min_block;
    int num_threads = std::min(min_block * thread_blocks, max_block);
    int num_blocks = (nkin + num_threads - 1) / num_threads;
    bremsstrahlung_cuda_kernel<double><<<num_blocks, num_threads>>>(
            result.data_ptr<double>(),
            kinetic_energies.data_ptr<double>(),
            recoil_energies.data_ptr<double>(), nkin,
            element, mass);
}

