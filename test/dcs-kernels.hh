#pragma once

#include <noa/pms/dcs.hh>
#include <torch/types.h>

using namespace noa::pms;


template<typename Dtype>
void bremsstrahlung_cuda(
        const torch::Tensor &result,
        const torch::Tensor &kinetic_energies,
        const torch::Tensor &recoil_energies,
        const AtomicElement &element,
        const ParticleMass &mass);

template<typename Dtype>
inline void bremsstrahlung_cpu(
        const torch::Tensor &result,
        const torch::Tensor &kinetic_energies,
        const torch::Tensor &recoil_energies,
        const AtomicElement &element,
        const ParticleMass &mass) {
    int n = result.size(0);
    auto r = result.data_ptr<Dtype>();
    auto k = kinetic_energies.data_ptr<Dtype>();
    auto q = recoil_energies.data_ptr<Dtype>();
    const int Z = element.Z;
    const Dtype A = element.A;
    for (int i = 0; i < n; i++)
        r[i] = dcs::pumas::bremsstrahlung<Dtype>(k[i], q[i], Z, A, mass);
}

template<typename Dtype>
inline void bremsstrahlung_cpu_p(
        const torch::Tensor &result,
        const torch::Tensor &kinetic_energies,
        const torch::Tensor &recoil_energies,
        const AtomicElement &element,
        const ParticleMass &mass) {
    int n = result.size(0);
    auto r = result.data_ptr<Dtype>();
    auto k = kinetic_energies.data_ptr<Dtype>();
    auto q = recoil_energies.data_ptr<Dtype>();
    const int Z = element.Z;
    const Dtype A = element.A;
#pragma omp parallel for
    for (int i = 0; i < n; i++)
        r[i] = dcs::pumas::bremsstrahlung<Dtype>(k[i], q[i], Z, A, mass);
}
