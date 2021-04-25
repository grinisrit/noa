#pragma once

#include <noa/pms/constants.hh>
#include <torch/types.h>

using namespace noa::pms;


template<typename Dtype>
void bremsstrahlung_cuda(
        const torch::Tensor &result,
        const torch::Tensor &kinetic_energies,
        const torch::Tensor &recoil_energies,
        const AtomicElement<Dtype> &element,
        const Dtype &mass);
