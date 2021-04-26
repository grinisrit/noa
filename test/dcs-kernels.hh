#pragma once

#include <noa/pms/constants.hh>
#include <torch/types.h>

using namespace noa;
using namespace noa::pms;

void bremsstrahlung_cuda(
        const torch::Tensor &result,
        const torch::Tensor &kinetic_energies,
        const torch::Tensor &recoil_energies,
        const AtomicElement<pms::Scalar> &element,
        const pms::ParticleMass &mass);

