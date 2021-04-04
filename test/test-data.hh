#pragma once

#include <noa/utils/common.hh>

using namespace noa::utils;

inline torch::Tensor lazy_load_or_fail(TensorOpt &tensor, const Path &path)
{
    if (tensor.has_value())
    {
        return tensor.value();
    }
    else
    {
        if (load_tensor(tensor, path))
        {
            return tensor.value();
        }
        else
        {
            throw std::runtime_error("Corrupted test data");
        }
    }
}

inline const auto noa_test_data = noa::utils::Path{"noa-test-data"};

inline const auto ghmc_dir = noa_test_data / "ghmc";
inline const auto theta_pt = ghmc_dir / "theta.pt";
inline const auto momentum_pt = ghmc_dir / "momentum.pt";
inline const auto expected_fisher_pt = ghmc_dir / "expected_fisher.pt";
inline const auto expected_spectrum_pt = ghmc_dir / "expected_spectrum.pt";
inline const auto expected_energy_pt = ghmc_dir / "expected_energy.pt";
inline const auto expected_flow_theta_pt = ghmc_dir / "expected_flow_theta.pt";
inline const auto expected_flow_moment_pt = ghmc_dir / "expected_flow_moment.pt";

inline const auto pms_dir = noa_test_data / "pms";
inline const auto kinetic_energies_pt = pms_dir / "kinetic_energies.pt";
inline const auto recoil_energies_pt = pms_dir / "recoil_energies.pt";
inline const auto pumas_brems_pt = pms_dir / "pumas_brems.pt";
inline const auto pumas_brems_del_pt = pms_dir / "pumas_brems_del.pt";
inline const auto pumas_brems_cel_pt = pms_dir / "pumas_brems_cel.pt";
inline const auto pumas_pprod_pt = pms_dir / "pumas_pprod.pt";
inline const auto pumas_pprod_del_pt = pms_dir / "pumas_pprod_del.pt";
inline const auto pumas_pprod_cel_pt = pms_dir / "pumas_pprod_cel.pt";
inline const auto pumas_photo_pt = pms_dir / "pumas_photo.pt";
inline const auto pumas_photo_del_pt = pms_dir / "pumas_photo_del.pt";
inline const auto pumas_photo_cel_pt = pms_dir / "pumas_photo_cel.pt";
inline const auto pumas_ion_pt = pms_dir / "pumas_ion.pt";
inline const auto pumas_ion_del_pt = pms_dir / "pumas_ion_del.pt";
inline const auto pumas_ion_cel_pt = pms_dir / "pumas_ion_cel.pt";
inline const auto pumas_screening_pt = pms_dir / "pumas_screening.pt";
inline const auto pumas_invlambda_pt = pms_dir / "pumas_invlambda.pt";
inline const auto pumas_transport_pt = pms_dir / "pumas_transport.pt";
inline const auto pumas_mu0_pt = pms_dir / "pumas_mu0.pt";
inline const auto pumas_lb_h_pt = pms_dir / "pumas_lb_h.pt";
inline const auto pumas_soft_scatter_pt = pms_dir / "pumas_soft_scatter.pt";