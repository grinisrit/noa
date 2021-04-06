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


class GHMCData 
{
    inline static TensorOpt theta = std::nullopt;
    inline static TensorOpt momentum = std::nullopt;
    inline static TensorOpt expected_fisher = std::nullopt;
    inline static TensorOpt expected_spectrum = std::nullopt;
    inline static TensorOpt expected_energy = std::nullopt;
    inline static TensorOpt expected_flow_theta = std::nullopt;
    inline static TensorOpt expected_flow_moment = std::nullopt;

public:
    static torch::Tensor get_theta()
    {
        return lazy_load_or_fail(theta, theta_pt);
    }

    static torch::Tensor get_momentum()
    {
        return lazy_load_or_fail(momentum, momentum_pt);
    }

    static torch::Tensor get_expected_fisher()
    {
        return lazy_load_or_fail(expected_fisher, expected_fisher_pt);
    }

    static torch::Tensor get_expected_spectrum()
    {
        return lazy_load_or_fail(expected_spectrum, expected_spectrum_pt);
    }

    static torch::Tensor get_expected_energy()
    {
        return lazy_load_or_fail(expected_energy, expected_energy_pt);
    }

    static torch::Tensor get_expected_flow_theta()
    {
        return lazy_load_or_fail(expected_flow_theta, expected_flow_theta_pt);
    }

    static torch::Tensor get_expected_flow_moment()
    {
        return lazy_load_or_fail(expected_flow_moment, expected_flow_moment_pt);
    }

    static void get_all()
    {
        get_theta();
        get_momentum();
        get_expected_fisher();
        get_expected_spectrum();
        get_expected_energy();
        get_expected_flow_theta();
        get_expected_flow_moment();
    }
};

class DCSData 
{

    inline static TensorOpt kinetic_energies = std::nullopt;
    inline static TensorOpt recoil_energies = std::nullopt;
    inline static TensorOpt pumas_brems = std::nullopt;
    inline static TensorOpt pumas_brems_del = std::nullopt;
    inline static TensorOpt pumas_brems_cel = std::nullopt;
    inline static TensorOpt pumas_pprod = std::nullopt;
    inline static TensorOpt pumas_pprod_del = std::nullopt;
    inline static TensorOpt pumas_pprod_cel = std::nullopt;
    inline static TensorOpt pumas_photo = std::nullopt;
    inline static TensorOpt pumas_photo_del = std::nullopt;
    inline static TensorOpt pumas_photo_cel = std::nullopt;
    inline static TensorOpt pumas_ion = std::nullopt;
    inline static TensorOpt pumas_ion_del = std::nullopt;
    inline static TensorOpt pumas_ion_cel = std::nullopt;
    inline static TensorOpt pumas_screening = std::nullopt;
    inline static TensorOpt pumas_invlambda = std::nullopt;
    inline static TensorOpt pumas_transport = std::nullopt;
    inline static TensorOpt pumas_mu0 = std::nullopt;
    inline static TensorOpt pumas_lb_h = std::nullopt;
    inline static TensorOpt pumas_soft_scatter = std::nullopt;

public:
    static torch::Tensor get_kinetic_energies()
    {
        return lazy_load_or_fail(kinetic_energies, kinetic_energies_pt);
    }

    static torch::Tensor get_recoil_energies()
    {
        return lazy_load_or_fail(recoil_energies, recoil_energies_pt);
    }

    static torch::Tensor get_pumas_brems()
    {
        return lazy_load_or_fail(pumas_brems, pumas_brems_pt);
    }
    static torch::Tensor get_pumas_brems_del()
    {
        return lazy_load_or_fail(pumas_brems_del, pumas_brems_del_pt);
    }
    static torch::Tensor get_pumas_brems_cel()
    {
        return lazy_load_or_fail(pumas_brems_cel, pumas_brems_cel_pt);
    }
    static torch::Tensor get_pumas_pprod()
    {
        return lazy_load_or_fail(pumas_pprod, pumas_pprod_pt);
    }
    static torch::Tensor get_pumas_pprod_del()
    {
        return lazy_load_or_fail(pumas_pprod_del, pumas_pprod_del_pt);
    }
    static torch::Tensor get_pumas_pprod_cel()
    {
        return lazy_load_or_fail(pumas_pprod_cel, pumas_pprod_cel_pt);
    }
    static torch::Tensor get_pumas_photo()
    {
        return lazy_load_or_fail(pumas_photo, pumas_photo_pt);
    }
    static torch::Tensor get_pumas_photo_del()
    {
        return lazy_load_or_fail(pumas_photo_del, pumas_photo_del_pt);
    }
    static torch::Tensor get_pumas_photo_cel()
    {
        return lazy_load_or_fail(pumas_photo_cel, pumas_photo_cel_pt);
    }
    static torch::Tensor get_pumas_ion()
    {
        return lazy_load_or_fail(pumas_ion, pumas_ion_pt);
    }
    static torch::Tensor get_pumas_ion_del()
    {
        return lazy_load_or_fail(pumas_ion_del, pumas_ion_del_pt);
    }
    static torch::Tensor get_pumas_ion_cel()
    {
        return lazy_load_or_fail(pumas_ion_cel, pumas_ion_cel_pt);
    }
    static torch::Tensor get_pumas_screening()
    {
        return lazy_load_or_fail(pumas_screening, pumas_screening_pt);
    }
    static torch::Tensor get_pumas_invlambda()
    {
        return lazy_load_or_fail(pumas_invlambda, pumas_invlambda_pt);
    }
    static torch::Tensor get_pumas_transport()
    {
        return lazy_load_or_fail(pumas_transport, pumas_transport_pt);
    }
    static torch::Tensor get_pumas_mu0()
    {
        return lazy_load_or_fail(pumas_mu0, pumas_mu0_pt);
    }
    static torch::Tensor get_pumas_lb_h()
    {
        return lazy_load_or_fail(pumas_lb_h, pumas_lb_h_pt);
    }
    static torch::Tensor get_pumas_soft_scatter()
    {
        return lazy_load_or_fail(pumas_soft_scatter, pumas_soft_scatter_pt);
    }

    
    static void get_all()
    {
        get_kinetic_energies();
        get_recoil_energies();
        get_pumas_brems();
        get_pumas_brems_del();
        get_pumas_brems_cel();
        get_pumas_pprod();
        get_pumas_pprod_del();
        get_pumas_pprod_cel();
        get_pumas_photo();
        get_pumas_photo_del();
        get_pumas_photo_cel();
        get_pumas_ion();
        get_pumas_ion_del();
        get_pumas_ion_cel();
        get_pumas_screening();
        get_pumas_invlambda();
        get_pumas_transport();
        get_pumas_mu0();
        get_pumas_lb_h();
        get_pumas_soft_scatter();
    }
};