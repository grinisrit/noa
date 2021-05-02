#include <noa/pms/model.hh>
#include <noa/pms/pumas-model.hh>
#include <noa/pms/pms.hh>
#include <noa/pms/muon.hh>

#include <iostream>
#include <chrono>
#include <gflags/gflags.h>

using namespace std::chrono;
using namespace noa::pms;


DEFINE_string(materials, "noa-pms-models", "Path to the PMS materials data");


auto main(int argc, char **argv) -> int {
    gflags::ParseCommandLineFlags(&argc, &argv, true);

    const auto materials_dir = noa::utils::Path{FLAGS_materials};

    auto mdf_path = materials_dir / "mdf" / "standard.xml";
    auto dedx_path = materials_dir / "dedx" / "muon";

    auto mdf_settings = mdf::load_settings(mdf_path);
    if (mdf_settings.has_value()) {
        mdf::print_elements(std::get<mdf::Elements>(*mdf_settings));
        mdf::print_materials(std::get<mdf::Materials>(*mdf_settings));
    } else return 1;

    auto dedx_data = mdf::load_materials_data(
            std::get<mdf::Materials>(*mdf_settings), dedx_path);
    if (dedx_data.has_value())
        print_dedx_headers(*dedx_data);
    else return 1;



    auto begin = steady_clock::now();

    auto muon_physics = load_muon_physics_from(
            mdf_path, dedx_path, dcs::default_kernels);
    if (!muon_physics.has_value())
        std::cerr << "Failed to load the physics model from:\n"
                  << mdf_path << "\n"
                  << dedx_path << std::endl;
    else {
        auto end = steady_clock::now();
        std::cout << "Loading model took " << duration_cast<microseconds>(end - begin).count() / 1E+6
                  << " seconds" << std::endl;
    }

    return !muon_physics.has_value();
}