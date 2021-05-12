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

    auto begin = steady_clock::now();

    auto mdf_settings = mdf::load_settings(mdf_path);
    if (!mdf_settings.has_value()) return 1;

    auto dedx_data = mdf::load_materials_data(*mdf_settings, dedx_path);
    if (!dedx_data.has_value()) return 1;

    auto muon_physics = MuonPhysics{}.set_mdf_settings(*mdf_settings);

    if (!muon_physics.load_dedx_data(*dedx_data)){
        std::cerr << "Failed to load the physics model from:\n"
                  << mdf_path << "\n"
                  << dedx_path << std::endl;
        return 1;
    } else {
        auto end = steady_clock::now();
        std::cout << "Loading model took " << duration_cast<microseconds>(end - begin).count() / 1E+6
                  << " seconds" << std::endl;
    }

    mdf::print_elements(std::get<mdf::Elements>(*mdf_settings));
    mdf::print_materials(std::get<mdf::Materials>(*mdf_settings));
    print_dedx_headers(*dedx_data);

    return 0;
}