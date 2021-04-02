#include <ghmc/pms/model.hh>
#include <ghmc/pms/dcs.hh>

#include <iostream>
#include <chrono>
#include <gflags/gflags.h>

using namespace std::chrono;

DEFINE_string(materials, "materials", "Path to the materials data");

auto main(int argc, char **argv) -> int
{
    gflags::ParseCommandLineFlags(&argc, &argv, true);

    const auto materials_dir = ghmc::utils::Path{FLAGS_materials};

    auto mdf = materials_dir / "mdf" / "standard.xml";
    auto dedx = materials_dir / "dedx" / "muon";

    auto begin = steady_clock::now();

    auto muon_physics = ghmc::pms::load_muon_physics_from(
        mdf, dedx, ghmc::pms::dcs::default_kernels);
    if (!muon_physics.has_value())
        std::cerr << "Failed to load the physics model from:\n"
                  << mdf << "\n"
                  << dedx << std::endl;
    else
    {
        auto end = steady_clock::now();
        std::cout << "Loading model took " << duration_cast<microseconds>(end - begin).count() / 1E+6
                  << " seconds" << std::endl;
    }

    return !muon_physics.has_value();
}