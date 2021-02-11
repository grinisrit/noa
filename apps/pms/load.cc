#include <ghmc/pms/model.hh>
#include <ghmc/pms/dcs.hh>

#include <iostream>
#include <chrono>

using namespace std::chrono;

int main()
{
    auto mdf = "materials/mdf/standard.xml";
    auto dedx = "materials/dedx/muon";

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