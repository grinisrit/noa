#include <iostream>

#include <ghmc/pms/model.hh>

int main()
{
    auto mdf = "materials/mdf/standard.xml";
    auto dedx = "materials/dedx/muon";

    auto muon_physics = ghmc::pms::load_muon_physics_from(mdf, dedx);
    if (!muon_physics.has_value())
        std::cerr << "Failed to load the physics model from:\n"
                  << mdf << "\n"
                  << dedx << std::endl;
    else
        std::cout << "Ok" << std::endl;

    return !muon_physics.has_value();
}