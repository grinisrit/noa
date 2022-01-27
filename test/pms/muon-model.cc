#include <noa/utils/meshes.hh>
#include <noa/pms/pumas.hh>

#include <gflags/gflags.h>


DEFINE_string(materials, "pumas-materials", "Path to PUMAS materials data");
DEFINE_string(binary_model, "materials.pumas", "Pre-computed PUMAS materials model");
DEFINE_string(mesh, "mesh.vtu", "Path to tetrahedron mesh");

using namespace noa;


auto main(int argc, char **argv) -> int {

    gflags::SetUsageMessage("Functional tests for PUMAS bindings in PMS");

    gflags::ParseCommandLineFlags(&argc, &argv, true);

    auto meshOpt = utils::meshes::load_tetrahedron_mesh(FLAGS_mesh);

    const auto material_dir = utils::Path{FLAGS_materials};
    const auto mdf_file = material_dir / "mdf" / "examples" / "standard.xml";
    const auto dedx_dir = material_dir / "dedx";

    auto modelOpt = pms::pumas::MuonModel::load_from_binary(FLAGS_binary_model);
    if (!modelOpt.has_value()){
        modelOpt = pms::pumas::MuonModel::load_from_mdf(mdf_file, dedx_dir);
        if (!modelOpt.has_value()) return 1;
    }

    const auto &model = modelOpt.value();
    model.save_binary(FLAGS_binary_model);

    gflags::ShutDownCommandLineFlags();

    return 0;
}