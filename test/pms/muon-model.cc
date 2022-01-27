#include <noa/utils/meshes.hh>
#include <noa/pms/pumas.hh>

#include <gflags/gflags.h>


DEFINE_string(materials, "pumas-materials", "Path to PUMAS materials data");
DEFINE_string(mesh, "mesh.vtu", "Path to tetrahedron mesh");

using namespace noa;


auto main(int argc, char **argv) -> int {

    gflags::SetUsageMessage("Functional tests for PUMAS bindings in PMS");

    gflags::ParseCommandLineFlags(&argc, &argv, true);

    auto meshOpt = utils::meshes::load_tetrahedron_mesh(FLAGS_mesh);

    const auto material_dir = utils::Path{FLAGS_materials};
    const auto mdf_file = material_dir / "mdf" / "examples" / "standard.xml";
    const auto dedx_dir = material_dir / "dedx";

    const auto modelOpt = pms::pumas::MuonModel::load_from_mdf(mdf_file, dedx_dir);

    gflags::ShutDownCommandLineFlags();

    return 0;
}