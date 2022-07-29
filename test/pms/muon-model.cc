#include <iostream>
#include <limits>
#include <fstream>

#include "particleworld.hh"

#include <gflags/gflags.h>
#include "gflags_required.hh"

DEFINE_double_required(elevation, "Height at which muon flux needs to be calculated");
DEFINE_double_required(kenergy_min, "Minimal muon kinetic energy");
DEFINE_double(kenergy_max, doubleNaN, "Maximal muon kinetic energy (optional, default = kenergy_min)");
DEFINE_string(dump_file, "materials.pumas", "Pre-computed PUMAS materials model");
DEFINE_string(materials_dir, "pumas-materials", "Path to PUMAS materials data directory");
DEFINE_string(mesh, "mesh.vtk", "Path to rock mesh");
DEFINE_string(track_dump, "", "Path to particle track dump");
DEFINE_bool(create_dump, false, "If possible, create a pre-computed PUMAS materials model when one is not available");

namespace noa::pms::pumas {
#include "flux.c"
}

using namespace std;
using namespace noa;

constexpr auto matNameAir = "Air";
constexpr auto matNameRock = "StandardRock";

constexpr double primary_altitude = 1e3;

int main(int argc, char* argv[]) {
	// Set up gflags
	gflags::SetUsageMessage("A re-implementation of PUMAS 'geometry' example in C++ for NOA\n"
			"See https://github.com/niess/pumas/blob/master/examples/pumas/geometry.c");
	gflags::ParseCommandLineFlags(&argc, &argv, true);

	// Max kinetic energy is optional: if not specified, it will be set to be the same as min energy
	if (isnan(FLAGS_kenergy_max)) FLAGS_kenergy_max = FLAGS_kenergy_min;

	test::pms::MuonWorld<float> world{};
	world.init(FLAGS_dump_file, FLAGS_materials_dir);

	auto& context = world.get_context();
	context->mode.direction = pms::pumas::PUMAS_MODE_BACKWARD;
	context->event = (pms::pumas::Event) ((int)context->event | pms::pumas::PUMAS_EVENT_LIMIT_ENERGY);

	const auto& model = world.get_model();
	const auto airIndex = model.get_material_index(matNameAir).value();
	const auto rockIndex = model.get_material_index(matNameRock).value();

	cout << "Material indices: " << endl;
	cout << "\t" << matNameAir << ":\t" << airIndex << endl;
	cout << "\t" << matNameRock << ":\t" << rockIndex << endl;

	auto& rockDomain = world.add_domain();
	rockDomain.loadFrom(FLAGS_mesh);

	auto& rockCellLayers = rockDomain.getLayers(3);
	rockCellLayers.template add<int>(rockIndex); // 0th layer is for material index by default

	world.environment = [&airIndex, &model] (pms::pumas::Context* context_p, pms::pumas::State* state_p, pms::pumas::Medium** medium_p, double* step_p) -> pms::pumas::Step {
		const auto& state = *state_p;
		const auto& z = state->position[2];
		const auto& uz = state->direction[2] * context_p->sgn();

		double step = 1e3;

		if (z < 0) {
			if (medium_p != nullptr) *medium_p = nullptr;
			step = -1;
		} else if (z < primary_altitude) {
			if (medium_p != nullptr)
				*medium_p = const_cast<pms::pumas::Medium*>(model.get_medium(airIndex));
			if (uz > std::numeric_limits<float>::epsilon())
				step = (primary_altitude - z) / uz;
			if (uz < -std::numeric_limits<float>::epsilon())
				step = - z / uz;
		} else {
			if (medium_p != nullptr) *medium_p = nullptr;
			step = -1;
		}

		if (step_p != nullptr) {
			*step_p = step;
			if (*step_p > 0) *step_p += std::numeric_limits<float>::epsilon();
		}

		return pms::pumas::PUMAS_STEP_RAW;
	};

	world.add_medium(
		matNameAir,
		[] (
			pms::pumas::Medium* medium,
			pms::pumas::State* state,
			pms::pumas::Locals* locals
		) -> double {
			// Assume uniform magnetic field
			locals->magnet[0] = 0.;
			locals->magnet[1] = 2e-5;
			locals->magnet[2] = -4e-5;

			// Atmosphere density
			constexpr double rho0 = 1.205;
			constexpr double h = 12e+3;
			locals->density = rho0 * exp(-(*state)->position[2] / h);

			// Local stepping distance proposed
			constexpr double eps = 5e-2;
			const double uz = fabs((*state)->direction[2]);
			return h / ((uz <= eps) ? eps : uz);
		}
	);
	world.add_medium(
		matNameRock,
		[] (
			pms::pumas::Medium* medium,
			pms::pumas::State* state,
			pms::pumas::Locals* locals
		) -> double {
			locals->density = 2.65e3;
			return 0;
		}
	);

	// Mote-Carlo
	std::ofstream particles;
	if (FLAGS_track_dump != "") particles.open(FLAGS_track_dump);
	const double cos_theta = cos((90. - FLAGS_elevation) / 180. * M_PI);
	const double sin_theta = sqrt(1. - cos_theta * cos_theta);
	const double rk = log(FLAGS_kenergy_max / FLAGS_kenergy_min);
	double w = 0., w2 = 0.;
	constexpr int n = 10000;
	for (int i = 0; i < n; ++i) {
		cout << "\rSimulating " << i;
		cout.flush();
		if (particles.is_open()) particles << "particle " << i << endl;
		// Set the muon final state
		double kf, wf;
		if (rk) {
			kf = FLAGS_kenergy_min * exp(rk * context.rnd());
			wf = kf * rk;
		} else {
			kf = FLAGS_kenergy_min;
			wf = 1;
		}

		const double cf = (context.rnd() > 0.5) ? 1 : -1;
		wf *= 2;

		auto state = context.create_state();
		state->charge = cf;
		state->energy = kf;
		state->weight = wf,
		state->direction[0] = -sin_theta;
		state->direction[1] = 0;
		state->direction[2] = -cos_theta;

		// Simulate muon trajectory with PUMAS
		const double energyThreshold = FLAGS_kenergy_max * 1e3;
		if (particles.is_open()) particles <<	state->position[0] << ", " <<
							state->position[1] << ", " <<
							state->position[2] << endl;

		while (state->energy < energyThreshold - numeric_limits<double>::epsilon()) {
			if (state->energy < 1e2 - numeric_limits<double>::epsilon()) {
				context->mode.energy_loss = pms::pumas::PUMAS_MODE_STRAGGLED;
				context->mode.scattering = pms::pumas::PUMAS_MODE_MIXED;
				context->limit.energy = 1e2;
			} else {
				context->mode.energy_loss = pms::pumas::PUMAS_MODE_MIXED;
				context->mode.scattering = pms::pumas::PUMAS_MODE_DISABLED;
				context->limit.energy = energyThreshold;
			}

			pms::pumas::Medium* medium[2];
			pms::pumas::Event event = context.do_transport(state, medium);
			if (particles.is_open()) particles <<	state->position[0] << ", " <<
								state->position[1] << ", " <<
								state->position[2] << endl;

			if ((event == pms::pumas::PUMAS_EVENT_MEDIUM) && (medium[1] == nullptr)) {
				if (state->position[2] >= primary_altitude - numeric_limits<double>::epsilon()) {
					if (state->position[2] > primary_altitude * 1.1) {
						cout << " Out of bounds by a lot!" << endl;
						cout << "cf = " << cf << "; kf = " << kf << "; wf = " << wf << endl;
					}
					const double wi = state->weight * pms::pumas::flux_gccly(-state->direction[2], state->energy, state->charge);
					w += wi;
					w2 += wi * wi;
				}
				break;
			} else if (event != pms::pumas::PUMAS_EVENT_LIMIT_ENERGY) {
				cerr << "Error: unexpected PUMAS event " << event << endl;
				return EXIT_FAILURE;
			}
		}
	}
	cout << endl;
	if (particles.is_open()) particles.close();

	// Print the calculation result
	w /= n;
	const double sigma = sqrt(((w2 / n) - w * w) / n);
	const auto unit = rk ? "" : "GeV^{-1} ";
	cout << "Flux: " << scientific << w << " \\pm " << sigma << " " << unit << "m^{-2} s^{-1} sr^{-1}" << endl;

	gflags::ShutDownCommandLineFlags();

	return EXIT_SUCCESS;
}
