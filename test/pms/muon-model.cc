#include <iostream>
#include <limits>

#include <noa/pms/pumas.hh>

#include <gflags/gflags.h>
#include "gflags_required.hh"

DEFINE_double_required(rock_thickness, "Thickness of the rock layer above the detector");
DEFINE_double_required(elevation, "Height at which muon flux needs to be calculated");
DEFINE_double_required(kenergy_min, "Minimal muon kinetic energy");
DEFINE_double(kenergy_max, doubleNaN, "Maximal muon kinetic energy (optional, default = kenergy_min)");
DEFINE_string(dump_file, "materials.pumas", "Pre-computed PUMAS materials model");
DEFINE_string(materials_dir, "pumas-materials", "Path to PUMAS materials data directory");
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

	// Initialize PUMAS physics from a binary dump
	auto modelOpt = pms::pumas::MuonModel::load_from_binary(FLAGS_dump_file);

	if (!modelOpt.has_value()) {
		cerr << "Failed to load muon model physics from a binary dump. Trying MDF..." << endl;
		const auto materialsDir = utils::Path{FLAGS_materials_dir};
		const auto mdfFile = materialsDir / "mdf" / "examples" / "standard.xml";
		const auto dedxDir = materialsDir / "dedx";
		modelOpt = pms::pumas::MuonModel::load_from_mdf(mdfFile, dedxDir);

		if (!modelOpt.has_value()) {
			cerr << "Failed to load muon model physics from MDF!" << endl;
			return EXIT_FAILURE;
		}

		if (FLAGS_create_dump) modelOpt.value().save_binary(FLAGS_dump_file);
	}

	auto model = &modelOpt.value();

	cout << "Material indices: " << endl;
	cout << "\tStandardRock:\t" << model->get_material_index(matNameRock).value() << endl;
	cout << "\tAir:\t" << model->get_material_index(matNameAir).value() << endl;

	const auto mediumAir = model->add_medium(
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
				locals->density = rho0 * exp(-state->position[2] / h);

				// Local stepping distance proposed
				constexpr double eps = 5e-2;
				const double uz = fabs(state->direction[2]);
				return h / ((uz <= eps) ? eps : uz);
			}
		);
	const auto mediumRock = model->add_medium(
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

	model->medium_callback = [&model] (pms::pumas::Context *context, pms::pumas::State *state, pms::pumas::Medium **medium_ptr, double *step_ptr) -> pms::pumas::Step {
		if ((medium_ptr == nullptr) && (step_ptr == nullptr))
			return pms::pumas::PUMAS_STEP_RAW;

		// Check muon position and direction
		const double z = state->position[2];
		const double uz = -state->direction[2]; // context->mode.direction = PUMAS_MODE_FORWARD in our case

		double step = 1e3;

		if (z < 0) {
			// Outside of the simulation area
			if (medium_ptr != nullptr) *medium_ptr = nullptr;
			step = -1;
		} else if (z < FLAGS_rock_thickness) {
			if (medium_ptr != nullptr) *medium_ptr = model->get_medium(matNameRock);
			if (uz > numeric_limits<double>::epsilon()) // Upgoing muon, next boundary is rock->air
				step = (FLAGS_rock_thickness - z) / uz;
			else if (uz < -numeric_limits<double>::epsilon()) // Next boundary is rock->nothing
				step = -z / uz;
		} else if (z < primary_altitude) {
			if (medium_ptr != nullptr) *medium_ptr = model->get_medium(matNameAir);
			if (uz > numeric_limits<double>::epsilon()) // Next boundary is air->nothing
				step = (primary_altitude - z) / uz;
			else if (uz < -numeric_limits<double>::epsilon()) // Next boundary is air->rock
				step = (FLAGS_rock_thickness - z) / uz;
		} else {
			// Outside of sim area
			if (medium_ptr != nullptr) *medium_ptr = nullptr;
			step = 1;
		}

		if (step_ptr != nullptr) {
			constexpr double step_epsilon = 1e-7;
			if (step > 0) step += step_epsilon;
			*step_ptr = step;
		}

		return pms::pumas::PUMAS_STEP_RAW;
	};

	// Create a new PUMAS simulation context
	auto* modelContext = model->create_context();
	modelContext->mode.direction = pms::pumas::PUMAS_MODE_BACKWARD;
	modelContext->event = (pms::pumas::Event) ((int)modelContext->event | pms::pumas::PUMAS_EVENT_LIMIT_ENERGY);

	// Mote-Carlo
	const double cos_theta = cos((90. - FLAGS_elevation) / 180. * M_PI);
	const double sin_theta = sqrt(1. - cos_theta * cos_theta);
	const double rk = log(FLAGS_kenergy_max / FLAGS_kenergy_min);
	double w = 0., w2 = 0.;
	constexpr int n = 10000;
	for (int i = 0; i < n; ++i) {
		// Set the muon final state
		double kf, wf;
		if (rk) {
			kf = FLAGS_kenergy_min * exp(rk * modelContext->random(modelContext));
			wf = kf * rk;
		} else {
			kf = FLAGS_kenergy_min;
			wf = 1;
		}

		const double cf = (modelContext->random(modelContext) > 0.5) ? 1 : -1;
		wf *= 2;

		pms::pumas::State state{};
		state.charge = cf;
		state.energy = kf;
		state.weight = wf,
		state.direction[0] = -sin_theta;
		state.direction[1] = 0;
		state.direction[2] = -cos_theta;

		const double energyThreshold = FLAGS_kenergy_max * 1e3;
		while (state.energy < energyThreshold - numeric_limits<double>::epsilon()) {
			if (state.energy < 1e2 - numeric_limits<double>::epsilon()) {
				modelContext->mode.energy_loss = pms::pumas::PUMAS_MODE_STRAGGLED;
				modelContext->mode.scattering = pms::pumas::PUMAS_MODE_MIXED;
				modelContext->limit.energy = 1e2;
			} else {
				modelContext->mode.energy_loss = pms::pumas::PUMAS_MODE_MIXED;
				modelContext->mode.scattering = pms::pumas::PUMAS_MODE_DISABLED;
				modelContext->limit.energy = energyThreshold;
			}

			pms::pumas::Medium* medium[2];
			pms::pumas::Event event = model->do_transport(&state, medium);

			if ((event == pms::pumas::PUMAS_EVENT_MEDIUM) && (medium[1] == nullptr)) {
				if (state.position[2] >= primary_altitude - numeric_limits<double>::epsilon()) {
					const double wi = state.weight * pms::pumas::flux_gccly(-state.direction[2], state.energy, state.charge);
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

	// Print the calculation result
	w /= n;
	const double sigma = (FLAGS_rock_thickness <= 0) ? 0 : sqrt(((w2 / n) - w * w) / n);
	const auto unit = rk ? "" : "GeV^{-1} ";
	cout << "Flux: " << scientific << w << " \\pm " << sigma << " " << unit << "m^{-2} s^{-1} sr^{-1}" << endl;

	gflags::ShutDownCommandLineFlags();

	return EXIT_SUCCESS;
}
