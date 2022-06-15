/*
 * Copyright (C) 2017 Université Clermont Auvergne, CNRS/IN2P3, LPC
 * Author: Valentin NIESS (niess@in2p3.fr)
 *
 * This software is a C library whose purpose is to transport high energy
 * muons or taus in various media.
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>
 */

#ifndef pumas_h
#define pumas_h
#ifdef __cplusplus
extern "C" {
#endif

#ifndef PUMAS_API
#define PUMAS_API
#endif

/* For C standard streams. */
#ifndef FILE
#include <stdio.h>
#endif

/* PUMAS library version. */
#define PUMAS_VERSION_MAJOR 1
#define PUMAS_VERSION_MINOR 2
#define PUMAS_VERSION_PATCH 1

/**
 * Projectiles supported by PUMAS.
 */
enum pumas_particle {
        /** The muon or anti-muon lepton. */
        PUMAS_PARTICLE_MUON = 0,
        /** The tau or anti-tau lepton. */
        PUMAS_PARTICLE_TAU
};

/**
 * Physics properties tabulated by PUMAS.
 */
enum pumas_property {
        /**
         * The restricted cross-section for inelastic and radiative
         * processes, in m^(2)/kg.
         */
        PUMAS_PROPERTY_CROSS_SECTION = 0,
        /**
         * Cutoff angle for hard elastic events in the center of mass frame,
         * in rad.
         */
        PUMAS_PROPERTY_ELASTIC_CUTOFF_ANGLE,
        /**
         * The mean free path for hard elastic (Coulomb) collisions,
         * in kg/m^(2).
         */
        PUMAS_PROPERTY_ELASTIC_PATH,
        /** The material stopping power, in GeV/(kg/m^(2)). */
        PUMAS_PROPERTY_STOPPING_POWER,
        /** The particle grammage range, in kg/m^(2). */
        PUMAS_PROPERTY_RANGE,
        /** The particle kinetic energy, in GeV. */
        PUMAS_PROPERTY_KINETIC_ENERGY,
        /** The total magnetic rotation angle, in rad kg/m^(3). */
        PUMAS_PROPERTY_MAGNETIC_ROTATION,
        /**
         * The transport mean free path for soft processes, in kg/m^(2).
         */
        PUMAS_PROPERTY_TRANSPORT_PATH,
        /** The particle proper time, in kg/m^(2). */
        PUMAS_PROPERTY_PROPER_TIME
};

/**
 * Modes for the Monte Carlo transport.
 */
enum pumas_mode {
        /** The simulation of the corresponding property is disabled.
         *
         * **Note** : When running without energy losses a distance / grammage
         * limit must be defined or geometry callback provided.
         *
         * **Note** : When scattering is disabled, charged particles are still
         * deflected by external electromagnetic fields.
         */
        PUMAS_MODE_DISABLED = -1,
        /** Energy losses are purely determinstic as given by the Continuously
         * Slowing Down Approximation (CSDA).
         */
        PUMAS_MODE_CSDA = 0,
        /** Energy losses or scattering are simulated using a mixed (class II)
         * Monte-Carlo algorithm with a split between soft and hard collisions.
         */
        PUMAS_MODE_MIXED = 1,
        /** In addition to the mixed algorithm, the energy loss due to soft
         * electronic collisions is straggled.
         */
        PUMAS_MODE_STRAGGLED = 2,
        /**
         * Decays are accounted for by a weight factor. This is efficient
         * for muons but irrelevant -numerically instable- for the forward
         * transport of taus since they decay in flight. Hence this mode is
         * not allowed in the latter case.
         */
        PUMAS_MODE_WEIGHTED = 0,
        /** Decay vertices are randomised as a specific Monte-Carlo process.
         *
         * **Note** : the transported particle stops at the decay vertex but
         * its decay is not simulated, i.e. no daughter particles are
         * generated.
         */
        PUMAS_MODE_RANDOMISED = 1,
        /** Do a forward Monte Carlo transport.
         *
         * **Note** : the forward Monte Carlo transport is analog, i.e.
         * unweighted. However, if the decay mode is set to
         * `PUMAS_MODE_WEIGHTED` then the particle weight is updated
         * accordingly.
         */
        PUMAS_MODE_FORWARD = 0,
        /** Do a backward Monte Carlo transport.
         *
         * **Note** : the backward Monte Carlo transport is **not** analog. I.e.
         * the transported particle is weighted.
         */
        PUMAS_MODE_BACKWARD = 1
};

/** Return codes for the API functions. */
enum pumas_return {
        /** Execution was successful. */
        PUMAS_RETURN_SUCCESS = 0,
        /** The requested accuracy is not valid. */
        PUMAS_RETURN_ACCURACY_ERROR,
        /** End of file was reached. */
        PUMAS_RETURN_END_OF_FILE,
        /** The specified decay mode is not valid. */
        PUMAS_RETURN_DECAY_ERROR,
        /** Some medium has a wrong density value. */
        PUMAS_RETURN_DENSITY_ERROR,
        /** Some medium has a wrong density value. */
        PUMAS_RETURN_DIRECTION_ERROR,
        /** A non unit direction is provided. */
        PUMAS_RETURN_INCOMPLETE_FILE,
        /** Some index is out of validity range. */
        PUMAS_RETURN_INDEX_ERROR,
        /** The physics is not initialised or a NULL pointer was provided. */
        PUMAS_RETURN_PHYSICS_ERROR,
        /** An internal library error occured. */
        PUMAS_RETURN_INTERNAL_ERROR,
        /** Some read /write error occured. */
        PUMAS_RETURN_IO_ERROR,
        /** Some file is badly formated. */
        PUMAS_RETURN_FORMAT_ERROR,
        /** Wrong propagation medium. */
        PUMAS_RETURN_MEDIUM_ERROR,
        /** Some memory could not be allocated. */
        PUMAS_RETURN_MEMORY_ERROR,
        /** An invalid (unknown) DCS model was requested. */
        PUMAS_RETURN_MODEL_ERROR,
        /** A user supplied limit is required. */
        PUMAS_RETURN_MISSING_LIMIT,
        /** The random callback is not defined. */
        PUMAS_RETURN_MISSING_RANDOM,
        /** Some file could not be found. */
        PUMAS_RETURN_PATH_ERROR,
        /** A raise was called without any catch. */
        PUMAS_RETURN_RAISE_ERROR,
        /** Some input string is too long. */
        PUMAS_RETURN_TOO_LONG,
        /** No MDF file specified. */
        PUMAS_RETURN_UNDEFINED_MDF,
        /** An unkwon element was specified. */
        PUMAS_RETURN_UNKNOWN_ELEMENT,
        /** An unkwon material was specified. */
        PUMAS_RETURN_UNKNOWN_MATERIAL,
        /** The particle type is not known. */
        PUMAS_RETURN_UNKNOWN_PARTICLE,
        /** Some input value is not valid. */
        PUMAS_RETURN_VALUE_ERROR,
        /** The number of PUMAS return codes.  */
        PUMAS_N_RETURNS
};

/** Flags for transport events. */
enum pumas_event {
        /** No event occured or is foreseen. */
        PUMAS_EVENT_NONE = 0,
        /** A kinetic energy limit was reached or is foreseen. */
        PUMAS_EVENT_LIMIT_ENERGY = 1,
        /** A distance limit was reached or is foreseen. */
        PUMAS_EVENT_LIMIT_DISTANCE = 2,
        /** A grammage limit was reached or is foreseen. */
        PUMAS_EVENT_LIMIT_GRAMMAGE = 4,
        /** A proper time limit was reached or is foreseen. */
        PUMAS_EVENT_LIMIT_TIME = 8,
        /** Shortcut for any external limit. */
        PUMAS_EVENT_LIMIT = 15,
        /** A change of medium occured or is foreseen. */
        PUMAS_EVENT_MEDIUM = 16,
        /** A Bremsstrahlung occured or is foreseen. */
        PUMAS_EVENT_VERTEX_BREMSSTRAHLUNG = 32,
        /** A Pair creation occured or is foreseen. */
        PUMAS_EVENT_VERTEX_PAIR_CREATION = 64,
        /** A Photonuclear interaction occured or is foreseen. */
        PUMAS_EVENT_VERTEX_PHOTONUCLEAR = 128,
        /** A Delta ray occured or is foreseen. */
        PUMAS_EVENT_VERTEX_DELTA_RAY = 256,
        /** Shortcut for any Discrete Energy Loss (DEL). */
        PUMAS_EVENT_VERTEX_DEL = 480,
        /** A hard Coulombian interaction occured or is foreseen. */
        PUMAS_EVENT_VERTEX_COULOMB = 512,
        /** A decay has occured or is foreseen. */
        PUMAS_EVENT_VERTEX_DECAY = 1024,
        /** Shortcut for any interaction vertex. */
        PUMAS_EVENT_VERTEX = 2016,
        /** The particle has a nul or negative weight. */
        PUMAS_EVENT_WEIGHT = 2048,
        /** Extra flag for records tagging the first transport step. */
        PUMAS_EVENT_START = 4096,
        /** Extra flag for records tagging the last transport step. */
        PUMAS_EVENT_STOP = 8192
};

/** Radiative processes available in PUMAS. */
enum pumas_process {
        /** The Bremstrahlung process */
        PUMAS_PROCESS_BREMSSTRAHLUNG = 0,
        /** The e+e- pair production process */
        PUMAS_PROCESS_PAIR_PRODUCTION,
        /** The photonuclear process */
        PUMAS_PROCESS_PHOTONUCLEAR
};

/** Physics constants used by PUMAS. */
enum pumas_constant {
        /** The electromagnetic coupling constant, alpha. */
        PUMAS_CONSTANT_ALPHA_EM = 0,
        /** The Avogadro number, in mol. */
        PUMAS_CONSTANT_AVOGADRO_NUMBER,
        /** The electron Bohr radius, in m.  */
        PUMAS_CONSTANT_BOHR_RADIUS,
        /** The electron mass, in GeV/c^2. */
        PUMAS_CONSTANT_ELECTRON_MASS,
        /** The classical electron radius, in m. */
        PUMAS_CONSTANT_ELECTRON_RADIUS,
        /* The planck constant, in GeV m. */
        PUMAS_CONSTANT_HBAR_C,
        /** The muon decay length, in m. */
        PUMAS_CONSTANT_MUON_C_TAU,
        /** The muon mass, in GeV/c^2. */
        PUMAS_CONSTANT_MUON_MASS,
        /** The neutron mass, in GeV/c^2. */
        PUMAS_CONSTANT_NEUTRON_MASS,
        /** The mass of charged pions, in GeV/c^2 */
        PUMAS_CONSTANT_PION_MASS,
        /** The proton mass, in GeV/c^2. */
        PUMAS_CONSTANT_PROTON_MASS,
        /** The tau decay length, in m. */
        PUMAS_CONSTANT_TAU_C_TAU,
        /** The tau mass, in GeV/c^2. */
        PUMAS_CONSTANT_TAU_MASS,
        /** The number of PUMAS constants.  */
        PUMAS_N_CONSTANTS
};

/**
 * Container for a Monte-Carlo state.
 *
 * This structure contains data defining a particle Monte Carlo state. It must
 * be directly instancianted by the user.
 *
 * __Note__: this structure might be wrapped (sub-classed) in a larger one by
 * the user.
 */
struct pumas_state {
        /** The particle's electric charge. Note that non physical values,
         * i.e. different from 1 or -1, could be set. */
        double charge;
        /** The current kinetic energy, in GeV. */
        double energy;
        /** The total travelled distance, in m. */
        double distance;
        /** The total travelled grammage, in kg/m^2. */
        double grammage;
        /** The particle's proper time, in m/c. */
        double time;
        /** The Monte-Carlo weight. */
        double weight;
        /** The absolute location, in m. */
        double position[3];
        /** The momentum's unit direction. Must be normalised to one. */
        double direction[3];
        /** Status flag telling if the particle has decayed or not.  */
        int decayed;
};

/**
 * The local properties of a propagation medium.
 */
struct pumas_locals {
        /** The material local density, in kg/m^3. Setting a null or negative
         * value results in the material's default density being used.
         */
        double density;
        /** The local magnetic field components, in T. */
        double magnet[3];
};

struct pumas_medium;
/**
 * Callback for setting the local properties of a propagation medium.
 *
 * @param medium    The propagation medium.
 * @param state     The Monte-Carlo state for which the local properties are
 *                  requested.
 * @param locals    A pointer to a `pumas_locals` structure to update.
 * @return The size of local inhomogeneities (see below).
 *
 * The callback must return a length, in m, consistent with the size of the
 * propagation medium inhomogeneities, e. g. &rho; / |&nabla; &rho;| for a
 * density gradient. Returning zero or less signs that the propagation medium is
 * fully uniform.
 *
 * **Note** that inhomogeneities modelled by the `pumas_locals` callback must be
 * **continuous**. If the geometry has a density or magnetic field discontinuity
 * then this must be modelled by using separate media on both sides of the
 * discontinuity.
 *
 * **Warning** : it is an error to return zero or less for any position of the
 * medium if at least one area is not uniform. Instead one should use two
 * different media even though they have the same material base.
 *
 */
typedef double pumas_locals_cb (struct pumas_medium * medium,
    struct pumas_state * state, struct pumas_locals * locals);

/**
 * Description of a propagation medium.
 *
 * A propagation medium is fully defined by:
 *
 * - a *material* composition with a constant relative content.
 *
 * - Optionally, local properties set by a `pumas_locals_cb` callback.
 *
 * __Note__: this structure might be wrapped (sub-classed) in a larger one by
 * the user.
 */
struct pumas_medium {
        /**
         * The material index in the Material Description File (MDF). It can be
         * mapped to the corresponding name with the
         * `pumas_physics_material_name` function.
         */
        int material;
        /**
         * The user supplied callback for setting the medium local properties.
         * Setting a `NULL` callback results in the material's default density
         * being used with no magnetic field.
         */
        pumas_locals_cb * locals;
};

/** A recorded Monte-Carlo frame.
 *
 * This structure exposes data relative to a recorded Monte Carlo frame.  It is
 * not meant to be modified by the user.
 *
 * See the `pumas_recorder` structure for more information on recording Monte
 * Carlo steps.
 */
struct pumas_frame {
        /** The recorded Monte Carlo state. */
        struct pumas_state state;
        /** The corresponding target medium. */
        struct pumas_medium * medium;
        /** The corresponding Monte Carlo event. */
        enum pumas_event event;
        /** Link to the next frame in the record. */
        struct pumas_frame * next;
};

struct pumas_context;
/** A user supplied recorder callback.
 * @param context The recording simulation context.
 * @param state   The recorded particle state.
 * @param medium  The corresponding medium.
 * @param event   The step event.
 *
 * This callback allows to customize the recording of PUMAS Monte-Carlo events.
 *
 * **Note** : by default the recorder uses an in-memory copy with dynamic
 * allocation. Setting a custom recorder disables the default recording.
 */
typedef void pumas_recorder_cb (struct pumas_context * context,
    struct pumas_state * state, struct pumas_medium * medium,
    enum pumas_event event);

/**
 * A Monte-Carlo recorder.
 *
 * This structure is used for recording Monte Carlo steps and/or accessing
 * them. Although it exposes some public data that the user may alter it also
 * encloses other opaque data. Therefore, it **must** be handled with the
 * `pumas_recorder_create`, `pumas_recorder_clear` and `pumas_recorder_destroy`
 * functions.
 *
 * By default a newly created recorder is configured for saving all Monte Carlo
 * steps as `pumas_frame` objects. This behaviour can be modified by setting a
 * `pumas_recorder_cb` callback as *record* field. Other attributes of the
 * structure control the sampling rate of Monte Carlo steps and allow to access
 * the sampled `pumas_frame`, as detailed herein.
 *
 * **Note** : A recorder is enabled (disabled) by setting (unsetting) it to
 * (from) the *recorder* field of a `pumas_context`. Only the corresponding
 * context is recorded.
 */
struct pumas_recorder {
        /** Link to the 1^(st) recorded frame or `NULL` if none. This field
         * should not be modified.
         */
        struct pumas_frame * first;
        /** The total number of recorded frames. This field should not be
         * modified.
         */
        int length;
        /**
         * The sampling period of the recorder. If set to zero or less only
         * medium changes and user specified events are recorded. Defaults to 1,
         * i.e. all Monte-Carlo steps are recorded.
         */
        int period;
        /**
         * Link to an external (user supplied) recording callback. Note that
         * setting this value disables the in-memory frame recording. Defaults
         * to `NULL`.
         */
        pumas_recorder_cb * record;
        /**
         * A pointer to additional memory, if any is requested at
         * initialisation.
         */
        void * user_data;
};

/** Return codes for the medium callback. */
enum pumas_step {
        /** The proposed step is cross-checked by PUMAS beforehand.
         *
         * This is the safest option. Use this mode if you are unsure about
         * the compatibility of your geometry ray tracer with PUMAS.
         */
        PUMAS_STEP_CHECK = 0,
        /** The proposed step is used by PUMAS as is.
         *
         * This mode is intended for expert usage. Depending on the geometry ray
         * tracer used, it can save PUMAS from performing some redundant
         * geometry checks.
         */
        PUMAS_STEP_RAW
};

/**
 * Callback for locating the propagation medium of a `pumas_state`.
 *
 * @param context   The Monte-Carlo context requiring a medium.
 * @param state     The Monte-Carlo state for which the medium is requested.
 * @param medium    A pointer to store the medium or `NULL` if not requested.
 * @param step      The proposed step size or zero or less for an infinite
 *                    medium. If not requested this points to `NULL`.
 * @return If the proposed step size should be cross-checked by PUMAS
 * `PUMAS_STEP_CHECK` should be returned otherwise `PUMAS_STEP_RAW`.
 *
 * If *step* is not `NULL`, this callback must propose a Monte-Carlo stepping
 * distance, in m, consistent with the geometry. Note that returning zero or
 * less signs that the corresponding medium has no boundaries. When *medium* is
 * not `NULL` it must be set to the located `pumas_medium`.
 *
 * In addition the user must return a `pumas_step` enum indicating if the
 * proposed *step* needs to be cross-checked by PUMAS or if it should be used
 * raw. Managing steps that end on a geometry boundary can be tricky
 * numerically. Therefore it is recommended to return `PUMAS_STEP_CHECK` if you
 * are unsure of what to do since it is more robust. The raw mode is usefull if
 * your geometry engine already performs those checks in order to avoid double
 * work.
 *
 * **Warning** : it is an error to return zero or less for any state if the
 * extension is finite.
 *
 * **Warning** : in backward Monte Carlo mode the particle is propagated reverse
 * to the state direction. The user must take care to provide a *step* size
 * accordingly, i.e. consistent with the geometry in both forward and backward
 * modes.
 */
typedef enum pumas_step pumas_medium_cb (
    struct pumas_context * context, struct pumas_state * state,
    struct pumas_medium ** medium, double * step);

/**
 * Generic function pointer.
 *
 * This is a generic function pointer used to identify the library functions,
 * e.g. for error handling.
 */
typedef void pumas_function_t (void);

/**
 * Callback for error handling.
 *
 * @param rc          The PUMAS return code.
 * @param caller      The API function where the error occured.
 * @param message     Brief description of the error.
 *
 * The user can override the PUMAS default error handler by providing its own
 * error handler. It will be called at the return of any PUMAS library function
 * providing an error code.
 */
typedef void pumas_handler_cb (enum pumas_return rc, pumas_function_t * caller,
    const char * message);

/**
 * Callback providing a stream of pseudo random numbers.
 *
 * @param context The simulation context requiring a random number.
 * @return A uniform pseudo random number in [0;1].
 *
 * **Note** : this is the only random stream used by PUMAS. If overriding the
 * default `pumas_context` callback then the user must unsure proper behaviour,
 * i.e. that a flat distribution in [0;1] is indeed returned.
 *
 * **Warning** : if multiple contexts are used the user must also ensure that
 * this callback is thread safe, e.g. by using independant streams for each
 * context or a locking mechanism in order to share a single random stream.
 * The default `pumas_context` random callback uses distinct random streams per
 * context which ensures thread safety.
 */
typedef double pumas_random_cb (struct pumas_context * context);

/** Mode flags for the Monte Carlo transport. */
struct pumas_context_mode {
        /**
        * The mode used for the computation of energy losses. Default
        * is `PUMAS_MODE_STRAGGLED`. Other options are `PUMAS_MODE_DISABLED`,
        * `PUMAS_MODE_CSDA` and `PUMAS_MODE_MIXED`.
        */
        enum pumas_mode energy_loss;
        /**
        * The mode for handling decays. Default is `PUMAS_MODE_WEIGHTED`
        * for a muon or `PUMAS_MODE_RANDOMISED` for a tau. Set this to
        * `PUMAS_MODE_DISABLED` in order to disable decays at all.
        */
        enum pumas_mode decay;
        /**
        * Direction of the Monte Carlo flow. Default is
        * `PUMAS_MODE_FORWARD`. Set this to `PUMAS_MODE_BACKWARD` for a
        * reverse Monte Carlo.
        */
        enum pumas_mode direction;
        /**
        * Algorithm for the simulation of the scattering. Default is
        * `PUMAS_MODE_MIXED`. Other option is `PUMAS_MODE_DISABLED` which
        * neglects any scattering.
        */
        enum pumas_mode scattering;
};

/** External limits for the Monte Carlo transport. */
struct pumas_context_limit {
        /**
         * The minimum kinetic energy for forward transport, or the
         * maximum one for backward transport, in GeV.
         */
        double energy;
        /** The maximum travelled distance, in m. */
        double distance;
        /** The maximum travelled grammage, in kg/m^2. */
        double grammage;
        /** The maximum travelled proper time, in m/c. */
        double time;
};

/**
 * A simulation context.
 *
 * This structure manages thread specific data for a PUMAS Monte Carlo
 * simulation.  It also exposes configuration parameters for the Monte Carlo
 * transport. The exposed parameters can be directly modified by the user.
 *
 * __Warning__: since the simulation context wraps opaque data it **must** be
 * created (destroyed) with the `pumas_context_create`
 * (`pumas_context_destroy`) function.
 *
 * A context created with `pumas_context_create` is initialised with default
 * settings. That is, the transport is configured for forward Monte Carlo with
 * the highest level of detail available, i.e. energy straggling and scattering
 * enabled.  This can be modified by overriding the *mode* attribute of the
 * simulation context.
 *
 * **Note**: in the case of a muon projectile, the default initialisation is to
 * account for decays by weighting according to the proper time
 * (`PUMAS_MODE_WEIGHTED`). However, for a tau projectile the default is to
 * randomise the decay location (`PUMAS_MODE_RANDOMISE`).
 *
 * Each simulation context natively embeds a pseudo random engine. A Mersenne
 * Twister algorithm is used. The random engine can be seeded with the
 * `pumas_context_random_seed_set` function.  Note that two contexts seeded
 * with the same value are 100% correlated. If no seed is provided then one is
 * picked randomly from the OS, e.g.  from `/dev/urandom` on UNIX.
 * Alternatively, a custom random engine can be used instead of the native one
 * by overriding the *random* callback.
 *
 * The geometry of the simulation is specified by setting the *medium* field
 * with a `pumas_medium_cb` callback.  By default the *medium* field is `NULL`.
 * Note that it must be set by the user before calling
 * `pumas_context_transport`.
 *
 * The *event* field of the simulation context allows to specify end conditions
 * for the Monte Carlo transport. E.g. a lower (upper) limit can be set on the
 * kinetic energy of the projectile in forward (backward) mode. The limit value
 * is specified by setting the corresponding *limit* field.
 */
struct pumas_context {
        /** The geometry of the simulation specified as a callback. It must be
         * provided by the user.
         */
        pumas_medium_cb * medium;
        /** The pseudo random generator of the simulation context. An
         *  alternative generator can be used by overriding this callback.
         */
        pumas_random_cb * random;
        /** An optionnal recorder for Monte Carlo steps. */
        struct pumas_recorder * recorder;
        /** A pointer to additional memory if any is requested at
         * initialisation. Otherwise this points to `NULL`.
         */
        void * user_data;

        /** Settings controlling the Monte Carlo transport algirithm. */
        struct pumas_context_mode mode;
        /**
         * The events that stop the transport. Default is `PUMAS_EVENT_NONE`,
         * i.e. the transport stops only if the particle exits the simulation
         * media, or if it looses all of its energy.
         */
        enum pumas_event event;

        /** External limits for the Monte Carlo transport. */
        struct pumas_context_limit limit;

        /** Tuning parameter for the accuracy of the Monte Carlo integration.
         *
         * The Monte Carlo transport is discretized in elementary steps. This
         * parameter directly controls the length of these steps. The smaller
         * the *accuracy* value the smaller the step length. Thus, the longer
         * the Monte Carlo simulation.
         */
        double accuracy;
};

/**
 * Physics tables for the Monte Carlo transport
 *
 * This is an **opaque** structure wrapping physics tables for the Monte Carlo
 * transport.  See `pumas_physics_create` for informations on how to create a
 * physics object.
 *
 * __Note__: the physics is configured during its instantiation. It cannot
 * be modified afterwards. Only the composition of composite materials can be
 * updated with the `pumas_physics_composite_update` function.
 *
 * The settings of a `pumas_physics` instance can be inspected with the
 * `pumas_physics_cutoff`, `pumas_physics_dcs` and `pumas_physics_elastic_ratio`
 * functions. The materials data are retrieved with the
 * `pumas_physics_element_*`, and `pumas_physics_material_*` functions.
 * Alternatively, the `pumas_physics_print` function can be used in order to
 * print out a human readable summary of the physics.
 *
 * Physics properties are tabulated as function of the projectile kinetic
 * energy.  The tabulated values can be retrieved with the
 * `pumas_physics_table_value` function. In addition, the
 * `pumas_physics_property_*` functions provide smooth interpolations of physics
 * properties for arbitrary kinetic energy values.
 */
struct pumas_physics;

/**
 * Prototype for a Differential Cross-Section (DCS).
 *
 * @param Z       The charge number of the target atom.
 * @param A       The mass number of the target atom.
 * @param m       The projectile rest mass, in GeV
 * @param K       The projectile kinetic energy, in GeV.
 * @param q       The projectile energy loss, in GeV.
 * @return The corresponding value of the atomic DCS, in m^(2)/GeV.
 *
 * The `pumas_dcs_get` function allows to retrieve the DCS for a given process
 * and model. Extra DCSs can be registered with the `pumas_dcs_register`
 * function.
 *
 * **Note** : only the Bremsstrahlung, pair creation and photonuclear processes
 * can be modified.
 */
typedef double pumas_dcs_t (double Z, double A, double m, double K, double q);

/**
 */
struct pumas_physics_settings {
        /** Relative cutoff between soft and hard energy losses.
         *
         * Setting a null or negative value results in the default cutoff value
         * to be used i.e. 5% which is a good compromise between speed and
         * accuracy for transporting a continuous spectrumm, see e.g.  [Sokalski
         * et al.](https://doi.org/10.1103/PhysRevD.64.074015)
         *
         * __Warning__ : In backward mode, with mixed or straggled energy loss,
         * cutoff values lower than 1% are not currently supported.
         */
        double cutoff;
        /** Ratio of the mean free path for hard elastic events to the smallest
         * of the transport mean free path or CSDA range.
         *
         * The lower the ratio the more detailed the simulation of elastic
         * scattering, see e.g. [Fernandez-Varea et al. (1993)](
         * https://doi.org/10.1016/0168-583X(93)95827-R)  Setting a null or
         * negative value results in the default ratio to be used i.e. 5%.
         */
        double elastic_ratio;
        /** Physics model for the Bremsstrahlung process.
         *
         *  Available models are:
         *
         *  - `ABB`: Andreev, Bezrukov and Bugaev, Physics of Atomic Nuclei 57
         *           (1994) 2066.
         *
         *  - `KKP`: Kelner, Kokoulin and Petrukhin, Moscow Engineering Physics
         *           Inst., Moscow, 1995.
         *
         *  - `SSR`: Sandrock, Soedingresko and Rhode, [ICRC 2019](
         *           https://arxiv.org/abs/1910.07050).
         *
         * Setting a `NULL` value results in PUMAS default Bremsstrahlung model
         * to be used, i.e. `SSR`.
         * */
        const char * bremsstrahlung;
        /** Physics model for e^(+)e^(-) pair production.
         *
         *  Available models are:
         *
         *  - `KKP`: Kelner, Kokoulin and Petrukhin, Soviet Journal of Nuclear
         *           Physics 7 (1968) 237.
         *
         *  - `SSR`: Sandrock, Soedingresko and Rhode, [ICRC 2019](
         *           https://arxiv.org/abs/1910.07050).
         *
         * Setting a `NULL` value results in PUMAS default pair production model
         * to be used, i.e. `SSR`.
         */
        const char * pair_production;
        /** Physics model for photonuclear interactions.
         *
         *  Available models are:
         *
         *  - `BBKS`: Bezrukov, Bugaev, Sov. J. Nucl. Phys. 33 (1981), 635.
         *            with improved photon-nucleon cross-section according to
         *            [Kokoulin](https://doi.org/10.1016/S0920-5632(98)00475-7)
         *            and hard component from [Bugaev and Shlepin](
         *            https://doi.org/10.1103/PhysRevD.67.034027).
         *
         *  - `BM`  : Butkevich and Mikheyev, Soviet Journal of Experimental and
         *            Theoretical Physics 95 (2002) 11.
         *
         *  - `DRSS`: Dutta, Reno, Sarcevic and Seckel, [Phys.Rev. D63 (2001)
         *            094020](https://arxiv.org/abs/hep-ph/0012350).
         *
         * Setting a `NULL` value results in PUMAS default photonuclear model to
         * be used, i.e. `DRSS`.
         */
        const char * photonuclear;
        /** The number of kinetic energy values to tabulate. Providing a value
         * of zero or less results in the PDG energy grid being used.
         */
        int n_energies;
        /** Array of kinetic energy values to tabulate. Providing a `NULL`
         * value results in the PDG energy grid being used.
         */
        double * energy;
        /** Flag to force updating existing stopping power table(s). The default
         * behaviour is to not overwrite any already existing file.
         */
        int update;
        /** Flag to enable dry mode.
         *
         * In dry mode energy loss files are generated but the physics is
         * not created. This is usefull e.g. if only energy loss files are
         * needed as a speed up.
         *
         * __Warning__: in dry mode no physics is returned, i.e. the *physics*
         * pointer provided by `pumas_physics_create` points to `NULL`.
         */
        int dry;
};

/**
 * Create physics tables.
 *
 * @param physics      The physics tables.
 * @param particle     The type of the particle to transport.
 * @param mdf_path     The path to a Material Description File (MDF), or `NULL`.
 * @param dedx_path    The path to the energy loss tabulation(s), or `NULL`.
 * @param settings     Extra physics settings or `NULL`.
 * @return On success `PUMAS_RETURN_SUCCESS` is returned otherwise an error
 * code is returned as detailed below.
 *
 * Create physics tables for a set of materials and a given *particle*. These
 * tables are looked-up by the Monte Carlo engine for fast evaluation of physics
 * properties during the transport. Tabulated properties are cross-sections,
 * materials stopping power, transport mean free path length, etc.
 *
 * The materials to tabulate are specified in a Materials Description File (MDF)
 * provided with the *mdf_path* argument. If a `NULL` argument is given then the
 * path is read from the `PUMAS_MDF` environment variable. Examples of MDF are
 * available from the [pumas-materials
 * repository](https://github.com/niess/pumas-materials).
 *
 * **Note**: a MDF must be provided in any case.
 *
 * The physics creation generates stopping power table(s) in the Particle Data
 * Group (PDG) format. These tables are written to the *dedx_path* directory.
 * If the latter is `NULL` then it is read from the `PUMAS_DEDX` environment
 * variable. If both are `NULL` then the tables are dumped beside the MDF, i.e.
 * in the same directory.
 *
 * Specific physics settings can be selected by providing a
 * `pumas_physics_settings` structure. If `NULL` is provided then PUMAS default
 * physics settings are used which should perform well for most use cases.
 *
 * Call `pumas_physics_destroy` in order to unload the physics and release the
 * corresponding alocated memory.
 *
 * __Note__: computing the physics tables can be long, e.g. a few seconds per
 * material defined in the MDF. The `pumas_physics_dump` and
 * `pumas_physics_load` functions allow to save and load the tables to/from a
 * file. This can be used in order to greatly speed up the physics
 * initialisation.
 *
 * __Warning__: this function is **not** thread safe.
 *
 * __Error codes__
 *
 *     PUMAS_RETURN_END_OF_FILE             And unexpected EOF occured.
 *
 *     PUMAS_RETURN_FORMAT_ERROR            A file has a wrong format.
 *
 *     PUMAS_RETURN_INCOMPLETE_FILE         There are missing entries in
 * the MDF.
 *
 *     PUMAS_RETURN_IO_ERROR                A file could not be read.
 *
 *     PUMAS_RETURN_MEMORY_ERROR            Could not allocate memory.
 *
 *     PUMAS_RETURN_MODEL_ERROR             A requested DCS model is not valid.
 *
 *     PUMAS_RETURN_PATH_ERROR              A file could not be opened.
 *
 *     PUMAS_RETURN_PHYSICS_ERROR           A `NULL` physics pointer was
 * provided.
 *
 *     PUMAS_RETURN_TOO_LONG                Some XML node in the MDF is
 * too long.
 *
 *     PUMAS_RETURN_UNDEFINED_MDF           No MDF was provided.
 *
 *     PUMAS_RETURN_UNKNOWN_ELEMENT         An element in the MDF wasn't
 * defined.
 *
 *     PUMAS_RETURN_UNKNOWN_MATERIAL        An material in the MDF wasn't
 * defined.
 *
 *     PUMAS_RETURN_UNKNOWN_PARTICLE        The given type is not supported.
 *
 *     PUMAS_RETURN_VALUE_ERROR             A bad cutoff or elastic ratio
 * was provided.
 */
PUMAS_API enum pumas_return pumas_physics_create(
    struct pumas_physics ** physics, enum pumas_particle particle,
    const char * mdf_path, const char * dedx_path,
    const struct pumas_physics_settings * settings);

/**
 * Destroy a physics instance.
 *
 * @param physics      The physics tables.
 *
 * Finalise the physics and free its memory. Call `pumas_physics_create` or
 * `pumas_physics_load` in order to reload the physics.
 *
 * __Note__: at return the *physics* pointer points to `NULL`.
 *
 * __Note__: finalising the physics does not release the memory allocated for
 * related `pumas_context`. This must be done explictly with the
 * `pumas_context_destroy` function.
 *
 * __Warning__: it is the user responsability to not use any simulation context
 * whose physics would have been destroyed. Doing so would lead to
 * unexpected results, e.g. memory corruption.
 *
 * __Warning__: this function is **not** thread safe.
 *
 */
PUMAS_API void pumas_physics_destroy(struct pumas_physics ** physics);

/**
 * Dump the physics tables to a file.
 *
 * @param physics   The physics tables.
 * @param stream    The stream where to dump.
 * @return On success `PUMAS_RETURN_SUCCESS` is returned otherwise an error
 * code is returned as detailed below.
 *
 * Dump the *physics* tables to *stream* as a raw binary object.  This binary
 * dump can be re-loaded with the `pumas_physics_load` function. This provides a
 * fast initialisation of the physics tables for subsequent uses.
 *
 * __Warning__: the binary dump is raw formated, thus *a priori* platform
 * dependent.
 *
 * __Error codes__
 *
 *     PUMAS_RETURN_PHYSICS_ERROR           The physics is not initialised.
 *
 *     PUMAS_RETURN_PATH_ERROR              The output stream in invalid (NULL).
 *
 *     PUMAS_RETURN_IO_ERROR                Could not write to the stream.
 */
PUMAS_API enum pumas_return pumas_physics_dump(
    const struct pumas_physics * physics, FILE * stream);

/**
 * Load the physics tables from a file.
 *
 * @param physics   The physics tables.
 * @param stream    The stream to load from.
 * @return On success `PUMAS_RETURN_SUCCESS` is returned otherwise an error
 * code is returned as detailed below.
 *
 * Load the physics tables from a binary dump previously generated with
 * `pumas_physics_dump`.
 *
 * __Note__: loading to an already initialised physics instance generates an
 * error. The `pumas_physics_destroy` function must be called first.
 *
 * __Warning__: the binary dump is raw formated, thus *a priori* platform
 * dependent.
 *
 * __Error codes__
 *
 *     PUMAS_RETURN_FORMAT_ERROR            The binary dump is not compatible
 * with the current version.
 *
 *     PUMAS_RETURN_PHYSICS_ERROR           The physics is not initialised.
 *
 *     PUMAS_RETURN_PATH_ERROR              The input stream in invalid (null).
 *
 *     PUMAS_RETURN_IO_ERROR                Could not read from the stream.
 */
PUMAS_API enum pumas_return pumas_physics_load(
    struct pumas_physics ** physics, FILE * stream);

/**
 * Get the cutoff value used by the physics.
 *
 * @param physcis   The physics tables.
 * @return The cutoff value or -1 if the physics is not properly initialised.
 *
 * The cutoff value between soft and hard energy losses is specified during the
 * physics initialisation with `pumas_physics_create`. It cannot be modified
 * afterwards. Instead a new physics object must be created.
 */
PUMAS_API double pumas_physics_cutoff(const struct pumas_physics * physics);

/**
 * Get the elastic ratio value used by the physics.
 *
 * @param physics    The physics tables.
 * @return The elastic ratio or -1 if the physics is not properly initialised.
 *
 * The ratio of the m.f.p. to the transport m.f.p. for elastic events is
 * specified during the physics initialisation with `pumas_physics_create`.  It
 * cannot be modified afterwards. Instead a new physics object must be created.
 */
PUMAS_API double pumas_physics_elastic_ratio(
    const struct pumas_physics * physics);

/**
 * Transport a Monte Carlo particle.
 *
 * @param context The simulation context.
 * @param state   The initial state or the final state at return.
 * @param event   The end event or `NULL`.
* @param media   The initial and final media, or `NULL`.
* @return On success `PUMAS_RETURN_SUCCESS` is returned otherwise an error
* code is returned as detailed below.
*
* Transport a Monte Carlo *state* according to a simulation *context*.  The
* transport algorithm and the geometry are set by the `pumas_context`
* structure.
*
* At return the particle *state* is updated. If *event* is not `NULL` it is
* filled with the transport end condition. If *media* is not `NULL` it contains
 * the initial (index 0) and final (index 1) media seen by the particle.
 *
 * **Warning**: the state direction must be a unit vector. Otherwise an error
 * is returned (see below).
 *
 * __Error codes__
 *
 *     PUMAS_RETURN_ACCURACY_ERROR          The requested accuracy is not valid.
 *
 *     PUMAS_RETURN_DENSITY_ERROR           A null or negative density was
 * encountered.
 *
 *     PUMAS_RETURN_DIRECTION_ERROR         A non unit direction was provided.
 *
 *     PUMAS_RETURN_PHYSICS_ERROR           The physics is not initalised.
 *
 *     PUMAS_RETURN_MEDIUM_ERROR            The medium callback was not defined.
 *
 *     PUMAS_RETURN_MISSING_LIMIT           An external limit is needed.
 *
 *     PUMAS_RETURN_VALUE_ERROR             The State or the context is NULL.
 */
PUMAS_API enum pumas_return pumas_context_transport(
    struct pumas_context * context, struct pumas_state * state,
    enum pumas_event * event, struct pumas_medium * media[2]);

/**
 * Print a summary of the physics.
 *
 * @param physics       The physics tables.
 * @param stream        A stream where the summary will be formated to.
 * @param tabulation    The tabulation separator or `NULL`.
 * @param newline       The newline separator or `NULL`.
 * @return On success `PUMAS_RETURN_SUCCESS` is returned otherwise an error
 * code is returned as detailed below.
 *
 * The summary is JSON formated. It provides information on the physics settings
 * as well as a summary of the tabulated materials.  The *tabulation* and
 * *newline* parameters allow to control the output rendering. Empty strings
 * are used if these arguments are `NULL`.
 *
 * __Error codes__
 *
 *     PUMAS_RETURN_PHYSICS_ERROR           The physics is not initalised.
 *
 *     PUMAS_RETURN_IO_ERROR                Could not write to *stream*.
 */
PUMAS_API enum pumas_return pumas_physics_print(
    const struct pumas_physics * physics, FILE * stream,
    const char * tabulation, const char * newline);

/**
 * Get the version of the PUMAS library.
 *
 * @param major    The major version number or `NULL`.
 * @param minor    The minor version number or `NULL`.
 * @param patch    The patch version number or `NULL`.
 *
 * The PUMAS library version is given as MAJOR.MINOR.PATCH. If *major*, *minor*
 * or *patch* is not required, then the corresponding pointer can be set to
 * `NULL`.
 */
PUMAS_API void pumas_version(int * major, int * minor, int * patch);

/**
 * Get information on the transported particle.
 *
 * @param physics       The physics tables.
 * @param particle      The type of the transported particle or `NULL`.
 * @param lifetime      The proper lifetime, in m/c, or `NULL`.
 * @param mass          The mass of the transported particle, in GeV, or `NULL`.
 * @return On success `PUMAS_RETURN_SUCCESS` is returned otherwise an error
 * code is returned as detailed below.
 *
 * This function allows to retrieve information on the transported particle.
 *
 * __Note__: not needed arguments can be set to `NULL`.
 *
 * __Error codes__
 *
 *     PUMAS_RETURN_PHYSICS_ERROR    The physics is not initalised.
 */
PUMAS_API enum pumas_return pumas_physics_particle(
    const struct pumas_physics * physics, enum pumas_particle * particle,
    double * lifetime, double * mass);

/**
 * Return a string describing a PUMAS library function.
 *
 * @param function    The library function.
 * @return a static string.
 *
 * This function is meant for verbosing when handling errors.
 *
 * __Note__: this function **is** thread safe.
 */
PUMAS_API const char * pumas_error_function(pumas_function_t * function);

/**
 * Set or clear the error handler.
 *
 * @param handler    The error handler to set or `NULL`.
 *
 * Set the error handler callback for PUMAS library functions.  The user can
 * override the PUMAS default error handler by providing its own error handler.
 * If *handler* is set to `NULL` then error callbacks are disabled.
 *
 * __Warning__: this function is **not** thread safe.
 */
PUMAS_API void pumas_error_handler_set(pumas_handler_cb * handler);

/**
 * Get the current error handler.
 *
 * By default PUMAS is configured to printout to stderr whenever an error occurs
 * and to exit back to the OS. See `pumas_error_handler_set` in order to
 * override this behaviour.
 *
 * @return The current error handler or `NULL` if none.
 */
PUMAS_API pumas_handler_cb * pumas_error_handler_get(void);

/**
 * Catch the next error.
 *
 * @param enable   A flag for enabling or disabling error catch.
 *
 * Enable or disable the catch of the next PUMAS library error. If catching is
 * enabled then library errors do **not** trigger the error handler. Call
 * `pumas_error_raise` to enable the error handler again and raise any caught
 * error.
 *
 * __Note__: only the first error occuring is recorded. Subsequent error(s) are
 * muted but not recorded.
 *
 * __Warning__: this function is **not** thread safe. Only a single error stream
 * can be handled at a time.
 */
PUMAS_API void pumas_error_catch(int enable);

/**
 * Raise any caught error.
 *
 * @return If no error was caught `PUMAS_RETURN_SUCCESS` is returned otherwise
 * an error code is returned as detailed below.
 *
 * Raise any caught error. Error catching must have been enabled first with
 * `pumas_error_catch` otherwise a `PUMAS_RETURN_RAISE_ERROR` is returned.
 *
 * __Note__: calling this function disables further error's catching.
 *
 * __Warning__: this function is **not** thread safe. Only a single error stream
 * can be handled at a time.
 *
 * __Error codes__
 *
 *     PUMAS_RETURN_RAISE_ERROR    Error catching hasn't been enabled.
 *
 *     PUMAS_RETURN_*              Any caught error code.
 */
PUMAS_API enum pumas_return pumas_error_raise(void);

/**
 * Create a simulation context.
 *
 * @param context         The new simulation context.
 * @param physics         A physics instance.
 * @param extra_memory    Size of the user memory or 0 if none is requested.
 * @return On success `PUMAS_RETURN_SUCCESS` is returned otherwise an error
 * code is returned as detailed below.
 *
 * Create a new simulation *context* initialised with a default configuration.
 * Call `pumas_context_destroy` in order to release the memory allocated for the
 * context. 
 *
 * **Note**: the simulation context is bound to the provided *physics* instance.
 *
 * If *extra_memory* is strictly positive then the context memory is extended by
 * *extra_memory* bytes reserved to the user. This memory is accessed with the
 * *user_data* field of the instanciated context.
 *
 * See the `pumas_context` structure for more detailed usage of a simulation
 * context.
 *
 * __Error codes__
 *
 *     PUMAS_RETURN_MEMORY_ERROR            Could not allocate memory.
 *
 *     PUMAS_RETURN_PHYSICS_ERROR           The physics is not initialised.
 */
PUMAS_API enum pumas_return pumas_context_create(
    struct pumas_context ** context, const struct pumas_physics * physics,
    int extra_memory);

/**
 * Set the random seed of a simulation context.
 *
 * @param context         The simulation context.
 * @param seed            The random seed or `NULL`.
 * @return On success `PUMAS_RETURN_SUCCESS` is returned otherwise an error
 * code is returned as detailed below.
 *
 * Set the random seed of the simulation *context* and reset the random state
 * accordingly.  If `NULL` is provided then the seed is randomly initialised
 * from the OS, e.g.  using `/dev/urandom` on UNIX.
 *
 * **Note**: each simulation context manages its own random stream.  See the
 * `pumas_context` documentation for more detailed usage.
 *
 * __Error codes__
 *
 *     PUMAS_RETURN_MEMORY_ERROR            Could not allocate memory.
 *
 *     PUMAS_RETURN_PATH_ERROR              The OS random stream could not be
 * read.
 */
PUMAS_API enum pumas_return pumas_context_random_seed_set(
    struct pumas_context * context, const unsigned long * seed);

/**
 * Get the random seed of a simulation context.
 *
 * @param context         The simulation context.
 * @param seed            The corresponding random seed.
 * @return On success `PUMAS_RETURN_SUCCESS` is returned otherwise an error
 * code is returned as detailed below.
 *
 * Get the random seed of the simulation context. If the seed has not been
 * previously set with `pumas_context_random_seed_set` then it is randomly
 * initialised from the OS, e.g. using `/dev/urandom` on UNIX.
 *
 * **Note**: each simulation context manages its own random stream.  See the
 * `pumas_context` documentation for more detailed usage.
 *
 * __Error codes__
 *
 *     PUMAS_RETURN_MEMORY_ERROR            Could not allocate memory.
 *
 *     PUMAS_RETURN_PATH_ERROR              The OS random stream could not be
 * read.
 */
PUMAS_API enum pumas_return pumas_context_random_seed_get(
    struct pumas_context * context, unsigned long * seed);

/**
 * Load the random state of a simulation context.
 *
 * @param context         The simulation context.
 * @param stream          The stream to load from.
 * @return On success `PUMAS_RETURN_SUCCESS` is returned otherwise an error
 * code is returned as detailed below.
 *
 * Restore the random state of the simulation context from a *stream*.  See the
 * `pumas_context_random_dump` function for the converse, i.e. dumping the
 * random engine state to a file.
 *
 * __Error codes__
 *
 *     PUMAS_RETURN_FORMAT_ERROR            The dump version is invalid.
 *
 *     PUMAS_RETURN_MEMORY_ERROR            Could not allocate memory.
 *
 *     PUMAS_RETURN_IO_ERROR                Could not read from the stream.
 *
 *     PUMAS_RETURN_PATH_ERROR              The input stream is invalid (NULL).
 */
PUMAS_API enum pumas_return pumas_context_random_load(
    struct pumas_context * context, FILE * stream);

/**
 * Dump the random state of a simulation context.
 *
 * @param context         The simulation context.
 * @param stream          The stream to dump to.
 * @return On success `PUMAS_RETURN_SUCCESS` is returned otherwise an error
 * code is returned as detailed below.
 *
 * Save the random state of the simulation context to a *stream*. See the
 * `pumas_context_random_load` function for the converse, i.e. loading back the
 * random state from a stream.
 *
 * __Error codes__
 *
 *     PUMAS_RETURN_MEMORY_ERROR            Could not allocate memory.
 *
 *     PUMAS_RETURN_IO_ERROR                Could not write to the stream.
 *
 *     PUMAS_RETURN_PATH_ERROR              The input stream is invalid (NULL).
 */
PUMAS_API enum pumas_return pumas_context_random_dump(
    struct pumas_context * context, FILE * stream);

/**
 * Destroy a simulation context.
 *
 * @param context The simulation context.
 *
 * Destroy a simulation *context* previously created with
 * `pumas_context_create`.  The corresponding allocated memory is released.
 *
 * **Note**: on return the *context* pointer is set to `NULL`.
 */
PUMAS_API void pumas_context_destroy(struct pumas_context ** context);

/**
 * Get the physics used by a simulation context.
 *
 * @param context The simulation context.
 * @return The corresponding physics or `NULL`.
 *
 * The physics used by a simulation context cannot be changed. If an alternative
 * physics is needed then a new `pumas_context` object must be created.
 */
PUMAS_API const struct pumas_physics * pumas_context_physics_get(
    const struct pumas_context * context);

/**
 * The CSDA range.
 *
 * @param physics     The physics tables.
 * @param mode        The energy loss mode.
 * @param material    The material index.
 * @param energy      The initial kinetic energy, in GeV.
 * @param range       The grammage range in kg/m^(2).
 * @return On success `PUMAS_RETURN_SUCCESS` is returned otherwise an error
 * code is returned as detailed below.
 *
 * This function computes the CSDA range of the projectile in a given target
 * *material*. See the `pumas_physics_property_kinetic_energy` for the converse,
 * i.e. getting the minimum energy for a given range.
 *
 * __Note__: the energy loss mode must be one of `PUMAS_MODE_CSDA` or
 * `PUMAS_MODE_MIXED`.
 *
 * Divide the *range* value by the target density in order to get the
 * range in unit of distance.
 *
 * __Error codes__
 *
 *     PUMAS_RETURN_INDEX_ERROR             The mode or material index is
 * not valid.
 *
 *     PUMAS_RETURN_PHYSICS_ERROR           The physics is not initialised.
 */
PUMAS_API enum pumas_return pumas_physics_property_range(
    const struct pumas_physics * physics, enum pumas_mode mode,
    int material, double energy, double * range);

/**
 * The total proper time for continuous energy loss.
 *
 * @param physics     The physics tables.
 * @param mode        The energy loss mode.
 * @param material    The material index.
 * @param energy      The initial kinetic energy, in GeV.
 * @param time        The normalised proper time in kg/m^(2).
 * @return On success `PUMAS_RETURN_SUCCESS` is returned otherwise an error
 * code is returned as detailed below.
 *
 * This function computes the ellapsed proper time of a particle over
 * its total range. Continuous energy loss is assumed.
 *
 * __Note__: the energy loss mode must be one of `PUMAS_MODE_CSDA` or
 * `PUMAS_MODE_MIXED`.
 *
 * Divide the *time* value by the target density times *c* in order to get the
 * proper time in unit of time.
 *
 * __Error codes__
 *
 *     PUMAS_RETURN_INDEX_ERROR             The mode or material index is
 * not valid.
 *
 *     PUMAS_RETURN_PHYSICS_ERROR           The physics is not initialised.
 */
PUMAS_API enum pumas_return pumas_physics_property_proper_time(
    const struct pumas_physics * physics, enum pumas_mode mode,
    int material, double energy, double * time);

/**
 * Magnetic rotation angle for a uniform magnetic field.
 *
 * @param physics     The physics tables.
 * @param material    The material index.
 * @param energy      The initial kinetic energy, in GeV.
 * @param angle       The normalised rotation angle in rad kg/m^(3)/T.
 * @return On success `PUMAS_RETURN_SUCCESS` is returned otherwise an error
 * code is returned as detailed below.
 *
 * This function computes the magnetic rotation angle of a particle over
 * its total range. A uniform magnetic field is assumed with CSDA energy loss.
 *
 * Multiply the returned value by the amplitude of the transverse magnetic field
 * and divide by the target density in order to get the rotation angle in
 * radian.
 *
 * __Error codes__
 *
 *     PUMAS_RETURN_INDEX_ERROR             The material index is not valid.
 *
 *     PUMAS_RETURN_PHYSICS_ERROR           The physics is not initialised.
 */
PUMAS_API enum pumas_return pumas_physics_property_magnetic_rotation(
    const struct pumas_physics * physics, int material, double energy,
    double * angle);

/**
 * Kinetic energy for travelling over a given CSDA range.
 *
 * @param physics     The physics tables.
 * @param mode        The energy loss mode
 * @param material    The material index.
 * @param range       The requested grammage range, in kg/m^(2).
 * @param energy      The required kinetic energy in GeV.
 * @return On success `PUMAS_RETURN_SUCCESS` is returned otherwise an error
 * code is returned as detailed below.
 *
 * This is the inverse of the `pumas_physics_property_range` function. It
 * computes the minimum kinetic energy needed in order to cross a given range of
 * material, assuming deterministic (CSDA) energy loss.
 *
 * __Note__: the energy loss mode must be one of `PUMAS_MODE_CSDA` or
 * `PUMAS_MODE_MIXED`.
 *
 * __Error codes__
 *
 *     PUMAS_RETURN_INDEX_ERROR             The mode or material index is
 * not valid.
 *
 *     PUMAS_RETURN_PHYSICS_ERROR           The physics is not initialised.
 */
PUMAS_API enum pumas_return pumas_physics_property_kinetic_energy(
    const struct pumas_physics * physics, enum pumas_mode mode,
    int material, double range, double * energy);

/**
 * Stopping power per unit mass.
 *
 * @param physics     The physics tables.
 * @param mode        The energy loss mode
 * @param material    The material index.
 * @param energy      The kinetic energy, in GeV.
 * @param dedx        The computed stopping power in GeV/(kg/m^(2)).
 * @return On success `PUMAS_RETURN_SUCCESS` is returned otherwise an error
 * code is returned as detailed below.
 *
 * This function computes the stopping power in a given *material*. See the
 * `pumas_property_range` function in order to get the corresponding CSDA range.
 *
 * __Note__: the energy loss mode must be one of `PUMAS_MODE_CSDA` or
 * `PUMAS_MODE_MIXED`. In the latter case the stopping power is restricted to
 * soft collisions.
 *
 * The stopping power, *dedx*, is given per unit mass. Multiply by the target
 * density in order to get the stopping power per unit length.
 *
 * __Error codes__
 *
 *     PUMAS_RETURN_INDEX_ERROR             The mode or material index is
 * not valid.
 *
 *     PUMAS_RETURN_PHYSICS_ERROR           The physics is not initialised.
 */
PUMAS_API enum pumas_return pumas_physics_property_stopping_power(
    const struct pumas_physics * physics, enum pumas_mode mode,
    int material, double energy, double * dedx);

/**
 * Energy loss straggling parameter.
 *
 * @param physics     The physics tables.
 * @param material    The material index.
 * @param energy      The kinetic energy, in GeV.
 * @param straggling  The computed energy loss straggling in GeV^(2)/(kg/m^(2)).
 * @return On success `PUMAS_RETURN_SUCCESS` is returned otherwise an error
 * code is returned as detailed below.
 *
 * The energy loss straggling parameter, &Omega;<sub>s</sub>. This parameter
 * quantifies the fluctuations of the soft electronic energy loss in straggled
 * mode (`PUMAS_MODE_STRAGGLED`).
 *
 * **Note** : the energy loss straggling is applied to electronic collisions
 * only.
 *
 * The straggling per unit mass of the target is returned, in
 * GeV^(2)/(kg/m^(2)). Multiply by the target density in order to get the
 * straggling in unit of Gev^(2).
 *
 * __Error codes__
 *
 *     PUMAS_RETURN_INDEX_ERROR             The material index is not valid.
 *
 *     PUMAS_RETURN_PHYSICS_ERROR           The physics is not initialised.
 */
PUMAS_API enum pumas_return pumas_physics_property_energy_straggling(
    const struct pumas_physics * physics, int material, double energy,
    double * straggling);

/**
 * Cutoff angle for hard elastic collisions.
 *
 * @param physics     The physics tables.
 * @param material    The material index.
 * @param energy      The kinetic energy, in GeV.
 * @param angle       The corresponding angle, in rad.
 * @return On success `PUMAS_RETURN_SUCCESS` is returned otherwise an error
 * code is returned as detailed below.
 *
 * The cutoff angle is set from the physics *elastic\_ratio* following
 * [Fernandez-Varea et al.
 * (1993)](https://doi.org/10.1016/0168-583X(93)95827-R). It is computed at the
 * physics creation and cannot be modified afterwards.
 *
 * __Note__: the returned cutoff angle is defined in the center of mass frame of
 * the collision.
 *
 * __Error codes__
 *
 *     PUMAS_RETURN_INDEX_ERROR             The material index is not valid.
 *
 *     PUMAS_RETURN_PHYSICS_ERROR           The physics is not initialised.
 */
PUMAS_API enum pumas_return pumas_physics_property_elastic_cutoff_angle(
    const struct pumas_physics * physics, int material, double energy,
    double * angle);

/**
 * Mean free path for hard elastic collisions.
 *
 * @param physics     The physics tables.
 * @param material    The material index.
 * @param energy      The kinetic energy, in GeV.
 * @param length      The corresponding length, in kg/m^(2).
 * @return On success `PUMAS_RETURN_SUCCESS` is returned otherwise an error
 * code is returned as detailed below.
 *
 * The mean free path is restricted to hard elastic collisions with an angle
 * larger than a cutoff value, as returned by
 * `pumas_physics_property_elastic_cutoff_angle`. Soft collisions are included
 * in the multiple scattering (see `pumas_physics_property_transport_path`).
 *
 * The path per unit mass of the target is returned, in kg/m^(2). Divide by
 * the target density in order to get the path in unit of distance.
 *
 * __Error codes__
 *
 *     PUMAS_RETURN_INDEX_ERROR             The material index is not valid.
 *
 *     PUMAS_RETURN_PHYSICS_ERROR           The physics is not initialised.
 */
PUMAS_API enum pumas_return pumas_physics_property_elastic_path(
    const struct pumas_physics * physics, int material, double energy,
    double * length);

/**
 * Transport mean free path for soft collisions.
 *
 * @param physics     The physics tables.
 * @param mode        The energy loss mode.
 * @param material    The material index.
 * @param energy      The kinetic energy, in GeV.
 * @param path        The corresponding path per unit mass, in kg/m^(2).
 * @return On success `PUMAS_RETURN_SUCCESS` is returned otherwise an error
 * code is returned as detailed below.
 *
 * The transport m.f.p., &lambda;, is related to the standard deviation of the
 * polar multiple scattering angle's as &theta;^(2) = X/(2&lambda;), with X the
 * column depth.
 *
 * __Note__: the transport path includes all soft collisions, not only elastic
 * ones.
 *
 * The path per unit mass of the target is returned, in kg/m^(2). Divide by
 * the target density in order to get the path in unit of distance.
 *
 * __Error codes__
 *
 *     PUMAS_RETURN_INDEX_ERROR             The material index is not valid.
 *
 *     PUMAS_RETURN_PHYSICS_ERROR           The physics is not initialised.
 */
PUMAS_API enum pumas_return pumas_physics_property_transport_path(
    const struct pumas_physics * physics, enum pumas_mode mode, int material,
    double energy, double * path);

/**
 * Cross-section for hard collisions.
 *
 * @param physics          The physics tables.
 * @param material         The material index.
 * @param energy           The kinetic energy, in GeV.
 * @param cross_section    The computed cross-section value.
 * @return On success `PUMAS_RETURN_SUCCESS` is returned otherwise an error
 * code is returned as detailed below.
 *
 * The returned cross-section is restricted hard collisions with fractionnal
 * energy loss larger than the physics *cutoff*. Collisions with smaller energy
 * loss are included in the continuous energy loss given by
 * `pumas_physics_property_stopping_power`.
 *
 * __Note__: hard elastic collisions are not included in the cross-section but
 * in the elastic mean free path given by the 
 * `pumas_physics_property_elastic_path` function.
 *
 * The macroscopic cross-section is returned in unit m^(2)/kg. Multiply by the
 * target density in order to get the inverse of the interaction length in unit
 * of distance.
 *
 * __Error codes__
 *
 *     PUMAS_RETURN_INDEX_ERROR             The material index is not valid.
 *
 *     PUMAS_RETURN_PHYSICS_ERROR           The physics is not initialised.
 */
PUMAS_API enum pumas_return pumas_physics_property_cross_section(
    const struct pumas_physics * physics, int material, double energy,
    double * cross_section);

/**
 * The total number of atomic elements.
 *
 * @param physics    The physics tables.
 * @return The total number of atomic elements for the physics.
 */
PUMAS_API int pumas_physics_element_length(
    const struct pumas_physics * physics);

/**
 * Get the properties of an atomic element.
 *
 * @param physics    The physics tables.
 * @param index      The element index.
 * @param Z          The element charge number.
 * @param A          The element mass number in g/mol.
 * @param I          The element mean excitation energy in GeV.
 * @return On success `PUMAS_RETURN_SUCCESS` is returned otherwise an error
 * code is returned as detailed below.
 *
 * Get the properties of an atomic element. The *Z*, *A* or *I* pointers can be
 * `NULL` in which case the corresponding property is not retrieved.
 *
 * __Error codes__
 *
 *     PUMAS_RETURN_INDEX_ERROR               The provided index is not valid.
 *
 *     PUMAS_RETURN_PHYSICS_ERROR             The physics is not initialised.
 */
PUMAS_API enum pumas_return pumas_physics_element_properties(
    const struct pumas_physics * physics, int index, double * Z, double * A,
    double * I);

/**
 * The name of an atomic element given its index.
 *
 * @param physics    The physics tables.
 * @param index      The atomic element index.
 * @param element    The corresponding element name.
 * @return On success `PUMAS_RETURN_SUCCESS` is returned otherwise an error
 * code is returned as detailed below.
 *
 * The atomic element name is defined in the Material Description File (MDF).
 *
 * See the `pumas_physics_element_index` for the converse function, i.e. getting
 * an element index given its name.
 *
 * __Error codes__
 *
 *     PUMAS_RETURN_INDEX_ERROR               The provided index is not valid.
 *
 *     PUMAS_RETURN_PHYSICS_ERROR             The physics is not initialised.
 */
PUMAS_API enum pumas_return pumas_physics_element_name(
    const struct pumas_physics * physics, int index, const char ** element);

/**
 * The index of an atomic element given its name.
 *
 * @param physics     The physics tables.
 * @param element     The element name.
 * @param index       The corresponding index.
 * @return On success `PUMAS_RETURN_SUCCESS` is returned otherwise an error
 * code is returned as detailed below.
 *
 * The element index is given by its order of appeareance in the Material
 * Description File (MDF).
 *
 * See the `pumas_physics_element_name` for the converse function, i.e. getting
 * an element name given its index.
 *
 * __Error codes__
 *
 *     PUMAS_RETURN_PHYSICS_ERROR             The physics is not initialised.
 *
 *     PUMAS_RETURN_UNKNOWN_MATERIAL          The material is not defined.
 */
PUMAS_API enum pumas_return pumas_physics_element_index(
    const struct pumas_physics * physics, const char * element, int * index);

/**
 * The name of a material given its index.
 *
 * @param physics    The physics tables.
 * @param index      The material index.
 * @param material   The corresponding material name.
 * @return On success `PUMAS_RETURN_SUCCESS` is returned otherwise an error
 * code is returned as detailed below.
 *
 * The material name is defined in the Material Description File (MDF).
 *
 * __Note__: this function can be used for both base materials and composite
 * ones.
 *
 * See the `pumas_physics_material_index` for the converse function, i.e.
 * getting a material index given its name.
 *
 * __Error codes__
 *
 *     PUMAS_RETURN_INDEX_ERROR               The provided index is not valid.
 *
 *     PUMAS_RETURN_PHYSICS_ERROR             The physics is not initialised.
 */
PUMAS_API enum pumas_return pumas_physics_material_name(
    const struct pumas_physics * physics, int index, const char ** material);

/**
 * The index of a material given its name.
 *
 * @param physics     The physics tables.
 * @param material    The material name.
 * @param index       The corresponding index.
 * @return On success `PUMAS_RETURN_SUCCESS` is returned otherwise an error
 * code is returned as detailed below.
 *
 * The material index is given by its order of appeareance in the Material
 * Description File (MDF).
 *
 * __Note__: this function can be used for both base materials and composite
 * ones.
 *
 * See the `pumas_physics_material_name` for the converse function, i.e. getting
 * a material name given its index.
 *
 * __Error codes__
 *
 *     PUMAS_RETURN_PHYSICS_ERROR             The physics is not initialised.
 *
 *     PUMAS_RETURN_UNKNOWN_MATERIAL          The material is not defined.
 */
PUMAS_API enum pumas_return pumas_physics_material_index(
    const struct pumas_physics * physics, const char * material, int * index);

/**
 * The total number of materials.
 *
 * __Note__: this function returns the sum of the numbers of base materials and
 * of composite ones.
 *
 * @param physics    The physics tables.
 * @return The total number of known materials, base plus composite.
 */
PUMAS_API int pumas_physics_material_length(
    const struct pumas_physics * physics);

/**
 * Get the properties of a material.
 *
 * @param physics           The physics tables.
 * @param index             The material index.
 * @param length            The number of atomic elements.
 * @param density           The material reference density in kg/m^(3).
 * @param I                 The material mean excitation energy in GeV.
 * @param components        The vector of indices of the atomic elements.
 * @param fractions         The vector of mass fractions of the atomic elements.
 * @return On success `PUMAS_RETURN_SUCCESS` is returned otherwise an error
 * code is returned as detailed below.
 *
 * Get the properties of a base material. `length`, `density`, `I`, `components`
 * or `fractions` can be `NULL` in which case the corresponding property is not
 * retrieved.
 *
 * __Error codes__
 *
 *     PUMAS_RETURN_INDEX_ERROR               The provided index is not valid.
 *
 *     PUMAS_RETURN_PHYSICS_ERROR             The physics is not initialised.
 */
PUMAS_API enum pumas_return pumas_physics_material_properties(
    const struct pumas_physics * physics, int index, int * length,
    double * density, double * I, int * components, double * fractions);

/**
 * The number of composite materials.
 *
 * @param physics    The physics tables.
 * @return The number of composite materials.
 */
PUMAS_API int pumas_physics_composite_length(
    const struct pumas_physics * physics);

/**
 * Update the properties of a composite material.
 *
 * @param physics    The physics tables.
 * @param material   The composite material index.
 * @param fractions  The mass fractions of the constitutive base materials.
 * @return On success `PUMAS_RETURN_SUCCESS` is returned otherwise an error
 * code is returned as detailed below.
 *
 * Update the composition of a composite material, i.e the mass fractions of its
 * components.
 *
 * **Note**: the provided mass fraction values are normalised to one by PUMAS.
 * Thus, they can be given e.g. in percent. Negative values are treated as zero.
 *
 * __Error codes__
 *
 *     PUMAS_RETURN_VALUE_ERROR               The fractions pointer is NULL.
 *
 *     PUMAS_RETURN_INDEX_ERROR               The provided index is not valid.
 *
 *     PUMAS_RETURN_PHYSICS_ERROR             The physics is not initialised.
 */
PUMAS_API enum pumas_return pumas_physics_composite_update(
    struct pumas_physics * physics, int material, const double * fractions);

/**
 * Get the properties of a composite material.
 *
 * @param physics    The physics tables.
 * @param index      The composite material index.
 * @param length     The number of base materials componsing the composite.
 * @param components The indices of the base materials.
 * @param fractions  The mass fractions of the constitutive base materials.
 * @return On success `PUMAS_RETURN_SUCCESS` is returned otherwise an error
 * code is returned as detailed below.
 *
 * Get the properties of a composite material. The *length*, *components* or
 * *fractions* pointers can be `NULL` in which case the corresponding property
 * is not retrieved.
 *
 * __Error codes__
 *
 *     PUMAS_RETURN_INDEX_ERROR               The provided index is not valid.
 *
 *     PUMAS_RETURN_PHYSICS_ERROR             The physics is not initialised.
 */
PUMAS_API enum pumas_return pumas_physics_composite_properties(
    const struct pumas_physics * physics, int index, int * length,
    int * components, double * fractions);

/**
 * Get tabulated physics values.
 *
 * @param physics     The physics tables.
 * @param property    The column index of a property of interest.
 * @param mode        The energy loss mode, i.e. `PUMAS_MODE_CSDA` or
 *                      `PUMAS_MODE_MIXED`.
 * @param material    The material index.
 * @param row         The row index in the table.
 * @param value       The corresponding table value.
 * @return On success `PUMAS_RETURN_SUCCESS` is returned otherwise an error
 * code is returned as detailed below.
 *
 * This function returns the tabulated value for a given *property* and *row*
 * index. See `pumas_property` for the list of tabulated physics properties. The
 * rows of the table map to different kinetic energy values specified when
 * creating the `pumas_physics` object.  Those can be retrieved with the
 * `PUMAS_PROPERTY_KINETIC_ENERGY` property. The `pumas_physics_table_length`
 * returns the number of tabulated kinetic energy values.
 *
 * Except for the kinetic energy a *material* must be selected. In addition,
 * the energy loss *mode* must be specified for related properties, e.g. for
 * the stopping power (`PUMAS_PROPERTY_STOPPING_POWER`).
 *
 * __Note__: a negative row index can be provided in which case it refers to
 * the end of the table. E.g. `row = -1` is the last entry and `row = -2` is
 * the before last one.
 *
 * __Error codes__
 *
 *     PUMAS_RETURN_INDEX_ERROR             Some input index is not valid
 * (property, material or mode).
 *
 *     PUMAS_RETURN_PHYSICS_ERROR           The physics is not initialised.
 */
PUMAS_API enum pumas_return pumas_physics_table_value(
    const struct pumas_physics * physics, enum pumas_property property,
    enum pumas_mode mode, int material, int row, double * value);

/**
 * The number of rows in physics tabulations.
 *
 * @param physics    The physics tables.
 * @return The number of rows.
 *
 * Physics properties are tabulated as function of the projectile kinetic
 * energy. This function returns the number of tabulated kinetic energies, i.e.
 * the number of rows in physics tables.
 */
PUMAS_API int pumas_physics_table_length(const struct pumas_physics * physics);

/**
 * Compute the table row index for a given property and value.
 *
 * @param physics     The physics tables.
 * @param property    The column index of the property.
 * @param mode        The energy loss mode.
 * @param material    The material index.
 * @param value       The property value.
 * @param index       The row index from below for the given value.
 * @return On success `PUMAS_RETURN_SUCCESS` is returned otherwise an error
 * code is returned as detailed below.
 *
 * This function is the converse of the `pumas_physics_property_value`. It
 * returns the row index corresponding to a given property and value.
 *
 * __Warning__: in the case of an out of bounds value the closest index value is
 * provided and `PUMAS_RETURN_VALUE_ERROR` is returned.
 *
 * __Note__: only monotone properties are supported, that is when there is at
 * most one solution for the inverse. Supported properties are:
 * `PUMAS_PROPERTY_RANGE`, `PUMAS_PROPERTY_KINETIC_ENERGY`,
 * `PUMAS_PROPERTY_MAGNETIC_ROTATION` and `PUMAS_PROPERTY_PROPER_TIME`.
 *
 * __Error codes__
 *
 *     PUMAS_RETURN_INDEX_ERROR             Some input index is not valid
 * (property, material or mode).
 *
 *     PUMAS_RETURN_PHYSICS_ERROR           The physics is not initialised.
 *
 *     PUMAS_RETURN_VALUE_ERROR             The provided value is out of the
 * table.
 */
PUMAS_API enum pumas_return pumas_physics_table_index(
    const struct pumas_physics * physics, enum pumas_property property,
    enum pumas_mode mode, int material, double value, int * index);

/**
 * Create a new Monte Carlo recorder.
 *
 * @param recorder     The Monte Carlo recorder.
 * @param extra_memory The size of the user extra memory if any is claimed.
 * @return On success `PUMAS_RETURN_SUCCESS` is returned otherwise an error
 * code is returned as detailed below.
 *
 * Create a new *recorder* object for Monte Carlo steps. The recorder starts
 * configured with a built-in algorithm recording all Monte Carlo steps as
 * `pumas_frame` objects. See the `pumas_recorder` structure for configuration
 * options and usage.
 *
 * If *extra_memory* is strictly positive the recorder is extended by
 * *extra_memory* bytes for user usage. This memory can then be accessed with
 * the *user_data* field of the returned `pumas_recorder` structure.
 *
 * __Error codes__
 *
 *     PUMAS_RETURN_MEMORY_ERROR    Could not allocate memory.
 */
PUMAS_API enum pumas_return pumas_recorder_create(
    struct pumas_recorder ** recorder, int extra_memory);

/**
 * Clear all recorded frames.
 *
 * @param recorder The recorder handle.
 *
 * Erase all recorded `pumas_frame` instances from the recorder and reset the
 * frame count.
 */
PUMAS_API void pumas_recorder_clear(struct pumas_recorder * recorder);

/**
 * Destroy a Monte Carlo recorder.
 *
 * @param recorder The recorder handle.
 *
 * Destroy a Monte Carlo recorder by releasing its associated memory, i.e. the
 * recorder is cleared before beeing destroyed.
 *
 * __Note__: at return `recorder` is set to `NULL`.
 *
 * __Warning__: if a user supplied *record* callback is used instead of the
 * built-in `pumas_frame` recorder then it is the user responsibility to
 * properly manage any self allocated memory.
 */
PUMAS_API void pumas_recorder_destroy(struct pumas_recorder ** recorder);

/**
 * User supplied callback for memory allocation.
 *
 * @param size    The number of memory bytes to allocate.
 * @return The address of the allocated memory or `NULL` in case of faillure.
 *
 * The provided callback must conform to the `malloc` semantic and behaviour.
 */
typedef void * pumas_allocate_cb (size_t size);

/**
 * Set the memory allocation function for the PUMAS library.
 *
 * @param allocator    The user supplied memory allocator, or `NULL`.
 *
 * This function allows to specify a custom memory allocation function for
 * PUMAS. Passing a `NULL` value results in PUMAS using its default allocator,
 * i.e. `malloc`.
 *
 * __Warning__: this function is **not** thread safe.
 */
PUMAS_API void pumas_memory_allocator(pumas_allocate_cb * allocator);

/**
 * User supplied callback for memory re-allocation.
 *
 * @param ptr     The address of the memory to reallocate.
 * @param size    The number of memory bytes requested for the reallocation.
 * @return The address of the re-allocated memory or `NULL` in case of faillure.
 *
 * The provided callback must conform to the `realloc` semantic and behaviour.
 */
typedef void * pumas_reallocate_cb (void * ptr, size_t size);

/**
 * Set the memory re-allocation function for the PUMAS library.
 *
 * @param reallocator    The user supplied memory reallocator, or `NULL`.
 *
 * This function allows to specify a custom memory re-allocation function for
 * PUMAS. Passing a `NULL` value results in PUMAS using its default
 * reallocator, i.e. `realloc`.
 *
 * __Warning__: this function is **not** thread safe.
 */
PUMAS_API void pumas_memory_reallocator(pumas_reallocate_cb * reallocator);

/**
 * User supplied callback for memory deallocation.
 *
 * @param size    The address of the memory to deallocate.
 *
 * The provided callback must conform to the `free` semantic and behaviour.
 */
typedef void pumas_deallocate_cb (void * ptr);

/**
 * Set the memory deallocation function for the PUMAS library.
 *
 * @param deallocator    The user supplied memory deallocator, or `NULL`.
 *
 * This function allows to specify a custom memory deallocation function for
 * PUMAS. Passing a `NULL` value results in PUMAS using its default
 * deallocator, i.e. `free`.
 *
 * __Warning__: this function is **not** thread safe.
 */
PUMAS_API void pumas_memory_deallocator(pumas_deallocate_cb * deallocator);

/**
 * Get the Differential Cross-Section (DCS) used by the physics.
 *
 * @param physics      The physics tables.
 * @param process      The physics process.
 * @param model        The corresponding DCS model.
 * @param dcs          The corresponding DCS function.
 * @return On success `PUMAS_RETURN_SUCCESS` is returned otherwise an error
 * code is returned as detailed below.
 *
 * This function provides the DCSs for radiative processes used by a *physics*
 * instance. See the `pumas_dcs_get` function in order to get the DCS for a
 * specific model of radiative process.
 *
 *
 * The *dcs* and *model* return values are optionnal. If a `NULL` pointer is
 * provided then the corresponding return value is not filled.
 *
 * **Note**: the DCS models for radiative processes are set at the physics
 * creation and cannot be changed afterwards. Elastic and electronic collisions
 * use fixed models. The corresponding DCSs are given by the `pumas_elastic_dcs`
 * and `pumas_electronic_dcs` functions.
 *
 * __Error codes__
 *
 *     PUMAS_RETURN_INDEX_ERROR             The process index is not a valid.
 *
 *     PUMAS_RETURN_PHYSICS_ERROR           The physics is not initialised.
 */
PUMAS_API enum pumas_return pumas_physics_dcs(
    const struct pumas_physics * physics, enum pumas_process process,
    const char ** model, pumas_dcs_t ** dcs);

/**
 * Get a PUMAS library constant.
 *
 * @param index     The constant index.
 * @param value     The corresponding value.
 * @return On success `PUMAS_RETURN_SUCCESS` is returned otherwise an error
 * code is returned as detailed below.
 *
 * This function allows to retrieve the numeric values of physics constants used
 * in PUMAS. See the `pumas_constant` enum for a list of available constants.
 *
 * __Note__: values are returned in PUMAS system of units, i.e. GeV, m, etc.
 *
 * __Error codes__
 *
 *     PUMAS_RETURN_INDEX_ERROR             The index is not a valid.
 *
 *     PUMAS_RETURN_VALUE_ERROR             The value pointer is NULL.
 */
PUMAS_API enum pumas_return pumas_constant(
    enum pumas_constant index, double * value);

/**
 * Register a Differential Cross Section (DCS) model to PUMAS.
 *
 * @param process   The physics process index.
 * @param model     The model name.
 * @param dcs       The model DCS.
 * @return On success `PUMAS_RETURN_SUCCESS` is returned otherwise an error
 * code is returned as detailed below.
 *
 * This function allows to register a DCS model for a radiative *process*. Note
 * that the *model* name must not be already used otherwise an error is
 * returned.  Only the following processes can be user defined: Bremsstrahlung,
 * e^(+)e^(-) pair production and photonuclear interactions.  Electronic and
 * elastic collisions are built-in.
 *
 * __Note__: it is not possible to un-register a model.
 *
 * __Warning__: this function is **not** thread safe.
 *
 * __Error codes__
 *
 *     PUMAS_RETURN_INDEX_ERROR             The process index is not a valid.
 *
 *     PUMAS_RETURN_MEMORY_ERROR            The maximum number of models was
 * reached.
 *
 *     PUMAS_RETURN_MODEL_ERROR             The model name is already used.
 *
 *     PUMAS_RETURN_VALUE_ERROR             A NULL model or dcs was provided.
 */
PUMAS_API enum pumas_return pumas_dcs_register(
    enum pumas_process process, const char * model, pumas_dcs_t * dcs);

/**
 * Differential Cross Section (DCS) for a given model.
 *
 * @param process   The physics process index.
 * @param model     The model name.
 * @param dcs       The corresponding DCS.
 * @return On success `PUMAS_RETURN_SUCCESS` is returned otherwise an error
 * code is returned as detailed below.
 *
 * This function allows to retrieve the DCS for a given physics *process* and
 * *model*. See the `pumas_physics_settings` structure for a list of models
 * available by default. Extra models can be registered with the
 * `pumas_dcs_register` function.
 *
 * __Warning__: this function is **not** thread safe.
 *
 * __Error codes__
 *
 *     PUMAS_RETURN_INDEX_ERROR             The process index is not a valid.
 *
 *     PUMAS_RETURN_MODEL_ERROR             The model name is not valid.
 */
PUMAS_API enum pumas_return pumas_dcs_get(
    enum pumas_process process, const char * model, pumas_dcs_t ** dcs);

/**
 * Default Differential Cross Section (DCS) model.
 *
 * @param process   The physics process index.
 * @return On success the default model name is returned otherwise `NULL`.
 *
 * Get the name of the default DCS model for a given *process*.  If the
 * requested process index is not valid then `NULL` is returned. See the
 * `pumas_dcs_get` function in order to get the DCS for a given model.
 */
PUMAS_API const char * pumas_dcs_default(enum pumas_process process);

/**
 * Energy range for a given radiative process.
 *
 * @param process   The physics process index.
 * @param Z         The target atomic number.
 * @param mass      The projectile rest mass, in GeV.
 * @param energy    The projectile kinetic energy, in GeV.
 * @param min       The minimum allowed energy transfer, in GeV, or `NULL`.
 * @param min       The maximum allowed energy transfer, in GeV, or `NULL`.
 * @return On success `PUMAS_RETURN_SUCCESS` is returned otherwise an error
 * code is returned as detailed below.
 *
 * This function provides the range of valid energy transfers for the
 * Differential Cross Section (DCS) of a given radiative *process*.
 *
 * __Error codes__
 *
 *     PUMAS_RETURN_INDEX_ERROR             The process index is not a valid.
 */
PUMAS_API enum pumas_return pumas_dcs_range(enum pumas_process process,
    double Z, double mass, double energy, double * min, double * max);

/**
 * The differential cross section (DCS) for elastic collisions.
 *
 * @param Z       The charge number of the target atom.
 * @param A       The mass number of the target atom.
 * @param m       The projectile rest mass, in GeV
 * @param K       The projectile initial kinetic energy.
 * @param theta   The scattering angle, in rad.
 * @return The corresponding value of the atomic DCS, in m^(2) / rad.
 *
 * The elastic DCS is computed following [Salvat
 * (2013)](https://doi.org/10.1016/j.nimb.2013.08.035) and [Boschini et al.
 * (2014)](https://arxiv.org/abs/1011.4822). The first Born approximation is
 * used with coulomb corrections from [Kuraev et al.
 * (2014)](https://doi.org/10.1103/PhysRevD.89.116016). The target recoil is
 * taken into account with an effective projectile mass.
 */
PUMAS_API double pumas_elastic_dcs(
    double Z, double A, double m, double K, double theta);

/**
 * The (transport) mean free path for elastic collisions.
 *
 * @param order   The order of the distribution.
 * @param Z       The charge number of the target atom.
 * @param A       The mass number of the target atom.
 * @param m       The projectile rest mass, in GeV
 * @param K       The projectile initial kinetic energy.
 * @param theta   The scattering angle, in rad.
 * @return The corresponding path per unit mass, in kg / m^(2).
 *
 * The m.f.p. is computed analytically by integration of the elastic DCS (see
 * `pumas_elastic_dcs`). If *order* is 0 then the single collision m.f.p. is
 * returned. Else, if *order* is 1 then the transport m.f.p. is returned. For
 * other values of *order* -1 is returned.
 */
PUMAS_API double pumas_elastic_path(
    int order, double Z, double A, double mass, double kinetic);

/**
 * The electronic differential cross section restricted to close collisions.
 *
 * @param Z       The charge number of the target atom.
 * @param I       The mean excitation energy of the target atom.
 * @param m       The projectile rest mass, in GeV
 * @param K       The projectile initial kinetic energy.
 * @param q       The projectile energy loss, in GeV.
 * @return The corresponding value of the atomic DCS, in m^(2) / GeV.
 *
 * The electronic DCS restricted to close collisions is computed following
 * [Salvat (2013)](https://doi.org/10.1016/j.nimb.2013.08.035). An effective
 * model is used with a cutoff set as a fraction of the mean excitation energy,
 * I. This reproduces Salvat for energy losses, *q*, larger than the
 * electrons binding energies.
 */
PUMAS_API double pumas_electronic_dcs(
    double Z, double I, double m, double K, double q);

/**
 * The electronic density effect for a material.
 *
 * @param n_elements    The number of atomic elements in the material.
 * @param Z             The charge numbers of the constitutive atomic elements.
 * @param A             The mass numbers of the constitutive atomic elements.
 * @param w             The mass fractions of the atomic elements, or `NULL`.
 * @param I             The mean excitation energy of the material, in GeV.
 * @param density       The density of the material, in kg / m^(3).
 * @param gamma         The relativistic gamma factor of the projectile.
 * @return The corresponding density effect.
 *
 * The density effect is computed following [Fano
 * (1963)](https://doi.org/10.1146/annurev.ns.13.120163.000245). Oscillators
 * strength and level have been set from electrons binding energies of
 * individual atomic elements. A global scaling factor is applied in order to
 * match the Mean Excitation Energy.
 *
 * The mass fractions of the elements, *w*, can be `NULL` in wich case they are
 * assumed to be 1. The mass fractions do not need to be normalised to 1.
 */
PUMAS_API double pumas_electronic_density_effect(int n_elements,
    const double * Z, const double * A, const double * w, double I,
    double density, double gamma);

/**
 * The stopping power due to collisions with atomic electrons.
 *
 * @param n_elements    The number of atomic elements in the material.
 * @param Z             The charge numbers of the constitutive atomic elements.
 * @param A             The mass numbers of the constitutive atomic elements.
 * @param w             The mass fractions of the atomic elements, or `NULL`.
 * @param I             The mean excitation energy of the material, in GeV.
 * @param density       The density of the material, in kg / m^(3).
 * @param mass          The mass of the projectile, in GeV / c^(2).
 * @param energy        The energy of the projectile, in GeV
 * @return The corresponding stopping power per unit mass, in GeV m^(2) / kg.
 *
 * The electronic stopping power is computed following [Salvat
 * (2013)](https://doi.org/10.1016/j.nimb.2013.08.035). The result is identical
 * to [Groom et al. (2001)](https://doi.org/10.1006/adnd.2001.0861) except for
 * the density effect. The latter is computed following [Fano
 * (1963)](https://doi.org/10.1146/annurev.ns.13.120163.000245), see e.g.
 * `pumas_electronic_density_effect`.
 *
 * The mass fractions of the elements, *w*, can be `NULL` in wich case they are
 * assumed to be 1. The mass fractions do not need to be normalised to 1.
 */
PUMAS_API double pumas_electronic_stopping_power(int n_elements,
    const double * Z, const double * A, const double * w, double I,
    double density, double mass, double energy);

#ifdef __cplusplus
}
#endif
#endif
