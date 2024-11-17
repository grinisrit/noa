/**
 * \file except.hh
 * \brief NOA execeptions in MHFEM
 */

#pragma once

// Standard library
#include <stdexcept>

/// \brief A namespace for exceptions occuring in NOA
namespace noa::test::exceptions {

/// Gets thrown when a solver is met with an empty setup
struct empty_setup : public std::runtime_error {
	empty_setup() : std::runtime_error("Empty setup") {}
};

/// Gets thrown when a some conditions expected by a solver weren't met
struct invalid_setup : public std::runtime_error {
        invalid_setup() : std::runtime_error("Invalid or missing setup/initial conditions!") {}
};

} // <-- namespace noa::test::exceptions
