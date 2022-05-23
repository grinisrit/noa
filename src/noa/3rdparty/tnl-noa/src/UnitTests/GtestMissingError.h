#pragma once

#include <stdexcept>

struct GtestMissingError
   : public std::runtime_error
{
   GtestMissingError()
   : std::runtime_error( "The GTest library is needed to run the tests." )
   {}
};
