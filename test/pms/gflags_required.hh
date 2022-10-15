#pragma once

constexpr double doubleNaN = std::numeric_limits<double>::quiet_NaN();

inline bool validateRequiredDoubleFlag(const char* flagname, double value) {
	if (std::isnan(value)) {
		std::cerr << "Flag --" << flagname << " is not optional. Please specify it." << std::endl;
		return false;
	}
	return true;
}

#define DEFINE_double_required(name, help)	DEFINE_double(name, doubleNaN, help);\
						DEFINE_validator(name, validateRequiredDoubleFlag)
