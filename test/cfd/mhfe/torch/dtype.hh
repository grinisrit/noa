/**
 * \file dtype.hh
 * \brief Conversion between C++ types and Torch dtypes
 */

#pragma once

namespace noa::test::utils::pytorch {

/// \brief Get torch real dtype
template <std::size_t sz> struct TDTReal {
    static_assert((sz < 0), "Not Torch dtype for this real type!");
};

template <> struct TDTReal<2> { static constexpr auto value = torch::kFloat16; };
template <> struct TDTReal<4> { static constexpr auto value = torch::kFloat32; };
template <> struct TDTReal<8> { static constexpr auto value = torch::kFloat64; };

/// \brief Alias for TDTReal<sizeof(T)>::value
template <typename T>
constexpr auto tdtReal = TDTReal<sizeof(T)>::value;

/// \brief Get torch int dtype
template <std::size_t sz> struct TDTInt {
    static_assert((sz < 0), "Not Torch dtype for this integer type!");
};

template <> struct TDTInt<1> { static constexpr auto value = torch::kInt8; };
template <> struct TDTInt<2> { static constexpr auto value = torch::kInt16; };
template <> struct TDTInt<4> { static constexpr auto value = torch::kInt32; };
template <> struct TDTInt<8> { static constexpr auto value = torch::kInt64; };

/// \brief Alias for TDTInt<sizeof(T)>::value
template <typename T>
constexpr auto tdtInt = TDTInt<sizeof(T)>::value;

} // <-- namespace noa::test::utils::pytorch
