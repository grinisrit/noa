/**
 * \file wrappers.hh
 * \brief Defines wrappers used with PyBind to expose NOA classes
 */

#pragma once

/// \brief Namespace containing utility code for providing
/// PyBind/Torch extensions in NOA
namespace noa::test::utils::pytorch {

/// \brief Non-owning pointer wrapper
template <typename T>
class WeakWrapper {
	T* p = nullptr;

public:
	WeakWrapper(T* dp) : p(dp) {}
	T* operator->() { return p; }
	const T* operator->() const { return p; }
	T& get() { return *p; }
	const T& get() const { return *p; }
}; // <-- class WeakWrapper

/// \brief Non-owning pointer to a const wrapper
template <typename T>
class ConstWeakWrapper {
	const T* p = nullptr;

public:
	ConstWeakWrapper(const T* dp) : p(dp) {}
	const T* operator->() const { return p; }
	const T& get() const { return *p; }
}; // <-- class ConstWeakWrapper

} // <-- namespace noa::test::utils::pytorch
