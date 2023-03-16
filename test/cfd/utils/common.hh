/**
 * \file common.hh
 * \brief Candidates for inclusion in src/noa/utils/common.hh
 *
 * Implemented by: Gregory Dushkin
 */
#pragma once

// Standard library
#include <iostream>

namespace noa::utils::test {

/// \brief An empty type
///
/// Used as a placeholder for 'unimplemented' fields in
/// classes as a third argument for `std::conditional`
struct Empty {
	constexpr Empty() noexcept = default;
	template <typename T>
	Empty(T) = delete;
	template <typename T>
	Empty& operator=(T) = delete;
};

/// \brief A temporary ostream wrapper. Performs Handler::exit(stream)
template <typename StreamT, typename Handler>
class EndingOstreamWrapper {
	StreamT& stream;
public:
	EndingOstreamWrapper(StreamT& _stream, Handler handler) : stream(_stream) {}
	EndingOstreamWrapper(const EndingOstreamWrapper&_) = delete;
	EndingOstreamWrapper(EndingOstreamWrapper&&) = delete;
	EndingOstreamWrapper& operator=(const EndingOstreamWrapper&) = delete;
	EndingOstreamWrapper& operator=(EndingOstreamWrapper&&) = delete;
	~EndingOstreamWrapper() noexcept { Handler::exit(this->stream); }

	template <typename T>
	EndingOstreamWrapper& operator<<(T arg) { this->stream << arg; return *this; }
};

/// \brief Automatically flush the stream at the end of output
inline struct AutoFlush {
	static void exit(std::ostream& stream) { stream.flush(); }
} autoflush;

/// \brief Replace current terminal line message to argument
inline struct StatusLine : AutoFlush {} status_line;

/// Do nothing at the start of autoflush output
inline auto operator<<(std::ostream& stream, AutoFlush) {
	return EndingOstreamWrapper(stream, autoflush);
}

/// \brief Printing status_line to std::ostream results its line being cleared via VT100 code
///
/// Caret is also returned to the start of the line. End of output behavior inherited from AutoFlush
/// (flushes the stream)
inline auto operator<<(std::ostream& stream, StatusLine) {
	stream << "\033[2K\r";
	return EndingOstreamWrapper(stream, status_line);
}

} // <-- namespace noa::test::utils
