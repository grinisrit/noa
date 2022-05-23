#pragma once

#include <cctbx/pystreambuf.h>

namespace pybind11 { namespace detail {
    template <> struct type_caster<std::istream> {
    public:
        bool load(handle src, bool) {
            if (getattr(src, "read", none()).is_none()){
              return false;
            }

            obj = reinterpret_borrow<object>(src);
            value = std::unique_ptr<pystreambuf::istream>(new pystreambuf::istream(obj, 0));

            return true;
        }

    protected:
        object obj;
        std::unique_ptr<pystreambuf::istream> value;

    public:
        static constexpr auto name = _("istream");
        static handle cast(const std::istream *src, return_value_policy policy, handle parent) {
            return none().release();
        }
        operator std::istream*() { return value.get(); }
        operator std::istream&() { return *value; }
        template <typename _T> using cast_op_type = pybind11::detail::cast_op_type<_T>;
    };

    template <> struct type_caster<std::ostream> {
    public:
        bool load(handle src, bool) {
            if (getattr(src, "write", none()).is_none()){
              return false;
            }

            obj = reinterpret_borrow<object>(src);
            value = std::unique_ptr<pystreambuf::ostream>(new pystreambuf::ostream(obj, 0));

            return true;
        }

    protected:
        object obj;
        std::unique_ptr<pystreambuf::ostream> value;

    public:
        static constexpr auto name = _("ostream");
        static handle cast(const std::ostream *src, return_value_policy policy, handle parent) {
            return none().release();
        }
        operator std::ostream*() { return value.get(); }
        operator std::ostream&() { return *value; }
        template <typename _T> using cast_op_type = pybind11::detail::cast_op_type<_T>;
    };
}} // namespace pybind11::detail
