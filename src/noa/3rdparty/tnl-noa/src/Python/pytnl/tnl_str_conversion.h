#pragma once

#include <pybind11/pybind11.h>
namespace py = pybind11;

#include <TNL/String.h>

namespace pybind11 { namespace detail {

    template <>
    struct type_caster<TNL::String>
    {
        using StdStringCaster = type_caster<std::string>;
        StdStringCaster _caster;

    public:
        /**
         * This macro establishes the name 'TNL::String' in
         * function signatures and declares a local variable
         * 'value' of type TNL::String
         */
        PYBIND11_TYPE_CASTER(TNL::String, _("TNL::String"));

        /**
         * Conversion part 1 (Python->C++): convert a PyObject into a TNL::String
         * instance or return false upon failure. The second argument
         * indicates whether implicit conversions should be applied.
         */
        bool load(handle src, bool implicit)
        {
            if( ! _caster.load(src, implicit) )
                return false;
            const std::string& str = (std::string&) _caster;
            value = TNL::String(str.c_str());
            return true;
        }

        /**
         * Conversion part 2 (C++ -> Python): convert an TNL::String instance into
         * a Python object. The second and third arguments are used to
         * indicate the return value policy and parent object (for
         * ``return_value_policy::reference_internal``) and are generally
         * ignored by implicit casters.
         */
        static handle cast(TNL::String src, return_value_policy policy, handle parent)
        {
            return StdStringCaster::cast( src.getString(), policy, parent );
        }
    };

}} // namespace pybind11::detail
