#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
namespace py = pybind11;

#include <TNL/Containers/StaticArray.h>
#include <TNL/Containers/StaticVector.h>

namespace pybind11 { namespace detail {

    template< typename ArrayType >
    struct _tnl_tuple_caster
    {
        using Value = typename std::remove_reference< decltype(ArrayType()[0]) >::type;
        using StdArray = std::array< Value, ArrayType::getSize() >;
        using StdArrayCaster = type_caster< StdArray >;
//        StdArrayCaster _caster;
		using value_conv = make_caster<Value>;

    public:
//        PYBIND11_TYPE_CASTER(ArrayType, StdArrayCaster::name);
        PYBIND11_TYPE_CASTER(ArrayType, _("Tuple[") + value_conv::name + _<false>(_(""), _("[") + _<ArrayType::getSize()>() + _("]")) + _("]"));

        /**
         * Conversion part 1 (Python -> C++): convert a PyObject into an ArrayType
         * instance or return false upon failure. The second argument indicates
         * whether implicit conversions should be applied.
         */
        bool load(handle src, bool implicit)
        {
            // we don't use StdArrayCaster here because we want to convert Python tuples, not lists
//            if( ! _caster.load(src, implicit) )
//                return false;
//            const StdArray& arr = (StdArray&) _caster;
//            for( int i = 0; i < ArrayType::getSize(); i++ )
//                value[ i ] = arr[ i ];
//            return true;

            if (!isinstance<tuple>(src))
               return false;
            auto t = reinterpret_borrow<tuple>(src);
            if (t.size() != (std::size_t) ArrayType::getSize())
               return false;
            std::size_t ctr = 0;
            for (auto it : t) {
               value_conv conv;
               if (!conv.load(it, implicit))
                  return false;
               value[ctr++] = cast_op<Value &&>(std::move(conv));
            }
            return true;
        }

        /**
         * Conversion part 2 (C++ -> Python): convert an ArrayType instance into
         * a Python object. The second and third arguments are used to
         * indicate the return value policy and parent object (for
         * ``return_value_policy::reference_internal``) and are generally
         * ignored by implicit casters.
         */
        static handle cast(const ArrayType& src, return_value_policy policy, handle parent)
        {
            StdArray arr;
            for( int i = 0; i < ArrayType::getSize(); i++ )
                arr[ i ] = src[ i ];
            return StdArrayCaster::cast( arr, policy, parent );
        }
    };

    template< typename T, int Size >
    struct type_caster< TNL::Containers::StaticArray< Size, T > >
        : _tnl_tuple_caster< TNL::Containers::StaticArray< Size, T > > {};

    template< typename T, int Size >
    struct type_caster< TNL::Containers::StaticVector< Size, T > >
        : _tnl_tuple_caster< TNL::Containers::StaticVector< Size, T > > {};

}} // namespace pybind11::detail
