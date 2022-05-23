#pragma once

#include <pybind11/pybind11.h>
namespace py = pybind11;

#include "RawIterator.h"

template< typename Array, typename Scope >
void tnl_indexing( Scope & scope )
{
    using Index = typename Array::IndexType;
    using Value = typename Array::ValueType;

    scope.def("__len__", &Array::getSize);

    scope.def("__iter__",
        []( Array& array ) {
            return py::make_iterator(
                        RawIterator<Value>(array.getData()),
                        RawIterator<Value>(array.getData() + array.getSize()) );
        },
        py::keep_alive<0, 1>()  // keep array alive while iterator is used
    );

    scope.def("__getitem__",
        [](Array &a, Index i) {
            if (i >= a.getSize())
                throw py::index_error();
            return a[i];
        }
    );

    scope.def("__setitem__",
        [](Array &a, Index i, const Value& e) {
            if (i >= a.getSize())
                throw py::index_error();
            a[i] = e;
        }
    );
}

template< typename Array, typename Scope >
void tnl_slice_indexing( Scope & scope )
{
    /// Slicing protocol
    scope.def("__getitem__",
        [](const Array& a, py::slice slice) -> Array* {
            size_t start, stop, step, slicelength;

            if (!slice.compute(a.getSize(), &start, &stop, &step, &slicelength))
                throw py::error_already_set();

            Array* seq = new Array();
            seq->setSize(slicelength);

            for (size_t i = 0; i < slicelength; ++i) {
                seq->operator[](i) = a[start];
                start += step;
            }
            return seq;
        },
        "Retrieve list elements using a slice object"
    );

    scope.def("__setitem__",
        [](Array& a, py::slice slice,  const Array& value) {
            size_t start, stop, step, slicelength;
            if (!slice.compute(a.getSize(), &start, &stop, &step, &slicelength))
                throw py::error_already_set();

            if (slicelength != (size_t) value.getSize())
                throw std::runtime_error("Left and right hand size of slice assignment have different sizes!");

            for (size_t i = 0; i < slicelength; ++i) {
                a[start] = value[i];
                start += step;
            }
        },
        "Assign list elements using a slice object"
    );
}
