#pragma once

#include <pybind11/pybind11.h>
namespace py = pybind11;

#include "../typedefs.h"

template< typename Grid >
struct SpaceStepsProductsGetter
{};

template<>
struct SpaceStepsProductsGetter< Grid1D >
{
    static inline
    typename Grid1D::RealType
    get( const Grid1D & grid, const int & xPow )
    {
        if( xPow == -2 ) return grid.template getSpaceStepsProducts< -2 >();
        if( xPow == -1 ) return grid.template getSpaceStepsProducts< -1 >();
        if( xPow ==  0 ) return grid.template getSpaceStepsProducts<  0 >();
        if( xPow ==  1 ) return grid.template getSpaceStepsProducts<  1 >();
        if( xPow ==  2 ) return grid.template getSpaceStepsProducts<  2 >();
        const auto hx = grid.template getSpaceStepsProducts< 1 >();
        return pow( hx, xPow );
    }

    template< typename PyGrid >
    static void
    export_getSpaceSteps( PyGrid & scope, const char* name = "getSpaceSteps" )
    {
        scope.def("getSpaceStepsProducts", get,   py::arg("xPow") );
    }
};

template<>
struct SpaceStepsProductsGetter< Grid2D >
{
    static inline
    typename Grid2D::RealType
    get( const Grid2D & grid, const int & xPow, const int & yPow = 0 )
    {
        if( xPow == 0 && yPow == 0 ) return grid.template getSpaceStepsProducts< 0, 0 >();
        auto hx = grid.template getSpaceStepsProducts< 1, 0 >();
        auto hy = grid.template getSpaceStepsProducts< 0, 1 >();
        if( xPow != 1 ) hx = pow( hx, xPow );
        if( yPow != 1 ) hy = pow( hy, yPow );
        return hx * hy;
    }

    template< typename PyGrid >
    static void
    export_getSpaceSteps( PyGrid & scope, const char* name = "getSpaceSteps" )
    {
        scope.def("getSpaceStepsProducts", get,   py::arg("xPow"), py::arg("yPow")=0 );
    }
};

template<>
struct SpaceStepsProductsGetter< Grid3D >
{
    static inline
    typename Grid3D::RealType
    get( const Grid3D & grid, const int & xPow, const int & yPow = 0, const int & zPow = 0 )
    {
        if( xPow == 0 && yPow == 0 && zPow == 0 ) return grid.template getSpaceStepsProducts< 0, 0, 0 >();
        auto hx = grid.template getSpaceStepsProducts< 1, 0, 0 >();
        auto hy = grid.template getSpaceStepsProducts< 0, 1, 0 >();
        auto hz = grid.template getSpaceStepsProducts< 0, 0, 1 >();
        if( xPow != 1 ) hx = pow( hx, xPow );
        if( yPow != 1 ) hy = pow( hy, yPow );
        if( zPow != 1 ) hz = pow( hz, zPow );
        return hx * hy * hz;
    }

    template< typename PyGrid >
    static void
    export_getSpaceSteps( PyGrid & scope, const char* name = "getSpaceSteps" )
    {
        scope.def("getSpaceStepsProducts", get,   py::arg("xPow"), py::arg("yPow")=0, py::arg("zPow")=0 );
    }
};
