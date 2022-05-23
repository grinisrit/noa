// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#ifndef HAVE_CUDA

   #define __host__
   #define __device__
   #define __global__

struct dim3
{
   unsigned int x = 1;
   unsigned int y = 1;
   unsigned int z = 1;

   dim3() = default;
   constexpr dim3( const dim3& ) = default;
   constexpr dim3( dim3&& ) = default;

   constexpr dim3( unsigned int x, unsigned int y = 1, unsigned int z = 1 ) : x( x ), y( y ), z( z ) {}
};

struct cudaStream_t
{
   cudaStream_t() = default;
   cudaStream_t( int /*dummy*/ ) {}
};

#endif
