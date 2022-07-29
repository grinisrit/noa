/*****************************************************************************
 *   Copyright (c) 2022, Roland Grinis, GrinisRIT ltd.                       *
 *   (roland.grinis@grinisrit.com)                                           *
 *   All rights reserved.                                                    *
 *   See the file COPYING for full copying permissions.                      *
 *                                                                           *
 *   This program is free software: you can redistribute it and/or modify    *
 *   it under the terms of the GNU General Public License as published by    *
 *   the Free Software Foundation, either version 3 of the License, or       *
 *   (at your option) any later version.                                     *
 *                                                                           *
 *   This program is distributed in the hope that it will be useful,         *
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of          *
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the            *
 *   GNU General Public License for more details.                            *
 *                                                                           *
 *   You should have received a copy of the GNU General Public License       *
 *   along with this program.  If not, see <http://www.gnu.org/licenses/>.   *
 *****************************************************************************/
/**
 * Implemented by: Roland Grinis
 */

#pragma once

#include "noa/3rdparty/tinyxml2.hh"
#ifndef NOA_3RDPARTY_TINYXML_NOIMPL
        #define NOA_3RDPARTY_TINYXML_NOIMPL
        #include "noa/3rdparty/_tinyxml2/tinyxml2.cpp"
#endif

#ifdef NOA_3RDPARTY_PUMAS
namespace noa::pms::pumas {
        #include "noa/3rdparty/_pumas/pumas.h" // This .h include is inside of the namespace and is to be used
}
#ifndef NOA_3RDPARTY_PUMAS_NOIMPL
        #define NOA_3RDPARTY_PUMAS_NOIMPL
        #undef pumas_h
        #include "noa/3rdparty/_pumas/pumas.h" // This .h include is needed to compile PUMAS
        #include "noa/3rdparty/_pumas/pumas.c"
#endif
#endif
