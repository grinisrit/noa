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

#include "noa/pms/dcs.hh"
#include "noa/pms/physics.hh"
#include "noa/utils/common.cuh"

void noa::pms::dcs::cuda::vmap_bremsstrahlung(
        const Calculation &result,
        const Energies &kinetic_energies,
        const Energies &recoil_energies,
        const AtomicElement &element,
        const ParticleMass &mass) {
    const Scalar *pq = recoil_energies.data_ptr<Scalar>();
    const auto brems = [pq, element, mass] __device__(const Index i, const Scalar &k) {
        return _bremsstrahlung_(k, pq[i], element, mass);
    };
    utils::cuda::vmapi<Scalar>(kinetic_energies, brems, result);
}