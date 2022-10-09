// This file is originally a part of PUMAS
// See: https://github.com/niess/pumas/blob/master/examples/pumas/flux.c
/*
 * This is free and unencumbered software released into the public domain.
 *
 * Anyone is free to copy, modify, publish, use, compile, sell, or
 * distribute this software, either in source code form or as a compiled
 * binary, for any purpose, commercial or non-commercial, and by any
 * means.
 *
 * In jurisdictions that recognize copyright laws, the author or authors
 * of this software dedicate any and all copyright interest in the
 * software to the public domain. We make this dedication for the benefit
 * of the public at large and to the detriment of our heirs and
 * successors. We intend this dedication to be an overt act of
 * relinquishment in perpetuity of all present and future rights to this
 * software under copyright law.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS BE LIABLE FOR ANY CLAIM, DAMAGES OR
 * OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
 * ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
 * OTHER DEALINGS IN THE SOFTWARE.
 *
 * For more information, please refer to <http://unlicense.org>
 */

/* This is a simple example of a library of atmospheric muon fluxes based on
 * semi-analytical models. It implements Gaisser's and GCCLY models (see
 * references below).
 */

/* Standard library includes */
#include <math.h>
/* The atmospheric muon fluxes library */
#include "flux.h"

/* Fraction of the muon flux for a given charge */
static double charge_fraction(double charge)
{
        /* Use a constant charge ratio.
         * Ref: CMS (https://arxiv.org/abs/1005.5332)
         */
        const double charge_ratio = 1.2766;

        if (charge < 0.)
                return 1. / (1. + charge_ratio);
        else if (charge > 0.)
                return charge_ratio / (1. + charge_ratio);
        else
                return 1.;
}

/* Gaisser's flux model
 * Ref: see e.g. the ch.30 of the PDG (https://pdglive.lbl.gov)
 */
double flux_gaisser(double cos_theta, double kinetic_energy, double charge)
{
        const double Emu = kinetic_energy + 0.10566;
        const double ec = 1.1 * Emu * cos_theta;
        const double rpi = 1. + ec / 115.;
        const double rK = 1. + ec / 850.;
        return 1.4E+03 * pow(Emu, -2.7) * (1. / rpi + 0.054 / rK) *
            charge_fraction(charge);
}

/* Volkova's parameterization of cos(theta*) */
static double cos_theta_star(double cos_theta)
{
        const double p[] = { 0.102573, -0.068287, 0.958633, 0.0407253,
                0.817285 };
        const double cs2 =
            (cos_theta * cos_theta + p[0] * p[0] + p[1] * pow(cos_theta, p[2]) +
                p[3] * pow(cos_theta, p[4])) /
            (1. + p[0] * p[0] + p[1] + p[3]);
        return cs2 > 0. ? sqrt(cs2) : 0.;
}

/*
 * Guan et al. parameterization of the sea level flux of atmospheric muons
 * Ref: https://arxiv.org/abs/1509.06176
 */
double flux_gccly(double cos_theta, double kinetic_energy, double charge)
{
        const double Emu = kinetic_energy + 0.10566;
        const double cs = cos_theta_star(cos_theta);
        return pow(1. + 3.64 / (Emu * pow(cs, 1.29)), -2.7) *
            flux_gaisser(cs, kinetic_energy, charge);
}
