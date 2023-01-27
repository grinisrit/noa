#!/usr/bin/env python
# coding: utf-8

from pickle import FALSE
from AbstractAdapter import AbstractAdapter
import torch
from typing import Union
import xitorch as xt
from dqc.utils.datastruct import AtomCGTOBasis, ValGrad, SpinParam, DensityFitInfo
import numba as nb
import numpy as np
import pyscf

def _symm(scp: torch.Tensor):
    # forcely symmetrize the tensor
    return (scp + scp.transpose(-2, -1)) * 0.5

@nb.njit(parallel=True)
def _four_term_integral(m, b, o, bv):
    N = b.shape[1]  # size of basis set
    n = o.shape[1]  # number of occupied orbitals
    R = m.shape[0]  # size of grid
    res = np.zeros((N, n, N, n))
    for i in nb.prange(N):
        for a in range(n):
            for k in range(N):
                for c in range(n):
                    for r in range(R):
                        res[i, a, k, c] += m[r] * b[r, i] * o[r, a] * bv[r, k] * o[r, c]
    return res

class PySCFAdapter(AbstractAdapter):
    """ 
    PySCF Adapter
    
    """
    def __init__(self, mf):
        
        #density matrix 
        self.__dm = mf.make_rdm1() #one particle?
        
        #fockian
        self.__fockian = mf.get_fock()
        
        #overlap matrix
        self.__overlap = mf.get_ovlp()
        
        # number of occupied orbitals: 
        self.__noccorb = sum(map(lambda x: x > 0, mf.mo_occ))

        # number of all orbitals: 
        self.__nallorb = len(mf.mo_occ)
        
        
        # construct orbital energies and orbital coefficients:
        self.__energies = mf.mo_energy
        self.__coefficients = mf.mo_coeff
        
        ## another
        #mo_energy, mo_coeff = mf.eig(self.__fockian, self.__overlap)
        
        # orbital occupancies:
        self.__orboccs = mf.mo_occ

        # Adjoint equation components (for debugging)
        self._dE_wrt_dC = None
        self._dE_wrt_depsilon = None
        self._dnorm_wrt_dC = None
        self._dnorm_wrt_depsilon = None
        self._dRoothan_wrt_dC_first_term = None
        self._dRoothan_wrt_dC_second_term = None
        self._dRoothan_wrt_dC = None
        self._dRoothan_wrt_depsilon = None
        self._N = None
        self._dE_wrt_dX = None
        self._dY_wrt_dX = None
        self._adjoint = None
        self._dRoothan_wrt_dtheta = None
        self._dnorm_wrt_dtheta = None
        self._dY_wrt_dtheta = None
        self._derivative = None
        self._dX_wrt_dY = None
        self._chain = None


    def get_density_matrix(self):
        return self.__dm

    def get_overlap_matrix(self):
        return self.__overlap

    def get_number_of_occupied_orbitals(self):
        return self.__noccorb

    def get_number_of_all_orbitals(self):
        return self.__nallorb

    def get_orbital_energies(self):
        return self.__energies

    def get_orbital_coefficients(self):
        return self.__coefficients

    def get_orbital_occupancy(self):
        return self.__orboccs

    def get_fockian(self):
        return self.__fockian

