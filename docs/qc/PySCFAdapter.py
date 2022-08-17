from pickle import FALSE
from AbstractAdapter import AbstractAdapter
#import torch
from typing import Union
#import xitorch as xt
#from dqc.utils.datastruct import AtomCGTOBasis, ValGrad, SpinParam, DensityFitInfo
import numba as nb
import numpy as np
import scipy
import pyscf
import types


# def _symm(scp: torch.Tensor):
#     # forcely symmetrize the tensor
#     return (scp + scp.transpose(-2, -1)) * 0.5


@nb.njit(parallel=True)
def _four_term_integral(m, b, o):
    N = b.shape[1]  # size of basis set
    n = o.shape[1]  # number of occupied orbitals
    R = m.shape[0]  # size of grid
    res = np.zeros((N, n, N, n))
    for i in nb.prange(N):
        for a in range(n):
            for k in range(i+1):
                for c in range(a+1):
                    for r in range(R):
                        value = m[r] * b[r, i] * o[r, a] * b[r, k] * o[r, c]
                        res[i, a, k, c] += value
                        if a != c:
                            res[i, c, k, a] += value
                            if i != k:
                                res[k, a, i, c] += value
                                res[k, c, i, a] += value
                        elif i != k:
                            res[k, a, i, c] += value
    return res


class PySCFAdapter(AbstractAdapter):
    """
    PySCF Adapter
    """

    def __init__(self, qc):
        # copy density matrix:
        # self.__dm = qc._dm.detach().clone()
        
        # PySCF uses non-orthonormal basis set.
        # So we should make it orthonormal. 
        # At first, calculate S^{1/2} and S^{-1/2}
        self.__initial_overlap_minus_half = scipy.linalg.fractional_matrix_power(qc.get_ovlp(), -0.5)
        self.__initial_overlap_plus_half  = scipy.linalg.fractional_matrix_power(qc.get_ovlp(), +0.5)
        
        # construct overlap matrix:
        self.__overlap = self.__initial_overlap_minus_half @ qc.get_ovlp() @ self.__initial_overlap_minus_half

        # construct fockian:
        self.__fockian = self.__initial_overlap_minus_half @ qc.get_fock() @ self.__initial_overlap_minus_half
        
        # copy orbital energies, orbital coefficients and orbital occupancies:
        _, _, self.__energies, self.__coefficients, self.__orboccs = pyscf.scf.hf.kernel(qc)
        self.__coefficients = self.__initial_overlap_plus_half @ self.__coefficients # do not forget to modify coeffs
        
        # number of occupied orbitals:
        self.__noccorb = np.count_nonzero(self.__orboccs)

        # number of all orbitals:
        self.__nallorb = qc.mol.nao_nr()
        
        # only occupied orbitals:
        self.__coefficients = self.__coefficients.transpose()[:self.__noccorb]
        self.__energies     = self.__energies[:self.__noccorb]
        self.__orboccs      = self.__orboccs[:self.__noccorb]
                
        # number of model parameters:
        # NB: user has to assign number of parameters to eval_xc object before calling of define_xc_
        self.__number_of_parameters = qc._numint.eval_xc.number_of_parameters
        
        # temporary object, storing values on grid
        grid_values = self.__construct_on_grid_values(qc)
        
        # construct derivatives from exc and vxc with respect to all thetas
        self.__contruct_all_dxc_wrt_dtheta_tensors(qc, grid_values)
        
        # construct 4D tensor of dvxc/drho
        self.__four_center_xc_tensor = self.__construct_vxc_derivative_tensor(qc, grid_values)

        # construct 4D tensor of electron-electron repulsion
        four_center_elrep_tensor_tmp = np.einsum("mnop,im,jn,ko,pl->ijkl",
                                                    qc.mol.intor('int2e', aosym='s1'),
                                                    self.__initial_overlap_minus_half,
                                                    self.__initial_overlap_minus_half,
                                                    self.__initial_overlap_plus_half,
                                                    self.__initial_overlap_plus_half)

        self.__four_center_elrep_tensor = np.einsum("ijkl,cl,aj->iakc",
                                                    four_center_elrep_tensor_tmp,
                                                    self.get_orbital_coefficients(),
                                                    self.get_orbital_coefficients())

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

        self.__compute_adjoint()
        self.__compute_derivative()

    def get_derivative(self):
        return self._derivative

    def get_adjoint(self):
        return self._adjoint

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

    def get_four_center_xc_tensor(self):
        return self.__four_center_xc_tensor

    def get_four_center_elrep_tensor(self):
        return self.__four_center_elrep_tensor

    def get_number_of_parameters(self):
        return self.__number_of_parameters

    def get_derivative_of_exc_wrt_theta(self):
        return self.__dexc_wrt_dtheta

    def get_derivative_of_vxc_wrt_theta(self):
        return self.__dvxc_wrt_dtheta
    
    def __construct_on_grid_values(self, qc):
        grid_values = types.SimpleNamespace()
        
        dm_in_nonorthogonal_basis = qc.make_rdm1()
        grid_values.coords = qc.grids.coords
        grid_values.weights = qc.grids.weights
        non_orthogonal_ao = pyscf.dft.numint.eval_ao(qc.mol, grid_values.coords, deriv=0)
        grid_values.non_orthogonal_ao = non_orthogonal_ao
        grid_values.rho = pyscf.dft.numint.eval_rho(qc.mol, non_orthogonal_ao, 
                                                    dm_in_nonorthogonal_basis, xctype='LDA')
        grid_values.basis_set = np.einsum("ni,rn->ri", self.__initial_overlap_minus_half, non_orthogonal_ao)
        grid_values.final_orbs = np.einsum("ri,ai->ra", grid_values.basis_set, self.get_orbital_coefficients())
        return grid_values


    # calculate partitial derivative of (full) energy with respect to parameters of xc-functional
    def __contruct_all_dxc_wrt_dtheta_tensors(self, qc, grid_values):
        list_of_dexc_wrt_dtheta = []
        list_of_dvxc_wrt_dtheta = []
        for p_number in range(self.get_number_of_parameters()):
            dexc_wrt_dtheta_on_grid, dvxc_wrt_dtheta_on_grid = qc._numint.eval_xc.wrt[p_number]('LDA', grid_values.rho)[:2]

            current_dexc_wrt_dtheta = np.einsum('r,r,r->...', dexc_wrt_dtheta_on_grid, grid_values.rho, grid_values.weights)
            list_of_dexc_wrt_dtheta.append(current_dexc_wrt_dtheta)
            
            current_dvxc_wrt_dtheta = self.__initial_overlap_minus_half @ np.einsum('r,ri,rj,r->ij', 
                                                    dvxc_wrt_dtheta_on_grid[0],
                                                    grid_values.non_orthogonal_ao,
                                                    grid_values.non_orthogonal_ao,
                                                    grid_values.weights) @ self.__initial_overlap_minus_half
            list_of_dvxc_wrt_dtheta.append(current_dvxc_wrt_dtheta)
        self.__dexc_wrt_dtheta = np.array(list_of_dexc_wrt_dtheta)
        self.__dvxc_wrt_dtheta = np.array(list_of_dvxc_wrt_dtheta)

    def __construct_vxc_derivative_tensor(self, qc, grid_values):
        dvxc_wrt_drho_on_grid = qc._numint.eval_xc('LDA', grid_values.rho)[2]
        dvxc_wrt_drho_on_grid_dot_weights_of_grid = dvxc_wrt_drho_on_grid * grid_values.weights
        res = _four_term_integral(dvxc_wrt_drho_on_grid_dot_weights_of_grid, grid_values.basis_set, grid_values.final_orbs)
        return res

    def __compute_adjoint(self):
        self._dE_wrt_dC = 2 * np.einsum("b,bi,ij->bj",
                                        self.__orboccs, self.__coefficients, self.__fockian)

        self._dE_wrt_depsilon = np.zeros(self.__noccorb)
        print("Shape of dE/depsilon tensor:", self._dE_wrt_depsilon.shape)
        
        self._dE_wrt_dX = np.concatenate((self._dE_wrt_dC, 
                                  np.expand_dims(self._dE_wrt_depsilon, 1)), 
                                 1).transpose().reshape(-1, )

        occupied_orbitals_kronecker = np.eye(self.__noccorb)

        self._dnorm_wrt_dC = 2 * np.einsum("ac,ck->kac",
                                              occupied_orbitals_kronecker, self.__coefficients)

        self._dnorm_wrt_depsilon = np.zeros((
            self.__noccorb, self.__noccorb))

        all_orbitals_kronecker = np.eye(self.__nallorb)

        self._dRoothan_wrt_dC_first_term = np.einsum("ik,ac->iakc",
                                                        self.__fockian, occupied_orbitals_kronecker)
        self._dRoothan_wrt_dC_first_term -= np.einsum("c,ik,ac->iakc",
                                                      self.__energies,
                                                      all_orbitals_kronecker,
                                                      occupied_orbitals_kronecker)

        print("Shape of elrep tensor:", self.get_four_center_elrep_tensor().shape)
        print("Shape of dvxc/drho tensor:", self.get_four_center_elrep_tensor().shape)

        fourDtensor = self.get_four_center_elrep_tensor() + \
                      self.get_four_center_xc_tensor()
        
        print("Shape of fourDtensor:", fourDtensor.shape)


        self._dRoothan_wrt_dC_second_term = 2 * np.einsum("c,iakc->iakc", self.__orboccs, fourDtensor)

        self._dRoothan_wrt_dC = self._dRoothan_wrt_dC_first_term + \
                                self._dRoothan_wrt_dC_second_term

        self._dRoothan_wrt_depsilon = -1 * np.einsum("ac,ai->iac",
                                                     occupied_orbitals_kronecker, 
                                                     self.__coefficients)

        print("Shape of dE/dC:", self._dE_wrt_dC.shape)
        print("Shape of expand_dim'ed dE/depsilon:", np.expand_dims(self._dE_wrt_depsilon, 1).shape)
        
        print("Shape of dRoothan/dC:", self._dRoothan_wrt_dC.shape)
        print("Shape of dRoothan/depsilon:", self._dRoothan_wrt_depsilon.shape)

        self._N = self.__nallorb * self.__noccorb
        upper = np.concatenate((
            self._dRoothan_wrt_dC.reshape((self._N, self._N)),
            self._dRoothan_wrt_depsilon.reshape((self._N, self.__noccorb))), 1)
        lower = np.concatenate((
            self._dnorm_wrt_dC.reshape(self._N, self.__noccorb).transpose(),
            self._dnorm_wrt_depsilon), 1)

        self._dY_wrt_dX = np.concatenate((upper, lower), 0)

        self._adjoint = scipy.linalg.solve(self._dY_wrt_dX, self._dE_wrt_dX)

    def __compute_derivative(self):
        self._dRoothan_wrt_dtheta = np.einsum("ijt,aj->iat",
                                              self.__dvxc_wrt_dtheta, self.__coefficients)

        self._dnorm_wrt_dtheta = np.zeros((self.__noccorb, self.__number_of_parameters))

        self._dY_wrt_dtheta = np.concatenate((
            self._dRoothan_wrt_dtheta.reshape(
                self._N, self.__number_of_parameters),
            self._dnorm_wrt_dtheta), 0)

        print(f'g_p: {self.__dexc_wrt_dtheta}')
        print(f'g_x^t norm: {np.abs(self._dE_wrt_dX).sum()}')
        print(f'lambda norm: {np.abs(self._adjoint).sum()}')
        print(f'f_p norm: {np.abs(self._dY_wrt_dtheta).sum()}')
        print(f'lambda_t * f_p: {(self._adjoint @ self._dY_wrt_dtheta)}')
        self._dX_wrt_dY = scipy.linalg.inv(self._dY_wrt_dX)
        print(f'f_x^-t norm: {np.abs(self._dX_wrt_dY).sum()}')
        self._chain = (self._dX_wrt_dY.transpose() @ self._dY_wrt_dtheta)
        print(f'f_x^-1 * f_p norm: {np.abs(self._chain).sum()}')
        print(f'g_x^t * f_x^-1 * f_p: {(self._dE_wrt_dX @ self._chain)}')

        self._derivative = self.__dexc_wrt_dtheta - (self._adjoint @ self._dY_wrt_dtheta)
