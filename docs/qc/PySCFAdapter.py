from pickle import FALSE
from AbstractAdapter import AbstractAdapter
#import torch
from typing import Union
#import xitorch as xt
#from dqc.utils.datastruct import AtomCGTOBasis, ValGrad, SpinParam, DensityFitInfo
#import numba as nb
import numpy as np
import scipy
import pyscf


# def _symm(scp: torch.Tensor):
#     # forcely symmetrize the tensor
#     return (scp + scp.transpose(-2, -1)) * 0.5


# @nb.njit(parallel=True)
# def _four_term_integral(m, b, o):
#     N = b.shape[1]  # size of basis set
#     n = o.shape[1]  # number of occupied orbitals
#     R = m.shape[0]  # size of grid
#     res = np.zeros((N, n, N, n))
#     for i in nb.prange(N):
#         for a in range(n):
#             for k in range(i+1):
#                 for c in range(a+1):
#                     for r in range(R):
#                         value = m[r] * b[r, i] * o[r, a] * b[r, k] * o[r, c]
#                         res[i, a, k, c] += value
#                         if a != c:
#                             res[i, c, k, a] += value
#                             if i != k:
#                                 res[k, a, i, c] += value
#                                 res[k, c, i, a] += value
#                         elif i != k:
#                             res[k, a, i, c] += value
#     return res


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
        self.__coefficients = self.__coefficients[:np.count_nonzero(self.__orboccs)] # only occupied orbitals
        
        # number of occupied orbitals:
        self.__noccorb = np.count_nonzero(self.__orboccs)

        # number of all orbitals:
        self.__nallorb = qc.mol.nao_nr()
        
        # number of model parameters:
        # NB: user has to assign number of parameters to eval_xc object before calling of define_xc_
        self.__number_of_parameters = qc._numint.eval_xc.number_of_parameters
                
        # construct derivatives from Exc with respect to all thetas
        self.__contruct_all_dxc_wrt_dtheta_tensors(qc)
        
#         # construct 4D tensor of dXC/dro
#         self.__four_center_xc_tensor = self.__construct_vxc_derivative_tensor(qc._engine.hamilton, self.__dm)

#         # construct 4D tensor of electron-electron repulsion
#         self.__four_center_elrep_tensor = torch.einsum("ijkl,lc,ja->iakc",
#                                                        qc._engine.hamilton.el_mat.detach().clone(),
#                                                        self.get_orbital_coefficients(),
#                                                        self.get_orbital_coefficients())

#         # Adjoint equation components (for debugging)
#         self._dE_wrt_dC = None
#         self._dE_wrt_depsilon = None
#         self._dnorm_wrt_dC = None
#         self._dnorm_wrt_depsilon = None
#         self._dRoothan_wrt_dC_first_term = None
#         self._dRoothan_wrt_dC_second_term = None
#         self._dRoothan_wrt_dC = None
#         self._dRoothan_wrt_depsilon = None
#         self._N = None
#         self._dE_wrt_dX = None
#         self._dY_wrt_dX = None
#         self._adjoint = None
#         self._dRoothan_wrt_dtheta = None
#         self._dnorm_wrt_dtheta = None
#         self._dY_wrt_dtheta = None
#         self._derivative = None
#         self._dX_wrt_dY = None
#         self._chain = None

#         self.__compute_adjoint()
#         self.__compute_derivative()

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

    # calculate partitial derivative of (full) energy with respect to parameters of xc-functional
    def __contruct_all_dxc_wrt_dtheta_tensors(self, qc):
        dm_in_nonorthogonal_basis = qc.make_rdm1()
        coords_of_grid = qc.grids.coords
        weights_of_grid = qc.grids.weights
        ao_value_on_grid = pyscf.dft.numint.eval_ao(qc.mol, coords_of_grid, deriv=0)
        rho_on_grid = pyscf.dft.numint.eval_rho(qc.mol, ao_value_on_grid, dm_in_nonorthogonal_basis, xctype='LDA')

        list_of_dexc_wrt_dtheta = []
        list_of_dvxc_wrt_dtheta = []
        for parameter_number in range(self.get_number_of_parameters()):
            dexc_wrt_dtheta_on_grid, dvxc_wrt_dtheta_on_grid = qc._numint.eval_xc.wrt[parameter_number]('LDA', rho_on_grid)[:2]

            current_dexc_wrt_dtheta = np.einsum('r,r,r->...', dexc_wrt_dtheta_on_grid, rho_on_grid, weights_of_grid)
            list_of_dexc_wrt_dtheta.append(current_dexc_wrt_dtheta)
            
            current_dvxc_wrt_dtheta = self.__initial_overlap_minus_half @ np.einsum('r,ri,rj,r->ij', 
                                                    dvxc_wrt_dtheta_on_grid[0], 
                                                    ao_value_on_grid, 
                                                    ao_value_on_grid, 
                                                    weights_of_grid) @ self.__initial_overlap_minus_half
            list_of_dvxc_wrt_dtheta.append(current_dvxc_wrt_dtheta)     
        self.__dexc_wrt_dtheta = np.array(list_of_dexc_wrt_dtheta)
        self.__dvxc_wrt_dtheta = np.array(list_of_dvxc_wrt_dtheta)

#     def __construct_vxc_derivative_tensor(self, hamiltonian, dm):
#         densinfo = SpinParam.apply_fcn(lambda dm_: hamiltonian._dm2densinfo(dm_), dm)  # value: (*BD, ngrid)
#         print(f'density info {densinfo.value.shape}')
#         derivative_of_potinfo_wrt_ro = self.__get_dvxc_wrt_dro_xc(hamiltonian.xc, densinfo)  # value: (*BD, ngrid)
#         dvxc_wrt_dro = self.__get_dvxc_wrt_dro_from_derivative_of_potinfo_wrt_ro(hamiltonian,
#                                                                                  derivative_of_potinfo_wrt_ro)
#         return dvxc_wrt_dro  # (nallorb, noccorb, nallorb, noccorb)

#     def __get_dvxc_wrt_dro_xc(self, xc, densinfo):
#         """
#         calculating values of dvxc_wrt_dro at all points of grid
#         :param densinfo: (*BD, ngrid)
#         :return: derivative_of_potinfo_wrt_ro: (*BD, ngrid)
#         """

#         # mark the densinfo components as requiring grads
#         with xc._enable_grad_densinfo(densinfo):
#             with torch.enable_grad():
#                 edensity = xc.get_edensityxc(densinfo)  # (*BD, ngrid)
#             grad_outputs = torch.ones_like(edensity)
#             # print(f'edensity {edensity.shape}')
#             grad_enabled = torch.is_grad_enabled()

#             if not isinstance(densinfo, ValGrad):  # polarized case
#                 raise NotImplementedError("polarized case is not implemented")
#             else:  # unpolarized case
#                 if xc.family == 1:  # LDA
#                     # Here we are taking autograd.grad derivative twice:
#                     potinfo, = torch.autograd.grad(
#                         edensity, densinfo.value, create_graph=grad_enabled,
#                         grad_outputs=grad_outputs)
#                     # print(f'potinfo {potinfo.shape}')
#                     derivative_of_potinfo_wrt_ro, = torch.autograd.grad(
#                         potinfo, densinfo.value, grad_outputs=grad_outputs)
#                     return ValGrad(value=derivative_of_potinfo_wrt_ro)
#                 else:  # GGA and others
#                     raise NotImplementedError("Default dvxc wrt dro for this family is not implemented")

#     def __get_dvxc_wrt_dro_from_derivative_of_potinfo_wrt_ro(self, hamiltonian, derivative_of_potinfo_wrt_ro: ValGrad):
#         """
#         Calculate 4D tensor of derivative of V_XC with respect to density.
#         :param derivative_of_potinfo_wrt_ro: (ngrid), basis: (ngrid, nallorb)
#         :return: (nallorb, nallorb, nallorb, nallorb)
#         # TODO: do here the same stuff as DQC did
#         mat = torch.einsum("r,ri,rj,rk,rl->ijkl",
#                            derivative_of_potinfo_wrt_ro.value,
#                            hamiltonian.basis,
#                            hamiltonian.basis,
#                            hamiltonian.basis,
#                            hamiltonian.basis_dvolume)
#         """
#         print(f'started the four term integral computation for {hamiltonian.basis.shape}...')
#         molecular_orbitals_at_grid = hamiltonian.basis @ self.get_orbital_coefficients()
#         derivative_of_potinfo_wrt_ro_dvolume = derivative_of_potinfo_wrt_ro.value * hamiltonian.dvolume
#         mat = _four_term_integral(derivative_of_potinfo_wrt_ro_dvolume.numpy(),
#                                   hamiltonian.basis.numpy(),
#                                   molecular_orbitals_at_grid.numpy())
#         return torch.from_numpy(mat)

#     def __compute_adjoint(self):
#         self._dE_wrt_dC = 2 * torch.einsum("b,ib,ij->bj",
#                                            self.__orboccs, self.__coefficients, self.__fockian)

#         self._dE_wrt_depsilon = torch.zeros(self.__noccorb)

#         occupied_orbitals_kronecker = torch.eye(self.__noccorb)

#         self._dnorm_wrt_dC = 2 * torch.einsum("ac,kc->kac",
#                                               occupied_orbitals_kronecker, self.__coefficients)

#         self._dnorm_wrt_depsilon = torch.zeros((
#             self.__noccorb, self.__noccorb))

#         all_orbitals_kronecker = torch.eye(self.__nallorb)

#         self._dRoothan_wrt_dC_first_term = torch.einsum("ik,ac->iakc",
#                                                         self.__fockian, occupied_orbitals_kronecker)
#         self._dRoothan_wrt_dC_first_term -= torch.einsum("c,ik,ac->iakc",
#                                                          self.__energies,
#                                                          all_orbitals_kronecker,
#                                                          occupied_orbitals_kronecker)

#         fourDtensor = self.get_four_center_elrep_tensor() + \
#                       self.get_four_center_xc_tensor()

#         self._dRoothan_wrt_dC_second_term = 2 * torch.einsum(
#             "c,iakc->iakc",
#             self.__orboccs, fourDtensor)

#         self._dRoothan_wrt_dC = self._dRoothan_wrt_dC_first_term + \
#                                 self._dRoothan_wrt_dC_second_term

#         self._dRoothan_wrt_depsilon = -1 * torch.einsum("ac,ia->iac",
#                                                         occupied_orbitals_kronecker, self.__coefficients)

#         self._dE_wrt_dX = torch.cat((self._dE_wrt_dC,
#                                      self._dE_wrt_depsilon.unsqueeze(-1)), 1) \
#             .t().reshape(-1, )

#         self._N = self.__nallorb * self.__noccorb
#         upper = torch.cat((
#             self._dRoothan_wrt_dC.reshape((self._N, self._N)),
#             self._dRoothan_wrt_depsilon.reshape((self._N, self.__noccorb))), 1)
#         lower = torch.cat((
#             self._dnorm_wrt_dC.reshape(self._N, self.__noccorb).t(),
#             self._dnorm_wrt_depsilon), 1)

#         self._dY_wrt_dX = torch.cat((upper, lower), 0)

#         self._adjoint = torch.linalg.solve(self._dY_wrt_dX, self._dE_wrt_dX)

#     def __compute_derivative(self):
#         self._dRoothan_wrt_dtheta = torch.einsum("ijt,ja->iat",
#                                                  self.__dvxc_wrt_dtheta, self.__coefficients)

#         self._dnorm_wrt_dtheta = torch.zeros(
#             self.__noccorb, self.__number_of_parameters)

#         self._dY_wrt_dtheta = torch.cat((
#             self._dRoothan_wrt_dtheta.reshape(
#                 self._N, self.__number_of_parameters),
#             self._dnorm_wrt_dtheta), 0)

#         print(f'g_p: {self.__dexc_wrt_dtheta}')
#         print(f'g_x^t norm: {self._dE_wrt_dX.abs().sum()}')
#         print(f'lambda norm: {self._adjoint.abs().sum()}')
#         print(f'f_p norm: {self._dY_wrt_dtheta.abs().sum()}')
#         print(f'lambda_t * f_p: {torch.matmul(self._adjoint, self._dY_wrt_dtheta)}')
#         self._dX_wrt_dY = torch.linalg.inv(self._dY_wrt_dX)
#         print(f'f_x^-t norm: {self._dX_wrt_dY.abs().sum()}')
#         self._chain = torch.matmul(self._dX_wrt_dY.t(), self._dY_wrt_dtheta)
#         print(f'f_x^-1 * f_p norm: {self._chain.abs().sum()}')
#         print(f'g_x^t * f_x^-1 * f_p: {torch.matmul(self._dE_wrt_dX, self._chain)}')

#         self._derivative = self.__dexc_wrt_dtheta - \
#                            torch.matmul(self._adjoint, self._dY_wrt_dtheta)
