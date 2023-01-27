from pickle import FALSE
from AbstractAdapter import AbstractAdapter
import torch
from typing import Union
import xitorch as xt
from dqc.utils.datastruct import AtomCGTOBasis, ValGrad, SpinParam, DensityFitInfo
import numba as nb
import numpy as np


def _symm(scp: torch.Tensor):
    # forcely symmetrize the tensor
    return (scp + scp.transpose(-2, -1)) * 0.5


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


class DQCAdapter(AbstractAdapter):
    """
    DQC Adapter
    """

    def __init__(self, qc):
        # copy density matrix:
        self.__dm = qc._dm.detach().clone()

        # copy overlap matrix:
        self.__overlap = qc.get_system().get_hamiltonian().get_overlap()._fullmatrix()

        # copy fockian:
        self.__fockian = qc._engine.dm2scp(self.__dm).detach().clone()

        # number of occupied orbitals:
        self.__noccorb = qc._engine.hf_engine._norb

        # number of all orbitals:
        self.__nallorb = qc.get_system().get_hamiltonian().nao

        # construct orbital energies and orbital coefficients:
        # TODO: assert that dm can be constructed from these values
        self.__energies, self.__coefficients = self.__eigendecomposition(qc._engine.hf_engine)

        # copy orbital occupancies:
        # TODO: here can be some inconsistency with orbital coefficients, check it somehow
        self.__orboccs = qc._engine.orb_weight.detach().clone()

        # construct 4D tensor of dXC/dro
        self.__four_center_xc_tensor = self.__construct_vxc_derivative_tensor(qc._engine.hamilton, self.__dm)

        # construct 4D tensor of electron-electron repulsion
        self.__four_center_elrep_tensor = torch.einsum("ijkl,lc,ja->iakc",
                                                       qc._engine.hamilton.el_mat.detach().clone(),
                                                       self.get_orbital_coefficients(),
                                                       self.get_orbital_coefficients())

        # number of model parameters:
        self.__number_of_parameters = qc._engine.hamilton.xc.number_of_parameters

        # construct derivatives from Exc with respect to all thetas
        self.__contruct_all_dexc_wrt_dtheta_tensors(qc._engine.hamilton, self.__dm)

        # construct derivatives from vxc with respect to all thetas
        self.__contruct_all_dvxc_wrt_dtheta_tensors(qc._engine.hamilton, self.__dm)

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

    def __eigendecomposition(self, hf_engine):
        if not hf_engine._polarized:
            fock = xt.LinearOperator.m(_symm(self.__fockian), is_hermitian=True)
            return hf_engine.diagonalize(fock, self.__noccorb)
        else:
            fock_u = xt.LinearOperator.m(_symm(self.__fockian[0]), is_hermitian=True)
            fock_d = xt.LinearOperator.m(_symm(self.__fockian[1]), is_hermitian=True)
            return hf_engine.diagonalize(hf_engine, SpinParam(u=fock_u, d=fock_d))

    # calculate partitial derivative of (full) energy with respect to parameters of xc-functional
    def __construct_dexc_wrt_dtheta(self, hamiltonian, dm: Union[torch.Tensor, SpinParam[torch.Tensor]],
                                    parameter_number) -> torch.Tensor:
        densinfo = SpinParam.apply_fcn(
            lambda dm_: hamiltonian._dm2densinfo(dm_), dm)  # (spin) value: (*BD, nr)
        edens_derivative = hamiltonian.xc.get_edensityxc_derivative(densinfo, parameter_number).detach()  # (*BD, nr)
        return torch.sum(hamiltonian.grid.get_dvolume() * edens_derivative, dim=-1)

    def __contruct_all_dexc_wrt_dtheta_tensors(self, hamiltonian, dm):
        number_of_parameters = hamiltonian.xc.number_of_parameters
        list_of_tensors = []
        for parameter_number in range(number_of_parameters):
            current_tensor = self.__construct_dexc_wrt_dtheta(hamiltonian, dm, parameter_number)
            list_of_tensors.append(current_tensor.unsqueeze(-1))
        self.__dexc_wrt_dtheta = torch.cat(list_of_tensors, 0)

    def __get_dvxc_wrt_dtheta(self, xc, densinfo, parameter_number):
        """
        Returns the ValGrad for the xc potential given the density info
        for unpolarized case.
        """
        # This is the default implementation of vxc if there is no implementation
        # in the specific class of XC.

        # densinfo.value & lapl: (*BD, nr)
        # densinfo.grad: (*BD, ndim, nr)
        # return:
        # potentialinfo.value & lapl: (*BD, nr)
        # potentialinfo.grad: (*BD, ndim, nr)

        # mark the densinfo components as requiring grads
        with xc._enable_grad_densinfo(densinfo):
            with torch.enable_grad():
                edensity_derivative = xc.get_edensityxc_derivative(densinfo, parameter_number)  # (*BD, nr)
            grad_outputs = torch.ones_like(edensity_derivative)
            grad_enabled = torch.is_grad_enabled()

            if not isinstance(densinfo, ValGrad):  # polarized case
                raise NotImplementedError("polarized case is not implemented")
            else:  # unpolarized case
                if xc.family == 1:  # LDA
                    dedn, = torch.autograd.grad(
                        edensity_derivative, densinfo.value, create_graph=grad_enabled,
                        grad_outputs=grad_outputs)

                    return ValGrad(value=dedn)
                else:  # GGA and others
                    raise NotImplementedError("Default dvxc wrt dro for this family is not implemented")

    def __contruct_dvxc_wrt_dtheta(self, hamiltonian, dm, parameter_number):
        densinfo = SpinParam.apply_fcn(lambda dm_: hamiltonian._dm2densinfo(dm_), dm)  # value: (*BD, nr)
        potinfo = self.__get_dvxc_wrt_dtheta(hamiltonian.xc, densinfo, parameter_number)  # value: (*BD, nr)
        vxc_linop = hamiltonian._get_vxc_from_potinfo(potinfo)
        return vxc_linop

    def __contruct_all_dvxc_wrt_dtheta_tensors(self, hamiltonian, dm):
        number_of_parameters = hamiltonian.xc.number_of_parameters
        list_of_tensors = []
        for parameter_number in range(number_of_parameters):
            current_tensor = self.__contruct_dvxc_wrt_dtheta(hamiltonian, dm, parameter_number)
            list_of_tensors.append(current_tensor.fullmatrix().detach().unsqueeze(-1))
        self.__dvxc_wrt_dtheta = torch.cat(list_of_tensors, 2)

    def __construct_vxc_derivative_tensor(self, hamiltonian, dm):
        densinfo = SpinParam.apply_fcn(lambda dm_: hamiltonian._dm2densinfo(dm_), dm)  # value: (*BD, ngrid)
        print(f'density info {densinfo.value.shape}')
        derivative_of_potinfo_wrt_ro = self.__get_dvxc_wrt_dro_xc(hamiltonian.xc, densinfo)  # value: (*BD, ngrid)
        dvxc_wrt_dro = self.__get_dvxc_wrt_dro_from_derivative_of_potinfo_wrt_ro(hamiltonian,
                                                                                 derivative_of_potinfo_wrt_ro)
        return dvxc_wrt_dro  # (nallorb, noccorb, nallorb, noccorb)

    def __get_dvxc_wrt_dro_xc(self, xc, densinfo):
        """
        calculating values of dvxc_wrt_dro at all points of grid
        :param densinfo: (*BD, ngrid)
        :return: derivative_of_potinfo_wrt_ro: (*BD, ngrid)
        """

        # mark the densinfo components as requiring grads
        with xc._enable_grad_densinfo(densinfo):
            with torch.enable_grad():
                edensity = xc.get_edensityxc(densinfo)  # (*BD, ngrid)
            grad_outputs = torch.ones_like(edensity)
            # print(f'edensity {edensity.shape}')
            grad_enabled = torch.is_grad_enabled()

            if not isinstance(densinfo, ValGrad):  # polarized case
                raise NotImplementedError("polarized case is not implemented")
            else:  # unpolarized case
                if xc.family == 1:  # LDA
                    # Here we are taking autograd.grad derivative twice:
                    potinfo, = torch.autograd.grad(
                        edensity, densinfo.value, create_graph=grad_enabled,
                        grad_outputs=grad_outputs)
                    # print(f'potinfo {potinfo.shape}')
                    derivative_of_potinfo_wrt_ro, = torch.autograd.grad(
                        potinfo, densinfo.value, grad_outputs=grad_outputs)
                    return ValGrad(value=derivative_of_potinfo_wrt_ro)
                else:  # GGA and others
                    raise NotImplementedError("Default dvxc wrt dro for this family is not implemented")

    def __get_dvxc_wrt_dro_from_derivative_of_potinfo_wrt_ro(self, hamiltonian, derivative_of_potinfo_wrt_ro: ValGrad):
        """
        Calculate 4D tensor of derivative of V_XC with respect to density.
        :param derivative_of_potinfo_wrt_ro: (ngrid), basis: (ngrid, nallorb)
        :return: (nallorb, nallorb, nallorb, nallorb)
        # TODO: do here the same stuff as DQC did
        mat = torch.einsum("r,ri,rj,rk,rl->ijkl",
                           derivative_of_potinfo_wrt_ro.value,
                           hamiltonian.basis,
                           hamiltonian.basis,
                           hamiltonian.basis,
                           hamiltonian.basis_dvolume)
        """
        print(f'started the four term integral computation for {hamiltonian.basis.shape}...')
        molecular_orbitals_at_grid = hamiltonian.basis @ self.get_orbital_coefficients()
        derivative_of_potinfo_wrt_ro_dvolume = derivative_of_potinfo_wrt_ro.value * hamiltonian.dvolume
        mat = _four_term_integral(derivative_of_potinfo_wrt_ro_dvolume.numpy(),
                                  hamiltonian.basis.numpy(),
                                  molecular_orbitals_at_grid.numpy())
        return torch.from_numpy(mat)

    def __compute_adjoint(self):
        self._dE_wrt_dC = 2 * torch.einsum("b,ib,ij->bj",
                                           self.__orboccs, self.__coefficients, self.__fockian)

        self._dE_wrt_depsilon = torch.zeros(self.__noccorb)

        occupied_orbitals_kronecker = torch.eye(self.__noccorb)

        self._dnorm_wrt_dC = 2 * torch.einsum("ac,kc->kac",
                                              occupied_orbitals_kronecker, self.__coefficients)

        self._dnorm_wrt_depsilon = torch.zeros((
            self.__noccorb, self.__noccorb))

        all_orbitals_kronecker = torch.eye(self.__nallorb)

        self._dRoothan_wrt_dC_first_term = torch.einsum("ik,ac->iakc",
                                                        self.__fockian, occupied_orbitals_kronecker)
        self._dRoothan_wrt_dC_first_term -= torch.einsum("c,ik,ac->iakc",
                                                         self.__energies,
                                                         all_orbitals_kronecker,
                                                         occupied_orbitals_kronecker)

        fourDtensor = self.get_four_center_elrep_tensor() + \
                      self.get_four_center_xc_tensor()

        self._dRoothan_wrt_dC_second_term = 2 * torch.einsum(
            "c,iakc->iakc",
            self.__orboccs, fourDtensor)

        self._dRoothan_wrt_dC = self._dRoothan_wrt_dC_first_term + \
                                self._dRoothan_wrt_dC_second_term

        self._dRoothan_wrt_depsilon = -1 * torch.einsum("ac,ia->iac",
                                                        occupied_orbitals_kronecker, self.__coefficients)

        self._dE_wrt_dX = torch.cat((self._dE_wrt_dC,
                                     self._dE_wrt_depsilon.unsqueeze(-1)), 1) \
            .t().reshape(-1, )

        self._N = self.__nallorb * self.__noccorb
        upper = torch.cat((
            self._dRoothan_wrt_dC.reshape((self._N, self._N)),
            self._dRoothan_wrt_depsilon.reshape((self._N, self.__noccorb))), 1)
        lower = torch.cat((
            self._dnorm_wrt_dC.reshape(self._N, self.__noccorb).t(),
            self._dnorm_wrt_depsilon), 1)

        self._dY_wrt_dX = torch.cat((upper, lower), 0)

        self._adjoint = torch.linalg.solve(self._dY_wrt_dX, self._dE_wrt_dX)

    def __compute_derivative(self):
        self._dRoothan_wrt_dtheta = torch.einsum("ijt,ja->iat",
                                                 self.__dvxc_wrt_dtheta, self.__coefficients)

        self._dnorm_wrt_dtheta = torch.zeros(
            self.__noccorb, self.__number_of_parameters)

        self._dY_wrt_dtheta = torch.cat((
            self._dRoothan_wrt_dtheta.reshape(
                self._N, self.__number_of_parameters),
            self._dnorm_wrt_dtheta), 0)

        print(f'g_p: {self.__dexc_wrt_dtheta}')
        print(f'g_x^t norm: {self._dE_wrt_dX.abs().sum()}')
        print(f'lambda norm: {self._adjoint.abs().sum()}')
        print(f'f_p norm: {self._dY_wrt_dtheta.abs().sum()}')
        print(f'lambda_t * f_p: {torch.matmul(self._adjoint, self._dY_wrt_dtheta)}')
        self._dX_wrt_dY = torch.linalg.inv(self._dY_wrt_dX)
        print(f'f_x^-t norm: {self._dX_wrt_dY.abs().sum()}')
        self._chain = torch.matmul(self._dX_wrt_dY.t(), self._dY_wrt_dtheta)
        print(f'f_x^-1 * f_p norm: {self._chain.abs().sum()}')
        print(f'g_x^t * f_x^-1 * f_p: {torch.matmul(self._dE_wrt_dX, self._chain)}')

        self._derivative = self.__dexc_wrt_dtheta - \
                           torch.matmul(self._adjoint, self._dY_wrt_dtheta)
