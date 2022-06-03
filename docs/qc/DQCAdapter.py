from AbstractAdapter import AbstractAdapter
import torch
from typing import Union
import xitorch as xt
from dqc.utils.datastruct import AtomCGTOBasis, ValGrad, SpinParam, DensityFitInfo


def _symm(scp: torch.Tensor):
    # forcely symmetrize the tensor
    return (scp + scp.transpose(-2, -1)) * 0.5


class DQCAdapter(AbstractAdapter):
    """
    DQC Adapter
    """

    def __init__(self, qc):
        # copy density matrix:
        self.__dm = qc._dm.detach().clone()

        # copy fockian:
        self.__fockian = qc._engine.dm2scp(self.__dm).clone()

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

        # copy 4D tensor of electron-electron repulsion
        self.__four_center_elrep_tensor = qc._engine.hamilton.el_mat.detach().clone()

    def get_density_matrix(self):
        return self.__dm

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

    def __eigendecomposition(self, hf_engine):
        if not hf_engine._polarized:
            fock = xt.LinearOperator.m(_symm(self.__fockian), is_hermitian=True)
            return hf_engine.diagonalize(fock, self.__noccorb)
        else:
            fock_u = xt.LinearOperator.m(_symm(self.__fockian[0]), is_hermitian=True)
            fock_d = xt.LinearOperator.m(_symm(self.__fockian[1]), is_hermitian=True)
            return hf_engine.diagonalize(hf_engine, SpinParam(u=fock_u, d=fock_d))

    def __construct_vxc_derivative_tensor(self, hamiltonian, dm):
        densinfo = SpinParam.apply_fcn(lambda dm_: hamiltonian._dm2densinfo(dm_), dm)  # value: (*BD, ngrid)
        # print("density info\n", densinfo)
        derivative_of_potinfo_wrt_ro = self.__get_dvxc_wrt_dro_xc(hamiltonian.xc, densinfo)  # value: (*BD, ngrid)
        # print("derivative of potential info wrt density\n", derivative_of_potinfo_wrt_ro)
        dvxc_wrt_dro = self.__get_dvxc_wrt_dro_from_derivative_of_potinfo_wrt_ro(hamiltonian,
                                                                                 derivative_of_potinfo_wrt_ro)
        return dvxc_wrt_dro # (nallorb, nallorb, nallorb, nallorb)

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
            grad_enabled = torch.is_grad_enabled()

            if not isinstance(densinfo, ValGrad):  # polarized case
                raise NotImplementedError("polarized case is not implemented")
            else:  # unpolarized case
                if xc.family == 1:  # LDA
                    # TODO: compare numerical V_XC and dV_XC/dro with concrete analytical values
                    potinfo, = torch.autograd.grad(
                        edensity, densinfo.value, create_graph=grad_enabled,
                        grad_outputs=grad_outputs)
                    # print("potential info\n", potinfo)
                    derivative_of_potinfo_wrt_ro, = torch.autograd.grad(
                        potinfo, densinfo.value, create_graph=grad_enabled,
                        grad_outputs=grad_outputs)
                    return ValGrad(value=derivative_of_potinfo_wrt_ro)
                else:  # GGA and others
                    raise NotImplementedError("Default dvxc wrt dro for this family is not implemented")

    def __get_dvxc_wrt_dro_from_derivative_of_potinfo_wrt_ro(self, hamiltonian, derivative_of_potinfo_wrt_ro: ValGrad):
        """
        Calculate 4D tensor of derivative of V_XC with respect to density.
        :param derivative_of_potinfo_wrt_ro: (ngrid), basis: (ngrid, nallorb)
        :return: (nallorb, nallorb, nallorb, nallorb)
        """

        # TODO: do here the same stuff as DQC did
        mat = torch.einsum("r,ri,rj,rk,rl->ijkl",
                           derivative_of_potinfo_wrt_ro.value,
                           hamiltonian.basis,
                           hamiltonian.basis,
                           hamiltonian.basis,
                           hamiltonian.basis_dvolume)
        return mat
