from abc import abstractmethod, abstractproperty


class AbstractAdapter:
    """
    Adapter for different quantum chemistry packages
    """
    @abstractmethod
    def get_number_of_occupied_orbitals(self):
        pass

    @abstractmethod
    def get_number_of_all_orbitals(self):
        pass

    @abstractmethod
    def get_orbital_energies(self):
        pass

    @abstractmethod
    def get_orbital_coefficients(self):
        pass

    @abstractmethod
    def get_orbital_occupancy(self):
        pass

    @abstractmethod
    def get_fockian(self):
        pass

    @abstractmethod
    def get_four_center_xc_tensor(self):
        pass

    @abstractmethod
    def get_four_center_elrep_tensor(self):
        pass
