"""Kinetic library version 0.01"""

import json
from abc import abstractmethod
import numpy as np


class Particle:
    """
    Class for a particle
    """

    # Class property to store molecular information
    particle_data = {}
    # Class property to store constants
    constants = {}
    # Class propery to store model parameters
    parameters = {}

    def __init__(self, name):
        self.name = name
        self.particle_data = self.particle_data[name]
        self.mass = self.particle_data["m"] / 1000  # g -> kg
        self.R_specific = self.constants["k_B"] * self.constants["N_a"] / self.mass

    def __repr__(self):  # repr
        return self.name

    def _eps(self, n, i, j):
        """
        Internal energy for i-th vibrational, j-th rotational
        and n-th electronic level
        $$\varepsilon^{c}_{nij} + \varepsilon^{el,c}_{nij} + \varepsilon^{rot,c}_{nij} + \varepsilon^{vibr,c}_{nij}$$
        Parameters:
        - n (int) electronic level
        - i (int) rotational level
        - j (int) electronic level

        Returns:
        float: internal energy
        """
        return self._eps_el(n) + self._eps_v(i, n) + self._eps_r(i, j, n)

    def _eps_el(self, n):
        """
        Electronic energy for n-th electronic state
        """
        hc = self.constants["h"] * self.constants["c"]
        return self.particle_data["eps_el_n"][n] * hc * 100

    def _eps_v(self, i, n):
        """
        Vibrational energy for i-th vibrational level
        and n-th electronic state
        (anharmonic oscillator model)
        $$\varepsilon^{c,n}_{i} = hc \Big(\omega^{c,n}_{e}\Big(i+\frac{1}{2}\Big)
        - \omega^{c,n}_{e}x^{c,n}_{e}\Big(i+\frac{1}{2}\Big)^2\Big)
        + \omega^{c,n}_{e}y^{c,n}_{e}\Big(i+\frac{1}{2}\Big)^3\Big)$$
        """
        hc = self.constants["h"] * self.constants["c"]
        # wy, wz?
        #  model with I_c
        eps_divbyhc = (
            self.particle_data["omega_n"][n] * (i + 0.5)
            - self.particle_data["omega_ex_n"][n] * (i + 0.5) ** 2
            + self.particle_data["omega_ey_n"][n] * (i + 0.5) ** 2
        )
        # *100 cm -> m
        return eps_divbyhc * hc * 100

    def _eps_r(self, i, j, n):
        """
        Rotational energy for i-th vibrational, j-th rotational levels
        and n-th electronic state
        $$\varepsilon^{c,ni}_{j} = hc \Big(B^{c}_{ni}j(j+1)-D^{c}_{ni}j^2(j+1)^2 + ...\Big)$$
        $$B^{c}_{ni} = B^{c}_{n,e} - \alpha^{c}_{n,e}\Big(i + \frac{1}{2}\Big)$$
        $$D^{c}_{ni} = D^{c}_{n,e} - \beta^{c}_{n,e}\Big(i + \frac{1}{2}\Big)$$
        """
        hc = self.constants["h"] * self.constants["c"]
        B_ni = self.particle_data["B_n"][n] - self.particle_data["alpha_n"][n] * (
            i + 0.5
        )
        D_ni = self.particle_data["D_n"][n]  # - beta...
        eps_divbyhc = B_ni * j * (j + 1) - D_ni * (j * (j + 1)) ** 2
        # *100 cm -> m
        return eps_divbyhc * hc * 100

    def Z(self, T):
        """
        Partition Function
        $$Z = Z_{tr}Z_{int}$$

        Parameters:
        - T: Temperature in Kelvin
        """
        return self.Z_tr(T) * self.Z_int(T)

    def Z_tr(self, T):
        """
        Translational partition function
        $$Z_{tr,c} = \Big(\frac{2\pi m_{c}kT}{h^2}\Big)^{\frac{3}{2}}$$

        Parameters:
        - T (float): Temperature in Kelvin

        Returns:
        float: Translational partition function.
        """
        assert T > 0, "Temperature must be a positive number."

        return np.power(
            (
                2
                * np.pi
                * self.mass
                / self.constants["N_a"]
                * self.constants["k_B"]
                * T
                / self.constants["h"] ** 2
            ),
            1.5,
        )

    @abstractmethod
    def Z_int(self, T):
        """
        Internal partition function, defined in subclasses
        """

    def c_v(self, T):
        """
        Unit heat capacity at constant volume
        $$c_{v} = c_{v,tr} + c_{v,rot}$$
        """
        return self.c_v_int(T) + self.c_v_tr()

    def c_v_tr(self, T=None):
        """
        Translational unit heat capacity at constant volume
        $$c_{v,tr} = \frac{3}{2}\frac{k}{m}$$
        """
        return (3 / 2) * self.R_specific

    @abstractmethod
    def c_v_int(self, T):
        """
        Internal unit heat capacity at constant volume, defined in subclasses
        """

    def c_p(self, T):
        """
        Internal unit heat capacity at constant pressure
        """
        return self.R_specific + self.c_v_int(T)

    def e(self, T):
        """
        Full unit energy
        """
        return self.e_tr(T) + self.e_int(T)

    def e_tr(self, T):
        """
        Translational unit energy
        """
        return (3 / 2) * self.R_specific * T

    @abstractmethod
    def e_int(self, T):
        """
        Internal unit energy, defined in subclasses
        """

    @classmethod
    def load_particle_data(
        cls,
        particle_path="particle_data.json",
        constants_path="constants.json",
        parameters_path="parameters.json",
    ):
        """Loads particle data, constants and model settings from json"""
        with open(particle_path, "r", encoding="UTF-8") as file:
            cls.particle_data = json.load(file)

        with open(constants_path, "r", encoding="UTF-8") as file:
            cls.constants = json.load(file)

        with open(parameters_path, "r", encoding="UTF-8") as file:
            cls.parameters = json.load(file)


class Molecule(Particle):
    """
    Class for a molecule
    """

    def __init__(self, name):
        super().__init__(name)
        self.i_max = self.parameters["energy_levels"]["molecule"]["translational"]
        self.j_max = self.parameters["energy_levels"]["molecule"]["rotational"]
        self.n_max = self.parameters["energy_levels"]["molecule"]["electronic"]

    def _sum_over_gnij(self, term):
        res = sum(
            (2 * j + 1)  # g_{j} rotational
            * 1  # g_{i} vibrational
            * (2 * n + 1)  # g_{n} electronic
            * term(n, i, j)
            for i in range(self.i_max)
            for j in range(self.j_max)
            for n in range(self.n_max)
        )
        return res

    def Z_int(self, T):
        """
        Partition function for internal energy
        $$Z_{int,c} =
        \sum_{nij}g^{c}_{n}g^{c}_{i}g^{c}_{j}\exp{\Big(-\frac{\varepsilon^{c}_{nij}}{kT})}$$

        Parameters:
        - T: Temperature in Kelvin

        Returns:
        Partition function for internal energy
        """
        Z = self._sum_over_gnij(
            lambda n, i, j: np.exp(-self._eps(n, i, j) / (self.constants["k_B"] * T))
        )

        return Z

    def e_int(self, T):
        """
        Unit internal energy
        $$e_{int,c} =
        \frac{1}{m_{c}Z_{int,c}\sum_{nij}g_{c,nij}\varepsilon^{c}_{nij}\exp{(\frac{-\varepsilon^{c}_{nij}}{kT})}$$

        Parameters:
        - T: Temperature in Kelvin

        Returns:
        Unit internal energy
        """
        emZ = self._sum_over_gnij(
            lambda n, i, j: self._eps(n, i, j)
            * np.exp(-self._eps(n, i, j) / (self.constants["k_B"] * T))
        )
        e = emZ / ((self.mass / self.constants["N_a"]) * self.Z_int(T))
        return e

    def c_v_int(self, T):
        """
        Internal unit heat capacity at constant volume

         Parameters:
        - T: Temperature in Kelvin

        Returns:
        Internal unit heat capacity at constant volume

        """
        k = self.constants["k_B"]
        term1 = self._sum_over_gnij(
            lambda n, i, j: self._eps(n, i, j) ** 2
            * np.exp(-self._eps(n, i, j) / (k * T))
            / (k * T) ** 2
        ) / self.Z_int(T)
        term2 = self._sum_over_gnij(
            lambda n, i, j: self._eps(n, i, j)
            * np.exp(-self._eps(n, i, j) / (k * T))
            / (k * T)
        ) / self.Z_int(T)
        return (
            self.constants["k_B"]
            / (self.mass / self.constants["N_a"])
            * (term1 - term2**2)
        )


class Atom(Particle):
    """
    Class for a atom
    """

    def __init__(self, name):
        super().__init__(name)
        self.n_max = self.parameters["energy_levels"]["atom"]["electronic"]

    def _sum_over_gn(self):
        pass

    def e_int(self, T):
        pass

    def c_v_int(self, T):
        pass

    def Z_int(self, T):
        """
        Partition function for internal energy
        """
        Z = sum(
            (2 * n + 1) * np.exp(-self._eps_el(n) / (self.constants["k_B"] * T))
            for n in range(self.n_max)
        )

        return Z
