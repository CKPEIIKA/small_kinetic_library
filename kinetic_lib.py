"""Kinetic library version 0.02
with the capability to calculate various thermodynamic properties for arbitrary temperature:

    Statistical Sums: Total/Translational/Internal
    Specific Energy: Total/Translational/Internal
    Specific Heat at Constant Volume: Total/Translational/Internal

The calculations for these properties take into account translational/electronic degrees of freedom for atoms and translational/electronic/vibrational/rotational degrees of freedom for molecules. In the case of molecules, 
the rotational energy depend on the electronic and vibrational levels, 
and the vibrational energy depend on the electronic level.


"""

import json
from abc import abstractmethod
import numpy as np

CM_TO_M = 100
G_TO_KG = 1000


class Particle:
    """
    Class for a particle, with derived classes of Molecule and Atom
    """

    # Class property to store molecular information
    particle_data = {}
    # Class property to store constants
    constants = {}
    # Class propery to store model parameters
    parameters = {}

    def __init__(self, name):
        """
        Particle initialization

        Parameters:
        - name (string): the name of the particle, e.g. N2, N
        """
        self.name = name
        self.particle_data = self.particle_data[name]
        self.mass = self.particle_data["m"] / G_TO_KG
        self.R_specific = self.constants["k_B"] * self.constants["N_a"] / self.mass

    def __repr__(self):  # to print in console environment
        return self.name

    def _eps(self, n, i, j):
        """
        Internal energy for i-th vibrational, j-th rotational
        and n-th electronic state
        $$\varepsilon^{c}_{nij} + \varepsilon^{el,c}_{nij} + \varepsilon^{rot,c}_{nij} + \varepsilon^{vibr,c}_{nij}$$
        Parameters:
        - n (int) electronic state
        - i (int) rotational level
        - j (int) vibrational level

        Returns:
        float: internal energy
        """
        return self._eps_el(n) + self._eps_v(n, i) + self._eps_r(n, i, j)

    def _eps_el(self, n):
        """
        Electronic energy for n-th electronic state
        """
        hc = self.constants["h"] * self.constants["c"]
        return self.particle_data["eps_el_n"][n] * hc * 100

    def _eps_v(self, n, i):
        """
        Vibrational energy for i-th vibrational level
        and n-th electronic state
        (anharmonic oscillator model)
        $$\varepsilon^{c,n}_{i} = hc \Big(\omega^{c,n}_{e}\Big(i+\frac{1}{2}\Big)
        - \omega^{c,n}_{e}x^{c,n}_{e}\Big(i+\frac{1}{2}\Big)^2\Big)
        + \omega^{c,n}_{e}y^{c,n}_{e}\Big(i+\frac{1}{2}\Big)^3\Big)$$
        """
        hc = self.constants["h"] * self.constants["c"]
        # alt model with I_c
        eps_divbyhc = (
            self.particle_data["omega_n"][n] * (i + 0.5)
            - self.particle_data["omega_ex_n"][n] * (i + 0.5) ** 2
            + self.particle_data["omega_ey_n"][n] * (i + 0.5) ** 3
        )
        return eps_divbyhc * hc * CM_TO_M

    def _eps_r(self, n, i, j):
        """
        Rotational energy for i-th vibrational, j-th rotational levels
        and n-th electronic state
        $$\varepsilon^{c,ni}_{j} = hc \Big(B^{c}_{ni}j(j+1)-D^{c}_{ni}j^2(j+1)^2 + \dots\Big)$$
        $$B^{c}_{ni} = B^{c}_{n,e} - \alpha^{c}_{n,e}\Big(i + \frac{1}{2}\Big)$$
        $$D^{c}_{ni} = D^{c}_{n,e} - \beta^{c}_{n,e}\Big(i + \frac{1}{2}\Big)$$
        """
        hc = self.constants["h"] * self.constants["c"]
        B_ni = self.particle_data["B_n"][n] - self.particle_data["alpha_n"][n] * (
            i + 0.5
        )
        D_ni = self.particle_data["D_n"][n] - self.particle_data["beta_n"][n] * (
            i + 0.5
        )
        eps_divbyhc = B_ni * j * (j + 1) - D_ni * (j * (j + 1)) ** 2
        return eps_divbyhc * hc * CM_TO_M

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
        Unit heat capacity at constant pressure
        """
        return self.R_specific + self.c_v(T)

    def c_p_tr(self, T=None):
        """
        Translational unit heat capacity at constant pressure
        """
        return (5 / 2) * self.R_specific

    def c_p_int(self, T):
        """
        Translational unit heat capacity at constant pressure
        """
        return self.c_p(T) - self.c_p_tr(T)

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
        self.i_max = self.parameters["energy_levels"]["molecule"]["vibrational"]
        self.j_max = self.parameters["energy_levels"]["molecule"]["rotational"]
        self.n_max = self.parameters["energy_levels"]["molecule"]["electronic"]

    def _sum_over_gnij(self, expr):
        """
        Sum an expression multiplied by g_{i,j,n}:
        $$\sum_{nij}g^{c}_{n}g^{c}_{i}g^{c}_{j} expr(n,i,j)$$

        Parameters:
        - expr: function, depending on n,i,j

        Returns:
        (float) Computed sum
        """

        def g_i(i):  # g_{i} vibrational
            return 1

        def g_j(j):  # g_{j} rotational
            return 2 * j + 1

        def g_n(i):  # g_{n} electronic
            return 2 * i + 1

        res = sum(
            g_i(i + 1) * g_j(j + 1) * g_n(n + 1) * expr(n, i, j)
            for i in range(self.i_max)
            for j in range(self.j_max)
            for n in range(self.n_max)
        )
        return res

    def Z_int(self, T):
        """
        The internal partition function
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

    def _sum_over_gn(self, term):
        """
        Sum an expression multiplied by g_{n}:
        $$\sum_{n}g^{c}_{n} expr(n)$$

        Parameters:
        - expr: function, depending on n

        Returns:
        (float) Computed sum
        """
        res = sum((2 * n + 1) * term(n) for n in range(self.n_max))  # g_{n} electronic
        return res

    def e_int(self, T):
        """
        Unit internal energy
        for atom summing only over electronic states
        $$e_{int,c} =
        \frac{1}{m_{c}Z_{int,c}\sum_{n}g_{c,n}\varepsilon^{c}_{n}\exp{(\frac{-\varepsilon^{c}_{n}}{kT})}$$

        Parameters:
        - T: Temperature in Kelvin

        Returns:
        Unit internal energy
        """
        emZ = self._sum_over_gn(
            lambda n: self._eps_el(n)
            * np.exp(-self._eps_el(n) / (self.constants["k_B"] * T))
        )
        e = emZ / ((self.mass / self.constants["N_a"]) * self.Z_int(T))
        return e

    def c_v_int(self, T):
        """
        Internal unit heat capacity at constant volume
        for atom summing only over electronic states

         Parameters:
        - T: Temperature in Kelvin

        Returns:
        Internal unit heat capacity at constant volume

        """
        k = self.constants["k_B"]
        term1 = self._sum_over_gn(
            lambda n: self._eps_el(n) ** 2
            * np.exp(-self._eps_el(n) / (k * T))
            / (k * T) ** 2
        ) / self.Z_int(T)
        term2 = self._sum_over_gn(
            lambda n: self._eps_el(n) * np.exp(-self._eps_el(n) / (k * T)) / (k * T)
        ) / self.Z_int(T)
        return (
            self.constants["k_B"]
            / (self.mass / self.constants["N_a"])
            * (term1 - term2**2)
        )

    def Z_int(self, T):
        """
        Partition function for internal energy
        """
        Z = self._sum_over_gn(
            lambda n: np.exp(-self._eps_el(n) / (self.constants["k_B"] * T))
        )
        return Z
