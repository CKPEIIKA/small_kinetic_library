"""Kinetic library version 0.03
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
        Spectroscopic data is coming from the particle_data dictionary.

        Parameters:
        - name (string): the name of the particle, e.g. N2, N
        """
        self.name = name
        self.particle_data = self.particle_data[name]
        self.mole_number = 1
        self.mass = self.particle_data["M"] / G_TO_KG / self.constants["N_a"]
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
        Electronic energy for n-th electronic state.
        Tabulated in particle_data.
        $$\varepsilon_{el} = E_{el}hc$$

        Parameters:
        - n: electronic level

        """
        hc = self.constants["h"] * self.constants["c"]
        eps_el = self.particle_data["eps_el_n"][n] * hc * 100
        return eps_el

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
        # TODO: alt model with harmonic osc and I_c
        eps_divbyhc = (
            self.particle_data["omega_n"][n] * (i + 0.5)
            - self.particle_data["omega_ex_n"][n] * (i + 0.5) ** 2
            + self.particle_data["omega_ey_n"][n] * (i + 0.5) ** 3
            + self.particle_data["omega_ez_n"][n] * (i + 0.5) ** 4
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
        # TODO: parameters for rigid rotator
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
        Heat capacity at constant volume
        $$c_{v} = c_{v,tr} + c_{v,rot}$$
        """
        return self.c_v_int(T) + self.c_v_tr()

    def c_v_tr(self, T=None):
        """
        Translational heat capacity at constant volume
        $$c_{v,tr} = \frac{3}{2}\frac{k}{m}$$
        """
        return (3 / 2) * self.constants["k_B"] / self.mass

    @abstractmethod
    def c_v_int(self, T):
        """
        Internal heat capacity at constant volume, defined in subclasses
        """

    def c_p(self, T):
        """
        Heat capacity at constant pressure
        $$C_{p} = \frac{k}{m} + C_{v}$$
        """
        return self.constants["k_B"] / self.mass + self.c_v(T)

    def c_p_tr(self, T=None):
        """
        Translational heat capacity at constant pressure
        $$C_{p,tr} = \frac{5}{2}\frac{k}{m}$$
        """
        return (5 / 2) * self.constants["k_B"] / self.mass

    def c_p_int(self, T):
        """
        Translational heat capacity at constant pressure
        $$C_{p,int} = C_{p}-C_{p,tr}$$
        """
        return self.c_p(T) - self.c_p_tr(T)

    def e(self, T):
        """
        Full unit energy
        $$e = e_{tr} + e_{int}$$
        """
        return self.e_tr(T) + self.e_int(T)

    def e_tr(self, T):
        """
        Translational unit energy
        $$e_{tr} = \frac{3}{2}\frac{k}{m}T$$
        """
        return (3 / 2) * (self.constants["k_B"] / self.mass) * T

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

    def _pressure_to_temperature(self, p):
        """
        Convert pressure to temperature using the ideal gas law.
        $$p = nkT$$

        Parameters:
        - p: Pressure in Pascals (Pa)

        Returns:
        Temperature in Kelvin (K)
        """
        # Ensure pressure and the Boltzmann constant are positive
        if p <= 0:
            raise ValueError("Pressure and Boltzmann constant must be positive.")

        # Calculate temperature using the ideal gas law
        T = p / (self.mole_number * self.constants["k_B"])

        return T

    def _check_pressure_and_temperature(self, T, p):
        """
        Check which of temperature and pressure if provided,
        and convert pressure to temperature if needed.
        Parameters:
        - p: Pressure in Pascals
        - T: Temperature in Kelvin

        Returns:
        Temperature in Kelvin
        """
        if T is not None and p is not None:
            raise ValueError(
                "Provide either temperature or pressure, not both. (p=value)"
            )

        # If pressure is provided, convert it to temperature
        if p is not None:
            if p <= 0:
                raise ValueError("Pressure must be positive.")
            T = self._pressure_to_temperature(p)

        if T <= 0:
            raise ValueError("Temperature must be positive.")

        return T


class Molecule(Particle):
    """
    Class for a molecule
    """

    _n_max_list = []
    _i_max_list = []
    _j_max_list = []

    def __init__(self, name):
        super().__init__(name)
        self._find_max_possible_nij()

    def _get_allowed_levels(self, n, i=None):
        if i is None:
            return self._j_max_list[n]
        else:
            return self._j_max_list[n][i]

    def _sum_over_gnij(self, expr):
        """
        Sum an expression multiplied by g_{i,j,n}:
        $$\sum_{nij}g^{c}_{n}g^{c}_{i}g^{c}_{j} expr(n,i,j)$$
        $$g_{i} = 1, g_{j} = 2j+1, g_{n} = 2n+1$$

        Parameters:
        - expr: function, depending on n,i,j

        Returns:
        (float) Computed sum
        """

        def g_i(i):  # g_{i} vibrational
            return 1

        def g_j(j):  # g_{j} rotational
            return 2 * j + 1

        def g_n(n):  # g_{n} electronic
            if self.parameters["g_n_tabulated"]:
                return self.particle_data["g_n"][n]
            return 2 * n + 1

        res = sum(
            g_i(i) * g_j(j) * g_n(n) * expr(n, i, j)
            for n in self._n_max_list
            for i in self._i_max_list[n]
            for j in self._j_max_list[n][i]
        )
        return res

    def _find_max_possible_nij(self):
        """
        Finding maximum possible energy state based on dissociation energy E_diss,
        for each electronic level n finds maximum i vibrational level,
        then for each vibrational level i on electronic level n finds maximum rotational level j
        """
        hc = self.constants["h"] * self.constants["c"]
        n_max_list = []
        i_max_list = []
        j_max_list = []

        for n in range(
            self.parameters["limit_energy_levels"]["molecule"]["electronic"]
        ):
            E_diss = self.particle_data["E_diss"][n] * hc * CM_TO_M  # converting to J

            if self._eps(n, 0, 0) >= E_diss:
                break

            n_max = n
            n_max_list.append(n_max)

            i_max_list_for_n = []
            j_max_list_for_n = []
            for i in range(
                self.parameters["limit_energy_levels"]["molecule"]["vibrational"]
            ):
                if self._eps(n, i, 0) >= E_diss or (
                    i != 0 and self._eps_v(n, i) < self._eps_v(n, i - 1)
                ):
                    break

                i_max = i
                i_max_list_for_n.append(i_max)

                j_max_list_for_i = []
                for j in range(
                    self.parameters["limit_energy_levels"]["molecule"]["rotational"]
                ):
                    if self._eps(n, i, j) >= E_diss or (
                        j != 0 and self._eps_r(n, i, j) < self._eps_r(n, i, j - 1)
                    ):
                        break

                    j_max = j
                    j_max_list_for_i.append(j_max)

                j_max_list_for_n.append(j_max_list_for_i)

            j_max_list.append(j_max_list_for_n)
            i_max_list.append(i_max_list_for_n)

        self._i_max_list = i_max_list
        self._j_max_list = j_max_list
        self._n_max_list = n_max_list

    def Z_int(self, T, p=None):
        """
        The internal partition function
        $$Z_{int,c} = \frac{1}{\sigma}
        \sum_{nij}g^{c}_{n}g^{c}_{i}g^{c}_{j}\exp{\Big(-\frac{\varepsilon^{c}_{nij}}{kT})}$$

        Parameters:
        - T: Temperature in Kelvin
        - p: Pressure in Pascal

        Returns:
        Partition function for internal energy
        """

        T = self._check_pressure_and_temperature(T, p)

        if self.parameters["enable_symmetry_factor"]:
            sigma = self.particle_data["sigma"]
        else:
            sigma = 1

        def expression(n, i, j):
            return np.exp(-self._eps(n, i, j) / (self.constants["k_B"] * T))

        Z = self._sum_over_gnij(expression) / sigma
        return Z

    def e_int(self, T, p=None):
        """
        Unit internal energy
        $$e_{int,c} =
        \frac{1}{m_{c}Z_{int,c}\sum_{nij}g_{c,nij}\varepsilon^{c}_{nij}\exp{(\frac{-\varepsilon^{c}_{nij}}{kT})}$$

        Parameters:
        - T: Temperature in Kelvin
        - p: Pressure in Pascal

        Returns:
        Unit internal energy
        """
        T = self._check_pressure_and_temperature(T, p)

        def summand(n, i, j):
            return self._eps(n, i, j) * np.exp(
                -self._eps(n, i, j) / (self.constants["k_B"] * T)
            )

        emZ = self._sum_over_gnij(summand)
        e = emZ / ((self.mass) * self.Z_int(T))
        return e

    def c_v_int(self, T, p=None):
        """
        Internal unit heat capacity at constant volume
        $$C_{v,int} = \frac{dE_{v,int}}{dT}$$

         Parameters:
        - T: Temperature in Kelvin
        - p: Pressure in Pascals

        Returns:
        Internal unit heat capacity at constant volume

        """
        k = self.constants["k_B"]
        T = self._check_pressure_and_temperature(T, p)

        def summand1(n, i, j):
            return self._eps(n, i, j) ** 2 * np.exp(-self._eps(n, i, j) / (k * T))

        def summand2(n, i, j):
            return self._eps(n, i, j) * np.exp(-self._eps(n, i, j) / (k * T))

        term1 = self._sum_over_gnij(summand1) / (k * T) ** 2 / self.Z_int(T)
        term2 = self._sum_over_gnij(summand2) / (k * T) / self.Z_int(T)

        return (term1 - term2**2) * self.constants["k_B"] / (self.mass)


class Atom(Particle):
    """
    Class for an atom
    """

    def __init__(self, name):
        super().__init__(name)
        self.n_max = self.parameters["limit_energy_levels"]["atom"]["electronic"]

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
        e = emZ / (self.mass * self.Z_int(T))
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
        return self.constants["k_B"] / self.mass * (term1 - term2**2)

    def Z_int(self, T):
        """
        Partition function for internal energy
        """
        Z = self._sum_over_gn(
            lambda n: np.exp(-self._eps_el(n) / (self.constants["k_B"] * T))
        )
        return Z
