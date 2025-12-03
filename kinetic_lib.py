"""Kinetic library version 0.04
with the capability to calculate various thermodynamic properties for arbitrary temperature:

    Statistical Sums: Total/Translational/Internal
    Specific Energy: Total/Translational/Internal
    Specific Heat at Constant Volume: Total/Translational/Internal

The calculations for these properties take into account translational/electronic degrees of freedom
for atoms and translational/electronic/vibrational/rotational degrees of freedom for molecules.
In the case of molecules,
the rotational energy depend on the electronic and vibrational levels,
and the vibrational energy depend on the electronic level.


"""

import json
from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

CM_TO_M = 100
G_TO_KG = 1000


class Particle(ABC):
    """
    Class for a particle, with derived classes of Molecule and Atom
    """

    particle_catalog: dict[str, dict[str, Any]] = {}
    particle_data: dict[str, dict[str, Any]] = {}
    constants: dict[str, float] = {}
    parameters: dict[str, Any] = {}
    _allowed_levels: Sequence[Any] = []  # allowed energy levels

    @staticmethod
    def _as_bool(value: Any, default: bool = False) -> bool:
        """
        Normalize configuration values that may be stored as strings.
        """
        if value is None:
            return default
        if isinstance(value, str):
            return value.strip().lower() in {"true", "1", "yes", "on"}
        return bool(value)

    @staticmethod
    def _ensure_number(value: Any, name: str) -> float:
        """
        Ensure configuration values used in arithmetic are numeric.
        """
        if not isinstance(value, (int, float)):
            raise TypeError(f"{name} must be numeric, got {value!r}")
        return float(value)

    def _get_limit(self, *keys: str) -> int:
        """
        Retrieve nested numeric limits from parameters with validation.
        """
        data: Any = self.parameters
        path: list[str] = []
        for key in keys:
            path.append(key)
            if not isinstance(data, dict) or key not in data:
                raise KeyError(f"Missing parameter: {'/'.join(path)}")
            data = data[key]
        return int(self._ensure_number(data, ".".join(keys)))

    def _get_constant(self, name: str) -> float:
        """
        Fetch numeric constants with validation.
        """
        if name not in self.constants:
            raise KeyError(f"Missing constant: {name}")
        return self._ensure_number(self.constants[name], name)

    def _get_particle_numeric(self, key: str) -> float:
        """
        Fetch numeric particle property with validation.
        """
        if key not in self.particle_data:
            raise KeyError(f"Missing particle property: {self.name}.{key}")
        return self._ensure_number(self.particle_data[key], f"{self.name}.{key}")

    def _get_particle_series(self, key: str) -> list[float]:
        """
        Fetch a numeric sequence from particle data with validation.
        """
        if key not in self.particle_data:
            raise KeyError(f"Missing particle property: {self.name}.{key}")
        value = self.particle_data[key]
        if not isinstance(value, list):
            raise TypeError(f"{self.name}.{key} must be a list of numbers, got {type(value)!r}")
        return [
            self._ensure_number(item, f"{self.name}.{key}[{idx}]")
            for idx, item in enumerate(value)
        ]

    def __init__(self, name: str) -> None:
        """
        Particle initialization
        Spectroscopic data is coming from the particle_data dictionary.

        Parameters:
        - name (string): the name of the particle, e.g. N2, N
        """
        self.name = name
        if name not in self.particle_catalog:
            raise KeyError(f"Unknown particle '{name}'")
        self.particle_data: dict[str, Any] = self.particle_catalog[name]
        self.mole_number = 1
        molar_mass = self._get_particle_numeric("M")
        N_a = self._get_constant("N_a")
        self.mass = molar_mass / G_TO_KG / N_a
        self.R_specific = self._get_constant("k_B") * N_a / molar_mass * G_TO_KG

    def __repr__(self) -> str:  # to print in console environment
        return self.name

    def _eps(self, n: int, i: int, j: int) -> float:
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

    def _eps_el(self, n: int) -> float:
        """
        Electronic energy for n-th electronic state.
        Tabulated in particle_data.
        $$\varepsilon_{el} = E_{el}hc$$

        Parameters:
        - n: electronic level

        """
        hc = self._get_constant("h") * self._get_constant("c")
        eps_levels = self._get_particle_series("eps_el_n")
        eps_el = eps_levels[n] * hc * CM_TO_M
        return eps_el

    def _eps_v(self, n: int, i: int) -> float:
        """
        Vibrational energy for i-th vibrational level and n-th electronic state
        (only anharmonic oscillator model is available)
        $$\varepsilon^{c,n}_{i} = hc \Big(\omega^{c,n}_{e}\Big(i+\frac{1}{2}\Big)
        - \omega^{c,n}_{e}x^{c,n}_{e}\Big(i+\frac{1}{2}\Big)^2\Big)
        + \omega^{c,n}_{e}y^{c,n}_{e}\Big(i+\frac{1}{2}\Big)^3\Big)$$


        Parameters:
        - n: electronic level
        - i: vibrational level

        """
        hc = self._get_constant("h") * self._get_constant("c")
        vib_params = self.parameters.get("vibr_energy", {})
        include_anharmonic = self._as_bool(vib_params.get("anharmonicity"), default=True)
        series_terms = max(1, int(vib_params.get("anh_series_terms_number", 4)))
        base = i + 0.5
        omega_n = self._get_particle_series("omega_n")
        eps_divbyhc = omega_n[n] * base

        if include_anharmonic:
            omega_ex = self._get_particle_series("omega_ex_n")
            omega_ey = self._get_particle_series("omega_ey_n")
            omega_ez = self._get_particle_series("omega_ez_n")
            anharm_terms = [
                (-omega_ex[n], 2),
                (omega_ey[n], 3),
                (omega_ez[n], 4),
            ]
            for idx, (coeff, power) in enumerate(anharm_terms, start=2):
                if series_terms < idx:
                    break
                eps_divbyhc += coeff * base**power

        return eps_divbyhc * hc * CM_TO_M

    def _eps_r(self, n: int, i: int, j: int) -> float:
        """
        Rotational energy for i-th vibrational, j-th rotational levels
        and n-th electronic state
        $$\varepsilon^{c,ni}_{j} = hc \Big(B^{c}_{ni}j(j+1)-D^{c}_{ni}j^2(j+1)^2 + \dots\Big)$$
        $$B^{c}_{ni} = B^{c}_{n,e} - \alpha^{c}_{n,e}\Big(i + \frac{1}{2}\Big)$$
        $$D^{c}_{ni} = D^{c}_{n,e} - \beta^{c}_{n,e}\Big(i + \frac{1}{2}\Big)$$

        Parameters:
        - n: electronic level
        - i: vibrational level
        - j: rotational level
        """
        hc = self._get_constant("h") * self._get_constant("c")
        rot_params = self.parameters.get("rot_energy", {})
        rigid_rotor = self._as_bool(rot_params.get("rigid_rotator_model"), default=False)
        series_terms = max(1, int(rot_params.get("series_terms_number", 2)))
        j_term = j * (j + 1)

        B_values = self._get_particle_series("B_n")

        if rigid_rotor:
            eps_divbyhc = B_values[n] * j_term
            return eps_divbyhc * hc * CM_TO_M

        base = i + 0.5
        alpha_n = self._get_particle_series("alpha_n")
        B_ni = B_values[n] - alpha_n[n] * base
        eps_divbyhc = B_ni * j_term

        if series_terms >= 2:
            D_n = self._get_particle_series("D_n")
            beta_n = self._get_particle_series("beta_n")
            D_ni = D_n[n] - beta_n[n] * base
            eps_divbyhc -= D_ni * j_term**2

        return eps_divbyhc * hc * CM_TO_M

    def Z(self, T: float) -> float:
        """
        Partition Function
        $$Z = Z_{tr}Z_{int}$$

        Parameters:
        - T: Temperature in Kelvin
        """
        return self.Z_tr(T) * self.Z_int(T)

    def Z_tr(self, T: float) -> float:
        """
        Translational partition function
        $$Z_{tr,c} = \Big(\frac{2\pi m_{c}kT}{h^2}\Big)^{\frac{3}{2}}$$

        Parameters:
        - T (float): Temperature in Kelvin

        Returns:
        float: Translational partition function.
        """
        assert T > 0, "Temperature must be a positive number."
        k_B = self._get_constant("k_B")
        h = self._get_constant("h")
        return np.power((2 * np.pi * self.mass * k_B * T / h**2), 1.5)

    @abstractmethod
    def Z_int(self, T: float, p: float | None = None) -> float:
        """
        Internal partition function, defined in subclasses
        """

    def c_v(self, T: float) -> float:
        """
        Heat capacity at constant volume
        $$c_{v} = c_{v,tr} + c_{v,rot}$$
        """
        return self.c_v_int(T) + self.c_v_tr()

    def c_v_tr(self, T: float | None = None) -> float:
        """
        Translational heat capacity at constant volume
        $$c_{v,tr} = \frac{3}{2}\frac{k}{m}$$
        """
        return (3 / 2) * self._get_constant("k_B") / self.mass

    def c_v_int(self, T: float | None, p: float | None = None) -> float:
        """
        Internal unit heat capacity at constant volume
        $$C_{v,int} = \frac{dE_{v,int}}{dT}$$

         Parameters:
        - T: Temperature in Kelvin
        - p: Pressure in Pascals

        Returns:
        Internal unit heat capacity at constant volume

        """
        temperature = self._check_pressure_and_temperature(T, p)

        esquared = self.e_int(temperature, squared=True)
        squareofe = self.e_int(temperature) ** 2 * self.mass

        c_v = (esquared - squareofe) / (self._get_constant("k_B") * temperature**2)

        return c_v

    def c_p(self, T: float) -> float:
        """
        Heat capacity at constant pressure
        $$C_{p} = \frac{k}{m} + C_{v}$$
        """
        return self._get_constant("k_B") / self.mass + self.c_v(T)

    def c_p_tr(self, T: float | None = None) -> float:
        """
        Translational heat capacity at constant pressure
        $$C_{p,tr} = \frac{5}{2}\frac{k}{m}$$
        """
        return (5 / 2) * self._get_constant("k_B") / self.mass

    def c_p_int(self, T: float) -> float:
        """
        Translational heat capacity at constant pressure
        $$C_{p,int} = C_{p}-C_{p,tr}$$
        """
        return self.c_p(T) - self.c_p_tr(T)

    def e(self, T: float) -> float:
        """
        Full unit energy
        $$e = e_{tr} + e_{int}$$
        """
        return self.e_tr(T) + self.e_int(T)

    def e_tr(self, T: float) -> float:
        """
        Translational unit energy
        $$e_{tr} = \frac{3}{2}\frac{k}{m}T$$
        """
        return (3 / 2) * (self._get_constant("k_B") / self.mass) * T

    @abstractmethod
    def e_int(self, T: float, p: float | None = None, squared: bool = False) -> float:
        """
        Internal unit energy, defined in subclasses
        """

    @classmethod
    def load_particle_data(
        cls,
        particle_path: str = "particle_data.json",
        constants_path: str = "constants.json",
        parameters_path: str = "parameters.json",
    ) -> None:
        """Loads particle data, constants and model settings from json"""
        with open(particle_path, "r", encoding="UTF-8") as file:
            cls.particle_catalog = json.load(file)
            cls.particle_data = cls.particle_catalog

        with open(constants_path, "r", encoding="UTF-8") as file:
            cls.constants = json.load(file)

        with open(parameters_path, "r", encoding="UTF-8") as file:
            cls.parameters = json.load(file)

    def _pressure_to_temperature(self, p: float) -> float:
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
            raise ValueError("Pressure must be positive")

        # Calculate temperature using the ideal gas law
        T = p / (self.mole_number * self._get_constant("k_B"))

        return T

    def _check_pressure_and_temperature(self, T: float | None, p: float | None) -> float:
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
            raise ValueError("Provide either temperature or pressure, not both. (p=value)")

        # If pressure is provided, convert it to temperature
        if p is not None:
            if p <= 0:
                raise ValueError("Pressure must be positive.")
            T = self._pressure_to_temperature(p)

        if T is None:
            raise ValueError("Temperature must be provided.")

        if T <= 0:
            raise ValueError("Temperature must be positive.")

        return T


class Molecule(Particle):
    """
    Class for a molecule
    """

    def __init__(self, name: str) -> None:
        super().__init__(name)
        self._find_max_possible_nij()

    def _get_allowed_levels(self, n: int, i: int | None = None) -> Sequence[int]:
        if i is None:
            return self._allowed_levels[n]
        else:
            return self._allowed_levels[n][i]

    def _sum_over_gnij(self, expr: Callable[[int, int, int], float]) -> float:
        """
        Sum an expression multiplied by g_{i,j,n}:
        $$\sum_{nij}g^{c}_{n}g^{c}_{i}g^{c}_{j} expr(n,i,j)$$
        $$g_{i} = 1, g_{j} = 2j+1, g_{n} = 2n+1$$

        Parameters:
        - expr: function, depending on n,i,j

        Returns:
        (float) Computed sum
        """

        def g_i(i: int) -> int:  # g_{i} vibrational
            return 1

        def g_j(j: int) -> int:  # g_{j} rotational
            return 2 * j + 1

        g_n_values = self._get_particle_series("g_n") if self._as_bool(
            self.parameters.get("g_n_tabulated"), default=False
        ) else None

        def g_n(n: int) -> float:  # g_{n} electronic
            if g_n_values is not None:
                return g_n_values[n]
            return 2 * n + 1

        res = sum(
            g_i(i) * g_j(j) * g_n(n) * expr(n, i, j)
            for n in range(len(self._allowed_levels))
            for i in range(len(self._allowed_levels[n]))
            for j in self._allowed_levels[n][i]
        )
        return res

    def _find_max_possible_nij(self) -> None:
        """
        Finding maximum possible energy state based on dissociation energy E_diss,
        for each electronic level n finds maximum i vibrational level,
        then for each vibrational level i on electronic level n finds maximum rotational level j
        """
        hc = self._get_constant("h") * self._get_constant("c")
        n_max_list = []
        i_max_list = []
        j_max_list = []

        electronic_limit = self._get_limit("limit_energy_levels", "molecule", "electronic")
        vibrational_limit = self._get_limit("limit_energy_levels", "molecule", "vibrational")
        rotational_limit = self._get_limit("limit_energy_levels", "molecule", "rotational")

        E_diss_levels = self._get_particle_series("E_diss")

        for n in range(electronic_limit):
            E_diss = E_diss_levels[n] * hc * CM_TO_M  # converting to J

            if self._eps(n, 0, 0) >= E_diss:
                break

            n_max = n
            n_max_list.append(n_max)

            i_max_list_for_n = []
            j_max_list_for_n = []
            for i in range(vibrational_limit):
                if self._eps(n, i, 0) >= E_diss or (
                    i != 0 and self._eps_v(n, i) < self._eps_v(n, i - 1)
                ):
                    break

                i_max = i
                i_max_list_for_n.append(i_max)

                j_max_list_for_i = []
                for j in range(rotational_limit):
                    if self._eps(n, i, j) >= E_diss or (
                        j != 0 and self._eps_r(n, i, j) < self._eps_r(n, i, j - 1)
                    ):
                        break

                    j_max = j
                    j_max_list_for_i.append(j_max)

                j_max_list_for_n.append(j_max_list_for_i)

            j_max_list.append(j_max_list_for_n)
            i_max_list.append(i_max_list_for_n)

        assert len(n_max_list) > 0, "No allowed electronic states"
        assert len(i_max_list[0]) > 0, "No allowed vibrational levels"
        assert len(j_max_list[0][0]) > 0, "No allowed rotational levels"
        self._allowed_levels = j_max_list

    def Z_int(self, T: float | None, p: float | None = None) -> float:
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

        temperature = self._check_pressure_and_temperature(T, p)
        k_B = self._get_constant("k_B")

        sigma = (
            self._get_particle_numeric("sigma")
            if self._as_bool(self.parameters.get("enable_symmetry_factor"), default=False)
            else 1.0
        )

        def expression(n, i, j):
            return np.exp(-self._eps(n, i, j) / (k_B * temperature))

        Z = self._sum_over_gnij(expression) / sigma
        return Z

    def e_int(self, T: float | None, p: float | None = None, squared: bool = False) -> float:
        """
        Unit internal energy
        $$e_{int,c} =
        \frac{1}{m_{c}Z_{int,c}\sum_{nij}g_{c,nij}\varepsilon^{c}_{nij}\exp{(\frac{-\varepsilon^{c}_{nij}}{kT})}$$

        Parameters:
        - T: Temperature in Kelvin
        - p: Pressure in Pascal
        - squared: Whether to compute \varepsilon^{c}_{nij} or (\varepsilon^{c}_{nij}) before exponent

        Returns:
        Unit internal energy
        """
        temperature = self._check_pressure_and_temperature(T, p)
        k_B = self._get_constant("k_B")

        sigma = (
            self._get_particle_numeric("sigma")
            if self._as_bool(self.parameters.get("enable_symmetry_factor"), default=False)
            else 1.0
        )

        def summand(n, i, j):
            if squared:
                return self._eps(n, i, j) ** 2 * np.exp(-self._eps(n, i, j) / (k_B * temperature))
            return self._eps(n, i, j) * np.exp(-self._eps(n, i, j) / (k_B * temperature))

        emZ = self._sum_over_gnij(summand)
        e = emZ / (self.mass * self.Z_int(temperature))
        return e / sigma


class Atom(Particle):
    """
    Class for an atom
    """

    def __init__(self, name: str) -> None:
        super().__init__(name)
        atom_limit = self._get_limit("limit_energy_levels", "atom", "electronic")
        self._allowed_levels = range(atom_limit)

    def _sum_over_gn(self, term: Callable[[int], float]) -> float:
        """
        Sum an expression multiplied by g_{n}:
        $$\sum_{n}g^{c}_{n} expr(n)$$

        Parameters:
        - expr: function, depending on n

        Returns:
        (float) Computed sum
        """
        res = sum(
            (2 * n + 1) * term(n) for n in range(len(self._allowed_levels))
        )  # g_{n} electronic
        return res

    def e_int(
        self,
        T: float | None,
        p: float | None = None,
        squared: bool = False,
    ) -> float:
        """
        Unit internal energy
        for atom summing only over electronic states
        $$e_{int,c} =
        \frac{1}{m_{c}Z_{int,c}\sum_{n}g_{c,n}\varepsilon^{c}_{n}\exp{(\frac{-\varepsilon^{c}_{n}}{kT})}$$

        Parameters:
        - T: Temperature in Kelvin
        - p: Pressure in Pascal
        - squared: Whether to compute \varepsilon^{c}_{nij} or (\varepsilon^{c}_{nij}) before exponent

        Returns:
        Unit internal energy
        """

        temperature = self._check_pressure_and_temperature(T, p)
        k_B = self._get_constant("k_B")

        def summand(n):
            if squared:
                return self._eps_el(n) ** 2 * np.exp(-self._eps_el(n) / (k_B * temperature))
            return self._eps_el(n) * np.exp(-self._eps_el(n) / (k_B * temperature))

        emZ = self._sum_over_gn(summand)
        e = emZ / (self.mass * self.Z_int(temperature))
        return e

    def Z_int(self, T: float | None, p: float | None = None) -> float:
        """
        Partition function for internal energy

        Parameters:
        - T: Temperature in Kelvin
        - p: Pressure in Pascal
        """
        temperature = self._check_pressure_and_temperature(T, p)
        k_B = self._get_constant("k_B")
        Z = self._sum_over_gn(lambda n: np.exp(-self._eps_el(n) / (k_B * temperature)))
        return Z
