"""Kinetic library"""

import json
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
        self.info = self.particle_data[name]
        self.mass = self.info["mass"]

    def __repr__(self):  # repr
        return self.name

    def _calc_eps(self, n, i, j):
        """
        Internal energy for i-th vibrational, j-th rotational
        and n-th vibrational quantum number
        """
        return self._calc_eps_el(n) + self._calc_eps_v + self._calc_eps_r

    def _calc_eps_el(self, n):
        """
        Electronic energy for n-th
        """
        hc = self.constants["h"] * self.constants["c"]
        return self.particle_data["eps_el_n"][n] * hc * 100

    def _calc_eps_v(self, i, n):
        """
        Vibrational energy for i-th vibrational
        and n-th vibrational quantum number
        """
        hc = self.constants["h"] * self.constants["c"]
        # wy, wz?
        #  model with I_c
        eps_divbyhc = (
            self.particle_data["omega_e_n"][n] * (i + 0.5)
            - self.particle_data["omega_ex_e_n"][n] * (i + 0.5)
            ^ 2
        )
        return eps_divbyhc * hc * 100

    def _calc_eps_r(self, i, j, n):
        hc = self.constants["h"] * self.constants["c"]
        B_ni = self.particle_data["B_n"][n] - self.particle_data["alpha_n"][n] * (
            i + 0.5
        )
        D_ni = self.particle_data["D_n"][n]  # - beta...
        eps_divbyhc = B_ni * j * (j + 1) - D_ni * j ^ 2 * (j + 1) ^ 2
        return eps_divbyhc * hc * 100

    def calc_Z(self, T):
        """
        Partition Function
        $$Z = Z_{tr}Z_{int}$$
        """
        return self.calc_Z_tr(T) + self.calc_Z_int(T)

    def calc_Z_tr(self, T):
        """
        Translational partition function
        """
        return np.power(
            (
                2
                * np.pi
                * self.particle_data["m"]
                * self.constants["k"]
                * T
                / self.constants["h"]
                ^ 2
            ),
            1.5,
        )

    @classmethod
    def load_particle_data(cls, particle_path, constants_path, parameters_path):
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

    def __init__(self, name, i_max=3, j_max=3, n_max=5):
        super().__init__(name)

    def calc_Z_int(self, T):
        """
        Partition function for internal energy
        """

        return 0


class Atom(Particle):
    """
    Class for a atom
    """

    def __init__(self, name):
        super().__init__(name)

    def calc_Z_int(self, T):
        return 0
