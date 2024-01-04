"""Kinetic library"""

import json
import numpy as np

# расчет статсумм, теплоемкости
# перенос расчета вязкости по приближенным формулам
# возможность использования регрессора
# см. каппу


class Particle:
    """
    Class for a particle
    """

    # Class property to store molecular information
    molecular_info = {}
    # Class property to store constants
    constants = {}

    def __init__(self, name):
        self.name = name
        self.info = self.particle_info[name]
        self.mass = self.info["mass"]

    def __repr__(self):  # repr
        return self.name

    @classmethod
    def load_particle_data(cls, file_path):
        """Loads particle data from json"""
        with open(file_path, "r", encoding="UTF-8") as file:
            cls.particle_info = json.load(file)

    @classmethod
    def load_constants(cls, file_path):
        with open(file_path, "r", encoding="UTF-8") as file:
            cls.constants = json.load(file)


class Molecule(Particle):
    """
    Class for a molecule
    """

    def __init__(self, name, tr_dof=0, el_dof=0, vib_dof=0):
        super().__init__(name)


class Atom(Particle):
    """
    Class for a atom
    """

    def __init__(self, name, tr_dof=0, el_dof=0, vib_dof=0):
        super().__init__(name)
