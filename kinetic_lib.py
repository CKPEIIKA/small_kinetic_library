"""Kinetic library"""

# расчет статсумм, теплоемкости
# перенос расчета вязкости по приближенным формулам
# возможность использования регрессора
# см. каппу


class Particle:
    """
    Class for a particle
    """

    def __init__(self, name):
        self.name = name

    def __repr__(self):  # repr
        return self.name


class Molecule(Particle):
    """
    Class for a molecule
    """


class Atom(Particle):
    """
    Class for a atom
    """
