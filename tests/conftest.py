import os

import pytest

from kinetic_lib import Particle


@pytest.fixture
def load_data():
    data_path = os.path.dirname(os.path.dirname(__file__))
    print(data_path)
    return Particle.load_particle_data(
        data_path + "/particle_data.json",
        data_path + "/constants.json",
        data_path + "/parameters.json",
    )
