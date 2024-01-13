import numpy as np
from numpy.testing import assert_almost_equal, assert_array_almost_equal
from kinetic_lib import Molecule, Particle


def test_tests():
    assert True


def test_mass(load_data):
    """
    Test N2 mass
    """
    N2 = Molecule("N2")
    assert_almost_equal(N2.mass, 4.6517344343135997e-26, 10)


def test_R_specific(load_data):
    """
    Test N2 R_specific
    """
    N2 = Molecule("N2")
    assert_almost_equal(N2.R_specific, 296.8, 2)


def test_cv_tr(load_data):
    """
    Test N2 Cv_tr
    """
    N2 = Molecule("N2")
    assert_almost_equal(N2.c_v_tr(296.8), 445.205, 3)


def test_eps_v(load_data):
    """
    Test vibrational spectrum of N2
    """
    N2 = Molecule("N2")
    spectr_r = np.array(
        [
            0.145769,
            0.434641,
            0.719958,
            1.00172,
            1.27991,
            1.55454,
            1.8256,
            2.09308,
            2.35698,
            2.61728,
            2.87398,
            3.12708,
            3.37655,
            3.6224,
            3.86461,
            4.10316,
            4.33805,
            4.56927,
            4.79679,
            5.02061,
            5.24071,
            5.45707,
            5.66968,
            5.87852,
            6.08357,
            6.28481,
            6.48222,
            6.67579,
            6.86549,
            7.0513,
            7.2332,
            7.41116,
            7.58516,
            7.75518,
            7.92119,
            8.08317,
            8.24108,
            8.3949,
            8.54461,
            8.69017,
            8.83156,
            8.96874,
            9.10168,
            9.23035,
            9.35473,
            9.47477,
            9.59044,
            9.70171,
        ]
    )
    spectr_computed = np.zeros(spectr_r.shape)
    Q_e = 1.602176565e-19
    for i in range(len(spectr_r)):
        spectr_computed[i] = N2._eps_v(0, i) / Q_e
    assert_array_almost_equal(spectr_computed, spectr_r, 2)


def test_allowed_electronic_lvls(load_data):
    N2 = Molecule("N2")
    assert_almost_equal(len(N2._allowed_levels), 5)


def test_allowed_vibrational_levels(load_data):
    N2 = Molecule("N2")
    assert_almost_equal(len(N2._allowed_levels[0]), 48)
