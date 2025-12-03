import copy

import numpy as np
from numpy.testing import assert_almost_equal, assert_array_almost_equal

from kinetic_lib import CM_TO_M, Molecule, Particle


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


def test_harmonic_vibrational_model(load_data):
    original_vibr = copy.deepcopy(Particle.parameters["vibr_energy"])
    Particle.parameters["vibr_energy"]["anharmonicity"] = False
    Particle.parameters["vibr_energy"]["anh_series_terms_number"] = 1

    try:
        N2 = Molecule("N2")
        level = 3
        base = level + 0.5
        expected = (
            N2.particle_data["omega_n"][0] * base * N2.constants["h"] * N2.constants["c"] * CM_TO_M
        )
        assert_almost_equal(N2._eps_v(0, level), expected)
    finally:
        Particle.parameters["vibr_energy"] = original_vibr


def test_rigid_rotator_model(load_data):
    original_rot = copy.deepcopy(Particle.parameters["rot_energy"])
    Particle.parameters["rot_energy"]["rigid_rotator_model"] = True

    try:
        N2 = Molecule("N2")
        j = 10
        expected = (
            N2.particle_data["B_n"][0]
            * j
            * (j + 1)
            * N2.constants["h"]
            * N2.constants["c"]
            * CM_TO_M
        )
        assert_almost_equal(N2._eps_r(0, 0, j), expected)
    finally:
        Particle.parameters["rot_energy"] = original_rot


def test_rotational_series_terms(load_data):
    original_rot = copy.deepcopy(Particle.parameters["rot_energy"])

    try:
        N2 = Molecule("N2")
        Particle.parameters["rot_energy"]["rigid_rotator_model"] = False
        Particle.parameters["rot_energy"]["series_terms_number"] = 2
        j = 8
        with_second_term = N2._eps_r(0, 0, j)

        Particle.parameters["rot_energy"]["series_terms_number"] = 1
        without_second_term = N2._eps_r(0, 0, j)

        j_term = j * (j + 1)
        base = 0 + 0.5
        D_ni = N2.particle_data["D_n"][0] - N2.particle_data["beta_n"][0] * base
        expected_delta = -D_ni * (j_term**2) * N2.constants["h"] * N2.constants["c"] * CM_TO_M

        assert_almost_equal(with_second_term - without_second_term, expected_delta)
    finally:
        Particle.parameters["rot_energy"] = original_rot
