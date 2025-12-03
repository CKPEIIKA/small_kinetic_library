from numpy.testing import assert_allclose, assert_almost_equal

from kinetic_lib import Atom, Molecule, Particle


def test_partition_function_consistency(load_data):
    n2 = Molecule("N2")
    temp = 500.0
    assert_allclose(n2.Z(temp), n2.Z_tr(temp) * n2.Z_int(temp), rtol=1e-6)


def test_energy_relationship(load_data):
    n2 = Molecule("N2")
    temp = 750.0
    assert_allclose(n2.e(temp), n2.e_tr(temp) + n2.e_int(temp), rtol=1e-6)


def test_heat_capacity_relationship(load_data):
    n2 = Molecule("N2")
    temp = 600.0
    expected_cp = n2.c_v(temp) + Particle.constants["k_B"] / n2.mass
    assert_allclose(n2.c_p(temp), expected_cp, rtol=1e-6)


def test_pressure_temperature_equivalence(load_data):
    n2 = Molecule("N2")
    temp = 900.0
    equivalent_pressure = temp * n2.mole_number * Particle.constants["k_B"]
    from_temperature = n2.c_v_int(temp)
    from_pressure = n2.c_v_int(None, p=equivalent_pressure)
    assert_allclose(from_temperature, from_pressure, rtol=1e-6)


def test_atom_properties(load_data):
    nitrogen = Atom("N")
    temp = 800.0
    partition = nitrogen.Z_int(temp)
    assert partition > 0
    assert nitrogen.e_int(temp) > 0
    assert_almost_equal(nitrogen.e(temp), nitrogen.e_tr(temp) + nitrogen.e_int(temp))
