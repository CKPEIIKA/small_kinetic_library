"""The main module to show the usage"""

from kinetic_lib import Atom, Molecule, Particle

Particle.load_particle_data()
T1 = 298.15
# T1 = 1000
N2 = Molecule("N2")
N = Atom("N")
print("Z_tr", N2.Z_tr(T1))
print("Z_int", N2.Z_int(T1))
print("Z", N2.Z_int(T1))
print("c_v_tr", N2.c_v_tr(T1))
print("c_v_int", N2.c_v_int(T1))
print("c_v", N2.c_v(T1))
print("e_tr", N2.c_v_tr(T1))
print("e_int", N2.c_v_int(T1))
print("e", N2.c_v(T1))

print(N2.c_p(T1) / N2.R_specific)  # * N2.e_int(T1) * T1)
print(N2.e_int(T1) / T1 * N2.R_specific)
