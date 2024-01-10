"""The main module to show the example of library usage"""

from kinetic_lib import Atom, Molecule, Particle

Particle.load_particle_data()
T1 = 298.15
T2 = 1000
N2 = Molecule("N2")
N = Atom("N")
T = T1


print(
    f"Results for N2 at Temperature: {T} K with "
    f"{N2.n_max+1} electronic states, {N2.i_max+1} vibrational energy levels, "
    f"and {N2.j_max+1} rotational energy levels:"
)

print("\nPartition Functions:")
print("Z_tr:", N2.Z_tr(T))
print("Z_int:", N2.Z_int(T))
print("Total Partition Function (Z):", N2.Z(T))

print("\nSpecific Heat Capacities (J/(mol·K)):")
print("c_v_tr:", N2.c_v_tr(T))
print("c_v_int:", N2.c_v_int(T))
print("Total c_v:", N2.c_v(T))

print("\nc_p_int:", N2.c_p_int(T))
print("Total c_p:", N2.c_p(T))

print("\nInternal Energy (J/mol):")
print("e_tr:", N2.e_tr(T))
print("e_int:", N2.e_int(T))
print("Total Internal Energy (e):", N2.e(T))

print("\nAdditional Relations:")
print("c_p_int/R:", N2.c_p_int(T) / N2.R_specific)
print("e_int/RT:", N2.e_int(T) / (T * N2.R_specific))
