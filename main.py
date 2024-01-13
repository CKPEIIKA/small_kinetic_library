"""The main module to show the example of library usage"""

from kinetic_lib import Atom, Molecule, Particle

Particle.load_particle_data()
T1 = 298.15
T2 = 1000
N2 = Molecule("N2")
T = T1

print(f"Results for N2 at Temperature: {T}")
print("\nPartition Functions:")
print("Z_tr:", N2.Z_tr(T))
print("Z_int:", N2.Z_int(T))  # = 5288 Capitelli at T=298.15
print("Total Partition Function (Z):", N2.Z(T))

print("\nInternal Energy:")
print("e_tr:", N2.e_tr(T))
print("e_int:", N2.e_int(T))
print("Total Internal Energy (e):", N2.e(T))

print("\nHeat Capacities:")
print("c_v_tr:", N2.c_v_tr(T))
print("c_v_int:", N2.c_v_int(T))
print("Total c_v:", N2.c_v(T))

print("\nc_p_tr:", N2.c_p_tr(T))
print("c_p_int:", N2.c_p_int(T))
print("Total c_p:", N2.c_p(T))
# C_p = 2911 J/mol/K  = 1038.9 J/kg/K Capitelli at T=298.15

print("\nAdditional Relations:")
print("C_p_int/R:", N2.c_p_int(T) / N2.R_specific)
# C_p_int/R = 1.003 Capitelli at T=298.15
print("E_int/RT:", N2.e_int(T) / (T * N2.R_specific))
# E_int/RT = 0.9979 Capitelli at T=298.15
