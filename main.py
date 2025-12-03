"""Command-line sample demonstrating kinetic_lib computations."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from kinetic_lib import Atom, Molecule, Particle


@dataclass
class ResultRow:
    name: str
    temperature: float
    Z_tr: float
    Z_int: float
    e_total: float
    c_p: float


def evaluate_species(species: Iterable[str], temperature: float) -> list[ResultRow]:
    rows: list[ResultRow] = []
    for name in species:
        particle = Molecule(name) if len(name) > 1 else Atom(name)
        rows.append(
            ResultRow(
                name=name,
                temperature=temperature,
                Z_tr=particle.Z_tr(temperature),
                Z_int=particle.Z_int(temperature),
                e_total=particle.e(temperature),
                c_p=particle.c_p(temperature),
            )
        )
    return rows


def print_table(rows: list[ResultRow]) -> None:
    header = (
        f"{'Species':<8}{'T [K]':>10}{'Z_tr':>15}{'Z_int':>15}{'e [J/kg]':>15}{'c_p [J/kg/K]':>18}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{row.name:<8}{row.temperature:>10.2f}{row.Z_tr:>15.2e}{row.Z_int:>15.2e}{row.e_total:>15.2f}{row.c_p:>18.2f}"
        )


def main() -> None:
    Particle.load_particle_data()
    baseline_temp = 298.15
    high_temp = 2000.0
    species = ["N2"]

    print("=== Baseline conditions ===")
    table = evaluate_species(species, baseline_temp)
    print_table(table)

    print("\n=== High-temperature conditions ===")
    print_table(evaluate_species(species, high_temp))


if __name__ == "__main__":
    main()
