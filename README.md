# Simple kinetic library

## Overview

This repository contains a study project focused on developing a Python library for thermodynamic calculations. The goal is to create a program capable of computing various thermodynamic properties at arbitrary temperature. The calculations will consider the degrees of freedom for atoms and molecules, incorporating translational, electronic, vibrational, and rotational contributions.

## Features

The library provides functions for the following thermodynamic properties:

1. Statistical Sums: Total/Translational/Internal
2. Specific Energy: Total/Translational/Internal
3. Specific Heat at Constant Volume: Total/Translational/Internal

## Usage

The library is intended for educational purposes and can be used to explore thermodynamic concepts. Users can perform calculations based on temperature and particle type, file "main.py" is provided as example.

### Quick start

```bash
pip install -r requirements.txt
python main.py
```

### Programmatic usage

```python
from kinetic_lib import Molecule, Particle

Particle.load_particle_data()
nitrogen = Molecule("N2")
temperature = 1200.0

print("Partition:", nitrogen.Z(temperature))
print("Internal energy:", nitrogen.e_int(temperature))
print("Heat capacity:", nitrogen.c_p(temperature))
```

## Limitations

This project is a study endeavor, and as such, there may be mistakes and under-developed features. Users are encouraged to review the code critically and contribute to its improvement.

### Contributing

Feel free to contribute to the project by forking and fixing it. 

## License

This project is licensed under the MIT License.
