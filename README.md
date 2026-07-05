# PysTDA

PysTDA is an open, PySCF-compatible platform for prototyping semi-empirical excited-state methods. It separates method-specific approximations for the Tamm-Dancoff approximation (TDA) matrix from shared solver and analysis infrastructure, making it straightforward to test new approximations while reusing dense diagonalization, Davidson iteration, PySCF checkpoint loading, and oscillator-strength utilities.

## Features

- sTDA and XsTDA reference implementations.
- Dense diagonalization through SciPy.
- Optional Davidson solver through PySCF.
- PySCF checkpoint loading for molecular orbital data.
- Apache-2.0 license for open method development.

## Installation

```bash
pip install .
```

For checkpoint loading, AO integrals, and Davidson iteration, install the optional PySCF dependency:

```bash
pip install .[pyscf]
```

## Basic Usage

```python
import system_env, utils
from sTDA import sTDA

pmol, mf = utils.read_chk_pyscf("acetamide.chk")
mol = system_env.molecule(mf.mo_coeff, mf.mo_energy, mf.mo_occ,
    pmol.aoslice_by_atom(), pmol.atom_charges(), pmol.atom_coords(),
    basis=pmol.basis, int1e_ovlp=mf.get_ovlp(), cart=pmol.cart)

stda = sTDA(mol, ax=0.25, nroot=10)
print(stda.kernel_davidson()[0] * 27.21138)
```

Use `kernel_diag()` for small dense calculations and `kernel_davidson()` for larger truncated excitation spaces.

## Implementing a New Method

New semi-empirical excited-state methods should subclass `TDA_base`. A method implementation supplies the approximated TDA matrix and its matrix-vector product:

- `get_A_matrix()` builds the dense TDA matrix for validation and small systems.
- `matvec_A(x)` applies the TDA matrix without explicitly building it.
- `matmat_A(X)` can be overridden to batch Davidson matrix products for `X.shape == (nov, nvec)`.
- `get_diag()` can be overridden to provide an exact matrix-free Davidson preconditioner; the base class uses orbital energy gaps.

The shared base class handles eigenvalue solvers, conversion to full occupied-virtual amplitudes, and oscillator strengths. See `examples/new_method_template.py` for a minimal starting point.

## License And Attribution

PysTDA is licensed under Apache-2.0. Portions of the oscillator-strength and transition-multipole logic are adapted from PySCF TD-SCF utilities; see `NOTICE` for attribution.
