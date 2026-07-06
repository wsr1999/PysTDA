# Contributing To PysTDA

PysTDA is intended as an open platform for developing semi-empirical excited-state methods. Contributions should keep method-specific approximations separate from shared solver and analysis infrastructure.

## Adding A New Method

Add a new method by subclassing `TDA_base`. The implementation should provide:

- `get_A_matrix()`: dense TDA matrix for small systems and validation.
- `matvec_A(x)`: matrix-free TDA matrix-vector product for Davidson iteration.
- optional `matmat_A(X)`: batched matrix-free TDA products for efficient Davidson iteration.
- optional `get_diag()`: exact matrix-free diagonal for Davidson preconditioning. The base class uses orbital energy gaps.

Keep method parameters explicit in the class constructor and cache expensive intermediates only when they are reused.
The Davidson preconditioner diagonal is cached after first use, so method parameters and molecular orbital data should be treated as immutable after solver construction.

## Tests

New methods should include tests that:

- compare `matvec_A(x)` against explicit dense multiplication from `get_A_matrix()`;
- compare `matmat_A(X)` against column-by-column `matvec_A(x)`;
- compare `kernel_davidson()` roots against `kernel_diag()` roots on a small system;
- cover singlet/triplet or range-separated branches when applicable;
- verify clear errors for optional dependencies.

Run the test suite before submitting changes:

```bash
py -3 -m unittest discover -v
```

## Style

Use readable NumPy expressions and keep tensor index conventions close to the mathematical formula. Prefer small, direct methods over broad abstractions unless a new abstraction clearly helps multiple methods.

## Licensing

Contributions are accepted under the Apache License, Version 2.0. By contributing, you agree that your contribution may be distributed under the project license. Preserve third-party notices and attribution when adapting code or formulas from other projects.
