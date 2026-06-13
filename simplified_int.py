# Copyright 2026 Sheng-Rui Wang
# SPDX-License-Identifier: Apache-2.0

import numpy as np


def _sum_by_atom_blocks(values, aorange):
    starts = aorange[:, 2].astype(int)
    ends = aorange[:, 3].astype(int)
    csum = np.cumsum(values, axis=0)
    out = csum[ends - 1].copy()
    mask = starts > 0
    if np.any(mask):
        out[mask] -= csum[starts[mask] - 1]
    return out


def get_lowdin_pop(mol, type="ov", dependence="atom", if_trunc=True):
    C_MO = mol.C_MO
    if if_trunc:
        occidx = mol.occidx_trunc
        viridx = mol.viridx_trunc
    else:
        occidx = mol.occidx
        viridx = mol.viridx

    if type == "ov":
        C_p = C_MO[:, occidx]
        C_q = C_MO[:, viridx]
    elif type == "oo":
        C_p = C_MO[:, occidx]
        C_q = C_MO[:, occidx]
    elif type == "vv":
        C_p = C_MO[:, viridx]
        C_q = C_MO[:, viridx]
    elif type == "all":
        C_p = C_MO
        C_q = C_MO
    else:
        raise ValueError("Invalid type. Choose from 'ov', 'oo', 'vv', 'all'.")

    L_basis = C_p[:, :, None] * C_q[:, None, :]
    if dependence == "basis":
        return L_basis
    if dependence == "atom":
        return _sum_by_atom_blocks(L_basis, mol.aorange)
    raise ValueError("Invalid dependence. Choose 'atom' or 'basis'.")


def get_MNOK(dist, exp, eta, dependence="atom"):
    if dist.shape[0] != eta.shape[0]:
        raise ValueError("Shape of eta does not match dist matrix.")
    if dependence not in {"atom", "basis"}:
        raise ValueError("Invalid dependence. Choose 'atom' or 'basis'.")

    etaAB = 0.5 * (eta[:, None] + eta[None, :])
    return (dist**exp + etaAB ** (-exp)) ** (-1.0 / exp)


def _get_cart_norm_factors(l):
    n_cart = (l + 1) * (l + 2) // 2
    if l <= 1:
        return np.ones(n_cart)

    df = [1, 1, 3, 15, 105, 945, 10395]
    correction = np.sqrt((2 * l + 1) / (4 * np.pi))

    factors = []
    for lx in range(l, -1, -1):
        for ly in range(l - lx, -1, -1):
            lz = l - lx - ly
            norm = np.sqrt(df[l] / (df[lx] * df[ly] * df[lz]))
            factors.append(norm * correction)
    return np.array(factors)


def get_uuvv_pyscf(mol, cart=False, do_RSH=False, omega=None):
    mol_pyscf = pyscf_build_mol(mol, cart=cart)
    if do_RSH:
        mol_pyscf.set_range_coulomb(omega)

    uuvv = np.zeros((mol_pyscf.nao, mol_pyscf.nao), dtype=np.float64)
    ao_loc = mol_pyscf.ao_loc
    int_type = "int2e_cart" if mol_pyscf.cart else "int2e_sph"

    shell_idx = [np.arange(ao_loc[i + 1] - ao_loc[i]) for i in range(mol_pyscf.nbas)]
    norm_factors_sq = []
    for ish in range(mol_pyscf.nbas):
        l = mol_pyscf.bas_angular(ish)
        nctr = mol_pyscf.bas_nctr(ish)
        if mol_pyscf.cart:
            norm_factors_sq.append(np.tile(_get_cart_norm_factors(l), nctr) ** 2)
        else:
            norm_factors_sq.append(np.ones((2 * l + 1) * nctr))

    for i in range(mol_pyscf.nbas):
        ao_start_i, ao_end_i = ao_loc[i], ao_loc[i + 1]
        mu_idx = shell_idx[i]
        norm_i_sq = norm_factors_sq[i]

        for j in range(i, mol_pyscf.nbas):
            ints_block = mol_pyscf.intor_by_shell(int_type, (i, i, j, j))
            ao_start_j, ao_end_j = ao_loc[j], ao_loc[j + 1]
            nu_idx = shell_idx[j]
            diag_ints = ints_block[mu_idx, mu_idx, :, :][:, nu_idx, nu_idx]

            if cart:
                block = diag_ints * np.outer(norm_i_sq, norm_factors_sq[j])
            else:
                block = diag_ints
            uuvv[ao_start_i:ao_end_i, ao_start_j:ao_end_j] = block
            if i != j:
                uuvv[ao_start_j:ao_end_j, ao_start_i:ao_end_i] = block.T

    return uuvv


def pyscf_build_mol(mol, cart=False):
    from pyscf import gto

    mol_pyscf = gto.Mole()
    mol_pyscf.atom = [
        [mol.atom_charge[i], tuple(mol.atom_coord[i] * 0.52917721092)]
        for i in range(mol.natom)
    ]
    mol_pyscf.basis = mol.basis
    mol_pyscf.cart = cart
    mol_pyscf.symmetry = False
    mol_pyscf.build()
    return mol_pyscf
