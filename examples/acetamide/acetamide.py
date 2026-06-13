import system_env
from pyscf import gto, scf
from XsTDA import XsTDA
from sTDA import sTDA
import utils
import numpy as np
import os

nroot = 10
chkname = 'acetamide.chk'
mol_pyscf, mf = utils.read_chk_pyscf(chkname)
nocc = mf.mo_occ[mf.mo_occ > 0].shape[0]
nvir = mf.mo_occ[mf.mo_occ == 0].shape[0]
mol = system_env.molecule(mf.mo_coeff, mf.mo_energy, mf.mo_occ, mol_pyscf.aoslice_by_atom(), mol_pyscf.atom_charges(),
                          mol_pyscf.atom_coords(), nocc_trunc=nocc, nvir_trunc=nvir, basis=mol_pyscf.basis,
                          int1e_ovlp=mf.get_ovlp(), cart=mol_pyscf.cart)

with mol_pyscf.with_common_orig(mol._charge_center()):
    dipole_int = mol_pyscf.intor_symmetric('int1e_r', comp=3)

stda = sTDA(mol, ax=0.25, singlet=True, nroot=10)
e1, _ = stda.kernel_davidson()
f1 = stda.oscillator_strength(dipole_int)
stda.analyze(verbose=5)

xstda = XsTDA(mol, ax=0.25, singlet=True, nroot=10)
e1, _ = xstda.kernel_diag()
f1 = xstda.oscillator_strength(dipole_int)
xstda.analyze()
chkname = 'acetamide_CAM.chk'
mol_pyscf, mf = utils.read_chk_pyscf(chkname)
nocc = mf.mo_occ[mf.mo_occ > 0].shape[0]
nvir = mf.mo_occ[mf.mo_occ == 0].shape[0]
mol = system_env.molecule(mf.mo_coeff, mf.mo_energy, mf.mo_occ, mol_pyscf.aoslice_by_atom(), mol_pyscf.atom_charges(),
                          mol_pyscf.atom_coords(), nocc_trunc=nocc, nvir_trunc=nvir, basis=mol_pyscf.basis,
                          int1e_ovlp=mf.get_ovlp(), cart=mol_pyscf.cart)

stda = sTDA(mol, ax=0.38, alpha=1.86, beta=0.90, singlet=True, nroot=10)
e1, _ = stda.kernel_diag()
f1 = stda.oscillator_strength(dipole_int)
stda.analyze()

xstda = XsTDA(mol, ax=0.19, omega=0.33, betax=0.46, do_RSH=True, singlet=True, nroot=10)
e1, _ = xstda.kernel_diag()
f1 = xstda.oscillator_strength(dipole_int)
xstda.analyze()

