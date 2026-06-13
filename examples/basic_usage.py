# Copyright 2026 Sheng-Rui Wang
# SPDX-License-Identifier: Apache-2.0

import system_env
import utils
from sTDA import sTDA


pmol, mf = utils.read_chk_pyscf("acetamide.chk")
mol = system_env.molecule(
    mf.mo_coeff, mf.mo_energy, mf.mo_occ,
    pmol.aoslice_by_atom(), pmol.atom_charges(), pmol.atom_coords(),
    basis=pmol.basis, int1e_ovlp=mf.get_ovlp(), cart=pmol.cart,
)

stda = sTDA(mol, ax=0.25, nroot=10)
print(stda.kernel_davidson()[0] * 27.21138)
