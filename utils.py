# Copyright 2026 Sheng-Rui Wang
# SPDX-License-Identifier: Apache-2.0

import numpy as np

def read_chk_pyscf(chkname):
    '''
    Read mol and mf objects from a PySCF checkpoint file.
    '''
    from pyscf import lib, scf
    
    mol = lib.chkfile.load_mol(chkname)
    mf = scf.RHF(mol)
    mf.__dict__.update(lib.chkfile.load(chkname, 'scf'))
    return mol, mf

def trunc_orb_energy(mo_energy, mo_occ, e_occ, e_vir):
    '''
    Truncate the number of occupied and virtual orbitals based on energy thresholds.
    '''
    occ_threshold = e_occ / 27.21138
    vir_threshold = e_vir / 27.21138
    occidx = np.where((mo_occ == 2) & (mo_energy >= mo_energy[mo_occ == 2].max() - occ_threshold))[0]
    viridx = np.where((mo_occ == 0) & (mo_energy <= mo_energy[mo_occ == 0].min() + vir_threshold))[0]
    nocc, nvir = len(occidx), len(viridx)
    return nocc, nvir